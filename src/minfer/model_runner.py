"""Only this module performs model forward calls; no generation loop is delegated."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import DynamicCache

from .cache import HFCacheManager
from .config import EngineConfig, SamplingParams
from .device import select_device, select_dtype
from .sampler import Sampler


@dataclass
class ForwardResult:
    logits: torch.Tensor
    cache: DynamicCache


class ModelRunner:
    def __init__(
        self,
        config: EngineConfig,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.device = select_device(config.device)
        self.dtype = select_dtype(self.device, config.dtype)
        self.config = config
        self.cache_manager = HFCacheManager()
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        )
        self.model = (
            model
            if model is not None
            else AutoModelForCausalLM.from_pretrained(
                config.model,
                dtype=self.dtype,
                attn_implementation="sdpa",
                cache_dir=config.cache_dir,
            )
        )
        if getattr(self.model.config, "use_sliding_window", False) or any(
            t != "full_attention" for t in (getattr(self.model.config, "layer_types", None) or [])
        ):
            raise ValueError(
                "minfer supports full-attention caches; sliding layers are unsupported"
            )
        self.model.to(device=self.device, dtype=self.dtype).eval()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

    def tokenize(self, prompt: str) -> list[int]:
        if self.config.chat_template:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("Prompt must tokenize to at least one token")
            ids = [bos]
        return list(ids)

    def decode_text(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> ForwardResult:
        if not token_ids:
            raise ValueError("Prefill requires at least one token")
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        output = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            past_key_values=self.cache_manager.empty(),
            use_cache=True,
            logits_to_keep=1,
        )
        return ForwardResult(output.logits[:, -1, :], output.past_key_values)

    @torch.inference_mode()
    def prefill_batch(self, prompts: Sequence[list[int]]) -> list[ForwardResult]:
        """One left-padded forward; only real tokens survive in request caches."""
        if not prompts or any(not ids for ids in prompts):
            raise ValueError("Prefill requires a nonempty batch of nonempty prompts")
        if len(prompts) == 1:
            return [self.prefill(prompts[0])]
        lengths = [len(ids) for ids in prompts]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            pad = 0  # Masked input only; never a semantic prompt token.
        ids = torch.full((len(prompts), width), pad, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, -len(prompt) :] = torch.tensor(prompt, device=self.device)
            mask[row, -len(prompt) :] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=self.cache_manager.empty(),
            use_cache=True,
            logits_to_keep=1,
        )
        caches = self.cache_manager.split_prefill(output.past_key_values, lengths)
        return [
            ForwardResult(output.logits[row : row + 1, -1, :], cache)
            for row, cache in enumerate(caches)
        ]

    @torch.inference_mode()
    def decode_one(self, token_id: int, cache: DynamicCache) -> ForwardResult:
        self.cache_manager.validate(cache)
        length = self.cache_manager.length(cache)
        ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        mask = torch.ones((1, length + 1), device=self.device, dtype=torch.long)
        positions = torch.tensor([[length]], device=self.device, dtype=torch.long)
        # In 5.18 Qwen derives physical query offset from DynamicCache. No
        # cache_position is accepted/needed. RoPE still uses semantic positions.
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return ForwardResult(output.logits[:, -1, :], output.past_key_values)

    def generate_independent(self, token_ids: list[int], params: SamplingParams) -> list[int]:
        """Manual single-request oracle: prefill, then consume one pending token."""
        generator = torch.Generator()
        generator.seed() if params.seed is None else generator.manual_seed(params.seed)
        sampler = Sampler()
        state = self.prefill(token_ids)
        generated = []
        for _ in range(params.max_new_tokens):
            token = sampler.sample(state.logits[0], params, generator)
            generated.append(token)
            if params.stop_on_eos and token in self.eos_token_ids:
                break
            if len(generated) < params.max_new_tokens:
                state = self.decode_one(token, state.cache)
        return generated

    @torch.inference_mode()
    def decode_batch(
        self, pending_tokens: Sequence[int], caches: Sequence[DynamicCache]
    ) -> tuple[torch.Tensor, list[DynamicCache]]:
        if len(pending_tokens) != len(caches) or not caches:
            raise ValueError("Each cache needs exactly one pending token")
        if len(caches) == 1:
            state = self.decode_one(pending_tokens[0], caches[0])
            return state.logits, [state.cache]
        merged = self.cache_manager.merge(caches)
        ids = torch.tensor(pending_tokens, device=self.device, dtype=torch.long)[:, None]
        output = self.model(
            input_ids=ids,
            attention_mask=merged.attention_mask,
            position_ids=merged.position_ids,
            past_key_values=merged.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        split = self.cache_manager.split(output.past_key_values, merged.lengths)
        return output.logits[:, -1, :], split
