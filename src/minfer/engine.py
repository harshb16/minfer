"""A synchronous engine with one-token-per-request scheduler steps."""

from collections.abc import Sequence
from time import perf_counter
from uuid import uuid4

from .config import MODEL_ID, EngineConfig, SamplingParams
from .events import GenerationResult, TokenEvent
from .model_runner import ModelRunner
from .request import Request, RequestStatus
from .sampler import Sampler
from .scheduler import Scheduler


class LLMEngine:
    def __init__(
        self,
        model: str = MODEL_ID,
        *,
        config: EngineConfig | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_active_requests: int = 4,
        max_batched_tokens: int | None = None,
        max_kv_tokens: int | None = None,
        chat_template: bool = True,
        runner: ModelRunner | None = None,
    ) -> None:
        self.config = config or EngineConfig(
            model=model,
            device=device,
            dtype=dtype,
            max_active_requests=max_active_requests,
            max_batched_tokens=max_batched_tokens,
            max_kv_tokens=max_kv_tokens,
            chat_template=chat_template,
        )
        self.runner = runner if runner is not None else ModelRunner(self.config)
        self.scheduler = Scheduler(
            self.config.max_active_requests,
            self.config.max_batched_tokens,
            self.config.max_kv_tokens,
        )
        self.sampler = Sampler()
        self.last_step: dict[str, int] = {}

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str | None = None,
    ) -> str:
        """Tokenize immediately; admission and prefill happen on a later step."""
        arrival = perf_counter()
        params = sampling_params or SamplingParams()
        ids = self.runner.tokenize(prompt)
        context_limit = self.runner.model.config.max_position_embeddings
        if len(ids) + params.max_new_tokens - 1 > context_limit:
            raise ValueError(f"Prompt and generation exceed model context limit {context_limit}")
        request = Request(
            request_id if request_id is not None else uuid4().hex,
            prompt,
            ids,
            params,
            arrival_time=arrival,
        )
        self.scheduler.add(request)
        return request.request_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def get_request(self, request_id: str) -> Request:
        return self.scheduler.requests[request_id]

    def _emit(self, request: Request, token: int) -> TokenEvent:
        request.generated_token_ids.append(token)
        request.pending_token = token
        if request.first_token_time is None:
            request.first_token_time = perf_counter()
        reason = None
        if request.sampling_params.stop_on_eos and token in self.runner.eos_token_ids:
            reason = "eos"
        elif request.num_generated_tokens >= request.sampling_params.max_new_tokens:
            reason = "length"
        text = self.runner.decode_text(request.generated_token_ids)
        # Byte-level BPE can end mid-codepoint. Buffer replacement characters
        # until a subsequent token completes the character, or generation ends.
        if reason is None:
            text = text.rstrip("\ufffd")
        if not text.startswith(request.emitted_text):
            raise RuntimeError("Tokenizer rewrote an already emitted prefix")
        delta = text[len(request.emitted_text) :]
        request.emitted_text = text
        if reason:
            self.scheduler.finish(request, reason)
        return TokenEvent(request.request_id, token, delta, reason is not None, reason)

    def step(self) -> list[TokenEvent]:
        """Admit once, prefill new requests, decode the previous active snapshot.

        Each request emits at most one token per call. Newly admitted requests
        are excluded from this call's decode batch. Freed slots are reused on
        the next call, avoiding unbounded admission loops for one-token jobs.
        """
        plan = self.scheduler.plan_step()
        previous, admitted = plan.decode, plan.prefill
        self.last_step = {
            "prefill_batch_size": len(admitted),
            "prefill_tokens": sum(r.prompt_length for r in admitted),
            "decode_batch_size": len(previous),
            "scheduled_tokens_this_step": plan.scheduled_tokens,
        }
        events = []
        if admitted:
            try:
                states = self.runner.prefill_batch([r.prompt_token_ids for r in admitted])
                for request, state in zip(admitted, states, strict=True):
                    request.cache = state.cache
                    request.kv_tokens = request.prompt_length
                    token = self.sampler.sample(
                        state.logits[0], request.sampling_params, request.generator
                    )
                    events.append(self._emit(request, token))
            except Exception:
                for affected in admitted:
                    if affected.status == RequestStatus.RUNNING:
                        self.scheduler.abort(affected.request_id)
                raise
        if previous:
            if any(r.cache is None or r.pending_token is None for r in previous):
                raise RuntimeError("RUNNING requests need a cache and pending token")
            try:
                logits, caches = self.runner.decode_batch(
                    [r.pending_token for r in previous], [r.cache for r in previous]
                )
                for row, (request, cache) in enumerate(zip(previous, caches, strict=True)):
                    request.cache = cache
                    request.kv_tokens += 1
                    token = self.sampler.sample(
                        logits[row], request.sampling_params, request.generator
                    )
                    events.append(self._emit(request, token))
            except Exception:
                # Single-request DynamicCache mutates during forward. A failed
                # batch cannot safely be retried with potentially consumed tokens.
                for request in previous:
                    if request.status == RequestStatus.RUNNING:
                        self.scheduler.abort(request.request_id)
                raise
        return events

    def abort_request(self, request_id: str) -> TokenEvent:
        request = self.scheduler.abort(request_id)
        text = self.runner.decode_text(request.generated_token_ids)
        delta = text[len(request.emitted_text) :]
        request.emitted_text = text
        return TokenEvent(request_id, None, delta, True, "aborted")

    def get_result(self, request_id: str) -> GenerationResult:
        request = self.scheduler.finished[request_id]
        assert request.finish_time is not None and request.finish_reason is not None
        return GenerationResult(
            request_id,
            request.prompt,
            self.runner.decode_text(request.generated_token_ids),
            tuple(request.generated_token_ids),
            request.finish_reason,
            request.finish_time - request.arrival_time,
            None
            if request.first_token_time is None
            else request.first_token_time - request.arrival_time,
        )

    def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> GenerationResult:
        return self.generate_batch([prompt], sampling_params)[0]

    def generate_batch(
        self, prompts: Sequence[str], sampling_params: SamplingParams | None = None
    ) -> list[GenerationResult]:
        """Submit all prompts and drain the engine, including previously queued work."""
        ids = [self.add_request(prompt, sampling_params) for prompt in prompts]
        while self.has_unfinished_requests():
            self.step()
        return [self.get_result(request_id) for request_id in ids]
