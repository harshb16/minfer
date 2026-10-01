import torch
from transformers import GenerationConfig

from minfer.config import SamplingParams


def test_cached_decode_matches_full_forward(tiny_runner):
    runner = tiny_runner
    ids = [3, 7, 11, 15]
    state = runner.prefill(ids)
    for _ in range(4):
        token = int(state.logits[0].argmax())
        ids.append(token)
        state = runner.decode_one(token, state.cache)
        with torch.inference_mode():
            full = runner.model(input_ids=torch.tensor([ids]), use_cache=False).logits[:, -1]
        torch.testing.assert_close(state.logits, full, atol=1e-6, rtol=1e-5)
        assert runner.cache_manager.length(state.cache) == len(ids)


def test_manual_greedy_matches_hf(tiny_runner):
    runner = tiny_runner
    prompt = [5, 10, 20]
    ours = runner.generate_independent(prompt, SamplingParams(max_new_tokens=8))
    with torch.inference_mode():
        reference = runner.model.generate(
            torch.tensor([prompt]),
            attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
            generation_config=GenerationConfig(
                max_new_tokens=8, do_sample=False, eos_token_id=2, pad_token_id=0, use_cache=True
            ),
        )[0, len(prompt) :].tolist()
    assert ours == reference
