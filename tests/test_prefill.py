from unittest.mock import patch

import pytest
import torch

from minfer import EngineConfig, LLMEngine, SamplingParams


def assert_prefill_parity(runner, prompts, atol=1e-6, rtol=1e-5):
    expected = [runner.prefill(ids) for ids in prompts]
    calls = []
    hook = runner.model.register_forward_pre_hook(
        lambda _, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    try:
        actual = runner.prefill_batch(prompts)
    finally:
        hook.remove()
    assert len(calls) == 1
    assert calls[0]["input_ids"].shape == (len(prompts), max(map(len, prompts)))
    for row, (a, b) in enumerate(zip(actual, expected, strict=True)):
        length = len(prompts[row])
        assert calls[0]["position_ids"][row, -length:].tolist() == list(range(length))
        assert calls[0]["attention_mask"][row].tolist() == (
            [0] * (max(map(len, prompts)) - length) + [1] * length
        )
        torch.testing.assert_close(a.logits, b.logits, atol=atol, rtol=rtol)
        assert int(a.logits.argmax()) == int(b.logits.argmax())
        assert runner.cache_manager.length(a.cache) == length
        for (ak, av), (bk, bv) in zip(
            runner.cache_manager.layers(a.cache), runner.cache_manager.layers(b.cache), strict=True
        ):
            torch.testing.assert_close(ak, bk, atol=atol, rtol=rtol)
            torch.testing.assert_close(av, bv, atol=atol, rtol=rtol)
            for tensor in (ak, av):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_unequal_prefill_mask_positions_cache_and_logits(tiny_runner):
    prompts = [[3], [7, 9, 11, 13, 15], list(range(10, 22))]
    # Arbitrary nonzero padding must be masked, including its computed K/V.
    with patch.object(tiny_runner.tokenizer, "pad_token_id", 88):
        assert_prefill_parity(tiny_runner, prompts)


def test_single_and_invalid_prefill_batch(tiny_runner):
    assert_prefill_parity(tiny_runner, [[3, 4]])
    for prompts in ([], [[]], [[3], []]):
        with pytest.raises(ValueError, match="nonempty"):
            tiny_runner.prefill_batch(prompts)


def test_engine_batches_admitted_prompts_and_traces_budget(tiny_runner):
    e = LLMEngine(config=EngineConfig(max_batched_tokens=8), runner=tiny_runner)
    params = SamplingParams(max_new_tokens=3, stop_on_eos=False)
    ids = [e.add_request(p, params) for p in ("ab", "cde", "fghi")]
    calls = []
    hook = tiny_runner.model.register_forward_pre_hook(
        lambda _, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True
    )
    try:
        assert [event.request_id for event in e.step()] == ids[:2]
        assert e.last_step == dict(
            prefill_batch_size=2,
            prefill_tokens=5,
            decode_batch_size=0,
            scheduled_tokens_this_step=5,
        )
        events = e.step()
        assert [event.request_id for event in events] == [ids[2], *ids[:2]]
        assert e.last_step["scheduled_tokens_this_step"] == 6
        assert calls == [(2, 3), (1, 4), (2, 1)]
    finally:
        hook.remove()
