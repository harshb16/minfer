import pytest
import torch

from minfer.cache import HFCacheManager


def assert_ragged_parity(runner, prompts, *, atol=1e-6, rtol=1e-5):
    manager = runner.cache_manager
    states = [runner.prefill(ids) for ids in prompts]
    independent = [s.cache for s in states]
    batched = [manager.clone(s.cache) for s in states]
    pending = [int(s.logits[0].argmax()) for s in states]
    for step in range(5):
        lengths_before = [manager.length(c) for c in batched]
        expected = [
            runner.decode_one(token, cache)
            for token, cache in zip(pending, independent, strict=True)
        ]
        calls = []
        hook = runner.model.register_forward_pre_hook(
            lambda _, args, kwargs, calls=calls: calls.append(tuple(kwargs["input_ids"].shape)),
            with_kwargs=True,
        )
        try:
            logits, batched = runner.decode_batch(pending, batched)
        finally:
            hook.remove()
        assert calls == [(len(prompts), 1)]
        independent = [s.cache for s in expected]
        for row, state in enumerate(expected):
            torch.testing.assert_close(logits[row], state.logits[0], atol=atol, rtol=rtol)
            assert int(logits[row].argmax()) == int(state.logits[0].argmax())
            assert manager.length(batched[row]) == len(prompts[row]) + step + 1
            assert manager.length(batched[row]) == lengths_before[row] + 1
            for (bk, bv), (ik, iv) in zip(
                manager.layers(batched[row]), manager.layers(state.cache), strict=True
            ):
                torch.testing.assert_close(bk, ik, atol=atol, rtol=rtol)
                torch.testing.assert_close(bv, iv, atol=atol, rtol=rtol)
        pending = logits.argmax(-1).tolist()


def test_multistep_ragged_parity(tiny_runner):
    assert_ragged_parity(tiny_runner, [[3, 5], [7, 9, 11, 13, 15], list(range(10, 22))])


def test_merge_mask_positions_and_independent_storage(tiny_runner):
    manager = HFCacheManager()
    caches = [tiny_runner.prefill(ids).cache for ids in [[3, 5], [7, 9, 11, 13, 15]]]
    merged = manager.merge(caches)
    assert merged.attention_mask.tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    assert merged.position_ids.tolist() == [[2], [5]]
    # Poison masked K/V: parity must depend on the mask, not zero padding.
    for k, v in manager.layers(merged.cache):
        k[0, :, :3] = 77
        v[0, :, :3] = 99
    with torch.inference_mode():
        output = tiny_runner.model(
            input_ids=torch.tensor([[6], [8]]),
            attention_mask=merged.attention_mask,
            position_ids=merged.position_ids,
            past_key_values=merged.cache,
            use_cache=True,
        )
    expected = tiny_runner.decode_one(6, caches[0])
    torch.testing.assert_close(output.logits[0, -1], expected.logits[0], atol=1e-6, rtol=1e-5)
    split = manager.split(output.past_key_values, merged.lengths)
    for k, _ in manager.layers(split[0]):
        assert k.shape[-2] == 3
        assert k.untyped_storage().nbytes() == k.numel() * k.element_size()
    with pytest.raises(ValueError, match="empty"):
        manager.merge([])


def test_split_rejects_unconsumed_cache(tiny_runner):
    manager = HFCacheManager()
    merged = manager.merge([tiny_runner.prefill([3, 4]).cache])
    with pytest.raises(ValueError, match="exactly one"):
        manager.split(merged.cache, merged.lengths)
