from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from minfer import EngineConfig, LLMEngine, SamplingParams
from minfer.cache import HFCacheManager
from minfer.paged_cache import BlockManager, PagedKVManager
from minfer.scheduler import Scheduler
from tests.test_scheduler import sized


@pytest.mark.parametrize("tokens,blocks", [(1, 1), (16, 1), (17, 2), (32, 2), (33, 3)])
def test_lazy_block_boundaries(tokens, blocks):
    manager = BlockManager(16, 4)
    manager.allocate("A", 1)
    manager.grow("A", tokens)
    assert len(manager.table("A")) == blocks
    assert manager.length("A") == tokens
    m = manager.metrics()
    assert m.real_tokens_stored == tokens
    assert m.internal_fragmentation_tokens == blocks * 16 - tokens
    assert m.utilization == blocks / 4
    manager.validate()


def test_multiple_owners_free_reuse_exhaustion_and_accounting():
    m = BlockManager(4, 3)
    m.allocate("A", 5)
    m.allocate("B", 1)
    assert m.table("A") == (0, 1) and m.table("B") == (2,)
    assert asdict(m.metrics()) == dict(
        total_blocks=3,
        used_blocks=3,
        free_blocks=0,
        block_size=4,
        allocated_token_capacity=12,
        real_tokens_stored=6,
        internal_fragmentation_tokens=6,
        utilization=1.0,
    )
    with pytest.raises(MemoryError, match="exhausted"):
        m.allocate("C", 1)
    with pytest.raises(MemoryError, match="exhausted"):
        m.grow("A", 9)
    assert m.length("A") == 5 and m.table("A") == (0, 1)
    with pytest.raises(ValueError, match="already owns"):
        m.allocate("A", 1)
    with pytest.raises(ValueError, match="shrink"):
        m.grow("A", 4)
    m.free("A")
    with pytest.raises(ValueError, match="already freed"):
        m.free("A")
    m.allocate("C", 8)
    assert m.table("C") == (0, 1)
    m.validate()
    m.free("C")
    m.free("B")
    assert m.metrics().used_blocks == m.metrics().real_tokens_stored == 0
    m.validate()


@pytest.mark.parametrize("corruption", ["duplicate", "capacity", "length", "out_of_range"])
def test_invalid_block_state_detected(corruption):
    m = BlockManager(4, 4)
    m.allocate("A", 5)
    if corruption == "duplicate":
        m._tables["A"][1] = m._tables["A"][0]
    elif corruption == "capacity":
        m._tables["A"].pop()
    elif corruption == "length":
        m._lengths.clear()
    else:
        m._tables["A"][1] = 99
    with pytest.raises(ValueError):
        m.validate()


def synthetic(length, heads=2):
    return HFCacheManager.from_layers(
        (
            torch.arange(heads * length * 3, dtype=torch.float32).reshape(1, heads, length, 3) + i,
            torch.ones(1, heads, length, 3) * (i + 10),
        )
        for i in range(2)
    )


def assert_cache_equal(a, b, atol=0, rtol=0):
    for (ak, av), (bk, bv) in zip(HFCacheManager.layers(a), HFCacheManager.layers(b), strict=True):
        torch.testing.assert_close(ak, bk, atol=atol, rtol=rtol)
        torch.testing.assert_close(av, bv, atol=atol, rtol=rtol)


@torch.inference_mode()
def test_storage_partial_blocks_append_only_and_independent_materialization():
    p = PagedKVManager(4, 8)
    initial = synthetic(3)
    p.write("A", initial)
    p.write("B", synthetic(5))
    assert_cache_equal(p.materialize("A"), initial)
    # Poison unused slots: they must never appear in a materialized cache.
    for k, v in p._storage:
        k[p.blocks.table("A")[-1], :, 3:] = float("nan")
        v[p.blocks.table("A")[-1], :, 3:] = float("nan")
    assert_cache_equal(p.materialize("A"), initial)
    history = HFCacheManager.layers(initial)
    suffix = synthetic(7)
    # Use true old history plus a suffix that crosses two boundaries.
    grown = HFCacheManager.from_layers(
        (torch.cat((k, sk), dim=2), torch.cat((v, sv), dim=2))
        for (k, v), (sk, sv) in zip(history, HFCacheManager.layers(suffix), strict=True)
    )
    p.write("A", grown)
    assert_cache_equal(p.materialize("A"), grown)
    # Rewriting with poisoned history proves only newly appended tokens persist.
    poisoned = HFCacheManager().clone(grown)
    for k, v in HFCacheManager.layers(poisoned):
        k[:, :, :3] = 999
        v[:, :, :3] = 999
    p.write("A", poisoned)
    assert_cache_equal(p.materialize("A"), grown)
    materialized = p.materialize("A")
    for k, _ in HFCacheManager.layers(materialized):
        k.fill_(777)
    assert_cache_equal(p.materialize("A"), grown)
    assert_cache_equal(p.materialize("B"), synthetic(5))
    p.blocks.validate()
    with pytest.raises(ValueError, match="shrink"):
        p.write("A", initial)
    with pytest.raises(ValueError, match="Incompatible"):
        p.write("C", synthetic(3, heads=3))
    assert not p.blocks.contains("C")


def test_block_admission_rounding_and_release():
    blocks = BlockManager(4, 3)
    s = Scheduler(3, block_manager=blocks)
    a, b, c = sized("A", 4, 2), sized("B", 4, 2), sized("C", 1, 1)
    for r in (a, b, c):
        s.add(r)
    assert s.admit() == [a]  # Worst case 5 tokens rounds up to two blocks.
    blocks.allocate("A", 4)
    s.finish(a, "length")
    assert blocks.free_blocks == 3
    assert s.admit() == [b, c]
    with pytest.raises(ValueError, match="never fit KV block pool"):
        s.add(sized("D", 13, 1))


def assert_paged_engine_parity(runner, prompts):
    params = SamplingParams(max_new_tokens=6, stop_on_eos=False)
    expected = [runner.generate_independent(runner.tokenize(p), params) for p in prompts]
    engines = [
        LLMEngine(
            config=EngineConfig(cache_mode=mode, kv_block_size=4, num_kv_blocks=128), runner=runner
        )
        for mode in ("dynamic", "paged")
    ]
    ids = [[e.add_request(p, params) for p in prompts] for e in engines]
    atol, rtol = (3e-2, 3e-2) if runner.device.type == "mps" else (1e-4, 1e-4)
    while engines[0].has_unfinished_requests():
        for e in engines:
            events = e.step()
            assert len(events) == len({x.request_id for x in events})
        for a, b in zip(ids[0], ids[1], strict=True):
            da, pb = engines[0].get_request(a), engines[1].get_request(b)
            assert da.generated_token_ids == pb.generated_token_ids
            if da.cache is not None:
                assert pb.cache is None
                assert_cache_equal(da.cache, engines[1].paged_cache.materialize(b), atol, rtol)
                assert engines[1].paged_cache.blocks.length(b) == pb.kv_tokens
        engines[1].paged_cache.blocks.validate()
    for e, request_ids in zip(engines, ids, strict=True):
        assert [list(e.get_result(i).token_ids) for i in request_ids] == expected
    assert engines[1].paged_cache.blocks.metrics().used_blocks == 0


def test_paged_engine_dynamic_and_oracle_parity(tiny_runner):
    assert_paged_engine_parity(tiny_runner, ["a", "unequal", "a much longer input"])


@pytest.mark.parametrize("operation", ["abort", "decode_failure", "prefill_failure"])
def test_paged_cleanup_and_reuse(tiny_runner, operation):
    e = LLMEngine(cache_mode="paged", kv_block_size=4, num_kv_blocks=4, runner=tiny_runner)
    params = SamplingParams(max_new_tokens=3, stop_on_eos=False)
    ids = [e.add_request(p, params) for p in ("abc", "d")]
    if operation == "prefill_failure":
        # Simulate a failure after one cache has been persisted.
        original = e.paged_cache.write
        calls = []

        def failing_write(request_id, cache):
            original(request_id, cache)
            calls.append(request_id)
            if len(calls) == 2:
                raise RuntimeError("storage failed")

        with patch.object(e.paged_cache, "write", side_effect=failing_write):
            with pytest.raises(RuntimeError):
                e.step()
    else:
        e.step()
        if operation == "abort":
            for i in ids:
                e.abort_request(i)
        else:
            with patch.object(tiny_runner, "decode_batch", side_effect=RuntimeError("forward")):
                with pytest.raises(RuntimeError):
                    e.step()
    assert e.paged_cache.blocks.free_blocks == 4
    e.paged_cache.blocks.validate()
    assert not e.has_unfinished_requests()
    assert len(e.generate("abc", params).token_ids) == 3


def test_paged_waiting_progress_lazy_growth_and_terminal_peak(tiny_runner):
    e = LLMEngine(
        cache_mode="paged",
        kv_block_size=4,
        num_kv_blocks=2,
        max_kv_tokens=8,
        max_batched_tokens=6,
        runner=tiny_runner,
    )
    params = SamplingParams(max_new_tokens=3, stop_on_eos=False)
    a, b = [e.add_request(p, params) for p in ("abcd", "xy")]
    assert [event.request_id for event in e.step()] == [a]
    assert e.paged_cache.blocks.metrics().used_blocks == 1  # reservation is two, allocation one
    assert e.scheduler.usage().waiting_requests == 1
    e.step()
    assert e.paged_cache.blocks.metrics().used_blocks == 2
    assert e.step()[0].finished
    assert e.paged_cache.blocks.metrics().used_blocks == 0
    assert [event.request_id for event in e.step()] == [b]
    while e.has_unfinished_requests():
        e.step()
    c = e.add_request("z", SamplingParams(max_new_tokens=1))
    assert e.step()[0].request_id == c
    assert e.last_step["kv_metrics_peak"]["used_blocks"] == 1
    assert e.last_step["kv_metrics"]["used_blocks"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cache_mode="unknown"),
        dict(kv_block_size=0),
        dict(num_kv_blocks=0),
        dict(max_kv_tokens=0),
        dict(max_batched_tokens=-1),
    ],
)
def test_invalid_capacity_config(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)
