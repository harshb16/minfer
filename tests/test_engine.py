from unittest.mock import patch

import pytest
import torch

from minfer import EngineConfig, LLMEngine, RequestStatus, SamplingParams


def engine(runner, capacity=3):
    return LLMEngine(
        config=EngineConfig(device="cpu", max_active_requests=capacity, chat_template=False),
        runner=runner,
    )


def test_batch_matches_independent_and_cache_invariant(tiny_runner):
    prompts = ["hi", "unequal length", "a considerably longer input prompt"]
    params = SamplingParams(max_new_tokens=7, stop_on_eos=False)
    expected = [tiny_runner.generate_independent(tiny_runner.tokenize(p), params) for p in prompts]
    e = engine(tiny_runner)
    ids = [e.add_request(p, params) for p in prompts]
    stream = {i: "" for i in ids}
    while e.has_unfinished_requests():
        events = e.step()
        assert len({event.request_id for event in events}) == len(events)
        for event in events:
            stream[event.request_id] += event.token_text
        for r in e.scheduler.running.values():
            assert r.pending_token == r.generated_token_ids[-1]
            assert tiny_runner.cache_manager.length(r.cache) == (
                r.prompt_length + r.num_generated_tokens - 1
            )
    assert [list(e.get_result(i).token_ids) for i in ids] == expected
    for i in ids:
        assert stream[i] == e.get_result(i).text
        assert e.get_request(i).cache is None
        assert e.get_result(i).latency >= e.get_result(i).ttft >= 0


def test_late_arrivals_growing_shrinking_and_slot_reuse(tiny_runner):
    e = engine(tiny_runner, capacity=3)
    params = SamplingParams(max_new_tokens=5, stop_on_eos=False)
    a = e.add_request("A", params)
    b = e.add_request("BB", SamplingParams(max_new_tokens=2, stop_on_eos=False))
    assert [x.request_id for x in e.step()] == [a, b]
    c = e.add_request("CCC", params)
    calls = []
    hook = tiny_runner.model.register_forward_pre_hook(
        lambda _, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True
    )
    events = e.step()
    hook.remove()
    assert [x.request_id for x in events] == [c, a, b]
    assert calls == [(1, 3), (2, 1)]  # C is prefilled, never decoded this step.
    assert list(e.scheduler.running) == [a, c]
    d = e.add_request("DDDD", SamplingParams(max_new_tokens=1))
    events = e.step()
    assert [x.request_id for x in events] == [d, a, c]
    assert e.get_request(d).status == RequestStatus.FINISHED
    assert list(e.scheduler.running) == [a, c]
    while e.has_unfinished_requests():
        e.step()
    assert len(e.scheduler.finished) == 4


def test_eos_length_and_abort(tiny_runner):
    e = engine(tiny_runner, capacity=1)
    a = e.add_request("A", SamplingParams(max_new_tokens=10))
    with patch.object(e.sampler, "sample", return_value=2):
        event = e.step()[0]
    assert event.finished and event.finish_reason == "eos"
    assert e.get_result(a).token_ids == (2,)
    b = e.add_request("B", SamplingParams(max_new_tokens=1, stop_on_eos=False))
    with patch.object(e.sampler, "sample", return_value=2):
        assert e.step()[0].finish_reason == "length"
    assert e.get_result(b).token_ids == (2,)
    c = e.add_request("C")
    assert e.abort_request(c).finish_reason == "aborted"
    assert e.get_result(c).ttft is None
    d = e.add_request("D")
    e.step()
    e.abort_request(d)
    assert e.get_request(d).cache is None
    assert not e.has_unfinished_requests()


def test_sampling_same_request_seed_independent_of_batch(tiny_runner):
    params = SamplingParams(
        max_new_tokens=8, temperature=0.9, top_k=8, top_p=0.85, seed=123, stop_on_eos=False
    )
    prompts = ["abc", "much longer input", "z"]
    expected = [tiny_runner.generate_independent(tiny_runner.tokenize(p), params) for p in prompts]
    results = engine(tiny_runner).generate_batch(prompts, params)
    assert [list(r.token_ids) for r in results] == expected


def test_decode_failure_releases_mutated_cache(tiny_runner):
    e = engine(tiny_runner)
    ids = [e.add_request(p) for p in ["A", "BB"]]
    e.step()
    with patch.object(tiny_runner, "decode_batch", side_effect=RuntimeError("forward failed")):
        with pytest.raises(RuntimeError, match="forward failed"):
            e.step()
    assert not e.has_unfinished_requests()
    assert all(e.get_request(i).cache is None for i in ids)


def test_unicode_stream_buffers_incomplete_bytes(tiny_runner):
    e = engine(tiny_runner)
    i = e.add_request("hi", SamplingParams(max_new_tokens=3))
    with patch.object(tiny_runner, "decode_text", side_effect=["a\ufffd", "a界", "a界!"]):
        with patch.object(e.sampler, "sample", return_value=5):
            events = [e.step()[0], e.step()[0], e.step()[0]]
    assert [event.token_text for event in events] == ["a", "界", "!"]
    assert e.get_request(i).emitted_text == "a界!"


def test_prefill_failure_cleans_all_admissions(tiny_runner):
    e = engine(tiny_runner)
    ids = [e.add_request(p) for p in ["A", "B"]]
    with patch.object(tiny_runner, "prefill_batch", side_effect=RuntimeError("prefill failed")):
        with pytest.raises(RuntimeError, match="prefill failed"):
            e.step()
    assert not e.has_unfinished_requests()
    assert all(e.get_request(i).status == RequestStatus.ABORTED for i in ids)


def test_context_bound_and_empty_prompt(tiny_runner):
    e = engine(tiny_runner)
    with pytest.raises(ValueError, match="context"):
        e.add_request("a" * 250, SamplingParams(max_new_tokens=10))
    assert tiny_runner.tokenize("") == [1]


def test_greedy_sampling_does_not_change_rng(tiny_runner):
    e = engine(tiny_runner)
    i = e.add_request("a")
    before = e.get_request(i).generator.get_state().clone()
    e.step()
    assert torch.equal(before, e.get_request(i).generator.get_state())


def test_token_budgets_allow_completion_and_waiting_progress(tiny_runner):
    e = LLMEngine(
        config=EngineConfig(
            device="cpu", max_active_requests=3, max_batched_tokens=5, max_kv_tokens=8
        ),
        runner=tiny_runner,
    )
    params = SamplingParams(max_new_tokens=4, stop_on_eos=False)
    ids = [e.add_request(p, params) for p in ("abc", "def", "z")]
    steps = 0
    while e.has_unfinished_requests():
        events = e.step()
        assert len({x.request_id for x in events}) == len(events)
        assert e.scheduler.usage().used_kv_tokens <= 8
        assert e.scheduler.usage().scheduled_tokens_this_step <= 5
        steps += 1
        assert steps < 20
    assert e.scheduler.usage().used_kv_tokens == 0
    assert all(len(e.get_result(i).token_ids) == 4 for i in ids)
