import pytest

from minfer import Request, RequestStatus, SamplingParams
from minfer.scheduler import Scheduler


def request(name):
    return Request(name, name, [3], SamplingParams())


def test_fifo_capacity_completion_slot_reuse_and_arrivals():
    scheduler = Scheduler(2)
    a, b, c, d = [request(name) for name in "ABCD"]
    for r in [a, b, c]:
        scheduler.add(r)
    assert scheduler.admit() == [a, b]
    assert scheduler.admit() == []
    assert list(scheduler.running) == ["A", "B"]
    scheduler.finish(b, "length")
    assert b.status == RequestStatus.FINISHED
    assert scheduler.admit() == [c]
    scheduler.add(d)
    scheduler.finish(a, "eos")
    scheduler.finish(c, "length")
    assert scheduler.admit() == [d]
    scheduler.finish(d, "eos")
    assert not scheduler.has_unfinished()
    assert len(scheduler.finished) == 4


def test_duplicate_and_abort():
    scheduler = Scheduler(1)
    a, b = request("A"), request("B")
    scheduler.add(a)
    scheduler.add(b)
    with pytest.raises(ValueError, match="Duplicate"):
        scheduler.add(a)
    scheduler.admit()
    assert scheduler.abort("B").status == RequestStatus.ABORTED
    assert scheduler.abort("A").cache is None
    assert not scheduler.has_unfinished()
    with pytest.raises(ValueError, match="completed"):
        scheduler.abort("A")


def sized(name, prompt, output=2):
    return Request(name, name, [3] * prompt, SamplingParams(max_new_tokens=output))


def test_kv_reservation_fifo_and_capacity_release():
    s = Scheduler(3, max_kv_tokens=10)
    a, b, c = sized("A", 4, 3), sized("B", 4, 2), sized("C", 1, 1)
    for r in (a, b, c):
        s.add(r)
    assert s.admit() == [a]  # B blocks C even though C could fit.
    a.kv_tokens = 4
    usage = s.usage()
    assert (usage.used_kv_tokens, usage.reserved_kv_tokens) == (4, 6)
    assert (usage.available_kv_tokens, usage.unreserved_kv_tokens) == (6, 4)
    s.finish(a, "length")
    assert s.admit() == [b, c]
    assert a.kv_tokens == 0
    s.abort("B")
    assert s.usage().reserved_kv_tokens == 1


def test_step_budget_decode_first_whole_prompt_fifo():
    s = Scheduler(4, max_batched_tokens=6)
    a, b, c = sized("A", 3), sized("B", 4), sized("C", 1)
    for r in (a, b, c):
        s.add(r)
    plan = s.plan_step()
    assert plan.prefill == (a,) and plan.scheduled_tokens == 3
    plan = s.plan_step()
    assert plan.decode == (a,) and plan.prefill == (b, c)
    assert s.usage().scheduled_tokens_this_step == 6
    assert s.usage().active_requests == 3
    assert s.usage().waiting_requests == 0


def test_decode_budget_and_waiting_progress():
    s = Scheduler(4, max_batched_tokens=1)
    for name in "ABC":
        s.add(sized(name, 1))
        s.admit()
    assert [r.request_id for r in s.plan_step().decode] == ["A"]
    s.finish(s.running["A"], "length")
    assert [r.request_id for r in s.plan_step().decode] == ["B"]


@pytest.mark.parametrize("kwargs", [dict(max_kv_tokens=3), dict(max_batched_tokens=3)])
def test_impossible_request_rejected_without_queue_mutation(kwargs):
    s = Scheduler(2, **kwargs)
    with pytest.raises(ValueError, match="never fit"):
        s.add(sized("A", 4))
    assert not s.has_unfinished() and not s.requests


def test_decode_growth_can_make_request_impossible():
    s = Scheduler(2, max_kv_tokens=5)
    with pytest.raises(ValueError, match="decode growth"):
        s.add(sized("A", 3, 4))
