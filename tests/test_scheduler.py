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
