"""FIFO admission with independent active and completed request records."""

from collections import deque
from time import perf_counter

from .request import Request, RequestStatus


class Scheduler:
    def __init__(self, max_active_requests: int) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        self.max_active_requests = max_active_requests
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.finished: dict[str, Request] = {}
        self.requests: dict[str, Request] = {}

    def add(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate request ID: {request.request_id}")
        if request.status != RequestStatus.WAITING:
            raise ValueError("Only WAITING requests can be submitted")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def admit(self) -> list[Request]:
        admitted = []
        while self.waiting and len(self.running) < self.max_active_requests:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            admitted.append(request)
        return admitted

    def finish(self, request: Request, reason: str) -> None:
        if request.request_id not in self.running:
            raise ValueError("Only active requests can finish")
        self.running.pop(request.request_id)
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        request.finish_time = perf_counter()
        request.cache = None
        request.pending_token = None
        self.finished[request.request_id] = request

    def abort(self, request_id: str) -> Request:
        request = self.requests[request_id]
        if request.status in {RequestStatus.FINISHED, RequestStatus.ABORTED}:
            raise ValueError("Request already completed")
        self.running.pop(request_id, None)
        if request.status == RequestStatus.WAITING:
            self.waiting.remove(request)
        request.status = RequestStatus.ABORTED
        request.finish_reason = "aborted"
        request.finish_time = perf_counter()
        request.cache = None
        request.pending_token = None
        self.finished[request_id] = request
        return request

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
