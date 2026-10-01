"""FIFO admission with independent active and completed request records."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

from .paged_cache import BlockManager
from .request import Request, RequestStatus


@dataclass(frozen=True)
class ResourceUsage:
    active_requests: int
    waiting_requests: int
    used_kv_tokens: int
    reserved_kv_tokens: int
    available_kv_tokens: int | None
    unreserved_kv_tokens: int | None
    scheduled_tokens_this_step: int


@dataclass(frozen=True)
class StepPlan:
    decode: tuple[Request, ...]
    prefill: tuple[Request, ...]
    scheduled_tokens: int


class Scheduler:
    def __init__(
        self,
        max_active_requests: int,
        max_batched_tokens: int | None = None,
        max_kv_tokens: int | None = None,
        block_manager: BlockManager | None = None,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        for value in (max_batched_tokens, max_kv_tokens):
            if value is not None and value < 1:
                raise ValueError("Token budgets must be positive or None")
        self.max_active_requests = max_active_requests
        self.max_batched_tokens = max_batched_tokens
        self.max_kv_tokens = max_kv_tokens
        self.block_manager = block_manager
        self.scheduled_tokens_this_step = 0
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.finished: dict[str, Request] = {}
        self.requests: dict[str, Request] = {}

    def add(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate request ID: {request.request_id}")
        if request.status != RequestStatus.WAITING:
            raise ValueError("Only WAITING requests can be submitted")
        if self.max_batched_tokens is not None and request.prompt_length > self.max_batched_tokens:
            raise ValueError("Prompt can never fit max_batched_tokens (no chunked prefill)")
        if self.max_kv_tokens is not None and request.max_kv_tokens > self.max_kv_tokens:
            raise ValueError("Request can never fit max_kv_tokens including decode growth")
        if (
            self.block_manager is not None
            and self.block_manager.blocks_needed(request.max_kv_tokens)
            > self.block_manager.num_blocks
        ):
            raise ValueError("Request can never fit KV block pool including decode growth")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def usage(self) -> ResourceUsage:
        used = sum(r.kv_tokens for r in self.running.values())
        reserved = sum(r.max_kv_tokens for r in self.running.values())
        capacity = self.max_kv_tokens
        if self.block_manager is not None:
            physical = self.block_manager.num_blocks * self.block_manager.block_size
            capacity = physical if capacity is None else min(capacity, physical)
        return ResourceUsage(
            len(self.running),
            len(self.waiting),
            used,
            reserved,
            None if capacity is None else capacity - used,
            None if capacity is None else capacity - reserved,
            self.scheduled_tokens_this_step,
        )

    def admit(self, token_budget: int | None = None) -> list[Request]:
        """Whole prompts in FIFO order; a blocked head prevents overtaking.

        Reserve worst-case decode growth to guarantee progress without preemption.
        Reservations are accounting only, not eagerly allocated cache tensors.
        """
        admitted = []
        budget = self.max_batched_tokens if token_budget is None else token_budget
        reserved = self.usage().reserved_kv_tokens
        blocks = self.block_manager
        reserved_blocks = (
            0
            if blocks is None
            else sum(blocks.blocks_needed(r.max_kv_tokens) for r in self.running.values())
        )
        while self.waiting and len(self.running) < self.max_active_requests:
            request = self.waiting[0]
            if budget is not None and request.prompt_length > budget:
                break
            if (
                self.max_kv_tokens is not None
                and reserved + request.max_kv_tokens > self.max_kv_tokens
            ):
                break
            if blocks is not None:
                required = blocks.blocks_needed(request.max_kv_tokens)
                if (
                    reserved_blocks + required > blocks.num_blocks
                    or blocks.blocks_needed(request.prompt_length) > blocks.free_blocks
                ):
                    break
                reserved_blocks += required
            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            admitted.append(request)
            reserved += request.max_kv_tokens
            if budget is not None:
                budget -= request.prompt_length
        return admitted

    def plan_step(self) -> StepPlan:
        """Reserve FIFO decode tokens first, then spend the remainder on prefill."""
        previous = tuple(self.running.values())
        if self.max_batched_tokens is not None:
            previous = previous[: self.max_batched_tokens]
        remaining = (
            None if self.max_batched_tokens is None else self.max_batched_tokens - len(previous)
        )
        admitted = tuple(self.admit(remaining))
        self.scheduled_tokens_this_step = len(previous) + sum(r.prompt_length for r in admitted)
        return StepPlan(previous, admitted, self.scheduled_tokens_this_step)

    def finish(self, request: Request, reason: str) -> None:
        if request.request_id not in self.running:
            raise ValueError("Only active requests can finish")
        self.running.pop(request.request_id)
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        request.finish_time = perf_counter()
        if self.block_manager is not None and self.block_manager.contains(request.request_id):
            self.block_manager.free(request.request_id)
        request.cache = None
        request.kv_tokens = 0
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
        if self.block_manager is not None and self.block_manager.contains(request.request_id):
            self.block_manager.free(request.request_id)
        request.cache = None
        request.kv_tokens = 0
        request.pending_token = None
        self.finished[request_id] = request
        return request

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
