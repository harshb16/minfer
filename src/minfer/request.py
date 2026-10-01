"""Explicit request state and timing."""

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .config import SamplingParams

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache


class RequestStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    generated_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    cache: "DynamicCache | None" = None
    pending_token: int | None = None
    arrival_time: float = field(default_factory=perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None
    emitted_text: str = ""
    generator: torch.Generator = field(default_factory=torch.Generator, repr=False)
    kv_tokens: int = 0

    def __post_init__(self) -> None:
        if self.sampling_params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(self.sampling_params.seed)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def max_kv_tokens(self) -> int:
        """Worst-case consumed history; the final generated token is never cached."""
        return self.prompt_length + self.sampling_params.max_new_tokens - 1
