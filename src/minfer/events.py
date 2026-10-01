"""Immutable public streaming events and generation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int | None
    token_text: str
    finished: bool
    finish_reason: str | None


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt: str
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    latency: float
    ttft: float | None
