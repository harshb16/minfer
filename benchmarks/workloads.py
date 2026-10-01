"""Deliberately unequal prompt lengths and optional scheduled arrival offsets."""

import random
from dataclasses import dataclass

DEFAULT_PROMPTS = [
    "Explain KV caching in one sentence.",
    "Why is the first token often slower than later tokens during autoregressive generation?",
    "Consider an inference engine receiving requests with unequal prompt lengths. "
    "Each active request owns a KV cache and one pending generated token. "
    "Explain how a scheduler can temporarily batch decode work without forcing "
    "every request to arrive or finish at the same time.",
    "Name two differences between prefill and decode.",
    "A Transformer stores previous keys and values instead of recomputing them for every "
    "generated token. Explain why this helps and why attention still reads earlier tokens.",
]


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    arrival_offset: float


def workload(
    count: int, seed: int, arrival_interval: float, prompts: list[str] | None = None
) -> list[WorkItem]:
    if count < 1 or arrival_interval < 0:
        raise ValueError("count must be positive and arrival_interval nonnegative")
    choices = list(DEFAULT_PROMPTS if prompts is None else prompts)
    if not choices or any(not isinstance(p, str) or not p for p in choices):
        raise ValueError("Prompt set must be a nonempty list of nonempty strings")
    random.Random(seed).shuffle(choices)
    return [WorkItem(choices[i % len(choices)], i * arrival_interval) for i in range(count)]
