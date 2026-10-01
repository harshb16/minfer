"""Small, validated configuration objects."""

import math
from dataclasses import dataclass

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class EngineConfig:
    model: str = MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    max_active_requests: int = 4
    max_batched_tokens: int | None = None
    max_kv_tokens: int | None = None
    chat_template: bool = True
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        for name in ("max_batched_tokens", "max_kv_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    stop_on_eos: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
