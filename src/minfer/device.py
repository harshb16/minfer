"""All backend-specific operations live here."""

import torch


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return device


def select_dtype(device: torch.device, name: str = "auto") -> torch.dtype:
    if name != "auto":
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            name
        ]
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
    return torch.float16


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def allocated_memory(device: torch.device) -> int | None:
    """Current tensor allocation, not peak memory or total process RSS."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None
