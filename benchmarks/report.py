"""Machine-readable measurements and a compact human-readable table."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RequestMeasurement:
    latency_seconds: float
    ttft_seconds: float | None
    token_ids: tuple[int, ...]
    finish_reason: str


def percentile(values: list[float], p: float) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(
    name: str, wall: float, requests: list[RequestMeasurement], memory: int | None
) -> dict:
    latencies = [r.latency_seconds for r in requests]
    ttfts = [r.ttft_seconds for r in requests if r.ttft_seconds is not None]
    tokens = sum(len(r.token_ids) for r in requests)
    return {
        "name": name,
        "wall_seconds": wall,
        "requests": len(requests),
        "generated_tokens": tokens,
        "requests_per_second": len(requests) / wall,
        "tokens_per_second": tokens / wall,
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "ttft_p50_seconds": percentile(ttfts, 0.5),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "allocated_device_memory_bytes": memory,
        "per_request": [asdict(r) for r in requests],
    }


def print_table(rows: list[dict]) -> None:
    print(
        f"{'method':<20} {'wall(s)':>8} {'req/s':>8} {'tok/s':>8} "
        f"{'p50 lat':>9} {'p95 lat':>9} {'p50 TTFT':>9} {'p95 TTFT':>9}"
    )

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    for row in rows:
        values = [
            row[key]
            for key in [
                "wall_seconds",
                "requests_per_second",
                "tokens_per_second",
                "latency_p50_seconds",
                "latency_p95_seconds",
                "ttft_p50_seconds",
                "ttft_p95_seconds",
            ]
        ]
        print(f"{row['name']:<20} " + " ".join(f"{fmt(v):>9}" for v in values))
