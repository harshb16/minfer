"""Run from the repo: uv run python -m benchmarks.benchmark --help."""

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import transformers
from transformers.generation.streamers import BaseStreamer

from minfer import EngineConfig, LLMEngine, SamplingParams
from minfer.device import allocated_memory, synchronize
from minfer.model_runner import ModelRunner

from .reference import greedy_config
from .report import RequestMeasurement, print_table, summarize
from .workloads import WorkItem, workload


class FirstTokenTimer(BaseStreamer):
    """Observe HF reference tokens without replacing its generation algorithm."""

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self.prompt_seen = False
        self.first_token_time: float | None = None

    def put(self, value: torch.Tensor) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first_token_time is None:
            synchronize(self.runner.device)
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        pass


def wait_until(timestamp: float) -> None:
    delay = timestamp - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


@torch.inference_mode()
def run_reference(runner: ModelRunner, items: list[WorkItem], max_new_tokens: int) -> dict:
    measurements = []
    synchronize(runner.device)
    start = time.perf_counter()
    for item in items:
        arrival = start + item.arrival_offset
        wait_until(arrival)
        ids = runner.tokenize(item.prompt)
        inputs = torch.tensor([ids], device=runner.device, dtype=torch.long)
        timer = FirstTokenTimer(runner)
        output = runner.model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=greedy_config(runner, max_new_tokens),
            streamer=timer,
        )
        synchronize(runner.device)
        finish = time.perf_counter()
        tokens = tuple(output[0, len(ids) :].tolist())
        measurements.append(
            RequestMeasurement(
                finish - arrival,
                None if timer.first_token_time is None else timer.first_token_time - arrival,
                tokens,
                "eos" if tokens[-1] in runner.eos_token_ids else "length",
            )
        )
    synchronize(runner.device)
    wall = time.perf_counter() - start
    return summarize("hf-sequential", wall, measurements, allocated_memory(runner.device))


def run_engine(
    runner: ModelRunner, items: list[WorkItem], max_new_tokens: int, capacity: int
) -> dict:
    engine = LLMEngine(config=replace(runner.config, max_active_requests=capacity), runner=runner)
    params = SamplingParams(max_new_tokens=max_new_tokens)
    ids = []
    labels: dict[str, int] = {}
    iteration_trace: list[dict] = []
    next_item = 0
    synchronize(runner.device)
    start = time.perf_counter()
    while next_item < len(items) or engine.has_unfinished_requests():
        now = time.perf_counter()
        while next_item < len(items) and start + items[next_item].arrival_offset <= now:
            item = items[next_item]
            request_id = engine.add_request(item.prompt, params)
            # Scheduled arrival includes queueing between step boundaries.
            engine.get_request(request_id).arrival_time = start + item.arrival_offset
            ids.append(request_id)
            labels[request_id] = next_item
            next_item += 1
        if engine.has_unfinished_requests():
            before = list(engine.scheduler.running)
            waiting = len(engine.scheduler.waiting)
            events = engine.step()
            iteration_trace.append(
                {
                    "elapsed_seconds": time.perf_counter() - start,
                    "running_before": [labels[i] for i in before],
                    "waiting_before": waiting,
                    **engine.last_step,
                    "resources": asdict(engine.scheduler.usage()),
                    "prefilled": [
                        labels[e.request_id] for e in events if e.request_id not in before
                    ],
                    "finished": [labels[e.request_id] for e in events if e.finished],
                    "running_after": [labels[i] for i in engine.scheduler.running],
                }
            )
        elif next_item < len(items):
            wait_until(start + items[next_item].arrival_offset)
    synchronize(runner.device)
    wall = time.perf_counter() - start
    measurements = []
    for request_id in ids:
        result = engine.get_result(request_id)
        measurements.append(
            RequestMeasurement(result.latency, result.ttft, result.token_ids, result.finish_reason)
        )
    summary = summarize(
        f"engine-active-{capacity}", wall, measurements, allocated_memory(runner.device)
    )
    summary["concurrency"] = capacity
    summary["cache_mode"] = engine.config.cache_mode
    summary["iteration_trace"] = iteration_trace
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=EngineConfig().model)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-mode", choices=["dynamic", "paged"], default="dynamic")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=256)
    parser.add_argument("--max-kv-tokens", type=int)
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arrival-interval", type=float, default=0.0, help="Seconds between arrivals"
    )
    parser.add_argument("--prompts", type=Path, help="JSON array of prompt strings")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.threads < 1 or any(c < 1 for c in args.concurrency):
        parser.error("warmup must be nonnegative; threads and concurrency must be positive")
    SamplingParams(max_new_tokens=args.max_new_tokens)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    prompts = json.loads(args.prompts.read_text()) if args.prompts else None
    items = workload(args.requests, args.seed, args.arrival_interval, prompts)
    config = EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        cache_mode=args.cache_mode,
        kv_block_size=args.kv_block_size,
        num_kv_blocks=args.num_kv_blocks,
        max_kv_tokens=args.max_kv_tokens,
        max_batched_tokens=args.max_batched_tokens,
    )
    runner = ModelRunner(config)
    for item in items:
        if len(runner.tokenize(item.prompt)) + args.max_new_tokens - 1 > (
            runner.model.config.max_position_embeddings
        ):
            parser.error("Workload exceeds model context limit")
    warmup_items = [WorkItem(items[0].prompt, 0)]
    for _ in range(args.warmup):
        run_reference(runner, warmup_items, min(4, args.max_new_tokens))
    rows = [run_reference(runner, items, args.max_new_tokens)]
    capacities = list(dict.fromkeys([1, *args.concurrency]))
    for capacity in capacities:
        for _ in range(args.warmup):
            # Exercise a real multi-row decode during batched warmup.
            run_engine(
                runner,
                [WorkItem(item.prompt, 0) for item in items[:capacity]],
                min(4, args.max_new_tokens),
                capacity,
            )
        row = run_engine(runner, items, args.max_new_tokens, capacity)
        if [r["token_ids"] for r in row["per_request"]] != [
            r["token_ids"] for r in rows[0]["per_request"]
        ]:
            raise AssertionError(f"Greedy output parity failed for capacity {capacity}")
        rows.append(row)
    report = {
        "metadata": {
            "model": args.model,
            "cache_mode": args.cache_mode,
            "kv_block_size": args.kv_block_size,
            "num_kv_blocks": args.num_kv_blocks,
            "max_kv_tokens": args.max_kv_tokens,
            "max_batched_tokens": args.max_batched_tokens,
            "device": str(runner.device),
            "dtype": str(runner.dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": args.seed,
            "warmup": args.warmup,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "arrival_interval_seconds": args.arrival_interval,
            "prompt_token_lengths": [len(runner.tokenize(i.prompt)) for i in items],
            "prompts": [i.prompt for i in items],
            "note": "Sequential HF baseline; model loading excluded. Queueing included. "
            "Memory is current tensor allocation, not peak; CPU memory is N/A. "
            "Percentiles use interpolation; small samples are descriptive only.",
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print_table(rows)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
