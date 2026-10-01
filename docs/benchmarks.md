# Benchmarks and generated reports

[← README](../README.md) · [Dynamic smoke report](../benchmarks/reports/cpu-budget-dynamic/report.md) ·
[Paged smoke report](../benchmarks/reports/cpu-budget-paged/report.md)

Run benchmark modules from the repository root. Matplotlib is in the development
dependency group installed by `uv sync --locked`; it is not an engine runtime dependency.

```bash
uv run python -m benchmarks.benchmark --device cpu --requests 4 \
  --concurrency 1 2 4 --max-new-tokens 8 --warmup 1 \
  --max-batched-tokens 256 --max-kv-tokens 512 \
  --cache-mode paged --kv-block-size 16 --num-kv-blocks 32 \
  --output benchmarks/results/example-paged.json
uv run python -m benchmarks.plot benchmarks/results/example-paged.json \
  --output-dir benchmarks/reports/example-paged
```

Omit `--output-dir` to use `benchmarks/reports/<JSON stem>/`. The benchmark's default raw
output is `benchmarks/results/benchmark.json`. Keep measured JSON under `results/` and
curated charts/reports under `reports/`. Historical JSON is preserved: the old root
`benchmark-results.json` is now `benchmarks/results/mps-historical.json`, byte-for-byte.
Its generated report reflects its original measurements, before batched prefill.

## Workload and reference

The default seeded workload shuffles a mix of short, medium, and longer prompts, cycling
when request count exceeds the prompt set. `--prompts prompts.json` accepts a nonempty
JSON array of prompt strings. The instruction chat template contributes to recorded
prompt lengths. `--arrival-interval 0.1` schedules later arrivals rather than submitting
every request at time zero. `--threads` sets CPU intra-op threads, default four.

Methods share one model instance and identical prompts/output budgets: sequential HF
`model.generate()`, engine concurrency one, then requested capacities. The HF reference
uses ordinary DynamicCache regardless of the engine cache mode. All generation is greedy
with identical EOS behavior and explicit settings that disable the checkpoint's saved
repetition penalty of 1.1. Every engine row must match reference token IDs, or the run
fails. The benchmark always includes engine concurrency one and removes duplicate
capacities. There is no static padded HF batching baseline.

Warmup precedes each method's timed workload. Engine warmup exercises multi-row decode
when there are enough requests and the budget allows it. Model loading is excluded;
tokenization, sampling, text-delta decoding, queueing, and arrival waits are included.
Console result rendering happens after measurement. Run methods sequentially without
other benchmark processes competing for the same device. One smoke run is not a
statistical estimate of stable speed or a comparison with production engines.

## Timing and metrics

Times use `perf_counter()` in seconds. CUDA and MPS synchronize at measurement boundaries;
the reference streamer also synchronizes before recording its first output token.
Engine token selection reads the device result for sampling, so emitted-token timestamps
include completion of that forward. With synthetic arrivals, latency starts at the
scheduled arrival even if a step notices it later; queueing between step boundaries counts.

| Metric | Definition |
| --- | --- |
| Wall time | Measured workload duration, including idle arrival waits |
| Generated tokens/s | All generated token IDs, including terminal EOS, divided by wall time |
| Requests/s | Finished requests divided by wall time |
| TTFT | Scheduled/submitted arrival to first emitted token |
| End-to-end latency | Arrival to terminal event |
| p50 / p95 | Interpolation at index `(n - 1) * p` in sorted request samples |
| Device memory | Current allocated tensor memory after the run; not peak allocation or process RSS |

Percentiles are unavailable (`null`/N/A) for fewer than two samples. Small-sample p95
is descriptive. CPU memory is N/A. Output may end early at EOS, so generated counts can
be below the requested output budget. Token-capacity and block metrics are not GPU-byte
estimates; see [paged storage limitations](paged-kv.md).

Metadata records model, backend, platform, dtype, package versions, seed, warmup, CPU
threads, actual prompt lengths/texts, output budget, arrivals, and cache/scheduling
configuration. Block settings apply only in paged mode; recorded dynamic-mode defaults
do not mean a physical pool was allocated. Hardware identification here is the recorded OS/architecture and backend;
it does not capture a detailed chip/power/thermal profile.

## Iteration traces

Engine rows include stable zero-based workload indices for requests before/after each
step, waiting counts, prefills, and completions. New fields make actual work explicit:

```json
{
  "prefill_batch_size": 3,
  "prefill_tokens": 128,
  "decode_batch_size": 4,
  "scheduled_tokens_this_step": 132
}
```

This is a schema example, not a claimed measurement. `resources` contains scheduler
accounting at step end. Paged rows include `kv_metrics` at step end and `kv_metrics_peak`
after writes within the step. Completion frees blocks immediately, so the latter captures
brief usage that an end-only sample could miss. The trace does not measure GPU peak bytes
or a separate timestamp for each storage mutation.

## Plot and report generation

`benchmarks.plot` reads measurements; it never embeds hardcoded throughput/latency values.
It generates:

- Token throughput and requests/sec versus concurrency, with labeled HF reference lines.
- TTFT and end-to-end latency p50/p95 versus concurrency, displayed in milliseconds.
- Active-after-step, prefill, and actual decode batch sizes over elapsed seconds.
- Paged block occupancy over time, showing end-of-step and within-step peaks.
- `report.md` with metadata, results table, embedded PNGs, and descriptive observations.

Trace chart timestamps are step-end samples; connecting lines aid reading and do not
represent continuous measurements between samples. The generator accepts historical
method names (`engine-active-N`) when an explicit concurrency field is absent. It omits
charts with unavailable metrics, explains omissions, and removes stale generated chart
filenames on regeneration. Dynamic/historical reports do not fabricate paged metrics.

The checked-in 2026-10-02 CPU smoke runs use four prompts of 38/45/75/59 tokens, eight
new tokens each, capacities 1/2/4, one warmup, step budget 256, and logical KV budget 512.
The paged run uses 16-token blocks and 32 blocks. Both produced 32 output tokens in each
method with exact reference parity. Dynamic and paged runs were measured sequentially;
run-to-run timing differences do not establish a cache-mode speed advantage.
