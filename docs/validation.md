# Validation record

[← README](../README.md)

Validated on **2026-10-02**, using Python 3.12.14, PyTorch 2.14.1,
Transformers 5.18.0, and the locked environment on macOS arm64.

## Baseline before changes

`uv run pytest` passed **30 fast tests**, with eight canonical-checkpoint integration
cases deselected. The initial repository was clean. The inspection covered the engine,
request lifecycle, DynamicCache merge/split, FIFO scheduler, sampling, device abstraction,
all tests, CLI/examples, benchmark reference/workload/traces, and README/architecture guide.

The restricted environment could not use the default uv cache directory; commands used
`UV_CACHE_DIR=.cache/uv`. Model runs used `HF_HOME=.cache/huggingface`. Plotting used
`MPLCONFIGDIR=.cache/matplotlib`. These ignored directories are not in the distributions.

## Milestone validation

| Milestone | Fast tests | Canonical tests actually run |
| --- | ---: | --- |
| Token-budget admission | 37 passed | Deferred until prefill milestone |
| Batched prefill | 40 passed | Five CPU cases passed; sandbox hid MPS |
| Paged KV | 62 passed | All 12 CPU/MPS cases passed with host device access |
| Plots and reports | 65 passed | Earlier canonical results retained; model path unchanged |
| Positional API compatibility | 66 passed | Final canonical run below |

Ruff lint and formatting passed before each feature/compatibility commit. A final
format check caught the README's Python example formatting; it was corrected before
building and committing documentation.

## Final checks

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
uv run pytest -m integration
```

- Locked sync succeeded.
- Fast suite: **66 passed**, 12 integration cases deselected.
- Canonical suite with host CPU/MPS access: **12 passed**, no skips.
- Ruff lint and formatting passed.
- Wheel and source distribution built successfully; inspected archives contain the
  paged cache module and no `.cache` entries.

Canonical cases run on **both CPU float32 and MPS float16**: manual/HF greedy parity,
cached/full-sequence logits, multi-step ragged decode logits/K/V, late-arrival engine
parity, unequal-length batched prefill logits/K/V/positions, and paged/dynamic/independent
engine parity with materialized K/V comparisons. Greedy output IDs must match exactly.

The new MPS float16 batched-prefill comparison uses `atol=0.08, rtol=0.03` after measuring
maximum logit drift of 0.07324 between padded and independent forward shapes. Existing
MPS decode tolerances remain `0.03`; canonical CPU tolerances remain `1e-4`, with tighter
tiny-model tolerances. This is a numerical batch-shape limitation, not cross-device
bitwise reproducibility. CUDA was unavailable and is not claimed as tested.

## Real generation, benchmarks, and reports

A real paged-mode CPU CLI run completed using the canonical checkpoint:

```bash
uv run minfer generate --device cpu --cache-mode paged --kv-block-size 16 \
  --num-kv-blocks 32 --prompt "Say hello." --max-new-tokens 4
```

The curated CPU dynamic and paged benchmarks were collected **sequentially**, using:

```bash
uv run python -m benchmarks.benchmark --device cpu --requests 4 --concurrency 1 2 4 \
  --max-new-tokens 8 --warmup 1 --cache-mode dynamic \
  --max-batched-tokens 256 --max-kv-tokens 512 \
  --output benchmarks/results/cpu-budget-dynamic.json
uv run python -m benchmarks.benchmark --device cpu --requests 4 --concurrency 1 2 4 \
  --max-new-tokens 8 --warmup 1 --cache-mode paged --kv-block-size 16 --num-kv-blocks 32 \
  --max-batched-tokens 256 --max-kv-tokens 512 \
  --output benchmarks/results/cpu-budget-paged.json
uv run python -m benchmarks.plot benchmarks/results/cpu-budget-dynamic.json
uv run python -m benchmarks.plot benchmarks/results/cpu-budget-paged.json
uv run python -m benchmarks.plot benchmarks/results/mps-historical.json
```

Every measured method generated 32 tokens with exact HF reference parity. Concurrency
four performed a four-request, 217-real-token prefill batch. All six paged charts were
visually inspected, including labels, units, legends, and overlap of active/decode traces.
The generator was also tested on missing percentiles/traces and legacy method names.
Historical benchmark JSON was retained byte-for-byte under `benchmarks/results/`.

See [benchmark methodology and limitations](benchmarks.md) and
[paged storage tradeoffs](paged-kv.md).
