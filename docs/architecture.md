# Architecture and scheduler

[← README](../README.md) · [Paged KV](paged-kv.md) · [Benchmarks](benchmarks.md)

Hugging Face loads the tokenizer/checkpoint and computes Transformer forwards.
minfer retains its manual prefill/decode loop, per-request sampling, stopping,
streaming, and FIFO scheduling. The engine does not call `model.generate()`;
that API appears only in the external benchmark reference and correctness tests.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated engine configuration and immutable sampling parameters |
| `request.py`, `events.py` | Lifecycle records, streaming deltas, final results |
| `device.py` | Device selection, dtype, synchronization, current tensor memory |
| `model_runner.py` | HF loading, tokenization, prefill/decode forwards, independent oracle |
| `cache.py` | DynamicCache validation, merge, split, clone, padding removal |
| `paged_cache.py` | Physical block ownership, storage, and contiguous HF bridge |
| `scheduler.py` | FIFO admission, resource accounting, step plans, completion/cancellation |
| `engine.py` | Executes plans, updates KV state, samples, emits events |
| `benchmarks/` | Controlled HF reference, workload, measurement, charts, reports |

## Configuration and admission

`EngineConfig` and `LLMEngine` accept:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_active_requests` | 4 | Maximum RUNNING request count |
| `max_batched_tokens` | `None` | Maximum real tokens scheduled per logical step; `None` means unlimited |
| `max_kv_tokens` | `None` | Logical KV token capacity; `None` disables this additional limit |
| `cache_mode` | `dynamic` | Independent DynamicCaches or `paged` block storage |
| `kv_block_size` | 16 | Token slots per physical KV block |
| `num_kv_blocks` | 256 | Physical block pool capacity in paged mode |

The generation CLI and benchmark expose the corresponding hyphenated flags.
`max_active_requests` remains supported, and all original positional dataclass fields
keep their ordering. `max_batched_tokens` counts real prompt tokens, excluding artificial
padding, plus one consumed decode token for each scheduled running request. It is a
work-accounting abstraction: padded forwards can compute more tensor columns.

A request's persistent history grows to at most:

```text
prompt_length + max_new_tokens - 1
```

The final sampled token never needs to be consumed. Admission reserves this worst-case
length for every active request. `used_kv_tokens` reports actual consumed history, not
reservations. Reserving growth prevents several valid prompts from filling the pool and
then all stalling during decode. It is intentionally conservative: EOS may release a
large unused reservation. There is no preemption or waiting for active caches to shrink.

In paged mode the scheduler additionally reserves the sum of each request's rounded
worst-case block count. The block manager checks physical availability and allocates
IDs lazily as actual history grows. An explicit `max_kv_tokens` still applies alongside
block capacity. Logical available-token counts cannot account for block rounding;
inspect the block metrics for allocation availability and fragmentation.

Requests are admitted in submission order, stopping at the first head that cannot fit
active count, KV reservations, blocks, or remaining step budget. Smaller later requests
cannot overtake that head. Requests whose whole prompt exceeds the total per-step budget,
or whose worst-case history exceeds KV/pool capacity, fail at submission without entering
the queue. Model context overflow also fails early. There is no prompt truncation.

## Step semantics

`Scheduler.plan_step()` makes an explicit immutable `StepPlan`:

1. Snapshot previously RUNNING requests in admission order.
2. Schedule one decode token for each, up to the step token budget.
3. Spend the remainder on whole FIFO prompts that pass admission checks.

If running work exceeds a token budget, its FIFO prefix is scheduled and the rest stays
active. Finite older requests eventually complete. Whole prompts may wait until running
work finishes and enough step budget becomes available; there is no partial/chunked prefill.

The engine executes one batched prefill forward for admissions, then one decode forward
for the scheduled previous snapshot. Decode work is **reserved first**, even though its
forward executes after prefill. These are separate HF forwards, not a mixed kernel.
New requests are absent from the decode snapshot, so every request emits at most one
token per `step()`. Completions and aborts release DynamicCaches or block ownership
immediately; freed admission slots are reused on the next step boundary. There is no
same-step refill loop for one-token jobs.

Cache-mutating forward/storage failures abort affected work and propagate the error.
Other previously running requests remain valid after an admission prefill failure.

Inspect accounting with `engine.scheduler.usage()`:

- `active_requests`, `waiting_requests`: current lifecycle counts.
- `used_kv_tokens`: actual real KV tokens stored across active requests.
- `reserved_kv_tokens`: sum of active requests' worst-case history lengths.
- `available_kv_tokens`: effective token capacity minus used tokens.
- `unreserved_kv_tokens`: effective token capacity minus reservations.
- `scheduled_tokens_this_step`: real prefill tokens plus scheduled decode tokens.

The effective token capacity is the logical limit, the physical pool token capacity in
paged mode, or their minimum when both apply. Without either it is unlimited, represented
by `None` for the available counts. Paged rounding can constrain admission before a logical
available count reaches zero. `engine.last_step` records actual batch sizes and token work;
paged mode also records end-of-step and within-step peak block metrics.

## Autoregression and cache invariants

```text
RUNNING request:
  KV history = all tokens already consumed by the Transformer
  pending_token = latest generated token, not yet consumed

After prefill: cache = prompt; pending = first generated token
After decode:  cache = prompt + consumed output; pending = next token

real KV length = prompt_length + num_generated_tokens - 1
```

At rest in dynamic mode, each request owns a batch-size-one full-attention DynamicCache.
In paged mode, `request.cache` stays `None`; `request.kv_tokens` and the manager's logical
block table describe its persistent state. Finished/aborted records retain results but no
KV ownership. EOS takes precedence over length when both stop conditions hold.

`ModelRunner.prefill()` remains the single-request correctness oracle.
`generate_independent()` manually prefills once, then repeatedly consumes one pending
token with `decode_one()`; it delegates no generation loop to HF.

## Batched prefill

`prefill_batch()` left-pads unequal prompts so every final real token occupies the last
column. For prompt lengths 3 and 5:

```text
input_ids:      [PAD PAD a b c]    [d e f g h]
attention_mask: [ 0   0  1 1 1]    [1 1 1 1 1]
position_ids:   [ 0   0  0 1 2]    [0 1 2 3 4]
```

Positions are `(attention_mask.cumsum(-1) - 1).clamp_min(0)`. Real tokens retain semantic
positions 0, 1, 2, …, so padding never shifts RoPE. One HF forward returns final-column
logits for each request and a padded batched cache. `split_prefill()` selects each row,
trims every artificial left-padding column, and clones its K/V tensors. Cloning avoids
retaining a whole batch allocation through a short request's tensor views. Each request
samples its first token from its final real prompt position. A single prompt uses the
existing `prefill()` path.

## Ragged decode and positions

For history lengths 10, 6, and 14, `HFCacheManager.merge()` creates:

```text
K/V: [3, kv_heads, 14, head_dim]
A: [4 padding columns | 10 real KV columns]
B: [8 padding columns |  6 real KV columns]
C: [                   14 real KV columns]
input_ids: [3, 1]
attention_mask: [3, 15], masking left padding, including the new token
position_ids: [[10], [6], [14]]
```

Keys already contain their semantic RoPE rotations; moving storage columns does not
rotate them again. The appended token's physical column is 14 for all rows, while its
semantic position is its genuine history length. Transformers 5.18 Qwen derives its
physical causal-mask offset from DynamicCache and does not accept `cache_position`.
After the forward, `split()` trims padding and clones independent lengths 11, 7, and 15.
Paged mode materializes its histories before this same merge/split path and writes only
the newly consumed suffix back to blocks. Only full-attention layers are supported;
sliding-window models are rejected.

## Runtime, sampling, and streaming

Automatic device priority is CUDA → MPS → CPU. Default dtypes are CUDA bfloat16 when
supported (otherwise float16), MPS float16, CPU float32. Explicit unavailable backends
raise errors. HF attention uses PyTorch SDPA; no custom kernel or FlashAttention dependency
is required. The lockfile pins the tested Transformers cache API generation.

`temperature=0` takes raw argmax without advancing RNG. Otherwise temperature, top-k,
then top-p filtering apply; top-p retains the threshold-crossing candidate. Each request
owns a seeded CPU generator and samples CPU float32 probabilities, keeping its random
stream independent of unrelated requests and portable to MPS. Numerical differences
between devices/dtypes or batch shapes can still change nearly tied logits or samples.

`add_request()`, `step()`, `get_request()`, `get_result()`, `generate()`, and
`generate_batch()` share the same loop. Convenience calls drain all outstanding work.
`abort_request()` returns an immediate terminal event (`token_id=None`), rather than
queuing it for another step. Streaming events contain incremental text; incomplete
Unicode replacement characters are buffered. EOS may emit empty text. Results retain
output IDs including terminal EOS, finish reason, TTFT, and latency from submission,
including tokenization and queueing. The engine is synchronous and intended for one
calling thread; add arrivals between steps. Completed records accumulate for inspection.

See the [late-arrival example](../examples/dynamic_batch.py),
[validation record](validation.md), and [paged KV limitations](paged-kv.md).
