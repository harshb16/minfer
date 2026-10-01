# Paged KV storage

[← README](../README.md) · [Scheduler](architecture.md#configuration-and-admission)

minfer's paged mode demonstrates allocation, ownership, and storage of Transformer
keys and values in fixed-size physical blocks. It **does not implement PagedAttention**.
HF still executes ordinary attention over a materialized contiguous DynamicCache.
There are no zero-copy paged attention kernels or vLLM-equivalent memory behavior.

```bash
uv run minfer generate --prompt "Explain KV blocks" --cache-mode paged \
  --kv-block-size 16 --num-kv-blocks 64 --max-new-tokens 32
```

`--cache-mode dynamic` keeps the independent DynamicCache implementation for comparison.
The same `ModelRunner`, sampler, ragged decode merge/split, and device abstraction serve
both modes. Paged storage belongs to the engine, so separate engines sharing a runner
have independent pools.

## Physical pool and logical ownership

`BlockManager` tracks free physical IDs, each request's logical block table, and its
real KV token length. IDs come from a min-heap, so allocation/reuse is deterministic.
Logical block order defines token order; physical IDs need not be adjacent:

```text
Request A: [6, 1, 4]
Request B: [2, 7]

logical token i:
  physical block = table[i // block_size]
  slot within block = i % block_size
```

For 16-token blocks, lengths 1/16/17/32/33 own 1/1/2/2/3 blocks respectively. Assignment
is lazy: crossing a boundary adds one block. Completion or cancellation frees all IDs
immediately, and later work can reuse them. Exhaustion fails without changing ownership.
Double-free and duplicate allocation raise clear errors. `validate()` checks the exact
lazy table length and that owned plus free blocks partition the physical pool without
duplicate, missing, or out-of-range IDs. Public tables are immutable tuples.

`PagedKVManager` infers layer count, KV heads, head dimension, dtype, and device from
actual HF cache tensors. It hardcodes no Qwen dimensions. On the first write it allocates
per-layer backing tensors:

```text
K and V: [num_blocks, num_kv_heads, block_size, head_dim]
```

Each layer can infer its own head shape. Later writes validate compatibility. The backing
pool is a full fixed allocation once initialized, even when few physical IDs are owned.
Lazy block ownership does not imply lazy GPU byte allocation. Freeing a request returns
IDs but retains the backing pool for reuse. Slots outside a request's real length are
never materialized, even if they contain stale data from a previous owner.

## Bridge to Hugging Face

1. **Prefill:** the existing batched prompt forward returns independent, trimmed caches.
   The manager copies their real K/V into physical blocks and records logical lengths.
2. **Before decode:** `materialize(request_id)` copies block-table order into a fresh
   batch-size-one DynamicCache with exactly the real token length. Existing ragged
   merging makes the temporary padded decode batch.
3. **After decode:** the existing split produces updated independent caches. `write()`
   persists only the suffix beyond the manager's previous length. Normal decode appends
   one token, so unchanged history is never rewritten to the pool.
4. **Finish:** request block tables are removed, physical IDs become free, and the
   request retains no persistent contiguous cache or pending token.

Materialization, ragged merging, model-side cache append, and splitting all allocate or
copy tensors. Transient model memory exists in addition to the backing pool and can
exceed the configured token capacity in bytes. This implementation teaches block tables
and fragmentation, not kernel performance or a production memory envelope.

## Admission and reservations

Scheduler policy stays outside the allocator. For each admitted request, the scheduler
reserves `ceil((prompt_length + max_new_tokens - 1) / block_size)` block credits and checks
physical availability. Credits are logical reservations, not eagerly assigned IDs.
The sum must fit `num_kv_blocks`; an optional logical `max_kv_tokens` limit also applies.
This guarantees room for active decode growth without preemption. A request that could
never fit the pool is rejected at submission, and a temporarily blocked FIFO head waits.
Reservations are released at completion even when EOS ends generation early. The tradeoff
is unused reserved capacity and head-of-line blocking.

## Metrics

`engine.paged_cache.blocks.metrics()` returns an immutable `KVMetrics` record:

| Metric | Definition |
| --- | --- |
| `total_blocks` | Configured number of physical blocks |
| `used_blocks` | Blocks currently owned by requests |
| `free_blocks` | `total_blocks - used_blocks` |
| `block_size` | Token slots per block |
| `allocated_token_capacity` | `used_blocks * block_size`; owned slots, not full backing-pool bytes |
| `real_tokens_stored` | Sum of actual persistent request KV lengths |
| `internal_fragmentation_tokens` | `allocated_token_capacity - real_tokens_stored` |
| `utilization` | `used_blocks / total_blocks`; block occupancy, not real-token fill ratio |

A request's capacity is `len(table) * block_size`; its unused final-block slots are that
capacity minus its real length. Arbitrary physical IDs avoid a need for contiguous runs
of blocks; the reported fragmentation is internal unused token slots, not external
contiguity fragmentation. Reservations are separate from allocation/utilization.

Benchmark traces record both end-of-step metrics and a within-step peak snapshot sampled
after writes, before immediate completion frees ownership. Even a one-output-token job
therefore has visible usage. Plot percentages refer to block occupancy. These are storage
metrics, not peak GPU memory measurements.

## Correctness and scope

Synthetic tests cover boundaries, partial blocks, independent owners, reuse, exhaustion,
invalid tables, double-free, metrics, append-only persistence, and independent materialized
storage. They poison unused block slots to prove those slots never reach HF. Engine tests
compare dynamic mode, paged mode, and independent manual generation; canonical CPU/MPS
tests compare materialized K/V and require identical greedy output IDs.

Only full-attention caches and Qwen2.5-0.5B-Instruct are verified. There is no custom
attention, PagedAttention, prefix sharing, preemption, chunked prefill, swapping, or
multi-device allocation.
