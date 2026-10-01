"""Educational block-backed KV storage with a copying HF DynamicCache bridge.

This is allocation and storage, not PagedAttention. HF still reads contiguous K/V.
Physical tensors per layer use [block, kv_head, token_in_block, head_dim].
"""

import heapq
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .cache import HFCacheManager, LayerKV


@dataclass(frozen=True)
class KVMetrics:
    total_blocks: int
    used_blocks: int
    free_blocks: int
    block_size: int
    allocated_token_capacity: int
    real_tokens_stored: int
    internal_fragmentation_tokens: int
    utilization: float


class BlockManager:
    """Own physical IDs and logical tables; never decide scheduling policy."""

    def __init__(self, block_size: int = 16, num_blocks: int = 256) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("Block size and count must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._lengths: dict[str, int] = {}

    def blocks_needed(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("Token count must be nonnegative")
        return (tokens + self.block_size - 1) // self.block_size

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def contains(self, request_id: str) -> bool:
        return request_id in self._tables

    def table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._tables[request_id])

    def length(self, request_id: str) -> int:
        return self._lengths[request_id]

    def allocate(self, request_id: str, tokens: int) -> None:
        if self.contains(request_id):
            raise ValueError("Request already owns blocks")
        if tokens < 1:
            raise ValueError("Initial KV length must be positive")
        count = self.blocks_needed(tokens)
        if count > self.free_blocks:
            raise MemoryError("KV block pool exhausted")
        self._tables[request_id] = [heapq.heappop(self._free) for _ in range(count)]
        self._lengths[request_id] = tokens

    def grow(self, request_id: str, tokens: int) -> None:
        if tokens < self.length(request_id):
            raise ValueError("KV history cannot shrink")
        extra = self.blocks_needed(tokens) - len(self._tables[request_id])
        if extra > self.free_blocks:
            raise MemoryError("KV block pool exhausted")
        self._tables[request_id].extend(heapq.heappop(self._free) for _ in range(extra))
        self._lengths[request_id] = tokens

    def free(self, request_id: str) -> None:
        if not self.contains(request_id):
            raise ValueError("Request owns no blocks (already freed or unknown)")
        for block in self._tables.pop(request_id):
            heapq.heappush(self._free, block)
        self._lengths.pop(request_id)

    def metrics(self) -> KVMetrics:
        used = self.num_blocks - self.free_blocks
        capacity = used * self.block_size
        real = sum(self._lengths.values())
        return KVMetrics(
            self.num_blocks,
            used,
            self.free_blocks,
            self.block_size,
            capacity,
            real,
            capacity - real,
            used / self.num_blocks,
        )

    def validate(self) -> None:
        """Validate ownership, exact lazy allocation, and the pool partition."""
        if self._tables.keys() != self._lengths.keys():
            raise ValueError("Block tables and token lengths disagree")
        owned = []
        for request_id, table in self._tables.items():
            length = self._lengths[request_id]
            if length < 1 or len(table) != self.blocks_needed(length):
                raise ValueError("Invalid block table capacity for real token length")
            owned.extend(table)
        partition = owned + self._free
        if len(partition) != self.num_blocks or set(partition) != set(range(self.num_blocks)):
            raise ValueError("Invalid block ownership: duplicate, missing, or out-of-range block")


class PagedKVManager:
    """Infer layer shapes from real caches, store blocks, materialize for HF."""

    def __init__(self, block_size: int = 16, num_blocks: int = 256) -> None:
        self.blocks = BlockManager(block_size, num_blocks)
        self.hf = HFCacheManager()
        self._storage: list[LayerKV] = []

    def _initialize(self, layers: list[LayerKV]) -> None:
        if not self._storage:
            self._storage = [
                (
                    k.new_empty(
                        (self.blocks.num_blocks, k.shape[1], self.blocks.block_size, k.shape[3])
                    ),
                    v.new_empty(
                        (self.blocks.num_blocks, v.shape[1], self.blocks.block_size, v.shape[3])
                    ),
                )
                for k, v in layers
            ]
        if len(layers) != len(self._storage):
            raise ValueError("Incompatible KV layer count")
        for (k, v), (pk, pv) in zip(layers, self._storage, strict=True):
            for source, pool in ((k, pk), (v, pv)):
                if (source.shape[1], source.shape[3], source.device, source.dtype) != (
                    pool.shape[1],
                    pool.shape[3],
                    pool.device,
                    pool.dtype,
                ):
                    raise ValueError("Incompatible KV shapes, devices or dtypes")

    @torch.inference_mode()
    def write(self, request_id: str, cache: DynamicCache) -> None:
        """Persist initial prefill or appended suffix, leaving history untouched."""
        self.hf.validate(cache)
        layers = self.hf.layers(cache)
        self._initialize(layers)
        length = self.hf.length(cache)
        start = self.blocks.length(request_id) if self.blocks.contains(request_id) else 0
        if start:
            self.blocks.grow(request_id, length)
        else:
            self.blocks.allocate(request_id, length)
        table = self.blocks.table(request_id)
        offset = start
        while offset < length:
            block_index, slot = divmod(offset, self.blocks.block_size)
            count = min(length - offset, self.blocks.block_size - slot)
            self._copy(layers, table[block_index], slot, offset, count)
            offset += count

    def _copy(self, layers: list[LayerKV], block: int, slot: int, offset: int, count: int) -> None:
        for (k, v), (pk, pv) in zip(layers, self._storage, strict=True):
            pk[block, :, slot : slot + count, :] = k[0, :, offset : offset + count, :]
            pv[block, :, slot : slot + count, :] = v[0, :, offset : offset + count, :]

    @torch.inference_mode()
    def materialize(self, request_id: str) -> DynamicCache:
        """Copy only genuine KV columns; uninitialized pool slots are never read."""
        length = self.blocks.length(request_id)
        table = self.blocks.table(request_id)
        layers = []
        for pk, pv in self._storage:
            k = pk.new_empty((1, pk.shape[1], length, pk.shape[3]))
            v = pv.new_empty(k.shape)
            for logical, block in enumerate(table):
                start = logical * self.blocks.block_size
                count = min(self.blocks.block_size, length - start)
                k[0, :, start : start + count] = pk[block, :, :count]
                v[0, :, start : start + count] = pv[block, :, :count]
            layers.append((k, v))
        return self.hf.from_layers(layers)
