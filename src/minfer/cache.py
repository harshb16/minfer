"""Transformers-version-sensitive cache operations, isolated from the engine.

Only full-attention DynamicCache layers are supported. Qwen2.5-0.5B uses these.
DynamicCache's public iterator yields (keys, values, optional sliding metadata)
in Transformers 5.18; its public data constructor reconstructs those layers.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

LayerKV = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class MergedCache:
    cache: DynamicCache
    lengths: tuple[int, ...]
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


class HFCacheManager:
    @staticmethod
    def empty() -> DynamicCache:
        return DynamicCache()

    @staticmethod
    def length(cache: DynamicCache) -> int:
        return int(cache.get_seq_length())

    @staticmethod
    def layers(cache: DynamicCache) -> list[LayerKV]:
        result = []
        for entry in cache:
            keys, values = entry[:2]
            if len(entry) > 2 and entry[2] is not None:
                raise ValueError("Sliding-window caches are outside minfer V1's scope")
            if keys is None or values is None:
                raise ValueError("Expected an initialized cache layer")
            if keys.ndim != 4 or keys.shape != values.shape:
                raise ValueError("Expected matching [batch, kv_heads, length, head_dim] K/V")
            result.append((keys, values))
        return result

    @staticmethod
    def from_layers(layers: Iterable[LayerKV]) -> DynamicCache:
        return DynamicCache(ddp_cache_data=layers)

    def clone(self, cache: DynamicCache) -> DynamicCache:
        return self.from_layers((k.clone(), v.clone()) for k, v in self.layers(cache))

    def validate(self, cache: DynamicCache, batch_size: int = 1) -> None:
        length = self.length(cache)
        layers = self.layers(cache)
        if not layers or length == 0:
            raise ValueError("Cannot decode an empty cache")
        for keys, _ in layers:
            if keys.shape[0] != batch_size or keys.shape[-2] != length:
                raise ValueError("Inconsistent cache batch size or layer length")

    def merge(self, caches: Sequence[DynamicCache]) -> MergedCache:
        """Left-pad temporary K/V so unequal histories can share one forward.

        Existing cached keys already contain RoPE at their true token positions;
        moving their storage columns does not require rotating them again.
        Padding is excluded by the mask and discarded when splitting.
        """
        if not caches:
            raise ValueError("Cannot merge an empty batch")
        for cache in caches:
            self.validate(cache)
        lengths = tuple(self.length(c) for c in caches)
        max_past = max(lengths)
        all_layers = [self.layers(c) for c in caches]
        num_layers = len(all_layers[0])
        if any(len(layers) != num_layers for layers in all_layers):
            raise ValueError("Caches have different numbers of layers")
        merged = []
        for layer_index in range(num_layers):
            prototype = all_layers[0][layer_index][0]
            shape = (len(caches), prototype.shape[1], max_past, prototype.shape[3])
            keys = prototype.new_zeros(shape)
            values = prototype.new_zeros(shape)
            for row, layers in enumerate(all_layers):
                k, v = layers[layer_index]
                if (
                    (k.shape[1], k.shape[3], k.dtype, k.device)
                    != (prototype.shape[1], prototype.shape[3], prototype.dtype, prototype.device)
                    or v.dtype != k.dtype
                    or v.device != k.device
                ):
                    raise ValueError("Caches have incompatible head shapes, devices or dtypes")
                keys[row : row + 1, :, max_past - lengths[row] :, :] = k
                values[row : row + 1, :, max_past - lengths[row] :, :] = v
            merged.append((keys, values))
        device = merged[0][0].device
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        columns = torch.arange(max_past + 1, device=device)
        mask = (columns[None, :] >= (max_past - lengths_tensor[:, None])).long()
        return MergedCache(self.from_layers(merged), lengths, mask, lengths_tensor[:, None])

    def split(self, cache: DynamicCache, old_lengths: Sequence[int]) -> list[DynamicCache]:
        """Split after exactly one consumed token; never retain padded batch storage."""
        if not old_lengths or any(length < 1 for length in old_lengths):
            raise ValueError("Expected nonempty positive cache lengths")
        self.validate(cache, batch_size=len(old_lengths))
        max_past = max(old_lengths)
        if self.length(cache) != max_past + 1:
            raise ValueError("Batched decode must append exactly one token")
        layers = self.layers(cache)
        result = []
        for row, length in enumerate(old_lengths):
            start = max_past - length
            # clone is deliberate: views would keep the entire merged allocation
            # alive after other requests finish, breaking independent ownership.
            result.append(
                self.from_layers(
                    (k[row : row + 1, :, start:, :].clone(), v[row : row + 1, :, start:, :].clone())
                    for k, v in layers
                )
            )
        return result

    def split_prefill(self, cache: DynamicCache, lengths: Sequence[int]) -> list[DynamicCache]:
        """Discard left padding and clone each row into independent storage."""
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError("Expected nonempty positive prompt lengths")
        self.validate(cache, batch_size=len(lengths))
        if self.length(cache) != max(lengths):
            raise ValueError("Prefill cache length must match longest prompt")
        layers = self.layers(cache)
        return [
            self.from_layers(
                (k[row : row + 1, :, -length:, :].clone(), v[row : row + 1, :, -length:, :].clone())
                for k, v in layers
            )
            for row, length in enumerate(lengths)
        ]
