"""Raw-logit sampling without Hugging Face generation utilities."""

import torch

from .config import SamplingParams


class Sampler:
    @staticmethod
    def filtered_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # CPU float32 sampling gives each request its own portable random stream,
        # including on MPS, which does not support a device-local Generator.
        scores = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        if scores.ndim != 1:
            raise ValueError("Expected one vocabulary vector")
        if params.temperature > 0:
            scores /= params.temperature
        if params.top_k:
            k = min(params.top_k, scores.numel())
            threshold = torch.topk(scores, k).values[-1]
            scores[scores < threshold] = -torch.inf
        if params.top_p < 1:
            sorted_scores, indices = torch.sort(scores, descending=True)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(-1)
            remove = cumulative > params.top_p
            # Retain the first token that crosses the probability threshold.
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores[indices[remove]] = -torch.inf
        return scores

    def sample(
        self, logits: torch.Tensor, params: SamplingParams, generator: torch.Generator
    ) -> int:
        if params.temperature == 0:
            return int(logits.argmax().item())
        probabilities = torch.softmax(self.filtered_logits(logits, params), dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())
