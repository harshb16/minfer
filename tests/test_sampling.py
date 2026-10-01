import pytest
import torch

from minfer import SamplingParams
from minfer.sampler import Sampler


def test_greedy_ignores_filters():
    assert (
        Sampler().sample(
            torch.tensor([0.0, 4.0, 1.0]),
            SamplingParams(temperature=0, top_k=1, top_p=0.1),
            torch.Generator(),
        )
        == 1
    )


def test_temperature_top_k_top_p():
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    filtered = Sampler.filtered_logits(scores, SamplingParams(temperature=2, top_k=2))
    assert torch.isneginf(filtered[:2]).all()
    torch.testing.assert_close(filtered[2:], torch.tensor([1.0, 1.5]))
    # 0.6 then 0.3 crosses 0.7: keep both, exclude the remaining 0.1.
    filtered = Sampler.filtered_logits(
        torch.tensor([0.6, 0.3, 0.1]).log(), SamplingParams(temperature=1, top_p=0.7)
    )
    assert torch.isfinite(filtered[:2]).all()
    assert torch.isneginf(filtered[2])
    filtered = Sampler.filtered_logits(scores, SamplingParams(temperature=1, top_k=999))
    assert torch.isfinite(filtered).all()


def test_seed_reproducibility_and_no_global_rng_dependency():
    sampler = Sampler()
    params = SamplingParams(temperature=0.8, seed=42)
    a, b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    scores = torch.arange(10, dtype=torch.float32) / 10
    expected = [sampler.sample(scores, params, a) for _ in range(30)]
    actual = []
    for _ in range(30):
        torch.rand(10)
        actual.append(sampler.sample(scores, params, b))
    assert actual == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(temperature=-1),
        dict(temperature=float("nan")),
        dict(temperature=float("inf")),
        dict(top_k=-1),
        dict(top_p=0),
        dict(top_p=1.1),
        dict(max_new_tokens=0),
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
