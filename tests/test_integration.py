import pytest
import torch

from benchmarks.reference import greedy_config
from minfer import EngineConfig, LLMEngine, SamplingParams
from minfer.model_runner import ModelRunner
from tests.test_cache import assert_ragged_parity

pytestmark = pytest.mark.integration

PROMPTS = [
    "Say hello.",
    "Explain why the Transformer KV cache avoids repeating previous work.",
    "A language model processes prompts before generating each new token. "
    "Incoming requests have different lengths and arrive at different times. "
    "Explain in one sentence how continuous decode batching can handle this workload.",
]


@pytest.fixture(scope="module", params=["cpu", "mps"])
def qwen_runner(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.set_num_threads(4)
    return ModelRunner(EngineConfig(device=request.param))


def test_qwen_manual_greedy_hf_parity(qwen_runner):
    runner = qwen_runner
    for prompt in PROMPTS[:2]:
        ids = runner.tokenize(prompt)
        ours = runner.generate_independent(ids, SamplingParams(max_new_tokens=12))
        with torch.inference_mode():
            reference = runner.model.generate(
                torch.tensor([ids], device=runner.device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=runner.device),
                generation_config=greedy_config(runner, 12),
            )[0, len(ids) :].tolist()
        assert ours == reference


def test_qwen_cached_full_forward_parity(qwen_runner):
    runner = qwen_runner
    ids = runner.tokenize(PROMPTS[0])
    state = runner.prefill(ids)
    atol, rtol = (3e-2, 3e-2) if runner.device.type == "mps" else (1e-4, 1e-4)
    for _ in range(3):
        ids.append(int(state.logits[0].argmax()))
        state = runner.decode_one(ids[-1], state.cache)
        with torch.inference_mode():
            full = runner.model(
                input_ids=torch.tensor([ids], device=runner.device),
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1]
        torch.testing.assert_close(state.logits, full, atol=atol, rtol=rtol)
        assert int(state.logits.argmax()) == int(full.argmax())


def test_qwen_ragged_multistep_parity(qwen_runner):
    ids = [qwen_runner.tokenize(p) for p in PROMPTS]
    assert len(set(map(len, ids))) == 3
    atol, rtol = (3e-2, 3e-2) if qwen_runner.device.type == "mps" else (1e-4, 1e-4)
    assert_ragged_parity(qwen_runner, ids, atol=atol, rtol=rtol)


def test_qwen_engine_batch_and_late_arrival_parity(qwen_runner):
    runner = qwen_runner
    params = SamplingParams(max_new_tokens=10)
    expected = [runner.generate_independent(runner.tokenize(p), params) for p in PROMPTS]
    engine = LLMEngine(
        config=EngineConfig(device=str(runner.device), max_active_requests=2), runner=runner
    )
    ids = [engine.add_request(PROMPTS[0], params)]
    engine.step()
    engine.step()
    ids.extend(engine.add_request(p, params) for p in PROMPTS[1:])
    while engine.has_unfinished_requests():
        engine.step()
    assert [list(engine.get_result(i).token_ids) for i in ids] == expected
    assert not engine.scheduler.running
    assert all(engine.get_request(i).cache is None for i in ids)
