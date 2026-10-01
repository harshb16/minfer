from unittest.mock import patch

import pytest
import torch

from benchmarks.benchmark import run_engine, run_reference
from benchmarks.report import percentile
from benchmarks.workloads import WorkItem, workload
from minfer import EngineConfig, LLMEngine
from minfer.cli import main
from minfer.device import allocated_memory, select_device, select_dtype, synchronize


def test_device_priority_and_cpu_operations():
    with patch("torch.cuda.is_available", return_value=True):
        assert select_device().type == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            assert select_device().type == "mps"
        with patch("torch.backends.mps.is_available", return_value=False):
            assert select_device().type == "cpu"
            with pytest.raises(ValueError, match="unavailable"):
                select_device("mps")
    cpu = select_device("cpu")
    assert select_dtype(cpu) == torch.float32
    assert select_dtype(cpu, "float16") == torch.float16
    assert allocated_memory(cpu) is None
    synchronize(cpu)
    with pytest.raises(ValueError, match="Unsupported"):
        select_device("meta")


def test_cli_streaming_flags_and_errors(tiny_runner, capsys):
    engine = LLMEngine(runner=tiny_runner)
    with patch("minfer.cli.LLMEngine", return_value=engine):
        assert (
            main(
                [
                    "generate",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "3",
                    "--temperature",
                    ".8",
                    "--top-k",
                    "4",
                    "--top-p",
                    ".9",
                    "--seed",
                    "42",
                ]
            )
            == 0
        )
    output = capsys.readouterr().out
    request = next(iter(engine.scheduler.finished.values()))
    assert output == tiny_runner.decode_text(request.generated_token_ids) + "\n"
    assert request.sampling_params.seed == 42
    assert main(["generate", "--prompt", "abc", "--max-new-tokens", "0"]) == 1
    assert "positive" in capsys.readouterr().err


def test_benchmark_reference_and_engine_measurements(tiny_runner):
    items = [WorkItem("abc", 0), WorkItem("a longer prompt", 0.0001)]
    reference = run_reference(tiny_runner, items, 4)
    batched = run_engine(tiny_runner, items, 4, 2)
    assert [r["token_ids"] for r in reference["per_request"]] == [
        r["token_ids"] for r in batched["per_request"]
    ]
    for row in [reference, batched]:
        assert row["requests"] == 2
        assert row["generated_tokens"] > 0
        assert row["wall_seconds"] > 0
        assert row["allocated_device_memory_bytes"] is None
        for request in row["per_request"]:
            assert request["latency_seconds"] >= request["ttft_seconds"] >= 0


def test_workloads_and_percentiles():
    a = workload(6, 42, 0.01)
    assert a == workload(6, 42, 0.01)
    assert len(set(len(item.prompt) for item in a)) >= 3
    assert a[2].arrival_offset == 0.02
    assert percentile([1], 0.95) is None
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)
    with pytest.raises(ValueError):
        workload(0, 42, 0)
    with pytest.raises(ValueError):
        workload(1, 42, 0, [])


def test_existing_positional_config_fields_keep_their_meaning():
    config = EngineConfig("model", "cpu", "float32", 2, False, "weights")
    assert config.chat_template is False and config.cache_dir == "weights"
    assert config.max_batched_tokens is None and config.max_kv_tokens is None
    assert config.cache_mode == "dynamic"
