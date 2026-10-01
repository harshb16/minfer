import json
from dataclasses import replace
from unittest.mock import patch

import pytest
from PIL import Image

from benchmarks.benchmark import run_engine, run_reference
from benchmarks.plot import CHARTS, concurrency, generate_report, main, observations
from benchmarks.workloads import WorkItem


def test_measured_report_all_charts_and_legacy_names(tiny_runner, tmp_path):
    items = [WorkItem("ab", 0), WorkItem("cdefg", 0)]
    reference = run_reference(tiny_runner, items, 3)
    with patch.object(tiny_runner, "config", replace(tiny_runner.config, cache_mode="paged")):
        rows = [
            reference,
            run_engine(tiny_runner, items, 3, 1),
            run_engine(tiny_runner, items, 3, 2),
        ]
    source = tmp_path / "measured.json"
    source.write_text(
        json.dumps(
            dict(
                metadata=dict(
                    model="tiny Qwen",
                    device="cpu",
                    dtype="float32",
                    max_new_tokens=3,
                    prompt_token_lengths=[2, 5],
                ),
                results=rows,
            )
        )
    )
    output = tmp_path / "report"
    report = generate_report(source, output)
    text = report.read_text()
    assert "HF" in text and "hf-sequential" in text
    assert "Prompt token lengths:** [2, 5]" in text
    assert "Paged KV block utilization peaked at" in text
    assert "ms)" in text and "tiny Qwen" in text
    for filename in CHARTS:
        with Image.open(output / filename) as image:
            assert image.width > 500 and image.height > 300
            image.verify()
        assert f"]({filename})" in text
    assert concurrency(dict(name="engine-active-4")) == 4
    assert concurrency(reference) is None
    assert all(
        step["prefill_tokens"] + step["decode_batch_size"] == step["scheduled_tokens_this_step"]
        for step in rows[2]["iteration_trace"]
    )
    assert rows[2]["iteration_trace"][0]["prefill_batch_size"] == 2


def test_missing_data_skips_charts_and_removes_stale_output(tiny_runner, tmp_path):
    row = run_engine(tiny_runner, [WorkItem("abc", 0)], 2, 1)
    row.pop("iteration_trace")
    source = tmp_path / "single.json"
    source.write_text(json.dumps(dict(results=[row])))
    output = tmp_path / "plots"
    output.mkdir()
    (output / "kv-utilization.png").write_text("stale")
    text = generate_report(source, output).read_text()
    assert (output / "throughput.png").exists()
    assert (output / "requests.png").exists()
    for name in ("ttft.png", "latency.png", "batches.png", "kv-utilization.png"):
        assert not (output / name).exists()
        assert f"]({name})" not in text
    assert "plots omitted" in text and "N/A" in text
    assert observations([row]) == []


def test_plot_cli_and_invalid_results(tmp_path):
    source = tmp_path / "empty.json"
    source.write_text(json.dumps(dict(results=[])))
    with pytest.raises(ValueError, match="measured results"):
        generate_report(source, tmp_path / "out")
    with pytest.raises(SystemExit) as error:
        main([str(source), "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
