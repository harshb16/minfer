"""Generate charts and report.md from measured JSON; run from the repository root."""

import argparse
import json
import re
from pathlib import Path

CHARTS = {
    "throughput.png": "Generated token throughput",
    "requests.png": "Request throughput",
    "ttft.png": "Time to first token",
    "latency.png": "End-to-end latency",
    "batches.png": "Active, prefill, and decode batches over time",
    "kv-utilization.png": "Paged KV block utilization over time",
}


def concurrency(row: dict) -> int | None:
    """Older reports encoded capacity in the method name."""
    if "concurrency" in row:
        return int(row["concurrency"])
    match = re.fullmatch(r"engine-active-(\d+)", row.get("name", ""))
    return int(match[1]) if match else None


def observations(rows: list[dict]) -> list[str]:
    engines = sorted((r for r in rows if concurrency(r) is not None), key=concurrency)
    statements = []
    if len(engines) >= 2:
        first, last = engines[0], engines[-1]
        for key, label, unit, scale in (
            ("tokens_per_second", "Generated token throughput", "tokens/s", 1),
            ("ttft_p50_seconds", "p50 TTFT", "ms", 1000),
            ("latency_p95_seconds", "p95 end-to-end latency", "ms", 1000),
        ):
            if first.get(key) is not None and last.get(key) is not None:
                statements.append(
                    f"{label} changed from {first[key] * scale:.2f} to "
                    f"{last[key] * scale:.2f} {unit} as concurrency changed "
                    f"from {concurrency(first)} to {concurrency(last)}."
                )
    utilizations = [
        step[key]["utilization"]
        for row in engines
        for step in row.get("iteration_trace", [])
        for key in ("kv_metrics", "kv_metrics_peak")
        if key in step and "utilization" in step[key]
    ]
    if utilizations:
        statements.append(f"Paged KV block utilization peaked at {max(utilizations):.1%}.")
    return statements


def generate_report(source: Path, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = json.loads(source.read_text())
    rows = data.get("results", [])
    if not rows:
        raise ValueError("Benchmark JSON must contain measured results")
    meta = data.get("metadata", {})
    engines = sorted((r for r in rows if concurrency(r) is not None), key=concurrency)
    baseline = next((r for r in rows if r.get("name") == "hf-sequential"), None)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = " · ".join(str(meta.get(k, "unrecorded")) for k in ("model", "device", "dtype"))
    generated = []

    def save(fig, filename):
        fig.suptitle(f"{CHARTS[filename]}\n{context}", fontsize=11)
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(filename)

    def style(ax, xlabel, ylabel):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)

    for filename, specs, ylabel in (
        ("throughput.png", [("tokens_per_second", "minfer", 1)], "Generated tokens/s"),
        ("requests.png", [("requests_per_second", "minfer", 1)], "Requests/s"),
        (
            "ttft.png",
            [("ttft_p50_seconds", "p50", 1000), ("ttft_p95_seconds", "p95", 1000)],
            "TTFT (ms; includes queueing)",
        ),
        (
            "latency.png",
            [("latency_p50_seconds", "p50", 1000), ("latency_p95_seconds", "p95", 1000)],
            "End-to-end latency (ms)",
        ),
    ):
        if not any(r.get(key) is not None for r in engines for key, _, _ in specs):
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        for series, (key, label, scale) in enumerate(specs):
            measured = [r for r in engines if r.get(key) is not None]
            if measured:
                ax.plot(
                    [concurrency(r) for r in measured],
                    [r[key] * scale for r in measured],
                    marker="o",
                    color=f"C{series}",
                    label=f"minfer {label}" if label != "minfer" else label,
                )
            if baseline is not None and baseline.get(key) is not None:
                ax.axhline(
                    baseline[key] * scale,
                    linestyle="--",
                    color=f"C{series}",
                    alpha=0.7,
                    label=f"HF sequential {label}" if label != "minfer" else "HF sequential",
                )
        ax.set_xticks([concurrency(r) for r in engines])
        ax.set_ylim(bottom=0)
        style(ax, "Max active requests (concurrency)", ylabel)
        save(fig, filename)

    traced = [(r, r.get("iteration_trace", [])) for r in engines]
    traced = [
        (r, t)
        for r, t in traced
        if any("elapsed_seconds" in s and "decode_batch_size" in s for s in t)
    ]
    if traced:
        fig, axes = plt.subplots(
            len(traced), 1, figsize=(8, 2.8 * len(traced)), layout="constrained", squeeze=False
        )
        for ax, (row, trace) in zip(axes[:, 0], traced, strict=True):
            valid = [s for s in trace if "elapsed_seconds" in s and "decode_batch_size" in s]
            time = [s["elapsed_seconds"] for s in valid]
            ax.step(
                time,
                [s["decode_batch_size"] for s in valid],
                where="post",
                label="Decode",
                linestyle="--",
                marker=".",
                zorder=3,
            )
            for key, label in (
                ("running_after", "Active after step"),
                ("prefill_batch_size", "Prefill"),
            ):
                values = [
                    (s["elapsed_seconds"], len(s[key]) if isinstance(s[key], list) else s[key])
                    for s in valid
                    if key in s
                ]
                if values:
                    ax.step(*zip(*values, strict=True), where="post", label=label)
            ax.set_title(f"Concurrency {concurrency(row)}", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            style(ax, "Elapsed time (s; samples at step end)", "Requests")
        save(fig, "batches.png")

    paged = [
        (
            r,
            [
                s
                for s in r.get("iteration_trace", [])
                if "elapsed_seconds" in s and "kv_metrics" in s
            ],
        )
        for r in engines
    ]
    paged = [(r, t) for r, t in paged if t]
    if paged:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for series, (row, trace) in enumerate(paged):
            ax.step(
                [s["elapsed_seconds"] for s in trace],
                [100 * s["kv_metrics"]["utilization"] for s in trace],
                where="post",
                label=f"Concurrency {concurrency(row)}: end of step",
                color=f"C{series}",
            )
            peaks = [s for s in trace if "kv_metrics_peak" in s]
            if peaks:
                ax.plot(
                    [s["elapsed_seconds"] for s in peaks],
                    [100 * s["kv_metrics_peak"]["utilization"] for s in peaks],
                    linestyle=":",
                    color=f"C{series}",
                    label=f"Concurrency {concurrency(row)}: within-step peak",
                )
        ax.set_ylim(0, 100)
        style(ax, "Elapsed time (s)", "Owned physical blocks (%)")
        save(fig, "kv-utilization.png")

    # Regeneration must not leave a stale chart that the new data cannot support.
    for filename in CHARTS.keys() - set(generated):
        (output_dir / filename).unlink(missing_ok=True)
    lines = ["# minfer benchmark report", "", f"Source: `{source.name}`", ""]
    for key, label in (
        ("model", "Model"),
        ("platform", "Hardware/platform"),
        ("device", "Backend"),
        ("dtype", "Dtype"),
        ("cache_mode", "Cache mode"),
        ("max_new_tokens", "Output budget"),
        ("prompt_token_lengths", "Prompt token lengths"),
        ("warmup", "Warmup runs"),
        ("seed", "Workload seed"),
        ("arrival_interval_seconds", "Arrival interval (s)"),
        ("max_batched_tokens", "Per-step token limit"),
        ("max_kv_tokens", "Logical KV limit"),
        ("kv_block_size", "KV block size"),
        ("num_kv_blocks", "KV block pool size"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
    ):
        lines.append(f"- **{label}:** {meta.get(key, 'unrecorded')}")
    lines += [
        f"- **Workload:** {rows[0].get('requests', 'unrecorded')} requests; "
        "same recorded prompts and greedy output budget across methods.",
        "",
        "HF sequential is the reference baseline; minfer runs manual prefill/decode. "
        "Model loading is excluded. Queueing and tokenization are included. "
        "Small-sample percentiles are descriptive; N/A means unavailable.",
        "",
        "| Method | Wall (s) | Tokens/s | Requests/s | TTFT p50 (ms) | TTFT p95 (ms) "
        "| Latency p50 (ms) | Latency p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    columns = [
        ("wall_seconds", 1),
        ("tokens_per_second", 1),
        ("requests_per_second", 1),
        ("ttft_p50_seconds", 1000),
        ("ttft_p95_seconds", 1000),
        ("latency_p50_seconds", 1000),
        ("latency_p95_seconds", 1000),
    ]
    for row in rows:
        values = [
            "N/A" if row.get(key) is None else f"{row[key] * scale:.2f}" for key, scale in columns
        ]
        lines.append(f"| {row['name']} | " + " | ".join(values) + " |")
    statements = observations(rows)
    if statements:
        lines += ["", "## Observations", "", *[f"- {s}" for s in statements]]
    lines += ["", "## Plots", ""]
    for filename in generated:
        lines += [f"![{CHARTS[filename]}]({filename})", ""]
    omitted = [title for name, title in CHARTS.items() if name not in generated]
    if omitted:
        lines += ["Unavailable data; plots omitted: " + "; ".join(omitted) + ".", ""]
    report = output_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or Path("benchmarks/reports") / args.source.stem
    try:
        report = generate_report(args.source, output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
