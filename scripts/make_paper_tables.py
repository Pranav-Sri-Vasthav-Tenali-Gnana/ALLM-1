"""Collect every ``artifacts/*/report.json`` into paper-ready tables + a figure.

    python scripts/make_paper_tables.py

Outputs:
    paper/tables/results.md         one combined FP32 vs INT8 table
    paper/tables/results.csv        same data, for plotting elsewhere
    paper/figures/latency_vs_size.png  (if matplotlib is installed)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS, TABLES, FIGURES = ROOT / "artifacts", ROOT / "paper" / "tables", ROOT / "paper" / "figures"


def collect() -> list[dict]:
    rows = []
    for path in sorted(ARTIFACTS.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["name"].startswith("smoke"):
            continue  # pipeline smoke tests, not experiments
        m = report["primary_metric"]
        for r in report["rows"]:
            rows.append({
                "experiment": report["name"],
                "task": report["task"],
                "model": report["base_model"],
                "variant": r["variant"],
                "metric": m,
                "score": r[m],
                "delta": r["delta_" + m],
                "size_mb": r["size_mb"],
                "size_reduction_x": r["size_reduction_x"],
                "p50_ms_bs1": r["p50_ms_bs1"],
                "speedup_bs1_x": r["speedup_bs1_x"],
            })
    return rows


def write_tables(rows: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Experiment | Model | Variant | Metric | Score | Δ | Size (MB) | Size ↓ | p50 bs=1 (ms) | Speedup |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['model']} | {r['variant']} | {r['metric']} | {r['score']:.4g} | "
            f"{r['delta']:+.4g} | {r['size_mb']} | {r['size_reduction_x']}x | {r['p50_ms_bs1']} | {r['speedup_bs1_x']}x |"
        )
    (TABLES / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping figure")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for exp in sorted({r["experiment"] for r in rows}):
        pts = [r for r in rows if r["experiment"] == exp]
        ax.plot([p["size_mb"] for p in pts], [p["p50_ms_bs1"] for p in pts], marker="o", label=exp)
        for p in pts:
            ax.annotate(p["variant"], (p["size_mb"], p["p50_ms_bs1"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Weight size (MB)")
    ax.set_ylabel("p50 latency, batch=1 (ms, log scale)")
    ax.set_yscale("log")  # classification (~30 ms) and summarization (~2 s) differ by 50x
    ax.set_title("ALLM-1: size vs CPU latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "latency_vs_size.png", dpi=150)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    rows = collect()
    if not rows:
        raise SystemExit("No artifacts/*/report.json found -- run scripts/run_pipeline.py first")
    write_tables(rows)
    write_figure(rows)
    print((TABLES / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
