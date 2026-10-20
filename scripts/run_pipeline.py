"""End-to-end experiment: train -> eval FP32 -> quantize -> eval INT8 -> benchmark -> report.

    python scripts/run_pipeline.py --config configs/classification/distilbert_agnews.yaml
    python scripts/run_pipeline.py --config configs/classification/smoke.yaml --register smoke-cls

Writes ``artifacts/<name>/report.json`` + ``report.md`` with FP32 vs INT8
accuracy/ROUGE, latency and size side by side.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bench.benchmark import benchmark_variant  # noqa: E402
from core.data.datasets import load_raw_splits  # noqa: E402
from core.eval.evaluate import evaluate_variant  # noqa: E402
from core.eval.metrics import primary_metric  # noqa: E402
from core.quant.quantize import quantize_run  # noqa: E402
from core.train.train import train  # noqa: E402
from core.utils.common import get_logger, save_json, set_seed  # noqa: E402
from core.utils.config import ExperimentConfig, add_config_args, config_from_args  # noqa: E402

log = get_logger("pipeline")


def build_report(cfg: ExperimentConfig, evals: dict, benches: dict) -> dict:
    metric = primary_metric(cfg.task)
    base = evals["fp32"]["metrics"][metric]
    base_lat = benches["fp32"]["latency"][0]["p50_ms"]
    base_size = benches["fp32"]["size_mb"]
    rows = []
    for variant in evals:
        e, b = evals[variant]["metrics"], benches[variant]
        lat = b["latency"][0]["p50_ms"]
        rows.append({
            "variant": variant,
            **e,
            "delta_" + metric: round(e[metric] - base, 4),
            "size_mb": b["size_mb"],
            "size_reduction_x": round(base_size / b["size_mb"], 2) if b["size_mb"] else None,
            "p50_ms_bs1": lat,
            "speedup_bs1_x": round(base_lat / lat, 2) if lat else None,
            "latency_by_batch": b["latency"],
        })
    return {"name": cfg.name, "task": cfg.task, "base_model": cfg.model.name_or_path,
            "primary_metric": metric, "rows": rows, "env": benches["fp32"]["env"]}


def report_markdown(report: dict) -> str:
    m = report["primary_metric"]
    lines = [
        f"# {report['name']} ({report['task']}, {report['base_model']})", "",
        f"| Variant | {m} | Δ {m} | Size (MB) | Size ↓ | p50 bs=1 (ms) | Speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['variant']} | {r[m]:.4g} | {r['delta_' + m]:+.4g} | {r['size_mb']} | "
            f"{r['size_reduction_x']}x | {r['p50_ms_bs1']} | {r['speedup_bs1_x']}x |"
        )
    env = report["env"]
    lines += ["", f"_CPU: {env['processor']} · threads={env['torch_threads']} · torch {env['torch']}_"]
    return "\n".join(lines) + "\n"


def run(cfg: ExperimentConfig, skip_train: bool = False) -> dict:
    set_seed(cfg.seed)
    splits = load_raw_splits(cfg)
    if not skip_train:
        train(cfg, splits=splits)

    variants = ["fp32"]
    if cfg.quant.enabled:
        variants += [p.name for p in quantize_run(cfg)]

    bench_texts = list(splits.eval["text"][: max(cfg.bench.batch_sizes)])
    evals = {v: evaluate_variant(cfg, v, splits.eval) for v in variants}
    benches = {v: benchmark_variant(cfg, v, bench_texts) for v in variants}

    report = build_report(cfg, evals, benches)
    save_json(report, cfg.run_dir / "report.json")
    (cfg.run_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    print(report_markdown(report))
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing fp32/")
    parser.add_argument("--register", metavar="MODEL_NAME", help="Register a variant in the model registry")
    parser.add_argument("--register-variant", default="int8")
    parser.add_argument("--registry-dir", default="artifacts/registry")
    args = parser.parse_args()

    cfg = config_from_args(args)
    run(cfg, skip_train=args.skip_train)

    if args.register:
        from service.registry.registry import ModelRegistry, metrics_for_variant

        resolved = ModelRegistry(args.registry_dir).register(
            args.register,
            cfg.variant_dir(args.register_variant),
            metrics_for_variant(cfg.run_dir, args.register_variant),
            alias="production",
            notes=f"pipeline run {cfg.name}",
        )
        log.info("Registered %s (%s) as production", resolved.ref, resolved.variant)


if __name__ == "__main__":
    main()
