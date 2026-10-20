"""CPU latency / throughput / size benchmark for a saved variant.

Methodology (kept fixed so FP32 vs INT8 numbers are comparable):

* fixed thread count (``bench.num_threads``) for torch and ORT
* real eval-set texts, the same ones for every variant
* ``warmup`` untimed runs, then ``iters`` timed runs per batch size
* report p50 / p95 / mean latency, throughput (items/s), weight size on disk

Usage::

    python -m core.bench.benchmark --config <yaml> --variant int8
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import numpy as np

from core.data.datasets import load_raw_splits
from core.models.factory import VARIANTS
from core.models.predictor import load_predictor
from core.utils.common import dir_size_mb, env_info, get_logger, save_json
from core.utils.config import ExperimentConfig, add_config_args, config_from_args

log = get_logger(__name__)


def _time_call(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_variant(cfg: ExperimentConfig, variant: str, texts: list[str] | None = None) -> dict[str, Any]:
    b = cfg.bench
    variant_dir = cfg.variant_dir(variant)
    predictor = load_predictor(variant_dir, num_threads=b.num_threads)
    texts = texts or list(load_raw_splits(cfg, max_eval=max(b.batch_sizes)).eval["text"])

    results = []
    for bs in b.batch_sizes:
        batch = (texts * (bs // len(texts) + 1))[:bs]
        if cfg.task == "classification":
            fn = lambda: predictor.predict(batch, batch_size=bs)  # noqa: E731
        else:
            fn = lambda: predictor.predict(  # noqa: E731
                batch, batch_size=bs, max_new_tokens=b.max_new_tokens, num_beams=b.num_beams
            )
        times = _time_call(fn, b.warmup, b.iters)
        results.append({
            "batch_size": bs,
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "mean_ms": round(statistics.fmean(times), 2),
            "throughput_items_s": round(bs / (statistics.fmean(times) / 1000), 2),
        })
        log.info("[%s] bs=%d p50=%.1fms p95=%.1fms", variant, bs, results[-1]["p50_ms"], results[-1]["p95_ms"])

    report = {
        "variant": variant,
        "size_mb": dir_size_mb(variant_dir),
        "latency": results,
        "settings": {"warmup": b.warmup, "iters": b.iters, "num_threads": b.num_threads,
                     "max_new_tokens": b.max_new_tokens, "num_beams": b.num_beams},
        "env": env_info(),
    }
    save_json(report, cfg.run_dir / f"bench_{variant}.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--variant", choices=VARIANTS, default="fp32")
    args = parser.parse_args()
    benchmark_variant(config_from_args(args), args.variant)


if __name__ == "__main__":
    main()
