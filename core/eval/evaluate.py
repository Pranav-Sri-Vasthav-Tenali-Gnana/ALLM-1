"""Score a saved variant on the eval split through the inference predictor.

Usage::

    python -m core.eval.evaluate --config <yaml> --variant int8

Writes ``artifacts/<name>/eval_<variant>.json`` (metrics + a few examples).
"""
from __future__ import annotations

import argparse
import time
from typing import Any

from core.data.datasets import load_raw_splits
from core.eval.metrics import classification_metrics, rouge_metrics
from core.models.factory import VARIANTS
from core.models.predictor import load_predictor
from core.utils.common import get_logger, save_json
from core.utils.config import ExperimentConfig, add_config_args, config_from_args

log = get_logger(__name__)


def evaluate_variant(cfg: ExperimentConfig, variant: str, eval_ds=None) -> dict[str, Any]:
    eval_ds = eval_ds if eval_ds is not None else load_raw_splits(cfg).eval
    # Quality is device-independent, so FP32 eval may use the GPU for speed.
    # Latency is measured separately (core.bench), always on CPU.
    device = cfg.resolved_device() if variant == "fp32" else "cpu"
    predictor = load_predictor(cfg.variant_dir(variant), num_threads=cfg.bench.num_threads, device=device)
    texts, targets = list(eval_ds["text"]), list(eval_ds["target"])

    start = time.perf_counter()
    if cfg.task == "classification":
        preds = predictor.predict(texts, batch_size=cfg.train.eval_batch_size)
        metrics = classification_metrics([p["label_id"] for p in preds], targets)
        examples = [
            {"text": t[:200], "gold": predictor.meta["label_names"][g], "pred": p["label"]}
            for t, g, p in list(zip(texts, targets, preds))[:5]
        ]
    else:
        preds = predictor.predict(
            texts,
            batch_size=8,
            max_new_tokens=cfg.bench.max_new_tokens,
            num_beams=cfg.bench.num_beams,
        )
        metrics = rouge_metrics(preds, targets)
        examples = [
            {"text": t[:300], "reference": r, "summary": p}
            for t, r, p in list(zip(texts, targets, preds))[:3]
        ]

    result = {
        "variant": variant,
        "n": len(texts),
        "metrics": metrics,
        "device": device,
        "eval_wall_seconds": round(time.perf_counter() - start, 2),
        "examples": examples,
    }
    save_json(result, cfg.run_dir / f"eval_{variant}.json")
    log.info("[%s] %s", variant, metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--variant", choices=VARIANTS, default="fp32")
    args = parser.parse_args()
    evaluate_variant(config_from_args(args), args.variant)


if __name__ == "__main__":
    main()
