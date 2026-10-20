"""Hyperparameter search with Optuna (TPE sampler).

The search space lives in the YAML under ``tune.search_space`` using dotted
config keys, so *any* config field is tunable without code changes::

    tune:
      n_trials: 12
      search_space:
        train.learning_rate: {type: float, low: 1.0e-5, high: 1.0e-4, log: true}
        train.batch_size:    {type: categorical, choices: [16, 32]}

Objective: eval accuracy (classification, maximize) or eval loss
(summarization, minimize -- cheap proxy for ROUGE that avoids generation).

Usage::

    python -m core.tune.search --config <yaml>
    python -m core.train.train --config <yaml> --params-file artifacts/<name>/tune/best_params.yaml
"""
from __future__ import annotations

import argparse
from typing import Any

import optuna
import yaml

from core.data.datasets import load_raw_splits
from core.train.train import train
from core.utils.common import get_logger, save_json
from core.utils.config import ExperimentConfig, add_config_args, apply_overrides, config_from_args

log = get_logger(__name__)


def suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search-space type {kind!r} for {name}")


def objective_spec(task: str) -> tuple[str, str]:
    return ("eval_accuracy", "maximize") if task == "classification" else ("eval_loss", "minimize")


def run_search(cfg: ExperimentConfig) -> dict[str, Any]:
    if not cfg.tune.search_space:
        raise ValueError("tune.search_space is empty -- nothing to search")
    metric, direction = objective_spec(cfg.task)

    # Load (and subsample) data once; every trial reuses the same split.
    splits = load_raw_splits(cfg, cfg.tune.max_train_samples, cfg.tune.max_eval_samples)

    def objective(trial: optuna.Trial) -> float:
        params = {k: suggest(trial, k, spec) for k, spec in cfg.tune.search_space.items()}
        trial_cfg = apply_overrides(cfg, params)
        metrics = train(trial_cfg, splits=splits, save=False)
        log.info("Trial %d %s -> %s=%.4f", trial.number, params, metric, metrics[metric])
        return metrics[metric]

    study = optuna.create_study(
        study_name=cfg.name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
    )
    study.optimize(objective, n_trials=cfg.tune.n_trials)

    out_dir = cfg.run_dir / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    (out_dir / "best_params.yaml").write_text(yaml.safe_dump(study.best_params), encoding="utf-8")
    summary = {
        "metric": metric,
        "direction": direction,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    save_json(summary, out_dir / "summary.json")
    log.info("Best %s=%.4f with %s", metric, study.best_value, study.best_params)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    run_search(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
