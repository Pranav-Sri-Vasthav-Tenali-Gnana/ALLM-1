"""Fine-tune a small transformer for classification or summarization.

Usage::

    python -m core.train.train --config configs/classification/distilbert_agnews.yaml
    python -m core.train.train --config configs/summarization/t5small_cnndm.yaml --set train.epochs=1

Output: ``artifacts/<name>/fp32/`` (model + tokenizer + allm_meta.json)
and ``artifacts/<name>/train_metrics.json``.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from core.data.datasets import RawSplits, load_raw_splits, tokenize
from core.eval.metrics import classification_metrics
from core.models.factory import build_model_for_training, load_tokenizer, make_meta, write_meta
from core.utils.common import get_logger, save_json, set_seed
from core.utils.config import ExperimentConfig, add_config_args, config_from_args

log = get_logger(__name__)


def _training_args(cfg: ExperimentConfig, output_dir: str):
    t = cfg.train
    common = dict(
        output_dir=output_dir,
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=float(t.warmup_ratio),  # transformers>=5: float in [0, 1) = ratio
        logging_steps=t.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",          # we save the final model ourselves
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        disable_tqdm=False,
        use_cpu=cfg.resolved_device() == "cpu",
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
    )
    if cfg.task == "summarization":
        return Seq2SeqTrainingArguments(**common, predict_with_generate=False)
    return TrainingArguments(**common)


def build_trainer(cfg: ExperimentConfig, splits: RawSplits, output_dir: str):
    tokenizer = load_tokenizer(cfg.model.name_or_path)
    model = build_model_for_training(cfg, splits.label_names)
    train_ds = tokenize(splits.train, tokenizer, cfg)
    eval_ds = tokenize(splits.eval, tokenizer, cfg)
    args = _training_args(cfg, output_dir)

    if cfg.task == "classification":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            return classification_metrics(np.argmax(logits, axis=-1), labels)

        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

    # Summarization: eval loss only during training (generation is slow);
    # ROUGE is computed afterwards by core.eval on raw text.
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )


def train(cfg: ExperimentConfig, splits: RawSplits | None = None, save: bool = True) -> dict[str, Any]:
    """Train and (optionally) save the fp32 variant. Returns eval metrics."""
    set_seed(cfg.seed)
    splits = splits or load_raw_splits(cfg)
    log.info("Train=%d  Eval=%d  Labels=%s  Device=%s",
             len(splits.train), len(splits.eval), splits.label_names, cfg.resolved_device())

    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(cfg, splits, tmp)
        start = time.perf_counter()
        train_out = trainer.train()
        metrics = {
            **{k: v for k, v in train_out.metrics.items()},
            **trainer.evaluate(),
            "train_wall_seconds": round(time.perf_counter() - start, 1),
            "n_train": len(splits.train),
            "n_eval": len(splits.eval),
        }

        if save:
            out = cfg.variant_dir("fp32")
            trainer.save_model(str(out))
            trainer.processing_class.save_pretrained(str(out))
            write_meta(out, make_meta(cfg, "fp32", splits.label_names))
            save_json(metrics, cfg.run_dir / "train_metrics.json")
            save_json(cfg.to_dict(), cfg.run_dir / "run_config.json")
            log.info("Saved FP32 model -> %s", out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    metrics = train(config_from_args(parser.parse_args()))
    log.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
