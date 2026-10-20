"""Dataset adapters.

Turns any HF hub dataset (or local JSONL) into a uniform shape:

* raw splits with two columns: ``text`` and ``target``
* tokenized splits ready for the HF Trainer

Keeping the raw eval split around matters: evaluation runs the *predictor*
on raw text, so FP32 / INT8 / ONNX are scored through the exact same path.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import ClassLabel, Dataset, load_dataset

from core.utils.config import ExperimentConfig


@dataclass
class RawSplits:
    train: Dataset
    eval: Dataset
    label_names: list[str] | None  # classification only


def _load_split(cfg: ExperimentConfig, which: str) -> Dataset:
    d = cfg.data
    if d.dataset == "local":
        path = d.train_file if which == "train" else d.eval_file
        if not path:
            raise ValueError(f"data.{which}_file is required when dataset == 'local'")
        return load_dataset("json", data_files=path, split="train")
    split = d.train_split if which == "train" else d.eval_split
    return load_dataset(d.dataset, d.subset, split=split)


def _subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_raw_splits(
    cfg: ExperimentConfig,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> RawSplits:
    d = cfg.data
    train = _subsample(_load_split(cfg, "train"), max_train or d.max_train_samples, cfg.seed)
    eval_ = _subsample(_load_split(cfg, "eval"), max_eval or d.max_eval_samples, cfg.seed)

    label_names = None
    if cfg.task == "classification":
        feature = train.features[d.target_field]
        if isinstance(feature, ClassLabel):
            label_names = d.label_names or list(feature.names)
        else:
            # String or raw-int labels: use configured names or derive a stable order.
            label_names = d.label_names or sorted({str(v) for v in train[d.target_field]})

    def to_uniform(ds: Dataset) -> Dataset:
        def fn(ex):
            target = ex[d.target_field]
            if label_names is not None and not isinstance(target, int):
                target = label_names.index(str(target))
            return {"text": ex[d.text_field], "target": target}

        return ds.map(fn, remove_columns=ds.column_names)

    return RawSplits(to_uniform(train), to_uniform(eval_), label_names)


def tokenize_classification(ds: Dataset, tokenizer, cfg: ExperimentConfig) -> Dataset:
    def fn(batch):
        enc = tokenizer(
            [cfg.model.prefix + t for t in batch["text"]],
            truncation=True,
            max_length=cfg.model.max_length,
        )
        enc["labels"] = batch["target"]
        return enc

    return ds.map(fn, batched=True, remove_columns=ds.column_names)


def tokenize_summarization(ds: Dataset, tokenizer, cfg: ExperimentConfig) -> Dataset:
    def fn(batch):
        enc = tokenizer(
            [cfg.model.prefix + t for t in batch["text"]],
            truncation=True,
            max_length=cfg.model.max_length,
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            max_length=cfg.model.max_target_length,
        )
        enc["labels"] = labels["input_ids"]
        return enc

    return ds.map(fn, batched=True, remove_columns=ds.column_names)


def tokenize(ds: Dataset, tokenizer, cfg: ExperimentConfig) -> Dataset:
    if cfg.task == "classification":
        return tokenize_classification(ds, tokenizer, cfg)
    return tokenize_summarization(ds, tokenizer, cfg)
