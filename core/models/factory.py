"""Model construction + the self-describing "variant directory" format.

A *variant* is one deployable form of a trained model: ``fp32``, ``int8``
(PyTorch dynamic quantization) or ``onnx-uint8`` (ONNX Runtime). Each variant
directory contains an ``allm_meta.json`` so anything downstream (eval, bench,
the API server) can load it without needing the original experiment config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from core.utils.common import load_json, save_json
from core.utils.config import ExperimentConfig

META_FILE = "allm_meta.json"
VARIANTS = ("fp32", "int8", "onnx-uint8")


def auto_model_cls(task: str):
    return AutoModelForSequenceClassification if task == "classification" else AutoModelForSeq2SeqLM


def load_tokenizer(name_or_path: str | Path):
    return AutoTokenizer.from_pretrained(str(name_or_path))


def build_model_for_training(cfg: ExperimentConfig, label_names: list[str] | None):
    if cfg.task == "classification":
        id2label = dict(enumerate(label_names))
        return AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name_or_path,
            num_labels=len(label_names),
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
    return AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)


def build_empty_model(task: str, config_dir: str | Path):
    """Architecture only (random weights) -- used to rebuild a quantized model."""
    config = AutoConfig.from_pretrained(str(config_dir))
    return auto_model_cls(task).from_config(config)


def make_meta(cfg: ExperimentConfig, variant: str, label_names: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "task": cfg.task,
        "variant": variant,
        "base_model": cfg.model.name_or_path,
        "max_length": cfg.model.max_length,
        "max_target_length": cfg.model.max_target_length,
        "prefix": cfg.model.prefix,
        "label_names": label_names,
    }


def write_meta(variant_dir: str | Path, meta: dict[str, Any]) -> None:
    save_json(meta, Path(variant_dir) / META_FILE)


def read_meta(variant_dir: str | Path) -> dict[str, Any]:
    path = Path(variant_dir) / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- is this a variant directory?")
    return load_json(path)
