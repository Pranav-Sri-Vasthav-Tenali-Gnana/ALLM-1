"""Typed experiment config.

Every entrypoint (train / tune / quantize / eval / bench) takes the same YAML
file, so one config fully describes one experiment. Values can be overridden
from the CLI with dotted keys, e.g. ``--set train.learning_rate=3e-5``.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS = ("classification", "summarization")


@dataclass
class ModelConfig:
    name_or_path: str
    max_length: int = 256          # max input tokens
    max_target_length: int = 64    # summarization only: max summary tokens
    prefix: str = ""               # task prefix, e.g. "summarize: " for T5


@dataclass
class DataConfig:
    dataset: str                   # HF hub id, or "local" for JSONL files
    subset: str | None = None      # HF config name, e.g. "3.0.0" for cnn_dailymail
    train_file: str | None = None  # used when dataset == "local"
    eval_file: str | None = None
    text_field: str = "text"
    target_field: str = "label"    # class label (cls) or reference summary (sum)
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    label_names: list[str] | None = None


@dataclass
class TrainConfig:
    epochs: float = 1.0
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    grad_accum: int = 1
    logging_steps: int = 50
    max_steps: int = -1            # >0 overrides epochs (handy for smoke runs)
    fp16: bool = False             # mixed precision: only worth it on GPUs with fast fp16 (Volta+)
    gradient_checkpointing: bool = False  # trade compute for memory on small GPUs


@dataclass
class TuneConfig:
    n_trials: int = 10
    max_train_samples: int | None = None   # tune on a subset to keep it cheap
    max_eval_samples: int | None = None
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class QuantConfig:
    enabled: bool = True
    onnx: bool = False             # also export ONNX + uint8 (classification only)


@dataclass
class BenchConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8])
    warmup: int = 3
    iters: int = 20
    num_threads: int = 4
    max_new_tokens: int = 48       # summarization only
    num_beams: int = 1


@dataclass
class ExperimentConfig:
    name: str
    task: str
    model: ModelConfig
    data: DataConfig
    seed: int = 42
    device: str = "auto"           # training + FP32 eval: "auto" (GPU if available), "cuda" or "cpu"
    output_dir: str = "artifacts"
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def variant_dir(self, variant: str) -> Path:
        return self.run_dir / variant

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass, rejecting unknown keys (catches typos)."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        ftype = fields[key].type
        nested = _NESTED.get(ftype) if isinstance(ftype, str) else None
        kwargs[key] = _build(nested, value or {}) if nested else value
    return cls(**kwargs)


_NESTED = {
    "ModelConfig": ModelConfig,
    "DataConfig": DataConfig,
    "TrainConfig": TrainConfig,
    "TuneConfig": TuneConfig,
    "QuantConfig": QuantConfig,
    "BenchConfig": BenchConfig,
}

# YAML parses "1e-5" (no dot) as a string; coerce known float fields.
_FLOAT_FIELDS = {"learning_rate", "weight_decay", "warmup_ratio", "epochs"}


def _coerce(d: dict[str, Any]) -> dict[str, Any]:
    for key, value in d.items():
        if isinstance(value, dict):
            _coerce(value)
        elif key in _FLOAT_FIELDS and isinstance(value, str):
            d[key] = float(value)
    return d


def set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Return a new config with dotted-key overrides applied."""
    raw = copy.deepcopy(cfg.to_dict())
    for key, value in overrides.items():
        set_dotted(raw, key, value)
    return _build(ExperimentConfig, _coerce(raw))


def parse_set_args(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``["train.learning_rate=3e-5", ...]`` into a dict (values via YAML)."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    params_file: str | Path | None = None,
) -> ExperimentConfig:
    """Load YAML config, then apply a tuned-params file, then CLI overrides."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    merged: dict[str, Any] = {}
    if params_file:
        with open(params_file, encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(overrides or {})
    for key, value in merged.items():
        set_dotted(raw, key, value)
    return _build(ExperimentConfig, _coerce(raw))


def add_config_args(parser) -> None:
    """Shared CLI flags for every entrypoint."""
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dotted overrides, e.g. train.epochs=1")
    parser.add_argument("--params-file", default=None,
                        help="YAML of tuned params (output of core.tune)")


def config_from_args(args) -> ExperimentConfig:
    return load_config(args.config, parse_set_args(args.set), args.params_file)
