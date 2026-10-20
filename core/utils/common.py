"""Small shared helpers: logging, seeding, JSON I/O, sizes, environment info."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("ALLM_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so runs are reproducible."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dir_size_mb(path: str | Path, suffixes: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".onnx")) -> float:
    """Size of the weight files in a directory (tokenizer/config files excluded)."""
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.suffix in suffixes]
    return round(sum(p.stat().st_size for p in files) / 1024**2, 2)


def env_info() -> dict[str, Any]:
    """Hardware/software fingerprint saved alongside every benchmark."""
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "quant_engine": torch.backends.quantized.engine,
    }
