import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.models.factory import write_meta  # noqa: E402


def make_variant_dir(path: Path, task: str = "classification", variant: str = "int8") -> Path:
    """A minimal variant directory (meta only) for registry/API tests."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "model_int8.pt").write_bytes(b"\0" * 1024)
    write_meta(path, {
        "name": "test", "task": task, "variant": variant, "base_model": "tiny",
        "max_length": 32, "max_target_length": 16, "prefix": "",
        "label_names": ["neg", "pos"] if task == "classification" else None,
    })
    return path


@pytest.fixture
def variant_factory(tmp_path):
    counter = {"n": 0}

    def _make(task="classification", variant="int8"):
        counter["n"] += 1
        return make_variant_dir(tmp_path / f"src{counter['n']}" / variant, task, variant)

    return _make
