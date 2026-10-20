"""File-system model registry with immutable versions and movable aliases.

Layout::

    <root>/
      registry.json
      models/<name>/v1/   <- frozen copy of a variant dir (fp32 / int8 / onnx-uint8)
      models/<name>/v2/

A model reference is ``name``, ``name@3`` or ``name@production``.
A bare ``name`` resolves to the ``production`` alias if set, else the latest.

Versions are copies, never pointers into ``artifacts/<run>``, so retraining
can't silently change what the server is serving.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models.factory import read_meta
from core.utils.common import dir_size_mb, load_json

DEFAULT_ALIAS = "production"


class RegistryError(KeyError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    version: int
    path: Path
    task: str
    variant: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def parse_ref(ref: str) -> tuple[str, str | None]:
    name, _, tag = ref.partition("@")
    return name, (tag or None)


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "registry.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------- persistence
    def _read(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"models": {}}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        # Atomic replace: a crash mid-write never leaves a corrupt index.
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------ writes
    def register(
        self,
        name: str,
        variant_dir: str | Path,
        metrics: dict[str, Any] | None = None,
        alias: str | None = None,
        notes: str = "",
    ) -> ResolvedModel:
        variant_dir = Path(variant_dir)
        meta = read_meta(variant_dir)  # validates it's a real variant dir
        with self._lock:
            data = self._read()
            entry = data["models"].setdefault(
                name, {"task": meta["task"], "versions": {}, "aliases": {}}
            )
            if entry["task"] != meta["task"]:
                raise RegistryError(f"{name} is a {entry['task']} model; got {meta['task']}")

            version = max(map(int, entry["versions"]), default=0) + 1
            dest = self.root / "models" / name / f"v{version}"
            shutil.copytree(variant_dir, dest)
            entry["versions"][str(version)] = {
                # POSIX separators so a registry built on Windows works in Linux containers.
                "path": dest.relative_to(self.root).as_posix(),
                "variant": meta["variant"],
                "base_model": meta.get("base_model"),
                "source": variant_dir.as_posix(),
                "size_mb": dir_size_mb(dest),
                "metrics": metrics or {},
                "notes": notes,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if alias:
                entry["aliases"][alias] = version
            self._write(data)
        return self.resolve(f"{name}@{version}")

    def set_alias(self, name: str, alias: str, version: int) -> None:
        with self._lock:
            data = self._read()
            entry = self._entry(data, name)
            if str(version) not in entry["versions"]:
                raise RegistryError(f"{name} has no version {version}")
            entry["aliases"][alias] = int(version)
            self._write(data)

    # ------------------------------------------------------------------- reads
    def _entry(self, data: dict[str, Any], name: str) -> dict[str, Any]:
        try:
            return data["models"][name]
        except KeyError:
            raise RegistryError(f"Unknown model {name!r}") from None

    def resolve(self, ref: str) -> ResolvedModel:
        name, tag = parse_ref(ref)
        entry = self._entry(self._read(), name)
        if tag is None:
            version = entry["aliases"].get(DEFAULT_ALIAS) or max(map(int, entry["versions"]))
        elif tag.isdigit():
            version = int(tag)
        elif tag in entry["aliases"]:
            version = entry["aliases"][tag]
        else:
            raise RegistryError(f"{name} has no alias {tag!r}")
        info = entry["versions"].get(str(version))
        if info is None:
            raise RegistryError(f"{name} has no version {version}")
        return ResolvedModel(name, int(version), self.root / info["path"], entry["task"], info["variant"])

    def list_models(self) -> dict[str, Any]:
        return self._read()["models"]


def metrics_for_variant(run_dir: str | Path, variant: str) -> dict[str, Any]:
    """Pull eval + bench results for a variant from its run dir (if present)."""
    run_dir = Path(run_dir)
    out: dict[str, Any] = {}
    if (run_dir / f"eval_{variant}.json").exists():
        out.update(load_json(run_dir / f"eval_{variant}.json")["metrics"])
    if (run_dir / f"bench_{variant}.json").exists():
        bench = load_json(run_dir / f"bench_{variant}.json")
        out["p50_ms_bs1"] = bench["latency"][0]["p50_ms"]
    return out
