"""Assemble a self-contained deploy bundle (code + selected models) for Docker / HF Spaces.

    python scripts/build_space.py --models agnews-cls
    hf upload <your-username>/allm-1 dist/space . --repo-type space

The bundle contains only what the server needs: ``core/``, ``service/``, the
Dockerfile, serving requirements, a trimmed registry holding just the
production version of each requested model, and those experiments' benchmark reports.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def build(models: list[str], registry_dir: Path, out: Path) -> Path:
    index = json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for pkg in ("core", "service"):
        shutil.copytree(ROOT / pkg, out / pkg, ignore=IGNORE)
    shutil.copy(ROOT / "deploy" / "Dockerfile", out / "Dockerfile")
    shutil.copy(ROOT / "deploy" / "requirements-serve.txt", out / "requirements-serve.txt")
    shutil.copy(ROOT / "deploy" / "SPACE_README.md", out / "README.md")

    # Trimmed registry: only the version each model's "production" alias points to.
    trimmed = {"models": {}}
    for name in models:
        entry = index["models"].get(name)
        if entry is None:
            raise SystemExit(f"{name!r} is not in {registry_dir / 'registry.json'}")
        version = str(entry["aliases"].get("production") or max(map(int, entry["versions"])))
        info = dict(entry["versions"][version])
        info["path"] = info["path"].replace("\\", "/")  # registries made before the POSIX-path fix
        info["source"] = info.get("source", "").replace("\\", "/")
        shutil.copytree(registry_dir / info["path"], out / "registry" / info["path"], ignore=IGNORE)
        # Ship the experiment's benchmark report so the demo page can show FP32 vs INT8.
        report = ROOT / Path(info["source"]).parent / "report.json"
        if report.exists():
            (out / "reports").mkdir(exist_ok=True)
            shutil.copy(report, out / "reports" / f"{report.parent.name}.json")
        trimmed["models"][name] = {
            "task": entry["task"],
            "versions": {version: info},
            "aliases": {"production": int(version)},
        }
    (out / "registry" / "registry.json").write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    print(f"Bundle ready: {out} ({size:.0f} MB) with models {models}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["agnews-cls"])
    parser.add_argument("--registry-dir", default=str(ROOT / "artifacts" / "registry"))
    parser.add_argument("--out", default=str(ROOT / "dist" / "space"))
    args = parser.parse_args()
    build(args.models, Path(args.registry_dir), Path(args.out))


if __name__ == "__main__":
    main()
