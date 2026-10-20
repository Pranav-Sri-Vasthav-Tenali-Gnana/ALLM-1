"""Registry CLI.

    python -m service.registry register --name agnews-cls --variant-dir artifacts/cls-distilbert-agnews/int8 --alias production
    python -m service.registry promote --name agnews-cls --version 2 --alias production
    python -m service.registry list
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from service.registry.registry import ModelRegistry, metrics_for_variant

DEFAULT_ROOT = os.environ.get("ALLM_REGISTRY_DIR", "artifacts/registry")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m service.registry")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Freeze a variant dir as a new version")
    reg.add_argument("--name", required=True)
    reg.add_argument("--variant-dir", required=True)
    reg.add_argument("--alias", default=None)
    reg.add_argument("--notes", default="")

    pro = sub.add_parser("promote", help="Point an alias at a version")
    pro.add_argument("--name", required=True)
    pro.add_argument("--version", type=int, required=True)
    pro.add_argument("--alias", default="production")

    sub.add_parser("list", help="Show all models")

    args = parser.parse_args()
    registry = ModelRegistry(args.root)

    if args.cmd == "register":
        vdir = Path(args.variant_dir)
        metrics = metrics_for_variant(vdir.parent, vdir.name)
        resolved = registry.register(args.name, vdir, metrics, args.alias, args.notes)
        print(f"Registered {resolved.ref} ({resolved.variant}) -> {resolved.path}")
    elif args.cmd == "promote":
        registry.set_alias(args.name, args.alias, args.version)
        print(f"{args.name}@{args.alias} -> v{args.version}")
    else:
        print(json.dumps(registry.list_models(), indent=2))


if __name__ == "__main__":
    main()
