"""Sync sasrec_core from repo root into SASRec/sasrec_core (for standalone package)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILES = (
    "__init__.py",
    "config.py",
    "data.py",
    "model.py",
    "trainer.py",
    "estimator.py",
    "README.md",
    "TEAM_GUIDE.md",
    "SASREC_原理与实现.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy repo-root sasrec_core into SASRec/sasrec_core.")
    parser.add_argument("--source", type=Path, default=None, help="Source dir (default: <repo>/sasrec_core)")
    parser.add_argument("--target", type=Path, default=None, help="Target dir (default: SASRec/sasrec_core)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sasrec_dir = Path(__file__).resolve().parents[1]
    repo_root = sasrec_dir.parent

    source = args.source or (repo_root / "sasrec_core")
    target = args.target or (sasrec_dir / "sasrec_core")

    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    target.mkdir(parents=True, exist_ok=True)
    print(f"source: {source}")
    print(f"target: {target}")

    for name in FILES:
        src_file = source / name
        dst_file = target / name
        if not src_file.exists():
            raise FileNotFoundError(f"Missing source file: {src_file}")
        if dst_file.exists() and not args.force:
            print(f"skip (exists): {name}")
            continue
        shutil.copy2(src_file, dst_file)
        print(f"copied: {name}")

    print("done.")


if __name__ == "__main__":
    main()
