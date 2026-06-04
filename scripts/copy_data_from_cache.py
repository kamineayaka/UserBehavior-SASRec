"""Copy SASRec training parquet files from repo SASRec_cache to SASRec/data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PARQUET_FILES = (
    "train.parquet",
    "valid.parquet",
    "test.parquet",
    "item2idx_mapping.parquet",
)

BASELINE_JSON = (
    "baseline_sasrec_20260425_130830",
    "baseline_sasrec_20260425_130830.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync SASRec/data from SASRec_cache.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source cache dir (default: <repo>/SASRec_cache)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target data dir (default: <SASRec>/data)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def copy_file(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"skip (exists): {dst.name}")
        return
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"copied: {dst.name} ({size_mb:.1f} MB)")


def main() -> None:
    args = parse_args()
    sasrec_dir = Path(__file__).resolve().parents[1]
    repo_root = sasrec_dir.parent

    source = args.source or (repo_root / "SASRec_cache")
    target = args.target or (sasrec_dir / "data")

    print(f"source: {source}")
    print(f"target: {target}")

    for name in PARQUET_FILES:
        copy_file(source / name, target / name, args.force)

    baseline_src = source / BASELINE_JSON[0] / BASELINE_JSON[1]
    baseline_dst = target / "baseline" / BASELINE_JSON[1]
    copy_file(baseline_src, baseline_dst, args.force)

    print("done.")


if __name__ == "__main__":
    main()
