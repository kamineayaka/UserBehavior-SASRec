"""Download SASRec training data and optional baseline weights from GitHub Release."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "kamineayaka/UserBehavior-SASRec"
RELEASE_TAG = "latest"  # or e.g. "v1.0-data"

PARQUET_FILES = (
    "train.parquet",
    "valid.parquet",
    "test.parquet",
    "item2idx_mapping.parquet",
)

RELEASE_URL = f"https://github.com/{REPO}/releases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Release assets into SASRec/data/ (and data/baseline/ for .pt)."
    )
    parser.add_argument(
        "--tag",
        default=RELEASE_TAG,
        help=f"Release tag (default: {RELEASE_TAG})",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Where gh saves assets (default: <SASRec>/.release_download)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data files")
    return parser.parse_args()


def gh_available() -> bool:
    try:
        subprocess.run(
            ["gh", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_with_gh(repo: str, tag: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "release", "download", "--repo", repo, "--dir", str(dest)]
    if tag == "latest":
        cmd.append("--latest")
    else:
        cmd.extend([tag])
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def move_parquet(src: Path, data_dir: Path, force: bool) -> None:
    for name in PARQUET_FILES:
        found = list(src.rglob(name))
        if not found:
            continue
        src_file = found[0]
        dst = data_dir / name
        if dst.exists() and not force:
            print(f"skip (exists): {dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src_file.resolve() != dst.resolve():
            shutil.copy2(src_file, dst)
        print(f"placed: {dst.name} ({dst.stat().st_size / (1024**2):.1f} MB)")


def move_pt_files(src: Path, baseline_dir: Path, force: bool) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for pt in src.rglob("*.pt"):
        dst = baseline_dir / pt.name
        if dst.exists() and not force:
            print(f"skip (exists): {dst.name}")
            continue
        if pt.resolve() != dst.resolve():
            shutil.copy2(pt, dst)
        print(f"placed: {dst} ({dst.stat().st_size / (1024**2):.1f} MB)")


def extract_zips(directory: Path) -> None:
    for zpath in list(directory.glob("*.zip")):
        extract_to = directory / zpath.stem
        extract_to.mkdir(parents=True, exist_ok=True)
        print(f"extracting: {zpath.name} -> {extract_to.name}/")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_to)


def verify(data_dir: Path) -> list[str]:
    missing = [name for name in PARQUET_FILES if not (data_dir / name).exists()]
    return missing


def print_manual_help() -> None:
    print(f"\nManual download: {RELEASE_URL}")
    print("See 数据与模型下载.md for file list and paths.")


def main() -> int:
    args = parse_args()
    sasrec_dir = Path(__file__).resolve().parents[1]
    data_dir = sasrec_dir / "data"
    baseline_dir = data_dir / "baseline"
    download_dir = args.download_dir or (sasrec_dir / ".release_download")

    if not gh_available():
        print("error: GitHub CLI (gh) not found or not working.", file=sys.stderr)
        print_manual_help()
        return 1

    try:
        if download_dir.exists() and args.force:
            shutil.rmtree(download_dir)
        download_with_gh(REPO, args.tag, download_dir)
    except subprocess.CalledProcessError as exc:
        print(f"error: gh release download failed ({exc})", file=sys.stderr)
        print_manual_help()
        return 1

    extract_zips(download_dir)
    move_parquet(download_dir, data_dir, args.force)
    move_pt_files(download_dir, baseline_dir, args.force)

    missing = verify(data_dir)
    if missing:
        print("\nwarning: still missing parquet files:", ", ".join(missing))
        print_manual_help()
        return 1

    print("done. data/ is ready for notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
