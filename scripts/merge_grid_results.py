"""Merge all grid_search CSV files under results/grid_search and rank by metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and rank grid search CSV results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="SASRec directory (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output merged csv path (default: results/grid_search/_merged_all.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sasrec_dir = args.root or Path(__file__).resolve().parents[1]
    grid_dir = sasrec_dir / "results" / "grid_search"
    out_path = args.output or (grid_dir / "_merged_all.csv")

    csv_files = sorted(grid_dir.rglob("*.csv"))
    csv_files = [p for p in csv_files if p.name != out_path.name and not p.name.startswith("_merged")]

    if not csv_files:
        print(f"No csv files under {grid_dir}")
        return

    frames = []
    for p in csv_files:
        df = pd.read_csv(p)
        df["source_file"] = str(p.relative_to(sasrec_dir))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("valid_ndcg10", "valid_hr10", "test_ndcg10", "test_hr10") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"merged {len(csv_files)} files, {len(merged)} rows -> {out_path}")
    if sort_cols:
        print("\nTop 10 by valid_ndcg10:")
        print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
