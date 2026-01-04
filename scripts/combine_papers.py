from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine_papers(csv_paths: list[str], output_path: str) -> None:
    dfs: list[pd.DataFrame] = []

    for csv_path in csv_paths:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"Skip (not found): {csv_path}")
            continue

        df = pd.read_csv(csv_file, keep_default_na=False, na_filter=False)
        print(f"Loaded {csv_file.name}: {len(df)} rows × {len(df.columns)} columns")
        dfs.append(df)

    if not dfs:
        print("No valid CSV files found")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--output", "-o", type=str, default="output/conversions/combined_dataset.csv")
    args = parser.parse_args()
    combine_papers(args.csv_files, args.output)


if __name__ == "__main__":
    main()
