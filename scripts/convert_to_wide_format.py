from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_MISSING_NORMALIZED = {
    "na",
    "n a",
    "not reported",
    "not available",
    "not found",
    "missing",
    "no data",
    "none",
    "not applicable",
    "not specified",
    "unknown",
}


def _is_missing_value(text: str) -> bool:
    if not text:
        return True
    norm = text.strip().lower()
    norm_clean = "".join(c for c in norm if c.isalnum() or c.isspace())
    norm_clean = " ".join(norm_clean.split())
    return norm_clean in _MISSING_NORMALIZED


def convert_json_to_wide_csv(json_path: str, output_path: str | None = None) -> pd.DataFrame:
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    paper_name = json_file.stem.replace("_output", "")

    if output_path is None:
        output_dir = Path("output/conversions")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{paper_name}_wide.csv"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    row_data: dict[str, str] = {}

    for question, qa_data in data.items():
        answer = (qa_data or {}).get("answer", "") or ""
        status = (qa_data or {}).get("status", "") or ""
        if status == "MISSING_IN_PAPER" or _is_missing_value(answer):
            answer = ""
        row_data[question] = answer

    df = pd.DataFrame([row_data])
    df.insert(0, "paper_name", paper_name)
    df.to_csv(out_file, index=False)

    print(f"Converted: {json_file.name} -> {out_file}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str)
    parser.add_argument("output_csv", nargs="?", default=None)
    args = parser.parse_args()

    try:
        convert_json_to_wide_csv(args.json_file, args.output_csv)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
