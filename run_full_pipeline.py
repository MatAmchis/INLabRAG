from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config.preprocessing import PreprocessingConfig, set_seed as set_preprocess_seed
from config.postprocessing import PostprocessingConfig, set_seed as set_postprocess_seed

set_preprocess_seed(PreprocessingConfig.SEED)
set_postprocess_seed(PostprocessingConfig.SEED)


def run_full_pipeline_single(pdf_path: Path, output_base: Path) -> None:
    pdf_stem = pdf_path.stem

    print("\n" + "=" * 70)
    print(f"STEP 1/3: PRE-PROCESSING - Extracting Q&A from {pdf_stem}")
    print("=" * 70 + "\n")

    from src.pre_processing.pipeline_preprocess import process_pdf_full

    preprocess_dir = output_base / "preprocessing" / pdf_stem
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_pdf_full(pdf_path, preprocess_dir)
        print(f"Saved to: {preprocess_dir}")
    except Exception as e:
        print(f"Pre-processing failed: {type(e).__name__}: {str(e)[:100]}")
        return

    print("\n" + "=" * 70)
    print("STEP 2/3: CONVERTING TO WIDE FORMAT")
    print("=" * 70 + "\n")

    from scripts.convert_to_wide_format import convert_json_to_wide_csv

    long_json = preprocess_dir / f"{pdf_stem}_output.json"
    conversions_dir = output_base / "conversions"
    conversions_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = conversions_dir / f"{pdf_stem}_wide.csv"

    try:
        convert_json_to_wide_csv(str(long_json), str(wide_csv))
        print(f"Converted to wide format: {wide_csv}")
    except Exception as e:
        print(f"Conversion failed: {type(e).__name__}: {str(e)[:100]}")
        return

    print("\n" + "=" * 70)
    print("STEP 3/3: POST-PROCESSING - Filling missing data")
    print("=" * 70 + "\n")

    from src.post_processing.pipeline import run_pipeline

    postprocess_dir = output_base / "post_processing"
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(
            csv_path=str(wide_csv),
            outdir=str(postprocess_dir),
            model_name=PostprocessingConfig.POST_MODEL,
            force_reprocess=False,
        )
        print(f"Post-processing complete: {postprocess_dir}")
    except Exception as e:
        print(f"Post-processing failed: {type(e).__name__}: {str(e)[:100]}")
        return

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE - {pdf_stem}")
    print("=" * 70)
    print("\nOutputs:")
    print(f"Pre-processing: {preprocess_dir}")
    print(f"Wide format: {wide_csv}")
    print(f"Post-processing: {postprocess_dir}")


def run_postprocess_only(input_path: Path, output_base: Path) -> None:
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        return

    postprocess_dir = output_base / "post_processing"
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    from src.post_processing.pipeline import run_pipeline, run_pipeline_batch

    if input_path.is_dir():
        print("\n" + "=" * 70)
        print("POST-PROCESSING ONLY (BATCH) - Filling missing data from all CSVs")
        print("=" * 70 + "\n")

        try:
            run_pipeline_batch(
                csv_folder=str(input_path),
                outdir=str(postprocess_dir),
                model_name=PostprocessingConfig.POST_MODEL,
                force_reprocess=False,
            )
            print("\n" + "=" * 70)
            print("BATCH POST-PROCESSING COMPLETE")
            print("=" * 70)
            print("\nOutputs:")
            print(f"Input folder: {input_path}")
            print(f"Post-processing: {postprocess_dir}")
            print(f"Combined output: {postprocess_dir / 'dataset_with_fills.csv'}")
        except Exception as e:
            print(f"Batch post-processing failed: {type(e).__name__}: {str(e)[:100]}")
        return

    if input_path.suffix.lower() != ".csv":
        print(f"Error: Input must be a CSV file or directory: {input_path}")
        return

    print("\n" + "=" * 70)
    print("POST-PROCESSING ONLY - Filling missing data")
    print("=" * 70 + "\n")

    try:
        run_pipeline(
            csv_path=str(input_path),
            outdir=str(postprocess_dir),
            model_name=PostprocessingConfig.POST_MODEL,
            force_reprocess=False,
        )
        print("\n" + "=" * 70)
        print("POST-PROCESSING COMPLETE")
        print("=" * 70)
        print("\nOutputs:")
        print(f"Input CSV: {input_path}")
        print(f"Post-processing: {postprocess_dir}")
    except Exception as e:
        print(f"Post-processing failed: {type(e).__name__}: {str(e)[:100]}")


def run_full_pipeline_batch(pdf_folder: Path, output_base: Path) -> None:
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {pdf_folder}")
        return

    print("\n" + "=" * 70)
    print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
    print("=" * 70 + "\n")

    conversions_dir = output_base / "conversions"
    conversions_dir.mkdir(parents=True, exist_ok=True)

    from src.pre_processing.pipeline_preprocess import process_pdf_full
    from scripts.convert_to_wide_format import convert_json_to_wide_csv

    wide_csvs: list[Path] = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        pdf_stem = pdf_path.stem
        preprocess_dir = output_base / "preprocessing" / pdf_stem
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        try:
            process_pdf_full(pdf_path, preprocess_dir)
        except Exception as e:
            print(f"Pre-processing failed: {type(e).__name__}")
            continue

        long_json = preprocess_dir / f"{pdf_stem}_output.json"
        wide_csv = conversions_dir / f"{pdf_stem}_wide.csv"

        try:
            convert_json_to_wide_csv(str(long_json), str(wide_csv))
            wide_csvs.append(wide_csv)
            print(f"Converted: {wide_csv}")
        except Exception as e:
            print(f"Conversion failed: {type(e).__name__}")
            continue

    if not wide_csvs:
        print("\nNo files converted successfully. Skipping post-processing.")
        return

    print("\n" + "=" * 70)
    print("BATCH POST-PROCESSING - Filling missing data across all papers")
    print("=" * 70 + "\n")

    from src.post_processing.pipeline import run_pipeline_batch

    postprocess_dir = output_base / "post_processing"
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline_batch(
            csv_folder=str(conversions_dir),
            outdir=str(postprocess_dir),
            model_name=PostprocessingConfig.POST_MODEL,
            force_reprocess=False,
        )
    except Exception as e:
        print(f"Batch post-processing failed: {type(e).__name__}: {str(e)[:100]}")
        return

    print("\n" + "=" * 70)
    print("BATCH PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print(f"Pre-processing: {output_base / 'preprocessing'}")
    print(f"Wide format: {conversions_dir}")
    print(f"Post-processing: {postprocess_dir}")
    print(f"Combined output: {postprocess_dir / 'dataset_with_fills.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="PDF file, folder, or CSV file (if --postprocess-only)",
    )
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()

    if not args.input:
        print("Error: Input path required")
        parser.print_help()
        raise SystemExit(1)

    input_path = Path(args.input)
    output_base = Path(args.output)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        raise SystemExit(1)

    if args.postprocess_only:
        run_postprocess_only(input_path, output_base)
        return

    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            print("Error: --batch requires a directory path")
            raise SystemExit(1)
        run_full_pipeline_batch(input_path, output_base)
        return

    run_full_pipeline_single(input_path, output_base)


if __name__ == "__main__":
    main()
