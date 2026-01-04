import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config.postprocessing import PostprocessingConfig, set_seed as set_post_seed
from src.post_processing import run_pipeline_batch, run_pipeline

set_post_seed(PostprocessingConfig.SEED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--outdir", type=str, default="output/post_processing")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    model_name = args.model or os.getenv("POST_MODEL", "gpt-4o-mini")

    try:
        if input_path.is_dir():
            run_pipeline_batch(
                csv_folder=str(input_path),
                outdir=args.outdir,
                model_name=model_name,
                force_reprocess=args.force,
            )
        else:
            stem = input_path.stem
            outdir = f"{args.outdir}/{stem}"
            run_pipeline(
                csv_path=str(input_path),
                outdir=outdir,
                model_name=model_name,
            )
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError:
        sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
