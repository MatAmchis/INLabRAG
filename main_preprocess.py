from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config.preprocessing import PreprocessingConfig, set_seed as set_preprocess_seed
from src.pre_processing.pipeline_preprocess import process_pdf_full

set_preprocess_seed(PreprocessingConfig.SEED)

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def process_pdf(pdf_path: Path, out_dir: Path) -> None:
    process_pdf_full(pdf_path, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=str)
    parser.add_argument("output_dir", type=str, nargs="?", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("output/preprocessing") / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_pdf(pdf_path, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
