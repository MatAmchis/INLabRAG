from __future__ import annotations
from pathlib import Path
import fitz

def load_pdf_pages(pdf_path: str | Path):
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc):
        pages.append(
            {
                "page": page_num,
                "text": page.get_text(),
                "blocks": page.get_text("blocks"),
                "images": page.get_images(full=True),
                "tables": [],
            }
        )

    return pages


def extract_page_blocks(page_dict: dict):
    return page_dict.get("blocks", [])


def extract_page_images(page_dict: dict):
    return page_dict.get("images", [])


def extract_page_text(page_dict: dict):
    return page_dict.get("text", "")
