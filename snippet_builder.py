from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from src.pre_processing.vlm_captioning import caption_figure_gpt4o, caption_table_gpt4o

LOGGER = logging.getLogger("snippet_builder")

FLATTEN_TABLE_CELLS = True
FLATTEN_TABLE_MAX_ROWS = 4
FLATTEN_TABLE_MAX_COLS = 8
FLATTEN_TABLE_MAX_CHARS = 600
FLATTEN_TABLE_MAX_CELL_CHARS = 40


@dataclass
class Snippet:
    text: str
    page: int
    para_idx: int
    kind: str
    raw: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None


def _split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _table_to_tsv(table) -> str:
    try:
        rows = table.extract()
    except Exception:
        return ""

    out = io.StringIO()
    w = csv.writer(out, delimiter="\t")

    def _cell(c) -> str:
        if c is None:
            return ""
        return str(c)

    for r in rows:
        w.writerow([_cell(c) for c in r])

    return out.getvalue().strip()


def _flatten_table(tsv: str) -> str:
    if not tsv:
        return ""
    lines = tsv.splitlines()
    if not lines:
        return ""

    header = lines[0].split("\t")
    rows = lines[1 : 1 + FLATTEN_TABLE_MAX_ROWS]

    flattened: List[str] = []
    for r in rows:
        cells = r.split("\t")
        for h, v in zip(header, cells):
            flattened.append(f"{h}={v[:FLATTEN_TABLE_MAX_CELL_CHARS]}")
    out = "; ".join(flattened)
    return out[:FLATTEN_TABLE_MAX_CHARS]


def load_pdf_with_vlm(pdf_path: str | Path, client):
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)

    snippets: List[Snippet] = []
    full_text: List[str] = []

    for pno, page in enumerate(doc):
        text = page.get_text()
        full_text.append(text)

        for idx, para in enumerate(_split_into_paragraphs(text)):
            snippets.append(Snippet(text=para, page=pno, para_idx=idx, kind="text", raw=para))

        try:
            tables = page.get_tables() or []
            for t_idx, t in enumerate(tables):
                tsv = _table_to_tsv(t)
                flat = _flatten_table(tsv) if FLATTEN_TABLE_CELLS else tsv

                try:
                    rect = fitz.Rect(t.bbox)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                    image_bytes = pix.tobytes("png")
                    table_key = f"table:{pno}:{t_idx}"
                    cap = caption_table_gpt4o(image_bytes, table_key, client)
                except Exception:
                    cap = "Table detected."

                snippets.append(
                    Snippet(
                        text=f"[TABLE p{pno}] {cap}\n{flat}",
                        page=pno,
                        para_idx=1000 + t_idx,
                        kind="table",
                        raw=tsv,
                    )
                )
        except Exception:
            pass

        try:
            blocks = page.get_text("dict").get("blocks", [])
            img_blocks = [b for b in blocks if b.get("type") == 1]

            for ib_idx, b in enumerate(img_blocks):
                x0, y0, x1, y1 = b["bbox"]
                rect = fitz.Rect(x0, y0, x1, y1)

                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                    image_bytes = pix.tobytes("png")
                    fig_id = f"figure:{pno}:{ib_idx}"
                    cap = caption_figure_gpt4o(image_bytes, fig_id, client, role="figure")
                except Exception:
                    cap = "Figure detected."

                snippets.append(
                    Snippet(
                        text=f"[FIGURE p{pno}] {cap}",
                        page=pno,
                        para_idx=2000 + ib_idx,
                        kind="image",
                        raw=None,
                        bbox=(x0, y0, x1, y1),
                    )
                )
        except Exception:
            pass

    return snippets, doc.metadata, "\n".join(full_text)
