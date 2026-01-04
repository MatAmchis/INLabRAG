from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import base64
import csv
import io
import logging
import re

import fitz

try:
    import pymupdf_layout  # noqa: F401
    PYMUPDF_LAYOUT_AVAILABLE = True
except ImportError:
    PYMUPDF_LAYOUT_AVAILABLE = False

from .vlm_captioning import (
    caption_table_gpt4o,
    caption_figure_gpt4o,
    _CAPTION_CACHE,
    _save_caption_cache,
    _compute_pdf_digest,
    _hash_text,
)

LOGGER = logging.getLogger("snippet_builder")

USE_VLM_FOR_TABLES = True
SANITIZE_TABLE_CAPTIONS = True

NUMERIC_Q_PAT = re.compile(
    r"("
    r"cost|expens|expenditure|spend|price|budget|fund|financial|economic|"
    r"usd|dollar|\$|rupee|inr|euro|£|¥|"
    r"infect|disease|preval|inciden|mortal|morbid|death|fatalit|surviv|"
    r"diagnos|detect|screen|positive|negative|test|"
    r"rate|percent|%|ratio|proportion|frequen|odds|risk|probabil|likelihood|"
    r"statistic|significan|p-value|confidence|interval|"
    r"coverage|uptake|enrol|particip|adher|complian|retention|dropout|attrition|"
    r"cases?|patient|subject|sample|population|cohort|recruit|"
    r"number|count|total|mean|median|average|range|estimat|"
    r"measure|outcome|result|finding|value|level|score|"
    r"duration|period|year|month|week|day|follow-?up|baseline|"
    r"sensitiv|specific|accurac|efficac|effect|impact|reduc|increas|improv"
    r")",
    re.I,
)
NUMERIC_TSV_CHARS = 10000
NUMERIC_TSV_ROWS = 100

BAD_CAPTION_PAT = re.compile(
    r"\b(i\s*am|i'?m|unable|cant|cannot|sorry|no\s+table|does\s+not\s+contain)\b",
    re.I,
)

MAX_TOKENS = 300
STRIDE = 60
MERGE_LIMIT = 540

MAX_CAPTION_TOKENS = 120
MAX_CTX_TOKENS = 120

CITATION_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)
TOKEN_SPLIT = re.compile(r"\s+")
_SENT_END = re.compile(r"[.!?]['\")\]]?$")
_TOKEN_PAT = re.compile(r"[a-z0-9]+")

FLATTEN_TABLE_CELLS = True
FLATTEN_TABLE_MAX_ROWS = 10
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


def _pixmap_bytes(page: fitz.Page, rect: fitz.Rect, dpi: int = 150) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pm.tobytes("png")


try:
    import pytesseract
    from PIL import Image

    def _ocr_png(png_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(png_bytes))
            txt = pytesseract.image_to_string(img)
            return re.sub(r"\s+", " ", txt).strip()
        except Exception:
            return ""

except Exception:

    def _ocr_png(png_bytes: bytes) -> str:
        return ""


def _b64_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _plausible(title: str) -> bool:
    bad = {"microsoft", "powerpoint", "slide", "untitled", ".com", "http", "supplement", "table"}
    words = title.strip().split()
    return (
        5 <= len(words) <= 50
        and not all(w.isupper() for w in words)
        and not any(tok in title.lower() for tok in bad)
    )


def _first_page_title(pdf_path: str) -> str:
    page0 = fitz.open(pdf_path)[0]
    top_block = max(
        page0.get_text("dict")["blocks"],
        key=lambda b: max(
            (s["size"] for ln in b.get("lines", []) for s in ln["spans"]), default=0
        ),
    )
    text = " ".join(s["text"] for ln in top_block.get("lines", []) for s in ln["spans"])
    return re.sub(r"\s+", " ", text).strip()


def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        toks = TOKEN_SPLIT.split(p)
        if len(toks) <= MAX_TOKENS:
            chunks.append(p)
        else:
            step = MAX_TOKENS - STRIDE
            for i in range(0, len(toks), step):
                chunks.append(" ".join(toks[i : i + MAX_TOKENS]).strip())
    return chunks


def _reflow_block_text(raw: str) -> str:
    if not raw:
        return ""
    lines = raw.splitlines()
    out_lines, buf = [], []
    for ln in lines:
        s = ln.strip()
        if not s:
            if buf:
                out_lines.append(" ".join(buf))
                buf = []
            out_lines.append("")
            continue
        if not buf:
            buf.append(s)
        else:
            if _SENT_END.search(buf[-1]):
                buf.append(s)
            else:
                buf[-1] = buf[-1] + " " + s
    if buf:
        out_lines.append(" ".join(buf))
    cleaned, prev_blank = [], False
    for l in out_lines:
        if l == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(l)
            prev_blank = False
    return "\n".join(cleaned).strip()


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_text_blocks(page: fitz.Page) -> List[str]:
    result = []
    blocks = page.get_text("dict")["blocks"]

    for b in blocks:
        if b["type"] != 0:
            continue

        text_lines = []
        for line in b.get("lines", []):
            words = [w["text"] for w in line.get("spans", [])]
            if words:
                text_lines.append(" ".join(words))

        if not text_lines:
            continue

        block_text = " ".join(text_lines).strip()
        block_text = _clean_text(block_text)

        if block_text:
            result.append(block_text)

    return result


def _table_to_tsv(table: Any) -> str:
    try:
        rows = table.extract()
    except Exception as e:
        LOGGER.debug("table.extract() failed: %s", e)
        return ""

    out = io.StringIO()
    w = csv.writer(out, delimiter="\t")

    def _cell_text(c: Any) -> str:
        if c is None:
            return ""
        if isinstance(c, (list, tuple)):
            c = c[0] if c else ""
        try:
            return str(c).strip()
        except Exception:
            return ""

    for r in rows or []:
        w.writerow([_cell_text(c) for c in r])

    return out.getvalue()


def _table_preview(rows_tsv: str, max_lines: int = 8, max_chars: int = 400) -> str:
    lines = rows_tsv.splitlines()[:max_lines]
    txt = "\n".join(lines)
    return txt[:max_chars]


def _flatten_table_cells_for_retrieval(
    tsv: str,
    max_rows: int = FLATTEN_TABLE_MAX_ROWS,
    max_cols: int = FLATTEN_TABLE_MAX_COLS,
    max_chars: int = FLATTEN_TABLE_MAX_CHARS,
    max_cell_chars: int = FLATTEN_TABLE_MAX_CELL_CHARS,
) -> str:
    if not tsv:
        return ""
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    if not lines:
        return ""

    header = [c.strip() for c in lines[0].split("\t")]
    data_lines = lines[1 : 1 + max_rows]

    pieces = []
    for ln in data_lines:
        cells = [c.strip() for c in ln.split("\t")]
        for j, (h, v) in enumerate(zip(header, cells)):
            if j >= max_cols:
                break
            if not h or not v:
                continue
            v = re.sub(r"\s+", " ", v)
            if len(v) > max_cell_chars:
                v = v[: max_cell_chars - 1] + "…"
            pieces.append(f"{h}={v}")
        if len(pieces) >= max_rows * max_cols:
            break

    flat = "; ".join(pieces)
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1] + "…"
    return flat


def _flatten_table(tsv: str) -> str:
    if not tsv:
        return ""
    lines = tsv.splitlines()
    if not lines:
        return ""

    header = lines[0].split("\t")
    rows = lines[1:5]

    parts = []
    for r in rows:
        cells = r.split("\t")
        for h, v in zip(header, cells):
            parts.append(f"{h}={v[:40]}")

    flat = "; ".join(parts)
    return flat[:600]


def _coerce_bbox(bbox_like: Any) -> Optional[fitz.Rect]:
    if bbox_like is None:
        return None
    if isinstance(bbox_like, fitz.Rect):
        return bbox_like
    if isinstance(bbox_like, (tuple, list)) and len(bbox_like) == 4:
        try:
            return fitz.Rect(*bbox_like)
        except Exception:
            return None
    try:
        return fitz.Rect(bbox_like.x0, bbox_like.y0, bbox_like.x1, bbox_like.y1)
    except Exception:
        return None


def _image_blocks(page: fitz.Page) -> List[Tuple[fitz.Rect, int]]:
    blocks = page.get_text("dict")["blocks"]
    out: List[Tuple[fitz.Rect, int]] = []
    idx = 0
    for b in blocks:
        if b.get("type") == 1:
            x0, y0, x1, y1 = b["bbox"]
            out.append((fitz.Rect(x0, y0, x1, y1), idx))
            idx += 1
    return out


def _classify_image_role(page: fitz.Page, rect: fitz.Rect, page_number: int) -> str:
    try:
        page_h = page.rect.height
        page_w = page.rect.width
    except Exception:
        return "figure"

    y0_norm = rect.y0 / page_h
    width_norm = (rect.x1 - rect.x0) / page_w
    height_norm = (rect.y1 - rect.y0) / page_h

    if page_number in (1, 2) and y0_norm < 0.30 and width_norm > 0.55 and height_norm > 0.10:
        if height_norm < 0.45:
            return "abstract"

    if page_number == 1 and 0.12 < y0_norm < 0.40 and height_norm < 0.10 and width_norm > 0.40:
        return "authors"

    return "figure"


def load_pdf_with_vlm(pdf_path: str | Path):
    pdf_path = Path(pdf_path).expanduser().resolve()
    doc = fitz.open(str(pdf_path))
    pdf_digest = _compute_pdf_digest(str(pdf_path))

    snippets: List[Snippet] = []
    full_text_parts: List[str] = []

    for page in doc:
        pno = page.number + 1

        try:
            raw_text = page.get_text() or ""
            raw_text = _reflow_block_text(raw_text)
        except Exception as exc:
            LOGGER.warning("get_text error p%s: %s", pno, exc)
            raw_text = ""

        full_text_parts.append(raw_text)

        for idx, para in enumerate(_split_into_paragraphs(raw_text)):
            text = para.strip()
            if not text:
                continue
            snippets.append(
                Snippet(
                    text=text,
                    page=pno,
                    para_idx=idx,
                    kind="text",
                    raw=para,
                )
            )

        table_rects: List[fitz.Rect] = []
        try:
            if hasattr(page, "find_tables"):
                tf = page.find_tables()
                tables = tf.tables if tf else []
            else:
                tables = page.get_tables() or []

            for t_idx, table in enumerate(tables):
                try:
                    tsv = _table_to_tsv(table)
                    csv_preview = _table_preview(tsv)

                    rect = _coerce_bbox(getattr(table, "bbox", None))
                    if rect is None:
                        rect = page.rect

                    png_bytes = None
                    try:
                        png_bytes = _pixmap_bytes(page, rect)
                    except Exception as e_img:
                        LOGGER.debug("table pixmap fallback p%s#%s: %s", pno, t_idx, e_img)

                    cache_key = f"{pdf_digest}:{pno}:{1000 + t_idx}:{_hash_text(tsv or csv_preview)}"

                    if png_bytes:
                        cap = caption_table_gpt4o(
                            pno,
                            png_bytes,
                            csv_preview,
                            cache_key=cache_key,
                        )
                    else:
                        if cache_key in _CAPTION_CACHE:
                            cap = _CAPTION_CACHE[cache_key]
                        else:
                            if csv_preview:
                                cap = f"Table detected (no image capture). Headers preview: {csv_preview[:160]}"
                            else:
                                cap = "Table detected (no preview)."
                            _CAPTION_CACHE[cache_key] = cap
                            _save_caption_cache()

                    rect_tuple = (
                        (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                        if rect
                        else None
                    )

                    flat_cells = (
                        _flatten_table_cells_for_retrieval(tsv)
                        if FLATTEN_TABLE_CELLS
                        else ""
                    )
                    if flat_cells:
                        table_text = f"[TABLE p{pno}] {cap} || {flat_cells}"
                    else:
                        table_text = f"[TABLE p{pno}] {cap}"

                    snippets.append(
                        Snippet(
                            text=table_text,
                            page=pno,
                            para_idx=1000 + t_idx,
                            kind="table",
                            raw=tsv if tsv else None,
                            bbox=rect_tuple,
                        )
                    )

                    if rect is not None:
                        table_rects.append(rect)

                except Exception as e_tbl:
                    LOGGER.warning("Table parse fail p%s#%s: %s", pno, t_idx, e_tbl)
                    snippets.append(
                        Snippet(
                            text=f"[TABLE p{pno}] (table present; parse failed)",
                            page=pno,
                            para_idx=1000 + t_idx,
                            kind="table",
                            raw=None,
                            bbox=None,
                        )
                    )
        except Exception as exc:
            LOGGER.warning("Table detection error p%s: %s", pno, exc)

        try:
            for ib_rect, ib_idx in _image_blocks(page):
                if not isinstance(ib_rect, fitz.Rect):
                    ib_rect = _coerce_bbox(ib_rect) or page.rect

                try:
                    if any(ib_rect.intersects(tr) for tr in table_rects):
                        continue
                except Exception:
                    pass

                png_bytes = None
                try:
                    png_bytes = _pixmap_bytes(page, ib_rect)
                except Exception as e_img:
                    LOGGER.debug("image pixmap fallback p%s#%s: %s", pno, ib_idx, e_img)

                role = _classify_image_role(page, ib_rect, pno)
                kind_map = {
                    "abstract": "abstract_image",
                    "authors": "authors_image",
                    "figure": "figure",
                }
                snippet_kind = kind_map.get(role, "image")

                if png_bytes:
                    cap = caption_figure_gpt4o(
                        pno,
                        png_bytes,
                        pdf_digest=pdf_digest,
                        role=role,
                    )
                else:
                    cap = f"{role.capitalize()} image detected (no capture) [p{pno}]"

                rect_tuple = (
                    (float(ib_rect.x0), float(ib_rect.y0), float(ib_rect.x1), float(ib_rect.y1))
                    if ib_rect
                    else None
                )

                if role == "abstract":
                    prefix = "[ABSTRACT_IMAGE"
                elif role == "authors":
                    prefix = "[AUTHORS_IMAGE"
                else:
                    prefix = "[IMAGE"

                snippets.append(
                    Snippet(
                        text=f"{prefix} p{pno}] {cap}",
                        page=pno,
                        para_idx=2000 + ib_idx,
                        kind=snippet_kind,
                        raw=None,
                        bbox=rect_tuple,
                    )
                )
        except Exception as exc:
            LOGGER.warning("Image detection error p%s: %s", pno, exc)

    snippets.sort(key=lambda s: (s.page, s.para_idx))
    full_text = "\n".join(full_text_parts)
    meta = doc.metadata or {}
    doc.close()

    return snippets, meta, full_text
