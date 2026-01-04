```python
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Optional

from .env_setup import ensure_openai_api_key

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from config.preprocessing import PreprocessingConfig

    MAX_CAPTION_TOKENS = PreprocessingConfig.MAX_CAPTION_TOKENS
    VLM_MODEL = PreprocessingConfig.VLM_MODEL
    USE_VLM_FOR_TABLES = PreprocessingConfig.USE_VLM_FOR_TABLES
    SANITIZE_TABLE_CAPTIONS = PreprocessingConfig.SANITIZE_TABLE_CAPTIONS
    API_MAX_RETRIES = PreprocessingConfig.API_MAX_RETRIES
    API_INITIAL_BACKOFF = PreprocessingConfig.API_INITIAL_BACKOFF
    API_BACKOFF_MULT = PreprocessingConfig.API_BACKOFF_MULT
    API_BACKOFF_MAX = PreprocessingConfig.API_BACKOFF_MAX
    API_JITTER_FRAC = PreprocessingConfig.API_JITTER_FRAC
    MODEL_CACHE_DIR = PreprocessingConfig.MODEL_CACHE_DIR
except ImportError:
    MAX_CAPTION_TOKENS = 120
    VLM_MODEL = os.getenv("VLM_MODEL", "gpt-5.2")
    USE_VLM_FOR_TABLES = True
    SANITIZE_TABLE_CAPTIONS = True
    API_MAX_RETRIES = 6
    API_INITIAL_BACKOFF = 1.0
    API_BACKOFF_MULT = 2.0
    API_BACKOFF_MAX = 30.0
    API_JITTER_FRAC = 0.25
    MODEL_CACHE_DIR = os.path.expanduser("~/.cache/research_pipeline")

CITATION_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)
BAD_CAPTION_PAT = re.compile(
    r"\b(i\s*am|i'?m|unable|cant|cannot|sorry|no\s+table|does\s+not\s+contain)\b",
    re.I,
)

_OPENAI_CLIENT = None
_CAPTION_CACHE: Dict[str, str] = {}
_CAPTION_CACHE_PATH: Optional[Path] = None


def _init_caption_cache() -> None:
    global _CAPTION_CACHE, _CAPTION_CACHE_PATH
    _CAPTION_CACHE_PATH = Path(MODEL_CACHE_DIR) / "_caption_cache.json"
    _CAPTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _CAPTION_CACHE_PATH.exists():
        try:
            with open(_CAPTION_CACHE_PATH, "r", encoding="utf-8") as f:
                _CAPTION_CACHE = json.load(f)
            LOGGER.info("Loaded %d cached captions", len(_CAPTION_CACHE))
        except Exception:
            _CAPTION_CACHE = {}


def _fallback_table_caption(page_num: int, csv_preview: str) -> str:
    headers = []
    if csv_preview:
        lines = csv_preview.splitlines()
        if lines:
            first = lines[0]
            headers = [h.strip() for h in first.split("\t") if h.strip()]
    if headers:
        head_txt = ", ".join(headers[:8])
        return f"Table; columns: {head_txt} [p{page_num}]"
    if csv_preview:
        row_count = len(csv_preview.splitlines())
        return f"Table (no clear headers; {row_count} rows parsed) [p{page_num}]"
    return f"Table (no preview parsed) [p{page_num}]"


def _sanitize_table_caption(cap: str, page_num: int, csv_preview: str) -> str:
    if not cap or BAD_CAPTION_PAT.search(cap) or len(cap) > 400:
        return _fallback_table_caption(page_num, csv_preview)
    cap = re.sub(r"\s+", " ", cap).strip()
    if f"[p{page_num}]" not in cap:
        cap = CITATION_RE.sub("", cap).strip()
        cap = cap + f" [p{page_num}]"
    return cap.strip()


def _compute_pdf_digest(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return "unknown"


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _load_caption_cache(path: Optional[Path] = None) -> Dict[str, str]:
    global _CAPTION_CACHE, _CAPTION_CACHE_PATH
    if path is None:
        path = Path(MODEL_CACHE_DIR) / "_caption_cache.json"
    _CAPTION_CACHE_PATH = path

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                _CAPTION_CACHE = json.load(f)
        except Exception:
            _CAPTION_CACHE = {}
    return _CAPTION_CACHE


def _save_caption_cache() -> None:
    if _CAPTION_CACHE_PATH is None:
        return
    try:
        _CAPTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CAPTION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CAPTION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.debug("Failed to save caption cache: %s", e)


def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        ensure_openai_api_key()
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(timeout=600)
    return _OPENAI_CLIENT


def _call_with_backoff(fn, *args, **kwargs):
    delay = API_INITIAL_BACKOFF
    for attempt in range(API_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            is_rate = ("RateLimit" in msg) or ("429" in msg)
            if (not is_rate) and (attempt == API_MAX_RETRIES - 1):
                raise
            jitter = delay * random.uniform(1 - API_JITTER_FRAC, 1 + API_JITTER_FRAC)
            time.sleep(jitter)
            delay = min(delay * API_BACKOFF_MULT, API_BACKOFF_MAX)
    raise RuntimeError("API retries exhausted")


def _png_to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _caption_via_responses(
    prompt: str,
    img_data_uri: Optional[str],
    max_tokens: int = MAX_CAPTION_TOKENS,
) -> str:
    client = _get_client()
    try:
        if img_data_uri is not None:
            try:
                resp = _call_with_backoff(
                    client.responses.create,
                    model=VLM_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": img_data_uri},
                            ],
                        }
                    ],
                    max_output_tokens=max_tokens,
                    temperature=0,
                )
                try:
                    text = resp.output_text.strip()
                    if text:
                        return text
                except Exception:
                    pass

                out_texts = []
                for item in getattr(resp, "output", []):
                    if getattr(item, "type", None) == "message":
                        for c in getattr(item.message, "content", []):
                            if getattr(c, "type", None) == "output_text":
                                out_texts.append(getattr(c, "text", ""))
                text = " ".join(out_texts).strip()
                return text if text else "(no caption)"
            except Exception:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_data_uri, "detail": "high"},
                            },
                        ],
                    }
                ]
                resp = _call_with_backoff(
                    client.chat.completions.create,
                    model=VLM_MODEL,
                    temperature=0,
                    max_completion_tokens=max_tokens,
                    messages=messages,
                )
                return resp.choices[0].message.content.strip()

        resp = _call_with_backoff(
            client.chat.completions.create,
            model=VLM_MODEL,
            temperature=0,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        LOGGER.warning("Caption error (%s); fallback text-only.", e)
        try:
            resp = _call_with_backoff(
                client.chat.completions.create,
                model=VLM_MODEL,
                temperature=0,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "(caption error)"


def caption_table_gpt4o(
    page_num: int,
    png_bytes: bytes,
    csv_preview: str,
    cache_key: Optional[str] = None,
) -> str:
    key = cache_key or _hash_bytes(png_bytes + csv_preview.encode("utf-8"))
    if key in _CAPTION_CACHE:
        cap_cached = _CAPTION_CACHE[key]
        if SANITIZE_TABLE_CAPTIONS:
            cap_clean = _sanitize_table_caption(cap_cached, page_num, csv_preview)
            if cap_clean != cap_cached:
                _CAPTION_CACHE[key] = cap_clean
                _save_caption_cache()
            return cap_clean
        return cap_cached

    cap_raw = ""
    if USE_VLM_FOR_TABLES:
        cap_raw = _caption_via_responses(
            prompt=(
                "You are describing a table extracted from a scientific PDF for vector retrieval. "
                "Write a concise, high-signal caption of less than 120 tokens that names the table's subject, "
                "key column headers, units, and any striking values or comparisons. "
                f"Include the inline token [p{page_num}] exactly once so downstream models can cite the page. "
                "Do not reproduce the full table. Summarize. If the table is mostly empty say so. "
                "Preview of first rows (TSV): "
                + csv_preview[:2000]
            ),
            img_data_uri=_png_to_data_uri(png_bytes),
        )

    cap = _sanitize_table_caption(cap_raw, page_num, csv_preview) if SANITIZE_TABLE_CAPTIONS else cap_raw
    _CAPTION_CACHE[key] = cap
    _save_caption_cache()
    return cap


def caption_figure_gpt4o(
    page_num: int,
    png_bytes: bytes,
    pdf_digest: Optional[str] = None,
    role: str = "figure",
) -> str:
    prefix = f"{pdf_digest}:" if pdf_digest else ""
    key = prefix + _hash_bytes(png_bytes) + f":{role}"
    if key in _CAPTION_CACHE:
        return _CAPTION_CACHE[key]

    if role == "abstract":
        prompt = (
            "You are shown an IMAGE of an ABSTRACT section extracted from a scientific PDF. "
            "Summarize the study abstract concisely in less than 120 tokens, capturing objective, setting/population, "
            "methods (brief), key quantitative or qualitative results, and main conclusion. "
            f"Include exactly one page token [p{page_num}]. Do NOT add extra commentary."
        )
    elif role == "authors":
        prompt = (
            "You are shown an IMAGE containing the paper's title block (authors / affiliations). "
            "Extract ONLY the list of author surnames in order, separated by commas, plus year if visible. "
            f"Include exactly one page token [p{page_num}]. No affiliations, no initials beyond disambiguating letters."
        )
    else:
        prompt = (
            "You are describing a figure / diagram / image block from a scientific PDF for retrieval. "
            "State what it depicts (variables, population, time frame, notable patterns) in <120 tokens. "
            f"Include exactly one page token [p{page_num}] once."
        )

    cap = _caption_via_responses(prompt, _png_to_data_uri(png_bytes))
    cap = re.sub(r"\s+", " ", cap).strip()

    if not cap or cap.lower().startswith("(caption error)") or len(cap) < 10:
        ocr_txt = _ocr_png(png_bytes)
        if ocr_txt:
            ocr_txt = ocr_txt[:250].strip()
            if f"[p{page_num}]" not in ocr_txt:
                cap = f"{ocr_txt} [p{page_num}]"
            else:
                cap = ocr_txt

    if f"[p{page_num}]" not in cap:
        cap = CITATION_RE.sub("", cap).strip()
        cap = f"{cap} [p{page_num}]".strip()

    _CAPTION_CACHE[key] = cap
    _save_caption_cache()
    return cap


def _b64_data_uri(png_bytes: bytes) -> str:
    return _png_to_data_uri(png_bytes)


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


_init_caption_cache()
