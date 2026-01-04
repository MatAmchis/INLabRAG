from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from .env_setup import ensure_openai_api_key

env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from config.preprocessing import PreprocessingConfig

    MERGE_LIMIT = PreprocessingConfig.MERGE_LIMIT
    TOP_K_SBERT = PreprocessingConfig.TOP_K_SBERT
    TOP_K_OPENAI = PreprocessingConfig.TOP_K_OPENAI
    TOP_BM25 = PreprocessingConfig.TOP_BM25
    TOP_N_FINAL = PreprocessingConfig.TOP_N_FINAL
    MAX_CTX_TOKENS = PreprocessingConfig.MAX_CTX_TOKENS
    CTX_MODEL = PreprocessingConfig.CTX_MODEL
    USE_QUERY_EXPANSION = PreprocessingConfig.USE_QUERY_EXPANSION
    MAX_EXPANSION_TERMS = PreprocessingConfig.MAX_EXPANSION_TERMS
except ImportError:
    MERGE_LIMIT = 540
    TOP_K_SBERT = 200
    TOP_K_OPENAI = 200
    TOP_BM25 = 200
    TOP_N_FINAL = 30
    MAX_CTX_TOKENS = 120
    CTX_MODEL = os.getenv("CTX_MODEL", "gpt-5.2")
    USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "1") not in {"0", "false", "False", "no", "No"}
    MAX_EXPANSION_TERMS = int(os.getenv("MAX_EXPANSION_TERMS", "5"))

TOKEN_SPLIT = re.compile(r"\s+")
CITATION_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)
_SENT_END = re.compile(r"[.!?]['\")\]]?$")
_TOKEN_PAT = re.compile(r"[a-z0-9]+")

QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "tb": ["tuberculosis", "mtb", "mycobacterium", "pulmonary tuberculosis", "ptb"],
    "tuberculosis": ["tb", "mtb", "mycobacterium"],
    "hiv": ["human immunodeficiency virus", "aids", "hiv/aids", "antiretroviral"],
    "aids": ["hiv", "acquired immunodeficiency syndrome"],
    "malaria": ["plasmodium", "mosquito-borne", "antimalarial"],
    "diabetes": ["diabetic", "blood sugar", "glucose", "hba1c", "insulin"],
    "hypertension": ["high blood pressure", "bp", "blood pressure", "antihypertensive"],
    "chagas": ["trypanosoma cruzi", "t. cruzi", "american trypanosomiasis"],
    "ntd": ["neglected tropical disease", "neglected tropical diseases"],
    "cost": ["expenditure", "expense", "price", "budget", "spending", "financial", "economic"],
    "cost-effective": ["cost-effectiveness", "value for money", "economic evaluation", "cea"],
    "usd": ["dollar", "dollars", "$", "us dollar", "united states dollar"],
    "budget": ["funding", "finance", "resource", "allocation"],
    "detection": ["screening", "diagnosis", "identification", "case finding", "detected"],
    "case finding": ["active case finding", "acf", "detection", "screening"],
    "screening": ["testing", "examination", "diagnosis", "detection"],
    "diagnosis": ["diagnosed", "diagnostic", "detection", "identified"],
    "patient": ["patients", "participant", "subject", "individual", "case", "person"],
    "population": ["community", "cohort", "sample", "group", "subjects"],
    "children": ["child", "pediatric", "paediatric", "infant", "adolescent", "youth"],
    "women": ["woman", "female", "maternal", "pregnant", "mother"],
    "men": ["man", "male", "father"],
    "rct": ["randomized controlled trial", "randomised controlled trial", "clinical trial"],
    "cohort": ["cohort study", "longitudinal", "prospective", "follow-up"],
    "cross-sectional": ["cross sectional", "survey", "prevalence study"],
    "qualitative": ["interview", "focus group", "fgd", "kii", "in-depth"],
    "quantitative": ["statistical", "numeric", "measurement", "survey"],
    "outcome": ["result", "finding", "effect", "impact", "endpoint"],
    "efficacy": ["effectiveness", "effect", "impact", "benefit"],
    "mortality": ["death", "deaths", "fatality", "died", "survival"],
    "morbidity": ["illness", "disease burden", "sickness"],
    "intervention": ["program", "programme", "initiative", "strategy", "approach"],
    "integration": ["integrated", "integrating", "combine", "combined", "linkage"],
    "implementation": ["implemented", "implementing", "rollout", "scale-up"],
    "training": ["trained", "capacity building", "education", "workshop"],
    "coverage": ["uptake", "utilization", "access", "reach", "enrollment"],
    "uptake": ["coverage", "adoption", "acceptance", "utilization"],
    "adherence": ["compliance", "retention", "follow-up", "continuation"],
    "barrier": ["barriers", "challenge", "obstacle", "constraint", "limitation"],
    "facilitator": ["facilitators", "enabler", "strength", "opportunity", "support"],
    "healthcare": ["health care", "medical care", "health service", "health system"],
    "hospital": ["clinic", "facility", "health center", "health centre"],
    "community health worker": ["chw", "lay health worker", "village health worker", "asha"],
    "primary care": ["primary health care", "phc", "first-level care"],
    "year": ["years", "annual", "annually", "per year", "yearly"],
    "month": ["months", "monthly", "per month"],
    "duration": ["period", "length", "time", "timeframe"],
    "prevalence": ["prevalent", "proportion", "frequency", "rate"],
    "incidence": ["incident", "new cases", "occurrence"],
    "rate": ["ratio", "proportion", "percentage", "percent"],
    "percent": ["%", "percentage", "proportion"],
}

Q_AWARE_TABLE_SUMMARIES = True
TABLE_Q_SUM_MAX_CHARS_IN = 4000
TABLE_Q_SUM_MAX_ROWS_OUT = 25
TABLE_Q_SUM_LLM_TOKENS = 160
_TABLE_Q_CACHE: Dict[Tuple[str, int, int], str] = {}

_METHODISH = (
    "method",
    "design",
    "data",
    "interview",
    "survey",
    "fgd",
    "focus",
    "questionnaire",
    "observation",
    "kii",
    "kiis",
    "kid",
    "kids",
    "sample",
    "analysis",
    "procedure",
    "protocol",
    "approach",
)

_COSTISH = (
    "cost",
    "cost-",
    "usd",
    "$",
    "expense",
    "economic",
    "efficien",
    "budget",
    "expenditure",
    "price",
    "funding",
    "financial",
    "per case",
)

_OUTCOMEISH = (
    "outcome",
    "result",
    "impact",
    "effect",
    "change",
    "diff",
    "finding",
    "improvement",
    "reduction",
    "measure",
    "indicator",
)

CTX_SYSTEM_MSG = (
    "Give a short succinct context to situate this chunk within the overall document for the purposes of improving "
    "search retrival of the chunk. Answer only with the succinct context and nothing else."
)


def expand_query(query: str) -> str:
    if not USE_QUERY_EXPANSION:
        return query

    query_lower = query.lower()
    words = set(_TOKEN_PAT.findall(query_lower))
    expansions: List[str] = []

    for word in words:
        syns = QUERY_EXPANSIONS.get(word)
        if not syns:
            continue
        for syn in syns[:MAX_EXPANSION_TERMS]:
            if syn.lower() not in query_lower:
                expansions.append(syn)

    for key, syns in QUERY_EXPANSIONS.items():
        if " " in key and key in query_lower:
            for syn in syns[:MAX_EXPANSION_TERMS]:
                if syn.lower() not in query_lower:
                    expansions.append(syn)

    if not expansions:
        return query

    expansion_str = " ".join(expansions[: MAX_EXPANSION_TERMS * 2])
    expanded = f"{query} {expansion_str}"
    LOGGER.debug("Query expansion: '%s' -> '%s'", query, expanded)
    return expanded


def get_query_keywords(query: str) -> List[str]:
    query_lower = query.lower()
    keywords: List[str] = []

    for word in _TOKEN_PAT.findall(query_lower):
        keywords.append(word)
        syns = QUERY_EXPANSIONS.get(word)
        if syns:
            keywords.extend(syns[:3])

    return list(set(keywords))


def _bm25_tokenize(text: str) -> List[str]:
    return _TOKEN_PAT.findall(text.lower())


def _contextualise_chunk(full_doc: str, chunk: str) -> str:
    ensure_openai_api_key()

    from openai import OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()
    prompt = (
        f"<document>\n{full_doc[:8000]}\n</document>\n\n"
        f"<chunk>\n{chunk}\n</chunk>\n\n"
        "Give a concise context to situate this chunk."
    )

    try:
        resp = _call_with_backoff(
            client.chat.completions.create,
            model=CTX_MODEL,
            temperature=0,
            max_completion_tokens=MAX_CTX_TOKENS,
            messages=[
                {"role": "system", "content": CTX_SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        LOGGER.warning("Contextualise error: %s", e)
        summary = ""

    return f"{chunk}\n\n{summary}" if summary else chunk


def _rrf(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for lst in ranked_lists:
        for r, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=scores.get, reverse=True)


def _merge_neighbours(ids: List[int], snippets: List[Any]) -> List[Any]:
    ids.sort(key=lambda i: (snippets[i].page, snippets[i].para_idx))
    merged: List[Any] = []
    buf_txt = ""
    buf_page: Optional[int] = None
    buf_idx: Optional[int] = None

    def flush() -> None:
        nonlocal buf_txt, buf_page, buf_idx
        if not buf_txt:
            return
        from collections import namedtuple

        Snippet = namedtuple("Snippet", ["text", "page", "para_idx", "kind"])
        merged.append(Snippet(buf_txt, buf_page or 0, buf_idx or 0, kind="text"))
        buf_txt = ""
        buf_page = None
        buf_idx = None

    for i in ids:
        s = snippets[i]
        if s.kind != "text":
            flush()
            merged.append(s)
            continue

        if buf_page is None:
            buf_txt, buf_page, buf_idx = s.text, s.page, s.para_idx
            continue

        adjacent = (s.page == buf_page) and (s.para_idx == (buf_idx or 0) + 1)
        if adjacent:
            tentative = f"{buf_txt}\n\n{s.text}"
            if len(TOKEN_SPLIT.split(tentative)) <= MERGE_LIMIT:
                buf_txt = tentative
                buf_idx = s.para_idx
                continue

        flush()
        buf_txt, buf_page, buf_idx = s.text, s.page, s.para_idx

    flush()
    return merged


def _format_context(snips: List[Any]) -> Tuple[str, List[int]]:
    ctx: List[str] = []
    pages: List[int] = []

    for s in snips:
        if s.kind == "table":
            tag = "TABLE"
        elif s.kind in ("figure", "image"):
            tag = "FIG"
        elif s.kind == "abstract_image":
            tag = "ABSTRACT_IMAGE"
        elif s.kind == "authors_image":
            tag = "AUTHORS_IMAGE"
        else:
            tag = "Snippet"

        ctx.append(f"{tag} [p{s.page}]: {s.text.strip()}")
        pages.append(s.page)

    return "\n".join(ctx), pages


def _qa_col_keywords(question: str) -> Tuple[str, ...]:
    ql = question.lower()

    if "method" in ql:
        return _METHODISH
    if "design" in ql:
        return ("design",) + _METHODISH
    if "cost" in ql:
        return _COSTISH
    if "acceptab" in ql:
        return ("accept", "satisf", "perception", "qual", "interview") + _METHODISH
    if "outcome" in ql or "impact" in ql:
        return _OUTCOMEISH

    toks = tuple(t for t in re.split(r"\W+", ql) if t)
    return toks or ("data",)


def _parse_tsv(tsv: str) -> Tuple[List[str], List[List[str]]]:
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    if not lines:
        return [], []

    rows = [ln.split("\t") for ln in lines]
    header: Optional[List[str]] = None
    header_row: Optional[List[str]] = None

    for r in rows:
        non_empty = [c.strip() for c in r if c.strip()]
        if len(non_empty) >= 2:
            header = [c.strip() for c in r]
            header_row = r
            break

    if header is None:
        width = max(len(r) for r in rows)
        header = [f"Col{j}" for j in range(width)]
        data_rows = rows
    else:
        idx = rows.index(header_row)  # type: ignore[arg-type]
        data_rows = rows[idx + 1 :]
        for dr in data_rows:
            if len(dr) < len(header):
                dr.extend([""] * (len(header) - len(dr)))

    return header, data_rows


def _extract_relevant_table_text(
    question: str,
    tsv: str,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_cell_chars: Optional[int] = None,
) -> str:
    from .snippet_builder import FLATTEN_TABLE_MAX_CELL_CHARS, FLATTEN_TABLE_MAX_COLS

    if not tsv:
        return ""

    if max_rows is None:
        max_rows = TABLE_Q_SUM_MAX_ROWS_OUT
    if max_cols is None:
        max_cols = FLATTEN_TABLE_MAX_COLS
    if max_chars is None:
        max_chars = TABLE_Q_SUM_MAX_CHARS_IN
    if max_cell_chars is None:
        max_cell_chars = FLATTEN_TABLE_MAX_CELL_CHARS

    if len(tsv) > max_chars:
        tsv = tsv[:max_chars]

    header, rows = _parse_tsv(tsv)
    if not header:
        return ""

    kws = _qa_col_keywords(question)
    col_mask = [any(k in h.lower() for k in kws) for h in header]
    if not any(col_mask):
        col_mask = [(i < max_cols) for i in range(len(header))]

    keep_idx = [i for i, m in enumerate(col_mask) if m][:max_cols]
    hdr_out = [header[i] for i in keep_idx]

    out_lines: List[str] = []
    for r in rows[:max_rows]:
        vals: List[str] = []
        for hi, i in enumerate(keep_idx):
            v = re.sub(r"\s+", " ", r[i]).strip() if i < len(r) else ""
            if not v:
                continue
            if len(v) > max_cell_chars:
                v = v[: max_cell_chars - 1] + "..."
            vals.append(f"{hdr_out[hi]}={v}")
        if vals:
            out_lines.append(" | ".join(vals))

    return "\n".join(out_lines).strip()


def _summarize_table_for_question(question: str, page: int, extract_text: str) -> str:
    if not extract_text:
        return ""

    prompt = (
        "You are helping extract data from a scientific table.\n"
        f"Question: {question}\n"
        f"This content comes from a table on page {page} of the PDF.\n"
        "Below are selected rows and columns likely relevant.\n"
        "Write a concise digest of less than 120 tokens that ONLY includes information that helps answer the question.\n"
        "List key values. Do NOT fabricate! Do NOT include page tags. No prose beyond what is needed.\n\n"
        f"{extract_text}"
    )

    try:
        ensure_openai_api_key()
        from openai import OpenAI
        from .embeddings import _call_with_backoff

        client = OpenAI()
        resp = _call_with_backoff(
            client.chat.completions.create,
            model=CTX_MODEL,
            temperature=0,
            max_completion_tokens=TABLE_Q_SUM_LLM_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        LOGGER.debug("Q-table LLM summarize fail: %s", e)
        return extract_text[:TABLE_Q_SUM_MAX_CHARS_IN]


def _qa_enrich_table_snippet(question: str, snip: Any) -> str:
    from .qa_extractor import NUMERIC_TSV_CHARS, NUMERIC_TSV_ROWS, _is_numeric_q
    from .snippet_builder import FLATTEN_TABLE_MAX_COLS, FLATTEN_TABLE_MAX_ROWS

    if not Q_AWARE_TABLE_SUMMARIES or snip.kind != "table" or not getattr(snip, "raw", None):
        return snip.text

    key = (question.lower(), snip.page, snip.para_idx)
    cached = _TABLE_Q_CACHE.get(key)
    if cached is not None:
        return cached

    if _is_numeric_q(question):
        mr = NUMERIC_TSV_ROWS
        mchars = NUMERIC_TSV_CHARS
    else:
        mr = FLATTEN_TABLE_MAX_ROWS
        mchars = TABLE_Q_SUM_MAX_CHARS_IN

    mc = FLATTEN_TABLE_MAX_COLS

    extract_txt = _extract_relevant_table_text(
        question,
        snip.raw,
        max_rows=mr,
        max_cols=mc,
        max_chars=mchars,
    )

    if extract_txt:
        sum_txt = _summarize_table_for_question(question, snip.page, extract_txt)
        if sum_txt and re.search(r"\d", sum_txt):
            payload = sum_txt
        else:
            payload = "\n".join(extract_txt.splitlines()[:20])
        enriched = f"{snip.text}\n[Q-relevant extract] {payload} [p{snip.page}]"
    else:
        enriched = snip.text

    _TABLE_Q_CACHE[key] = enriched
    return enriched


def _format_context_qaware(question: str, snips: List[Any]) -> Tuple[str, List[int]]:
    ctx: List[str] = []
    pages: List[int] = []

    for s in snips:
        if s.kind == "table":
            txt = _qa_enrich_table_snippet(question, s)
            tag = "TABLE"
        elif s.kind in ("figure", "image"):
            txt = s.text
            tag = "FIG"
        elif s.kind == "abstract_image":
            txt = s.text
            tag = "ABSTRACT_IMAGE"
        elif s.kind == "authors_image":
            txt = s.text
            tag = "AUTHORS_IMAGE"
        else:
            txt = s.text
            tag = "Snippet"

        ctx.append(f"{tag} [p{s.page}]: {txt.strip()}")
        pages.append(s.page)

    return "\n".join(ctx), pages


def _search_hybrid(query: str, sbert_index, openai_index, bm25, use_expansion: bool = True) -> List[int]:
    from .embeddings import embed_texts_openai, embed_texts_sbert

    lists: List[List[int]] = []
    expanded_query = expand_query(query) if (use_expansion and USE_QUERY_EXPANSION) else query

    if sbert_index is not None:
        q_sbert = embed_texts_sbert([query])
        if q_sbert is not None:
            _, sbert_ids = sbert_index.search(q_sbert.astype(np.float32), TOP_K_SBERT)
            lists.append(sbert_ids[0].tolist())

    if openai_index is not None:
        q_openai = embed_texts_openai([query])
        _, openai_ids = openai_index.search(q_openai.astype(np.float32), TOP_K_OPENAI)
        lists.append(openai_ids[0].tolist())

    if bm25 is not None:
        bm_scores_expanded = bm25.get_scores(_bm25_tokenize(expanded_query))
        bm_ids_expanded = np.argsort(bm_scores_expanded)[::-1][:TOP_BM25].tolist()

        if use_expansion and expanded_query != query:
            bm_scores_original = bm25.get_scores(_bm25_tokenize(query))
            bm_ids_original = np.argsort(bm_scores_original)[::-1][: max(1, TOP_BM25 // 2)].tolist()
            combined_bm25 = _rrf([bm_ids_expanded, bm_ids_original])[:TOP_BM25]
            lists.append(combined_bm25)
        else:
            lists.append(bm_ids_expanded)

    if not lists:
        return []

    fused_ids = _rrf(lists)[: max(TOP_K_SBERT, TOP_K_OPENAI, TOP_BM25)]
    return fused_ids


def build_corpus_indexes(snippets: List[Any], full_doc: str):
    from .embeddings import DUAL_DENSE, embed_texts_openai, embed_texts_sbert
    from tqdm import tqdm

    try:
        import faiss
    except ImportError:
        faiss = None

    ctx_texts: List[str] = []
    for s in tqdm(snippets, desc="Contextualising"):
        if s.kind == "text":
            ctx_texts.append(_contextualise_chunk(full_doc, s.text))
        else:
            ctx_texts.append(s.text)

    sbert_index = None
    if DUAL_DENSE and faiss is not None:
        sbert_vecs = embed_texts_sbert(ctx_texts)
        if sbert_vecs is not None:
            sbert_index = faiss.IndexFlatIP(sbert_vecs.shape[1])
            sbert_index.add(sbert_vecs)

    openai_index = None
    if faiss is not None:
        openai_vecs = embed_texts_openai(ctx_texts)
        openai_index = faiss.IndexFlatL2(openai_vecs.shape[1])
        openai_index.add(openai_vecs)

    bm25 = BM25Okapi([_bm25_tokenize(t) for t in ctx_texts])
    return sbert_index, openai_index, bm25, ctx_texts


def build_fallback_indexes(snippets: List[Any]):
    from .embeddings import DUAL_DENSE, embed_texts_openai, embed_texts_sbert

    try:
        import faiss
    except ImportError:
        faiss = None

    texts = [s.text for s in snippets]

    sbert_index = None
    if DUAL_DENSE and faiss is not None:
        sbert_vecs = embed_texts_sbert(texts)
        if sbert_vecs is not None:
            sbert_index = faiss.IndexFlatIP(sbert_vecs.shape[1])
            sbert_index.add(sbert_vecs)

    openai_index = None
    if faiss is not None:
        openai_vecs = embed_texts_openai(texts)
        openai_index = faiss.IndexFlatL2(openai_vecs.shape[1])
        openai_index.add(openai_vecs)

    bm25 = BM25Okapi([_bm25_tokenize(t) for t in texts])
    return sbert_index, openai_index, bm25
