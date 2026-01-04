from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .env_setup import ensure_openai_api_key
from .qa_examples import FEW_SHOT_EXAMPLES
from .qa_prompts import QUESTION_PROMPT_OVERRIDES
from .qa_reasoning import REASONING_STEPS

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from config.preprocessing import PreprocessingConfig

    CTX_MODEL = PreprocessingConfig.CTX_MODEL
    API_MAX_RETRIES = PreprocessingConfig.API_MAX_RETRIES
    API_INITIAL_BACKOFF = PreprocessingConfig.API_INITIAL_BACKOFF
    API_BACKOFF_MULT = PreprocessingConfig.API_BACKOFF_MULT
    API_BACKOFF_MAX = PreprocessingConfig.API_BACKOFF_MAX
    API_JITTER_FRAC = PreprocessingConfig.API_JITTER_FRAC
except ImportError:
    CTX_MODEL = os.getenv("CTX_MODEL", "gpt-5.2")
    API_MAX_RETRIES = 6
    API_INITIAL_BACKOFF = 1.0
    API_BACKOFF_MULT = 2.0
    API_BACKOFF_MAX = 30.0
    API_JITTER_FRAC = 0.25

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
CITATION_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)
CITATION_OPTIONAL = {"First author", "Year published", "Title", "Language of publication"}

MAX_PROMPT_TOKENS = 16000
SLEEP_PER_Q_SUCCESS = 0.0
SLEEP_PER_Q_FALLBACK = 0.0

QUESTIONS = list(QUESTION_PROMPT_OVERRIDES.keys())


@dataclass
class QAResult:
    answer: str
    status: str


def _is_numeric_q(q: str) -> bool:
    return bool(NUMERIC_Q_PAT.search(q))


def _normalize_missing(text: str) -> str:
    if not text:
        return ""
    text = CITATION_RE.sub("", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


_MISSING_CANON = {
    _normalize_missing(v)
    for v in (
        "n/a",
        "na",
        "not reported",
        "not available",
        "not found",
        "missing",
        "",
    )
}


def _is_missing(text: str) -> bool:
    norm = _normalize_missing(text)
    if norm in _MISSING_CANON:
        return True
    return (
        "not reported" in norm
        or "notavailable" in norm
        or "notfound" in norm
        or "no data" in norm
        or norm == ""
    )


def _clean_na_prefix(text: str) -> str:
    if not text:
        return text

    text = text.strip()
    na_prefix_patterns = [
        r"^[Nn][/\\]?[Aa][\s.,;:\-–—]+(?:however|but|though|although|yet|still|nonetheless|the|a|an|this|that|it|there|in|on|for|from|with|community|study|paper|data|results?|findings?)",
        r"^[Nn][/\\]?[Aa][\s.,;:\-–—]+[A-Z]",
    ]

    for pattern in na_prefix_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            clean_start = re.match(r"^[\s.,;:\-–—]*", text[3:])
            if clean_start:
                actual_content = text[3 + clean_start.end() :].strip()
                if actual_content and len(actual_content) > 10:
                    if actual_content[0].islower():
                        actual_content = actual_content[0].upper() + actual_content[1:]
                    return actual_content

    return text


def _build_prompt(question: str, context: str) -> str:
    instr = QUESTION_PROMPT_OVERRIDES.get(question, "")
    examples = FEW_SHOT_EXAMPLES.get(question, [])
    reasoning = REASONING_STEPS.get(question, [])

    few_shot_block = ""
    if examples:
        few_shot_block = "\n\nExamples:\n" + "\n".join(f"- {ex}" for ex in examples)

    reasoning_block = ""
    if reasoning:
        if isinstance(reasoning, (list, tuple)):
            reasoning_block = "\n\nReasoning steps:\n" + "\n".join(f"- {r}" for r in reasoning)
        else:
            reasoning_block = f"\n\nReasoning steps:\n{reasoning}"

    prompt = (
        f"Question: {question}\n\n"
        f"{instr}{few_shot_block}{reasoning_block}\n\n"
        f"PDF context:\n{context}"
    )

    if any(tag in context for tag in ("[TABLE", "[FIG", "[IMAGE", "[ABSTRACT_IMAGE", "[AUTHORS_IMAGE")):
        prompt += (
            "\n\n(Note: Snippets tagged [TABLE pX], [FIG pX], etc. are vision-model captions "
            "of non-text PDF regions. Use them as evidence; cite the page number(s).)"
        )

    if len(prompt.split()) > MAX_PROMPT_TOKENS:
        prompt = "...\n" + " ".join(prompt.split()[-MAX_PROMPT_TOKENS:])
    return prompt


def _gpt_answer(question: str, context: str, retrieved_pages: Optional[List[int]] = None) -> str:
    ensure_openai_api_key()

    from openai import APITimeoutError, OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()

    try:
        resp = _call_with_backoff(
            client.chat.completions.create,
            model=CTX_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a global-health research assistant extracting structured data "
                        "from a scientific paper. Follow the user prompt exactly. "
                        "If the information is not present, answer 'N/A'. Cite pages like [p5]. "
                        "Do NOT start your answer with 'N/A' if you have actual content to provide. "
                        "Be concise: report only what is in the paper. "
                        "Do NOT write 'the article does not describe...' — just report what's there. "
                        "If truly nothing relevant exists, answer exactly 'N/A'."
                    ),
                },
                {"role": "user", "content": _build_prompt(question, context)},
            ],
            temperature=0,
        )
        ans = resp.choices[0].message.content.strip()
        return _clean_na_prefix(ans)
    except APITimeoutError:
        return "N/A"
    except Exception as e:
        LOGGER.warning("_gpt_answer error: %s", e)
        return "N/A"


def _gpt_answer_quantitative(question: str, context: str) -> str:
    ensure_openai_api_key()

    from openai import OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()

    ans = _call_with_backoff(
        client.chat.completions.create,
        model=CTX_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quantitative extraction specialist. "
                    "Respond with numeric indicators, ratios, denominators, cost figures or percentages "
                    "that directly answer the question. If no numbers exist, answer 'N/A'. "
                    "Cite pages like [p7]. Be concise: report only what is in the paper. "
                    "Do NOT start with 'N/A' if you have actual numbers to report."
                ),
            },
            {"role": "user", "content": _build_prompt(question, context)},
        ],
    ).choices[0].message.content.strip()
    return _clean_na_prefix(ans)


def _gpt_answer_qualitative(question: str, context: str) -> str:
    ensure_openai_api_key()

    from openai import OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()

    ans = _call_with_backoff(
        client.chat.completions.create,
        model=CTX_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a qualitative extraction specialist. "
                    "Provide concise thematic or descriptive answers. Cite pages like [p7]. "
                    "If no qualitative info exists, answer 'N/A'. "
                    "Be concise: report only what is in the paper. "
                    "Do NOT start with 'N/A' if you have actual content to provide."
                ),
            },
            {"role": "user", "content": _build_prompt(question, context)},
        ],
    ).choices[0].message.content.strip()
    return _clean_na_prefix(ans)


def _gpt_synthesise(question: str, quant_ans: str, qual_ans: str) -> str:
    ensure_openai_api_key()

    from openai import OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()

    ans = _call_with_backoff(
        client.chat.completions.create,
        model=CTX_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the synthesiser.\n"
                    f"Question: {question}\n"
                    f"Answer A (quantitative specialist): {quant_ans}\n"
                    f"Answer B (qualitative specialist): {qual_ans}\n\n"
                    "Task:\n"
                    "1) Read both answers.\n"
                    "2) Produce ONE final answer most faithful to the paper.\n"
                    "3) Keep every correct number/ratio/statistic from either answer.\n"
                    "4) Keep non-numeric insights that add meaning.\n"
                    "5) If both answers are 'N/A', output exactly: N/A.\n"
                    "6) Remove duplicates. If statements conflict, prefer the more precise/better-cited one.\n"
                    "7) Preserve existing page citations. Do NOT invent new ones.\n"
                    "8) Do NOT start with 'N/A' if you have actual content."
                ),
            }
        ],
    ).choices[0].message.content.strip()
    return _clean_na_prefix(ans)


def _gpt_reflect(question: str, draft: str, context: str, valid_pages: List[int]) -> str:
    ensure_openai_api_key()

    from openai import APITimeoutError, OpenAI
    from .embeddings import _call_with_backoff

    client = OpenAI()

    try:
        resp = _call_with_backoff(
            client.chat.completions.create,
            model=CTX_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are verifying an extracted answer. "
                        f"Approve only if it contains at least one valid page citation from {sorted(set(valid_pages))} "
                        "and matches the snippets. If it does not, respond with N/A."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nDraft answer: {draft}\n\nPDF context:\n{context}",
                },
            ],
            temperature=0,
        )
        decision = resp.choices[0].message.content.strip().upper()
        return draft if decision != "N/A" else "N/A"
    except APITimeoutError:
        return "N/A"
    except Exception as e:
        LOGGER.warning("_gpt_reflect error: %s", e)
        return "N/A"


def _heuristic_pass(draft: str, valid_pages: List[int], question: str) -> bool:
    if _is_missing(draft):
        return False
    if question in CITATION_OPTIONAL:
        return True
    cites = [int(p) for p in CITATION_RE.findall(draft)]
    return bool(cites) and all(c in valid_pages for c in cites)


def extract_answers(
    snippets: List[Any],
    full_body_text: str,
    metadata: dict | None = None,
    pdf_path: Optional[str] = None,
) -> Dict[str, QAResult]:
    import time

    from tqdm import tqdm
    from .reranker import _rerank_crossencoder
    from .retrieval import _format_context_qaware, _merge_neighbours, _search_hybrid, build_corpus_indexes
    from .snippet_builder import _first_page_title, _plausible

    sbert_index, openai_index, bm25_index, _ = build_corpus_indexes(snippets, full_body_text)
    results: Dict[str, QAResult] = {}

    for q in tqdm(QUESTIONS, desc="Questions", unit="Q"):
        if q == "Title":
            meta_title = (metadata or {}).get("title", "").strip()
            page_title = _first_page_title(str(pdf_path)) if pdf_path else ""
            meta_ok, page_ok = _plausible(meta_title), _plausible(page_title)
            chosen = (
                meta_title
                if meta_ok and not page_ok
                else page_title
                if page_ok and not meta_ok
                else page_title
                if page_ok and len(page_title) > len(meta_title)
                else meta_title
            )
            if _plausible(chosen):
                results[q] = QAResult(answer=f"{chosen} [auto]", status="VERIFIED")
                continue

        cand_ids = _search_hybrid(q, sbert_index, openai_index, bm25_index)
        if not cand_ids:
            results[q] = QAResult("N/A", "MISSING_IN_PAPER")
            continue

        top_ids = _rerank_crossencoder(q, cand_ids, snippets)
        merged_snips = _merge_neighbours(top_ids, snippets)
        ctx, pages = _format_context_qaware(q, merged_snips)

        if q in CITATION_OPTIONAL:
            draft = _gpt_answer(q, ctx, pages)
        else:
            quant_ans = _gpt_answer_quantitative(q, ctx)
            qual_ans = _gpt_answer_qualitative(q, ctx)
            draft = _gpt_synthesise(q, quant_ans, qual_ans)

        if not _heuristic_pass(draft, pages, q):
            results[q] = QAResult("N/A", "Needs_verification")
            continue

        final = _gpt_reflect(q, draft, ctx, pages)
        is_missing_val = _is_missing(final)

        results[q] = QAResult(
            "N/A" if is_missing_val else final,
            "MISSING_IN_PAPER" if is_missing_val else "VERIFIED",
        )

        if SLEEP_PER_Q_SUCCESS:
            time.sleep(SLEEP_PER_Q_SUCCESS)

    return results


def fallback_rerun(
    results: Dict[str, QAResult],
    snippets: List[Any],
    full_body_text: str,
) -> Dict[str, QAResult]:
    import time

    from .reranker import _rerank_crossencoder
    from .retrieval import _format_context, _merge_neighbours, _search_hybrid, build_fallback_indexes

    sbert_index, openai_index, bm25 = build_fallback_indexes(snippets)

    def get_snippets_by_pages(snips: List[Any], pages: List[int], max_snips: int = 80) -> List[Any]:
        seen = set()
        selected = []
        for s in snips:
            if s.page in pages and (s.page, s.para_idx) not in seen:
                selected.append(s)
                seen.add((s.page, s.para_idx))
            if len(selected) >= max_snips:
                break
        return selected

    def build_context(question: str, snips: List[Any]) -> str:
        from .retrieval import _qa_enrich_table_snippet

        parts = []
        for s in snips:
            txt = _qa_enrich_table_snippet(question, s) if s.kind == "table" else s.text
            parts.append(f"[p{s.page}] {txt.strip()[:800]}")
        return "\n\n".join(parts)

    for q, qa in results.items():
        if qa.status != "Needs_verification":
            continue

        cand_ids = _search_hybrid(q, sbert_index, openai_index, bm25)
        top_ids = _rerank_crossencoder(q, cand_ids, snippets)
        merged = _merge_neighbours(top_ids, snippets)
        _, pages = _format_context(merged)

        fallback_snips = get_snippets_by_pages(snippets, pages, max_snips=80)
        full_ctx = build_context(q, fallback_snips)
        draft = _gpt_answer(q, full_ctx, pages)

        if draft == "N/A":
            brute_force_ctx = build_context(q, snippets[:200])
            draft = _gpt_answer(q, brute_force_ctx, [s.page for s in snippets[:200]])

        is_missing_val = _is_missing(draft)
        results[q] = QAResult(
            "N/A" if is_missing_val else draft,
            "MISSING_IN_PAPER" if is_missing_val else "VERIFIED",
        )

        if SLEEP_PER_Q_FALLBACK:
            time.sleep(SLEEP_PER_Q_FALLBACK)

    return results


def save_csv(mapping: Dict[str, QAResult], path: str) -> None:
    import csv

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Answer", "Status"])
        for k, v in mapping.items():
            w.writerow([k, v.answer, v.status])


def save_json(mapping: Dict[str, QAResult], path: str) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v.__dict__ for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
