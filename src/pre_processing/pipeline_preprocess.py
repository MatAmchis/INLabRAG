from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.preprocessing import PreprocessingConfig, set_seed
from .qa_extractor import (
    CITATION_OPTIONAL,
    QAResult,
    QUESTIONS,
    _gpt_answer,
    _gpt_answer_qualitative,
    _gpt_answer_quantitative,
    _gpt_reflect,
    _gpt_synthesise,
    _heuristic_pass,
    _is_missing,
    save_csv,
    save_json,
)
from .reranker import _rerank_crossencoder
from .retrieval import (
    _format_context,
    _format_context_qaware,
    _merge_neighbours,
    _qa_enrich_table_snippet,
    _search_hybrid,
    build_corpus_indexes,
    build_fallback_indexes,
)
from .snippet_builder import load_pdf_with_vlm, _first_page_title, _plausible

set_seed(PreprocessingConfig.SEED)

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def extract_answers_full(
    snippets: List[Any],
    full_body_text: str,
    metadata: dict | None = None,
    pdf_path: Optional[str] = None,
) -> Dict[str, QAResult]:
    from tqdm import tqdm

    LOGGER.info("Building corpus indexes with contextualization...")
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

    return results


def fallback_rerun_full(
    results: Dict[str, QAResult],
    snippets: List[Any],
    full_body_text: str,
) -> Dict[str, QAResult]:
    LOGGER.info("Building fallback indexes for verification...")
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
        parts = []
        for s in snips:
            txt = _qa_enrich_table_snippet(question, s) if s.kind == "table" else s.text
            parts.append(f"[p{s.page}] {txt.strip()[:800]}")
        return "\n\n".join(parts)

    for q, qa in results.items():
        if qa.status != "Needs_verification":
            continue

        LOGGER.info("Fallback rerun for: %s", q)

        cand_ids = _search_hybrid(q, sbert_index, openai_index, bm25)
        top_ids = _rerank_crossencoder(q, cand_ids, snippets)
        merged = _merge_neighbours(top_ids, snippets)
        _, pages = _format_context(merged)

        fallback_snips = get_snippets_by_pages(snippets, pages, max_snips=80)
        full_ctx = build_context(q, fallback_snips)
        draft = _gpt_answer(q, full_ctx, pages)

        if draft == "N/A" or _is_missing(draft):
            brute_force_ctx = build_context(q, snippets[:200])
            draft = _gpt_answer(q, brute_force_ctx, [s.page for s in snippets[:200]])

        is_missing_val = _is_missing(draft)
        results[q] = QAResult(
            "N/A" if is_missing_val else draft,
            "MISSING_IN_PAPER" if is_missing_val else "VERIFIED",
        )

    return results


def process_pdf_full(pdf_path: Path, out_dir: Path) -> Dict[str, QAResult]:
    LOGGER.info("Processing: %s", pdf_path.name)

    snippets, meta, full_body_text = load_pdf_with_vlm(str(pdf_path), client=None)
    LOGGER.info("Loaded %d snippets", len(snippets))

    results = extract_answers_full(snippets, full_body_text, metadata=meta, pdf_path=str(pdf_path))
    results = fallback_rerun_full(results, snippets, full_body_text)

    for _, qa in results.items():
        if _is_missing(qa.answer):
            qa.answer = "N/A"
            qa.status = "MISSING_IN_PAPER"

    stem = pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / f"{stem}_output.csv"
    json_out = out_dir / f"{stem}_output.json"

    save_csv(results, str(csv_out))
    save_json(results, str(json_out))

    LOGGER.info("Saved CSV: %s", csv_out)
    LOGGER.info("Saved JSON: %s", json_out)

    return results


def process_pdf_and_answer(
    pdf_path: str | Path,
    question: str,
    client,
) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    LOGGER.info("Processing PDF: %s", pdf_path)

    snippets, meta, full_text = load_pdf_with_vlm(str(pdf_path), client=client)
    LOGGER.info("Loaded %d snippets", len(snippets))

    sbert_index, openai_index, bm25, _ = build_corpus_indexes(snippets, full_text)
    retrieved_ids = _search_hybrid(question, sbert_index, openai_index, bm25)
    LOGGER.info("Hybrid retrieved %d candidates", len(retrieved_ids))

    final_ids = _rerank_crossencoder(question, retrieved_ids, snippets)
    LOGGER.info("Final candidate count: %d", len(final_ids))

    merged = _merge_neighbours(final_ids, snippets)
    context, pages = _format_context_qaware(question, merged)

    if question in CITATION_OPTIONAL:
        ans = _gpt_answer(question, context, pages)
    else:
        quant = _gpt_answer_quantitative(question, context)
        qual = _gpt_answer_qualitative(question, context)
        ans = _gpt_synthesise(question, quant, qual)

    return {
        "question": question,
        "answer": ans,
        "snippets_used": final_ids,
        "snippet_count": len(snippets),
        "metadata": meta,
        "full_text": full_text,
    }
