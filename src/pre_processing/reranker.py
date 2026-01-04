from __future__ import annotations

import logging
import os
from typing import Any, List

try:
    import torch
except ImportError:
    torch = None

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from config.preprocessing import PreprocessingConfig

    HF_CROSS_MODEL_NAME = PreprocessingConfig.CROSS_ENCODER_MODEL
    TOP_N_FINAL = PreprocessingConfig.TOP_N_FINAL
    MODEL_CACHE_DIR = PreprocessingConfig.MODEL_CACHE_DIR
    USE_HF_CROSS_RERANK = PreprocessingConfig.USE_HF_CROSS_RERANK
    HF_CROSS_DEVICE = PreprocessingConfig.HF_CROSS_DEVICE
    HF_CROSS_BATCH = PreprocessingConfig.HF_CROSS_BATCH
except ImportError:
    HF_CROSS_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-large")
    TOP_N_FINAL = int(os.getenv("TOP_N_FINAL", "30"))
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/research_pipeline"))
    USE_HF_CROSS_RERANK = os.getenv("USE_HF_CROSS_RERANK", "1") not in {"0", "false", "False", "no", "No"}
    HF_CROSS_DEVICE = os.getenv("HF_CROSS_DEVICE", "cuda")
    HF_CROSS_BATCH = int(os.getenv("HF_CROSS_BATCH", "64"))

os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
os.environ["HF_HOME"] = MODEL_CACHE_DIR

_cross_model = None


def _ensure_cross_model():
    global _cross_model, USE_HF_CROSS_RERANK

    if not USE_HF_CROSS_RERANK:
        return None
    if _cross_model is not None:
        return _cross_model

    try:
        from sentence_transformers import CrossEncoder as _CE
    except Exception:
        LOGGER.warning("sentence-transformers not installed; cross-rerank disabled.")
        USE_HF_CROSS_RERANK = False
        return None

    cuda_disabled = os.getenv("CUDA_VISIBLE_DEVICES") == ""
    device = "cpu" if cuda_disabled else HF_CROSS_DEVICE

    if device == "cuda":
        if torch is None or not torch.cuda.is_available():
            device = "cpu"
        else:
            try:
                torch.cuda.current_device()
            except Exception:
                device = "cpu"

    LOGGER.info("Loading cross-encoder %s on %s (cache: %s)", HF_CROSS_MODEL_NAME, device, MODEL_CACHE_DIR)

    try:
        _cross_model = _CE(HF_CROSS_MODEL_NAME, max_length=512, device=device)
    except Exception as e:
        LOGGER.warning("Failed to load cross-encoder: %s", type(e).__name__)
        if device == "cuda":
            try:
                _cross_model = _CE(HF_CROSS_MODEL_NAME, max_length=512, device="cpu")
            except Exception:
                USE_HF_CROSS_RERANK = False
                _cross_model = None
        else:
            USE_HF_CROSS_RERANK = False
            _cross_model = None

    return _cross_model


def _rerank_crossencoder(query: str, cand_ids: List[int], snippets: List[Any]) -> List[int]:
    if not cand_ids:
        return []

    model = _ensure_cross_model()
    if model is None:
        return cand_ids[:TOP_N_FINAL]

    pairs = [[query, snippets[i].text] for i in cand_ids]
    scores = model.predict(pairs, batch_size=HF_CROSS_BATCH, show_progress_bar=False)
    ranked = sorted(zip(cand_ids, scores), key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in ranked[:TOP_N_FINAL]]
