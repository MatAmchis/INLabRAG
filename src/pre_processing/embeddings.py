```python
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .env_setup import ensure_openai_api_key

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

LOGGER = logging.getLogger("pdf_vlm_qna")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from config.preprocessing import PreprocessingConfig

    SBERT_MODEL_NAME = PreprocessingConfig.SBERT_MODEL
    EMBED_MODEL_OPENAI = PreprocessingConfig.EMBED_MODEL
    MODEL_CACHE_DIR = PreprocessingConfig.MODEL_CACHE_DIR
    API_MAX_RETRIES = PreprocessingConfig.API_MAX_RETRIES
    API_INITIAL_BACKOFF = PreprocessingConfig.API_INITIAL_BACKOFF
    API_BACKOFF_MULT = PreprocessingConfig.API_BACKOFF_MULT
    API_BACKOFF_MAX = PreprocessingConfig.API_BACKOFF_MAX
    API_JITTER_FRAC = PreprocessingConfig.API_JITTER_FRAC
    SBERT_DEVICE = PreprocessingConfig.SBERT_DEVICE
    SBERT_BATCH_SIZE = PreprocessingConfig.SBERT_BATCH_SIZE
    OPENAI_EMBED_BATCH = PreprocessingConfig.OPENAI_EMBED_BATCH
except Exception:
    SBERT_MODEL_NAME = os.getenv("SBERT_MODEL", "intfloat/e5-large-v2")
    EMBED_MODEL_OPENAI = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/research_pipeline"))
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "6"))
    API_INITIAL_BACKOFF = float(os.getenv("API_INITIAL_BACKOFF", "1.0"))
    API_BACKOFF_MULT = float(os.getenv("API_BACKOFF_MULT", "2.0"))
    API_BACKOFF_MAX = float(os.getenv("API_BACKOFF_MAX", "30.0"))
    API_JITTER_FRAC = float(os.getenv("API_JITTER_FRAC", "0.25"))
    SBERT_DEVICE = os.getenv("SBERT_DEVICE", "cuda")
    SBERT_BATCH_SIZE = int(os.getenv("SBERT_BATCH_SIZE", "256"))
    OPENAI_EMBED_BATCH = int(os.getenv("OPENAI_EMBED_BATCH", "512"))

Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
os.environ["HF_HOME"] = MODEL_CACHE_DIR

_OPENAI_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        ensure_openai_api_key()
        _OPENAI_CLIENT = OpenAI(timeout=600)
    return _OPENAI_CLIENT


def _call_with_backoff(fn, *args, **kwargs):
    delay = API_INITIAL_BACKOFF
    last_err: Optional[Exception] = None

    for attempt in range(API_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate = ("RateLimit" in msg) or ("429" in msg)

            if (not is_rate) and attempt == API_MAX_RETRIES - 1:
                raise

            jitter = delay * random.uniform(1 - API_JITTER_FRAC, 1 + API_JITTER_FRAC)
            LOGGER.warning(
                "API error (%s); retrying in %.1fs (attempt %d/%d)",
                type(e).__name__,
                jitter,
                attempt + 1,
                API_MAX_RETRIES,
            )
            time.sleep(jitter)
            delay = min(delay * API_BACKOFF_MULT, API_BACKOFF_MAX)

    raise RuntimeError("API retries exhausted") from last_err


def _openai_embed_batch(batch: List[str]) -> np.ndarray:
    client = _get_client()
    resp = _call_with_backoff(client.embeddings.create, model=EMBED_MODEL_OPENAI, input=batch)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def embed_texts_openai(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), OPENAI_EMBED_BATCH):
        batch = texts[i : i + OPENAI_EMBED_BATCH]
        all_vecs.append(_openai_embed_batch(batch))

    vecs = np.vstack(all_vecs).astype(np.float32)
    return vecs


_sbert_model: Optional["SentenceTransformer"] = None
_sbert_warned = False


def _ensure_sbert() -> Optional["SentenceTransformer"]:
    global _sbert_model, _sbert_warned

    if SentenceTransformer is None:
        if not _sbert_warned:
            LOGGER.warning("sentence-transformers not installed; SBERT disabled.")
            _sbert_warned = True
        return None

    if _sbert_model is not None:
        return _sbert_model

    cuda_disabled = os.getenv("CUDA_VISIBLE_DEVICES") == ""
    device = "cpu" if cuda_disabled else SBERT_DEVICE

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
            else:
                try:
                    torch.cuda.current_device()
                except Exception:
                    device = "cpu"
        except Exception:
            device = "cpu"

    LOGGER.info("Loading SBERT (%s) on %s (cache: %s)", SBERT_MODEL_NAME, device, MODEL_CACHE_DIR)

    try:
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device, cache_folder=MODEL_CACHE_DIR)
    except Exception as e:
        LOGGER.warning("Failed to load SBERT on %s: %s", device, type(e).__name__)
        if device == "cuda":
            try:
                _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu", cache_folder=MODEL_CACHE_DIR)
            except Exception:
                _sbert_model = None
        else:
            _sbert_model = None

    return _sbert_model


def embed_texts_sbert(texts: List[str]) -> Optional[np.ndarray]:
    model = _ensure_sbert()
    if model is None:
        return None

    vecs = model.encode(
        texts,
        batch_size=SBERT_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32)

    if faiss is not None:
        try:
            faiss.normalize_L2(vecs)
        except Exception:
            pass

    return vecs
