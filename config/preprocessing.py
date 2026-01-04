import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def check_gpu_compatibility() -> bool:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False

    try:
        props = torch.cuda.get_device_properties(0)
        capability = props.major + props.minor / 10.0

        if capability < 7.0:
            print(f"GPU detected but incompatible (CUDA {capability}), using CPU")
            return False

        x = torch.zeros(1).cuda()
        _ = x + 1
        return True
    except Exception as e:
        print(f"GPU test failed: {type(e).__name__}, using CPU")
        return False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if not TORCH_AVAILABLE:
        return

    torch.manual_seed(seed)

    gpu_ok = check_gpu_compatibility()
    if gpu_ok:
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            gpu_ok = False

    if not gpu_ok:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreprocessingConfig:
    SEED = int(os.getenv("SEED", "42"))

    VLM_MODEL = os.getenv("VLM_MODEL", "gpt-5.2")
    CTX_MODEL = os.getenv("CTX_MODEL", "gpt-5.2")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    SBERT_MODEL = os.getenv("SBERT_MODEL", "intfloat/e5-large-v2")
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-large")

    SBERT_BATCH_SIZE = 256
    SBERT_DEVICE = "cuda"
    OPENAI_EMBED_BATCH = 512
    DUAL_DENSE = True

    USE_HF_CROSS_RERANK = True
    HF_CROSS_DEVICE = "cuda"
    HF_CROSS_BATCH = 64

    MODEL_CACHE_DIR = os.getenv(
        "MODEL_CACHE_DIR", os.path.expanduser("~/.cache/research_pipeline")
    )

    MAX_TOKENS = 300
    STRIDE = 60
    MERGE_LIMIT = 540

    TOP_K_SBERT = 200
    TOP_K_OPENAI = 200
    TOP_BM25 = 200
    TOP_N_FINAL = 30
    MAX_CTX_TOKENS = 120

    MAX_CAPTION_TOKENS = 120
    USE_VLM_FOR_TABLES = True
    SANITIZE_TABLE_CAPTIONS = True

    API_MAX_RETRIES = 6
    API_INITIAL_BACKOFF = 1.0
    API_BACKOFF_MULT = 2.0
    API_BACKOFF_MAX = 30.0
    API_JITTER_FRAC = 0.25

    NUMERIC_TSV_CHARS = 10000
    NUMERIC_TSV_ROWS = 100

    FLATTEN_TABLE_CELLS = True
    FLATTEN_TABLE_MAX_ROWS = 10
    FLATTEN_TABLE_MAX_COLS = 8
    FLATTEN_TABLE_MAX_CHARS = 600
    FLATTEN_TABLE_MAX_CELL_CHARS = 40

    Q_AWARE_TABLE_SUMMARIES = True
    TABLE_Q_SUM_MAX_CHARS_IN = 4000
    TABLE_Q_SUM_MAX_ROWS_OUT = 25
    TABLE_Q_SUM_LLM_TOKENS = 160

    USE_QUERY_EXPANSION = True
    MAX_EXPANSION_TERMS = 5

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


__all__ = ["PreprocessingConfig", "set_seed", "check_gpu_compatibility"]
