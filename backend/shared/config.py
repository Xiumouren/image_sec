from __future__ import annotations

import os
from pathlib import Path


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
SOURCE_ROI_DIR = OUTPUTS_DIR / "source_roi"
TEMP_DIR = BASE_DIR / ".tmp"
TFHUB_CACHE_DIR = TEMP_DIR / "tfhub_cache"
DATA_DIR = BASE_DIR / "data"
SQLITE_DIR = DATA_DIR / "sqlite"

DEFAULT_MODEL_PATH = BASE_DIR / "rebuild" / "nsfw_mobilenetv2_gradcam_ready.h5"
DEFAULT_LABELS_PATH = BASE_DIR / "mobilenet_v2_140_224" / "class_labels.txt"
DEFAULT_TEST_IMAGE_DIR = BASE_DIR / "test_images"
DEFAULT_SQLITE_PATH = Path(os.getenv("NET_SEC_SQLITE_PATH", str(SQLITE_DIR / "source_engine.db")))
SOURCE_ASSETS_ROOT = Path(os.getenv("NET_SEC_SOURCE_ASSETS_ROOT", r"E:\net_sec_assets"))
SOURCE_IMAGES_DIR = SOURCE_ASSETS_ROOT / "images"
SOURCE_ELA_DIR = SOURCE_ASSETS_ROOT / "ela"
SOURCE_ROI_ASSETS_DIR = SOURCE_ASSETS_ROOT / "roi"
SOURCE_ASSETS_URL_PREFIX = "/source-assets"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SINGLE_METHOD_DEFAULT = "gradcam"
BATCH_METHOD_DEFAULT = "saliency"
ALLOWED_METHODS = {"gradcam", "saliency", "integrated_gradients"}
PHASH_MATCH_THRESHOLD = 8
PHASH_TOP_K = 5
ELA_JPEG_QUALITY = 90
ELA_PIXEL_THRESHOLD = 24
ELA_TAMPER_RATIO_THRESHOLD = 5.0
SQLITE_TIMEOUT_SECONDS = 10.0
LLM_API_KEY = _env_first("NET_SEC_QWEN_API_KEY", "NET_SEC_LLM_API_KEY")
LLM_BASE_URL = _env_first(
    "NET_SEC_QWEN_BASE_URL",
    "NET_SEC_LLM_BASE_URL",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
LLM_MODEL = _env_first("NET_SEC_QWEN_MODEL", "NET_SEC_LLM_MODEL", default="qwen3-vl-plus")
LLM_TIMEOUT_SECONDS = float(
    _env_first("NET_SEC_QWEN_TIMEOUT_SECONDS", "NET_SEC_LLM_TIMEOUT_SECONDS", default="30")
)
LLM_ENABLE_THINKING = _env_flag("NET_SEC_QWEN_ENABLE_THINKING", False)
LLM_VL_HIGH_RESOLUTION_IMAGES = _env_flag("NET_SEC_QWEN_VL_HIGH_RESOLUTION_IMAGES", False)

SEMANTIC_EMBEDDING_API_KEY = _env_first("NET_SEC_QWEN_EMBEDDING_API_KEY", "NET_SEC_QWEN_API_KEY")
SEMANTIC_EMBEDDING_BASE_URL = _env_first(
    "NET_SEC_QWEN_EMBEDDING_BASE_URL",
    "NET_SEC_QWEN_BASE_URL",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
SEMANTIC_EMBEDDING_MODEL = os.getenv("NET_SEC_QWEN_EMBEDDING_MODEL", "qwen3-vl-embedding")
SEMANTIC_EMBEDDING_STORE_DIR = Path(
    os.getenv("NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR", r"E:\net_sec_data\semantic_embeddings")
)
LOCAL_CLIP_MODEL_NAME = os.getenv("NET_SEC_LOCAL_CLIP_MODEL_NAME", "ViT-B/32")
SEMANTIC_RETRIEVAL_TOP_K = int(os.getenv("NET_SEC_SEMANTIC_RETRIEVAL_TOP_K", "5"))
GRADCAM_ROI_HEATMAP_PERCENTILE = float(os.getenv("NET_SEC_GRADCAM_ROI_HEATMAP_PERCENTILE", "75"))
GRADCAM_ROI_MASK_THRESHOLD = float(os.getenv("NET_SEC_GRADCAM_ROI_MASK_THRESHOLD", "0.45"))
GRADCAM_ROI_FALLBACK_THRESHOLD = float(os.getenv("NET_SEC_GRADCAM_ROI_FALLBACK_THRESHOLD", "0.30"))
GRADCAM_ROI_MIN_COVERAGE_RATIO = float(os.getenv("NET_SEC_GRADCAM_ROI_MIN_COVERAGE_RATIO", "0.01"))
GRADCAM_ROI_EXPANSION_RATIO = float(os.getenv("NET_SEC_GRADCAM_ROI_EXPANSION_RATIO", "0.20"))
