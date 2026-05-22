from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.shared.config import LOCAL_CLIP_MODEL_NAME


class ClipEmbeddingService:
    def __init__(self, model_name: str = LOCAL_CLIP_MODEL_NAME, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._preprocess: Any | None = None

    @property
    def storage_model_name(self) -> str:
        return f"clip:{self.model_name}"

    def embed_image(self, image_path: Path) -> np.ndarray:
        self._ensure_model_loaded()
        torch = self._import_torch()
        with Image.open(image_path) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(image_tensor)
        embedding = features.detach().cpu().numpy().astype("float32")[0]
        return self._normalize(embedding)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._preprocess is not None:
            return
        clip = self._import_clip()
        model, preprocess = clip.load(self.model_name, device=self.device)
        model.eval()
        self._model = model
        self._preprocess = preprocess

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype="float32")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("CLIP embedding has zero norm.")
        return (vector / norm).astype("float32")

    @staticmethod
    def _import_clip() -> Any:
        try:
            import clip
        except Exception as exc:
            raise RuntimeError(f"Unable to import local clip package: {exc}") from exc
        return clip

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"Unable to import torch package: {exc}") from exc
        return torch
