from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.shared.config import SEMANTIC_EMBEDDING_STORE_DIR, SEMANTIC_RETRIEVAL_TOP_K
from backend.source_engine.clip_feature import ClipEmbeddingService
from backend.storage_layer.sqlite.models import SourceAssetRecord, SourceRoiAssetRecord
from backend.storage_layer.sqlite.repository import SourceRepository


class SemanticRetrievalService:
    def __init__(
        self,
        repository: SourceRepository | None = None,
        embedding_service: ClipEmbeddingService | None = None,
        embedding_store_dir: Path = SEMANTIC_EMBEDDING_STORE_DIR,
        top_k: int = SEMANTIC_RETRIEVAL_TOP_K,
    ) -> None:
        self.repository = repository or SourceRepository()
        self.embedding_service = embedding_service or ClipEmbeddingService()
        self.embedding_store_dir = Path(embedding_store_dir)
        self.top_k = top_k
        self._index = None
        self._asset_by_row: list[int] = []
        self._assets_by_id: dict[int, SourceAssetRecord] = {}
        self._index_dim: int | None = None
        self._built = False

    @property
    def model_name(self) -> str:
        return self.embedding_service.storage_model_name

    def search(self, image_path: Path) -> dict[str, object]:
        query_embedding = self.embedding_service.embed_image(image_path)
        self._ensure_index()
        if self._index is None or not self._asset_by_row:
            return {
                "model_name": self.model_name,
                "embedding_dim": int(query_embedding.shape[0]),
                "indexed_asset_count": 0,
                "candidates": [],
                "error": None,
            }
        distances, indices = self._index.search(query_embedding.reshape(1, -1), self.top_k)
        candidates = []
        for score, row_index in zip(distances[0], indices[0]):
            if int(row_index) < 0:
                continue
            asset_id = self._asset_by_row[int(row_index)]
            asset = self._assets_by_id.get(asset_id)
            if asset is None:
                continue
            candidates.append(
                {
                    "asset_id": asset.id,
                    "image_name": asset.original_filename,
                    "relative_path": asset.archived_relative_path,
                    "source_url": asset.archived_url,
                    "semantic_similarity": round(float(score), 4),
                    "model_name": self.model_name,
                }
            )
        return {
            "model_name": self.model_name,
            "embedding_dim": int(query_embedding.shape[0]),
            "indexed_asset_count": len(self._asset_by_row),
            "candidates": candidates,
            "error": None,
        }

    def add_or_update_asset(self, asset: SourceAssetRecord, source_image_path: Path | None = None) -> dict[str, object]:
        embedding = self._load_or_create_embedding(asset, source_image_path or Path(asset.archived_path))
        self._ensure_index()
        if asset.id in self._asset_by_row:
            self._assets_by_id = {**self._assets_by_id, asset.id: asset}
            return {
                "asset_id": asset.id,
                "model_name": self.model_name,
                "embedding_dim": int(embedding.shape[0]),
                "indexed": True,
                "error": None,
            }
        if self._index is None:
            self._reset_index(int(embedding.shape[0]))
        self._index.add(embedding.reshape(1, -1))
        self._asset_by_row.append(asset.id)
        self._assets_by_id = {**self._assets_by_id, asset.id: asset}
        return {
            "asset_id": asset.id,
            "model_name": self.model_name,
            "embedding_dim": int(embedding.shape[0]),
            "indexed": True,
            "error": None,
        }

    def rebuild_index(self) -> dict[str, object]:
        assets = self.repository.list_assets()
        self._index = None
        self._asset_by_row = []
        self._assets_by_id = {}
        self._index_dim = None
        errors = []
        for asset in assets:
            try:
                embedding = self._load_or_create_embedding(asset, Path(asset.archived_path))
                if self._index is None:
                    self._reset_index(int(embedding.shape[0]))
                if int(embedding.shape[0]) != self._index_dim:
                    errors.append(f"Skipped asset {asset.id}: embedding dimension mismatch.")
                    continue
                self._index.add(embedding.reshape(1, -1))
                self._asset_by_row.append(asset.id)
                self._assets_by_id[asset.id] = asset
            except Exception as exc:
                errors.append(f"Skipped asset {asset.id}: {exc}")
        self._built = True
        return {
            "model_name": self.model_name,
            "indexed_asset_count": len(self._asset_by_row),
            "errors": errors,
        }

    def _ensure_index(self) -> None:
        if not self._built:
            self.rebuild_index()

    def _reset_index(self, embedding_dim: int) -> None:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(f"Unable to import faiss package: {exc}") from exc
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._index_dim = embedding_dim

    def _load_or_create_embedding(self, asset: SourceAssetRecord, image_path: Path) -> np.ndarray:
        embedding_record = self.repository.find_embedding(asset.id, self.model_name)
        if embedding_record:
            embedding_path = Path(embedding_record.embedding_path)
            if embedding_path.exists():
                embedding = np.load(embedding_path).astype("float32")
                return self._normalize(embedding)

        embedding = self.embedding_service.embed_image(image_path)
        embedding_path = self._embedding_path(asset.id)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embedding)
        self.repository.upsert_embedding(
            asset_id=asset.id,
            model_name=self.model_name,
            embedding_dim=int(embedding.shape[0]),
            embedding_path=str(embedding_path),
        )
        return embedding

    def _embedding_path(self, asset_id: int) -> Path:
        safe_model_name = self.model_name.replace(":", "_").replace("/", "_")
        return self.embedding_store_dir / safe_model_name / f"asset_{asset_id}.npy"

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype="float32")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Stored embedding has zero norm.")
        return (vector / norm).astype("float32")


class RoiSemanticRetrievalService:
    def __init__(
        self,
        repository: SourceRepository | None = None,
        embedding_service: ClipEmbeddingService | None = None,
        embedding_store_dir: Path = SEMANTIC_EMBEDDING_STORE_DIR,
        top_k: int = SEMANTIC_RETRIEVAL_TOP_K,
    ) -> None:
        self.repository = repository or SourceRepository()
        self.embedding_service = embedding_service or ClipEmbeddingService()
        self.embedding_store_dir = Path(embedding_store_dir)
        self.top_k = top_k
        self._index = None
        self._roi_by_row: list[int] = []
        self._roi_by_id: dict[int, SourceRoiAssetRecord] = {}
        self._assets_by_id: dict[int, SourceAssetRecord] = {}
        self._index_dim: int | None = None
        self._built = False

    @property
    def model_name(self) -> str:
        return self.embedding_service.storage_model_name

    def search(self, roi_image_path: Path, query_phash: str | None = None) -> dict[str, object]:
        query_embedding = self.embedding_service.embed_image(roi_image_path)
        self._ensure_index()
        if self._index is None or not self._roi_by_row:
            return {
                "model_name": self.model_name,
                "embedding_dim": int(query_embedding.shape[0]),
                "indexed_roi_count": 0,
                "candidates": [],
                "error": None,
            }

        distances, indices = self._index.search(query_embedding.reshape(1, -1), self.top_k)
        candidates = []
        for score, row_index in zip(distances[0], indices[0]):
            if int(row_index) < 0:
                continue
            roi_id = self._roi_by_row[int(row_index)]
            roi = self._roi_by_id.get(roi_id)
            if roi is None:
                continue
            asset = self._assets_by_id.get(roi.asset_id)
            phash_distance = self._phash_distance(query_phash, roi.roi_phash) if query_phash else None
            candidates.append(
                {
                    "roi_id": roi.id,
                    "asset_id": roi.asset_id,
                    "image_name": asset.original_filename if asset else "",
                    "relative_path": asset.archived_relative_path if asset else "",
                    "source_url": asset.archived_url if asset else "",
                    "roi_url": roi.roi_url,
                    "roi_relative_path": roi.roi_relative_path,
                    "source_bbox": self._safe_bbox(roi.bbox_json),
                    "target_label": roi.target_label,
                    "roi_semantic_similarity": round(float(score), 4),
                    "roi_phash_distance": phash_distance,
                    "model_name": self.model_name,
                }
            )
        return {
            "model_name": self.model_name,
            "embedding_dim": int(query_embedding.shape[0]),
            "indexed_roi_count": len(self._roi_by_row),
            "candidates": candidates,
            "error": None,
        }

    def add_or_update_roi(self, roi: SourceRoiAssetRecord, roi_image_path: Path | None = None) -> dict[str, object]:
        embedding = self._load_or_create_embedding(roi, roi_image_path or Path(roi.roi_path))
        self._ensure_index()
        if roi.id in self._roi_by_row:
            self._roi_by_id = {**self._roi_by_id, roi.id: roi}
            return {
                "roi_id": roi.id,
                "model_name": self.model_name,
                "embedding_dim": int(embedding.shape[0]),
                "indexed": True,
                "error": None,
            }
        if self._index is None:
            self._reset_index(int(embedding.shape[0]))
        self._index.add(embedding.reshape(1, -1))
        self._roi_by_row.append(roi.id)
        self._roi_by_id = {**self._roi_by_id, roi.id: roi}
        self._assets_by_id = {asset.id: asset for asset in self.repository.list_assets()}
        return {
            "roi_id": roi.id,
            "model_name": self.model_name,
            "embedding_dim": int(embedding.shape[0]),
            "indexed": True,
            "error": None,
        }

    def rebuild_index(self) -> dict[str, object]:
        roi_assets = self.repository.list_roi_assets()
        self._index = None
        self._roi_by_row = []
        self._roi_by_id = {}
        self._assets_by_id = {asset.id: asset for asset in self.repository.list_assets()}
        self._index_dim = None
        errors = []
        for roi in roi_assets:
            try:
                embedding = self._load_or_create_embedding(roi, Path(roi.roi_path))
                if self._index is None:
                    self._reset_index(int(embedding.shape[0]))
                if int(embedding.shape[0]) != self._index_dim:
                    errors.append(f"Skipped ROI {roi.id}: embedding dimension mismatch.")
                    continue
                self._index.add(embedding.reshape(1, -1))
                self._roi_by_row.append(roi.id)
                self._roi_by_id[roi.id] = roi
            except Exception as exc:
                errors.append(f"Skipped ROI {roi.id}: {exc}")
        self._built = True
        return {
            "model_name": self.model_name,
            "indexed_roi_count": len(self._roi_by_row),
            "errors": errors,
        }

    def _ensure_index(self) -> None:
        if not self._built:
            self.rebuild_index()

    def _reset_index(self, embedding_dim: int) -> None:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(f"Unable to import faiss package: {exc}") from exc
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._index_dim = embedding_dim

    def _load_or_create_embedding(self, roi: SourceRoiAssetRecord, image_path: Path) -> np.ndarray:
        embedding_record = self.repository.find_roi_embedding(roi.id, self.model_name)
        if embedding_record:
            embedding_path = Path(embedding_record.embedding_path)
            if embedding_path.exists():
                embedding = np.load(embedding_path).astype("float32")
                return SemanticRetrievalService._normalize(embedding)

        embedding = self.embedding_service.embed_image(image_path)
        embedding_path = self._embedding_path(roi.id)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embedding)
        self.repository.upsert_roi_embedding(
            roi_id=roi.id,
            model_name=self.model_name,
            embedding_dim=int(embedding.shape[0]),
            embedding_path=str(embedding_path),
        )
        return embedding

    def _embedding_path(self, roi_id: int) -> Path:
        safe_model_name = self.model_name.replace(":", "_").replace("/", "_")
        return self.embedding_store_dir / safe_model_name / f"roi_{roi_id}.npy"

    @staticmethod
    def _safe_bbox(bbox_json: str) -> dict[str, object] | None:
        try:
            import json

            value = json.loads(bbox_json)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    @staticmethod
    def _phash_distance(left: str | None, right: str | None) -> int | None:
        if not left or not right:
            return None
        try:
            import imagehash

            return int(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right))
        except Exception:
            return None
