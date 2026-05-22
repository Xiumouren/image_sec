from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceAssetRecord:
    id: int
    original_filename: str
    archived_path: str
    archived_relative_path: str
    archived_url: str
    sha256: str
    phash: str
    file_size: int
    created_at: str


@dataclass(frozen=True)
class SourceAssetEmbeddingRecord:
    asset_id: int
    model_name: str
    embedding_dim: int
    embedding_path: str
    created_at: str


@dataclass(frozen=True)
class SourceRoiAssetRecord:
    id: int
    asset_id: int
    target_label: str
    bbox_json: str
    coverage_ratio: float
    roi_path: str
    roi_relative_path: str
    roi_url: str
    roi_sha256: str
    roi_phash: str
    created_at: str


@dataclass(frozen=True)
class SourceRoiEmbeddingRecord:
    roi_id: int
    model_name: str
    embedding_dim: int
    embedding_path: str
    created_at: str


@dataclass(frozen=True)
class PersistSourceAnalysisResult:
    asset: SourceAssetRecord
    record_saved: bool
    asset_reused: bool
