from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
from uuid import uuid4

from backend.shared.config import (
    SOURCE_ASSETS_ROOT,
    SOURCE_ASSETS_URL_PREFIX,
    SOURCE_ELA_DIR,
    SOURCE_IMAGES_DIR,
    SOURCE_ROI_ASSETS_DIR,
)


@dataclass(frozen=True)
class ArchivedAsset:
    original_filename: str
    archived_path: Path
    archived_relative_path: str
    archived_url: str
    sha256: str
    file_size: int


@dataclass(frozen=True)
class ElaAsset:
    output_path: Path
    output_relative_path: str
    output_url: str


@dataclass(frozen=True)
class ArchivedRoiAsset:
    roi_path: Path
    roi_relative_path: str
    roi_url: str
    roi_sha256: str


class AssetManager:
    def __init__(
        self,
        source_root: Path = SOURCE_ASSETS_ROOT,
        images_dir: Path = SOURCE_IMAGES_DIR,
        ela_dir: Path = SOURCE_ELA_DIR,
        roi_dir: Path = SOURCE_ROI_ASSETS_DIR,
        source_url_prefix: str = SOURCE_ASSETS_URL_PREFIX,
    ) -> None:
        self.source_root = Path(source_root)
        self.images_dir = Path(images_dir)
        self.ela_dir = Path(ela_dir)
        self.roi_dir = Path(roi_dir)
        self.source_url_prefix = source_url_prefix.rstrip("/")

    def ensure_directories(self) -> None:
        for path in (self.source_root, self.images_dir, self.ela_dir, self.roi_dir):
            path.mkdir(parents=True, exist_ok=True)

    def archive_image(self, image_path: Path) -> ArchivedAsset:
        self.ensure_directories()
        sha256 = self.compute_sha256(image_path)
        timestamp = datetime.now()
        target_dir = self.images_dir / timestamp.strftime("%Y%m%d")
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(image_path.name).name
        archived_name = f"{timestamp.strftime('%H%M%S')}_{uuid4().hex[:8]}_{safe_name}"
        archived_path = target_dir / archived_name
        shutil.copy2(image_path, archived_path)

        return ArchivedAsset(
            original_filename=safe_name,
            archived_path=archived_path,
            archived_relative_path=self.relative_to_source_root(archived_path),
            archived_url=self.build_source_url(archived_path),
            sha256=sha256,
            file_size=archived_path.stat().st_size,
        )

    def record_to_archived_asset(self, record) -> ArchivedAsset:
        archived_path = Path(record.archived_path)
        return ArchivedAsset(
            original_filename=record.original_filename,
            archived_path=archived_path,
            archived_relative_path=record.archived_relative_path,
            archived_url=record.archived_url,
            sha256=record.sha256,
            file_size=record.file_size,
        )

    def build_ela_asset(self, image_name: str) -> ElaAsset:
        self.ensure_directories()
        timestamp = datetime.now()
        target_dir = self.ela_dir / timestamp.strftime("%Y%m%d")
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(image_name).stem
        output_name = f"{timestamp.strftime('%H%M%S')}_{uuid4().hex[:8]}_{stem}_ela.jpg"
        output_path = target_dir / output_name
        return ElaAsset(
            output_path=output_path,
            output_relative_path=self.relative_to_source_root(output_path),
            output_url=self.build_source_url(output_path),
        )

    def archive_roi_crop(self, *, crop_path: Path, asset_id: int, target_label: str) -> ArchivedRoiAsset:
        self.ensure_directories()
        timestamp = datetime.now()
        target_dir = self.roi_dir / timestamp.strftime("%Y%m%d")
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in target_label)
        safe_label = safe_label or "roi"
        suffix = Path(crop_path).suffix or ".jpg"
        archived_name = f"{timestamp.strftime('%H%M%S')}_asset_{asset_id}_{safe_label}_{uuid4().hex[:8]}{suffix}"
        archived_path = target_dir / archived_name
        shutil.copy2(crop_path, archived_path)

        return ArchivedRoiAsset(
            roi_path=archived_path,
            roi_relative_path=self.relative_to_source_root(archived_path),
            roi_url=self.build_source_url(archived_path),
            roi_sha256=self.compute_sha256(archived_path),
        )

    def relative_to_source_root(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.source_root.resolve()).as_posix()

    def build_source_url(self, path: Path) -> str:
        return f"{self.source_url_prefix}/{self.relative_to_source_root(path)}"

    @staticmethod
    def delete_file(path: Path | str) -> None:
        target = Path(path)
        if target.exists() and target.is_file():
            target.unlink()

    @staticmethod
    def compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
