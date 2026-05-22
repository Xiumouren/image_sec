from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_ROOT = Path(r"E:\nsfw_data_source_urls-master\downloaded_images")
DEFAULT_SOURCE_ASSETS_ROOT = Path(r"E:\net_sec_assets")
DEFAULT_SQLITE_PATH = Path(r"E:\net_sec_data\source_engine.db")
DEFAULT_EMBEDDING_STORE_DIR = Path(r"E:\net_sec_data\semantic_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import local image folders into the source-analysis asset archive and CLIP embedding store."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-assets-root", type=Path, default=DEFAULT_SOURCE_ASSETS_ROOT)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--embedding-store-dir", type=Path, default=DEFAULT_EMBEDDING_STORE_DIR)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["NET_SEC_SOURCE_ASSETS_ROOT"] = str(args.source_assets_root)
    os.environ["NET_SEC_SQLITE_PATH"] = str(args.sqlite_path)
    os.environ["NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR"] = str(args.embedding_store_dir)
    os.environ["NET_SEC_LOCAL_CLIP_MODEL_NAME"] = args.clip_model
    os.environ["NET_SEC_SEMANTIC_RETRIEVAL_TOP_K"] = str(args.top_k)


def iter_image_paths(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def category_for(path: Path, input_root: Path) -> str:
    try:
        relative_parts = path.relative_to(input_root).parts
    except ValueError:
        return "_unknown"
    return relative_parts[0] if len(relative_parts) > 1 else "_root"


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def make_empty_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "source_assets_root": str(args.source_assets_root),
        "sqlite_path": str(args.sqlite_path),
        "embedding_store_dir": str(args.embedding_store_dir),
        "clip_model": args.clip_model,
        "dry_run": bool(args.dry_run),
        "limit": args.limit,
        "summary": {
            "scanned": 0,
            "processed": 0,
            "created_assets": 0,
            "duplicates": 0,
            "embedding_available": 0,
            "embedding_created_or_verified": 0,
            "failed": 0,
            "skipped_non_images": 0,
        },
        "by_category": {},
        "failures": [],
    }


def increment_category(report: dict[str, Any], category: str, key: str) -> None:
    by_category = report["by_category"]
    if category not in by_category:
        by_category[category] = {
            "processed": 0,
            "created_assets": 0,
            "duplicates": 0,
            "embedding_available": 0,
            "failed": 0,
        }
    by_category[category][key] += 1


def write_report(report: dict[str, Any]) -> Path:
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"import_source_dataset_{timestamp}.json"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def run_dry_scan(args: argparse.Namespace, report: dict[str, Any]) -> None:
    image_paths = iter_image_paths(args.input_root)
    selected_paths = image_paths[: args.limit] if args.limit else image_paths
    report["summary"]["scanned"] = len(image_paths)
    report["summary"]["skipped_non_images"] = count_non_images(args.input_root)

    for image_path in selected_paths:
        category = category_for(image_path, args.input_root)
        try:
            verify_image(image_path)
            report["summary"]["processed"] += 1
            increment_category(report, category, "processed")
        except Exception as exc:
            report["summary"]["failed"] += 1
            increment_category(report, category, "failed")
            report["failures"].append(
                {
                    "path": str(image_path),
                    "category": category,
                    "error": f"Image verification failed: {exc}",
                }
            )


def count_non_images(input_root: Path) -> int:
    if not input_root.exists():
        return 0
    return sum(
        1
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() not in SUPPORTED_EXTENSIONS
    )


def run_import(args: argparse.Namespace, report: dict[str, Any]) -> None:
    from backend.source_engine.phash.phash_service import PHashService
    from backend.source_engine.semantic_retrieval import SemanticRetrievalService
    from backend.storage_layer.filesystem.asset_manager import AssetManager
    from backend.storage_layer.sqlite.repository import SourceRepository

    repository = SourceRepository(database_path=args.sqlite_path)
    asset_manager = AssetManager(
        source_root=args.source_assets_root,
        images_dir=args.source_assets_root / "images",
        ela_dir=args.source_assets_root / "ela",
    )
    phash_service = PHashService()
    semantic_service = SemanticRetrievalService(
        repository=repository,
        embedding_store_dir=args.embedding_store_dir,
        top_k=args.top_k,
    )

    image_paths = iter_image_paths(args.input_root)
    selected_paths = image_paths[: args.limit] if args.limit else image_paths
    report["summary"]["scanned"] = len(image_paths)
    report["summary"]["skipped_non_images"] = count_non_images(args.input_root)

    for index, image_path in enumerate(selected_paths, start=1):
        category = category_for(image_path, args.input_root)
        try:
            verify_image(image_path)
            sha256 = asset_manager.compute_sha256(image_path)
            existing_asset = repository.find_asset_by_sha256(sha256)
            created = False

            if existing_asset is not None:
                asset = existing_asset
                source_image_path = Path(existing_asset.archived_path)
                report["summary"]["duplicates"] += 1
                increment_category(report, category, "duplicates")
            else:
                phash = phash_service.compute_hash(image_path)
                archived_asset = asset_manager.archive_image(image_path)
                asset, created = repository.upsert_asset(
                    asset_payload={
                        "original_filename": archived_asset.original_filename,
                        "archived_path": str(archived_asset.archived_path),
                        "archived_relative_path": archived_asset.archived_relative_path,
                        "archived_url": archived_asset.archived_url,
                        "sha256": archived_asset.sha256,
                        "phash": phash,
                        "file_size": archived_asset.file_size,
                    }
                )
                source_image_path = archived_asset.archived_path
                if created:
                    report["summary"]["created_assets"] += 1
                    increment_category(report, category, "created_assets")
                else:
                    report["summary"]["duplicates"] += 1
                    increment_category(report, category, "duplicates")

            embedding_before = repository.find_embedding(asset.id, semantic_service.model_name)
            embedding_status = semantic_service.add_or_update_asset(asset, source_image_path)
            if embedding_status.get("error"):
                raise RuntimeError(str(embedding_status["error"]))
            embedding_after = repository.find_embedding(asset.id, semantic_service.model_name)
            if embedding_after is not None:
                report["summary"]["embedding_available"] += 1
                increment_category(report, category, "embedding_available")
            if embedding_before is None and embedding_after is not None:
                report["summary"]["embedding_created_or_verified"] += 1

            report["summary"]["processed"] += 1
            increment_category(report, category, "processed")
            if index % 25 == 0:
                print(f"Processed {index}/{len(selected_paths)} images...")
        except Exception as exc:
            report["summary"]["failed"] += 1
            increment_category(report, category, "failed")
            report["failures"].append(
                {
                    "path": str(image_path),
                    "category": category,
                    "error": str(exc),
                }
            )


def main() -> int:
    args = parse_args()
    configure_environment(args)
    report = make_empty_report(args)

    if not args.input_root.exists():
        print(f"Input root does not exist: {args.input_root}", file=sys.stderr)
        return 2

    if args.dry_run:
        run_dry_scan(args, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        if report["failures"]:
            print(json.dumps(report["failures"][:10], ensure_ascii=False, indent=2))
        return 0 if report["summary"]["failed"] == 0 else 1

    run_import(args, report)
    report_path = write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Import report: {report_path}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
