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

DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite" / "source_engine.db"
DEFAULT_SOURCE_ASSETS_ROOT = Path(r"E:\net_sec_assets")
DEFAULT_EMBEDDING_STORE_DIR = Path(r"E:\net_sec_data\semantic_embeddings")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "rebuild" / "nsfw_mobilenetv2_gradcam_ready.h5"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "mobilenet_v2_140_224" / "class_labels.txt"
VIOLATION_LABELS = {"porn", "sexy", "hentai"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Grad-CAM ROI crops, ROI hashes, and ROI CLIP embeddings for existing source_assets."
    )
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--source-assets-root", type=Path, default=DEFAULT_SOURCE_ASSETS_ROOT)
    parser.add_argument("--embedding-store-dir", type=Path, default=DEFAULT_EMBEDDING_STORE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Create a new ROI record even if an asset already has ROI rows.")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing ROI rows and ROI embedding files for each selected asset before backfilling.",
    )
    parser.add_argument(
        "--fallback-image-root",
        type=Path,
        action="append",
        default=[PROJECT_ROOT / "outputs" / "uploads" / "single"],
        help="Directory to search by original filename when archived_path is missing. Can be passed multiple times.",
    )
    parser.add_argument(
        "--repair-missing-archives",
        action="store_true",
        help="Copy a resolved fallback image back to the archived_path stored in source_assets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["NET_SEC_SQLITE_PATH"] = str(args.sqlite_path)
    os.environ["NET_SEC_SOURCE_ASSETS_ROOT"] = str(args.source_assets_root)
    os.environ["NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR"] = str(args.embedding_store_dir)
    os.environ["NET_SEC_LOCAL_CLIP_MODEL_NAME"] = args.clip_model
    os.environ["NET_SEC_SEMANTIC_RETRIEVAL_TOP_K"] = str(args.top_k)


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "sqlite_path": str(args.sqlite_path),
        "source_assets_root": str(args.source_assets_root),
        "embedding_store_dir": str(args.embedding_store_dir),
        "clip_model": args.clip_model,
        "limit": args.limit,
        "force": bool(args.force),
        "replace_existing": bool(args.replace_existing),
        "fallback_image_roots": [str(path) for path in args.fallback_image_root],
        "repair_missing_archives": bool(args.repair_missing_archives),
        "dry_run": bool(args.dry_run),
        "summary": {
            "scanned_assets": 0,
            "processed_assets": 0,
            "skipped_existing_roi": 0,
            "replaced_existing_roi": 0,
            "repaired_archives": 0,
            "roi_assets_created_or_reused": 0,
            "roi_embeddings_available": 0,
            "failed": 0,
        },
        "results": [],
        "failures": [],
    }


def write_report(report: dict[str, Any]) -> Path:
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"backfill_roi_assets_{timestamp}.json"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def resolve_image_path(asset, fallback_roots: list[Path]) -> tuple[Path, Path | None]:
    archived_path = Path(asset.archived_path)
    if archived_path.exists():
        return archived_path, None
    for root in fallback_roots:
        root = Path(root)
        if not root.exists():
            continue
        for candidate in root.rglob(asset.original_filename):
            if candidate.is_file():
                return candidate, archived_path
    raise FileNotFoundError(f"Archived image is missing and no fallback matched {asset.original_filename}: {archived_path}")


def repair_archive_if_needed(source_path: Path, archived_path: Path | None, enabled: bool) -> bool:
    if archived_path is None or not enabled:
        return False
    archived_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(source_path, archived_path)
    return True


def delete_file_if_safe(path: str) -> None:
    target = Path(path)
    if target.exists() and target.is_file():
        target.unlink()


def replace_existing_roi_assets(repository, asset_id: int, model_name: str) -> int:
    existing = [roi for roi in repository.list_roi_assets() if roi.asset_id == asset_id]
    if not existing:
        return 0

    for roi in existing:
        embedding = repository.find_roi_embedding(roi.id, model_name)
        if embedding is not None:
            delete_file_if_safe(embedding.embedding_path)
        delete_file_if_safe(roi.roi_path)

    deleted = repository.delete_roi_assets_for_asset(asset_id)
    return len(deleted)


def select_source_roi_target(prediction: dict[str, object], labels: list[str]) -> tuple[int, str]:
    predicted_index = int(prediction["predicted_class_index"])
    predicted_label = str(prediction["predicted_label"])
    if predicted_label in VIOLATION_LABELS:
        return predicted_index, predicted_label

    raw_scores = prediction.get("raw_scores")
    best_index = predicted_index
    best_label = predicted_label
    best_score = -1.0
    for index, label in enumerate(labels):
        if label not in VIOLATION_LABELS:
            continue
        score = float(raw_scores[index]) if raw_scores is not None and index < len(raw_scores) else 0.0
        if score > best_score:
            best_index = index
            best_label = label
            best_score = score
    return best_index, best_label


def source_roi_target_candidates(prediction: dict[str, object], labels: list[str]) -> list[tuple[int, str]]:
    predicted_index = int(prediction["predicted_class_index"])
    predicted_label = str(prediction["predicted_label"])
    raw_scores = prediction.get("raw_scores")

    scored: list[tuple[float, int, str]] = []
    for index, label in enumerate(labels):
        if label not in VIOLATION_LABELS:
            continue
        score = float(raw_scores[index]) if raw_scores is not None and index < len(raw_scores) else 0.0
        boost = 1.0 if index == predicted_index and label in VIOLATION_LABELS else 0.0
        scored.append((boost + score, index, label))

    if not scored:
        return [(predicted_index, predicted_label)]

    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    candidates: list[tuple[int, str]] = []
    seen: set[int] = set()
    for _, index, label in ordered:
        if index in seen:
            continue
        seen.add(index)
        candidates.append((index, label))
    if predicted_label not in VIOLATION_LABELS and predicted_index not in seen:
        candidates.append((predicted_index, predicted_label))
    return candidates


def build_services(args: argparse.Namespace) -> dict[str, Any]:
    from backend.detection_engine.tensorflow.detector_service import DetectorService
    from backend.explanation_generation.tensorflow.explainer_service import ExplainerService
    from backend.source_engine.gradcam_roi import GradcamRoiService
    from backend.source_engine.phash.phash_service import PHashService
    from backend.source_engine.semantic_retrieval import RoiSemanticRetrievalService
    from backend.storage_layer.filesystem.asset_manager import AssetManager
    from backend.storage_layer.sqlite.repository import SourceRepository

    repository = SourceRepository(database_path=args.sqlite_path)
    asset_manager = AssetManager(
        source_root=args.source_assets_root,
        images_dir=args.source_assets_root / "images",
        ela_dir=args.source_assets_root / "ela",
        roi_dir=args.source_assets_root / "roi",
    )
    return {
        "repository": repository,
        "asset_manager": asset_manager,
        "phash_service": PHashService(),
        "roi_service": GradcamRoiService(),
        "roi_semantic_service": RoiSemanticRetrievalService(
            repository=repository,
            embedding_store_dir=args.embedding_store_dir,
            top_k=args.top_k,
        ),
        "detector_service": DetectorService(model_path=args.model_path, labels_path=args.labels_path),
        "explainer_service": ExplainerService(),
    }


def existing_roi_count(repository, asset_id: int) -> int:
    return sum(1 for roi in repository.list_roi_assets() if roi.asset_id == asset_id)


def backfill_asset(asset, services: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    repository = services["repository"]
    asset_manager = services["asset_manager"]
    phash_service = services["phash_service"]
    roi_service = services["roi_service"]
    roi_semantic_service = services["roi_semantic_service"]
    detector_service = services["detector_service"]
    explainer_service = services["explainer_service"]

    image_path, missing_archive_path = resolve_image_path(asset, args.fallback_image_root)
    verify_image(image_path)
    repaired_archive = repair_archive_if_needed(
        source_path=image_path,
        archived_path=missing_archive_path,
        enabled=args.repair_missing_archives,
    )
    if repaired_archive and missing_archive_path is not None:
        image_path = missing_archive_path

    prediction = detector_service.predict(image_path)
    labels = detector_service.get_labels()
    model = detector_service.load_model()
    roi_payload = None
    target_label = ""
    errors = []
    for target_index, candidate_label in source_roi_target_candidates(prediction, labels):
        target_label = candidate_label
        try:
            roi_explanation_path = explainer_service.build_output_path(
                image_path.name,
                f"gradcam_roi_{candidate_label}",
                folder="roi_backfill",
            )
            roi_explanation = explainer_service.explain_for_source_roi(
                model,
                image_path=image_path,
                target_class=target_index,
                output_path=roi_explanation_path,
            )
            candidate_roi_payload = roi_service.extract_roi(
                image_path=image_path,
                heatmap=roi_explanation["heatmap"],
                target_label=candidate_label,
            )
            if candidate_roi_payload.get("available"):
                roi_payload = candidate_roi_payload
                break
            errors.append(str(candidate_roi_payload.get("error") or f"{candidate_label} ROI unavailable."))
        except Exception as exc:
            errors.append(f"{candidate_label}: {exc}")

    if roi_payload is None:
        raise RuntimeError("; ".join(errors) or "ROI extraction did not produce a crop.")

    archived_roi = asset_manager.archive_roi_crop(
        crop_path=Path(str(roi_payload["crop_path"])),
        asset_id=asset.id,
        target_label=target_label,
    )
    roi_phash = phash_service.compute_hash(archived_roi.roi_path)
    roi_record = repository.upsert_roi_asset(
        asset_id=asset.id,
        target_label=target_label,
        bbox_json=json.dumps(roi_payload.get("bbox") or {}, ensure_ascii=False),
        coverage_ratio=float(roi_payload.get("coverage_ratio") or 0.0),
        roi_path=str(archived_roi.roi_path),
        roi_relative_path=archived_roi.roi_relative_path,
        roi_url=archived_roi.roi_url,
        roi_sha256=archived_roi.roi_sha256,
        roi_phash=roi_phash,
    )
    embedding_status = roi_semantic_service.add_or_update_roi(roi_record, archived_roi.roi_path)
    if embedding_status.get("error"):
        raise RuntimeError(str(embedding_status["error"]))

    return {
        "asset_id": asset.id,
        "image_name": asset.original_filename,
        "predicted_label": prediction["predicted_label"],
        "target_label": target_label,
        "roi_id": roi_record.id,
        "roi_url": roi_record.roi_url,
        "bbox": roi_payload.get("bbox"),
        "coverage_ratio": roi_payload.get("coverage_ratio"),
        "repaired_archive": repaired_archive,
        "embedding_status": embedding_status,
    }


def run(args: argparse.Namespace) -> int:
    configure_environment(args)
    report = make_report(args)
    services = build_services(args)
    repository = services["repository"]
    assets = repository.list_assets()
    selected_assets = assets[: args.limit] if args.limit else assets
    report["summary"]["scanned_assets"] = len(assets)

    if args.dry_run:
        for asset in selected_assets:
            try:
                image_path, missing_archive_path = resolve_image_path(asset, args.fallback_image_root)
                verify_image(image_path)
                if args.repair_missing_archives and missing_archive_path is not None:
                    report["summary"]["repaired_archives"] += 1
                has_existing_roi = existing_roi_count(repository, asset.id) > 0
                if args.replace_existing and has_existing_roi:
                    report["summary"]["replaced_existing_roi"] += 1
                    report["summary"]["processed_assets"] += 1
                elif not args.force and has_existing_roi:
                    report["summary"]["skipped_existing_roi"] += 1
                else:
                    report["summary"]["processed_assets"] += 1
            except Exception as exc:
                report["summary"]["failed"] += 1
                report["failures"].append({"asset_id": asset.id, "path": asset.archived_path, "error": str(exc)})
        report_path = write_report(report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"Dry-run report: {report_path}")
        return 0 if report["summary"]["failed"] == 0 else 1

    for index, asset in enumerate(selected_assets, start=1):
        try:
            if args.replace_existing:
                report["summary"]["replaced_existing_roi"] += replace_existing_roi_assets(
                    repository,
                    asset.id,
                    services["roi_semantic_service"].model_name,
                )
            elif not args.force and existing_roi_count(repository, asset.id):
                report["summary"]["skipped_existing_roi"] += 1
                continue
            result = backfill_asset(asset, services, args)
            report["summary"]["processed_assets"] += 1
            report["summary"]["repaired_archives"] += int(bool(result.get("repaired_archive")))
            report["summary"]["roi_assets_created_or_reused"] += 1
            report["summary"]["roi_embeddings_available"] += int(bool(result["embedding_status"].get("indexed")))
            report["results"].append(result)
            print(f"ROI backfilled {index}/{len(selected_assets)} asset_id={asset.id} roi_id={result['roi_id']}")
        except Exception as exc:
            report["summary"]["failed"] += 1
            report["failures"].append({"asset_id": asset.id, "path": asset.archived_path, "error": str(exc)})
            print(f"ROI backfill failed asset_id={asset.id}: {exc}", file=sys.stderr)

    report_path = write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Backfill report: {report_path}")
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
