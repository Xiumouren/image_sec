from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
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
        description=(
            "Rebuild source-analysis SQLite records from the currently archived source_assets, "
            "then rerun NSFW detection, Grad-CAM ROI extraction, and source analysis for every asset."
        )
    )
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--source-assets-root", type=Path, default=DEFAULT_SOURCE_ASSETS_ROOT)
    parser.add_argument("--embedding-store-dir", type=Path, default=DEFAULT_EMBEDDING_STORE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-rebuild", action="store_true")
    parser.add_argument(
        "--skip-embedding-backup",
        action="store_true",
        help="Do not copy the current CLIP embedding model directory before clearing it.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["NET_SEC_SQLITE_PATH"] = str(args.sqlite_path)
    os.environ["NET_SEC_SOURCE_ASSETS_ROOT"] = str(args.source_assets_root)
    os.environ["NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR"] = str(args.embedding_store_dir)
    os.environ["NET_SEC_LOCAL_CLIP_MODEL_NAME"] = args.clip_model
    os.environ["NET_SEC_SEMANTIC_RETRIEVAL_TOP_K"] = str(args.top_k)


def load_manifest(database_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {database_path}")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, original_filename, archived_path, archived_relative_path, archived_url, sha256, phash, file_size, created_at
            FROM source_assets
            ORDER BY id ASC
            """
        ).fetchall()
    selected_rows = rows[:limit] if limit else rows
    return [dict(row) for row in selected_rows]


def verify_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures = []
    for item in manifest:
        image_path = Path(str(item["archived_path"]))
        if not image_path.exists():
            failures.append({"path": str(image_path), "error": "Archived image is missing."})
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            failures.append({"path": str(image_path), "error": f"Image verification failed: {exc}"})
    return failures


def make_report(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "confirm_rebuild": bool(args.confirm_rebuild),
        "sqlite_path": str(args.sqlite_path),
        "source_assets_root": str(args.source_assets_root),
        "embedding_store_dir": str(args.embedding_store_dir),
        "clip_model": args.clip_model,
        "top_k": args.top_k,
        "limit": args.limit,
        "backup": {},
        "summary": {
            "manifest_assets": len(manifest),
            "verified_assets": 0,
            "created_assets": 0,
            "embedding_available": 0,
            "analysis_records": 0,
            "failed": 0,
        },
        "asset_id_map": [],
        "results": [],
        "failures": [],
    }


def write_report(report: dict[str, Any]) -> Path:
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"rebuild_source_analysis_{timestamp}.json"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def backup_sqlite(database_path: Path) -> Path:
    backup_path = database_path.with_name(
        f"{database_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak{database_path.suffix}"
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(database_path)
    try:
        backup_connection = sqlite3.connect(backup_path)
        try:
            source_connection.backup(backup_connection)
        finally:
            backup_connection.close()
    finally:
        source_connection.close()
    return backup_path


def model_embedding_dir(embedding_store_dir: Path, clip_model: str) -> Path:
    safe_model_name = f"clip:{clip_model}".replace(":", "_").replace("/", "_")
    return embedding_store_dir / safe_model_name


def backup_and_clear_embeddings(args: argparse.Namespace) -> Path | None:
    embedding_dir = model_embedding_dir(args.embedding_store_dir, args.clip_model)
    if not embedding_dir.exists():
        embedding_dir.mkdir(parents=True, exist_ok=True)
        return None
    backup_path = embedding_dir.with_name(
        f"{embedding_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    )
    if not args.skip_embedding_backup:
        shutil.copytree(embedding_dir, backup_path)
    shutil.rmtree(embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    return None if args.skip_embedding_backup else backup_path


def clear_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN")
        try:
            connection.execute("DELETE FROM source_asset_embeddings")
            connection.execute("DELETE FROM source_analysis_records")
            connection.execute("DELETE FROM source_assets")
            connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('source_assets', 'source_analysis_records')")
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise


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


def build_detection_payload(image_path: Path, detector_service, explainer_service, model, labels: list[str]) -> tuple[dict[str, Any], Any, str]:
    from backend.explanation_generation.tensorflow.gradcam_tf import DEFAULT_IG_STEPS

    prediction = detector_service.predict(image_path)
    explanation_error = None
    explanation = None
    try:
        explanation_path = explainer_service.build_output_path(image_path.name, "gradcam", folder="rebuild")
        explanation = explainer_service.explain(
            model,
            image_path=image_path,
            predicted_class=int(prediction["predicted_class_index"]),
            method="gradcam",
            output_path=explanation_path,
            ig_steps=DEFAULT_IG_STEPS,
        )
    except Exception as exc:
        explanation_error = f"Explanation generation failed: {exc}"

    roi_heatmap = None
    roi_target_label = ""
    try:
        roi_target_index, roi_target_label = select_source_roi_target(prediction, labels)
        roi_explanation_path = explainer_service.build_output_path(image_path.name, "gradcam_roi", folder="rebuild")
        roi_explanation = explainer_service.explain_for_source_roi(
            model,
            image_path=image_path,
            target_class=roi_target_index,
            output_path=roi_explanation_path,
        )
        roi_heatmap = roi_explanation["heatmap"]
    except Exception as exc:
        if explanation_error:
            explanation_error = f"{explanation_error}; Source ROI Grad-CAM failed: {exc}"
        else:
            explanation_error = f"Source ROI Grad-CAM failed: {exc}"

    detection_payload = {
        "image_name": image_path.name,
        "predicted_class_index": int(prediction["predicted_class_index"]),
        "predicted_label": str(prediction["predicted_label"]),
        "scores": prediction["scores"],
        "explanation": explanation,
        "error": explanation_error,
    }
    return detection_payload, roi_heatmap, roi_target_label


def reinsert_assets(manifest: list[dict[str, Any]], services: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    repository = services["repository"]
    asset_manager = services["asset_manager"]
    phash_service = services["phash_service"]
    semantic_service = services["semantic_service"]

    rebuilt_assets = []
    for item in manifest:
        image_path = Path(str(item["archived_path"]))
        try:
            sha256 = asset_manager.compute_sha256(image_path)
            phash = phash_service.compute_hash(image_path)
            asset, created = repository.upsert_asset(
                asset_payload={
                    "original_filename": str(item["original_filename"]),
                    "archived_path": str(image_path),
                    "archived_relative_path": str(item["archived_relative_path"]),
                    "archived_url": str(item["archived_url"]),
                    "sha256": sha256,
                    "phash": phash,
                    "file_size": image_path.stat().st_size,
                }
            )
            embedding_status = semantic_service.add_or_update_asset(asset, image_path)
            if embedding_status.get("error"):
                raise RuntimeError(str(embedding_status["error"]))
            report["summary"]["created_assets"] += int(created)
            report["summary"]["embedding_available"] += 1
            report["asset_id_map"].append({"old_asset_id": item["id"], "new_asset_id": asset.id})
            rebuilt_assets.append({"old": item, "asset": asset, "image_path": image_path})
        except Exception as exc:
            report["summary"]["failed"] += 1
            report["failures"].append(
                {
                    "stage": "asset_reinsert",
                    "old_asset_id": item.get("id"),
                    "path": str(image_path),
                    "error": str(exc),
                }
            )
    return rebuilt_assets


def rerun_analysis(rebuilt_assets: list[dict[str, Any]], services: dict[str, Any], report: dict[str, Any]) -> None:
    detector_service = services["detector_service"]
    explainer_service = services["explainer_service"]
    source_orchestrator = services["source_orchestrator"]
    labels = detector_service.get_labels()
    model = detector_service.load_model()

    for index, item in enumerate(rebuilt_assets, start=1):
        image_path = item["image_path"]
        asset = item["asset"]
        try:
            detection_payload, roi_heatmap, roi_target_label = build_detection_payload(
                image_path=image_path,
                detector_service=detector_service,
                explainer_service=explainer_service,
                model=model,
                labels=labels,
            )
            source_payload = source_orchestrator.analyze(
                image_path=image_path,
                detection_payload=detection_payload,
                gradcam_roi_heatmap=roi_heatmap,
                gradcam_roi_target_label=roi_target_label,
            )
            report["summary"]["analysis_records"] += 1
            report["results"].append(
                {
                    "asset_id": asset.id,
                    "image_name": asset.original_filename,
                    "predicted_label": detection_payload["predicted_label"],
                    "source_credibility_level": source_payload.get("source_credibility_level"),
                    "top_candidate": (source_payload.get("candidates") or [{}])[0].get("image_name"),
                    "errors": source_payload.get("errors", []),
                }
            )
            if index % 10 == 0:
                print(f"Analyzed {index}/{len(rebuilt_assets)} images...")
        except Exception as exc:
            report["summary"]["failed"] += 1
            report["failures"].append(
                {
                    "stage": "analysis",
                    "asset_id": asset.id,
                    "path": str(image_path),
                    "error": str(exc),
                }
            )


def build_services(args: argparse.Namespace) -> dict[str, Any]:
    from backend.detection_engine.tensorflow.detector_service import DetectorService
    from backend.explanation_generation.tensorflow.explainer_service import ExplainerService
    from backend.source_engine.ela.ela_service import ElaService
    from backend.source_engine.exif.exif_service import ExifService
    from backend.source_engine.gradcam_roi import GradcamRoiService
    from backend.source_engine.phash.phash_service import PHashService
    from backend.source_engine.semantic_retrieval import SemanticRetrievalService
    from backend.source_engine.source_orchestrator import SourceOrchestrator
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
    source_orchestrator = SourceOrchestrator(
        repository=repository,
        asset_manager=asset_manager,
        phash_service=phash_service,
        exif_service=ExifService(),
        ela_service=ElaService(),
        semantic_service=semantic_service,
        gradcam_roi_service=GradcamRoiService(),
    )
    return {
        "repository": repository,
        "asset_manager": asset_manager,
        "phash_service": phash_service,
        "semantic_service": semantic_service,
        "source_orchestrator": source_orchestrator,
        "detector_service": DetectorService(model_path=args.model_path, labels_path=args.labels_path),
        "explainer_service": ExplainerService(),
    }


def run_dry_run(args: argparse.Namespace, report: dict[str, Any], manifest: list[dict[str, Any]]) -> int:
    failures = verify_manifest(manifest)
    report["summary"]["verified_assets"] = len(manifest) - len(failures)
    report["summary"]["failed"] = len(failures)
    report["failures"].extend({"stage": "manifest_verify", **failure} for failure in failures)
    write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if failures:
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def run_rebuild(args: argparse.Namespace, report: dict[str, Any], manifest: list[dict[str, Any]]) -> int:
    if not args.confirm_rebuild:
        print("Refusing to clear database without --confirm-rebuild.", file=sys.stderr)
        return 2

    failures = verify_manifest(manifest)
    report["summary"]["verified_assets"] = len(manifest) - len(failures)
    if failures:
        report["summary"]["failed"] = len(failures)
        report["failures"].extend({"stage": "manifest_verify", **failure} for failure in failures)
        write_report(report)
        print("Refusing to rebuild because one or more archived images failed verification.", file=sys.stderr)
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
        return 1

    sqlite_backup = backup_sqlite(args.sqlite_path)
    report["backup"]["sqlite_path"] = str(sqlite_backup)
    embedding_backup = backup_and_clear_embeddings(args)
    report["backup"]["embedding_path"] = str(embedding_backup) if embedding_backup else ""

    clear_database(args.sqlite_path)
    services = build_services(args)
    rebuilt_assets = reinsert_assets(manifest, services, report)
    services["semantic_service"].rebuild_index()
    rerun_analysis(rebuilt_assets, services, report)

    report_path = write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"SQLite backup: {sqlite_backup}")
    if embedding_backup:
        print(f"Embedding backup: {embedding_backup}")
    print(f"Rebuild report: {report_path}")
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    args = parse_args()
    configure_environment(args)
    manifest = load_manifest(args.sqlite_path, args.limit)
    report = make_report(args, manifest)
    if args.dry_run:
        return run_dry_run(args, report, manifest)
    return run_rebuild(args, report, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
