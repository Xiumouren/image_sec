from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.source_engine.ela.ela_service import ElaService
from backend.source_engine.exif.exif_service import ExifService
from backend.source_engine.gradcam_roi import GradcamRoiService
from backend.source_engine.phash.phash_service import PHashService
from backend.source_engine.semantic_retrieval import RoiSemanticRetrievalService, SemanticRetrievalService
from backend.storage_layer.filesystem.asset_manager import AssetManager
from backend.storage_layer.sqlite.repository import SourceRepository


class SourceOrchestrator:
    def __init__(
        self,
        repository: SourceRepository | None = None,
        asset_manager: AssetManager | None = None,
        phash_service: PHashService | None = None,
        exif_service: ExifService | None = None,
        ela_service: ElaService | None = None,
        semantic_service: SemanticRetrievalService | None = None,
        roi_semantic_service: RoiSemanticRetrievalService | None = None,
        gradcam_roi_service: GradcamRoiService | None = None,
    ) -> None:
        self.repository = repository or SourceRepository()
        self.asset_manager = asset_manager or AssetManager()
        self.phash_service = phash_service or PHashService()
        self.exif_service = exif_service or ExifService()
        self.ela_service = ela_service or ElaService()
        self.semantic_service = semantic_service or SemanticRetrievalService(repository=self.repository)
        self.roi_semantic_service = roi_semantic_service or RoiSemanticRetrievalService(repository=self.repository)
        self.gradcam_roi_service = gradcam_roi_service or GradcamRoiService()

    def ensure_runtime_dirs(self) -> None:
        self.asset_manager.ensure_directories()

    def analyze(
        self,
        image_path: Path,
        detection_payload: dict,
        gradcam_roi_heatmap: np.ndarray | None = None,
        gradcam_roi_target_label: str = "",
    ) -> dict[str, object]:
        self.ensure_runtime_dirs()
        source_sha256 = self.asset_manager.compute_sha256(image_path)
        existing_asset = self.repository.find_asset_by_sha256(source_sha256)
        archived_asset = self.asset_manager.record_to_archived_asset(existing_asset) if existing_asset else None

        current_phash = existing_asset.phash if existing_asset else self.phash_service.compute_hash(image_path)
        historical_assets = self.repository.list_assets_except_sha256(source_sha256)

        phash_result = self._safe_phash_analysis(current_phash, historical_assets, existing_asset)
        exif_result = self._safe_exif_analysis(image_path)
        ela_result = self._safe_ela_analysis(image_path.name, image_path)
        semantic_result = self._safe_semantic_search(image_path)
        gradcam_roi_result = self._safe_gradcam_roi_search(
            image_path=image_path,
            heatmap=gradcam_roi_heatmap,
            target_label=gradcam_roi_target_label,
        )
        storage_result = {
            "archived": archived_asset is not None,
            "record_saved": False,
            "asset_reused": archived_asset is not None,
            "error": None,
        }

        created_archive = False
        if archived_asset is None:
            archived_asset = self.asset_manager.archive_image(image_path)
            created_archive = True

        source_payload = self._build_evidence_chain(
            query_sha256=source_sha256,
            query_phash=current_phash,
            archived_asset=archived_asset,
            phash_result=phash_result,
            exif_result=exif_result,
            ela_result=ela_result,
            semantic_result=semantic_result,
            gradcam_roi_result=gradcam_roi_result,
            storage_result={
                "archived": True,
                "record_saved": True,
                "asset_reused": existing_asset is not None,
                "error": None,
            },
        )

        try:
            persist_result = self.repository.persist_analysis(
                asset_payload={
                    "original_filename": archived_asset.original_filename,
                    "archived_path": str(archived_asset.archived_path),
                    "archived_relative_path": archived_asset.archived_relative_path,
                    "archived_url": archived_asset.archived_url,
                    "sha256": archived_asset.sha256,
                    "phash": current_phash,
                    "file_size": archived_asset.file_size,
                },
                predicted_class_index=int(detection_payload["predicted_class_index"]),
                predicted_label=str(detection_payload["predicted_label"]),
                detection_payload=detection_payload,
                source_payload=source_payload,
            )
            embedding_status = self._safe_index_archived_asset(persist_result.asset, archived_asset.archived_path)
            roi_embedding_status = self._safe_persist_and_index_roi(
                asset_id=persist_result.asset.id,
                gradcam_roi_result=gradcam_roi_result,
            )
            gradcam_roi_result = {
                **gradcam_roi_result,
                "embedding_status": roi_embedding_status,
            }
            storage_result = {
                "archived": True,
                "record_saved": persist_result.record_saved,
                "asset_reused": persist_result.asset_reused,
                "error": None,
            }
            source_payload = self._build_evidence_chain(
                query_sha256=source_sha256,
                query_phash=current_phash,
                archived_asset=archived_asset,
                phash_result=phash_result,
                exif_result=exif_result,
                ela_result=ela_result,
                semantic_result={**semantic_result, "embedding_status": embedding_status},
                gradcam_roi_result=gradcam_roi_result,
                storage_result=storage_result,
            )
        except Exception as exc:
            if created_archive:
                self.asset_manager.delete_file(archived_asset.archived_path)
            if ela_result["output_relative_path"]:
                self.asset_manager.delete_file(self.asset_manager.source_root / ela_result["output_relative_path"])
                ela_result = {**ela_result, "output_relative_path": "", "output_url": ""}
            storage_result = {
                "archived": not created_archive,
                "record_saved": False,
                "asset_reused": existing_asset is not None,
                "error": f"Storage persistence failed: {exc}",
            }
            source_payload = self._build_evidence_chain(
                query_sha256=source_sha256,
                query_phash=current_phash,
                archived_asset=archived_asset,
                phash_result=phash_result,
                exif_result=exif_result,
                ela_result=ela_result,
                semantic_result=semantic_result,
                gradcam_roi_result=gradcam_roi_result,
                storage_result=storage_result,
            )

        return source_payload

    def _safe_phash_analysis(
        self,
        current_phash: str,
        historical_assets: list,
        exact_match_asset=None,
    ) -> dict[str, object]:
        try:
            return self.phash_service.analyze(current_hash=current_phash, historical_assets=historical_assets, exact_match_asset=exact_match_asset)
        except Exception as exc:
            return {
                "hash": current_phash,
                "exact_match": False,
                "matched": False,
                "threshold": self.phash_service.threshold,
                "top_match": None,
                "candidates": [],
                "error": f"pHash analysis failed: {exc}",
            }

    def _safe_exif_analysis(self, image_path: Path) -> dict[str, object]:
        try:
            return self.exif_service.analyze(image_path)
        except Exception as exc:
            return {
                "has_exif": False,
                "summary": {},
                "raw": {},
                "software_present": False,
                "possible_postprocess_hint": False,
                "error": f"EXIF analysis failed: {exc}",
            }

    def _safe_ela_analysis(self, image_name: str, image_path: Path) -> dict[str, object]:
        try:
            ela_asset = self.asset_manager.build_ela_asset(image_name)
            return self.ela_service.analyze(
                image_path=image_path,
                output_path=ela_asset.output_path,
                output_relative_path=ela_asset.output_relative_path,
                output_url=ela_asset.output_url,
            )
        except Exception as exc:
            return {
                "supported_for_detection": False,
                "unsupported_reason": "ELA generation failed.",
                "is_tampered": False,
                "anomaly_ratio": 0.0,
                "max_error_level": 0,
                "description": "ELA generation failed.",
                "output_relative_path": "",
                "output_url": "",
                "error": f"ELA analysis failed: {exc}",
            }

    def _safe_semantic_search(self, image_path: Path) -> dict[str, object]:
        try:
            return self.semantic_service.search(image_path)
        except Exception as exc:
            return {
                "model_name": self.semantic_service.model_name,
                "embedding_dim": None,
                "indexed_asset_count": 0,
                "candidates": [],
                "error": f"Semantic retrieval failed: {exc}",
            }

    def _safe_gradcam_roi_search(
        self,
        *,
        image_path: Path,
        heatmap: np.ndarray | None,
        target_label: str,
    ) -> dict[str, object]:
        if heatmap is None:
            return {
                "available": False,
                "target_label": target_label,
                "bbox": None,
                "coverage_ratio": 0.0,
                "crop_relative_path": "",
                "candidates": [],
                "error": None,
            }
        roi_payload = self.gradcam_roi_service.extract_roi(
            image_path=image_path,
            heatmap=heatmap,
            target_label=target_label,
        )
        if not roi_payload.get("available"):
            return {**roi_payload, "candidates": []}
        try:
            roi_phash = self.phash_service.compute_hash(Path(str(roi_payload["crop_path"])))
            search_result = self.roi_semantic_service.search(
                Path(str(roi_payload["crop_path"])),
                query_phash=roi_phash,
            )
            return {
                **roi_payload,
                "roi_phash": roi_phash,
                "model_name": search_result.get("model_name"),
                "embedding_dim": search_result.get("embedding_dim"),
                "indexed_roi_count": search_result.get("indexed_roi_count", 0),
                "candidates": search_result.get("candidates", []),
                "error": search_result.get("error"),
            }
        except Exception as exc:
            return {
                **roi_payload,
                "candidates": [],
                "error": f"Grad-CAM ROI semantic retrieval failed: {exc}",
            }

    def _safe_persist_and_index_roi(
        self,
        *,
        asset_id: int,
        gradcam_roi_result: dict[str, object],
    ) -> dict[str, object]:
        if not gradcam_roi_result.get("available") or not gradcam_roi_result.get("crop_path"):
            return {
                "status": "unavailable",
                "indexed": False,
                "error": None,
            }
        try:
            archived_roi = self.asset_manager.archive_roi_crop(
                crop_path=Path(str(gradcam_roi_result["crop_path"])),
                asset_id=asset_id,
                target_label=str(gradcam_roi_result.get("target_label") or "roi"),
            )
            roi_phash = str(gradcam_roi_result.get("roi_phash") or self.phash_service.compute_hash(archived_roi.roi_path))
            roi_record = self.repository.upsert_roi_asset(
                asset_id=asset_id,
                target_label=str(gradcam_roi_result.get("target_label") or ""),
                bbox_json=json.dumps(gradcam_roi_result.get("bbox") or {}, ensure_ascii=False),
                coverage_ratio=float(gradcam_roi_result.get("coverage_ratio") or 0.0),
                roi_path=str(archived_roi.roi_path),
                roi_relative_path=archived_roi.roi_relative_path,
                roi_url=archived_roi.roi_url,
                roi_sha256=archived_roi.roi_sha256,
                roi_phash=roi_phash,
            )
            index_status = self.roi_semantic_service.add_or_update_roi(roi_record, archived_roi.roi_path)
            return {
                **index_status,
                "status": "available" if not index_status.get("error") else "error",
                "roi_url": roi_record.roi_url,
                "roi_relative_path": roi_record.roi_relative_path,
                "roi_sha256": roi_record.roi_sha256,
                "roi_phash": roi_record.roi_phash,
            }
        except Exception as exc:
            return {
                "status": "error",
                "indexed": False,
                "error": f"ROI evidence persistence failed: {exc}",
            }

    def _safe_index_archived_asset(self, asset, archived_path: Path) -> dict[str, object]:
        try:
            return self.semantic_service.add_or_update_asset(asset, archived_path)
        except Exception as exc:
            return {
                "asset_id": getattr(asset, "id", None),
                "model_name": self.semantic_service.model_name,
                "embedding_dim": None,
                "indexed": False,
                "error": f"Semantic embedding archive failed: {exc}",
            }

    def _build_evidence_chain(
        self,
        *,
        query_sha256: str,
        query_phash: str,
        archived_asset,
        phash_result: dict[str, object],
        exif_result: dict[str, object],
        ela_result: dict[str, object],
        semantic_result: dict[str, object],
        gradcam_roi_result: dict[str, object],
        storage_result: dict[str, object],
    ) -> dict[str, object]:
        candidate_groups = self._build_candidate_groups(phash_result, semantic_result, gradcam_roi_result)
        ranked_candidates = [
            *candidate_groups["full_image"],
            *candidate_groups["roi"],
        ]
        combined_ranked_candidates = sorted(
            ranked_candidates,
            key=lambda item: (
                1 if item.get("exact_sha256_match") else 0,
                self._credibility_rank(str(item["credibility_level"])),
                float(item.get("roi_semantic_similarity") or item.get("gradcam_roi_similarity") or 0.0),
                float(item.get("semantic_similarity") or 0.0),
                float(item.get("phash_similarity") or 0.0),
            ),
            reverse=True,
        )
        errors = [
            error
            for error in (
                phash_result.get("error"),
                exif_result.get("error"),
                ela_result.get("error"),
                semantic_result.get("error"),
                gradcam_roi_result.get("error"),
                storage_result.get("error"),
                (semantic_result.get("embedding_status") or {}).get("error")
                if isinstance(semantic_result.get("embedding_status"), dict)
                else None,
                (gradcam_roi_result.get("embedding_status") or {}).get("error")
                if isinstance(gradcam_roi_result.get("embedding_status"), dict)
                else None,
            )
            if error
        ]
        source_level = self._overall_credibility(combined_ranked_candidates, errors, semantic_result)
        evidence_summary = self._evidence_summary(
            combined_ranked_candidates,
            exif_result,
            ela_result,
            semantic_result,
            gradcam_roi_result,
            errors,
        )
        grouped_candidates = {
            **candidate_groups,
            "ranking_summary": self._ranking_summary(
                candidate_groups["full_image"],
                candidate_groups["roi"],
            ),
        }
        return {
            "query": {
                "sha256": query_sha256,
                "phash": query_phash,
                "archived": bool(storage_result.get("archived")),
                "record_saved": bool(storage_result.get("record_saved")),
                "asset_reused": bool(storage_result.get("asset_reused")),
                "archived_url": getattr(archived_asset, "archived_url", ""),
                "embedding": {
                    "model_name": semantic_result.get("model_name"),
                    "embedding_dim": semantic_result.get("embedding_dim"),
                    "indexed_asset_count": semantic_result.get("indexed_asset_count", 0),
                    "status": "error" if semantic_result.get("error") else "available",
                    "archive_status": semantic_result.get("embedding_status"),
                },
                "roi": {
                    "available": bool(gradcam_roi_result.get("available")),
                    "target_label": gradcam_roi_result.get("target_label", ""),
                    "bbox": gradcam_roi_result.get("bbox"),
                    "coverage_ratio": gradcam_roi_result.get("coverage_ratio", 0.0),
                    "mask_threshold": gradcam_roi_result.get("mask_threshold"),
                    "fallback_used": bool(gradcam_roi_result.get("fallback_used")),
                    "mask_coverage_ratio": gradcam_roi_result.get("mask_coverage_ratio", 0.0),
                    "roi_url": (gradcam_roi_result.get("embedding_status") or {}).get("roi_url", "")
                    if isinstance(gradcam_roi_result.get("embedding_status"), dict)
                    else "",
                    "crop_relative_path": gradcam_roi_result.get("crop_relative_path", ""),
                    "embedding": {
                        "model_name": gradcam_roi_result.get("model_name"),
                        "embedding_dim": gradcam_roi_result.get("embedding_dim"),
                        "indexed_roi_count": gradcam_roi_result.get("indexed_roi_count", 0),
                        "status": "error" if gradcam_roi_result.get("error") else (
                            "available" if gradcam_roi_result.get("available") else "unavailable"
                        ),
                        "archive_status": gradcam_roi_result.get("embedding_status"),
                    },
                },
            },
            "candidates": grouped_candidates,
            "evidence_summary": evidence_summary,
            "source_credibility_level": source_level,
            "review_recommendation": self._review_recommendation(source_level, combined_ranked_candidates, errors),
            "errors": errors,
            "signals": {
                "phash": phash_result,
                "exif": exif_result,
                "ela": ela_result,
                "semantic_retrieval": semantic_result,
                "gradcam_roi": gradcam_roi_result,
                "roi_semantic_retrieval": {
                    "model_name": gradcam_roi_result.get("model_name"),
                    "embedding_dim": gradcam_roi_result.get("embedding_dim"),
                    "indexed_roi_count": gradcam_roi_result.get("indexed_roi_count", 0),
                    "candidate_count": len(gradcam_roi_result.get("candidates", []) or []),
                    "status": "error" if gradcam_roi_result.get("error") else (
                        "available" if gradcam_roi_result.get("available") else "unavailable"
                    ),
                    "error": gradcam_roi_result.get("error"),
                },
                "storage": storage_result,
            },
        }

    def _build_candidate_groups(
        self,
        phash_result: dict[str, object],
        semantic_result: dict[str, object],
        gradcam_roi_result: dict[str, object],
    ) -> dict[str, list[dict[str, object]]]:
        return {
            "full_image": self._build_full_image_candidates(phash_result, semantic_result),
            "roi": self._build_roi_candidates(gradcam_roi_result),
        }

    def _build_full_image_candidates(
        self,
        phash_result: dict[str, object],
        semantic_result: dict[str, object],
    ) -> list[dict[str, object]]:
        by_asset_id: dict[int, dict[str, object]] = {}
        for item in semantic_result.get("candidates", []) or []:
            asset_id = int(item["asset_id"])
            by_asset_id[asset_id] = {
                "asset_id": asset_id,
                "image_name": item.get("image_name", ""),
                "relative_path": item.get("relative_path", ""),
                "source_url": item.get("source_url", ""),
                "semantic_similarity": item.get("semantic_similarity"),
                "phash_distance": None,
                "phash_similarity": None,
                "exact_sha256_match": False,
                "sha256": item.get("sha256", ""),
                "exif_evidence": {},
                "ela_evidence": {},
                "evidence": [],
            }
        for item in phash_result.get("candidates", []) or []:
            asset_id = int(item["asset_id"])
            candidate = by_asset_id.get(asset_id, {
                "asset_id": asset_id,
                "image_name": item.get("image_name", ""),
                "relative_path": item.get("relative_path", ""),
                "source_url": item.get("source_url", ""),
                "semantic_similarity": None,
                "phash_distance": None,
                "phash_similarity": None,
                "exact_sha256_match": False,
                "sha256": item.get("sha256", ""),
                "exif_evidence": {},
                "ela_evidence": {},
                "evidence": [],
            })
            candidate = {
                **candidate,
                "phash_distance": item.get("distance"),
                "phash_similarity": item.get("similarity"),
                "exact_sha256_match": bool(phash_result.get("exact_match") and int(item.get("distance", 1)) == 0),
                "sha256": item.get("sha256", candidate.get("sha256", "")),
            }
            by_asset_id[asset_id] = candidate

        enriched = []
        for candidate in by_asset_id.values():
            evidence = self._candidate_evidence(candidate)
            level = self._candidate_credibility(candidate, evidence)
            enriched.append(
                {
                    **candidate,
                    "evidence": evidence,
                    "credibility_level": level,
                    "needs_human_review": level == "needs_human_review",
                }
            )
        return sorted(
            enriched,
            key=lambda item: (
                1 if item.get("exact_sha256_match") else 0,
                self._credibility_rank(str(item["credibility_level"])),
                float(item.get("semantic_similarity") or 0.0),
                float(item.get("phash_similarity") or 0.0),
            ),
            reverse=True,
        )

    def _build_roi_candidates(self, gradcam_roi_result: dict[str, object]) -> list[dict[str, object]]:
        enriched = []
        for item in gradcam_roi_result.get("candidates", []) or []:
            candidate = {
                "roi_id": item.get("roi_id"),
                "asset_id": int(item["asset_id"]),
                "image_name": item.get("image_name", ""),
                "relative_path": item.get("relative_path", ""),
                "source_url": item.get("source_url", ""),
                "roi_url": item.get("roi_url", ""),
                "roi_relative_path": item.get("roi_relative_path", ""),
                "source_bbox": item.get("source_bbox"),
                "target_label": item.get("target_label") or gradcam_roi_result.get("target_label", ""),
                "roi_semantic_similarity": item.get("roi_semantic_similarity"),
                "roi_phash_distance": item.get("roi_phash_distance"),
                "query_bbox": gradcam_roi_result.get("bbox"),
                "query_coverage_ratio": gradcam_roi_result.get("coverage_ratio"),
                "evidence": [],
            }
            evidence = self._roi_candidate_evidence(candidate)
            level = self._roi_candidate_credibility(candidate, evidence)
            enriched.append(
                {
                    **candidate,
                    "evidence": evidence,
                    "credibility_level": level,
                    "needs_human_review": level == "needs_human_review",
                }
            )
        return sorted(
            enriched,
            key=lambda item: (
                self._credibility_rank(str(item["credibility_level"])),
                float(item.get("roi_semantic_similarity") or 0.0),
                -float(item.get("roi_phash_distance") or 99),
            ),
            reverse=True,
        )

    @staticmethod
    def _candidate_evidence(candidate: dict[str, object]) -> list[dict[str, object]]:
        evidence = []
        semantic_similarity = candidate.get("semantic_similarity")
        if isinstance(semantic_similarity, (int, float)):
            level = "strong" if semantic_similarity >= 0.88 else "medium" if semantic_similarity >= 0.78 else "weak"
            evidence.append(
                {
                    "type": "semantic_clip",
                    "level": level,
                    "message": "CLIP semantic image-to-image retrieval signal.",
                    "score": round(float(semantic_similarity), 4),
                }
            )
        phash_distance = candidate.get("phash_distance")
        phash_similarity = candidate.get("phash_similarity")
        if isinstance(phash_distance, int) and isinstance(phash_similarity, (int, float)):
            level = "strong" if phash_distance <= 6 else "medium" if phash_distance <= 12 else "weak"
            evidence.append(
                {
                    "type": "perceptual_hash",
                    "level": level,
                    "message": f"pHash distance {phash_distance}; this is a perceptual hash signal, not semantic proof.",
                    "score": round(float(phash_similarity), 4),
                }
            )
        if candidate.get("exact_sha256_match"):
            evidence.append(
                {
                    "type": "sha256",
                    "level": "strong",
                    "message": "SHA-256 exactly matches an archived asset.",
                    "score": 1.0,
                }
            )
        return evidence

    @staticmethod
    def _roi_candidate_evidence(candidate: dict[str, object]) -> list[dict[str, object]]:
        evidence = []
        roi_similarity = candidate.get("roi_semantic_similarity")
        if isinstance(roi_similarity, (int, float)):
            level = "strong" if roi_similarity >= 0.82 else "medium" if roi_similarity >= 0.72 else "weak"
            evidence.append(
                {
                    "type": "roi_semantic_clip",
                    "level": level,
                    "message": "CLIP retrieval signal from historical violation ROI crops.",
                    "score": round(float(roi_similarity), 4),
                }
            )
        roi_phash_distance = candidate.get("roi_phash_distance")
        if isinstance(roi_phash_distance, int):
            level = "strong" if roi_phash_distance <= 6 else "medium" if roi_phash_distance <= 12 else "weak"
            evidence.append(
                {
                    "type": "roi_perceptual_hash",
                    "level": level,
                    "message": f"ROI pHash distance {roi_phash_distance}.",
                    "score": None,
                }
            )
        return evidence

    @staticmethod
    def _candidate_credibility(candidate: dict[str, object], evidence: list[dict[str, object]]) -> str:
        semantic = candidate.get("semantic_similarity")
        phash_distance = candidate.get("phash_distance")
        if candidate.get("exact_sha256_match"):
            return "high"
        if isinstance(semantic, (int, float)) and isinstance(phash_distance, int):
            if semantic >= 0.88 and phash_distance <= 8:
                return "high"
            if semantic >= 0.88 and phash_distance > 18:
                return "needs_human_review"
            if semantic < 0.72 and phash_distance <= 6:
                return "needs_human_review"
            if semantic >= 0.78:
                return "medium"
        if any(item["level"] == "medium" for item in evidence):
            return "medium"
        if evidence:
            return "low"
        return "unknown"

    @staticmethod
    def _roi_candidate_credibility(candidate: dict[str, object], evidence: list[dict[str, object]]) -> str:
        roi_similarity = candidate.get("roi_semantic_similarity")
        roi_phash_distance = candidate.get("roi_phash_distance")
        if isinstance(roi_similarity, (int, float)):
            if roi_similarity >= 0.86 and isinstance(roi_phash_distance, int) and roi_phash_distance <= 8:
                return "high"
            if roi_similarity >= 0.82 and isinstance(roi_phash_distance, int) and roi_phash_distance > 18:
                return "needs_human_review"
            if roi_similarity >= 0.78:
                return "medium"
        if any(item["level"] == "medium" for item in evidence):
            return "medium"
        if evidence:
            return "low"
        return "unknown"

    @staticmethod
    def _ranking_summary(full_image_candidates: list[dict[str, object]], roi_candidates: list[dict[str, object]]) -> str:
        if not full_image_candidates and not roi_candidates:
            return "No full-image or ROI source candidates were retrieved."
        if full_image_candidates and not roi_candidates:
            return "Only full-image retrieval produced candidates; ROI retrieval is empty or unavailable."
        if roi_candidates and not full_image_candidates:
            return "Only ROI retrieval produced candidates; full-image retrieval is empty or unavailable."

        full_top = full_image_candidates[0]
        roi_top = roi_candidates[0]
        if full_top.get("asset_id") == roi_top.get("asset_id"):
            return "Full-image and ROI retrieval agree on the same top archived asset."
        return "Full-image and ROI retrieval point to different top assets; compare both modules manually."

    @staticmethod
    def _overall_credibility(candidates: list[dict[str, object]], errors: list[str], semantic_result: dict[str, object]) -> str:
        if errors and not candidates:
            return "unknown"
        if not candidates:
            return "unknown"
        if any(candidate.get("needs_human_review") for candidate in candidates):
            return "needs_human_review"
        top_level = str(candidates[0].get("credibility_level", "unknown"))
        if len(candidates) > 1:
            top_score = float(candidates[0].get("semantic_similarity") or 0.0)
            next_score = float(candidates[1].get("semantic_similarity") or 0.0)
            top_roi_score = float(candidates[0].get("roi_semantic_similarity") or 0.0)
            next_roi_score = float(candidates[1].get("roi_semantic_similarity") or 0.0)
            if top_score and next_score and abs(top_score - next_score) < 0.03:
                return "needs_human_review"
            if top_roi_score and next_roi_score and abs(top_roi_score - next_roi_score) < 0.03:
                return "needs_human_review"
        if semantic_result.get("indexed_asset_count", 0) == 0 and not candidates:
            return "unknown"
        return top_level

    @staticmethod
    def _evidence_summary(
        candidates: list[dict[str, object]],
        exif_result: dict[str, object],
        ela_result: dict[str, object],
        semantic_result: dict[str, object],
        gradcam_roi_result: dict[str, object],
        errors: list[str],
    ) -> list[str]:
        summary = []
        if candidates:
            top = candidates[0]
            summary.append(
                f"Top candidate {top.get('image_name')} has credibility {top.get('credibility_level')}."
            )
        else:
            summary.append("No archived source candidate was retrieved.")
        summary.append(f"Semantic index contains {semantic_result.get('indexed_asset_count', 0)} archived assets.")
        if gradcam_roi_result.get("available"):
            summary.append(
                "Grad-CAM ROI retrieval used the "
                f"{gradcam_roi_result.get('target_label') or 'violation'} hot area as a local source signal."
            )
        if exif_result.get("has_exif"):
            summary.append("EXIF metadata is present and available as supporting context.")
        if ela_result.get("is_tampered"):
            summary.append("ELA reported compression anomalies; manual review is recommended.")
        if errors:
            summary.append("Some source-analysis modules degraded; inspect errors before attribution.")
        return summary

    @staticmethod
    def _review_recommendation(level: str, candidates: list[dict[str, object]], errors: list[str]) -> str:
        if level == "high":
            return "Treat the top candidate as a strong source match, then verify context and policy category."
        if level == "needs_human_review":
            return "Manually compare the query image with top candidates because evidence is strong but conflicting or close."
        if level == "medium":
            return "Review the top candidate and supporting metadata before making a source attribution."
        if level == "low":
            return "Use candidates only as weak leads; do not attribute source without additional evidence."
        if errors:
            return "Source retrieval degraded; retry after resolving module errors or inspect archived assets manually."
        return "No reliable source attribution is available from the current archive."

    @staticmethod
    def _credibility_rank(level: str) -> int:
        return {
            "high": 5,
            "needs_human_review": 4,
            "medium": 3,
            "low": 2,
            "unknown": 1,
        }.get(level, 0)
