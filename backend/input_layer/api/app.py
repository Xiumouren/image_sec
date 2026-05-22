from __future__ import annotations

from pathlib import Path
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from backend.detection_engine.tensorflow.detector_service import DetectorService
from backend.explanation_generation.tensorflow.explainer_service import ExplainerService
from backend.explanation_generation.tensorflow.gradcam_tf import DEFAULT_IG_STEPS
from backend.input_layer.api.schemas import (
    BatchResponse,
    BatchResultResponse,
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
)
from backend.source_engine.source_orchestrator import SourceOrchestrator
from backend.source_engine.llm_review import LlmSafetyReviewService
from backend.shared.config import (
    ALLOWED_METHODS,
    BATCH_METHOD_DEFAULT,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    OUTPUTS_DIR,
    SINGLE_METHOD_DEFAULT,
    SOURCE_ASSETS_ROOT,
    SOURCE_ASSETS_URL_PREFIX,
    SOURCE_ROI_DIR,
    SUPPORTED_EXTENSIONS,
    TEMP_DIR,
    UPLOADS_DIR,
)
from backend.storage_layer.reporting.report_service import ReportService


app = FastAPI(
    title="NSFW Demo API",
    version="0.1.0",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount(SOURCE_ASSETS_URL_PREFIX, StaticFiles(directory=str(SOURCE_ASSETS_ROOT), check_dir=False), name="source-assets")

detector_service = DetectorService(model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH)
explainer_service = ExplainerService()
report_service = ReportService()
source_orchestrator = SourceOrchestrator()
llm_safety_review_service = LlmSafetyReviewService()


def ensure_runtime_dirs() -> None:
    for path in (OUTPUTS_DIR, UPLOADS_DIR, SOURCE_ROI_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
    source_orchestrator.ensure_runtime_dirs()


def select_source_roi_target(prediction: dict[str, object], labels: list[str]) -> tuple[int, str]:
    violation_labels = {"porn", "sexy", "hentai"}
    predicted_index = int(prediction["predicted_class_index"])
    predicted_label = str(prediction["predicted_label"])
    if predicted_label in violation_labels:
        return predicted_index, predicted_label

    raw_scores = prediction.get("raw_scores")
    best_index = predicted_index
    best_label = predicted_label
    best_score = -1.0
    for index, label in enumerate(labels):
        if label not in violation_labels:
            continue
        score = float(raw_scores[index]) if raw_scores is not None and index < len(raw_scores) else 0.0
        if score > best_score:
            best_index = index
            best_label = label
            best_score = score
    return best_index, best_label


def validate_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
    return normalized


def validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")


def save_upload(upload: UploadFile, folder: str) -> Path:
    validate_extension(upload.filename or "")
    target_dir = UPLOADS_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / Path(upload.filename).name
    with output_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return output_path


@app.on_event("startup")
def startup_event() -> None:
    ensure_runtime_dirs()
    detector_service.load_model()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        detector_service.load_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=str(DEFAULT_MODEL_PATH),
        labels=detector_service.get_labels(),
    )


@app.post("/api/detect", response_model=DetectionResponse)
def detect(
    file: UploadFile = File(...),
    method: str = Form(SINGLE_METHOD_DEFAULT),
    ig_steps: int = Form(DEFAULT_IG_STEPS),
) -> DetectionResponse:
    ensure_runtime_dirs()
    normalized_method = validate_method(method)
    image_path = save_upload(file, "single")

    try:
        prediction = detector_service.predict(image_path)
        explanation = None
        explanation_error = None
        try:
            explanation_path = explainer_service.build_output_path(image_path.name, normalized_method, folder="single")
            explanation = explainer_service.explain(
                detector_service.load_model(),
                image_path=image_path,
                predicted_class=prediction["predicted_class_index"],
                method=normalized_method,
                output_path=explanation_path,
                ig_steps=ig_steps,
            )
        except Exception as exc:
            explanation_error = f"Explanation generation failed: {exc}"

        detection_payload = {
            "image_name": image_path.name,
            "predicted_class_index": prediction["predicted_class_index"],
            "predicted_label": prediction["predicted_label"],
            "scores": prediction["scores"],
            "explanation": explanation,
            "error": explanation_error,
        }
        source_roi_heatmap = None
        source_roi_target_label = ""
        try:
            roi_target_index, source_roi_target_label = select_source_roi_target(
                prediction,
                detector_service.get_labels(),
            )
            roi_explanation_path = explainer_service.build_output_path(
                image_path.name,
                "gradcam_roi",
                folder="single",
            )
            roi_explanation = explainer_service.explain_for_source_roi(
                detector_service.load_model(),
                image_path=image_path,
                target_class=roi_target_index,
                output_path=roi_explanation_path,
            )
            source_roi_heatmap = roi_explanation["heatmap"]
        except Exception as exc:
            source_roi_target_label = ""
            if explanation_error:
                explanation_error = f"{explanation_error}; Source ROI Grad-CAM failed: {exc}"
            else:
                explanation_error = f"Source ROI Grad-CAM failed: {exc}"
        detection_payload["error"] = explanation_error
        try:
            source_analysis = source_orchestrator.analyze(
                image_path=image_path,
                detection_payload=detection_payload,
                gradcam_roi_heatmap=source_roi_heatmap,
                gradcam_roi_target_label=source_roi_target_label,
            )
        except Exception as exc:
            source_analysis = {
                "query": {
                    "sha256": "",
                    "phash": "",
                    "archived": False,
                    "record_saved": False,
                    "asset_reused": False,
                    "embedding": {"status": "error"},
                    "roi": {
                        "available": False,
                        "target_label": source_roi_target_label,
                        "bbox": None,
                        "coverage_ratio": 0.0,
                        "roi_url": "",
                        "embedding": {"status": "error"},
                    },
                },
                "candidates": {
                    "full_image": [],
                    "roi": [],
                    "ranking_summary": "Source analysis orchestration failed before retrieval modules completed.",
                },
                "evidence_summary": ["Source analysis orchestration failed."],
                "source_credibility_level": "unknown",
                "review_recommendation": "Retry source analysis after resolving the orchestration error.",
                "errors": [f"Source analysis orchestration failed: {exc}"],
                "signals": {
                    "roi_semantic_retrieval": {
                        "status": "error",
                        "indexed_roi_count": 0,
                        "candidate_count": 0,
                        "error": f"Source analysis orchestration failed: {exc}",
                    }
                },
            }
            if explanation_error:
                explanation_error = f"{explanation_error}; Source analysis failed: {exc}"
            else:
                explanation_error = f"Source analysis failed: {exc}"

        llm_safety_analysis = llm_safety_review_service.analyze(
            image_path=image_path,
            detection_payload=detection_payload,
            source_analysis=source_analysis,
        )

        response = DetectionResponse(
            image_name=image_path.name,
            original_image_relative_path=report_service.relative_to_outputs(image_path),
            predicted_class_index=prediction["predicted_class_index"],
            predicted_label=prediction["predicted_label"],
            scores=prediction["scores"],
            explanation=explanation,
            source_analysis=source_analysis,
            llm_safety_analysis=llm_safety_analysis,
            error=explanation_error,
        )
        report_service.write_single_report(response.model_dump())
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/detect/batch", response_model=BatchResponse)
def detect_batch(
    files: list[UploadFile] | None = File(default=None),
    method: str = Form(BATCH_METHOD_DEFAULT),
    ig_steps: int = Form(DEFAULT_IG_STEPS),
) -> BatchResponse:
    ensure_runtime_dirs()
    normalized_method = validate_method(method)
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    saved_paths = [save_upload(upload, "batch") for upload in files]
    model = detector_service.load_model()
    rows: list[dict] = []
    results: list[BatchResultResponse] = []

    for image_path in saved_paths:
        try:
            prediction = detector_service.predict(image_path)
            explanation_path = explainer_service.build_output_path(
                image_name=image_path.name,
                method=normalized_method,
                folder="batch",
            )
            explanation = explainer_service.explain(
                model=model,
                image_path=image_path,
                predicted_class=prediction["predicted_class_index"],
                method=normalized_method,
                output_path=explanation_path,
                ig_steps=ig_steps,
            )
            result = BatchResultResponse(
                image_name=image_path.name,
                relative_path=image_path.name,
                predicted_class_index=prediction["predicted_class_index"],
                predicted_label=prediction["predicted_label"],
                predicted_score=float(prediction["raw_scores"][prediction["predicted_class_index"]]),
                explanation=explanation,
            )
            rows.append(
                {
                    "image_name": result.image_name,
                    "relative_path": result.relative_path,
                    "predicted_class_index": result.predicted_class_index,
                    "predicted_label": result.predicted_label,
                    "predicted_score": result.predicted_score,
                    "method": normalized_method,
                    "output_path": explanation["output_path"],
                    "error": "",
                }
            )
            results.append(result)
        except Exception as exc:
            error_result = BatchResultResponse(
                image_name=image_path.name,
                relative_path=image_path.name,
                predicted_class_index=-1,
                predicted_label="error",
                predicted_score=0.0,
                error=str(exc),
            )
            rows.append(
                {
                    "image_name": error_result.image_name,
                    "relative_path": error_result.relative_path,
                    "predicted_class_index": error_result.predicted_class_index,
                    "predicted_label": error_result.predicted_label,
                    "predicted_score": error_result.predicted_score,
                    "method": normalized_method,
                    "output_path": "",
                    "error": error_result.error or "",
                }
            )
            results.append(error_result)

    payload = {
        "count": len(results),
        "method": normalized_method,
        "results": [item.model_dump() for item in results],
    }
    json_path, csv_path = report_service.write_batch_reports(rows, payload)
    return BatchResponse(
        count=len(results),
        method=normalized_method,
        results=results,
        report_json_path=str(json_path),
        report_json_relative_path=report_service.relative_to_outputs(json_path),
        report_csv_path=str(csv_path),
        report_csv_relative_path=report_service.relative_to_outputs(csv_path),
    )
