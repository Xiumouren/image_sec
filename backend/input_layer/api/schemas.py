from typing import Any

from pydantic import BaseModel, Field


class ExplanationResponse(BaseModel):
    method: str
    target_layer: str
    output_path: str
    output_relative_path: str
    error: str | None = None


class SourceAssetMatchResponse(BaseModel):
    image_name: str
    relative_path: str
    source_url: str
    distance: int
    similarity: float


class PHashResponse(BaseModel):
    hash: str
    exact_match: bool = False
    matched: bool
    threshold: int
    top_match: SourceAssetMatchResponse | None = None
    candidates: list[SourceAssetMatchResponse] = Field(default_factory=list)
    error: str | None = None


class ExifResponse(BaseModel):
    has_exif: bool
    summary: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, str] = Field(default_factory=dict)
    software_present: bool = False
    possible_postprocess_hint: bool = False
    error: str | None = None


class ElaResponse(BaseModel):
    supported_for_detection: bool
    unsupported_reason: str | None = None
    is_tampered: bool
    anomaly_ratio: float
    max_error_level: int
    description: str
    output_relative_path: str
    output_url: str
    error: str | None = None


class SourceStorageResponse(BaseModel):
    archived: bool
    record_saved: bool
    asset_reused: bool = False
    error: str | None = None


class SourceAnalysisResponse(BaseModel):
    query: dict[str, Any] = Field(default_factory=dict)
    candidates: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: list[str] = Field(default_factory=list)
    source_credibility_level: str = "unknown"
    review_recommendation: str = ""
    errors: list[str] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)


class LlmSafetyAnalysisResponse(BaseModel):
    content_risk_level: str
    pornographic_assessment: str
    source_credibility_level: str
    source_credibility_assessment: str
    evidence_summary: list[str] = Field(default_factory=list)
    review_recommendation: str
    limitations: list[str] = Field(default_factory=list)
    error: str | None = None


class DetectionResponse(BaseModel):
    image_name: str
    original_image_relative_path: str
    predicted_class_index: int
    predicted_label: str
    scores: dict[str, float]
    explanation: ExplanationResponse | None = None
    source_analysis: SourceAnalysisResponse | None = None
    llm_safety_analysis: LlmSafetyAnalysisResponse | None = None
    error: str | None = None


class BatchResultResponse(BaseModel):
    image_name: str
    relative_path: str
    predicted_class_index: int
    predicted_label: str
    predicted_score: float
    explanation: ExplanationResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    count: int
    method: str
    results: list[BatchResultResponse]
    report_json_path: str
    report_json_relative_path: str
    report_csv_path: str
    report_csv_relative_path: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    labels: list[str]


class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = Field(default=None)
