# Static UI Implementation Guide

## Goal
Build a Vue 3 + Element Plus static demo UI for the current NSFW detection backend. The first version is mock-driven but shaped around the real API so it can be wired later without structural rewrites.

## Scope
- Two-page flow inside one frontend app:
  - Upload page
  - Result page
- Desktop-first layout with mobile-safe stacking
- Mock data for health status, detection success, and request failure
- No real API calls in v1, but preserve the exact backend response shape from `/api/health` and `/api/detect`

## Expected Pages
### Upload Page
- App header and short system summary
- Health status card using `/api/health` fields
- Drag-and-drop upload panel with image preview
- Explanation method selector:
  - `gradcam`
  - `saliency`
  - `integrated_gradients`
- Primary submit button and inline error state

### Result Page
- Original image panel
- Prediction summary card
- Five-class score breakdown
- Heatmap preview panel
- Explanation metadata panel
- Back button for restarting the flow

## Component Plan
- `AppShell`
- `UploadPage`
- `ResultPage`
- `HealthStatusBadge`
- `ImageUploadPanel`
- `MethodSelector`
- `PredictionSummaryCard`
- `ScoreBreakdownCard`
- `HeatmapPreviewCard`
- `ExplanationMetaCard`

## Data Contract
- Health mock should match:
  - `status`
  - `model_loaded`
  - `model_path`
  - `labels`
- Detection mock should match:
  - `image_name`
  - `predicted_class_index`
  - `predicted_label`
  - `scores`
  - `explanation.method`
  - `explanation.target_layer`
  - `explanation.output_path`
  - `explanation.output_relative_path`

## Implementation Notes
- Use Vue Router for `/upload` and `/result`
- Keep API access isolated in one module so mock-to-real replacement is local
- Keep visual tone as a professional security-analysis console, not a landing page
- Preserve a visible placeholder area for future explanation text and source-analysis modules
