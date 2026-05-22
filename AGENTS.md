# Repository Guidelines

## Project Structure & Module Organization
- `backend/` holds the production code aligned to `revised_implementation_plan.md`.
- `backend/input_layer/api/` contains the FastAPI app and request/response schemas.
- `backend/detection_engine/tensorflow/` contains TensorFlow model loading and prediction logic.
- `backend/explanation_generation/tensorflow/` contains Grad-CAM, saliency, and explanation services.
- `backend/storage_layer/reporting/` writes JSON and CSV reports.
- `backend/shared/` stores shared configuration such as paths and defaults.
- `frontend/presentation_layer/` is reserved for the future UI.
- `rebuild/`, `mobilenet_v2_140_224/`, and `test_images/` contain rebuilt model assets, original model files, and local test inputs.
- Utility scripts are grouped under `scripts/manual/` and `scripts/checks/`.

## Build, Test, and Development Commands
- Activate the main environment first: `conda activate image_violation`
- Run the API locally: `C:\Anoco\envs\image_violation\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000`
- Run a single-image prediction smoke test: `C:\Anoco\envs\image_violation\python.exe scripts\manual\test_mobilenet_model.py`
- Generate a Grad-CAM explanation: `C:\Anoco\envs\image_violation\python.exe gradcam_tf.py --model rebuild\\nsfw_mobilenetv2_gradcam_ready.h5 --image test_images\\porn.jpg --class-index 3 --output outputs\\porn_gradcam.jpg --method gradcam`
- Batch explanations: `C:\Anoco\envs\image_violation\python.exe scripts\manual\batch_explain_nsfw.py --method saliency`

## Coding Style & Naming Conventions
- Use Python 3.10+ with 4-space indentation and type hints for new code.
- Prefer small, focused modules under the architecture folders instead of adding new root-level files.
- Use `snake_case` for files, functions, and variables; use `PascalCase` for classes and Pydantic models.
- Keep path and runtime constants in `backend/shared/config.py`.
- Reuse existing services before duplicating model, explanation, or report logic.

## Testing Guidelines
- No formal test suite is committed yet; use reproducible smoke tests before merging.
- Validate both importability and runtime behavior in the Conda environment.
- For API changes, test `GET /api/health` and at least one `POST /api/detect` request.
- When adding tests later, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits, for example: `feat: add batch explanation endpoint` or `fix: correct Grad-CAM output path`.
- Keep PRs focused on one subsystem when possible.
- Include a short summary, affected paths, manual test commands, and example output paths or screenshots for UI/heatmap changes.

## Security & Configuration Tips
- Do not hardcode secrets, tokens, or external service credentials.
- Keep model paths and output directories configurable through `backend/shared/config.py`.
- Large generated files belong in `outputs/`; do not mix them into source directories.
