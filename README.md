# NSFW Detection, Source Analysis, and LLM Safety Review Demo

This repository is a local image-safety demo. It combines a TensorFlow NSFW classifier, visual explanations, source-analysis signals, and an optional external multimodal LLM review in one single-image workflow.

The current frontend is a Vue 3 + Element Plus app connected to the FastAPI backend. Uploading one image returns:
- NSFW classification scores
- Grad-CAM, saliency, or integrated gradients heatmap
- source evidence chain with local CLIP semantic retrieval, FAISS candidates, pHash, SHA-256, EXIF, and ELA signals
- EXIF metadata summary
- ELA compression-anomaly analysis
- LLM safety review for pornographic risk and source credibility

## Current Architecture
- `backend/input_layer/api/`: FastAPI entrypoint and response schemas
- `backend/detection_engine/tensorflow/`: TensorFlow model loading and prediction
- `backend/explanation_generation/tensorflow/`: Grad-CAM, saliency, and integrated gradients
- `backend/source_engine/`: source evidence chain, local CLIP embeddings, FAISS retrieval, pHash, EXIF, ELA, and LLM review orchestration
- `backend/source_engine/clip_feature/`: local CLIP image embedding service
- `backend/storage_layer/sqlite/`: SQLite persistence for source-analysis records and embedding metadata
- `backend/storage_layer/filesystem/`: archived image and ELA asset management
- `backend/storage_layer/reporting/`: JSON and CSV report output
- `backend/shared/config.py`: paths, thresholds, and runtime settings
- `frontend/presentation_layer/`: Vue 3 + Element Plus UI
- `rebuild/`: rebuilt Grad-CAM-ready model and rebuild script
- `test_images/`: local sample images
- `outputs/`: uploaded images, explanations, and generated reports

## Runtime Environment
You do not need to run `conda activate` for daily startup. The project already has a local Python interpreter, and all backend commands in this README call that interpreter directly from ordinary PowerShell.

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe --version
```

Exact interpreter path:

```powershell
C:\Code\net_sec\.conda\image_violation\python.exe
```

Frontend Node/NPM path used in this repo:

```powershell
C:\nvm\v20.18.0\npm.cmd
```

The project-local Conda environment is intentionally ignored by Git through `.gitignore`:

```text
.conda/
```

Only use Conda when you need to recreate the project-local environment from the original Anaconda environment:

```powershell
cd C:\Code\net_sec
& 'C:\Anoco\Scripts\conda.exe' create --prefix .\.conda\image_violation --clone image_violation -y
```

Verify the cloned environment:

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe -c "import tensorflow as tf; import fastapi; import PIL; import exifread; import imagehash; import uvicorn; print('deps ok', tf.__version__)"
```

## Recommended Startup With Qwen3-VL-Plus
Use this flow for normal local work on this machine. It runs from ordinary PowerShell and does not require `conda activate`.

1. Connect the `E:\` mobile hard drive.
2. Open PowerShell.
3. Configure your DashScope API key in this same PowerShell session:

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
```

4. Start the backend with the E-drive profile:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

This script calls `C:\Code\net_sec\.conda\image_violation\python.exe` internally.

5. Open a second PowerShell window and start the frontend:

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

6. Open `http://127.0.0.1:5173`, upload one image, and check the result page.

Expected result:
- classification scores and heatmap are returned by local TensorFlow code
- the source-analysis card shows an evidence chain: query hash, CLIP/FAISS candidates, pHash distance, evidence tags, credibility level, and manual-review recommendation
- Qwen3-VL-Plus review appears in the LLM safety-review card

If the Qwen API key is missing or invalid, the detection flow still runs and the LLM card shows a degradation message.

## Optional Qwen3-VL-Plus Configuration
The external multimodal safety review is enabled through environment variables. The current recommended provider is Alibaba Cloud Model Studio with `qwen3-vl-plus`.

```powershell
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
$env:NET_SEC_QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:NET_SEC_QWEN_MODEL="qwen3-vl-plus"
```

Optional Qwen settings:

```powershell
$env:NET_SEC_QWEN_TIMEOUT_SECONDS="30"
$env:NET_SEC_QWEN_ENABLE_THINKING="false"
$env:NET_SEC_QWEN_VL_HIGH_RESOLUTION_IMAGES="false"
```

If no API key is configured, `/api/detect` still works and returns a warning inside `llm_safety_analysis.error`. Do not hardcode API keys in source files, README files, or startup scripts.

The code also keeps backward-compatible OpenAI-compatible variables:

```powershell
$env:NET_SEC_LLM_API_KEY="your_api_key"
$env:NET_SEC_LLM_BASE_URL="https://api.openai.com/v1"
$env:NET_SEC_LLM_MODEL="gpt-4.1-mini"
```

When both variable families are present, `NET_SEC_QWEN_*` takes priority over `NET_SEC_LLM_*`.

Source-analysis storage can also be configured:

```powershell
$env:NET_SEC_SQLITE_PATH="C:\Code\net_sec\data\sqlite\source_engine.db"
$env:NET_SEC_SOURCE_ASSETS_ROOT="E:\net_sec_assets"
$env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR="E:\net_sec_data\semantic_embeddings"
$env:NET_SEC_LOCAL_CLIP_MODEL_NAME="ViT-B/32"
$env:NET_SEC_SEMANTIC_RETRIEVAL_TOP_K="5"
```

By default, archived original images and ELA images are stored under `E:\net_sec_assets`, local CLIP embedding `.npy` files are stored under `E:\net_sec_data\semantic_embeddings`, and temporary uploads, heatmaps, and reports stay under `outputs/`.

## E: Mobile Hard Drive Layout
This machine is configured to use the `E:\` mobile hard drive for large source-analysis data:

```text
E:\
  net_sec_assets\
    images\
    ela\
  net_sec_data\
    source_engine.db
    clip_features\
    semantic_embeddings\
    faiss_index\
  net_sec_backup\
```

Current usage:
- `E:\net_sec_assets\images`: archived uploaded source images
- `E:\net_sec_assets\ela`: generated ELA images
- `E:\net_sec_data\source_engine.db`: SQLite source-analysis database
- `E:\net_sec_data\clip_features`: reserved for legacy CLIP/OpenCLIP embedding experiments
- `E:\net_sec_data\semantic_embeddings`: local CLIP embedding `.npy` files used by the evidence-chain source retriever
- `E:\net_sec_data\faiss_index`: reserved; the current FAISS `IndexFlatIP` is rebuilt in memory from SQLite and embedding files at runtime
- `E:\net_sec_backup`: reserved for model, database, and environment backups

Start the backend with the E-drive storage profile:

```powershell
cd C:\Code\net_sec
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

The script sets:

```powershell
$env:NET_SEC_SOURCE_ASSETS_ROOT="E:\net_sec_assets"
$env:NET_SEC_SQLITE_PATH="E:\net_sec_data\source_engine.db"
$env:NET_SEC_QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:NET_SEC_QWEN_MODEL="qwen3-vl-plus"
$env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR="E:\net_sec_data\semantic_embeddings"
$env:NET_SEC_LOCAL_CLIP_MODEL_NAME="ViT-B/32"
$env:NET_SEC_SEMANTIC_RETRIEVAL_TOP_K="5"
```

Keep the mobile hard drive connected while the backend is running. If `E:\` is disconnected, source-asset storage, SQLite writes, and embedding persistence will fail. Detection still attempts to return a degraded source-analysis response when retrieval or embedding generation fails.

## Recommended Multimodal Safety Models
As of 2026-05-07, the recommended model stack for this project is:

| Priority | Model | How to Use | Why |
| --- | --- | --- | --- |
| 1 | `qwen3-vl-plus` | Default `NET_SEC_QWEN_MODEL` through Alibaba Cloud Model Studio OpenAI-compatible Chat Completions | Best fit for the current local/E-drive setup: direct image input, JSON review output, Chinese safety-review wording, and no local GPU memory pressure. |
| 2 | local `clip` `ViT-B/32` | Default local semantic source retrieval, CPU execution, no external API call | Generates normalized image embeddings for archived assets and query images. FAISS retrieves candidates with cosine similarity through `IndexFlatIP`. |
| 3 | `gpt-4.1-mini` | Backward-compatible `NET_SEC_LLM_MODEL` option through OpenAI-compatible config | Good alternative if you switch providers; supports image input and structured review output. |
| 4 | `gpt-4.1` | Use when review quality matters more than latency/cost | Stronger multimodal reasoning for borderline cases, evidence-chain explanation, and source-credibility analysis. |
| 5 | `qwen3-vl-embedding` | Reserved for a later remote embedding path, not used by the current source retriever | Keep retrieval model choices separate from the LLM review model. The current implementation uses local CLIP first. |
| 6 | `omni-moderation-latest` | Add later as a dedicated `/v1/moderations` pre-check, not as `NET_SEC_QWEN_MODEL` or `NET_SEC_LLM_MODEL` | Purpose-built moderation model that accepts image input and returns category scores. It is better for policy classification than narrative source-review explanation. |

Recommended setup for the current code:

```powershell
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
$env:NET_SEC_QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:NET_SEC_QWEN_MODEL="qwen3-vl-plus"
```

Qwen3-VL-Plus is used for multimodal safety review and evidence explanation. Local CLIP + FAISS is used for image-to-image source retrieval. Keep the evidence chain separated:

```text
sha256/archive record > local CLIP + FAISS candidates > pHash distance > EXIF/ELA > Qwen3-VL-Plus review
```

Current local semantic retrieval settings:

```powershell
$env:NET_SEC_LOCAL_CLIP_MODEL_NAME="ViT-B/32"
$env:NET_SEC_SEMANTIC_RETRIEVAL_TOP_K="5"
$env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR="E:\net_sec_data\semantic_embeddings"
```

Do not set `NET_SEC_QWEN_MODEL` or `NET_SEC_LLM_MODEL` to an embedding model or `omni-moderation-latest`; those belong to separate API paths. The current source retriever does not require `NET_SEC_QWEN_EMBEDDING_API_KEY`.

References:
- Alibaba Cloud Model Studio Qwen OpenAI Chat API: https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions
- Alibaba Cloud Model Studio model list: https://www.alibabacloud.com/help/en/model-studio/models
- OpenAI image moderation: https://platform.openai.com/docs/guides/moderation/overview
- OpenAI `omni-moderation-latest`: https://platform.openai.com/docs/models/omni-moderation-latest
- OpenAI image and vision guide: https://platform.openai.com/docs/guides/images-vision
- Gemini model list: https://ai.google.dev/models/gemini
- Vertex AI safety overview: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/safety-overview
- Anthropic model overview: https://docs.anthropic.com/en/docs/about-claude/models/all-models
- Anthropic vision guide: https://docs.anthropic.com/en/docs/build-with-claude/vision

## Run the Backend API
Recommended startup on this machine from ordinary PowerShell:

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

This uses the project-local Python environment, stores SQLite/source assets on `E:\`, and defaults to `qwen3-vl-plus`.

If you intentionally do not want to use the mobile hard drive, start the API manually:

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
.\.conda\image_violation\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

If you are not in the repo root:

```powershell
C:\Code\net_sec\.conda\image_violation\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --app-dir C:\Code\net_sec
```

Available endpoints:
- `GET /api/health`
- `POST /api/detect`
- `POST /api/detect/batch`

The backend exposes:
- `/outputs`: uploaded files, heatmaps, and reports
- `/source-assets`: archived source-analysis images and ELA images

## Run the Frontend

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd install
C:\nvm\v20.18.0\npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

The Vite dev server proxies `/api`, `/outputs`, and `/source-assets` to `http://127.0.0.1:8000`.

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

## Single-Image Usage Flow
1. Connect the `E:\` mobile hard drive.
2. Start the backend API with `scripts\start_backend_e_drive.ps1`.
3. Start the frontend.
4. Open `http://127.0.0.1:5173`.
5. Upload an image.
6. Select the explanation method:
   - `gradcam`
   - `saliency`
   - `integrated_gradients`
7. Click the detection button.
8. Review the result page:
   - classification summary
   - score breakdown
   - original image and heatmap
   - source evidence-chain card with top candidates, evidence tags, credibility level, and manual-review prompt
   - LLM safety-review card

The source evidence-chain card is the primary attribution surface. It groups SHA-256, local CLIP/FAISS retrieval, pHash, EXIF, and ELA into candidate-level evidence. The LLM card summarizes pornographic risk, source credibility, supporting evidence, review recommendation, and limitations. It should be treated as an audit assistant, not as the sole source of truth.

## Minimal Local Startup
Backend with E-drive storage:

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

Frontend:

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173` after both processes are running.

## Single-Image Usage Flow Without E Drive
Use this only when the mobile hard drive is not connected. Data will be written to the repository-local defaults instead of `E:\`.

1. Start the backend manually. Set `NET_SEC_QWEN_API_KEY` first if you want Qwen3-VL-Plus review.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Upload an image.
5. Select the explanation method:
   - `gradcam`
   - `saliency`
   - `integrated_gradients`
6. Click the detection button.
7. Review the result page:
   - classification summary
   - score breakdown
   - original image and heatmap
   - source evidence-chain card
   - LLM safety-review card

Manual backend command without `E:\`:

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
.\.conda\image_violation\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Quick Start: Generate a Heatmap

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe gradcam_tf.py --model rebuild\nsfw_mobilenetv2_gradcam_ready.h5 --image test_images\porn.jpg --class-index 3 --output outputs\porn_gradcam.jpg --method gradcam
```

## Manual Smoke Tests
Single-image prediction:

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe scripts\manual\test_mobilenet_model.py
```

Batch explanation:

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe scripts\manual\batch_explain_nsfw.py --method saliency
```

Frontend production build:

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd run build
```

Semantic retrieval dependency check:

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe -c "import clip, torch, faiss, numpy as np; idx=faiss.IndexFlatIP(3); x=np.array([[1,0,0]], dtype='float32'); idx.add(x); print(idx.search(x, 1)[0][0][0])"
```

## Third-Party Open Source

This project builds on several third-party open-source projects and model/tooling ecosystems. Keep their original licenses and attribution requirements when publishing, redistributing model files, or packaging this project.

- [GantMan/nsfw_model](https://github.com/GantMan/nsfw_model): NSFW MobileNet/TensorFlow model source used as the basis for the local NSFW classification workflow and Grad-CAM-ready rebuild process.
- TensorFlow / Keras: model loading, inference, and gradient-based explanation implementation.
- OpenAI CLIP / `clip`: local image embedding generation for semantic source retrieval.
- FAISS: in-memory vector search for CLIP embedding candidates.
- `ImageHash`: pHash-based perceptual hash comparison.
- `ExifRead`: EXIF metadata extraction.
- FastAPI / Uvicorn / Pydantic: backend API, request parsing, and response schemas.
- Vue 3, Vite, Element Plus, and Vue Router: frontend UI and local development tooling.
- Alibaba Cloud Model Studio Qwen3-VL-Plus or other OpenAI-compatible multimodal providers: optional external LLM safety review when an API key is configured.

Large model weights, generated outputs, local sample images, and runtime databases are intentionally excluded from Git. If model weights are shared, prefer GitHub Releases, Git LFS, or a documented download script, and include the upstream license/notice files alongside them.

## Notes
- Default model: `rebuild/nsfw_mobilenetv2_gradcam_ready.h5`.
- Single-image detection is the primary supported frontend workflow.
- Batch detection exists in the backend API, but the batch frontend page is not implemented yet.
- Source retrieval uses local CLIP embeddings and an in-memory FAISS `IndexFlatIP`; the index is rebuilt from SQLite assets and embedding files on startup or first request.
- Embedding files are stored as `.npy` files under `SEMANTIC_EMBEDDING_STORE_DIR`; SQLite stores metadata and paths, not raw vectors.
- New archived images are added to the in-memory FAISS index after successful persistence.
- pHash is a perceptual hash signal and is displayed as distance/similarity evidence, not as standalone visual proof.
- ELA tamper scoring is only enabled for JPEG-like inputs; other formats return a degradation notice.
- EXIF output is summarized and filtered; sensitive raw metadata is not intended for broad exposure.
- LLM review requires an external multimodal API key and may add request latency.
