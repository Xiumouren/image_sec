param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".conda\image_violation\python.exe"
$AssetsRoot = "E:\net_sec_assets"
$DataRoot = "E:\net_sec_data"
$SqlitePath = Join-Path $DataRoot "source_engine.db"

if (-not (Test-Path "E:\")) {
    throw "E:\ drive is not available. Please connect the mobile hard drive first."
}

foreach ($Path in @(
    (Join-Path $AssetsRoot "images"),
    (Join-Path $AssetsRoot "ela"),
    (Join-Path $DataRoot "clip_features"),
    (Join-Path $DataRoot "semantic_embeddings"),
    (Join-Path $DataRoot "faiss_index")
)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

if (-not (Test-Path $PythonPath)) {
    throw "Project-local Python was not found: $PythonPath"
}

$env:NET_SEC_SOURCE_ASSETS_ROOT = $AssetsRoot
$env:NET_SEC_SQLITE_PATH = $SqlitePath
if (-not $env:NET_SEC_QWEN_BASE_URL) {
    $env:NET_SEC_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
if (-not $env:NET_SEC_QWEN_MODEL) {
    $env:NET_SEC_QWEN_MODEL = "qwen3-vl-plus"
}
if (-not $env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR) {
    $env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR = Join-Path $DataRoot "semantic_embeddings"
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $PythonPath"
Write-Host "Source assets: $env:NET_SEC_SOURCE_ASSETS_ROOT"
Write-Host "SQLite: $env:NET_SEC_SQLITE_PATH"
Write-Host "Qwen base URL: $env:NET_SEC_QWEN_BASE_URL"
Write-Host "Qwen model: $env:NET_SEC_QWEN_MODEL"
Write-Host "Semantic embedding store: $env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR"
if (-not $env:NET_SEC_QWEN_API_KEY -and -not $env:NET_SEC_LLM_API_KEY) {
    Write-Host "Qwen API key is not configured; /api/detect will still run with LLM review degraded."
}

& $PythonPath -m uvicorn app:app --host $HostAddress --port $Port --app-dir $ProjectRoot
