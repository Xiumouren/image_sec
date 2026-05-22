
# NSFW 检测、溯源证据链与大模型安全复核演示

本仓库是一个本地图像安全检测演示项目。单图检测链路整合了 TensorFlow NSFW 分类、可视化解释、违规来源溯源证据链，以及可选的外部多模态大模型安全复核。

当前前端是 Vue 3 + Element Plus，已接入 FastAPI 后端。上传单张图片后会返回：
- NSFW 分类分数
- Grad-CAM、saliency 或 integrated gradients 热力图
- 本地 CLIP + FAISS 语义候选召回
- SHA-256、pHash、EXIF、ELA 组成的候选级证据链
- 来源可信等级和人工复核建议
- Qwen3-VL 等外部多模态大模型的安全复核结果

## 演示结果与流程

`image/` 目录中的截图展示了当前单图检测演示的完整链路。本次示例中，上传图片被分类为 `neutral`，TOP1 概率为 `54.10%`；其余类别分数分别为 `drawings` `16.85%`、`sexy` `14.73%`、`porn` `11.88%`、`hentai` `2.44%`。

![分类结果、原图与 Grad-CAM 热力图](<image/屏幕截图 2026-05-22 112650.png>)

结果页首先展示五分类概率分布，然后并排展示原图和后端生成的 Grad-CAM 热力图。解释元信息会记录解释方法（`gradcam`）、目标层（`Conv_1`）以及热力图输出路径（`outputs/explanations/gradcam/single/`）。

![热力图细节与溯源证据链概览](<image/屏幕截图 2026-05-22 112421.png>)

分类完成后，溯源证据链会把当前图片与已归档资产进行对比。本示例中，当前图片已计算 SHA-256、pHash，并通过 `clip:ViT-B/32` 建立本地 CLIP 向量索引；同时还生成了 `sexy` 区域的 ROI 摘要，该区域覆盖图片 `19.33%`，并与 17 条 ROI 库记录进行比对。

![完整原图检索候选](<image/屏幕截图 2026-05-22 112413.png>)

完整原图检索会按归档来源候选排序。本次演示的最高候选与当前图片 `100.00%` 语义相似、pHash 距离为 `0`，并且 SHA-256 完全一致，因此被标记为高可信。其他接近候选仍会展示语义相似度、pHash、SHA-256 和证据标签，方便审核人员比较相近或冲突结果。

![ROI 局部检索候选](<image/屏幕截图 2026-05-22 112431.png>)

ROI 局部检索会在热力图关注区域上重复相似度比较，而不是只看整张图片。列表中会展示 ROI 语义相似度、ROI pHash 距离、历史 bbox、类别标签和可信等级，用于区分“整图来源一致”和“局部区域相似”两类证据。

![大模型安全复核与证据摘要](<image/屏幕截图 2026-05-22 113117.png>)

大模型安全复核卡片会把检测分数、Grad-CAM 覆盖比例、SHA-256/pHash 身份校验、CLIP 检索、ELA、EXIF 和 ROI 证据合并成面向审核人员的摘要。本示例中，复核结论为内容低风险、来源高可信，并解释 Grad-CAM 命中的区域更可能是非显式内容特征；针对该结果，大模型给出的建议是不需要人工复核。

建议查看顺序：
1. 启动后端和前端。
2. 在前端上传单张图片。
3. 查看分类摘要和五分类概率图。
4. 检查原图与生成的热力图。
5. 查看溯源证据链中的 SHA-256、pHash、CLIP/FAISS 候选、EXIF 和 ELA。
6. 当证据接近或冲突时，对比完整原图检索和 ROI 局部检索。
7. 将大模型安全复核作为审核辅助，而不是唯一判定依据。

## 当前架构

- `backend/input_layer/api/`：FastAPI 入口和响应结构
- `backend/detection_engine/tensorflow/`：TensorFlow 模型加载与预测
- `backend/explanation_generation/tensorflow/`：Grad-CAM、saliency、integrated gradients
- `backend/source_engine/`：溯源证据链、pHash、EXIF、ELA、语义检索和大模型复核编排
- `backend/source_engine/clip_feature/`：本地 CLIP 图片向量服务
- `backend/storage_layer/sqlite/`：溯源记录、归档资产和 embedding 元数据持久化
- `backend/storage_layer/filesystem/`：归档原图、ELA 图和大文件资产管理
- `backend/storage_layer/reporting/`：JSON、CSV 报告输出
- `backend/shared/config.py`：路径、阈值和运行时配置
- `frontend/presentation_layer/`：Vue 3 + Element Plus 前端
- `rebuild/`：可用于 Grad-CAM 的重建模型
- `test_images/`：本地测试图片
- `outputs/`：上传图片、解释图和报告输出

## 运行环境

日常启动不需要执行 `conda activate`。项目内已有本地 Python 解释器，README 中的后端命令都直接调用它：

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe --version
```

解释器路径：

```powershell
C:\Code\net_sec\.conda\image_violation\python.exe
```

前端 Node/NPM 路径：

```powershell
C:\nvm\v20.18.0\npm.cmd
```

如果需要从原始 Anaconda 环境重新克隆项目内环境：

```powershell
cd C:\Code\net_sec
& 'C:\Anoco\Scripts\conda.exe' create --prefix .\.conda\image_violation --clone image_violation -y
```

验证依赖：

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe -c "import tensorflow as tf; import fastapi; import PIL; import exifread; import imagehash; import clip; import torch; import faiss; print('deps ok', tf.__version__)"
```

## 推荐启动流程

1. 连接 `E:\` 移动硬盘。
2. 打开 PowerShell。
3. 如果需要 Qwen3-VL-Plus 复核，在同一个 PowerShell 会话中配置 API Key：

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
```

4. 使用 E 盘配置启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

5. 另开一个 PowerShell 窗口启动前端：

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

6. 打开 `http://127.0.0.1:5173`，上传图片并查看结果页。

预期结果：
- 本地 TensorFlow 返回分类分数和热力图
- 溯源卡片展示当前图 hash、Top candidates、语义相似度、pHash 距离、证据标签、可信等级和人工复核建议
- 如配置了 API Key，大模型复核卡片会展示 Qwen3-VL-Plus 的安全复核结果

如果未配置 Qwen API Key，检测和溯源仍会运行，`llm_safety_analysis.error` 会返回降级提示。

## 溯源证据链

`/api/detect` 的 `source_analysis` 现在是统一证据链结构：

- `query`：当前图片的 `sha256`、`phash`、归档状态、embedding 状态
- `candidates`：候选来源列表，每个候选包含归档 URL、语义相似度、pHash 距离、SHA-256 是否完全一致、证据项和候选可信等级
- `evidence_summary`：全局证据摘要
- `source_credibility_level`：`high`、`medium`、`low`、`unknown` 或 `needs_human_review`
- `review_recommendation`：面向审核员的下一步建议
- `errors`：各子模块降级错误
- `signals`：pHash、EXIF、ELA、semantic retrieval、storage 的底层信号

第一版语义检索默认使用本地 `clip` 包加载 `ViT-B/32`，CPU 运行。图片向量会归一化为 `float32`，FAISS 使用 `IndexFlatIP` 做 inner product 检索；在归一化向量上等价于 cosine similarity。FAISS 索引不持久化为 `.index` 文件，而是在启动或首次请求时从 SQLite 资产和 embedding 文件重建。

可信度规则概览：
- `high`：SHA-256 完全一致，或 CLIP 高相似且 pHash 距离很小
- `medium`：CLIP 相似较高，但 pHash、EXIF、ELA 不充分或只提供部分支持
- `low`：只有单一弱信号
- `needs_human_review`：语义和 pHash 冲突、候选分数接近、ELA 明显异常等
- `unknown`：图库为空、检索失败或 embedding 生成失败

## 可选 Qwen3-VL-Plus 配置

外部多模态安全复核通过环境变量启用。当前推荐提供方是阿里云 Model Studio 的 `qwen3-vl-plus`：

```powershell
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
$env:NET_SEC_QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:NET_SEC_QWEN_MODEL="qwen3-vl-plus"
```

可选参数：

```powershell
$env:NET_SEC_QWEN_TIMEOUT_SECONDS="30"
$env:NET_SEC_QWEN_ENABLE_THINKING="false"
$env:NET_SEC_QWEN_VL_HIGH_RESOLUTION_IMAGES="false"
```

兼容的 OpenAI-compatible 变量仍然保留：

```powershell
$env:NET_SEC_LLM_API_KEY="your_api_key"
$env:NET_SEC_LLM_BASE_URL="https://api.openai.com/v1"
$env:NET_SEC_LLM_MODEL="gpt-4.1-mini"
```

如果两组变量同时存在，`NET_SEC_QWEN_*` 优先于 `NET_SEC_LLM_*`。不要把真实 API Key 写入代码、README 或启动脚本。

## 溯源存储配置

```powershell
$env:NET_SEC_SQLITE_PATH="C:\Code\net_sec\data\sqlite\source_engine.db"
$env:NET_SEC_SOURCE_ASSETS_ROOT="E:\net_sec_assets"
$env:NET_SEC_SEMANTIC_EMBEDDING_STORE_DIR="E:\net_sec_data\semantic_embeddings"
$env:NET_SEC_LOCAL_CLIP_MODEL_NAME="ViT-B/32"
$env:NET_SEC_SEMANTIC_RETRIEVAL_TOP_K="5"
```

默认情况下：
- 原图归档和 ELA 图保存在 `E:\net_sec_assets`
- CLIP embedding `.npy` 文件保存在 `E:\net_sec_data\semantic_embeddings`
- SQLite 保存归档资产、分析记录和 embedding 元数据
- 临时上传、热力图和报告仍保存在项目内 `outputs/`

## E 盘移动硬盘布局

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

当前用途：
- `E:\net_sec_assets\images`：归档上传原图
- `E:\net_sec_assets\ela`：生成的 ELA 图片
- `E:\net_sec_data\source_engine.db`：SQLite 溯源数据库
- `E:\net_sec_data\clip_features`：预留给旧版 CLIP/OpenCLIP 实验
- `E:\net_sec_data\semantic_embeddings`：当前本地 CLIP embedding 文件目录
- `E:\net_sec_data\faiss_index`：预留目录；当前 FAISS 索引在内存中重建，不写 `.index` 文件
- `E:\net_sec_backup`：预留给模型、数据库和环境备份

后端运行期间请保持移动硬盘连接。如果 `E:\` 被拔出，归档、SQLite 写入和 embedding 文件写入会失败；检测流程会尽量返回降级后的 `source_analysis`。

## 推荐模型分工

| 优先级 | 模型/能力 | 使用方式 | 说明 |
| --- | --- | --- | --- |
| 1 | `qwen3-vl-plus` | `NET_SEC_QWEN_MODEL` 默认值，用于外部多模态安全复核 | 负责风险解释、证据摘要和审核建议，不负责候选召回 |
| 2 | 本地 `clip` `ViT-B/32` | 默认语义溯源模型，CPU 运行 | 负责图片向量生成，结合 FAISS 召回候选来源 |
| 3 | `gpt-4.1-mini` | `NET_SEC_LLM_MODEL` 兼容选项 | 切换到其他 OpenAI-compatible 供应商时可用 |
| 4 | `gpt-4.1` | 质量优先时使用 | 更适合边界样例和复杂证据解释 |
| 5 | `qwen3-vl-embedding` | 后续远程 embedding 路径预留 | 当前实现不依赖它 |
| 6 | `omni-moderation-latest` | 后续可作为独立 moderation 预检 | 不应配置到 `NET_SEC_QWEN_MODEL` 或 `NET_SEC_LLM_MODEL` |

当前证据链分层：

```text
sha256/归档记录 > 本地 CLIP + FAISS 候选召回 > pHash 距离 > EXIF/ELA > Qwen3-VL-Plus 复核解释
```

## 启动后端 API

推荐命令：

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend_e_drive.ps1
```

不使用 E 盘时，可以手动启动：

```powershell
cd C:\Code\net_sec
$env:NET_SEC_QWEN_API_KEY="your_dashscope_api_key"
.\.conda\image_violation\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

可用接口：
- `GET /api/health`
- `POST /api/detect`
- `POST /api/detect/batch`

后端暴露：
- `/outputs`：上传文件、热力图和报告
- `/source-assets`：归档溯源图片和 ELA 图片

## 启动前端

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd install
C:\nvm\v20.18.0\npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Vite 开发服务器会把 `/api`、`/outputs` 和 `/source-assets` 代理到 `http://127.0.0.1:8000`。

打开：
- 前端：`http://127.0.0.1:5173`
- 后端文档：`http://127.0.0.1:8000/docs`

## 单图检测流程

1. 连接 `E:\` 移动硬盘。
2. 使用 `scripts\start_backend_e_drive.ps1` 启动后端。
3. 启动前端。
4. 打开 `http://127.0.0.1:5173`。
5. 上传图片。
6. 选择解释方法：`gradcam`、`saliency` 或 `integrated_gradients`。
7. 点击检测。
8. 在结果页查看：
   - 分类摘要
   - 各类别分数
   - 原图和热力图
   - 溯源证据链卡片
   - 大模型安全复核卡片

溯源证据链卡片是主要的来源归因界面；大模型复核卡片适合作为审核辅助，不应作为唯一判定依据。

## 手工冒烟测试

单图预测：

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe scripts\manual\test_mobilenet_model.py
```

批量解释：

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe scripts\manual\batch_explain_nsfw.py --method saliency
```

语义检索依赖检查：

```powershell
cd C:\Code\net_sec
.\.conda\image_violation\python.exe -c "import clip, torch, faiss, numpy as np; idx=faiss.IndexFlatIP(3); x=np.array([[1,0,0]], dtype='float32'); idx.add(x); print(idx.search(x, 1)[0][0][0])"
```

前端生产构建：

```powershell
cd C:\Code\net_sec\frontend\presentation_layer
C:\nvm\v20.18.0\npm.cmd run build
```

## 第三方开源项目说明

本项目基于多个第三方开源项目和模型/工具生态构建。公开发布、重新分发模型文件或打包项目时，请保留原项目的许可证和署名要求。

- [GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)：本地 NSFW 分类流程和 Grad-CAM-ready 重建流程所使用的 MobileNet/TensorFlow NSFW 模型基础来源。
- TensorFlow / Keras：用于模型加载、推理和基于梯度的可视化解释。
- OpenAI CLIP / `clip`：用于本地图像 embedding 生成和语义溯源召回。
- FAISS：用于 CLIP embedding 候选的内存向量检索。
- `ImageHash`：用于 pHash 感知哈希比较。
- `ExifRead`：用于 EXIF 元数据提取。
- FastAPI / Uvicorn / Pydantic：用于后端 API、请求解析和响应结构。
- Vue 3、Vite、Element Plus、Vue Router：用于前端界面和本地开发构建。
- 阿里云 Model Studio Qwen3-VL-Plus 或其他 OpenAI-compatible 多模态供应商：配置 API Key 后用于可选的大模型安全复核。

模型权重、生成输出、本地测试样例图片和运行时数据库已通过 `.gitignore` 排除在 Git 之外。如需分享模型权重，建议使用 GitHub Releases、Git LFS 或明确的下载脚本，并同时附带上游许可证/声明文件。

## 说明

- 默认检测模型是 `rebuild/nsfw_mobilenetv2_gradcam_ready.h5`。
- 当前前端主要支持单图检测流程。
- 批量检测后端接口已存在，但批量前端页面尚未实现。
- pHash 是感知哈希信号，只作为证据项展示，不能单独等同于来源证明。
- CLIP 相似度用于候选召回，也不能单独给出高可信归因。
- ELA 篡改评分只对 JPEG 类输入启用，PNG、WebP 等格式会返回降级提示。
- EXIF 输出会做摘要和白名单过滤，不建议直接暴露完整敏感元数据。
- 大模型复核需要外部多模态 API Key，可能增加接口耗时。
