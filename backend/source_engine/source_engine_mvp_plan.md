# 溯源能力 MVP 初次设计方案

## 1. 目标
在现有 NSFW 分类、热力图解释、FastAPI 后端和 Vue 前端基础上，补齐第一版溯源闭环。  
这一版只做最小可用方案，不追求图库平台化，也不接入复杂向量检索。

目标能力：
- pHash 相似图比对
- EXIF 元数据提取
- ELA 篡改分析
- SQLite 持久化检测记录
- 前端结果页展示溯源摘要

## 2. 范围与原则
第一版范围固定为：
- `pHash + EXIF + ELA + SQLite + 文件系统`

明确不做：
- CLIP 向量检索
- FAISS
- 外部 URL 色情图库初始化
- MySQL
- 历史页和图库页

设计原则：
- 单次上传只调用一次 `/api/detect`
- 分类、热力图、溯源结果一次性返回
- 任一溯源子模块失败，不阻断整体检测结果
- 重文件资产和数据库分离

## 3. 存储设计
### 3.1 数据库存储
第一版使用 SQLite，数据库文件仍放在项目目录内，便于开发和迁移。

数据库负责保存：
- 检测记录
- 图片资产记录
- pHash 指纹
- EXIF 摘要
- ELA 摘要结果

### 3.2 文件系统存储
`E:\` 作为重文件资产根目录，第一版承载：
- 历史原图
- ELA 可视化图片

建议目录结构：
```text
E:\net_sec_assets\
  images\
  ela\
```

项目目录下的 `outputs/` 继续保留：
- 上传临时文件
- 热力图
- 报告

## 4. Python 后端模块设计
### 4.1 source_engine
- `backend/source_engine/phash/phash_service.py`
  - 计算图片 pHash
  - 比较 Hamming 距离
  - 输出最相似历史图摘要

- `backend/source_engine/exif/exif_service.py`
  - 提取 EXIF 原始字段
  - 生成可读摘要

- `backend/source_engine/ela/ela_service.py`
  - 生成 ELA 图
  - 计算异常比例
  - 输出是否疑似篡改

- `backend/source_engine/source_orchestrator.py`
  - 串联执行 `pHash + EXIF + ELA`
  - 汇总统一响应结构

### 4.2 storage_layer
- `backend/storage_layer/sqlite/db.py`
  - SQLite 初始化和连接管理

- `backend/storage_layer/sqlite/models.py`
  - 定义检测记录、图片资产、哈希与摘要模型

- `backend/storage_layer/sqlite/repository.py`
  - 封装查询、写入、历史匹配

- `backend/storage_layer/filesystem/`
  - 统一管理 `E:\net_sec_assets\images` 和 `E:\net_sec_assets\ela`

## 5. 数据流设计
上传图片后，`/api/detect` 的执行流程为：

1. 保存上传图到当前上传目录
2. 执行现有 NSFW 分类
3. 执行热力图生成
4. 将图片归档或复制到 `E:\net_sec_assets\images\`
5. 执行 `source_orchestrator.analyze(image_path)`
6. 使用 pHash 和历史归档图做相似图比对
7. 提取 EXIF
8. 生成 ELA 图并写入 `E:\net_sec_assets\ela\`
9. 将检测结果和溯源摘要写入 SQLite
10. 将分类结果、热力图、溯源结果一次性返回前端

## 6. 接口设计
### 6.1 继续使用现有接口
仍使用：
- `POST /api/detect`

不新增独立溯源分析接口，避免前端二次请求。

### 6.2 响应结构扩展
在现有检测响应中新增：

```json
{
  "source_analysis": {
    "phash": {
      "hash": "ff00aa...",
      "matched": true,
      "top_match": {
        "image_name": "old_001.jpg",
        "relative_path": "uploads/history/old_001.jpg",
        "distance": 4,
        "similarity": 0.9375
      },
      "candidates": []
    },
    "exif": {
      "has_exif": true,
      "summary": {
        "datetime_original": "2024:01:01 10:00:00",
        "make": "Apple",
        "model": "iPhone 13",
        "software": "Photoshop"
      },
      "raw": {}
    },
    "ela": {
      "is_tampered": true,
      "anomaly_ratio": 8.4,
      "max_error_level": 37,
      "description": "检测到局部篡改痕迹",
      "output_relative_path": "source/ela/example.jpg"
    }
  }
}
```

要求：
- `phash` 提供是否命中历史图和最相似摘要
- `exif` 提供摘要字段和原始字段
- `ela` 提供是否疑似篡改、异常比例和 ELA 图路径
- 任一子模块失败时，应返回错误字段，而不是整单失败

## 7. 前端调用与展示
### 7.1 调用方式
上传页保持不变，仍只调用一次：
- `POST /api/detect`

### 7.2 结果页展示
在现有结果页新增一个 `SourceAnalysisCard`，包含三块：

1. 相似图摘要
- 是否命中历史图
- 最相似图片名
- Hamming 距离
- 相似度

2. EXIF 信息
- 拍摄时间
- 设备品牌 / 型号
- 软件字段
- 是否存在编辑痕迹提示

3. ELA 分析
- 是否疑似篡改
- 异常比例
- 文字描述
- ELA 图预览

展示原则：
- 第一版只展示摘要，不做复杂交互
- 子模块失败时，只在对应区域显示降级提示
- 结果页整体仍保持可用

## 8. 测试要求
### 8.1 后端
- 首次上传图片：
  - `phash.matched = false`
  - EXIF/ELA 正常返回或明确降级

- 再次上传同图：
  - `phash.matched = true`
  - 可命中历史图片

- 无 EXIF 图片：
  - `has_exif = false`

- 可编辑或压缩过的图片：
  - ELA 返回异常比例和可视化图

- 子模块失败：
  - `/api/detect` 不应整体 500
  - 分类与热力图仍需返回

### 8.2 存储
- 上传后：
  - 原图落到 `E:\net_sec_assets\images\`
  - ELA 图落到 `E:\net_sec_assets\ela\`
  - SQLite 中存在检测记录

### 8.3 前端
- 结果页可展示 `source_analysis`
- ELA 图可正常访问
- 子模块失败时能展示降级提示，不白屏

## 9. 第一版默认决策
- 比对基准：只对历史上传图做 pHash 比对
- 存储后端：SQLite + 文件系统
- `E:\` 用途：历史原图 + ELA 图
- 不做图库初始化
- 不做 CLIP / FAISS / MySQL
- 不做历史记录页与图库页

## 10. 后续扩展路线
第一版完成后，再按优先级逐步扩展：

1. 历史记录页
2. 外部图库导入脚本
3. CLIP 特征提取
4. FAISS 相似图检索
5. MySQL 或远端部署
