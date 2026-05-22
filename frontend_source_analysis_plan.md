# 前端接入 `source_analysis` 结果页计划

## Summary
本轮只做现有结果页的溯源摘要接入，不新增历史页或新路由。目标是把后端已返回的 `source_analysis` 以“作品集可演示、信息不过载”的方式接入前端：用户上传图片后，在同一结果页看到相似图命中、EXIF 摘要、ELA 判断和存储状态，并保留稳定的错误降级展示。

## Key Changes
### 前端结果页
- 在现有结果页新增一个 `SourceAnalysisCard`，按 4 个摘要块展示：
  - `pHash`：是否命中历史图、是否精确命中、最相似图片、相似度或距离
  - `EXIF`：是否存在 EXIF、拍摄时间、设备、软件字段、后处理提示
  - `ELA`：是否支持检测、是否疑似压缩异常、异常比例、ELA 图链接或预览入口
  - `storage`：是否归档成功、是否写库成功、是否复用已有资产
- 默认只展示摘要信息；`candidates` 和 `raw` 不直接平铺，只在折叠区或“技术详情”区域展示。
- 结果页继续保留现有分类、分数、热力图、解释信息模块；溯源卡片放在这些模块之后，不改变主流程。
- 顺手修正当前结果页和数据层里已有的中文乱码文案，避免影响作品集展示。

### 前端数据层
- `detectionApi` 增加 `buildSourceAssetUrl`，统一处理 `/source-assets/*` 路径。
- `demoStore` 继续沿用当前 `result` 持久化方案，不新增状态机；只补辅助 selector/normalizer，把 `source_analysis` 转成适合组件消费的展示数据。
- 新增前端容错映射：
  - `source_analysis` 缺失时，组件显示“未提供溯源结果”
  - 某个子块 `error` 非空时，只降级该子块
  - `ELA.supported_for_detection = false` 时，显示“不支持 JPEG 之外的可靠评分”

### 允许的后端轻量调整
- 只做有利于展示的一次小收敛，不推翻当前契约。
- 推荐补齐或统一以下点：
  - 保证 `source_analysis` 始终完整返回 4 个子块：`phash / exif / ela / storage`
  - 保持 `output_url`、`source_url` 作为前端直接可用的 URL 字段
  - 如前端需要，补一个更直观的 `display_name` 或说明字段，但不改现有字段语义
- 不在这一轮新增历史查询接口、批量溯源接口或新的页面级 API。

## Test Plan
- JPEG 正常上传：
  - 结果页展示分类结果、热力图、`source_analysis` 四块摘要
  - `ELA` 可展示有效状态和图片链接
- 同图重复上传：
  - `pHash.exact_match = true`
  - `storage.asset_reused = true`
  - 前端摘要文案正确
- PNG 上传：
  - `ELA.supported_for_detection = false`
  - 页面显示降级提示，不显示错误白屏
- 子块失败场景：
  - 某一块有 `error` 时，其它块仍正常展示
- 路径验证：
  - `/outputs/*` 继续可显示热力图
  - `/source-assets/*` 可用于 ELA 或相似图链接
- 会话恢复：
  - 刷新结果页后，已有 `sessionStorage` 结果仍可展示溯源摘要

## Assumptions
- 范围固定为“现有结果页摘要接入”，不新增历史页。
- 展示深度采用“摘要优先”，原始 EXIF 和 pHash 候选仅折叠展示或后续再接。
- 允许对后端返回做一次小幅契约收敛，但以前端接入为主，不做新的业务接口扩展。
- 当前视觉基线沿用现有 Vue 结果页，不重做整页布局。
