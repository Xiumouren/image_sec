<template>
  <section class="panel source-panel">
    <div class="panel-head">
      <h2>溯源证据链</h2>
      <el-tag :type="credibilityTagType" effect="light">{{ credibilityLabel }}</el-tag>
    </div>

    <div v-if="!sourceAnalysis" class="kv-stack">
      <el-alert title="当前结果未提供溯源证据链。" type="info" :closable="false" show-icon />
    </div>

    <div v-else class="source-chain">
      <el-alert
        v-if="needsHumanReview"
        title="整图与 ROI 证据存在冲突或候选接近，建议人工复核后再做来源归因。"
        type="warning"
        :closable="false"
        show-icon
      />

      <div class="source-grid">
        <article class="source-card source-card-wide">
          <div class="source-card-head">
            <h3>当前图片</h3>
            <el-tag :type="queryArchived ? 'success' : 'info'" effect="light">
              {{ queryArchived ? "已归档" : "未归档" }}
            </el-tag>
          </div>
          <div class="kv-stack compact-stack">
            <div>
              <span class="kv-label">SHA-256</span>
              <strong class="hash-line">{{ query.sha256 || "未生成" }}</strong>
            </div>
            <div>
              <span class="kv-label">pHash 指纹</span>
              <strong>{{ query.phash || "未生成" }}</strong>
            </div>
            <div>
              <span class="kv-label">整图向量索引</span>
              <strong>{{ embeddingStatus }}</strong>
            </div>
          </div>
        </article>

        <article class="source-card">
          <div class="source-card-head">
            <h3>当前 ROI</h3>
            <el-tag :type="queryRoi.available ? 'success' : 'info'" effect="light">
              {{ queryRoi.available ? "可用" : "不可用" }}
            </el-tag>
          </div>
          <div class="kv-stack compact-stack">
            <div>
              <span class="kv-label">区域摘要</span>
              <strong>{{ roiSummary }}</strong>
            </div>
            <a v-if="queryRoiUrl" :href="queryRoiUrl" class="asset-link" target="_blank" rel="noreferrer">
              查看当前 ROI
            </a>
          </div>
        </article>
      </div>

      <article class="source-card source-card-full">
        <div class="source-card-head">
          <h3>复核建议</h3>
          <el-tag type="info" effect="light">{{ rankingSummary }}</el-tag>
        </div>
        <p class="review-text">{{ sourceAnalysis.review_recommendation || "暂无建议。" }}</p>
      </article>

      <section class="candidate-section">
        <div class="source-card-head">
          <h3>完整原图检索</h3>
          <el-tag effect="light">{{ fullImageCandidates.length }}</el-tag>
        </div>

        <div class="candidate-list">
          <article
            v-for="candidate in fullImageCandidates"
            :key="candidate.asset_id || candidate.source_url || candidate.image_name"
            class="candidate-card"
          >
            <img
              v-if="candidateImageUrl(candidate)"
              :src="candidateImageUrl(candidate)"
              alt="完整源图缩略图"
              class="candidate-thumb"
            />
            <div v-else class="placeholder-box candidate-thumb">无缩略图</div>

            <div class="candidate-body">
              <div class="candidate-head">
                <div>
                  <h3>{{ candidate.image_name || "未命名候选" }}</h3>
                  <a
                    v-if="candidateImageUrl(candidate)"
                    :href="candidateImageUrl(candidate)"
                    class="asset-link"
                    target="_blank"
                    rel="noreferrer"
                  >
                    查看完整源图
                  </a>
                </div>
                <el-tag :type="candidateTagType(candidate)" effect="light">
                  {{ levelLabel(candidate.credibility_level) }}
                </el-tag>
              </div>

              <div class="candidate-metrics">
                <span>整图语义：{{ formatSimilarity(candidate.semantic_similarity) }}</span>
                <span>整图 pHash 距离：{{ formatDistance(candidate.phash_distance) }}</span>
                <span>SHA-256：{{ candidate.exact_sha256_match ? "完全一致" : "未完全一致" }}</span>
              </div>

              <div class="evidence-tags">
                <el-tag
                  v-for="item in candidate.evidence || []"
                  :key="`${candidate.asset_id}-${item.type}-${item.score}`"
                  :type="evidenceTagType(item.level)"
                  effect="plain"
                >
                  {{ evidenceLabel(item) }}
                </el-tag>
              </div>
            </div>
          </article>

          <div v-if="!fullImageCandidates.length" class="placeholder-box">
            未召回完整原图候选；整图语义检索、pHash 或 SHA-256 模块可能为空或降级。
          </div>
        </div>
      </section>

      <section class="candidate-section">
        <div class="source-card-head">
          <h3>ROI 局部检索</h3>
          <el-tag effect="light">{{ roiCandidates.length }}</el-tag>
        </div>

        <div class="candidate-list">
          <article
            v-for="candidate in roiCandidates"
            :key="candidate.roi_id || `${candidate.asset_id}-${candidate.roi_url}`"
            class="candidate-card"
          >
            <img
              v-if="candidateRoiUrl(candidate)"
              :src="candidateRoiUrl(candidate)"
              alt="历史 ROI 裁剪图"
              class="candidate-thumb"
            />
            <div v-else class="placeholder-box candidate-thumb">无 ROI 图</div>

            <div class="candidate-body">
              <div class="candidate-head">
                <div>
                  <h3>{{ candidate.image_name || "未命名 ROI 候选" }}</h3>
                  <a
                    v-if="candidateImageUrl(candidate)"
                    :href="candidateImageUrl(candidate)"
                    class="asset-link"
                    target="_blank"
                    rel="noreferrer"
                  >
                    查看对应完整源图
                  </a>
                  <a
                    v-if="candidateRoiUrl(candidate)"
                    :href="candidateRoiUrl(candidate)"
                    class="asset-link asset-link-spaced"
                    target="_blank"
                    rel="noreferrer"
                  >
                    查看历史 ROI
                  </a>
                </div>
                <el-tag :type="candidateTagType(candidate)" effect="light">
                  {{ levelLabel(candidate.credibility_level) }}
                </el-tag>
              </div>

              <div class="candidate-metrics">
                <span>ROI 语义：{{ formatSimilarity(candidate.roi_semantic_similarity) }}</span>
                <span>ROI pHash 距离：{{ formatDistance(candidate.roi_phash_distance) }}</span>
                <span>历史 bbox：{{ formatBBox(candidate.source_bbox) }}</span>
                <span>类别：{{ candidate.target_label || "未提供" }}</span>
              </div>

              <div class="evidence-tags">
                <el-tag
                  v-for="item in candidate.evidence || []"
                  :key="`${candidate.roi_id}-${item.type}-${item.score}`"
                  :type="evidenceTagType(item.level)"
                  effect="plain"
                >
                  {{ evidenceLabel(item) }}
                </el-tag>
              </div>
            </div>
          </article>

          <div v-if="!roiCandidates.length" class="placeholder-box">
            未召回历史 ROI 候选；当前 ROI 不可用、ROI 索引为空或局部检索已降级。
          </div>
        </div>
      </section>

      <div class="source-grid">
        <article class="source-card source-card-wide">
          <div class="source-card-head">
            <h3>证据摘要</h3>
          </div>
          <ul class="review-list">
            <li v-for="item in evidenceSummary" :key="item">{{ item }}</li>
          </ul>
        </article>

        <article v-if="errors.length" class="source-card">
          <div class="source-card-head">
            <h3>降级错误</h3>
            <el-tag type="warning" effect="light">{{ errors.length }}</el-tag>
          </div>
          <ul class="review-list">
            <li v-for="item in errors" :key="item">{{ item }}</li>
          </ul>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

import { buildOutputsAssetUrl, buildSourceAssetUrl } from "../services/detectionApi";

const props = defineProps({
  sourceAnalysis: {
    type: Object,
    default: null
  }
});

const query = computed(() => props.sourceAnalysis?.query ?? {});
const queryRoi = computed(() => query.value?.roi ?? props.sourceAnalysis?.signals?.gradcam_roi ?? {});
const candidateGroups = computed(() => props.sourceAnalysis?.candidates ?? {});
const fullImageCandidates = computed(() => candidateGroups.value?.full_image ?? []);
const roiCandidates = computed(() => candidateGroups.value?.roi ?? []);
const rankingSummary = computed(() => candidateGroups.value?.ranking_summary || "暂无综合判断");
const evidenceSummary = computed(() => props.sourceAnalysis?.evidence_summary ?? []);
const errors = computed(() => props.sourceAnalysis?.errors ?? []);
const credibility = computed(() => props.sourceAnalysis?.source_credibility_level || "unknown");
const queryArchived = computed(() => Boolean(query.value?.archived));
const needsHumanReview = computed(() => credibility.value === "needs_human_review");

const embeddingStatus = computed(() => {
  const embedding = query.value?.embedding || {};
  if (embedding.status === "error") {
    return "生成或检索失败";
  }
  const count = embedding.indexed_asset_count ?? 0;
  const dim = embedding.embedding_dim || "--";
  return `${embedding.model_name || "local CLIP"} / ${dim} 维 / 已索引 ${count} 张`;
});

const roiSummary = computed(() => {
  if (!queryRoi.value?.available) {
    return "未提取可用局部区域";
  }
  const target = queryRoi.value?.target_label || "违规区域";
  const coverage = formatCoverage(queryRoi.value?.coverage_ratio);
  const indexed = queryRoi.value?.embedding?.indexed_roi_count ?? queryRoi.value?.indexed_roi_count ?? 0;
  return `${target} / 覆盖 ${coverage} / ROI 库 ${indexed} 条`;
});

const queryRoiUrl = computed(() => {
  if (queryRoi.value?.roi_url) {
    return buildSourceAssetUrl(queryRoi.value.roi_url);
  }
  if (queryRoi.value?.crop_relative_path) {
    return buildOutputsAssetUrl(queryRoi.value.crop_relative_path);
  }
  return "";
});

const credibilityLabel = computed(() => levelLabel(credibility.value));
const credibilityTagType = computed(() => {
  if (credibility.value === "high") return "danger";
  if (credibility.value === "medium" || credibility.value === "needs_human_review") return "warning";
  return "info";
});

function candidateImageUrl(candidate) {
  return buildSourceAssetUrl(candidate?.source_url || candidate?.relative_path || "");
}

function candidateRoiUrl(candidate) {
  return buildSourceAssetUrl(candidate?.roi_url || candidate?.roi_relative_path || "");
}

function candidateTagType(candidate) {
  const level = candidate?.credibility_level;
  if (level === "high") return "danger";
  if (level === "medium" || level === "needs_human_review") return "warning";
  return "info";
}

function evidenceTagType(level) {
  if (level === "strong") return "danger";
  if (level === "medium") return "warning";
  return "info";
}

function levelLabel(level) {
  const labels = {
    high: "高可信",
    medium: "中等可信",
    low: "低可信",
    unknown: "未知",
    needs_human_review: "需人工复核"
  };
  return labels[level] || "未知";
}

function evidenceLabel(item) {
  const labels = {
    semantic_clip: "CLIP 整图",
    roi_semantic_clip: "ROI CLIP",
    perceptual_hash: "整图 pHash",
    roi_perceptual_hash: "ROI pHash",
    sha256: "SHA-256"
  };
  const score = typeof item?.score === "number" ? ` ${formatSimilarity(item.score)}` : "";
  return `${labels[item?.type] || item?.type || "证据"} · ${item?.level || "weak"}${score}`;
}

function formatSimilarity(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatDistance(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return String(value);
}

function formatCoverage(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatBBox(value) {
  if (!value || typeof value !== "object") {
    return "--";
  }
  return `${value.x_min ?? "?"},${value.y_min ?? "?"} - ${value.x_max ?? "?"},${value.y_max ?? "?"}`;
}
</script>
