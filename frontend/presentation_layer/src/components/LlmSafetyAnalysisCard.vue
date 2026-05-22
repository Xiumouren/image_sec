<template>
  <section class="panel llm-panel">
    <div class="panel-head">
      <h2>大模型安全复核</h2>
      <span class="muted">色情风险、证据链与溯源可信度</span>
    </div>

    <div v-if="!analysis" class="kv-stack">
      <el-alert
        title="当前结果未提供大模型复核。"
        type="info"
        :closable="false"
        show-icon
      />
    </div>

    <div v-else-if="analysis.error" class="kv-stack">
      <el-alert
        :title="analysis.error"
        type="warning"
        :closable="false"
        show-icon
      />
      <div class="placeholder-box compact-placeholder">
        原有分类、热力图和溯源结果仍可用于演示；大模型复核需要配置外部多模态 API。
      </div>
    </div>

    <div v-else class="llm-grid">
      <article class="llm-summary-block">
        <div class="source-card-head">
          <h3>内容风险</h3>
          <el-tag :type="riskTagType" effect="light">{{ riskLabel }}</el-tag>
        </div>
        <p class="review-text">{{ analysis.pornographic_assessment || "未返回色情内容复核结论。" }}</p>
      </article>

      <article class="llm-summary-block">
        <div class="source-card-head">
          <h3>溯源可信度</h3>
          <el-tag :type="credibilityTagType" effect="light">{{ credibilityLabel }}</el-tag>
        </div>
        <p class="review-text">{{ analysis.source_credibility_assessment || "未返回溯源可信度结论。" }}</p>
      </article>

      <article class="llm-summary-block llm-wide-block">
        <h3>证据摘要</h3>
        <ul v-if="analysis.evidence_summary?.length" class="review-list">
          <li v-for="item in analysis.evidence_summary" :key="item">{{ item }}</li>
        </ul>
        <div v-else class="placeholder-box compact-placeholder">未返回关键证据摘要。</div>
      </article>

      <article class="llm-summary-block">
        <h3>审核建议</h3>
        <p class="review-text">{{ analysis.review_recommendation || "未返回审核建议。" }}</p>
      </article>

      <article class="llm-summary-block">
        <h3>不确定性</h3>
        <ul v-if="analysis.limitations?.length" class="review-list">
          <li v-for="item in analysis.limitations" :key="item">{{ item }}</li>
        </ul>
        <div v-else class="placeholder-box compact-placeholder">未返回额外限制说明。</div>
      </article>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  analysis: {
    type: Object,
    default: null
  }
});

const riskLabelMap = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
  critical: "严重风险"
};

const credibilityLabelMap = {
  low: "低可信",
  medium: "中可信",
  high: "高可信",
  unknown: "未知"
};

const riskLabel = computed(() => {
  return riskLabelMap[props.analysis?.content_risk_level] || "未知";
});

const riskTagType = computed(() => {
  const level = props.analysis?.content_risk_level;
  if (level === "critical" || level === "high") {
    return "danger";
  }
  if (level === "medium") {
    return "warning";
  }
  return "success";
});

const credibilityLabel = computed(() => {
  return credibilityLabelMap[props.analysis?.source_credibility_level] || "未知";
});

const credibilityTagType = computed(() => {
  const level = props.analysis?.source_credibility_level;
  if (level === "high") {
    return "success";
  }
  if (level === "medium") {
    return "warning";
  }
  return "info";
});
</script>
