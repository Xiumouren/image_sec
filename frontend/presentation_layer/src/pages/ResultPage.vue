<template>
  <section v-if="store.state.result" class="page-grid result-layout">
    <PredictionSummaryCard :result="store.state.result" />
    <ScoreBreakdownCard :scores="store.state.result.scores" />
    <HeatmapPreviewCard
      :original-src="originalSrc"
      :heatmap-src="heatmapSrc"
      :heatmap-error="heatmapError"
    />
    <ExplanationMetaCard
      :explanation="store.state.result.explanation"
      :request-error="store.state.result.error"
    />
    <SourceAnalysisCard :source-analysis="store.state.result.source_analysis" />
    <LlmSafetyAnalysisCard :analysis="store.state.result.llm_safety_analysis" />
    <section class="panel result-actions">
      <div class="panel-head">
        <h2>下一步</h2>
        <span class="muted">继续保留重新上传入口，便于快速对比结果</span>
      </div>
      <div class="button-row">
        <el-button @click="goBack">重新选择图片</el-button>
        <el-button type="primary" plain disabled>批量页面后续接入</el-button>
      </div>
    </section>
  </section>
  <section v-else class="panel empty-state">
    <h2>暂无检测结果</h2>
    <p class="muted">请先从上传页选择图片并提交检测。</p>
    <el-button type="primary" @click="$router.push('/upload')">返回上传页</el-button>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useRouter } from "vue-router";

import ExplanationMetaCard from "../components/ExplanationMetaCard.vue";
import HeatmapPreviewCard from "../components/HeatmapPreviewCard.vue";
import LlmSafetyAnalysisCard from "../components/LlmSafetyAnalysisCard.vue";
import PredictionSummaryCard from "../components/PredictionSummaryCard.vue";
import ScoreBreakdownCard from "../components/ScoreBreakdownCard.vue";
import SourceAnalysisCard from "../components/SourceAnalysisCard.vue";
import { buildOutputsAssetUrl } from "../services/detectionApi";
import { useDemoStore } from "../store/demoStore";

const store = useDemoStore();
const router = useRouter();
const localPreviewUrl = ref("");

function revokePreview() {
  if (localPreviewUrl.value.startsWith("blob:")) {
    URL.revokeObjectURL(localPreviewUrl.value);
  }
  localPreviewUrl.value = "";
}

watch(
  () => store.state.selectedFile,
  (nextFile) => {
    revokePreview();
    if (nextFile) {
      localPreviewUrl.value = URL.createObjectURL(nextFile);
    }
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  revokePreview();
});

const originalSrc = computed(() => {
  if (localPreviewUrl.value) {
    return localPreviewUrl.value;
  }
  return buildOutputsAssetUrl(store.state.result?.original_image_relative_path || "");
});

const heatmapSrc = computed(() => {
  return buildOutputsAssetUrl(store.state.result?.explanation?.output_relative_path || "");
});

const heatmapError = computed(() => {
  const explanationError = store.state.result?.explanation?.error;
  return explanationError || store.state.result?.error || "";
});

function goBack() {
  store.resetFlow();
  router.push("/upload");
}
</script>
