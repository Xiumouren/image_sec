<template>
  <section class="page-grid upload-layout">
    <div class="hero-panel panel">
      <p class="eyebrow">Live Demo</p>
      <h2>单图检测演示流程</h2>
      <p class="hero-copy">
        当前页面已接入真实后端接口。上传图片后会调用模型检测，并在结果页展示分类结果与热力图解释。
      </p>
      <el-alert
        title="当前联调范围为单图检测，批量能力仅保留服务层预留。"
        type="info"
        :closable="false"
        show-icon
      />
    </div>

    <HealthStatusBadge v-if="health" :health="health" />

    <ImageUploadPanel :file="store.state.selectedFile" @change="store.setSelectedFile" />

    <MethodSelector
      :model-value="store.state.selectedMethod"
      @update:model-value="store.setSelectedMethod"
    />

    <section class="panel action-panel">
      <div class="panel-head">
        <h2>提交检测</h2>
        <span class="muted">请求后端 `/api/detect`</span>
      </div>
      <el-alert
        v-if="store.state.error"
        :title="store.state.error"
        type="error"
        :closable="false"
        show-icon
        class="inline-alert"
      />
      <el-button
        type="primary"
        size="large"
        :loading="isSubmitting"
        @click="submit"
      >
        开始检测
      </el-button>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted } from "vue";
import { useRouter } from "vue-router";

import HealthStatusBadge from "../components/HealthStatusBadge.vue";
import ImageUploadPanel from "../components/ImageUploadPanel.vue";
import MethodSelector from "../components/MethodSelector.vue";
import { detectImage, getHealthStatus } from "../services/detectionApi";
import { useDemoStore } from "../store/demoStore";

const store = useDemoStore();
const router = useRouter();

const health = computed(() => store.state.health);
const isSubmitting = computed(() => store.state.requestStatus === "submitting");

onMounted(async () => {
  if (store.state.health) {
    return;
  }
  try {
    store.setRequestStatus("health-loading");
    const payload = await getHealthStatus();
    store.setHealth(payload);
  } catch (error) {
    store.setHealth({
      status: "degraded",
      model_loaded: false,
      model_path: "Unavailable",
      labels: []
    });
    store.setError(error instanceof Error ? error.message : "无法获取服务状态。");
  } finally {
    if (store.state.requestStatus === "health-loading") {
      store.setRequestStatus("idle");
    }
  }
});

async function submit() {
  store.setError("");
  store.setRequestStatus("submitting");
  try {
    const payload = await detectImage({
      file: store.state.selectedFile,
      method: store.state.selectedMethod
    });
    store.setResult(payload);
    store.setRequestStatus("success");
    router.push("/result");
  } catch (error) {
    store.setRequestStatus("error");
    store.setError(error instanceof Error ? error.message : "检测请求失败。");
  }
}
</script>
