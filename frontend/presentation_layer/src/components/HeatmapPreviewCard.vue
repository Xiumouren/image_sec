<template>
  <section class="panel">
    <div class="panel-head">
      <h2>热力图预览</h2>
      <span class="muted">原图优先使用本地预览，热力图来自后端输出</span>
    </div>
    <div class="image-grid">
      <figure>
        <figcaption>原图</figcaption>
        <img :src="originalSrc" alt="original" class="result-image" />
      </figure>
      <figure>
        <figcaption>热力图</figcaption>
        <template v-if="heatmapSrc && !heatmapError">
          <img :src="heatmapSrc" alt="heatmap" class="result-image" />
        </template>
        <div v-else class="placeholder-box image-placeholder">
          {{ heatmapError || "当前没有可用的热力图输出。" }}
        </div>
      </figure>
    </div>
  </section>
</template>

<script setup>
defineProps({
  originalSrc: {
    type: String,
    required: true
  },
  heatmapSrc: {
    type: String,
    default: ""
  },
  heatmapError: {
    type: String,
    default: ""
  }
});
</script>
