<template>
  <section class="panel accent-panel">
    <div class="panel-head">
      <h2>分类结果</h2>
      <el-tag :type="tagType" effect="dark">{{ result.predicted_label }}</el-tag>
    </div>
    <div class="summary-grid">
      <div>
        <span class="kv-label">Top1 类别</span>
        <strong>{{ result.predicted_label }}</strong>
      </div>
      <div>
        <span class="kv-label">类别索引</span>
        <strong>{{ result.predicted_class_index }}</strong>
      </div>
      <div>
        <span class="kv-label">Top1 概率</span>
        <strong>{{ topScore }}</strong>
      </div>
      <div>
        <span class="kv-label">图片名称</span>
        <strong>{{ result.image_name }}</strong>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  result: {
    type: Object,
    required: true
  }
});

const topScore = computed(() => {
  const value = props.result.scores?.[props.result.predicted_label] ?? 0;
  return `${(value * 100).toFixed(2)}%`;
});

const tagType = computed(() => {
  if (props.result.predicted_label === "neutral") {
    return "success";
  }
  if (props.result.predicted_label === "sexy") {
    return "warning";
  }
  return "danger";
});
</script>
