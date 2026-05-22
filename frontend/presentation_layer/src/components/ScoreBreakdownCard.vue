<template>
  <section class="panel">
    <div class="panel-head">
      <h2>五类概率分布</h2>
      <span class="muted">按 `/api/detect.scores` 展示</span>
    </div>
    <div class="score-list">
      <div v-for="item in scoreItems" :key="item.label" class="score-row">
        <div class="score-head">
          <span>{{ item.label }}</span>
          <strong>{{ item.percent }}</strong>
        </div>
        <el-progress :percentage="item.value" :show-text="false" :stroke-width="10" :color="item.color" />
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  scores: {
    type: Object,
    required: true
  }
});

const palette = {
  drawings: "#4e89ff",
  hentai: "#ff7a59",
  neutral: "#43b883",
  porn: "#ff4d67",
  sexy: "#f3b23c"
};

const scoreItems = computed(() =>
  Object.entries(props.scores).map(([label, value]) => ({
    label,
    value: Number((value * 100).toFixed(2)),
    percent: `${(value * 100).toFixed(2)}%`,
    color: palette[label] || "#6b7280"
  }))
);
</script>
