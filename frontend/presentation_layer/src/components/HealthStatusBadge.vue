<template>
  <section class="panel">
    <div class="panel-head">
      <h2>服务状态</h2>
      <el-tag :type="tagType" effect="dark">{{ statusLabel }}</el-tag>
    </div>
    <div class="kv-grid">
      <div>
        <span class="kv-label">状态</span>
        <strong>{{ health.status }}</strong>
      </div>
      <div>
        <span class="kv-label">模型已加载</span>
        <strong>{{ health.model_loaded ? "是" : "否" }}</strong>
      </div>
    </div>
    <p class="muted path-line">{{ health.model_path }}</p>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  health: {
    type: Object,
    required: true
  }
});

const tagType = computed(() => (props.health.model_loaded ? "success" : "danger"));
const statusLabel = computed(() => (props.health.model_loaded ? "系统正常" : "状态异常"));
</script>
