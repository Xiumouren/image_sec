<template>
  <section class="panel">
    <div class="panel-head">
      <h2>解释元信息</h2>
      <span class="muted">当前展示解释方法、目标层和输出位置</span>
    </div>
    <div v-if="explanation" class="kv-stack">
      <div>
        <span class="kv-label">解释方法</span>
        <strong>{{ explanation.method }}</strong>
      </div>
      <div>
        <span class="kv-label">目标层</span>
        <strong>{{ explanation.target_layer }}</strong>
      </div>
      <div>
        <span class="kv-label">输出路径</span>
        <p class="path-line">{{ explanation.output_path }}</p>
      </div>
      <el-alert
        v-if="explanation.error || requestError"
        :title="explanation.error || requestError"
        type="warning"
        :closable="false"
        show-icon
      />
      <div class="placeholder-box">
        这里预留给后续“解释文本”“违规面积占比”“溯源摘要”等增强信息。
      </div>
    </div>
    <div v-else class="kv-stack">
      <el-alert
        :title="requestError || '当前请求已返回分类结果，但没有可展示的解释输出。'"
        type="warning"
        :closable="false"
        show-icon
      />
      <div class="placeholder-box">
        模型分类已完成。解释阶段失败时，这里会显示降级提示，而不是让结果页空白。
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  explanation: {
    type: Object,
    default: null
  },
  requestError: {
    type: String,
    default: ""
  }
});
</script>
