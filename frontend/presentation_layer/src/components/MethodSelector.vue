<template>
  <section class="panel">
    <div class="panel-head">
      <h2>解释方法</h2>
      <span class="muted">字段与后端 `/api/detect` 对齐</span>
    </div>
    <el-radio-group
      :model-value="modelValue"
      @update:model-value="$emit('update:modelValue', $event)"
      class="method-grid"
    >
      <el-radio-button label="gradcam">Grad-CAM</el-radio-button>
      <el-radio-button label="saliency">Saliency</el-radio-button>
      <el-radio-button label="integrated_gradients">Integrated Gradients</el-radio-button>
    </el-radio-group>
  </section>
</template>

<script setup>
defineProps({
  modelValue: {
    type: String,
    required: true
  }
});

defineEmits(["update:modelValue"]);
</script>
