<template>
  <section class="panel">
    <div class="panel-head">
      <h2>图片上传</h2>
      <span class="muted">支持 jpg、jpeg、png、bmp、webp</span>
    </div>
    <el-upload
      class="upload-box"
      drag
      action="#"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleChange"
      :limit="1"
      accept=".jpg,.jpeg,.png,.bmp,.webp"
    >
      <el-icon class="upload-icon"><UploadFilled /></el-icon>
      <div class="el-upload__text">拖拽图片到这里，或 <em>点击上传</em></div>
    </el-upload>
    <div v-if="previewUrl" class="preview-block">
      <img :src="previewUrl" alt="preview" class="preview-image" />
      <div class="preview-meta">
        <strong>{{ file?.name }}</strong>
        <span class="muted">{{ fileSize }}</span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { UploadFilled } from "@element-plus/icons-vue";

const props = defineProps({
  file: {
    type: Object,
    default: null
  }
});

const emit = defineEmits(["change"]);

const previewUrl = ref("");

function revokePreview() {
  if (previewUrl.value.startsWith("blob:")) {
    URL.revokeObjectURL(previewUrl.value);
  }
}

watch(
  () => props.file,
  (nextFile) => {
    revokePreview();
    previewUrl.value = nextFile ? URL.createObjectURL(nextFile) : "";
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  revokePreview();
});

function handleChange(uploadFile) {
  emit("change", uploadFile.raw || null);
}

const fileSize = computed(() => {
  if (!props.file?.size) {
    return "";
  }
  return `${(props.file.size / 1024 / 1024).toFixed(2)} MB`;
});
</script>
