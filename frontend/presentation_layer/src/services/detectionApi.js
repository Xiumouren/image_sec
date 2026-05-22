import { request } from "./apiClient";

export function getHealthStatus() {
  return request("/api/health");
}

export function detectImage({ file, method, igSteps = 16 }) {
  if (!file) {
    throw new Error("请先选择图片文件。");
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("method", method);
  formData.append("ig_steps", String(igSteps));

  return request("/api/detect", {
    method: "POST",
    body: formData
  });
}

export function detectBatch({ files, method, igSteps = 16 }) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("method", method);
  formData.append("ig_steps", String(igSteps));

  return request("/api/detect/batch", {
    method: "POST",
    body: formData
  });
}

export function buildOutputsAssetUrl(relativePath) {
  if (!relativePath) {
    return "";
  }
  const normalizedRelativePath = normalizeAssetPath(relativePath);
  return normalizedRelativePath.startsWith("/")
    ? normalizedRelativePath
    : `/outputs/${normalizedRelativePath}`;
}

export function buildSourceAssetUrl(relativePath) {
  if (!relativePath) {
    return "";
  }
  const normalizedRelativePath = normalizeAssetPath(relativePath);
  return normalizedRelativePath.startsWith("/")
    ? normalizedRelativePath
    : `/source-assets/${normalizedRelativePath}`;
}

function normalizeAssetPath(value) {
  return String(value).replaceAll("\\", "/");
}
