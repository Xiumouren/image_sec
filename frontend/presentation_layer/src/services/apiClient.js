function buildErrorMessage(status, detail) {
  if (detail) {
    return detail;
  }
  if (status >= 500) {
    return "后端服务处理失败，请稍后重试。";
  }
  if (status >= 400) {
    return "请求参数无效，请检查上传文件和解释方法。";
  }
  return "请求失败，请检查后端服务是否已启动。";
}

export async function request(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    throw new Error(buildErrorMessage(response.status, payload?.detail));
  }

  return payload;
}
