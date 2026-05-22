import { reactive, readonly } from "vue";

const state = reactive({
  health: null,
  selectedFile: null,
  selectedMethod: "gradcam",
  result: null,
  batchResult: null,
  requestStatus: "idle",
  error: ""
});

export function useDemoStore() {
  function setHealth(payload) {
    state.health = payload;
  }

  function setSelectedFile(file) {
    state.selectedFile = file;
  }

  function setSelectedMethod(method) {
    state.selectedMethod = method;
  }

  function setResult(payload) {
    state.result = payload;
    if (payload) {
      sessionStorage.setItem("nsfw-demo-last-result", JSON.stringify(payload));
    } else {
      sessionStorage.removeItem("nsfw-demo-last-result");
    }
  }

  function setBatchResult(payload) {
    state.batchResult = payload;
  }

  function setRequestStatus(value) {
    state.requestStatus = value;
  }

  function setError(message) {
    state.error = message;
  }

  function resetFlow() {
    state.selectedFile = null;
    state.result = null;
    state.batchResult = null;
    state.requestStatus = "idle";
    state.error = "";
    state.selectedMethod = "gradcam";
    sessionStorage.removeItem("nsfw-demo-last-result");
  }

  return {
    state: readonly(state),
    setHealth,
    setSelectedFile,
    setSelectedMethod,
    setResult,
    setBatchResult,
    setRequestStatus,
    setError,
    resetFlow
  };
}
