import {
  detectionSuccessMock,
  healthSuccessMock
} from "../mocks/demoData";

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function getHealthStatus() {
  await sleep(250);
  return clone(healthSuccessMock);
}

export async function detectImage({ file, method }) {
  await sleep(600);

  if (!file) {
    throw new Error("请先选择图片文件。");
  }

  return {
    ...clone(detectionSuccessMock),
    image_name: file.name,
    explanation: {
      ...clone(detectionSuccessMock).explanation,
      method
    }
  };
}
