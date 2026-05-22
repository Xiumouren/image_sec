from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import urllib.error
import urllib.request

from PIL import Image

from backend.shared.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_VL_HIGH_RESOLUTION_IMAGES,
)


RISK_LEVELS = {"low", "medium", "high", "critical"}
CREDIBILITY_LEVELS = {"low", "medium", "high", "unknown"}


class LlmSafetyReviewService:
    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        enable_thinking: bool = LLM_ENABLE_THINKING,
        vl_high_resolution_images: bool = LLM_VL_HIGH_RESOLUTION_IMAGES,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking
        self.vl_high_resolution_images = vl_high_resolution_images

    def analyze(self, image_path: Path, detection_payload: dict, source_analysis: dict) -> dict[str, object]:
        if not self.api_key:
            return self._error_result("NET_SEC_QWEN_API_KEY or NET_SEC_LLM_API_KEY is not configured.")

        try:
            payload = self._build_request_payload(image_path, detection_payload, source_analysis)
            response_payload = self._post_chat_completion(payload)
            content = self._extract_message_content(response_payload)
            return self._parse_review_content(content)
        except Exception as exc:
            return self._error_result(f"LLM safety review failed: {exc}")

    def _build_request_payload(self, image_path: Path, detection_payload: dict, source_analysis: dict) -> dict[str, object]:
        evidence_payload = {
            "detection": {
                "predicted_class_index": detection_payload.get("predicted_class_index"),
                "predicted_label": detection_payload.get("predicted_label"),
                "scores": detection_payload.get("scores", {}),
                "explanation_method": (detection_payload.get("explanation") or {}).get("method"),
                "explanation_error": detection_payload.get("error"),
            },
            "source_analysis": source_analysis,
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a multimodal safety reviewer for an image moderation and source-analysis demo. "
                        "Assess pornographic risk and source credibility from the image and structured evidence. "
                        "Do not overstate ELA, EXIF, pHash, or heatmap evidence. Treat them as supporting signals, not proof. "
                        "Return only valid JSON with the requested keys."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review this image for pornographic safety risk and source credibility. "
                                "Use this JSON evidence from the local detector and source module:\n"
                                f"{json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
                                "Return JSON with keys: content_risk_level, pornographic_assessment, "
                                "source_credibility_level, source_credibility_assessment, evidence_summary, "
                                "review_recommendation, limitations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_data_url(image_path)},
                        },
                    ],
                },
            ],
        }
        payload.update(self._provider_extra_body())
        return payload

    def _provider_extra_body(self) -> dict[str, object]:
        is_qwen = "dashscope" in self.base_url.lower() or self.model.lower().startswith("qwen")
        if not is_qwen:
            return {}
        extra_body: dict[str, object] = {"enable_thinking": self.enable_thinking}
        if self.vl_high_resolution_images:
            extra_body["vl_high_resolution_images"] = True
        return extra_body

    def _post_chat_completion(self, payload: dict[str, object]) -> dict:
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _extract_message_content(response_payload: dict) -> str:
        choices = response_payload.get("choices") or []
        if not choices:
            raise ValueError("LLM API response did not include choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM API response did not include text content.")
        return content

    def _parse_review_content(self, content: str) -> dict[str, object]:
        parsed = json.loads(self._strip_json_fence(content))
        content_risk_level = self._coerce_choice(parsed.get("content_risk_level"), RISK_LEVELS, "unknown")
        if content_risk_level == "unknown":
            content_risk_level = "medium"
        source_credibility_level = self._coerce_choice(
            parsed.get("source_credibility_level"),
            CREDIBILITY_LEVELS,
            "unknown",
        )
        return {
            "content_risk_level": content_risk_level,
            "pornographic_assessment": str(parsed.get("pornographic_assessment") or ""),
            "source_credibility_level": source_credibility_level,
            "source_credibility_assessment": str(parsed.get("source_credibility_assessment") or ""),
            "evidence_summary": self._coerce_string_list(parsed.get("evidence_summary")),
            "review_recommendation": str(parsed.get("review_recommendation") or ""),
            "limitations": self._coerce_string_list(parsed.get("limitations")),
            "error": None,
        }

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.thumbnail((1024, 1024))
            buffer = BytesIO()
            rgb_image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _coerce_choice(value: object, allowed: set[str], default: str) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in allowed else default

    @staticmethod
    def _coerce_string_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value]
        return []

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    @staticmethod
    def _error_result(message: str) -> dict[str, object]:
        return {
            "content_risk_level": "medium",
            "pornographic_assessment": "",
            "source_credibility_level": "unknown",
            "source_credibility_assessment": "",
            "evidence_summary": [],
            "review_recommendation": "",
            "limitations": [],
            "error": message,
        }
