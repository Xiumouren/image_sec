from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from backend.shared.config import ELA_JPEG_QUALITY, ELA_PIXEL_THRESHOLD, ELA_TAMPER_RATIO_THRESHOLD


class ElaService:
    def __init__(
        self,
        jpeg_quality: int = ELA_JPEG_QUALITY,
        pixel_threshold: int = ELA_PIXEL_THRESHOLD,
        tamper_ratio_threshold: float = ELA_TAMPER_RATIO_THRESHOLD,
    ) -> None:
        self.jpeg_quality = jpeg_quality
        self.pixel_threshold = pixel_threshold
        self.tamper_ratio_threshold = tamper_ratio_threshold

    def analyze(
        self,
        image_path: Path,
        output_path: Path,
        output_relative_path: str,
        output_url: str,
    ) -> dict[str, object]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = image_path.suffix.lower()
        if suffix not in {".jpg", ".jpeg"}:
            return {
                "supported_for_detection": False,
                "unsupported_reason": "ELA tamper scoring is only reliable for JPEG-like inputs.",
                "is_tampered": False,
                "anomaly_ratio": 0.0,
                "max_error_level": 0,
                "description": "ELA scoring was skipped because the input is not JPEG-like.",
                "output_relative_path": "",
                "output_url": "",
                "error": None,
            }

        with Image.open(image_path) as original_image:
            original_rgb = original_image.convert("RGB")
            buffer = BytesIO()
            original_rgb.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            recompressed = Image.open(buffer).convert("RGB")
            difference = ImageChops.difference(original_rgb, recompressed)

        max_error_level = max(channel[1] for channel in difference.getextrema())
        brightness_scale = 255.0 / max(1, max_error_level)
        ela_image = ImageEnhance.Brightness(difference).enhance(brightness_scale)
        ela_image.save(output_path, format="JPEG")

        intensity_map = np.asarray(difference, dtype=np.uint8).max(axis=2)
        anomaly_ratio = float((intensity_map >= self.pixel_threshold).mean() * 100.0)
        is_tampered = bool(anomaly_ratio >= self.tamper_ratio_threshold)

        if is_tampered:
            description = "Detected localized compression anomalies that may indicate editing."
        else:
            description = "No strong localized compression anomalies were detected."

        return {
            "supported_for_detection": True,
            "unsupported_reason": None,
            "is_tampered": is_tampered,
            "anomaly_ratio": round(anomaly_ratio, 4),
            "max_error_level": int(max_error_level),
            "description": description,
            "output_relative_path": output_relative_path,
            "output_url": output_url,
            "error": None,
        }
