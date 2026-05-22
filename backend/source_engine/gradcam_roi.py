from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.shared.config import (
    GRADCAM_ROI_EXPANSION_RATIO,
    GRADCAM_ROI_FALLBACK_THRESHOLD,
    GRADCAM_ROI_HEATMAP_PERCENTILE,
    GRADCAM_ROI_MASK_THRESHOLD,
    GRADCAM_ROI_MIN_COVERAGE_RATIO,
    OUTPUTS_DIR,
    SOURCE_ROI_DIR,
)


@dataclass(frozen=True)
class GradcamRoiAsset:
    bbox: dict[str, int]
    coverage_ratio: float
    mask_threshold: float
    fallback_used: bool
    mask_coverage_ratio: float
    crop_path: Path
    output_relative_path: str


class GradcamRoiService:
    def __init__(
        self,
        output_root: Path = SOURCE_ROI_DIR,
        heatmap_percentile: float = GRADCAM_ROI_HEATMAP_PERCENTILE,
        mask_threshold: float = GRADCAM_ROI_MASK_THRESHOLD,
        fallback_threshold: float = GRADCAM_ROI_FALLBACK_THRESHOLD,
        min_coverage_ratio: float = GRADCAM_ROI_MIN_COVERAGE_RATIO,
        expansion_ratio: float = GRADCAM_ROI_EXPANSION_RATIO,
    ) -> None:
        self.output_root = Path(output_root)
        self.heatmap_percentile = heatmap_percentile
        self.mask_threshold = mask_threshold
        self.fallback_threshold = fallback_threshold
        self.min_coverage_ratio = min_coverage_ratio
        self.expansion_ratio = expansion_ratio

    def extract_roi(
        self,
        *,
        image_path: Path,
        heatmap: np.ndarray,
        target_label: str,
    ) -> dict[str, object]:
        image_path = Path(image_path)
        try:
            roi_asset = self._extract_roi_asset(
                image_path=image_path,
                heatmap=np.asarray(heatmap, dtype=np.float32),
                target_label=target_label,
            )
        except Exception as exc:
            return {
                "available": False,
                "target_label": target_label,
                "bbox": None,
                "coverage_ratio": 0.0,
                "mask_threshold": None,
                "fallback_used": False,
                "mask_coverage_ratio": 0.0,
                "crop_relative_path": "",
                "crop_path": "",
                "error": f"Grad-CAM ROI extraction failed: {exc}",
            }

        return {
            "available": True,
            "target_label": target_label,
            "bbox": roi_asset.bbox,
            "coverage_ratio": roi_asset.coverage_ratio,
            "mask_threshold": roi_asset.mask_threshold,
            "fallback_used": roi_asset.fallback_used,
            "mask_coverage_ratio": roi_asset.mask_coverage_ratio,
            "crop_relative_path": roi_asset.output_relative_path,
            "crop_path": str(roi_asset.crop_path),
            "error": None,
        }

    def _extract_roi_asset(
        self,
        *,
        image_path: Path,
        heatmap: np.ndarray,
        target_label: str,
    ) -> GradcamRoiAsset:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            resized_heatmap = self._resize_heatmap(heatmap, width=width, height=height)
            bbox, coverage_ratio, mask_threshold, fallback_used = self._bbox_from_heatmap(resized_heatmap)
            expanded_bbox = self._expand_bbox(bbox, width=width, height=height)
            crop = rgb_image.crop(
                (
                    expanded_bbox["x_min"],
                    expanded_bbox["y_min"],
                    expanded_bbox["x_max"],
                    expanded_bbox["y_max"],
                )
            )
            output_path = self._build_output_path(image_path, target_label)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)

        return GradcamRoiAsset(
            bbox=expanded_bbox,
            coverage_ratio=coverage_ratio,
            mask_threshold=mask_threshold,
            fallback_used=fallback_used,
            mask_coverage_ratio=coverage_ratio,
            crop_path=output_path,
            output_relative_path=self._relative_output_path(output_path),
        )

    @staticmethod
    def _resize_heatmap(heatmap: np.ndarray, *, width: int, height: int) -> np.ndarray:
        if heatmap.ndim != 2:
            raise ValueError("Grad-CAM heatmap must be a 2D array.")
        max_value = float(np.max(heatmap))
        if max_value <= 0.0:
            raise ValueError("Grad-CAM heatmap is empty.")
        normalized = np.clip(heatmap / max_value, 0.0, 1.0)
        heatmap_image = Image.fromarray(np.uint8(normalized * 255.0), mode="L")
        resized = heatmap_image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _bbox_from_heatmap(self, heatmap: np.ndarray) -> tuple[dict[str, int], float, float, bool]:
        threshold = self._clamp_threshold(self.mask_threshold)
        mask = self._postprocess_mask(heatmap >= threshold)
        coverage_ratio = float(np.count_nonzero(mask) / mask.size)

        fallback_used = False
        if coverage_ratio < self.min_coverage_ratio:
            fallback_used = True
            threshold = self._clamp_threshold(self.fallback_threshold)
            mask = self._postprocess_mask(heatmap >= threshold)
            coverage_ratio = float(np.count_nonzero(mask) / mask.size)

        if coverage_ratio < self.min_coverage_ratio:
            raise ValueError(
                f"Grad-CAM hot area is too small: {coverage_ratio:.4f} < {self.min_coverage_ratio:.4f}."
            )

        y_indices, x_indices = np.where(mask)
        if not len(x_indices) or not len(y_indices):
            raise ValueError("Grad-CAM heatmap did not produce a usable hot area.")
        return (
            {
                "x_min": int(np.min(x_indices)),
                "y_min": int(np.min(y_indices)),
                "x_max": int(np.max(x_indices)) + 1,
                "y_max": int(np.max(y_indices)) + 1,
            },
            coverage_ratio,
            threshold,
            fallback_used,
        )

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        binary_mask = np.asarray(mask, dtype=bool)
        if not np.any(binary_mask):
            return binary_mask
        radius = 2
        closed = self._erode_mask(self._dilate_mask(binary_mask, radius=radius), radius=radius)
        return self._dilate_mask(closed, radius=1)

    @staticmethod
    def _dilate_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        result = np.zeros_like(mask, dtype=bool)
        for y_offset in range(radius * 2 + 1):
            for x_offset in range(radius * 2 + 1):
                result |= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
        return result

    @staticmethod
    def _erode_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        result = np.ones_like(mask, dtype=bool)
        for y_offset in range(radius * 2 + 1):
            for x_offset in range(radius * 2 + 1):
                result &= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
        return result

    def _expand_bbox(self, bbox: dict[str, int], *, width: int, height: int) -> dict[str, int]:
        box_width = max(1, bbox["x_max"] - bbox["x_min"])
        box_height = max(1, bbox["y_max"] - bbox["y_min"])
        pad_x = int(round(box_width * self.expansion_ratio))
        pad_y = int(round(box_height * self.expansion_ratio))
        return {
            "x_min": max(0, bbox["x_min"] - pad_x),
            "y_min": max(0, bbox["y_min"] - pad_y),
            "x_max": min(width, bbox["x_max"] + pad_x),
            "y_max": min(height, bbox["y_max"] + pad_y),
        }

    def _build_output_path(self, image_path: Path, target_label: str) -> Path:
        safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in target_label)
        return self.output_root / f"{image_path.stem}_{safe_label}_roi.jpg"

    @staticmethod
    def _relative_output_path(output_path: Path) -> str:
        try:
            return output_path.resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()
        except ValueError:
            return output_path.as_posix()
