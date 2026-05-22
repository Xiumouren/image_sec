from __future__ import annotations

from pathlib import Path

from backend.shared.config import EXPLANATIONS_DIR, OUTPUTS_DIR
from backend.explanation_generation.tensorflow.gradcam_tf import DEFAULT_IG_STEPS, generate_gradcam


class ExplainerService:
    def __init__(self, output_root: Path = EXPLANATIONS_DIR):
        self.output_root = Path(output_root)

    def build_output_path(self, image_name: str, method: str, folder: str = "single") -> Path:
        target_dir = self.output_root / method / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_name).stem
        return target_dir / f"{stem}_{method}.jpg"

    def explain(
        self,
        model,
        image_path: Path,
        predicted_class: int,
        method: str,
        output_path: Path,
        ig_steps: int = DEFAULT_IG_STEPS,
    ) -> dict[str, str]:
        result = generate_gradcam(
            model=model,
            image_path=str(image_path),
            predicted_class=predicted_class,
            output_path=str(output_path),
            method=method,
            integrated_gradients_steps=ig_steps,
        )
        return {
            "method": result.method,
            "target_layer": result.target_layer,
            "output_path": str(output_path),
            "output_relative_path": output_path.relative_to(OUTPUTS_DIR).as_posix(),
        }

    def explain_for_source_roi(
        self,
        model,
        image_path: Path,
        target_class: int,
        output_path: Path,
    ) -> dict[str, object]:
        result = generate_gradcam(
            model=model,
            image_path=str(image_path),
            predicted_class=target_class,
            output_path=str(output_path),
            method="gradcam",
        )
        return {
            "method": result.method,
            "target_layer": result.target_layer,
            "target_class": target_class,
            "output_path": str(output_path),
            "output_relative_path": output_path.relative_to(OUTPUTS_DIR).as_posix(),
            "heatmap": result.heatmap,
        }
