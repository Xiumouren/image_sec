from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from backend.explanation_generation.tensorflow.gradcam_tf import load_keras_model, preprocess_image
from backend.shared.config import DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH


class DetectorService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, labels_path: Path = DEFAULT_LABELS_PATH):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self._model = None
        self._labels: list[str] | None = None
        self._lock = threading.Lock()

    def get_labels(self) -> list[str]:
        if self._labels is None:
            self._labels = [
                line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        return self._labels

    def load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = load_keras_model(str(self.model_path))
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_path: Path) -> dict[str, object]:
        model = self.load_model()
        labels = self.get_labels()
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        _, input_tensor = preprocess_image(
            image_path=str(image_path),
            target_size=(int(input_shape[2]), int(input_shape[1])),
            normalize=True,
        )
        scores = np.asarray(model.predict(input_tensor, verbose=0)[0], dtype=np.float32)
        predicted_class_index = int(np.argmax(scores))
        score_map = {
            labels[index] if index < len(labels) else f"class_{index}": float(score)
            for index, score in enumerate(scores.tolist())
        }

        return {
            "predicted_class_index": predicted_class_index,
            "predicted_label": labels[predicted_class_index] if predicted_class_index < len(labels) else f"class_{predicted_class_index}",
            "scores": score_map,
            "raw_scores": scores,
        }
