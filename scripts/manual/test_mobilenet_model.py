import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_DIR = Path(r"C:\Code\net_sec\mobilenet_v2_140_224")
DEFAULT_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a TensorFlow MobileNetV2 export with an image or a synthetic sample."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing saved_model.pb / saved_model.tflite / class_labels.txt",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a real test image. If omitted, a synthetic image is used.",
    )
    parser.add_argument(
        "--backend",
        choices=("saved_model", "tflite"),
        default="saved_model",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to print.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        return []
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_input_image(image_path: Path | None, height: int, width: int) -> tuple[np.ndarray, str]:
    if image_path is None:
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        grid_x, grid_y = np.meshgrid(x, y)
        synthetic = np.stack(
            [grid_x, grid_y, np.full((height, width), 127, dtype=np.uint8)],
            axis=-1,
        )
        return synthetic, "synthetic"

    image = Image.open(image_path).convert("RGB").resize((width, height))
    return np.asarray(image, dtype=np.uint8), str(image_path)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    flat_scores = np.asarray(scores).reshape(-1)
    if flat_scores.size == 0:
        raise RuntimeError("Model returned an empty prediction tensor.")

    if np.any(flat_scores < 0) or not np.isclose(np.sum(flat_scores), 1.0, atol=1e-3):
        flat_scores = softmax(flat_scores)

    return flat_scores


def top_predictions(scores: np.ndarray, labels: list[str], top_k: int) -> list[dict[str, float | int | str]]:
    normalized_scores = normalize_scores(scores)
    top_indices = np.argsort(normalized_scores)[::-1][:top_k]
    predictions: list[dict[str, float | int | str]] = []

    for rank, index in enumerate(top_indices, start=1):
        label = labels[index] if index < len(labels) else f"class_{index}"
        predictions.append(
            {
                "rank": rank,
                "index": int(index),
                "label": label,
                "score": float(normalized_scores[index]),
            }
        )

    return predictions


def print_top_predictions(predictions: list[dict[str, float | int | str]]) -> None:
    print("Top predictions:")
    for prediction in predictions:
        print(f"  {prediction['rank']}. {prediction['label']}: {prediction['score']:.6f}")


def load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the current environment. "
            "Install it in your Conda env before testing this model."
        ) from exc

    return tf


def run_saved_model(
    model_dir: Path, image_array: np.ndarray, labels: list[str], top_k: int
) -> dict[str, object]:
    tf = load_tensorflow()

    loaded = tf.saved_model.load(str(model_dir))
    infer = loaded.signatures.get("serving_default")
    if infer is None:
        infer = next(iter(loaded.signatures.values()))

    _, input_spec_map = infer.structured_input_signature
    if not input_spec_map:
        raise RuntimeError("Unable to determine SavedModel input signature.")

    input_name, input_spec = next(iter(input_spec_map.items()))
    expected_dtype = input_spec.dtype.as_numpy_dtype
    shape = list(input_spec.shape)
    target_height = shape[1] if len(shape) > 2 and shape[1] is not None else DEFAULT_IMAGE_SIZE
    target_width = shape[2] if len(shape) > 2 and shape[2] is not None else DEFAULT_IMAGE_SIZE

    resized = np.asarray(
        Image.fromarray(image_array).resize((target_width, target_height)).convert("RGB")
    )

    if np.issubdtype(expected_dtype, np.floating):
        model_input = resized.astype(expected_dtype) / 255.0
    else:
        model_input = resized.astype(expected_dtype)

    model_input = np.expand_dims(model_input, axis=0)
    outputs = infer(**{input_name: tf.convert_to_tensor(model_input)})
    output_name, output_tensor = next(iter(outputs.items()))
    predictions = top_predictions(output_tensor.numpy()[0], labels, top_k)

    return {
        "backend": "saved_model",
        "input_name": input_name,
        "input_shape": tuple(model_input.shape),
        "input_dtype": str(model_input.dtype),
        "output_name": output_name,
        "output_shape": tuple(output_tensor.shape),
        "output_dtype": output_tensor.dtype.name,
        "predictions": predictions,
    }


def run_tflite(model_dir: Path, image_array: np.ndarray, labels: list[str], top_k: int) -> dict[str, object]:
    tf = load_tensorflow()

    model_path = model_dir / "saved_model.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    _, target_height, target_width, _ = input_details["shape"]
    resized = np.asarray(
        Image.fromarray(image_array).resize((int(target_width), int(target_height))).convert("RGB")
    )

    if np.issubdtype(input_details["dtype"], np.floating):
        model_input = resized.astype(input_details["dtype"]) / 255.0
    else:
        model_input = resized.astype(input_details["dtype"])

    model_input = np.expand_dims(model_input, axis=0)
    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details["index"])
    predictions = top_predictions(output_tensor[0], labels, top_k)

    return {
        "backend": "tflite",
        "input_name": input_details["name"],
        "input_shape": tuple(model_input.shape),
        "input_dtype": str(model_input.dtype),
        "output_name": output_details["name"],
        "output_shape": tuple(output_tensor.shape),
        "output_dtype": str(output_tensor.dtype),
        "predictions": predictions,
    }


def predict_image(
    model_dir: Path, image_path: Path | None, backend: str, top_k: int
) -> dict[str, object]:
    labels = load_labels(model_dir / "class_labels.txt")
    image_array, image_source = build_input_image(image_path, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    if backend == "saved_model":
        result = run_saved_model(model_dir, image_array, labels, top_k)
    else:
        result = run_tflite(model_dir, image_array, labels, top_k)

    return {
        "model_dir": str(model_dir),
        "image_source": image_source,
        "labels": labels,
        **result,
    }


def main() -> int:
    args = parse_args()
    result = predict_image(args.model_dir, args.image, args.backend, args.top_k)

    print(f"Model directory: {result['model_dir']}")
    print(f"Image source: {result['image_source']}")
    print(f"Loaded labels: {result['labels'] if result['labels'] else 'none'}")
    print(f"Backend: {result['backend']}")
    print(
        f"Input tensor: {result['input_name']}, shape={result['input_shape']}, "
        f"dtype={result['input_dtype']}"
    )
    print(
        f"Output tensor: {result['output_name']}, shape={result['output_shape']}, "
        f"dtype={result['output_dtype']}"
    )
    print_top_predictions(result["predictions"])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
