import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.manual.batch_test_mobilenet import DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR, SUPPORTED_EXTENSIONS
from gradcam_tf import generate_gradcam, load_keras_model, preprocess_image


DEFAULT_MODEL_PATH = Path(r"C:\Code\net_sec\mobilenet_v2_140_224\saved_model.h5")
DEFAULT_LABELS_PATH = Path(r"C:\Code\net_sec\mobilenet_v2_140_224\class_labels.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate saliency or integrated-gradients overlays for NSFW images."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the NSFW Keras H5 model.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to the class_labels.txt file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where explanation images and reports will be written.",
    )
    parser.add_argument(
        "--method",
        choices=("saliency", "integrated_gradients"),
        default="integrated_gradients",
        help="Input-gradient explanation method to use.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=32,
        help="Number of integration steps when method=integrated_gradients.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.rglob("*"), key=lambda item: str(item.relative_to(image_dir)).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return images


def predict_top1(model, image_path: Path) -> tuple[int, np.ndarray]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    _, input_tensor = preprocess_image(
        image_path=str(image_path),
        target_size=(int(input_shape[2]), int(input_shape[1])),
        normalize=True,
    )
    scores = model.predict(input_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(scores))
    return predicted_index, np.asarray(scores, dtype=np.float32)


def build_output_path(output_dir: Path, image_dir: Path, image_path: Path, method: str) -> Path:
    relative_path = image_path.relative_to(image_dir)
    relative_without_suffix = relative_path.with_suffix("")
    output_path = output_dir / "explanations" / method / relative_without_suffix.parent
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{relative_without_suffix.name}_{method}.jpg"


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_name",
        "relative_path",
        "predicted_class_index",
        "predicted_label",
        "predicted_score",
        "method",
        "output_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(json_path: Path, payload: dict[str, object]) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    image_paths = find_images(args.image_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = load_keras_model(str(args.model))
    rows: list[dict[str, object]] = []
    results: list[dict[str, object]] = []

    print(f"Loading model: {args.model}")
    print(f"Scanning images in: {args.image_dir}")
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        predicted_index, scores = predict_top1(model, image_path)
        output_path = build_output_path(args.output_dir, args.image_dir, image_path, args.method)
        explanation = generate_gradcam(
            model=model,
            image_path=str(image_path),
            predicted_class=predicted_index,
            output_path=str(output_path),
            method=args.method,
            integrated_gradients_steps=args.ig_steps,
        )

        predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"class_{predicted_index}"
        relative_path = str(image_path.relative_to(args.image_dir))
        predicted_score = float(scores[predicted_index])

        print(f"{relative_path}: {predicted_label} ({predicted_score:.6f}) -> {output_path.name}")

        rows.append(
            {
                "image_name": image_path.name,
                "relative_path": relative_path,
                "predicted_class_index": predicted_index,
                "predicted_label": predicted_label,
                "predicted_score": predicted_score,
                "method": args.method,
                "output_path": str(output_path),
            }
        )
        results.append(
            {
                "image_path": str(image_path),
                "relative_path": relative_path,
                "predicted_class_index": predicted_index,
                "predicted_label": predicted_label,
                "predicted_score": predicted_score,
                "method": explanation.method,
                "output_path": explanation.output_path,
                "target_layer": explanation.target_layer,
            }
        )

    csv_path = args.output_dir / f"nsfw_explanations_{args.method}_{timestamp}.csv"
    json_path = args.output_dir / f"nsfw_explanations_{args.method}_{timestamp}.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "model_path": str(args.model),
            "labels_path": str(args.labels),
            "image_dir": str(args.image_dir),
            "method": args.method,
            "generated_at": timestamp,
            "results": results,
        },
    )

    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
