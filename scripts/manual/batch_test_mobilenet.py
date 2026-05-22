import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from scripts.manual.test_mobilenet_model import MODEL_DIR, predict_image


DEFAULT_IMAGE_DIR = Path(r"C:\Code\net_sec\test_images")
DEFAULT_OUTPUT_DIR = Path(r"C:\Code\net_sec\outputs")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference for all test images and save a summary report."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing the exported model files.",
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
        help="Directory where CSV and JSON reports will be written.",
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
        help="Number of top predictions to keep per image.",
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.rglob("*"), key=lambda item: str(item.relative_to(image_dir)).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return images


def build_report_rows(
    results: list[dict[str, object]], image_dir: Path, top_k: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        predictions = result["predictions"]
        row: dict[str, object] = {
            "image_name": Path(str(result["image_source"])).name,
            "image_path": result["image_source"],
            "relative_dir": str(Path(str(result["image_source"])).parent.relative_to(image_dir)),
            "predicted_label": predictions[0]["label"],
            "predicted_score": predictions[0]["score"],
            "backend": result["backend"],
        }
        for rank in range(top_k):
            if rank < len(predictions):
                prediction = predictions[rank]
                row[f"top_{rank + 1}_label"] = prediction["label"]
                row[f"top_{rank + 1}_score"] = prediction["score"]
            else:
                row[f"top_{rank + 1}_label"] = ""
                row[f"top_{rank + 1}_score"] = ""
        rows.append(row)
    return rows


def write_csv(csv_path: Path, rows: list[dict[str, object]], top_k: int) -> None:
    fieldnames = [
        "image_name",
        "image_path",
        "relative_dir",
        "predicted_label",
        "predicted_score",
        "backend",
    ]
    for rank in range(top_k):
        fieldnames.append(f"top_{rank + 1}_label")
        fieldnames.append(f"top_{rank + 1}_score")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(json_path: Path, payload: dict[str, object]) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    image_paths = find_images(args.image_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, object]] = []

    print(f"Scanning images in: {args.image_dir}")
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        result = predict_image(args.model_dir, image_path, args.backend, args.top_k)
        results.append(result)
        top_prediction = result["predictions"][0]
        print(
            f"{image_path.relative_to(args.image_dir)}: "
            f"{top_prediction['label']} ({top_prediction['score']:.6f})"
        )

    rows = build_report_rows(results, args.image_dir, args.top_k)
    csv_path = args.output_dir / f"mobilenet_batch_results_{timestamp}.csv"
    json_path = args.output_dir / f"mobilenet_batch_results_{timestamp}.json"

    write_csv(csv_path, rows, args.top_k)
    write_json(
        json_path,
        {
            "model_dir": str(args.model_dir),
            "image_dir": str(args.image_dir),
            "backend": args.backend,
            "top_k": args.top_k,
            "generated_at": timestamp,
            "results": results,
        },
    )

    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
