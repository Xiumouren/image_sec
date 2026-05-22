import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_TMP_DIR = BASE_DIR / ".tmp"
TFHUB_CACHE_DIR = WORKSPACE_TMP_DIR / "tfhub_cache"
DEFAULT_SOURCE_MODEL = BASE_DIR / "mobilenet_v2_140_224" / "saved_model.h5"
DEFAULT_OUTPUT_MODEL = BASE_DIR / "rebuild" / "nsfw_mobilenetv2_gradcam_ready.h5"
DEFAULT_LABELS = BASE_DIR / "mobilenet_v2_140_224" / "class_labels.txt"
DEFAULT_TEST_IMAGE_DIR = BASE_DIR / "test_images"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the TF Hub NSFW model as a standard Keras MobileNetV2 so Grad-CAM works."
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
        help="Path to the original TF Hub-backed H5 model.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save the rebuilt Grad-CAM-ready Keras model.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to class_labels.txt.",
    )
    parser.add_argument(
        "--test-image-dir",
        type=Path,
        default=DEFAULT_TEST_IMAGE_DIR,
        help="Directory of sample images used to compare old/new model predictions.",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=11,
        help="Maximum number of images used for old/new prediction comparison.",
    )
    return parser.parse_args()


def prepare_environment() -> None:
    WORKSPACE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(WORKSPACE_TMP_DIR)
    os.environ["TMP"] = str(WORKSPACE_TMP_DIR)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))


def load_original_model(source_model: Path):
    prepare_environment()
    import tensorflow_hub as hub
    import tf_keras

    return tf_keras.models.load_model(
        source_model,
        custom_objects={"KerasLayer": hub.KerasLayer},
        compile=False,
    )


def build_gradcam_ready_model(class_count: int, dropout_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.4,
        include_top=False,
        weights=None,
    )
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(1001, name="imagenet_logits")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(class_count, name="dense")(x)
    outputs = tf.keras.layers.Activation("softmax", name="prediction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="nsfw_mobilenetv2_gradcam_ready")
    return model


def keras_layer_to_hub_prefix(layer_name: str) -> str | None:
    if layer_name == "Conv1":
        return "MobilenetV2/Conv"
    if layer_name == "bn_Conv1":
        return "MobilenetV2/Conv/BatchNorm"
    if layer_name == "expanded_conv_depthwise":
        return "MobilenetV2/expanded_conv/depthwise"
    if layer_name == "expanded_conv_depthwise_BN":
        return "MobilenetV2/expanded_conv/depthwise/BatchNorm"
    if layer_name == "expanded_conv_project":
        return "MobilenetV2/expanded_conv/project"
    if layer_name == "expanded_conv_project_BN":
        return "MobilenetV2/expanded_conv/project/BatchNorm"
    if layer_name == "Conv_1":
        return "MobilenetV2/Conv_1"
    if layer_name == "Conv_1_bn":
        return "MobilenetV2/Conv_1/BatchNorm"

    if layer_name.startswith("block_"):
        parts = layer_name.split("_")
        block_id = parts[1]
        suffix = "_".join(parts[2:])
        suffix_to_prefix = {
            "expand": f"MobilenetV2/expanded_conv_{block_id}/expand",
            "expand_BN": f"MobilenetV2/expanded_conv_{block_id}/expand/BatchNorm",
            "depthwise": f"MobilenetV2/expanded_conv_{block_id}/depthwise",
            "depthwise_BN": f"MobilenetV2/expanded_conv_{block_id}/depthwise/BatchNorm",
            "project": f"MobilenetV2/expanded_conv_{block_id}/project",
            "project_BN": f"MobilenetV2/expanded_conv_{block_id}/project/BatchNorm",
        }
        return suffix_to_prefix.get(suffix)

    return None


def extract_old_backbone_weights(original_model) -> dict[str, np.ndarray]:
    return {weight.name: weight.numpy() for weight in original_model.layers[0].weights}


def set_conv_or_depthwise_weights(layer: tf.keras.layers.Layer, old_weights: dict[str, np.ndarray], prefix: str) -> None:
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        layer.set_weights([old_weights[f"{prefix}/depthwise_weights:0"]])
    else:
        layer.set_weights([old_weights[f"{prefix}/weights:0"]])


def set_batchnorm_weights(layer: tf.keras.layers.BatchNormalization, old_weights: dict[str, np.ndarray], prefix: str) -> None:
    layer.set_weights(
        [
            old_weights[f"{prefix}/gamma:0"],
            old_weights[f"{prefix}/beta:0"],
            old_weights[f"{prefix}/moving_mean:0"],
            old_weights[f"{prefix}/moving_variance:0"],
        ]
    )


def transfer_backbone_weights(rebuilt_model: tf.keras.Model, original_model) -> None:
    old_weights = extract_old_backbone_weights(original_model)
    backbone = rebuilt_model.get_layer("mobilenetv2_1.40_224")

    for layer in backbone.layers:
        prefix = keras_layer_to_hub_prefix(layer.name)
        if prefix is None or not layer.weights:
            continue

        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            set_conv_or_depthwise_weights(layer, old_weights, prefix)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            set_batchnorm_weights(layer, old_weights, prefix)


def transfer_classifier_weights(rebuilt_model: tf.keras.Model, original_model) -> None:
    old_backbone_weights = extract_old_backbone_weights(original_model)

    imagenet_kernel = old_backbone_weights["MobilenetV2/Logits/Conv2d_1c_1x1/weights:0"]
    imagenet_bias = old_backbone_weights["MobilenetV2/Logits/Conv2d_1c_1x1/biases:0"]
    rebuilt_model.get_layer("imagenet_logits").set_weights(
        [imagenet_kernel.reshape((imagenet_kernel.shape[2], imagenet_kernel.shape[3])), imagenet_bias]
    )

    dense_weights = original_model.layers[2].get_weights()
    rebuilt_model.get_layer("dense").set_weights(dense_weights)


def load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_test_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"), key=lambda item: str(item.relative_to(image_dir)).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def compare_models(
    original_model,
    rebuilt_model: tf.keras.Model,
    image_paths: list[Path],
    labels: list[str],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []

    for image_path in image_paths:
        input_tensor = preprocess_image(image_path)
        original_scores = original_model(input_tensor, training=False).numpy()[0]
        rebuilt_scores = rebuilt_model(input_tensor, training=False).numpy()[0]

        comparisons.append(
            {
                "image_path": str(image_path),
                "original_top1": labels[int(np.argmax(original_scores))],
                "rebuilt_top1": labels[int(np.argmax(rebuilt_scores))],
                "max_abs_diff": float(np.max(np.abs(original_scores - rebuilt_scores))),
                "mean_abs_diff": float(np.mean(np.abs(original_scores - rebuilt_scores))),
            }
        )

    return comparisons


def write_report(output_model: Path, comparisons: list[dict[str, object]]) -> Path:
    report_path = output_model.with_suffix(".comparison.json")
    report_path.write_text(json.dumps({"comparisons": comparisons}, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    original_model = load_original_model(args.source_model)
    dropout_rate = float(original_model.layers[1].rate)

    rebuilt_model = build_gradcam_ready_model(class_count=len(labels), dropout_rate=dropout_rate)
    rebuilt_model(tf.zeros((1, 224, 224, 3), dtype=tf.float32))

    transfer_backbone_weights(rebuilt_model, original_model)
    transfer_classifier_weights(rebuilt_model, original_model)

    test_images = find_test_images(args.test_image_dir)
    test_images = test_images[: args.max_test_images]
    comparisons = compare_models(original_model, rebuilt_model, test_images, labels)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    rebuilt_model.save(args.output_model, include_optimizer=False)
    report_path = write_report(args.output_model, comparisons)

    print(f"Saved rebuilt model: {args.output_model}")
    print(f"Saved comparison report: {report_path}")
    for item in comparisons:
        print(
            f"{Path(item['image_path']).name}: "
            f"{item['original_top1']} -> {item['rebuilt_top1']}, "
            f"max_abs_diff={item['max_abs_diff']:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
