import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


DEFAULT_ALPHA = 0.4
DEFAULT_SCORECAM_MAX_MAPS = 32
DEFAULT_IG_STEPS = 32
WORKSPACE_TMP_DIR = Path(__file__).resolve().parents[3] / ".tmp"
TFHUB_CACHE_DIR = WORKSPACE_TMP_DIR / "tfhub_cache"


@dataclass(frozen=True)
class GradCAMResult:
    image_path: str
    output_path: str
    predicted_class: int
    method: str
    target_layer: str
    scores: np.ndarray
    heatmap: np.ndarray


def load_image(image_path: str) -> np.ndarray:
    """Load an RGB image as a uint8 numpy array."""
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def preprocess_image(
    image_path: str,
    target_size: tuple[int, int],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image, resize it to the model input size, and optionally scale to 0-1.

    Returns:
        original_image: uint8 RGB image in original resolution
        input_tensor: float32 tensor shaped [1, H, W, 3]
    """
    original_image = load_image(image_path)
    resized = Image.fromarray(original_image).resize(target_size, Image.Resampling.BILINEAR)
    input_array = np.asarray(resized, dtype=np.float32)
    if normalize:
        input_array = input_array / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)
    return original_image, input_tensor


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """
    Resize the heatmap to the original image size and blend it with the RGB image.
    """
    if original_image.dtype != np.uint8:
        raise ValueError("original_image must be uint8 for visualization.")

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(original_image, 1.0 - alpha, colored_heatmap, alpha, 0)
    return blended


def save_overlay_image(image: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def aggregate_input_attribution(attribution: np.ndarray) -> np.ndarray:
    """
    Collapse RGB attributions into a single normalized heatmap.
    """
    attribution = np.abs(np.asarray(attribution, dtype=np.float32))
    if attribution.ndim == 3:
        attribution = np.max(attribution, axis=-1)

    max_value = float(np.max(attribution))
    if max_value <= 0.0:
        return np.zeros_like(attribution, dtype=np.float32)
    return attribution / max_value


def _walk_layers(model: tf.keras.Model) -> Iterable[tf.keras.layers.Layer]:
    """
    Yield layers recursively so nested models are searchable.
    """
    for layer in model.layers:
        yield layer
        if isinstance(layer, tf.keras.Model):
            yield from _walk_layers(layer)


def _walk_layers_with_parent(model: tf.keras.Model) -> Iterable[tuple[tf.keras.layers.Layer, tf.keras.Model]]:
    for layer in model.layers:
        yield layer, model
        if isinstance(layer, tf.keras.Model):
            yield from _walk_layers_with_parent(layer)


def find_target_layer(
    model: tf.keras.Model,
    target_layer_name: str | None = None,
) -> tuple[tf.keras.layers.Layer, tf.keras.Model]:
    """
    Pick the requested layer or the last convolution-like layer automatically.

    Grad-CAM requires access to a spatial feature map. If a model is wrapped into an
    opaque TensorFlow Hub KerasLayer / TFSMLayer, there is no internal conv layer to use.
    """
    all_layers = list(_walk_layers_with_parent(model))

    if target_layer_name:
        for layer, parent_model in all_layers:
            if layer.name == target_layer_name:
                return layer, parent_model
        raise ValueError(f"Layer '{target_layer_name}' was not found in the model.")

    convolutional_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    for layer, parent_model in reversed(all_layers):
        if isinstance(layer, convolutional_types):
            return layer, parent_model

    raise ValueError(
        "No convolutional layer was found. This usually means the classifier backbone is "
        "hidden inside an opaque KerasLayer/TFSMLayer, so true Grad-CAM cannot be computed."
    )


def ensure_keras_model(model: object) -> tf.keras.Model:
    """
    Validate that the input is a Keras model with accessible layers.
    """
    if isinstance(model, tf.keras.Model):
        return model

    if hasattr(model, "layers") and hasattr(model, "inputs") and hasattr(model, "outputs"):
        return model

    raise TypeError(
        "Grad-CAM requires a tf.keras.Model (or compatible object exposing layers/inputs/outputs). "
        "A bare tf.saved_model.load(...) object is not enough unless it preserves the Keras graph."
    )


def load_keras_model(model_path: str) -> tf.keras.Model:
    """
    Load a Keras model and transparently handle TensorFlow Hub KerasLayer models.

    GantMan/nsfw_model loads the H5 model with:
        tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    """
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except ValueError as exc:
        if "Unknown layer: 'KerasLayer'" not in str(exc):
            raise

        try:
            WORKSPACE_TMP_DIR.mkdir(parents=True, exist_ok=True)
            TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            os.environ["TEMP"] = str(WORKSPACE_TMP_DIR)
            os.environ["TMP"] = str(WORKSPACE_TMP_DIR)
            os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))
            import tensorflow_hub as hub
            import tf_keras
        except ImportError as hub_exc:
            raise RuntimeError(
                "This model uses tensorflow_hub.KerasLayer. Install tensorflow-hub and tf-keras first, "
                "then retry. Example: pip install tensorflow-hub tf-keras"
            ) from hub_exc

        return tf_keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
        )


def build_gradcam_model(
    model: tf.keras.Model,
    target_layer_name: str | None = None,
) -> tuple[object, tf.keras.layers.Layer]:
    """
    Create an auxiliary model that returns both target activations and classifier output.
    """
    keras_model = ensure_keras_model(model)
    target_layer, parent_model = find_target_layer(keras_model, target_layer_name)

    if parent_model is keras_model:
        grad_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[target_layer.output, keras_model.outputs[0]],
            name=f"{keras_model.name}_gradcam",
        )
        return grad_model, target_layer

    # For nested backbones, split the graph into:
    # input -> target activations -> remaining backbone -> classification head.
    feature_model = tf.keras.Model(
        inputs=parent_model.input,
        outputs=target_layer.output,
        name=f"{parent_model.name}_{target_layer.name}_features",
    )
    tail_model = tf.keras.Model(
        inputs=target_layer.output,
        outputs=parent_model.output,
        name=f"{parent_model.name}_{target_layer.name}_tail",
    )
    head_layers = []
    capture = False
    for layer in keras_model.layers:
        if layer is parent_model:
            capture = True
            continue
        if capture:
            head_layers.append(layer)
    return {"feature_model": feature_model, "tail_model": tail_model, "head_layers": head_layers}, target_layer


def _normalize_heatmap(heatmap: tf.Tensor) -> np.ndarray:
    heatmap = tf.maximum(heatmap, 0)
    denominator = tf.reduce_max(heatmap)
    if float(denominator) == 0.0:
        return np.zeros_like(heatmap.numpy(), dtype=np.float32)
    return (heatmap / denominator).numpy().astype(np.float32)


def compute_gradcam(
    grad_model,
    input_tensor: np.ndarray,
    predicted_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Grad-CAM: weight each channel by the mean gradient over spatial positions.
    """
    inputs = tf.convert_to_tensor(input_tensor)
    if isinstance(grad_model, dict):
        feature_model = grad_model["feature_model"]
        tail_model = grad_model["tail_model"]
        head_layers = grad_model["head_layers"]
        with tf.GradientTape() as tape:
            conv_outputs = feature_model(inputs, training=False)
            tape.watch(conv_outputs)
            predictions = tail_model(conv_outputs, training=False)
            for layer in head_layers:
                predictions = layer(predictions, training=False)
            class_score = predictions[:, predicted_class]
    else:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inputs, training=False)
            class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    return _normalize_heatmap(heatmap), predictions[0].numpy()


def compute_gradcam_plus_plus(
    grad_model,
    input_tensor: np.ndarray,
    predicted_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM++ uses higher-order gradient terms to better isolate multiple salient regions.
    """
    if isinstance(grad_model, dict):
        raise ValueError("Grad-CAM++ for nested Grad-CAM-ready models is not implemented in this script.")

    inputs = tf.convert_to_tensor(input_tensor)
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(inputs, training=False)
                class_score = predictions[:, predicted_class]
            first_grad = tape3.gradient(class_score, conv_outputs)
        second_grad = tape2.gradient(first_grad, conv_outputs)
    third_grad = tape1.gradient(second_grad, conv_outputs)

    conv_outputs = conv_outputs[0]
    first_grad = first_grad[0]
    second_grad = second_grad[0]
    third_grad = third_grad[0]

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
    alpha_num = second_grad
    alpha_denom = (2.0 * second_grad) + (third_grad * global_sum)
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alpha = alpha_num / alpha_denom

    positive_gradients = tf.nn.relu(first_grad)
    weights = tf.reduce_sum(alpha * positive_gradients, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    return _normalize_heatmap(heatmap), predictions[0].numpy()


def _select_scorecam_channels(activations: np.ndarray, max_maps: int) -> np.ndarray:
    if activations.shape[-1] <= max_maps:
        return np.arange(activations.shape[-1])

    variances = np.std(activations, axis=(0, 1))
    return np.argsort(variances)[::-1][:max_maps]


def compute_scorecam(
    grad_model,
    input_tensor: np.ndarray,
    predicted_class: int,
    max_maps: int = DEFAULT_SCORECAM_MAX_MAPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score-CAM perturbs the image with activation masks and uses forward scores as weights.
    """
    if isinstance(grad_model, dict):
        raise ValueError("Score-CAM for nested Grad-CAM-ready models is not implemented in this script.")

    inputs = tf.convert_to_tensor(input_tensor)
    conv_outputs, predictions = grad_model(inputs, training=False)
    activations = conv_outputs[0].numpy()
    base_scores = predictions[0].numpy()

    channel_indices = _select_scorecam_channels(activations, max_maps)
    heatmap = np.zeros(activations.shape[:2], dtype=np.float32)

    resized_input = input_tensor[0]
    for channel_index in channel_indices:
        activation_map = activations[:, :, channel_index]
        activation_map = np.maximum(activation_map, 0)
        max_value = activation_map.max()
        if max_value <= 0:
            continue

        normalized_map = activation_map / max_value
        masked_input = resized_input * normalized_map[..., np.newaxis]
        masked_input = np.expand_dims(masked_input, axis=0)
        _, masked_predictions = grad_model(tf.convert_to_tensor(masked_input), training=False)
        weight = float(masked_predictions[0, predicted_class].numpy())
        heatmap += weight * normalized_map

    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap.astype(np.float32), base_scores


def compute_saliency_map(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    predicted_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vanilla saliency map: gradient magnitude of the target class score with respect
    to the input pixels.
    """
    inputs = tf.convert_to_tensor(input_tensor)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, inputs)[0].numpy()
    heatmap = aggregate_input_attribution(gradients)
    return heatmap, predictions[0].numpy()


def compute_integrated_gradients(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    predicted_class: int,
    steps: int = DEFAULT_IG_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrated Gradients: integrate input gradients along the straight line from a
    zero baseline to the actual input. This is stable for opaque TF Hub backbones.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2 for integrated gradients.")

    inputs = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    baseline = tf.zeros_like(inputs)
    alphas = tf.linspace(0.0, 1.0, steps + 1)

    gradient_batches = []
    for alpha in alphas:
        interpolated = baseline + alpha * (inputs - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated, training=False)
            class_score = predictions[:, predicted_class]
        gradients = tape.gradient(class_score, interpolated)
        gradient_batches.append(gradients)

    stacked_gradients = tf.stack(gradient_batches, axis=0)
    avg_gradients = tf.reduce_mean(stacked_gradients[:-1] + stacked_gradients[1:], axis=0) / 2.0
    integrated_gradients = (inputs - baseline) * avg_gradients
    heatmap = aggregate_input_attribution(integrated_gradients[0].numpy())
    scores = model(inputs, training=False)[0].numpy()
    return heatmap, scores


def generate_gradcam(
    model,
    image_path: str,
    predicted_class: int,
    output_path: str,
    method: str = "gradcam",
    target_layer_name: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    scorecam_max_maps: int = DEFAULT_SCORECAM_MAX_MAPS,
    integrated_gradients_steps: int = DEFAULT_IG_STEPS,
) -> GradCAMResult:
    """
    Generate and save a Grad-CAM explanation for the requested class.

    Supported methods:
        - gradcam
        - gradcam++
        - scorecam
        - saliency
        - integrated_gradients
    """
    keras_model = ensure_keras_model(model)

    input_shape = keras_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    target_height = int(input_shape[1])
    target_width = int(input_shape[2])

    original_image, input_tensor = preprocess_image(
        image_path=image_path,
        target_size=(target_width, target_height),
        normalize=True,
    )

    normalized_method = method.lower().strip()
    target_layer = None
    if normalized_method in {"gradcam", "gradcam++", "grad_cam++", "gradcampp", "scorecam"}:
        grad_model, target_layer = build_gradcam_model(keras_model, target_layer_name)

    if normalized_method == "gradcam":
        heatmap, scores = compute_gradcam(grad_model, input_tensor, predicted_class)
    elif normalized_method in {"gradcam++", "grad_cam++", "gradcampp"}:
        heatmap, scores = compute_gradcam_plus_plus(grad_model, input_tensor, predicted_class)
        normalized_method = "gradcam++"
    elif normalized_method == "scorecam":
        heatmap, scores = compute_scorecam(
            grad_model,
            input_tensor,
            predicted_class,
            max_maps=scorecam_max_maps,
        )
    elif normalized_method == "saliency":
        heatmap, scores = compute_saliency_map(keras_model, input_tensor, predicted_class)
    elif normalized_method in {"integrated_gradients", "integrated-gradients", "ig"}:
        heatmap, scores = compute_integrated_gradients(
            keras_model,
            input_tensor,
            predicted_class,
            steps=integrated_gradients_steps,
        )
        normalized_method = "integrated_gradients"
    else:
        raise ValueError(f"Unsupported Grad-CAM method: {method}")

    overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)
    save_overlay_image(overlay, output_path)

    return GradCAMResult(
        image_path=image_path,
        output_path=output_path,
        predicted_class=predicted_class,
        method=normalized_method,
        target_layer=target_layer.name if target_layer is not None else "input_gradients",
        scores=np.asarray(scores, dtype=np.float32),
        heatmap=heatmap,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TensorFlow Grad-CAM overlays.")
    parser.add_argument("--model", required=True, help="Path to a Keras .h5/.keras model file.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--class-index", type=int, required=True, help="Predicted class index.")
    parser.add_argument("--output", required=True, help="Path to save the overlay image.")
    parser.add_argument(
        "--method",
        default="gradcam",
        choices=("gradcam", "gradcam++", "scorecam", "saliency", "integrated_gradients"),
        help="Explanation algorithm.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=DEFAULT_IG_STEPS,
        help="Number of integration steps used by integrated gradients.",
    )
    parser.add_argument(
        "--target-layer",
        default=None,
        help="Optional explicit convolution layer name. If omitted, the last conv layer is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_keras_model(args.model)
    result = generate_gradcam(
        model=model,
        image_path=args.image,
        predicted_class=args.class_index,
        output_path=args.output,
        method=args.method,
        target_layer_name=args.target_layer,
        integrated_gradients_steps=args.ig_steps,
    )
    print(f"Saved overlay: {result.output_path}")
    print(f"Method: {result.method}")
    print(f"Target layer: {result.target_layer}")
    print(f"Top class score: {float(result.scores[result.predicted_class]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
