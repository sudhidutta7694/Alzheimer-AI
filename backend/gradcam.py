from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def _find_base_and_last_conv(model: tf.keras.Model) -> tuple[tf.keras.Model, tf.keras.layers.Layer]:
    # Prefer searching within the first nested model if Sequential
    search_model = model
    if isinstance(model, tf.keras.Sequential) and model.layers and isinstance(model.layers[0], tf.keras.Model):
        search_model = model.layers[0]

    layers = search_model.layers
    for layer in reversed(layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D) or (len(layer.output_shape) == 4):
                return search_model, layer
        except Exception:
            continue
    return search_model, layers[-1]


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    # Resolve base model and last conv inside that base
    if last_conv_layer_name:
        # try resolving from the base if exists
        base_model = model.layers[0] if isinstance(model, tf.keras.Sequential) and model.layers and isinstance(model.layers[0], tf.keras.Model) else model
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    else:
        base_model, last_conv_layer = _find_base_and_last_conv(model)

    # Feature extractor stops at the last conv layer, tied to the same graph as base_model
    feature_extractor = tf.keras.models.Model(base_model.input, last_conv_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array, training=False)
        predictions = model(img_array, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / (max_val + 1e-12), heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    # Resize heatmap to the image size
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    # Apply colormap via matplotlib
    cmap = cm.get_cmap(colormap)
    colored = cmap(np.array(heatmap_resized) / 255.0)  # RGBA
    colored_img = Image.fromarray((colored[:, :, :3] * 255).astype("uint8"))

    # Overlay
    overlay = Image.blend(image, colored_img, alpha=alpha)
    return overlay


def make_input_gradient_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    class_index: Optional[int] = None,
) -> np.ndarray:
    x = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]
    grads = tape.gradient(class_channel, x)
    # Take maximum across color channels to get 2D map
    heatmap = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / (max_val + 1e-12), heatmap)
    return heatmap.numpy()
