import numpy as np
import tensorflow as tf


def _to_float32(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        return img.astype(np.float32)
    return img


def compute_occlusion_sensitivity(model: tf.keras.Model, input_batch: np.ndarray, patch_size: int = 32, stride: int = 16, top_k: int = 5) -> dict:
    """
    Computes occlusion sensitivity by sliding a zeroed patch over the image and
    measuring change in predicted probability for the top class.

    Args:
        model: Keras model expecting a batch shaped (1, H, W, C).
        input_batch: Preprocessed input array with shape (1, H, W, C).
        patch_size: Square occlusion patch size in pixels.
        stride: Sliding stride in pixels.
        top_k: Number of most influential regions to return.

    Returns:
        Dict containing base prediction, and a list of regions with bbox and delta probability.
    """
    inp = _to_float32(input_batch)
    H, W = inp.shape[1], inp.shape[2]

    base_probs = model.predict(inp, verbose=0)[0]
    top_idx = int(np.argmax(base_probs))
    base_conf = float(base_probs[top_idx])

    deltas = []
    y_range = range(0, H - patch_size + 1, stride)
    x_range = range(0, W - patch_size + 1, stride)

    for y in y_range:
        for x in x_range:
            occluded = inp.copy()
            occluded[0, y : y + patch_size, x : x + patch_size, :] = 0.0
            probs = model.predict(occluded, verbose=0)[0]
            delta = float(base_conf - probs[top_idx])
            deltas.append({
                "bbox": [x, y, patch_size, patch_size],
                "delta": delta,
            })

    # Sort by delta descending and take top_k
    deltas.sort(key=lambda r: r["delta"], reverse=True)
    influential = deltas[: top_k]

    return {
        "predicted_index": top_idx,
        "base_confidence": base_conf,
        "regions": influential,
        "grid": {
            "patchSize": patch_size,
            "stride": stride,
            "height": H,
            "width": W,
        },
    }


def compute_integrated_gradients(model: tf.keras.Model, input_batch: np.ndarray, target_index: int | None = None, baseline: float = 0.0, steps: int = 32, return_signed: bool = True) -> dict:
    """
    Integrated Gradients attribution for a single input.

    Args:
        model: Keras model expecting a batch shaped (1, H, W, C).
        input_batch: Preprocessed input array with shape (1, H, W, C).
        target_index: Class index to explain. If None, uses argmax.
        baseline: Baseline pixel value (post-preprocess space), typically 0.
        steps: Number of interpolation steps.

    Returns:
        Dict with attribution map and summary statistics.
    """
    inp = _to_float32(input_batch)
    H, W = inp.shape[1], inp.shape[2]

    if target_index is None:
        probs = model.predict(inp, verbose=0)[0]
        target_index = int(np.argmax(probs))

    baseline_img = np.full_like(inp, baseline, dtype=np.float32)

    # Interpolate between baseline and input
    alphas = tf.linspace(0.0, 1.0, steps)
    attributions = np.zeros_like(inp[0], dtype=np.float32)

    for alpha in alphas:
        x = baseline_img + alpha.numpy() * (inp - baseline_img)
        x_tf = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            preds = model(x_tf, training=False)
            target = preds[:, target_index]
        grads = tape.gradient(target, x_tf)[0].numpy()
        attributions += grads

    # Scale by input difference
    attributions *= (inp[0] - baseline_img[0]) / float(steps)

    # Aggregate over channels
    signed_map = attributions.mean(axis=-1)
    magnitude_map = np.abs(signed_map)
    # Normalize magnitude to [0,1]
    max_val = magnitude_map.max() if magnitude_map.max() > 0 else 1.0
    norm_mag = (magnitude_map / max_val).astype(np.float32)

    summary = {
        "mean": float(norm_mag.mean()),
        "std": float(norm_mag.std()),
        "max": float(norm_mag.max()),
        "min": float(norm_mag.min()),
    }

    out = {
        "target_index": int(target_index),
        "height": H,
        "width": W,
        "map": norm_mag.tolist(),  # normalized magnitude map for visualization
        "summary": summary,
    }
    if return_signed:
        # Clip signed map into [-1,1] for compact transport
        max_abs = np.max(np.abs(signed_map)) or 1.0
        out["signed_map"] = (signed_map / max_abs).astype(np.float32).tolist()
    return out


def compute_roi_contributions(signed_map: np.ndarray, labels: list[str] | None = None) -> list[dict]:
    """
    Compute per-ROI contribution percentages and direction from a signed IG map.

    ROIs are coarse image-based proxies for anatomical regions. If input images
    are not registered to a standard space, treat labels as approximations.

    Returns a list of {name, bbox, positivePercent, negativePercent, direction}.
    """
    H, W = signed_map.shape
    # Define coarse ROIs (x,y,w,h) as proportions of width/height
    def rect(px, py, pw, ph):
        return [int(px * W), int(py * H), int(pw * W), int(ph * H)]

    rois = [
        {"name": "Left Medial Temporal (hippocampal region)", "bbox": rect(0.10, 0.45, 0.18, 0.20)},
        {"name": "Right Medial Temporal (hippocampal region)", "bbox": rect(0.72, 0.45, 0.18, 0.20)},
        {"name": "Frontal Lobe", "bbox": rect(0.20, 0.10, 0.60, 0.20)},
        {"name": "Parietal Lobe", "bbox": rect(0.20, 0.30, 0.60, 0.20)},
        {"name": "Occipital Lobe", "bbox": rect(0.25, 0.70, 0.50, 0.20)},
    ]

    pos_total = float(np.sum(np.clip(signed_map, 0, None))) or 1e-8
    neg_total = float(np.sum(np.clip(signed_map, None, 0)))  # negative sum (<=0)

    results = []
    for roi in rois:
        x, y, w, h = roi["bbox"]
        sub = signed_map[y : y + h, x : x + w]
        pos_sum = float(np.sum(np.clip(sub, 0, None)))
        neg_sum = float(np.sum(np.clip(sub, None, 0)))
        pos_pct = (pos_sum / pos_total) * 100.0 if pos_total > 0 else 0.0
        # For negative, normalize by absolute negative mass
        neg_pct = (abs(neg_sum) / abs(neg_total)) * 100.0 if neg_total < 0 else 0.0
        direction = "increased" if pos_sum >= abs(neg_sum) else "decreased"
        results.append({
            "name": roi["name"],
            "bbox": roi["bbox"],
            "positivePercent": pos_pct,
            "negativePercent": neg_pct,
            "direction": direction,
        })

    # Sort by dominant contribution (max of pos/neg percent)
    results.sort(key=lambda r: max(r["positivePercent"], r["negativePercent"]), reverse=True)
    return results
