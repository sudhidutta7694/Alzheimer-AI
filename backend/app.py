from __future__ import annotations
import base64
import io
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

from tensorflow.keras.applications.efficientnet import preprocess_input

from . import config
from .utils import load_class_names, read_image_to_rgb, pil_to_model_array
from .custom_layers import SafeFlatten
from .gradcam import make_gradcam_heatmap, make_input_gradient_heatmap, overlay_heatmap_on_image
from .explanations import compute_occlusion_sensitivity, compute_integrated_gradients, compute_roi_contributions
from .reporting import build_pdf_report
from .reporting_html import build_html_pdf

app = Flask(__name__)
CORS(app)

MODEL: tf.keras.Model | None = None
CLASS_NAMES = []


def _load_model() -> tf.keras.Model:
    model_path = config.MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    # Load with a custom Flatten that tolerates list inputs from legacy graphs
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"Flatten": SafeFlatten},
    )

    # Some older SavedModels/H5 (Keras v2) may produce list/tuple outputs from
    # intermediate layers under Keras 3 runtime, breaking Flatten/Dense calls.
    # To ensure a proper tensor is passed through the stack, wrap the model
    # into a functional graph that selects the first tensor when a layer
    # returns a list/tuple.
    try:
        # Build a deterministic functional forward pass
        if isinstance(model, tf.keras.Sequential):
            # Use known input size to ensure the graph is well-defined
            inp = tf.keras.Input(shape=(config.INPUT_SIZE[0], config.INPUT_SIZE[1], 3))
            x = inp
            for lyr in model.layers:
                y = lyr(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                x = y
            model = tf.keras.Model(inp, x)
        else:
            # Functional models generally already have proper graph outputs
            pass
    except Exception:
        # If wrapping fails, fall back to the original model
        pass

    return model

def ensure_model_loaded():
    global MODEL, CLASS_NAMES
    if MODEL is None:
        MODEL = _load_model()
        try:
            n_classes = MODEL.output_shape[-1]
        except Exception:
            n_classes = 4
        CLASS_NAMES = load_class_names(config.CLASS_NAMES_PATH, num_classes=n_classes)


@app.get("/health")
def health():
    loaded = False
    try:
        ensure_model_loaded()
        loaded = True
    except Exception:
        loaded = False
    return jsonify({
        "status": "ok",
        "model_loaded": loaded,
        "model_path": str(config.MODEL_PATH),
    })


@app.get("/metadata")
def metadata():
    if not CLASS_NAMES:
        try:
            ensure_model_loaded()
        except Exception:
            # fall back to default class names length 4
            from .utils import load_class_names
            CLASS_NAMES.extend(load_class_names(config.CLASS_NAMES_PATH, num_classes=4))
    return jsonify({
        "input_size": list(config.INPUT_SIZE),
        "classes": CLASS_NAMES,
        "model_path": str(config.MODEL_PATH),
    })


@app.post("/predict")
def predict():
    global MODEL
    ensure_model_loaded()

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded under field 'image'"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # Read and preprocess
    pil_img = read_image_to_rgb(image_bytes)
    arr = pil_to_model_array(pil_img, config.INPUT_SIZE)
    arr_pp = preprocess_input(arr.copy())

    # Predict
    preds = MODEL.predict(arr_pp, verbose=0)[0]
    # Temperature scaling for calibration
    try:
        temp = float(os.getenv("TEMP_SCALE", "1.0"))
    except Exception:
        temp = 1.0
    logits = preds if preds.ndim == 1 else np.array(preds)
    probs = tf.nn.softmax(logits / max(1e-6, temp)).numpy() if logits.ndim == 1 else logits
    pred_index = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else str(pred_index)
    confidence = float(probs[pred_index])

    # Top-2 classes and margin
    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    top2_items = [
        {"index": top1, "label": CLASS_NAMES[top1] if top1 < len(CLASS_NAMES) else str(top1), "probability": float(probs[top1])},
        {"index": top2, "label": CLASS_NAMES[top2] if top2 < len(CLASS_NAMES) else str(top2), "probability": float(probs[top2])},
    ]
    margin = float(probs[top1] - probs[top2]) if len(order) > 1 else float(probs[top1])

    # Explainability heatmap (Grad-CAM first, fallback to input gradients)
    try:
        heatmap = make_gradcam_heatmap(arr_pp, MODEL, class_index=pred_index)
    except Exception:
        heatmap = make_input_gradient_heatmap(arr_pp, MODEL, class_index=pred_index)
    overlay = overlay_heatmap_on_image(pil_img.resize(config.INPUT_SIZE), heatmap, alpha=0.45)

    # Encode overlay to base64 PNG
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    heatmap_data_uri = f"data:image/png;base64,{heatmap_b64}"

    probs_list = [
        {"index": int(i), "label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), "probability": float(p)}
        for i, p in enumerate(probs)
    ]

    explanations_level = os.getenv("EXPLAIN_LEVEL", "full").lower()
    occlusion = None
    ig = None
    if explanations_level == "full":
        # Dynamic, image-specific reasons (heavy). Guarded by EXPLAIN_LEVEL=full
        try:
            occlusion = compute_occlusion_sensitivity(MODEL, arr_pp, patch_size=48, stride=32, top_k=5)
        except Exception as e:
            occlusion = {"error": str(e)}
        try:
            ig = compute_integrated_gradients(MODEL, arr_pp, target_index=pred_index, baseline=0.0, steps=24, return_signed=True)
        except Exception as e:
            ig = {"error": str(e)}
        # ROI contributions based on signed IG, when available
        try:
            if isinstance(ig, dict) and "signed_map" in ig:
                signed = np.array(ig["signed_map"], dtype=np.float32)
                roi_contrib = compute_roi_contributions(signed)
            else:
                roi_contrib = None
        except Exception as e:
            roi_contrib = {"error": str(e)}

    return jsonify({
        "predicted_index": pred_index,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": probs_list,
        "top2": top2_items,
        "margin": margin,
        "heatmap": heatmap_data_uri,
        "inputSize": list(config.INPUT_SIZE),
        "classes": CLASS_NAMES,
        "explanations": {
            "level": explanations_level,
            "occlusion": occlusion,
            "integratedGradients": ig,
            "roiContributions": roi_contrib,
        },
    })


@app.post("/report")
def report():
    """
    Generate a PDF report for a given image upload using the same pipeline as /predict.
    Returns application/pdf bytes.
    """
    ensure_model_loaded()

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded under field 'image'"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # Preprocess and run prediction with explanations level = full (for report)
    pil_img = read_image_to_rgb(image_bytes)
    arr = pil_to_model_array(pil_img, config.INPUT_SIZE)
    arr_pp = preprocess_input(arr.copy())

    preds = MODEL.predict(arr_pp, verbose=0)[0]
    try:
        temp = float(os.getenv("TEMP_SCALE", "1.0"))
    except Exception:
        temp = 1.0
    logits = preds if preds.ndim == 1 else np.array(preds)
    probs = tf.nn.softmax(logits / max(1e-6, temp)).numpy() if logits.ndim == 1 else logits
    pred_index = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else str(pred_index)
    confidence = float(probs[pred_index])

    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    top2_items = [
        {"index": top1, "label": CLASS_NAMES[top1] if top1 < len(CLASS_NAMES) else str(top1), "probability": float(probs[top1])},
        {"index": top2, "label": CLASS_NAMES[top2] if top2 < len(CLASS_NAMES) else str(top2), "probability": float(probs[top2])},
    ]
    margin = float(probs[top1] - probs[top2]) if len(order) > 1 else float(probs[top1])

    # Heatmap
    try:
        heatmap = make_gradcam_heatmap(arr_pp, MODEL, class_index=pred_index)
    except Exception:
        heatmap = make_input_gradient_heatmap(arr_pp, MODEL, class_index=pred_index)
    overlay = overlay_heatmap_on_image(pil_img.resize(config.INPUT_SIZE), heatmap, alpha=0.45)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    import base64
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    heatmap_data_uri = f"data:image/png;base64,{heatmap_b64}"

    # Explanations (full)
    occlusion = None
    ig = None
    roi_contrib = None
    try:
        occlusion = compute_occlusion_sensitivity(MODEL, arr_pp, patch_size=48, stride=32, top_k=5)
    except Exception as e:
        occlusion = {"error": str(e)}
    try:
        ig = compute_integrated_gradients(MODEL, arr_pp, target_index=pred_index, baseline=0.0, steps=24, return_signed=True)
        if isinstance(ig, dict) and "signed_map" in ig:
            signed = np.array(ig["signed_map"], dtype=np.float32)
            roi_contrib = compute_roi_contributions(signed)
    except Exception as e:
        ig = {"error": str(e)}

    probs_list = [
        {"index": int(i), "label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), "probability": float(p)}
        for i, p in enumerate(probs)
    ]

    # Prefer HTML/CSS-based PDF for professional layout
    orig_data_uri = None
    try:
        # use source image as data URI for consistent embedding
        buf_img = io.BytesIO()
        pil_img.resize(config.INPUT_SIZE).save(buf_img, format="PNG")
        import base64 as _b64
        orig_data_uri = f"data:image/png;base64,{_b64.b64encode(buf_img.getvalue()).decode('utf-8')}"
    except Exception:
        orig_data_uri = heatmap_data_uri

    pdf_bytes = build_html_pdf(
        original_img_data_uri=orig_data_uri,
        heatmap_data_uri=heatmap_data_uri,
        pred_label=pred_label,
        confidence=confidence,
        probabilities=probs_list,
        explanations={
            "occlusion": occlusion,
            "integratedGradients": ig,
            "roiContributions": roi_contrib,
        },
        extra={
            "top2": top2_items,
            "margin": margin,
        }
    )

    from flask import Response
    resp = Response(pdf_bytes, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f"attachment; filename=alzheimer_report_{pred_label.replace(' ', '_')}.pdf"
    return resp

# For WSGI servers
application = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, use_reloader=False)
