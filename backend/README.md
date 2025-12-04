# Backend Run Guide

This service loads the trained Keras/TensorFlow model and serves predictions and explanations.

## Environment variables
- `MODEL_PATH`: Absolute path to the H5/SavedModel.
- `EXPLAIN_LEVEL`: `basic` (default) returns Grad-CAM or input-gradient heatmap; `full` additionally computes occlusion sensitivity and integrated gradients (heavier CPU/memory).

Optional TensorFlow tuning to reduce memory usage:
- `TF_NUM_INTEROP_THREADS=1`
- `TF_NUM_INTRAOP_THREADS=1`
- `OMP_NUM_THREADS=1`

## Start (basic mode)
```zsh
source .venv/bin/activate
fuser -k 5000/tcp || true
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 OMP_NUM_THREADS=1 \
EXPLAIN_LEVEL=basic MODEL_PATH="/home/sudhi-sundar-dutta/Desktop/Alzheimer_AI/trained_model/Alzheimer_Detection_model.h5" \
python -m backend.app
```

## Start (full explanations)
Use only if sufficient memory is available.
```zsh
source .venv/bin/activate
fuser -k 5000/tcp || true
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 OMP_NUM_THREADS=1 \
EXPLAIN_LEVEL=full MODEL_PATH="/home/sudhi-sundar-dutta/Desktop/Alzheimer_AI/trained_model/Alzheimer_Detection_model.h5" \
python -m backend.app
```

## Test prediction
```zsh
curl -sS -X POST http://127.0.0.1:5000/predict \
  -F "image=@/home/sudhi-sundar-dutta/Desktop/Alzheimer_AI/sample_mri.png" | jq '.predicted_label, .confidence, .explanations.level'
```

The response includes:
- `predicted_label`, `confidence`, `probabilities`
- `heatmap` (data URI of PNG overlay)
- `explanations` object with `level`, and when full: `occlusion` regions and `integratedGradients` summary/map.Backend (Flask)

Endpoints
- GET /health: Service and model status
- GET /metadata: Input size and class names
- POST /predict: Multipart form with field `image` -> returns prediction JSON with Grad-CAM heatmap

Model
- Default model path: `../trained_model/Alzheimer_Detection_model.h5`
- Override via env: `MODEL_PATH=/abs/path/to/model.h5`

Class names
- Edit `backend/class_names.json` to match the exact order used during training. If unknown, set them carefully to avoid mislabeling.

Setup
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.app
```

POST example
```zsh
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/mri.jpg"
```
