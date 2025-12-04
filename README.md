# Alzheimer AI

Clinical-grade Alzheimer MRI classification with explainability, analytics, and professional PDF reporting.

## Features
- Flask backend with TensorFlow/Keras model inference
- Grad-CAM heatmap, occlusion sensitivity, integrated gradients, ROI contributions
- HTML/CSS WeasyPrint PDF report: images, charts, reasons, summary, quick facts, logo
- Next.js frontend with modern clinical UI and determinate analyzing animations

## Prerequisites
- Python 3.10+
- Node.js 18+
- Git

## Backend Setup
```zsh
# From repo root
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Optional: set temperature scaling for calibration (try 0.9–1.2)
export TEMP_SCALE=0.9

# Run server
python -m flask run
# Backend runs at http://localhost:5000
```

### Model
Place your trained model at:
```
trained_model/Alzheimer_Detection_model.h5
```
Update `backend/config.py` if your path differs.

## Frontend Setup
```zsh
# From repo root
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:3000
```
Set `NEXT_PUBLIC_BACKEND_URL` in `frontend/.env.local` if your backend is not at `http://localhost:5000`.

## Usage
1. Open the frontend, upload a brain MRI image.
2. Click "Analyze Image" to run predictions and view explainability.
3. Click "Download Report (PDF)" for a professional report.

## Report Contents
- Header with logo and timestamp
- Original MRI and Grad-CAM heatmap
- Quantitative Explainability
	- Occlusion patches list (top impacts)
	- Integrated Gradients summary and thumbnail
	- ROI contributions list and colored bars
- Class probabilities bars
- Summary bullets including top-2 classes and margin
- Quick facts and clinical note

## Troubleshooting
- Confidence <50%: consider calibration (`TEMP_SCALE`), test-time augmentation, and verify preprocessing alignment.
- Logo missing in PDF: ensure `frontend/assets/logo.png` exists; backend resolves multiple candidate paths.
- Memory issues: reduce explainability level via `EXPLAIN_LEVEL=basic` (environment variable).

## License
Proprietary — internal use.
