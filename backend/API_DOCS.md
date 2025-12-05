# Alzheimer AI Backend API Documentation

This document describes the REST API exposed by the Flask backend. It is written for newcomers and includes copy-paste examples.

## Base URL

- Local development: `http://localhost:5000`
- Ensure the server is running.

```bash
cd backend
python -m flask --app app:app run --host 0.0.0.0 --port 5000
```

## Authentication

- No authentication for local use. Add auth (e.g., reverse proxy, API gateway) for production.

## Endpoints

### GET /health

Check server health and model availability.

- Response 200:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "backend/trained_model/Alzheimer_Detection_model.h5"
}
```

- Example:

```bash
curl -s http://localhost:5000/health
```

### GET /metadata

Return input size and class names.

- Response 200:

```json
{
  "input_size": [224, 224],
  "classes": ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"],
  "model_path": "backend/trained_model/Alzheimer_Detection_model.h5"
}
```

- Example:

```bash
curl -s http://localhost:5000/metadata
```

### POST /predict

Upload an MRI image to get prediction, probabilities, and a heatmap.

- Request: `multipart/form-data` with field `image` containing the file.
- Response 200:

```json
{
  "predicted_index": 2,
  "predicted_label": "Mild Demented",
  "confidence": 0.82,
  "probabilities": [
    { "index": 0, "label": "Non Demented", "probability": 0.03 },
    { "index": 1, "label": "Very Mild Demented", "probability": 0.10 },
    { "index": 2, "label": "Mild Demented", "probability": 0.82 },
    { "index": 3, "label": "Moderate Demented", "probability": 0.05 }
  ],
  "top2": [
    { "index": 2, "label": "Mild Demented", "probability": 0.82 },
    { "index": 1, "label": "Very Mild Demented", "probability": 0.10 }
  ],
  "margin": 0.72,
  "heatmap": "data:image/png;base64,...",
  "inputSize": [224, 224],
  "classes": ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"],
  "explanations": {
    "level": "full",
    "occlusion": { /* may include details or an error */ },
    "integratedGradients": { /* may include details or an error */ },
    "roiContributions": { /* region importance if available */ }
  }
}
```

- Errors 400:

```json
{ "error": "No image file uploaded under field 'image'" }
```

- Example:

```bash
curl -s -X POST \
  -F image=@/path/to/mri.png \
  http://localhost:5000/predict | jq
```

- Environment:
  - `TEMP_SCALE`: Temperature scaling for calibration (default: 1.0)
  - `EXPLAIN_LEVEL`: Set to `full` for IG/occlusion; otherwise minimal

### POST /report

Generate a professional PDF report.

- Request: `multipart/form-data` with field `image`.
- Response: `application/pdf` bytes with `Content-Disposition: attachment`.

- Example:

```bash
curl -s -X POST \
  -F image=@/path/to/mri.png \
  http://localhost:5000/report \
  --output alzheimer_report.pdf
```

## Frontend Integration (TypeScript)

```ts
async function predict(imageFile: File) {
  const form = new FormData();
  form.append('image', imageFile);
  const res = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Prediction failed');
  return res.json();
}

async function downloadReport(imageFile: File) {
  const form = new FormData();
  form.append('image', imageFile);
  const res = await fetch('http://localhost:5000/report', {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Report generation failed');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'alzheimer_report.pdf';
  a.click();
  URL.revokeObjectURL(url);
}
```
