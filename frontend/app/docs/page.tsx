import Link from 'next/link'

export const metadata = {
  title: 'API Docs · Alzheimer AI',
  description: 'Backend API documentation for Alzheimer AI',
}

export default function DocsPage() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-slate-900">Backend API Documentation</h1>
        <p className="mt-1 text-sm text-slate-600">Everything you need to integrate the Alzheimer AI backend.</p>
      </div>

      <section className="grid gap-6">
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold">Base URL</h2>
          </div>
          <div className="px-5 py-4 text-sm text-slate-700">
            <p>Local development default: <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">http://localhost:5000</code></p>
            <p className="mt-2">Ensure the Flask server is running. From the repo root:</p>
            <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded text-slate-800 overflow-auto"><code>{`cd backend
python -m flask --app app:app run --host 0.0.0.0 --port 5000`}</code></pre>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold">Authentication</h2>
          </div>
          <div className="px-5 py-4 text-sm text-slate-700">
            <p>No authentication is required for local use. For production, place the API behind your preferred gateway and add auth there.</p>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold">Endpoints</h2>
          </div>
          <div className="px-5 py-4 text-sm text-slate-700">
            <div className="space-y-8">
              <div>
                <h3 className="text-base font-semibold">GET /health</h3>
                <p className="mt-1 text-slate-600">Check server health and model availability.</p>
                <div className="mt-2">
                  <p className="font-medium">Response</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`{
  "status": "ok",
  "model_loaded": true,
  "model_path": "backend/trained_model/Alzheimer_Detection_model.h5"
}`}</code></pre>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Curl</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`curl -s http://localhost:5000/health`}</code></pre>
                </div>
              </div>

              <div>
                <h3 className="text-base font-semibold">GET /metadata</h3>
                <p className="mt-1 text-slate-600">Model metadata such as input size and class names.</p>
                <div className="mt-2">
                  <p className="font-medium">Response</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`{
  "input_size": [224, 224],
  "classes": ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"],
  "model_path": "backend/trained_model/Alzheimer_Detection_model.h5"
}`}</code></pre>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Curl</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`curl -s http://localhost:5000/metadata`}</code></pre>
                </div>
              </div>

              <div>
                <h3 className="text-base font-semibold">POST /predict</h3>
                <p className="mt-1 text-slate-600">Upload a brain MRI image to receive prediction, probabilities, and heatmap.</p>
                <div className="mt-2">
                  <p className="font-medium">Request</p>
                  <p className="text-slate-600">Multipart form-data with field <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">image</code> containing the file.</p>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Response (200)</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`{
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
}`}</code></pre>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Errors (400)</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`{ "error": "No image file uploaded under field 'image'" }`}</code></pre>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Curl</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`curl -s -X POST \
  -F image=@/path/to/mri.png \
  http://localhost:5000/predict | jq`}</code></pre>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Environment variables</p>
                  <ul className="mt-2 list-disc list-inside text-slate-700">
                    <li><code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">TEMP_SCALE</code>: Temperature scaling for calibration (default: 1.0).</li>
                    <li><code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">EXPLAIN_LEVEL</code>: Set to <code>full</code> to compute IG/occlusion; otherwise minimal.</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-base font-semibold">POST /report</h3>
                <p className="mt-1 text-slate-600">Generate a professional PDF report including prediction, probabilities, heatmap, and explanations.</p>
                <div className="mt-2">
                  <p className="font-medium">Request</p>
                  <p className="text-slate-600">Multipart form-data with field <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">image</code>.</p>
                </div>
                <div className="mt-3">
                  <p className="font-medium">Response</p>
                  <p className="text-slate-600">Returns <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">application/pdf</code> bytes with <code>Content-Disposition: attachment</code>.</p>
                  <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`curl -s -X POST \
  -F image=@/path/to/mri.png \
  http://localhost:5000/report \
  --output alzheimer_report.pdf`}</code></pre>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold">Integration Notes</h2>
          </div>
          <div className="px-5 py-4 text-sm text-slate-700">
            <ul className="list-disc list-inside space-y-2">
              <li>Use <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">multipart/form-data</code> for <code>/predict</code> and <code>/report</code>.</li>
              <li>Expect <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">data:image/png;base64</code> heatmaps for quick rendering in browsers.</li>
              <li>For performance, keep <code>EXPLAIN_LEVEL</code> minimal unless you need detailed explanations.</li>
              <li>Handle errors gracefully; the API returns helpful messages with <code>error</code> fields and status 400.</li>
            </ul>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold">Frontend Example (TypeScript)</h2>
          </div>
          <div className="px-5 py-4 text-sm text-slate-700">
            <pre className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded overflow-auto"><code>{`async function predict(imageFile: File) {
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
}`}</code></pre>
          </div>
        </div>
      </section>

      <div className="mt-10 text-xs text-slate-500">
        Looking for the code? See <Link href="/" className="underline hover:text-slate-700">Home</Link> and the backend source under <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800">backend/</code>.
      </div>
    </div>
  )
}
