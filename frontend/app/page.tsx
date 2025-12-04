"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

type ProbItem = { index: number; label: string; probability: number };

type PredictResponse = {
  predicted_index: number;
  predicted_label: string;
  confidence: number;
  probabilities: ProbItem[];
  heatmap: string;
  inputSize: [number, number];
  classes: string[];
  explanations?: {
    level: "basic" | "full" | string;
    occlusion?: {
      predicted_index: number;
      base_confidence: number;
      regions: { bbox: [number, number, number, number]; delta: number }[];
      grid: { patchSize: number; stride: number; height: number; width: number };
      error?: string;
    } | null;
    integratedGradients?: {
      target_index: number;
      height: number;
      width: number;
      map?: number[][];
      summary?: { mean: number; std: number; max: number; min: number };
      error?: string;
    } | null;
    roiContributions?: {
      name: string;
      bbox: [number, number, number, number];
      positivePercent: number;
      negativePercent: number;
      direction: "increased" | "decreased" | string;
    }[] | null | { error: string };
  };
};

function SimpleBarChart({ data }: { data: ProbItem[] }) {
  // Render a simple responsive bar chart using divs with black labels beneath
  return (
    <div>
      <div className="bar-chart" aria-hidden>
        {data.map((d) => {
          const h = Math.max(2, Math.round(d.probability * 100)); // 0-100
          return (
            <div className="bar" key={d.index}>
              <div className="bar-rect" style={{ height: 120 }}>
                <div
                  className="bar-fill"
                  style={{ height: `${h}%`, transitionDelay: `${d.index * 30}ms` }}
                  title={`${d.label}: ${(d.probability * 100).toFixed(1)}%`}
                />
              </div>
              <div className="bar-label">{d.label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showOcclusionOverlay, setShowOcclusionOverlay] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const igCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const ig = result?.explanations?.integratedGradients;
    if (!ig?.map || !igCanvasRef.current) return;
    const h = ig.height || ig.map.length;
    const w = ig.width || (ig.map[0]?.length ?? 0);
    const canvas = igCanvasRef.current;
    const size = 160; // thumbnail
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = Math.max(0, Math.min(1, ig.map[y][x] ?? 0));
        const i = (y * w + x) * 4;
        const r = Math.round(255 * v);
        const g = Math.round(255 * (1 - v) * 0.2);
        const b = Math.round(255 * (1 - v) * 0.2);
        imgData.data[i + 0] = r;
        imgData.data[i + 1] = g;
        imgData.data[i + 2] = b;
        imgData.data[i + 3] = 255;
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    const tctx = tmp.getContext("2d");
    if (!tctx) return;
    tctx.putImageData(imgData, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(tmp, 0, 0, size, size);
  }, [result?.explanations?.integratedGradients]);

  const title = useMemo(() => {
    if (!result) return "Alzheimer MRI Analysis";
    return `Prediction: ${result.predicted_label} — ${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

  const handleFile = (f: File | null) => {
    setFile(f);
    setError(null);
    setResult(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  };

  const onChangeFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    handleFile(f);
    // Do not auto-start; user clicks Analyze after upload
  };

  const onDrop = (ev: React.DragEvent) => {
    ev.preventDefault();
    ev.stopPropagation();
    setDragOver(false);
    const f = ev.dataTransfer.files?.[0] || null;
    if (f) handleFile(f);
  };

  const onSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setError(null);
    setResult(null);
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    const form = new FormData();
    form.append("image", file);
    setLoading(true);
    // Simulated determinate progress: smoothly increases to ~90% while loading
    setProgress(0);
    let prog = 0;
    const tick = () => {
      // Ease towards 90% with diminishing increments
      const target = 90;
      const remaining = target - prog;
      const inc = Math.max(0.5, remaining * 0.05); // smaller as it approaches target
      prog = Math.min(target, prog + inc);
      setProgress(Math.round(prog));
    };
    const progTimer = setInterval(tick, 120);
    try {
      // small UX: animate spinner for at least 600ms for perceived polish
      const start = Date.now();
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Request failed: ${res.status}`);
      }
      const data = (await res.json()) as PredictResponse;
      const elapsed = Date.now() - start;
      if (elapsed < 600) await new Promise((r) => setTimeout(r, 600 - elapsed));
      setResult(data);
      // Fill to 100% on completion
      setProgress(100);
      await new Promise((r) => setTimeout(r, 250));
    } catch (err: any) {
      setError(err.message || "Request failed");
      // On error, still complete briefly to avoid abrupt stop
      setProgress(100);
      await new Promise((r) => setTimeout(r, 200));
    } finally {
      clearInterval(progTimer);
      setLoading(false);
      // Reset progress shortly after hiding overlay
      setTimeout(() => setProgress(0), 200);
    }
  };

  const onDownloadReport = async () => {
    if (!file) return;
    try {
      setReportLoading(true);
      const form = new FormData();
      form.append("image", file);
      const res = await fetch(`${BACKEND_URL}/report`, { method: "POST", body: form });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Report request failed: ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `alzheimer_report_${result?.predicted_label?.replace(/\s+/g, "_") || "analysis"}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("Failed to download report.");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <main className="max-content mt-8 pb-12">
      <div className="container-card">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="text-2xl font-semibold mb-1">{title}</h1>
            <p className="text-sm text-slate-500">Upload a brain MRI image to run classification and view explainability. For clinician review and reporting.</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-xs text-slate-600">Model v1.2 • trained on combined cohort</div>
          </div>
        </div>

        {/* Controls: separate upload and analyze from dropzone */}
        <div className="mt-6 flex flex-wrap items-center gap-3">
          {/* <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              const input = document.getElementById("file-input") as HTMLInputElement | null;
              if (input) {
                input.value = ""; // allow reselecting same file
                input.click();
              }
            }}
          >
            Choose MRI
          </button> */}
          <button
            type="button"
            className="btn-primary"
            disabled={loading || !file}
            onClick={() => onSubmit()}
          >
            {loading ? <span className="flex items-center gap-2 loading-overlay"><span className="spinner" />Analyzing…</span> : "Analyze Image"}
          </button>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={onChangeFile}
            className="hidden"
          />
        </div>

        <form onSubmit={(e) => onSubmit(e)} className="mt-3 grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
          <div className="lg:col-span-2">
            <div
              className={`dropzone ${dragOver ? "dragover" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragOver(false);
              }}
              onDrop={onDrop}
              role="button"
              aria-label="Upload brain MRI (drag & drop or click)"
              title="Drag & drop or click to choose file"
            >
              <div style={{ width: 56, height: 56, borderRadius: 10, background: "linear-gradient(180deg,#0e7490,#075985)", display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontWeight: 700 }}>
                MRI
              </div>
              <div>
                <div className="text-sm font-medium">Drop image here or click to choose</div>
                <div className="text-xs text-slate-500">Supported: jpg, png. Prefer axial slices, 224×224 or higher.</div>
              </div>
              {/* Clicking inside dropzone also opens file picker for convenience */}
              <div style={{ marginLeft: "auto" }}>
                <button type="button" className="btn-compact" onClick={() => {
                  const input = document.getElementById("file-input") as HTMLInputElement | null;
                  if (input) { input.value = ""; input.click(); }
                }}>Upload MRI</button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-1 text-right">
            <div className="card inline-flex items-center gap-3">
              <div className="flex flex-col items-end">
                <div className="text-xs text-slate-500">Predicted</div>
                <div className="text-sm font-semibold">{result?.predicted_label ?? "—"}</div>
              </div>
              <div className="px-3 py-2 bg-slate-50 rounded-md">
                <div className="text-xs text-slate-500">Confidence</div>
                <div className="font-semibold text-lg">{result ? `${(result.confidence * 100).toFixed(1)}%` : "—"}</div>
              </div>
            </div>
          </div>
        </form>

        <div className="analysis-grid mt-6">
          <div className="space-y-4">
            <div className="card grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="section-title mb-3">Uploaded Image</h3>
                <div className={`w-full h-72 bg-slate-50 rounded-md overflow-hidden flex items-center justify-center border border-slate-100 ${loading ? 'skeleton' : ''}`}>
                  {preview ? <img src={preview} alt="preview" className="object-contain h-full w-full" /> : <div className="text-sm text-slate-500">No image selected</div>}
                </div>
              </div>
              <div>
                <h3 className="section-title mb-3">Grad-CAM Heatmap</h3>
                <div className={`w-full h-72 rounded-md overflow-hidden border border-slate-100 relative ${loading ? 'skeleton' : ''}`}>
                  {result?.heatmap ? (
                    <>
                      <img src={result.heatmap} alt="heatmap" className="object-cover w-full h-full" />
                      {showOcclusionOverlay && result?.explanations?.occlusion?.regions?.length ? (
                        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${result.inputSize[0]} ${result.inputSize[1]}`} preserveAspectRatio="none">
                          {result.explanations.occlusion.regions.slice(0, 6).map((r, idx) => {
                            const [x, y, w, h] = r.bbox;
                            return (
                              <g key={idx}>
                                <rect x={x} y={y} width={w} height={h} fill="none" stroke="#FFD54F" strokeWidth={2} rx={4} />
                                <text x={x + 6} y={y + 14} fill="#FFD54F" fontSize={12} fontWeight={700}>{`Δ ${(r.delta * 100).toFixed(1)}%`}</text>
                              </g>
                            );
                          })}
                        </svg>
                      ) : null}
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-full text-sm text-slate-500">Heatmap will appear after prediction</div>
                  )}
                  {loading ? (
                    <div className="analyzing-overlay">
                      <div className="progress" aria-label="Analysis progress">
                        <div className="progress-bar" style={{ width: `${progress}%` }} />
                      </div>
                      <div className="text-xs text-slate-700 mt-2">Analyzing MRI…</div>
                    </div>
                  ) : null}
                </div>

                <div className="flex items-center justify-between mt-3">
                  <div className="text-xs text-slate-500">Explainability mode: {result?.explanations?.level ?? "basic"}</div>
                  <label className="text-xs inline-flex items-center gap-2">
                    <input type="checkbox" checked={showOcclusionOverlay} onChange={(e) => setShowOcclusionOverlay(e.target.checked)} />
                    Show occlusion regions
                  </label>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="section-title mb-2">Reasons & Explainability</h3>
              {result ? (
                <div className="explain-grid">
                  <div>
                    <div className="text-sm text-slate-700">The heatmap and occlusion tests highlight the image regions influencing the model. Below are quantitative and visual explainers to aid interpretation.</div>

                    <div className="mt-4">
                      <div className="text-sm font-medium">Occlusion (top impactful patches)</div>
                      <ul className="text-sm list-disc pl-5 mt-2 text-slate-700">
                        {result.explanations?.occlusion?.regions?.slice(0, 5).map((r, idx) => {
                          const [x, y, w, h] = r.bbox;
                          return <li key={idx}>{`Patch (${x},${y}), ${w}×${h} — Δ ${(r.delta * 100).toFixed(1)}%`}</li>;
                        }) ?? <li className="text-slate-500">Unavailable</li>}
                      </ul>
                    </div>

                    <div className="mt-4">
                      <div className="text-sm font-medium">Integrated Gradients (summary)</div>
                      <div className="text-xs text-slate-500 mt-1">{result.explanations?.integratedGradients?.summary ? `mean ${(result.explanations.integratedGradients.summary.mean).toFixed(3)} • std ${(result.explanations.integratedGradients.summary.std).toFixed(3)}` : "Unavailable"}</div>
                      {result.explanations?.integratedGradients?.map ? (
                        <div className="mt-3">
                          <canvas ref={igCanvasRef} className="ig-thumb" />
                          <div className="text-xs text-slate-500 mt-2">IG magnitude thumbnail</div>
                        </div>
                      ) : null}
                    </div>
                  </div>

                  <aside>
                    <div className="text-sm font-medium mb-2">Region contributions (atlas)</div>
                    <div className="bg-slate-50 rounded-md p-3">
                      {Array.isArray(result.explanations?.roiContributions) ? (
                        <>
                          <ul className="text-sm list-disc pl-4 text-slate-700">
                            {result.explanations!.roiContributions!.slice(0, 6).map((r, idx) => (
                              <li key={idx}>{`${r.name}: ${(r.direction === 'increased' ? r.positivePercent : r.negativePercent).toFixed(1)}% (${r.direction})`}</li>
                            ))}
                          </ul>
                          {/* Simple bar chart for ROI contributions */}
                          <div className="mt-3">
                            {result.explanations!.roiContributions!.slice(0, 5).map((r, idx) => {
                              const pct = r.direction === 'increased' ? r.positivePercent : r.negativePercent;
                              return (
                                <div key={idx} className="mb-1">
                                  <div className="text-[11px] text-slate-600 mb-0.5">{r.name}</div>
                                  <div className="w-full h-2 rounded bg-slate-200">
                                    <div className="h-2 rounded bg-[var(--accent-600)]" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </>
                      ) : (
                        <div className="text-xs text-slate-500">ROI contributions unavailable</div>
                      )}
                    </div>
                  </aside>
                </div>
              ) : (
                <div className="text-sm text-slate-500">Run an analysis to reveal explainability insights and region contributions.</div>
              )}
            </div>
          </div>

          <aside className="sticky-panel">
            <div className="card">
              <h3 className="section-title mb-3">Analytics</h3>

              {result ? (
                <>
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="text-xs text-slate-500">Predicted</div>
                      <div className="font-semibold">{result.predicted_label}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-slate-500">Confidence</div>
                      <div className="text-lg font-semibold">{(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  <div className="analytics-chart mb-5">
                    <SimpleBarChart data={result.probabilities} />
                  </div>

                  <div className="hr" />

                  <div className="text-sm font-medium mb-2">Suggested next steps</div>
                  <ul className="text-sm list-disc pl-5 text-slate-700">
                    <li>Combine with clinical cognitive tests (MMSE, CDR)</li>
                    <li>If confidence &lt; 60%: recommend second read</li>
                    <li>Consider hippocampal volumetry for follow-up</li>
                  </ul>

                  {/* Contextual summary synthesized from explainability to use remaining space */}
                  <div className="hr" />
                  <div className="text-sm font-medium mb-2">Summary</div>
                  {(() => {
                    const bullets: string[] = [];
                    const lbl = result.predicted_label;
                    const confPct = (result.confidence * 100).toFixed(1);
                    bullets.push(`Model indicates ${lbl} (confidence ${confPct}%).`);
                    const ex = result.explanations;
                    if (ex?.level === 'full') {
                      const topPatch = ex.occlusion?.regions?.[0];
                      if (topPatch) bullets.push(`Top occlusion patch at (${topPatch.bbox[0]},${topPatch.bbox[1]}) reduces confidence by ${(topPatch.delta * 100).toFixed(1)}%.`);
                      if (Array.isArray(ex.roiContributions) && ex.roiContributions.length) {
                        const roi = ex.roiContributions[0];
                        const pct = (roi.direction === 'increased' ? roi.positivePercent : roi.negativePercent).toFixed(1);
                        bullets.push(`${roi.name} contributed ${pct}% (${roi.direction} probability).`);
                      }
                      const ig = ex.integratedGradients?.summary;
                      if (ig) bullets.push(`IG magnitude: mean ${ig.mean.toFixed(3)}, std ${ig.std.toFixed(3)}.`);
                    } else {
                      bullets.push('Enable full explainability to view detailed reasons.');
                    }
                    return (
                      <ul className="text-sm list-disc pl-5 text-slate-700">
                        {bullets.map((b, i) => (<li key={i}>{b}</li>))}
                      </ul>
                    );
                  })()}

                  {/* Move Quick facts here to occupy remaining space elegantly */}
                  <div className="hr" />
                  <div className="text-sm font-medium mb-2">Quick facts</div>
                  <div className="text-xs text-slate-500">
                    <div>Model version: v1.2</div>
                    <div>Threshold suggestion: if confidence &lt; 60%, request second read</div>
                    <div className="mt-3">
                      <button className="btn-primary w-full" onClick={onDownloadReport} disabled={reportLoading}>
                        {reportLoading ? <span className="flex items-center gap-2"><span className="spinner" /> Preparing PDF…</span> : "Download Report (PDF)"}
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-sm text-slate-500">Run an analysis to see probabilities, ROI contributions and clinical suggestions.</div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
