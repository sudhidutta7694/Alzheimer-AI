from __future__ import annotations
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from weasyprint import HTML, CSS

# Minimal inline CSS for a professional medical report layout
BASE_CSS = CSS(string="""
@page { size: A4; margin: 18mm; }
body { font-family: 'Inter', 'IBM Plex Sans', Arial, sans-serif; color: #0f172a; }
.header { display:flex; align-items:center; justify-content:space-between; margin-bottom: 12px; }
.header .title { font-size: 20px; font-weight: 700; }
.header .ts { font-size: 11px; color: #475569; }
.divider { height:1px; background: linear-gradient(90deg, rgba(2,6,23,0.06), rgba(2,6,23,0)); margin: 10px 0 16px; }
.summary { display:flex; align-items:center; justify-content:space-between; margin-bottom: 12px; }
.summary .label { font-size: 13px; font-weight: 600; }
.summary .value { font-size: 13px; }
.note { border: 1px solid rgba(15,23,42,0.08); border-radius: 6px; padding: 8px; font-size: 11px; color: #334155; background: #f8fafc; }
.row { display:flex; gap: 12px; margin-top: 12px; }
.card { border: 1px solid rgba(15,23,42,0.06); border-radius: 8px; padding: 10px; background: white; }
.section-title { font-size: 14px; font-weight: 700; margin-bottom: 8px; }
.img { width: 100%; height: auto; border-radius: 6px; border: 1px solid rgba(15,23,42,0.06); }
.grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.small { font-size: 12px; color: #475569; }
.list { font-size: 12px; color: #0f172a; }
.bar { width: 100%; height: 8px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; }
.bar .fill { height: 8px; background: #0e7490; border-radius: 9999px; }
.bar .fill.neg { background: #cc5151; }
.hr { height:1px; background: linear-gradient(90deg, rgba(2,6,23,0.06), rgba(2,6,23,0)); margin: 10px 0; }
.footer-note { font-size: 10px; color: #64748b; margin-top: 12px; }
.logo { width: 28px; height: 28px; object-fit: contain; border-radius: 6px; }
""")


def _img_data_uri_from_pil(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _resolve_logo_data_uri() -> str | None:
    candidates = [
        Path(__file__).resolve().parent.parent / 'frontend' / 'assets' / 'logo.png',
        Path(__file__).resolve().parent / 'assets' / 'logo.png',
        Path.cwd() / 'frontend' / 'assets' / 'logo.png',
        Path.cwd() / 'assets' / 'logo.png',
    ]
    for p in candidates:
        if p.exists():
            b = p.read_bytes()
            return f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}"
    return None


def build_html_pdf(
    original_img_data_uri: str,
    heatmap_data_uri: str,
    pred_label: str,
    confidence: float,
    probabilities: List[Dict[str, Any]],
  explanations: Dict[str, Any] | None,
  extra: Dict[str, Any] | None = None,
) -> bytes:
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    logo_uri = _resolve_logo_data_uri()

    # Build probabilities bars
    prob_rows = []
    for p in probabilities:
        lbl = p.get('label', str(p.get('index', '')))
        val = float(p.get('probability', 0.0))
        width_pct = max(0, min(100, round(val * 100)))
        prob_rows.append(f"""
        <div class=\"list\" style=\"margin-bottom:6px\">{lbl}
          <div class=\"bar\"><div class=\"fill\" style=\"width:{width_pct}%\"></div></div>
        </div>
        """)

    # Occlusion list
    occl_html = "<div class=\"small\">Occlusion analysis not available.</div>"
    if explanations and isinstance(explanations.get('occlusion'), dict) and explanations['occlusion'].get('regions'):
        items = []
        for r in explanations['occlusion']['regions'][:5]:
            x, y, w, h = r['bbox']
            items.append(f"<li>Patch at ({x},{y}) size {w}×{h}: Δ {(r['delta']*100):.1f}%</li>")
        occl_html = f"<ul class=\"list\">{''.join(items)}</ul>"

    # IG summary + optional thumbnail
    ig = explanations.get('integratedGradients') if explanations else None
    ig_summary_html = "<div class=\"small\">Integrated Gradients summary unavailable.</div>"
    ig_thumb_html = ""
    if isinstance(ig, dict):
        if ig.get('summary'):
            s = ig['summary']
            ig_summary_html = f"<div class=\"small\">mean {s['mean']:.3f} • std {s['std']:.3f}</div>"
        if ig.get('map'):
            # Render IG map as heat image
            import numpy as np
            from PIL import Image
            arr = np.array(ig['map'], dtype=float)
            arr = np.clip(arr, 0, 1)
            rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
            rgb[...,0] = (255*arr).astype(np.uint8)
            rgb[...,1] = (255*(1-arr)*0.2).astype(np.uint8)
            rgb[...,2] = (255*(1-arr)*0.2).astype(np.uint8)
            pil = Image.fromarray(rgb, 'RGB')
            ig_thumb_html = f"<img class=\"img\" style=\"max-width:160px\" src=\"{_img_data_uri_from_pil(pil)}\" alt=\"IG thumbnail\"/>"

    # ROI list + bars
    roi_html = "<div class=\"small\">ROI contributions unavailable.</div>"
    roi_bars = ""
    rc = explanations.get('roiContributions') if explanations else None
    if isinstance(rc, list) and rc:
        items = []
        bars = []
        for r in rc[:6]:
            pct = r['positivePercent'] if r['direction'] == 'increased' else r['negativePercent']
            items.append(f"<li>{r['name']}: {pct:.1f}% ({r['direction']} probability)</li>")
        for r in rc[:5]:
            pct = r['positivePercent'] if r['direction'] == 'increased' else r['negativePercent']
            cl = '' if r['direction'] == 'increased' else 'neg'
            width_pct = max(0, min(100, round(pct)))
            bars.append(f"<div class=\"small\" style=\"margin-bottom:4px\">{r['name']}<div class=\"bar\"><div class=\"fill {cl}\" style=\"width:{width_pct}%\"></div></div></div>")
        roi_html = f"<ul class=\"list\">{''.join(items)}</ul>"
        roi_bars = ''.join(bars)

    # Synthesized summary bullets
    bullets = [f"Model indicates {pred_label} (confidence {confidence*100:.1f}%)."]
    if extra:
      top2 = extra.get('top2') or []
      margin = extra.get('margin')
      if isinstance(top2, list) and len(top2) >= 2:
        bullets.append(f"Top-2: {top2[0]['label']} ({top2[0]['probability']*100:.1f}%), {top2[1]['label']} ({top2[1]['probability']*100:.1f}%). Margin: {(margin*100):.1f}%.")
    if explanations:
        occl = explanations.get('occlusion')
        if isinstance(occl, dict) and occl.get('regions'):
            top = occl['regions'][0]
            bullets.append(f"Top occlusion patch at ({top['bbox'][0]},{top['bbox'][1]}) reduces confidence by {(top['delta']*100):.1f}%.")
        if isinstance(rc, list) and rc:
            r0 = rc[0]
            p0 = r0['positivePercent'] if r0['direction'] == 'increased' else r0['negativePercent']
            bullets.append(f"{r0['name']} contributed {p0:.1f}% ({r0['direction']} probability).")
        if isinstance(ig, dict) and ig.get('summary'):
            s = ig['summary']
            bullets.append(f"IG magnitude: mean {s['mean']:.3f}, std {s['std']:.3f}.")
    summary_bullets = ''.join([f"<li>{b}</li>" for b in bullets])

    logo_img = f"<img class=\"logo\" src=\"{logo_uri}\" alt=\"Logo\"/>" if logo_uri else ""

    html = f"""
    <html>
      <head><meta charset=\"utf-8\" /></head>
      <body>
        <div class=\"header\">
          <div style=\"display:flex;align-items:center;gap:8px\">{logo_img}<div class=\"title\">Alzheimer MRI Analysis Report</div></div>
          <div class=\"ts\">Generated: {ts}</div>
        </div>
        <div class=\"divider\"></div>

        <div class=\"summary\">
          <div><span class=\"label\">Prediction:</span> <span class=\"value\">{pred_label}</span></div>
          <div><span class=\"label\">Confidence:</span> <span class=\"value\">{confidence*100:.2f}%</span></div>
        </div>
        {('<div class=\'small\'>Top-2 classes shown in Summary below with margin.</div>') if extra else ''}
        <div class=\"note\">For clinical decision support only. Not a diagnostic device.</div>

        <div class=\"row\">
          <div class=\"card\" style=\"flex:1\">
            <div class=\"section-title\">Original MRI</div>
            <img class=\"img\" src=\"{original_img_data_uri}\" alt=\"Original MRI\" />
          </div>
          <div class=\"card\" style=\"flex:1\">
            <div class=\"section-title\">Grad-CAM Heatmap</div>
            <img class=\"img\" src=\"{heatmap_data_uri}\" alt=\"Heatmap\" />
          </div>
        </div>

        <div class=\"card\" style=\"margin-top:12px\">
          <div class=\"section-title\">Quantitative Explainability</div>
          <div class=\"grid-2\">
            <div>
              <div class=\"small\" style=\"font-weight:600\">Occlusion (top patches)</div>
              {occl_html}
              <div class=\"hr\"></div>
              <div class=\"small\" style=\"font-weight:600\">Integrated Gradients</div>
              {ig_summary_html}
              <div style=\"margin-top:6px\">{ig_thumb_html}</div>
            </div>
            <div>
              <div class=\"small\" style=\"font-weight:600\">Region contributions (atlas)</div>
              {roi_html}
              <div style=\"margin-top:6px\">{roi_bars}</div>
            </div>
          </div>
        </div>

        <div class=\"card\" style=\"margin-top:12px\">
          <div class=\"section-title\">Class Probabilities</div>
          {''.join(prob_rows)}
        </div>

        <div class=\"card\" style=\"margin-top:12px\">
          <div class=\"section-title\">Summary</div>
          <ul class=\"list\">{summary_bullets}</ul>
        </div>

        <div class=\"card\" style=\"margin-top:12px\">
          <div class=\"section-title\">Quick Facts</div>
          <div class=\"small\">Model version: v1.2</div>
          <div class=\"small\">Threshold suggestion: if confidence < 60%, request second read</div>
          <div class=\"footer-note\">Explainability outputs are model-based and approximate; interpret with clinical context.</div>
        </div>
      </body>
    </html>
    """

    pdf_bytes = HTML(string=html).write_pdf(stylesheets=[BASE_CSS])
    return pdf_bytes
