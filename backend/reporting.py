from __future__ import annotations
import io
from datetime import datetime
from typing import Any, List, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from PIL import Image, ImageDraw
import numpy as np

from .gradcam import overlay_heatmap_on_image


def _pil_from_data_uri(data_uri: str) -> Image.Image:
    import base64
    import re

    m = re.match(r"^data:image/[^;]+;base64,(.*)$", data_uri)
    if not m:
        raise ValueError("Invalid data URI")
    b = base64.b64decode(m.group(1))
    return Image.open(io.BytesIO(b)).convert("RGBA")


def _draw_occlusion_boxes(base: Image.Image, regions: List[Dict[str, Any]]) -> Image.Image:
    im = base.copy()
    draw = ImageDraw.Draw(im)
    for r in regions[:5]:
        x, y, w, h = r["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 235, 59, 255), width=3)
        draw.text((x + 3, y + 3), f"Δ {(r['delta']*100):.1f}%", fill=(255, 235, 59, 255))
    return im


def build_pdf_report(
    original_img: Image.Image,
    heatmap_overlay_datauri: str,
    pred_label: str,
    confidence: float,
    probabilities: List[Dict[str, Any]],
    explanations: Dict[str, Any] | None,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Margins
    margin = 18 * mm
    left = margin
    right = W - margin
    top = H - margin
    y = top

    # Title + timestamp
    c.setFont("Helvetica-Bold", 18)
    # Try to draw logo (robust path resolution)
    try:
        from pathlib import Path
        here = Path(__file__).resolve().parent
        # common locations relative to backend/reporting.py
        candidate_paths = [
            here.parent / 'frontend' / 'assets' / 'logo.png',
            here / 'assets' / 'logo.png',
            Path.cwd() / 'frontend' / 'assets' / 'logo.png',
            Path.cwd() / 'assets' / 'logo.png',
        ]
        logo_file = next((p for p in candidate_paths if p.exists()), None)
        if logo_file:
            with open(logo_file, 'rb') as lf:
                logo_bytes = lf.read()
            logo_reader = ImageReader(io.BytesIO(logo_bytes))
            c.drawImage(logo_reader, left, y - 10 * mm, width=12 * mm, height=12 * mm, mask='auto')
            c.drawString(left + 14 * mm, y, "Alzheimer MRI Analysis Report")
        else:
            c.drawString(left, y, "Alzheimer MRI Analysis Report")
    except Exception:
        c.drawString(left, y, "Alzheimer MRI Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawRightString(right, y, datetime.utcnow().strftime("Generated: %Y-%m-%d %H:%M UTC"))
    y -= 8 * mm
    # subtle divider
    c.setStrokeColorRGB(0.85, 0.88, 0.92)
    c.line(left, y, right, y)
    y -= 8 * mm

    # Summary (box)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Patient Image Analysis Summary")
    y -= 6 * mm
    c.setFont("Helvetica", 11)
    c.drawString(left, y, f"Prediction: {pred_label}")
    c.drawRightString(right, y, f"Confidence: {confidence*100:.2f}%")
    y -= 8 * mm
    c.setStrokeColorRGB(0.9, 0.92, 0.95)
    c.roundRect(left, y - 12 * mm, right - left, 12 * mm, 3 * mm, stroke=1, fill=0)
    c.setFont("Helvetica", 9)
    c.drawString(left + 3 * mm, y - 9 * mm, "Note: Model inference to be interpreted with clinical context.")
    y -= 18 * mm

    # Images row (Original + Heatmap)
    row_y = y
    # Left: Original thumbnail
    orig_thumb = original_img.copy()
    orig_thumb.thumbnail((85 * mm, 85 * mm))
    ib = io.BytesIO()
    orig_thumb.save(ib, format="PNG")
    c.drawImage(ImageReader(io.BytesIO(ib.getvalue())), left, row_y - 90 * mm, width=85 * mm, height=85 * mm)

    # Right: Heatmap overlay
    try:
        heatmap_img = _pil_from_data_uri(heatmap_overlay_datauri)
        heatmap_img = heatmap_img.convert("RGBA")
    except Exception:
        heatmap_img = original_img.copy()
    heatmap_thumb = heatmap_img.copy()
    heatmap_thumb.thumbnail((85 * mm, 85 * mm))
    hb = io.BytesIO()
    heatmap_thumb.save(hb, format="PNG")
    c.drawImage(ImageReader(io.BytesIO(hb.getvalue())), left + 95 * mm, row_y - 90 * mm, width=85 * mm, height=85 * mm)

    y = row_y - 100 * mm
    # Section: Quantitative Explainability
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left, y, "Quantitative Explainability")
    y -= 6 * mm
    c.setFont("Helvetica", 10)

    # Occlusion regions
    if explanations and explanations.get("occlusion") and isinstance(explanations["occlusion"], dict) and explanations["occlusion"].get("regions"):
        c.drawString(left, y, "Top occlusion patches (reduced probability when removed):")
        y -= 5 * mm
        for r in explanations["occlusion"]["regions"][:5]:
            x, yy, w, h = r["bbox"]
            c.drawString(left + 4 * mm, y, f"Patch at ({x},{yy}) size {w}x{h}: Δ {(r['delta']*100):.1f}%")
            y -= 4 * mm
    else:
        c.drawString(left, y, "Occlusion analysis not available.")
        y -= 5 * mm

    # Integrated Gradients summary and thumbnail
    ig = explanations.get("integratedGradients") if explanations else None
    if isinstance(ig, dict) and ig.get("summary"):
        s = ig["summary"]
        c.drawString(left, y, f"Integrated Gradients: mean {s['mean']:.3f} • std {s['std']:.3f}")
        y -= 5 * mm
    if isinstance(ig, dict) and ig.get("map"):
        try:
            ig_map = np.array(ig["map"], dtype=np.float32)
            h, w = ig_map.shape
            # Build a colored thumbnail from IG magnitude
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            v = np.clip(ig_map, 0, 1)
            rgb[..., 0] = (255 * v).astype(np.uint8)
            rgb[..., 1] = (255 * (1 - v) * 0.2).astype(np.uint8)
            rgb[..., 2] = (255 * (1 - v) * 0.2).astype(np.uint8)
            ig_img = Image.fromarray(rgb, mode="RGB")
            ig_img.thumbnail((60 * mm, 60 * mm))
            ib = io.BytesIO()
            ig_img.save(ib, format="PNG")
            c.drawImage(ImageReader(io.BytesIO(ib.getvalue())), left, y - 62 * mm, width=60 * mm, height=60 * mm)
            c.drawString(left + 62 * mm, y - 5 * mm, "IG magnitude thumbnail")
            y -= 66 * mm
        except Exception:
            pass

    # Probability chart (bars) — always include
    y -= 2 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Class probabilities")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    if probabilities:
        bar_left = left
        bar_w = 85 * mm
        bar_h = 6
        for p in probabilities:
            lbl = p.get("label", str(p.get("index", "")))
            val = float(p.get("probability", 0.0))
            c.drawString(bar_left, y, lbl)
            y -= 4 * mm
            c.setFillColorRGB(0.87, 0.90, 0.94)
            c.roundRect(bar_left, y, bar_w, bar_h, 1, stroke=0, fill=1)
            c.setFillColorRGB(0.06, 0.46, 0.44)
            c.roundRect(bar_left, y, min(bar_w, bar_w * val), bar_h, 1, stroke=0, fill=1)
            y -= 7 * mm
        c.setFillColorRGB(0, 0, 0)
    else:
        c.drawString(left, y, "Probabilities unavailable.")
        y -= 5 * mm

    # ROI contributions
    if explanations and explanations.get("roiContributions") and isinstance(explanations["roiContributions"], list):
        c.drawString(left, y, "Region contributions (Integrated Gradients):")
        y -= 5 * mm
        # Text list
        for r in explanations["roiContributions"][:6]:
            pct = r["positivePercent"] if r["direction"] == "increased" else r["negativePercent"]
            c.drawString(left + 4 * mm, y, f"{r['name']}: {pct:.1f}% ({r['direction']} probability)")
            y -= 4.5 * mm
        # Simple horizontal bar chart visualization
        y -= 2 * mm
        c.setFont("Helvetica", 9)
        bar_left = left + 4 * mm
        bar_width = 80 * mm
        bar_height = 4
        for r in explanations["roiContributions"][:5]:
            pct = r["positivePercent"] if r["direction"] == "increased" else r["negativePercent"]
            pct_clamped = max(0.0, min(100.0, pct))
            filled = (pct_clamped / 100.0) * bar_width
            # label
            c.drawString(bar_left, y, r["name"])
            y -= 3.5 * mm
            # background bar
            c.setFillColorRGB(0.87, 0.90, 0.94)  # slate-200
            c.roundRect(bar_left, y, bar_width, bar_height, 1, stroke=0, fill=1)
            # filled bar
            if r["direction"] == "increased":
                c.setFillColorRGB(0.06, 0.46, 0.44)  # teal tone
            else:
                c.setFillColorRGB(0.80, 0.33, 0.33)  # soft red for decreased
            c.roundRect(bar_left, y, filled, bar_height, 1, stroke=0, fill=1)
            y -= 6 * mm
        c.setFillColorRGB(0, 0, 0)
    else:
        c.drawString(left, y, "Region-wise contributions not available.")
        y -= 5 * mm

    # Summary bullets synthesized from explainability
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Summary")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    bullets: List[str] = []
    bullets.append(f"Model indicates {pred_label} (confidence {confidence*100:.1f}%).")
    if explanations:
        occl = explanations.get("occlusion")
        if isinstance(occl, dict) and occl.get("regions"):
            top = occl["regions"][0]
            bullets.append(f"Top occlusion patch at ({top['bbox'][0]},{top['bbox'][1]}) reduces confidence by {(top['delta']*100):.1f}%.")
        rc = explanations.get("roiContributions")
        if isinstance(rc, list) and rc:
            r0 = rc[0]
            pct = r0["positivePercent"] if r0["direction"] == "increased" else r0["negativePercent"]
            bullets.append(f"{r0['name']} contributed {pct:.1f}% ({r0['direction']} probability).")
        igs = explanations.get("integratedGradients")
        if isinstance(igs, dict) and igs.get("summary"):
            s = igs["summary"]
            bullets.append(f"IG magnitude: mean {s['mean']:.3f}, std {s['std']:.3f}.")
    for b in bullets:
        c.drawString(left + 4 * mm, y, f"• {b}")
        y -= 5 * mm

    # Footer
    # Quick facts and download section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Quick Facts")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    c.drawString(left, y, "Model version: v1.2")
    y -= 5 * mm
    c.drawString(left, y, "Threshold suggestion: if confidence < 60%, request second read")
    y -= 10 * mm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(left, 12 * mm, "Explainability outputs are model-based and approximate; interpret with clinical context.")

    c.showPage()
    c.save()
    return buf.getvalue()
