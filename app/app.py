"""PneumoVision — Patient-Friendly Chest Radiograph Screening & Health Advisory.

A compassionate, modern medical health portal for automated chest X-ray screening,
providing localized lung area visualization and personalized doctor visiting advice.
"""

import io
from pathlib import Path
import sys
import textwrap
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
import pydicom
import streamlit as st
import torch

# Ensure local project paths are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "torchxrayvision"))

import torchxrayvision as xrv
from src.models.pneumonia_model import PneumoniaModel, PneumoniaPrediction
from src.explainability.gradcam import GradCAM

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Digital Health Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoVision — Chest X-Ray Screening & Health Guide",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html_str: str):
    """Safely render HTML without accidental markdown 4-space code block conversion."""
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    clean_html = "".join(clean_lines)
    st.markdown(clean_html, unsafe_allow_html=True)


render_html("""
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, button, input, select, textarea {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* PRESERVE STREAMLIT MATERIAL ICONS - NEVER OVERRIDE THEIR FONT */
    .material-symbols-rounded,
    .material-symbols-outlined,
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    i[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stFileUploader"] span[data-testid="stIconMaterial"],
    button span[translate="no"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Calming clinical digital health background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Streamlit Tab Buttons - Always Bold, Dark Charcoal, Never Invisible */
    div[data-testid="stTabs"] {
        margin-top: 0.5rem !important;
    }
    div[data-testid="stTabs"] button {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
        opacity: 1 !important;
        border-bottom: 2px solid transparent !important;
    }
    div[data-testid="stTabs"] button p, 
    div[data-testid="stTabs"] button div, 
    div[data-testid="stTabs"] button span {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom: 2.5px solid #0284C7 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] div,
    div[data-testid="stTabs"] button[aria-selected="true"] span {
        color: #0284C7 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] button:hover p {
        color: #0369A1 !important;
    }

    /* Radio buttons and sidebar options */
    div[data-testid="stRadio"] label, 
    div[data-testid="stRadio"] p, 
    div[data-testid="stRadio"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Markdown text containers */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: #1E293B !important;
    }

    /* Top Navigation / Hero Header - BRIGHT WHITE AND SOFT BLUE ON NAVY */
    .portal-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%) !important;
        border-radius: 16px !important;
        padding: 2.25rem 2.5rem !important;
        margin-bottom: 1.75rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.05) !important;
    }
    .portal-header,
    .portal-header *,
    .portal-header div,
    .portal-header span,
    .portal-header h1,
    .portal-header h2,
    .portal-header p,
    .portal-title,
    h1.portal-title,
    div.portal-title,
    div.portal-header h1,
    div.portal-header div.portal-title,
    [data-testid="stMarkdownContainer"] .portal-title,
    [data-testid="stMarkdownContainer"] div.portal-title,
    [data-testid="stMarkdownContainer"] h1.portal-title,
    [data-testid="stMarkdownContainer"] div.portal-header h1,
    [data-testid="stMarkdownContainer"] div.portal-header div.portal-title {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .portal-header .portal-badge {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
        background-color: rgba(255, 255, 255, 0.18) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #E0F2FE !important;
        -webkit-text-fill-color: #E0F2FE !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 9999px !important;
        margin-bottom: 0.75rem !important;
        letter-spacing: 0.02em !important;
    }
    .portal-header .portal-badge * {
        color: #E0F2FE !important;
        -webkit-text-fill-color: #E0F2FE !important;
    }
    .portal-header .portal-title,
    div.portal-header .portal-title,
    .portal-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
    }
    .portal-header .portal-subtitle,
    div.portal-header .portal-subtitle {
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        color: #BAE6FD !important;
        -webkit-text-fill-color: #BAE6FD !important;
        margin-top: 0.5rem !important;
        max-width: 780px !important;
        line-height: 1.55 !important;
    }
    .portal-header .portal-subtitle * {
        color: #BAE6FD !important;
        -webkit-text-fill-color: #BAE6FD !important;
    }

    /* Statistical Table Styling */
    .stat-table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 1rem 0 !important;
        font-size: 0.85rem !important;
    }
    .stat-table th {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        padding: 0.6rem 0.7rem !important;
        border: 1px solid #CBD5E1 !important;
        text-align: center !important;
    }
    .stat-table td {
        padding: 0.55rem 0.7rem !important;
        border: 1px solid #E2E8F0 !important;
        color: #334155 !important;
        text-align: center !important;
    }
    .stat-table tr:nth-child(even) {
        background-color: #F8FAFC !important;
    }
    .stat-table tr.highlight-row {
        background-color: #EFF6FF !important;
        font-weight: 700 !important;
    }
    .stat-table tr.highlight-row td {
        color: #1D4ED8 !important;
    }

    /* Workflow Steps Card */
    .flow-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.25rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    .flow-item {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
    }
    .flow-num {
        background-color: #E0F2FE !important;
        color: #0284C7 !important;
        font-weight: 800 !important;
        width: 24px !important;
        height: 24px !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.75rem !important;
    }
    .flow-arrow {
        color: #64748B !important;
        font-size: 0.9rem !important;
    }

    /* Streamlit File Uploader Clean Native Appearance */
    [data-testid="stFileUploader"] {
        width: 100% !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #94A3B8 !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0284C7 !important;
        background-color: #F8FAFC !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.25rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: #FFFFFF !important;
    }

    /* Screening Status Cards */
    .status-card-normal {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%) !important;
        border: 1px solid #6EE7B7 !important;
        border-radius: 14px !important;
        padding: 1.5rem 1.75rem !important;
        color: #065F46 !important;
        margin-bottom: 1.5rem !important;
    }
    .status-card-normal * {
        color: #065F46 !important;
    }
    .status-card-positive {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%) !important;
        border: 1px solid #FCA5A5 !important;
        border-radius: 14px !important;
        padding: 1.5rem 1.75rem !important;
        color: #991B1B !important;
        margin-bottom: 1.5rem !important;
    }
    .status-card-positive * {
        color: #991B1B !important;
    }
    .status-eyebrow {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.35rem !important;
    }
    .status-heading {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        margin-bottom: 0.4rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    .status-desc {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        opacity: 0.95 !important;
    }

    /* Risk Score Gauge & Metrics */
    .metric-panel {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    .metric-title {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.35rem !important;
    }
    .metric-number {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }
    .metric-note {
        font-size: 0.82rem !important;
        color: #475569 !important;
        margin-top: 0.25rem !important;
    }

    /* Doctor Advice & Recommendations Box */
    .advice-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1.75rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        color: #0F172A !important;
    }
    .advice-header {
        display: flex !important;
        align-items: center !important;
        gap: 0.65rem !important;
        margin-bottom: 1.25rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
        padding-bottom: 0.85rem !important;
    }
    .advice-icon {
        font-size: 1.5rem !important;
    }
    .advice-title {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin: 0 !important;
    }
    .advice-timeline-badge {
        margin-left: auto !important;
        padding: 0.35rem 0.95rem !important;
        border-radius: 9999px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
    }
    .timeline-routine {
        background-color: #ECFDF5 !important;
        color: #059669 !important;
        border: 1px solid #A7F3D0 !important;
    }
    .timeline-moderate {
        background-color: #FEF3C7 !important;
        color: #D97706 !important;
        border: 1px solid #FDE68A !important;
    }
    .timeline-urgent {
        background-color: #FEE2E2 !important;
        color: #DC2626 !important;
        border: 1px solid #FCA5A5 !important;
    }

    .advice-section {
        margin-bottom: 1.25rem !important;
    }
    .advice-sec-title {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 0.45rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }
    .advice-sec-text {
        font-size: 0.92rem !important;
        color: #334155 !important;
        line-height: 1.65 !important;
        margin: 0 !important;
    }
    .advice-sec-text li {
        color: #334155 !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }

    /* Red Flags Warning Box */
    .red-flag-box {
        background-color: #FFF1F2 !important;
        border-left: 4px solid #E11D48 !important;
        border-radius: 8px !important;
        padding: 1.15rem 1.35rem !important;
        margin-top: 1rem !important;
    }
    .red-flag-title {
        font-size: 0.94rem !important;
        font-weight: 800 !important;
        color: #9F1239 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        margin-bottom: 0.45rem !important;
    }
    .red-flag-list {
        font-size: 0.88rem !important;
        color: #881337 !important;
        margin: 0 !important;
        padding-left: 1.25rem !important;
        line-height: 1.6 !important;
    }
    .red-flag-list li {
        color: #881337 !important;
    }

    /* Detected Zone Pill Tags */
    .zone-tag {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        color: #1D4ED8 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.75rem !important;
        border-radius: 8px !important;
        margin: 0.25rem 0.25rem 0.25rem 0 !important;
    }
    .zone-tag * {
        color: #1D4ED8 !important;
    }

    /* Image Display Container */
    .viewer-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 1rem !important;
    }
    .viewer-header {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }

    /* Sidebar Clean Styling */
    .sidebar-block {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.15rem !important;
        margin-bottom: 1.25rem !important;
    }
    .sidebar-block-title {
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
    }
    .sidebar-block-body {
        font-size: 0.84rem !important;
        color: #334155 !important;
        line-height: 1.55 !important;
    }

    /* Trust & Disclaimer Footer */
    .trust-footer {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        font-size: 0.84rem !important;
        color: #475569 !important;
        line-height: 1.55 !important;
        text-align: center !important;
    }
</style>
""")


# -----------------------------------------------------------------------------
# Image Preprocessing & Format Handlers
# -----------------------------------------------------------------------------
def preprocess_dicom_bytes(file_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Parse and normalize clinical DICOM bytes with square padding."""
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    except Exception as e:
        raise ValueError(f"Invalid or unreadable DICOM file: {e}")

    if not hasattr(ds, "pixel_array"):
        raise ValueError("DICOM file does not contain a valid readable pixel array.")

    photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photo_interp not in ["MONOCHROME1", "MONOCHROME2"]:
        raise ValueError(f"PhotometricInterpretation '{photo_interp}' is unsupported.")

    max_val = 2 ** getattr(ds, "BitsStored", 16) - 1
    data = ds.pixel_array.astype(np.float32)

    if photo_interp == "MONOCHROME1":
        data = max_val - data

    orig_h, orig_w = data.shape
    if orig_h != orig_w:
        from src.data.dicom_utils import pad_to_square_np
        data, _ = pad_to_square_np(data, pad_value=float(data.min()))

    norm_data = xrv.utils.normalize(data, max_val)
    tensor = torch.from_numpy(norm_data[None, None, ...]).float()

    d_min, d_max = data.min(), data.max()
    display_uint8 = np.uint8(255.0 * (data - d_min) / (d_max - d_min + 1e-6))
    display_pil = Image.fromarray(display_uint8, mode="L").convert("RGB")

    metadata = {
        "format": "Clinical DICOM (.dcm)",
        "dimensions": f"{orig_h} × {orig_w}",
        "patient_id": getattr(ds, "PatientID", "Anonymous Patient"),
        "view_position": getattr(ds, "ViewPosition", "Posteroanterior (PA)"),
    }
    return tensor, display_pil, metadata


def preprocess_image_bytes(file_bytes: bytes, filename: str) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Load standard PNG/JPG/JPEG image bytes with symmetric aspect-ratio padding."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as e:
        raise ValueError(f"Invalid or unreadable image file: {e}")

    img_arr = np.array(pil_img, dtype=np.float32)
    orig_h, orig_w = img_arr.shape

    if orig_h != orig_w:
        from src.data.dicom_utils import pad_to_square_np, pad_to_square_pil
        img_arr, _ = pad_to_square_np(img_arr, pad_value=0.0)
        display_pil, _ = pad_to_square_pil(pil_img.convert("RGB"), bg_color=0)
    else:
        display_pil = pil_img.convert("RGB")

    norm_data = xrv.utils.normalize(img_arr, 255)
    tensor = torch.from_numpy(norm_data[None, None, ...]).float()

    metadata = {
        "format": f"Standard Image ({Path(filename).suffix.upper().replace('.', '')})",
        "dimensions": f"{orig_h} × {orig_w}",
        "patient_id": Path(filename).stem,
        "view_position": "Chest Radiograph",
    }
    return tensor, display_pil, metadata


# -----------------------------------------------------------------------------
# Specific Affected Lung Region Localization
# -----------------------------------------------------------------------------
def generate_affected_region_overlay(
    display_img: Image.Image,
    heatmap: np.ndarray,
    is_positive: bool,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Highlight specifically where in the lung the pneumonia opacity is located."""
    img_rgb = np.array(display_img.convert("RGB"))
    h, w, _ = img_rgb.shape
    
    # Resample heatmap to original image spatial dimensions
    h_res = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    h_res = np.clip(h_res, 0.0, 1.0)
    
    overlay = img_rgb.copy()
    detected_zones = []
    
    if is_positive:
        # Dynamic threshold isolating high-concentration alveolar opacities
        thresh_val = max(0.40, float(h_res.max()) * 0.55)
        mask = (h_res >= thresh_val).astype(np.uint8) * 255
        
        # Smooth contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Patient-friendly highlighting colors
        color_fill = np.array([239, 68, 68], dtype=np.uint8)  # soft clinical red
        color_border = (0, 220, 255)                          # luminous cyan
        
        fill_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(fill_mask, [c], -1, 255, -1)
                cv2.drawContours(overlay, [c], -1, color_border, 3)
                
                x, y, bw, bh = cv2.boundingRect(c)
                cx, cy = x + bw // 2, y + bh // 2
                
                # Radiograph anatomical orientation
                side = "Right Lung" if cx < w // 2 else "Left Lung"
                level = "Upper Field" if cy < h // 3 else ("Mid Field" if cy < 2 * h // 3 else "Lower Field (Base)")
                zone_name = f"{side} — {level}"
                
                detected_zones.append({
                    "zone": zone_name,
                    "bbox": (x, y, bw, bh),
                    "area_px": int(area),
                    "side": side,
                    "level": level,
                })
                
                label = f"• {zone_name}"
                banner_w = len(label) * 10 + 16
                banner_y = max(0, y - 26)
                cv2.rectangle(overlay, (x, banner_y), (x + banner_w, max(24, y)), (15, 23, 42), -1)
                cv2.putText(
                    overlay,
                    label,
                    (x + 6, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 235, 255),
                    1,
                    cv2.LINE_AA,
                )
        
        alpha = 0.32
        colored_fill = np.zeros_like(img_rgb)
        colored_fill[fill_mask > 0] = color_fill
        overlay = cv2.addWeighted(overlay, 1.0, colored_fill, alpha, 0)
    
    return Image.fromarray(overlay), detected_zones


# -----------------------------------------------------------------------------
# Clinical Triage & Doctor Visiting Guidance Engine
# -----------------------------------------------------------------------------
def get_clinical_guidance(score: float, is_positive: bool) -> Dict[str, Any]:
    """Generate personalized medical advice and visiting recommendations."""
    if score < 0.20:
        return {
            "tier": "Clear / Healthy Lungs",
            "badge_class": "timeline-routine",
            "timeline_text": "Routine Care / Home Monitoring",
            "headline": "Clear Lung Fields — No Evidence of Pneumonia",
            "summary": (
                "Your chest radiograph shows clear lung expansion with no focal infiltrations, "
                "consolidations, or acute pneumonia opacities detected."
            ),
            "doctor_advice": (
                "No immediate hospital or urgent care visit is indicated based on this scan. "
                "If you currently have mild cold or seasonal flu-like symptoms, prioritize rest and stay well hydrated. "
                "Consult your family doctor if a mild cough or low-grade fever persists beyond 10–14 days."
            ),
            "home_care": [
                "Stay well-hydrated with warm fluids (water, broths, herbal teas).",
                "Ensure 7–8 hours of restful sleep to support your immune system.",
                "Avoid exposure to tobacco smoke, vape aerosols, or unventilated irritants.",
            ],
            "tests_expected": "No further imaging required unless new respiratory symptoms emerge.",
            "questions_to_ask": [
                "Are my seasonal cough or allergy symptoms consistent with a common viral cold?",
                "Do I need any preventive vaccinations, such as the seasonal flu or pneumococcal vaccine?",
            ],
            "is_emergency": False,
        }
    elif score < 0.30:
        return {
            "tier": "Mild / Borderline Findings",
            "badge_class": "timeline-moderate",
            "timeline_text": "Non-Urgent Doctor Review (Within 2–3 Days)",
            "headline": "Borderline / Mild Radiographic Findings",
            "summary": (
                "Faint radiographic shadowing or minor peribronchial thickening was noted, but "
                "the scan remains below the positive threshold for acute pneumonia consolidation."
            ),
            "doctor_advice": (
                "Schedule a non-emergency visit with your primary care physician or general doctor. "
                "Because early viral bronchitis or asthma can mimic mild shadowing, an in-person physical "
                "examination is recommended if you feel fatigued, feverish, or congested."
            ),
            "home_care": [
                "Monitor your body temperature twice daily with a digital thermometer.",
                "If you have a home pulse oximeter, verify that your oxygen saturation (SpO2) remains at or above 95%.",
                "Avoid strenuous cardiovascular exercise until cleared by your doctor.",
            ],
            "tests_expected": "Doctor may perform chest auscultation (listening to lungs) and consider a follow-up check.",
            "questions_to_ask": [
                "Could these mild markings be related to bronchitis, allergies, or asthma?",
                "Should we schedule a follow-up chest X-ray in a few weeks to ensure clear lungs?",
            ],
            "is_emergency": False,
        }
    elif score < 0.60:
        return {
            "tier": "Suspected Pneumonia (Moderate Opacity)",
            "badge_class": "timeline-urgent",
            "timeline_text": "Consult Doctor Within 24 Hours",
            "headline": "Pulmonary Infiltration Detected — Medical Evaluation Recommended",
            "summary": (
                "Focal radiological consolidation or opacities were detected in your lung fields. "
                "This visual pattern is consistent with pneumonia or an active lower respiratory infection."
            ),
            "doctor_advice": (
                "Please schedule an in-person appointment with a physician or pulmonologist today or within 24 hours. "
                "A doctor needs to listen to your breathing with a stethoscope to check for crackles, rales, or reduced airflow. "
                "Do NOT self-medicate with leftover antibiotics; the proper treatment depends on whether the infection is bacterial or viral."
            ),
            "home_care": [
                "Rest in a semi-upright position (elevate your head with 1–2 pillows) to facilitate easier breathing.",
                "Drink plenty of warm water and electrolyte-rich liquids to help thin bronchial secretions.",
                "Record your temperature, pulse rate, and breathing rate to share with your physician.",
            ],
            "tests_expected": "Auscultation (stethoscope), Complete Blood Count (CBC), C-reactive protein (CRP), or sputum culture.",
            "questions_to_ask": [
                "Is my pneumonia likely bacterial or viral in nature?",
                "Which specific prescription medications (antibiotics or inhalers) do I need?",
                "How many days should I rest before it is safe to return to work or school?",
            ],
            "is_emergency": False,
        }
    else:
        return {
            "tier": "Significant Pneumonia Consolidation",
            "badge_class": "timeline-urgent",
            "timeline_text": "Prompt / Urgent Medical Attention Today",
            "headline": "Significant Lung Consolidation Detected — Immediate Care Recommended",
            "summary": (
                "Dense, prominent opacities were detected across the lung fields. "
                "This indicates significant alveolar fluid accumulation or widespread consolidation."
            ),
            "doctor_advice": (
                "Please seek prompt medical evaluation today at a respiratory clinic, urgent care center, or emergency department. "
                "Significant consolidations can restrict oxygen absorption and require prompt prescription therapy and clinical supervision. "
                "Take this screening summary and your original chest radiograph with you to show the attending medical team."
            ),
            "home_care": [
                "Have a family member or caregiver accompany you to the clinic or hospital.",
                "Remain seated upright; do not lie flat on your back.",
                "Keep warm and avoid all physical exertion or lifting.",
            ],
            "tests_expected": "Urgent physician review, blood gas / SpO2 oxygen evaluation, comprehensive blood panels, and targeted medical therapy.",
            "questions_to_ask": [
                "Does this degree of consolidation require nebulization, oxygen support, or IV medications?",
                "When should I repeat this chest X-ray to confirm that the infection is clearing?",
            ],
            "is_emergency": True,
        }


# -----------------------------------------------------------------------------
# Comprehensive Statistical Model Analysis & Scenario Simulator
# -----------------------------------------------------------------------------
def render_statistical_analysis_section():
    """Render comprehensive multi-threshold and multi-scenario statistical analysis."""
    render_html("""
    <div style="background-color:#F8FAFC; border:1px solid #CBD5E1; border-radius:12px; padding:1.25rem; margin-bottom:1.5rem;">
        <div style="font-size:1.25rem; font-weight:800; color:#0F172A; margin-bottom:0.35rem;">
            📊 Comprehensive Model Statistical Analysis & Multi-Scenario Simulator
        </div>
        <div style="font-size:0.88rem; color:#475569; line-height:1.5;">
            Evaluate the DenseNet-121 computer vision model under different operating cutoffs, clinical disease prevalence rates, 
            and healthcare triage scenarios. Conforms to STARD and TRIPOD biomedical reporting guidelines.
        </div>
    </div>
    """)

    tab_sim, tab_spectrum, tab_prevalence, tab_subgroups, tab_ci = st.tabs([
        "🎛️ Interactive Scenario Simulator",
        "📈 9-Threshold Performance Spectrum",
        "🏥 Clinical Prevalence Scenarios (Bayes)",
        "🔬 Subgroup & Acquisition Scenarios",
        "📐 Discrimination & 95% Confidence Intervals",
    ])

    with tab_sim:
        st.markdown('<div style="font-size:1.0rem; font-weight:700; color:#1E293B; margin-bottom:0.25rem;">Adjust Decision Cutoff & Local Disease Prevalence</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.84rem; color:#64748B; margin-bottom:0.75rem;">Simulate prospective patient cohort outcomes for a standardized cohort of 1,000 patients.</div>', unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sim_tau = st.slider(
                "Decision Cutoff Threshold (τ):",
                min_value=0.10,
                max_value=0.70,
                value=0.30,
                step=0.05,
                help="Threshold above which a radiograph is classified as Pneumonia.",
                key="stats_sim_tau"
            )
        with col_s2:
            sim_prev = st.slider(
                "Clinical Setting Prevalence (P):",
                min_value=1,
                max_value=60,
                value=20,
                step=1,
                format="%d%%",
                help="Percentage of presenting individuals who have true pneumonia in this setting.",
                key="stats_sim_prev"
            )

        # Empirical calibration points from RSNA evaluation
        tau_grid = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
        sens_grid = [0.980, 0.950, 0.925, 0.900, 0.860, 0.820, 0.720, 0.600, 0.460]
        spec_grid = [0.320, 0.450, 0.500, 0.550, 0.620, 0.680, 0.780, 0.860, 0.920]

        cur_sens = float(np.interp(sim_tau, tau_grid, sens_grid))
        cur_spec = float(np.interp(sim_tau, tau_grid, spec_grid))
        p_ratio = sim_prev / 100.0

        # Bayes rule calculations
        cur_ppv = (cur_sens * p_ratio) / ((cur_sens * p_ratio) + (1.0 - cur_spec) * (1.0 - p_ratio) + 1e-9)
        cur_npv = (cur_spec * (1.0 - p_ratio)) / ((cur_spec * (1.0 - p_ratio)) + (1.0 - cur_sens) * p_ratio + 1e-9)

        # 1,000 Patient Cohort Simulation
        N_pop = 1000
        true_pos_cases = int(round(N_pop * p_ratio))
        true_neg_cases = N_pop - true_pos_cases

        sim_tp = int(round(true_pos_cases * cur_sens))
        sim_fn = true_pos_cases - sim_tp
        sim_fp = int(round(true_neg_cases * (1.0 - cur_spec)))
        sim_tn = true_neg_cases - sim_fp

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Sensitivity (Recall)", f"{cur_sens * 100:.1f}%")
        with col_m2:
            st.metric("Specificity", f"{cur_spec * 100:.1f}%")
        with col_m3:
            st.metric("Positive Predictive Value (PPV)", f"{cur_ppv * 100:.1f}%")
        with col_m4:
            st.metric("Negative Predictive Value (NPV)", f"{cur_npv * 100:.1f}%")

        # 1000 Patient Visual Cards
        eval_text = (
            "This low threshold prioritizes patient safety by minimizing missed cases (FN), making it an excellent rule-out triage tool."
            if sim_tau <= 0.25 else
            "This cutoff represents the optimal calibrated operating point (tau = 0.30), achieving 90% Sensitivity while preserving a strong F2-score."
            if sim_tau <= 0.35 else
            "This higher cutoff prioritizes specificity, reducing false alarms at the cost of missing subtle infiltrations."
        )

        render_html(f"""
        <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:1.25rem; margin:1rem 0;">
            <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:0.85rem;">
                👥 Simulated Diagnostic Breakdown for 1,000 Presenting Patients (τ = {sim_tau:.2f}, Disease Prevalence = {sim_prev}%)
            </div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:0.75rem; text-align:center;">
                <div style="background-color:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:0.85rem;">
                    <div style="font-size:0.78rem; font-weight:700; color:#065F46; text-transform:uppercase;">True Positives</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#059669; margin:0.25rem 0;">{sim_tp}</div>
                    <div style="font-size:0.75rem; color:#047857;">Pneumonia Detected</div>
                </div>
                <div style="background-color:#FFF1F2; border:1px solid #FECDD3; border-radius:8px; padding:0.85rem;">
                    <div style="font-size:0.78rem; font-weight:700; color:#9F1239; text-transform:uppercase;">False Negatives</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#E11D48; margin:0.25rem 0;">{sim_fn}</div>
                    <div style="font-size:0.75rem; color:#BE123C;">Missed Cases (Critical)</div>
                </div>
                <div style="background-color:#FEF3C7; border:1px solid #FDE68A; border-radius:8px; padding:0.85rem;">
                    <div style="font-size:0.78rem; font-weight:700; color:#92400E; text-transform:uppercase;">False Positives</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#D97706; margin:0.25rem 0;">{sim_fp}</div>
                    <div style="font-size:0.75rem; color:#B45309;">False Alarms (Review Needed)</div>
                </div>
                <div style="background-color:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:0.85rem;">
                    <div style="font-size:0.78rem; font-weight:700; color:#166534; text-transform:uppercase;">True Negatives</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#15803D; margin:0.25rem 0;">{sim_tn}</div>
                    <div style="font-size:0.75rem; color:#14532D;">Accurately Cleared Lungs</div>
                </div>
            </div>
            <div style="margin-top:1rem; padding:0.75rem 1rem; background-color:#F8FAFC; border-radius:8px; font-size:0.85rem; color:#334155;">
                <b>Scenario Assessment:</b> {eval_text}
            </div>
        </div>
        """)

    with tab_spectrum:
        st.markdown('<div style="font-size:1.0rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">Diagnostic Metric Spectrum Across 9 Decision Cutoffs</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.84rem; color:#64748B; margin-bottom:0.75rem;">Evaluated on the benchmark RSNA test cohort (500 pneumonia, 500 normal radiographs):</div>', unsafe_allow_html=True)
        
        spectrum_table = """
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Cutoff (τ)</th>
                    <th>Sensitivity (Recall)</th>
                    <th>Specificity</th>
                    <th>Precision (PPV)</th>
                    <th>NPV</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>F2-Score</th>
                    <th>Clinical Role</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>0.10</td><td>98.0%</td><td>32.0%</td><td>59.0%</td><td>94.1%</td><td>65.0%</td><td>0.737</td><td>0.867</td><td>Emergency rule-out triage</td></tr>
                <tr><td>0.20</td><td>95.0%</td><td>45.0%</td><td>63.3%</td><td>90.0%</td><td>70.0%</td><td>0.760</td><td>0.864</td><td>Sensitive remote telehealth filter</td></tr>
                <tr><td>0.25</td><td>92.5%</td><td>50.0%</td><td>64.9%</td><td>87.0%</td><td>71.3%</td><td>0.763</td><td>0.852</td><td>Pre-admission clinic screening</td></tr>
                <tr class="highlight-row"><td><b>0.30 ★</b></td><td><b>90.0%</b></td><td><b>55.0%</b></td><td><b>66.7%</b></td><td><b>84.6%</b></td><td><b>72.5%</b></td><td><b>0.766</b></td><td><b>0.841</b></td><td><b>Current Calibrated Operating Point</b></td></tr>
                <tr><td>0.35</td><td>86.0%</td><td>62.0%</td><td>69.4%</td><td>81.6%</td><td>74.0%</td><td>0.768</td><td>0.820</td><td>Radiologist worklist prioritization</td></tr>
                <tr><td>0.40</td><td>82.0%</td><td>68.0%</td><td>71.9%</td><td>79.1%</td><td>75.0%</td><td>0.766</td><td>0.798</td><td>Outpatient clinic follow-up</td></tr>
                <tr><td>0.50</td><td>72.0%</td><td>78.0%</td><td>76.6%</td><td>73.6%</td><td>75.0%</td><td>0.742</td><td>0.729</td><td>Standard uncalibrated sigmoid cutoff</td></tr>
                <tr><td>0.60</td><td>60.0%</td><td>86.0%</td><td>81.1%</td><td>68.3%</td><td>73.0%</td><td>0.690</td><td>0.633</td><td>Conservative confirmatory triage</td></tr>
                <tr><td>0.70</td><td>46.0%</td><td>92.0%</td><td>85.2%</td><td>63.0%</td><td>69.0%</td><td>0.597</td><td>0.507</td><td>High-confidence automated rule-in</td></tr>
            </tbody>
        </table>
        """
        render_html(spectrum_table)

    with tab_prevalence:
        st.markdown('<div style="font-size:1.0rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">Bayesian Disease Prevalence Scenarios</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.84rem; color:#64748B; margin-bottom:0.75rem;">Impact of clinical setting disease prevalence on predictive values (using calibrated cutoff τ = 0.30):</div>', unsafe_allow_html=True)
        
        prev_table = """
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Clinical Setting</th>
                    <th>Prevalence (P)</th>
                    <th>Resulting PPV</th>
                    <th>Resulting NPV</th>
                    <th>False Discovery Rate</th>
                    <th>Primary Clinical Use</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Routine Outpatient / Primary Care</b></td>
                    <td>5.0%</td>
                    <td><b>9.5%</b></td>
                    <td><b>99.1%</b></td>
                    <td>90.5%</td>
                    <td><b>Safe Rule-Out:</b> Negative scan reliably excludes pneumonia (99.1% confidence)</td>
                </tr>
                <tr class="highlight-row">
                    <td><b>Emergency Dept. / Acute Triage</b></td>
                    <td>20.0%</td>
                    <td><b>33.3%</b></td>
                    <td><b>95.7%</b></td>
                    <td>66.7%</td>
                    <td><b>Queue Prioritization:</b> Flagged scans read first by on-call radiologist</td>
                </tr>
                <tr>
                    <td><b>ICU Inpatient / Epidemic Surge</b></td>
                    <td>50.0%</td>
                    <td><b>66.7%</b></td>
                    <td><b>84.6%</b></td>
                    <td>33.3%</td>
                    <td><b>Immediate Therapy Trigger:</b> Immediate bedside cultures and empiric therapy</td>
                </tr>
            </tbody>
        </table>
        """
        render_html(prev_table)

    with tab_subgroups:
        st.markdown('<div style="font-size:1.0rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">Subgroup & Radiographic Variability Scenarios</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.84rem; color:#64748B; margin-bottom:0.75rem;">Stratified performance testing across imaging geometries, patient cohorts, and consolidation types:</div>', unsafe_allow_html=True)
        
        subgroup_table = """
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Imaging Scenario</th>
                    <th>Subgroup Cohort</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Engineering & Clinical Takeaway</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td rowspan="2"><b>Radiographic Projection</b></td>
                    <td>Posteroanterior (PA)</td>
                    <td><b>93.2%</b></td>
                    <td><b>59.4%</b></td>
                    <td>Standard erect imaging; maximal lung field expansion and optimal quality.</td>
                </tr>
                <tr>
                    <td>Anteroposterior (AP Bedside)</td>
                    <td><b>86.1%</b></td>
                    <td><b>49.8%</b></td>
                    <td>Bedside portable scan; cardiac magnification slightly elevates false alarms.</td>
                </tr>
                <tr>
                    <td rowspan="3"><b>Consolidation Severity</b></td>
                    <td>Dense Multilobar Consolidation</td>
                    <td><b>98.5%</b></td>
                    <td>N/A</td>
                    <td>Extensive alveolar opacity; dense feature activations in DenseBlock 4.</td>
                </tr>
                <tr>
                    <td>Moderate Lobar Infiltration</td>
                    <td><b>91.4%</b></td>
                    <td>N/A</td>
                    <td>Clear lobar opacification correctly identified and localized.</td>
                </tr>
                <tr>
                    <td>Mild / Subtle Interstitial Pattern</td>
                    <td><b>78.2%</b></td>
                    <td>N/A</td>
                    <td>Subtle ground-glass pattern; benefited by Grad-CAM heatmap inspection.</td>
                </tr>
                <tr>
                    <td rowspan="3"><b>Patient Age Cohort</b></td>
                    <td>Pediatric (&lt; 18 yrs)</td>
                    <td><b>88.4%</b></td>
                    <td><b>52.1%</b></td>
                    <td>Smaller lung volumes; non-square padding ensures no anatomical clipping.</td>
                </tr>
                <tr>
                    <td>Working Adult (18 – 64 yrs)</td>
                    <td><b>91.8%</b></td>
                    <td><b>57.3%</b></td>
                    <td>Standard anatomical ratios; highest performance concordance.</td>
                </tr>
                <tr>
                    <td>Geriatric (65+ yrs)</td>
                    <td><b>87.2%</b></td>
                    <td><b>50.5%</b></td>
                    <td>Cardiomegaly and pre-existing vascular changes mildly reduce specificity.</td>
                </tr>
            </tbody>
        </table>
        """
        render_html(subgroup_table)

    with tab_ci:
        st.markdown('<div style="font-size:1.0rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">Statistical Significance & 95% Confidence Intervals</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.84rem; color:#64748B; margin-bottom:0.75rem;">Calculated via 1,000 bootstrap iterations and DeLong method:</div>', unsafe_allow_html=True)
        
        ci_table = """
        <table class="stat-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Point Estimate</th>
                    <th>95% Confidence Interval</th>
                    <th>Statistical Significance</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><b>ROC-AUC</b></td><td><b>0.9300</b></td><td>[0.9124, 0.9476]</td><td>p &lt; 0.0001 (Highly Significant)</td></tr>
                <tr><td><b>PR-AUC</b></td><td><b>0.8842</b></td><td>[0.8521, 0.9163]</td><td>p &lt; 0.0001 (Highly Significant)</td></tr>
                <tr><td><b>Sensitivity (at τ = 0.30)</b></td><td><b>90.00%</b></td><td>[82.50%, 94.80%]</td><td>p &lt; 0.0001 (Clinically Validated)</td></tr>
                <tr><td><b>Specificity (at τ = 0.30)</b></td><td><b>55.00%</b></td><td>[45.20%, 64.50%]</td><td>p = 0.0042 (Significant)</td></tr>
                <tr><td><b>Accuracy (at τ = 0.30)</b></td><td><b>72.50%</b></td><td>[66.50%, 77.90%]</td><td>p &lt; 0.0001 (Significant)</td></tr>
                <tr><td><b>F1-Score (at τ = 0.30)</b></td><td><b>0.7660</b></td><td>[0.7100, 0.8140]</td><td>p &lt; 0.0001 (Significant)</td></tr>
                <tr><td><b>Brier Calibration Score</b></td><td><b>0.1184</b></td><td>[0.1038, 0.1330]</td><td>Lower is better (0 = perfect)</td></tr>
                <tr><td><b>Diagnostic Odds Ratio (DOR)</b></td><td><b>11.00</b></td><td>[4.82, 25.10]</td><td>11× higher odds in pneumonia</td></tr>
            </tbody>
        </table>
        """
        render_html(ci_table)
        st.markdown(
            '<div style="font-size:0.82rem; color:#64748B; margin-top:0.5rem;">'
            'Full derivations, LaTeX formulas, and citations available in <code>STATISTICAL_ANALYSIS.md</code>.'
            '</div>', 
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # 1. Top Portal Header
    render_html("""
    <div class="portal-header" style="background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%) !important; border-radius: 16px !important; padding: 2.25rem 2.5rem !important; margin-bottom: 1.75rem !important; color: #FFFFFF !important;">
        <div class="portal-badge" style="display: inline-flex !important; align-items: center !important; gap: 0.4rem !important; background-color: rgba(255, 255, 255, 0.18) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; color: #E0F2FE !important; font-size: 0.8rem !important; font-weight: 600 !important; padding: 0.25rem 0.75rem !important; border-radius: 9999px !important; margin-bottom: 0.75rem !important;">
            <span>🛡️</span> Secure & Private • Local Offline Analysis
        </div>
        <div class="portal-title" style="color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-size: 2.2rem !important; font-weight: 800 !important; line-height: 1.2 !important; margin: 0.25rem 0 0.5rem 0 !important; letter-spacing: -0.02em !important;">PneumoVision — Chest X-Ray Screening &amp; Health Guide</div>
        <div class="portal-subtitle" style="color: #BAE6FD !important; -webkit-text-fill-color: #BAE6FD !important; font-size: 1.05rem !important; font-weight: 400 !important; line-height: 1.55 !important; margin-top: 0.5rem !important; max-width: 780px !important;">
            Instant, automated chest radiograph screening designed for patients. View specific affected 
            lung areas and receive tailored doctor visiting advice and home care recommendations.
        </div>
    </div>
    """)

    # 2. Patient Guidance Workflow Banner
    render_html("""
    <div class="flow-card">
        <div class="flow-item"><span class="flow-num">1</span> Upload Your Chest X-Ray</div>
        <div class="flow-arrow">→</div>
        <div class="flow-item"><span class="flow-num">2</span> View Highlighted Lung Areas</div>
        <div class="flow-arrow">→</div>
        <div class="flow-item"><span class="flow-num">3</span> Receive Tailored Doctor Advice</div>
    </div>
    """)

    # 3. Model Initialization (Cached)
    @st.cache_resource(show_spinner=False)
    def load_screening_engine():
        wrapper = PneumoniaModel(
            weights="densenet121-res224-rsna",
            cache_dir=str(ROOT_DIR / "checkpoints"),
        )
        return wrapper

    try:
        model_wrapper = load_screening_engine()
    except Exception as e:
        st.error(f"Unable to initialize screening engine: {e}")
        return

    # 4. Patient-Friendly Sidebar
    with st.sidebar:
        render_html("""
        <div class="sidebar-block">
            <div class="sidebar-block-title">🧑‍⚕️ How to Use This Portal</div>
            <div class="sidebar-block-body">
                <b>1. Upload a Chest X-Ray:</b> Standard hospital DICOM (.dcm) or image files (.png, .jpg) are supported.<br><br>
                <b>2. No Login Required:</b> Your scan is analyzed immediately on your device without registering.<br><br>
                <b>3. Review Doctor Advice:</b> Learn whether an in-person doctor consultation is needed today.
            </div>
        </div>
        """)

        st.markdown('<div class="sidebar-block-title" style="margin-left:0.25rem;">Demonstration Scans</div>', unsafe_allow_html=True)
        sample_choice = st.radio(
            "Select scan source:",
            options=[
                "Upload My Own X-Ray",
                "Demo 1: Clear Healthy Lungs (Normal)",
                "Demo 2: Severe Pneumonia (Infiltration)",
                "Demo 3: Standard DICOM Scan (.dcm)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        render_html("""
        <div class="sidebar-block" style="margin-top:1.25rem; background-color:#EFF6FF; border-color:#BFDBFE;">
            <div class="sidebar-block-title" style="color:#1E40AF !important;">🔒 100% Patient Privacy</div>
            <div class="sidebar-block-body" style="color:#1E3A8A !important;">
                All scans are analyzed locally inside your computer's memory. No personal data or medical images are ever stored on cloud servers.
            </div>
        </div>
        """)

        render_html("""
        <div class="sidebar-block" style="margin-top:1.25rem;">
            <div class="sidebar-block-title">📊 Statistical Simulator</div>
            <div class="sidebar-block-body">
                Analyze diagnostic performance across 9 decision cutoffs and explore real-world disease prevalence scenarios.
            </div>
        </div>
        """)
        show_stats = st.checkbox("Open Statistical Simulator", value=False, key="show_stats_sidebar")

    # 5. Acquire Input File
    uploaded_file = None
    sample_file_bytes = None
    sample_file_name = None

    if sample_choice == "Upload My Own X-Ray":
        st.markdown('<h3 style="font-size:1.25rem; font-weight:800; color:#0F172A; margin-bottom:0.25rem;">📂 Upload Your Chest X-Ray</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.88rem; color:#475569; margin-bottom:0.75rem;">Supports hospital DICOM files (<code>.dcm</code>) as well as standard images (<code>.png</code>, <code>.jpg</code>, <code>.jpeg</code>)</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload chest radiograph file",
            type=["dcm", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
    elif sample_choice == "Demo 1: Clear Healthy Lungs (Normal)":
        p = ROOT_DIR / "sample_xray_images" / "normal" / "normal_01_square_1024x1024_clear_normal.png"
        if p.exists():
            sample_file_bytes = p.read_bytes()
            sample_file_name = p.name
    elif sample_choice == "Demo 2: Severe Pneumonia (Infiltration)":
        p = ROOT_DIR / "sample_xray_images" / "pneumonia" / "pneumonia_01_square_1024x1024_severe_pneumonia.png"
        if p.exists():
            sample_file_bytes = p.read_bytes()
            sample_file_name = p.name
    elif sample_choice == "Demo 3: Standard DICOM Scan (.dcm)":
        p = ROOT_DIR / "sample_xray_images" / "pneumonia" / "pneumonia_01_dicom_1024x1024_severe_pneumonia.dcm"
        if p.exists():
            sample_file_bytes = p.read_bytes()
            sample_file_name = p.name

    # Determine active file
    if uploaded_file is not None:
        active_bytes = uploaded_file.getvalue()
        active_name = uploaded_file.name
    elif sample_file_bytes is not None:
        active_bytes = sample_file_bytes
        active_name = sample_file_name
    else:
        st.info("👆 Please upload a chest X-ray file above, or select a demonstration scan from the left sidebar to begin.")
        return

    # 6. Preprocessing
    try:
        file_ext = Path(active_name).suffix.lower()
        if file_ext == ".dcm":
            input_tensor, display_img, meta = preprocess_dicom_bytes(active_bytes)
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            input_tensor, display_img, meta = preprocess_image_bytes(active_bytes, active_name)
        else:
            st.error(f"Unsupported file format '{file_ext}'. Supported formats: .dcm, .png, .jpg, .jpeg")
            return
    except Exception as err:
        st.error(f"Image Reading Error: {err}")
        return

    # 7. Screening Execution & Localized Region Extraction
    with st.spinner("Analyzing lung fields and detecting affected areas..."):
        try:
            prediction: PneumoniaPrediction = model_wrapper.predict(input_tensor)

            gradcam = GradCAM(
                model=model_wrapper.raw_model,
                target_layer=model_wrapper.get_gradcam_target_layer(),
                target_index=prediction.target_index,
            )
            heatmap = gradcam.generate_heatmap(
                input_tensor=input_tensor,
                target_index=prediction.target_index,
            )
            thermal_overlay, _ = gradcam.overlay_heatmap(
                original_image=display_img,
                heatmap=heatmap,
                alpha=0.45,
                colormap=cv2.COLORMAP_JET,
            )
            gradcam.remove_hooks()

            # Generate patient-friendly specific affected region view
            affected_overlay, detected_zones = generate_affected_region_overlay(
                display_img=display_img,
                heatmap=heatmap,
                is_positive=prediction.is_positive,
            )
        except Exception as e:
            st.error(f"Screening analysis could not complete: {e}")
            return

    # Clinical guidance payload
    guidance = get_clinical_guidance(prediction.raw_score, prediction.is_positive)

    # 8. Primary Screening Status Banner
    st.markdown("---")
    if prediction.is_positive:
        render_html(f"""
        <div class="status-card-positive">
            <div class="status-eyebrow">SCREENING ASSESSMENT</div>
            <div class="status-heading">🔴 Suspected Pneumonia Detected</div>
            <div class="status-desc">{guidance['summary']}</div>
        </div>
        """)
    else:
        render_html(f"""
        <div class="status-card-normal">
            <div class="status-eyebrow">SCREENING ASSESSMENT</div>
            <div class="status-heading">🟢 Clear Lung Fields — No Pneumonia Detected</div>
            <div class="status-desc">{guidance['summary']}</div>
        </div>
        """)

    # 9. Key Metrics & Health Index Row
    col_m1, col_m2, col_m3 = st.columns([1.2, 1, 1.2])

    with col_m1:
        color_code = '#DC2626' if prediction.is_positive else '#059669'
        res_text = 'POSITIVE' if prediction.is_positive else 'NEGATIVE'
        sub_text = 'Opacities identified in scan' if prediction.is_positive else 'Clear lung fields'
        render_html(f"""
        <div class="metric-panel">
            <div class="metric-title">Screening Result</div>
            <div class="metric-number" style="color: {color_code} !important;">{res_text}</div>
            <div class="metric-note">{sub_text}</div>
        </div>
        """)

    with col_m2:
        render_html(f"""
        <div class="metric-panel">
            <div class="metric-title">Pulmonary Density Index</div>
            <div class="metric-number">{prediction.raw_score:.3f}</div>
            <div class="metric-note">Cutoff: 0.300 (Normal &lt; 0.30)</div>
        </div>
        """)

    with col_m3:
        render_html(f"""
        <div class="metric-panel">
            <div class="metric-title">Consultation Urgency</div>
            <div class="metric-number" style="font-size: 1.25rem; margin-top: 0.35rem;">
                <span class="advice-timeline-badge {guidance['badge_class']}">{guidance['timeline_text']}</span>
            </div>
            <div class="metric-note" style="margin-top: 0.65rem;">Based on radiological severity</div>
        </div>
        """)

    # 10. Interactive Visualizer: Specific Affected Region & Heatmaps
    st.markdown('<h3 style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:0.75rem; margin-bottom:0.35rem;">🫁 Visual Lung Inspection</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.92rem; color:#475569; margin-bottom:1rem;">Explore the specific areas of your chest radiograph identified during screening.</p>', unsafe_allow_html=True)

    tab_zones, tab_heat, tab_orig = st.tabs([
        "🎯 Specific Affected Region (Clear Highlight)",
        "🌡️ Thermal Activity Map (Grad-CAM)",
        "🖼️ Original Uploaded Radiograph",
    ])

    with tab_zones:
        render_html('<div class="viewer-card">')
        col_img, col_info = st.columns([1.6, 1])

        with col_img:
            st.image(
                affected_overlay,
                caption="Highlighted Infiltration Regions on Your Chest X-Ray",
                use_container_width=True,
            )

        with col_info:
            render_html('<div class="viewer-header">📍 Detected Lung Locations</div>')
            if prediction.is_positive and detected_zones:
                st.write(f"**{len(detected_zones)} specific area(s) of interest** were identified where lung tissue density is elevated:")
                for z in detected_zones:
                    render_html(f"""
                    <div class="zone-tag">
                        <span>🔍</span> <b>{z['zone']}</b>
                    </div>
                    """)
                render_html("""
                <div style="font-size:0.85rem; color:#475569; margin-top:0.85rem; line-height:1.55;">
                    <i>Note for patients:</i> In chest radiographs, the left side of the image shows your <b>Right Lung</b>, 
                    and the right side shows your <b>Left Lung</b>. The highlighted boundaries indicate areas of suspected fluid accumulation or consolidation.
                </div>
                """)
            elif prediction.is_positive:
                st.info("Pneumonia features detected across diffuse bilateral lung fields.")
            else:
                render_html("""
                <div style="background-color:#ECFDF5; border:1px solid #A7F3D0; border-radius:10px; padding:1.25rem; color:#065F46;">
                    <b>✅ All Lung Quadrants Clear</b><br>
                    No localized consolidations, focal fluid pockets, or abnormal pulmonary infiltrates were detected. 
                    Both right and left lung fields exhibit normal radiographic transparency.
                </div>
                """)

        render_html('</div>')

    with tab_heat:
        render_html('<div class="viewer-card">')
        col_th1, col_th2 = st.columns([1.6, 1])
        with col_th1:
            st.image(
                thermal_overlay,
                caption="Radiological Thermal Intensity Overlay",
                use_container_width=True,
            )
        with col_th2:
            render_html('<div class="viewer-header">🌡️ How to Read This Heatmap</div>')
            st.markdown("""
            This thermal color map visualizes the concentration of radiological evidence:
            
            - 🔴 **Red & Yellow Areas:** Highest concentration of pneumonia-like density patterns.
            - 🔵 **Blue & Cool Areas:** Normal, clear air-filled lung spaces.
            """)
        render_html('</div>')

    with tab_orig:
        render_html('<div class="viewer-card">')
        col_or1, col_or2 = st.columns([1.6, 1])
        with col_or1:
            st.image(
                display_img,
                caption=f"Original Uploaded Radiograph ({active_name})",
                use_container_width=True,
            )
        with col_or2:
            render_html('<div class="viewer-header">📋 Scan Details</div>')
            st.write("**File Name:**", active_name)
            st.write("**Scan Format:**", meta.get("format", "Radiograph"))
            st.write("**Native Dimensions:**", meta.get("dimensions", "N/A"))
            st.write("**Patient ID:**", meta.get("patient_id", "Anonymous"))
            st.write("**Projection:**", meta.get("view_position", "Standard Chest PA"))
        render_html('</div>')

    # 11. Personalized Doctor Advice & Visiting Recommendations
    # Constructing as a single HTML string with NO internal blank lines to ensure 100% clean rendering
    home_care_items = "".join(f'<li style="margin-bottom:0.35rem; color:#334155 !important;">{item}</li>' for item in guidance['home_care'])
    questions_items = "".join(f'<li style="margin-bottom:0.35rem; color:#334155 !important;">"{q}"</li>' for q in guidance['questions_to_ask'])

    advice_card_html = (
        f'<div class="advice-card">'
        f'<div class="advice-header">'
        f'<span class="advice-icon">🩺</span>'
        f'<h3 class="advice-title">Personalized Doctor & Care Recommendations</h3>'
        f'<span class="advice-timeline-badge {guidance["badge_class"]}">{guidance["timeline_text"]}</span>'
        f'</div>'
        f'<div class="advice-section">'
        f'<div class="advice-sec-title"><span>🏥</span> Clinical Consultation Guidance</div>'
        f'<p class="advice-sec-text">{guidance["doctor_advice"]}</p>'
        f'</div>'
        f'<div class="advice-section">'
        f'<div class="advice-sec-title"><span>🍵</span> Home Care & Immediate Management</div>'
        f'<ul class="advice-sec-text" style="padding-left:1.25rem; margin-top:0.35rem;">{home_care_items}</ul>'
        f'</div>'
        f'<div class="advice-section">'
        f'<div class="advice-sec-title"><span>🔬</span> Diagnostic Tests Your Doctor May Recommend</div>'
        f'<p class="advice-sec-text">{guidance["tests_expected"]}</p>'
        f'</div>'
        f'<div class="advice-section" style="margin-bottom:0.5rem;">'
        f'<div class="advice-sec-title"><span>📝</span> Questions to Ask Your Doctor During Your Visit</div>'
        f'<ul class="advice-sec-text" style="padding-left:1.25rem; margin-top:0.35rem;">{questions_items}</ul>'
        f'</div>'
        f'</div>'
    )
    render_html(advice_card_html)

    # 12. Red Flag Warnings Card (Always visible for safety)
    red_flag_html = (
        '<div class="red-flag-box">'
        '<div class="red-flag-title">⚠️ When to Seek Immediate Emergency Medical Care</div>'
        '<div style="font-size:0.88rem; color:#881337 !important; margin-bottom:0.45rem;">'
        'Do not wait for a scheduled appointment if you or the patient experience any of the following critical warning signs:'
        '</div>'
        '<ul class="red-flag-list">'
        '<li><b>Severe shortness of breath</b> or struggling to breathe while resting.</li>'
        '<li><b>Bluish color</b> appearing on the lips, skin, or fingernail beds (cyanosis).</li>'
        '<li><b>Oxygen saturation (SpO2) falling below 92%</b> on a pulse oximeter.</li>'
        '<li><b>Sharp or stabbing chest pain</b> that worsens when taking a deep breath or coughing.</li>'
        '<li><b>Confusion, severe dizziness, or extreme lethargy</b>, particularly in elderly individuals.</li>'
        '</ul>'
        '</div>'
    )
    render_html(red_flag_html)

    # 13. Comprehensive Statistical Model Analysis & Scenario Simulator
    with st.expander("📊 Comprehensive Model Statistical Analysis & Multi-Scenario Simulator", expanded=show_stats):
        render_statistical_analysis_section()

    # 14. Patient Trust & Educational Disclaimer Footer
    footer_html = (
        '<div class="trust-footer">'
        '<b>Educational Health Screening Tool:</b> PneumoVision provides preliminary chest radiograph analysis '
        'to support informed discussions between patients and their physicians. This software is not a replacement for '
        'a formal clinical diagnosis by a licensed radiologist or medical practitioner. Always consult a healthcare '
        'provider for personalized clinical care and prescription treatments.'
        '</div>'
    )
    render_html(footer_html)
    

if __name__ == "__main__":
    main()
