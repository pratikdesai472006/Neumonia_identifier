"""PneumoVision — Patient-Friendly Chest Radiograph Screening & Health Advisory.

A compassionate, modern medical health portal for automated chest X-ray screening,
providing localized lung area visualization and personalized doctor visiting advice.
"""

import io
from pathlib import Path
import sys
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

st.markdown("""
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Calming clinical digital health background */
    .stApp {
        background-color: #F8FAFC;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(10px);
    }

    /* Top Navigation / Hero Header */
    .portal-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.75rem;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.05);
    }
    .portal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #E0F2FE;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.02em;
    }
    .portal-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .portal-subtitle {
        font-size: 1.02rem;
        font-weight: 400;
        color: #93C5FD;
        margin-top: 0.5rem;
        max-width: 780px;
        line-height: 1.55;
    }

    /* Workflow Steps Card */
    .flow-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .flow-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: #334155;
    }
    .flow-num {
        background-color: #E0F2FE;
        color: #0284C7;
        font-weight: 800;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
    }
    .flow-arrow {
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* Upload Panel */
    .upload-card {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
    }
    .upload-card:hover {
        border-color: #0284C7;
        background-color: #F8FAFC;
    }
    .upload-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .upload-sub {
        font-size: 0.88rem;
        color: #64748B;
    }

    /* Screening Status Cards */
    .status-card-normal {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #6EE7B7;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        color: #065F46;
        margin-bottom: 1.5rem;
    }
    .status-card-positive {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FCA5A5;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        color: #991B1B;
        margin-bottom: 1.5rem;
    }
    .status-eyebrow {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .status-heading {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-desc {
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.5;
        opacity: 0.95;
    }

    /* Risk Score Gauge & Metrics */
    .metric-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .metric-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-note {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Doctor Advice & Recommendations Box */
    .advice-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.75rem;
        margin-top: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .advice-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.85rem;
    }
    .advice-icon {
        font-size: 1.5rem;
    }
    .advice-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
    }
    .advice-timeline-badge {
        margin-left: auto;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .timeline-routine {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .timeline-moderate {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    .timeline-urgent {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }

    .advice-section {
        margin-bottom: 1.25rem;
    }
    .advice-sec-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .advice-sec-text {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.6;
        margin: 0;
    }

    /* Red Flags Warning Box */
    .red-flag-box {
        background-color: #FFF1F2;
        border-left: 4px solid #E11D48;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }
    .red-flag-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #9F1239;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.35rem;
    }
    .red-flag-list {
        font-size: 0.84rem;
        color: #881337;
        margin: 0;
        padding-left: 1.2rem;
        line-height: 1.5;
    }

    /* Detected Zone Pill Tags */
    .zone-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }

    /* Image Display Container */
    .viewer-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .viewer-header {
        font-size: 0.92rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Sidebar Clean Styling */
    .sidebar-block {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.15rem;
        margin-bottom: 1.25rem;
    }
    .sidebar-block-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-block-body {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Trust & Disclaimer Footer */
    .trust-footer {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


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
    """Highlight specifically where in the lung the pneumonia opacity is located.
    
    Draws smooth illuminated boundary contours around the affected areas,
    labels the anatomical lung quadrant, and computes the affected area.
    """
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
        
        # Patient-friendly highlighting colors: warm coral fill with luminous cyan border
        color_fill = np.array([239, 68, 68], dtype=np.uint8)  # soft clinical red
        color_border = (0, 220, 255)                          # luminous cyan
        
        fill_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            # Filter negligible pixel noise
            if area > 1000:
                cv2.drawContours(fill_mask, [c], -1, 255, -1)
                cv2.drawContours(overlay, [c], -1, color_border, 3)
                
                x, y, bw, bh = cv2.boundingRect(c)
                cx, cy = x + bw // 2, y + bh // 2
                
                # Radiograph anatomical orientation:
                # Left on image is patient's Right Lung; Right on image is patient's Left Lung
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
                
                # Draw subtle, legible anatomical banner directly above each bounding zone
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
        
        # Soft transparent tint over the affected lung tissue
        alpha = 0.32
        colored_fill = np.zeros_like(img_rgb)
        colored_fill[fill_mask > 0] = color_fill
        overlay = cv2.addWeighted(overlay, 1.0, colored_fill, alpha, 0)
    
    return Image.fromarray(overlay), detected_zones


# -----------------------------------------------------------------------------
# Clinical Triage & Doctor Visiting Guidance Engine
# -----------------------------------------------------------------------------
def get_clinical_guidance(score: float, is_positive: bool) -> Dict[str, Any]:
    """Generate personalized medical advice and visiting recommendations.
    
    Triage Tiers:
    - Normal / Clear: Score < 0.20
    - Mild / Borderline: 0.20 <= Score < 0.30
    - Moderate Opacity / Pneumonia: 0.30 <= Score < 0.60
    - High / Severe Consolidation: Score >= 0.60
    """
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
                "If you currently have mild cold or flu-like symptoms, prioritize rest and stay well hydrated. "
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
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # 1. Top Portal Header
    st.markdown("""
    <div class="portal-header">
        <div class="portal-badge">
            <span>🛡️</span> Secure & Private • Local Offline Analysis
        </div>
        <h1 class="portal-title">PneumoVision — Chest X-Ray Screening & Health Guide</h1>
        <div class="portal-subtitle">
            Instant, automated chest radiograph screening designed for patients. View specific affected 
            lung areas and receive tailored doctor visiting advice and home care recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Patient Guidance Workflow Banner
    st.markdown("""
    <div class="flow-card">
        <div class="flow-item"><span class="flow-num">1</span> Upload Your Chest X-Ray</div>
        <div class="flow-arrow">→</div>
        <div class="flow-item"><span class="flow-num">2</span> View Highlighted Lung Areas</div>
        <div class="flow-arrow">→</div>
        <div class="flow-item"><span class="flow-num">3</span> Receive Tailored Doctor Advice</div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div class="sidebar-block">
            <div class="sidebar-block-title">🧑‍⚕️ How to Use This Portal</div>
            <div class="sidebar-block-body">
                <b>1. Upload a Chest X-Ray:</b> Standard hospital DICOM (.dcm) or image files (.png, .jpg) are supported.<br><br>
                <b>2. No Login Required:</b> Your scan is analyzed immediately on your device without registering.<br><br>
                <b>3. Review Doctor Advice:</b> Learn whether an in-person doctor consultation is needed today.
            </div>
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="sidebar-block" style="margin-top:1.25rem; background-color:#EFF6FF; border-color:#BFDBFE;">
            <div class="sidebar-block-title" style="color:#1E40AF;">🔒 100% Patient Privacy</div>
            <div class="sidebar-block-body" style="color:#1E3A8A;">
                All scans are analyzed locally inside your computer's memory. No personal data or medical images are ever stored on cloud servers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Acquire Input File
    uploaded_file = None
    sample_file_bytes = None
    sample_file_name = None

    if sample_choice == "Upload My Own X-Ray":
        st.markdown("""
        <div class="upload-card">
            <div class="upload-heading">Select or Drag & Drop Your Chest X-Ray</div>
            <div class="upload-sub">Supports hospital DICOM files (.dcm) as well as standard images (.png, .jpg, .jpeg)</div>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown(
            f"""
            <div class="status-card-positive">
                <div class="status-eyebrow">SCREENING ASSESSMENT</div>
                <div class="status-heading">🔴 Suspected Pneumonia Detected</div>
                <div class="status-desc">{guidance['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-card-normal">
                <div class="status-eyebrow">SCREENING ASSESSMENT</div>
                <div class="status-heading">🟢 Clear Lung Fields — No Pneumonia Detected</div>
                <div class="status-desc">{guidance['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 9. Key Metrics & Health Index Row
    col_m1, col_m2, col_m3 = st.columns([1.2, 1, 1.2])

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-title">Screening Result</div>
                <div class="metric-number" style="color: {'#DC2626' if prediction.is_positive else '#059669'};">
                    {'POSITIVE' if prediction.is_positive else 'NEGATIVE'}
                </div>
                <div class="metric-note">{'Opacities identified in scan' if prediction.is_positive else 'Clear lung fields'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        # Normalized confidence level for patient clarity
        confidence_pct = min(100.0, max(0.0, prediction.raw_score * 100.0))
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-title">Pulmonary Density Index</div>
                <div class="metric-number">{prediction.raw_score:.3f}</div>
                <div class="metric-note">Cutoff: 0.300 (Normal &lt; 0.30)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-title">Consultation Urgency</div>
                <div class="metric-number" style="font-size: 1.25rem; margin-top: 0.35rem;">
                    <span class="advice-timeline-badge {guidance['badge_class']}">{guidance['timeline_text']}</span>
                </div>
                <div class="metric-note" style="margin-top: 0.65rem;">Based on radiological severity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 10. Interactive Visualizer: Specific Affected Region & Heatmaps
    st.markdown('<h3 style="font-size:1.35rem; font-weight:800; color:#0F172A; margin-top:0.75rem; margin-bottom:0.35rem;">🫁 Visual Lung Inspection</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.92rem; color:#64748B; margin-bottom:1rem;">Explore the specific areas of your chest radiograph identified during screening.</p>', unsafe_allow_html=True)

    tab_zones, tab_heat, tab_orig = st.tabs([
        "🎯 Specific Affected Region (Clear Highlight)",
        "🌡️ Thermal Activity Map (Grad-CAM)",
        "🖼️ Original Uploaded Radiograph",
    ])

    with tab_zones:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1.6, 1])

        with col_img:
            st.image(
                affected_overlay,
                caption="Highlighted Infiltration Regions on Your Chest X-Ray",
                use_container_width=True,
            )

        with col_info:
            st.markdown('<div class="viewer-header">📍 Detected Lung Locations</div>', unsafe_allow_html=True)
            if prediction.is_positive and detected_zones:
                st.write(f"**{len(detected_zones)} specific area(s) of interest** were identified where lung tissue density is elevated:")
                for z in detected_zones:
                    st.markdown(
                        f"""
                        <div class="zone-tag">
                            <span>🔍</span> <b>{z['zone']}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("""
                <div style="font-size:0.83rem; color:#64748B; margin-top:0.85rem; line-height:1.5;">
                    <i>Note for patients:</i> In chest radiographs, the left side of the image shows your <b>Right Lung</b>, 
                    and the right side shows your <b>Left Lung</b>. The highlighted boundaries indicate areas of suspected fluid accumulation or consolidation.
                </div>
                """, unsafe_allow_html=True)
            elif prediction.is_positive:
                st.info("Pneumonia features detected across diffuse bilateral lung fields.")
            else:
                st.markdown("""
                <div style="background-color:#ECFDF5; border:1px solid #A7F3D0; border-radius:10px; padding:1.25rem; color:#065F46;">
                    <b>✅ All Lung Quadrants Clear</b><br>
                    No localized consolidations, focal fluid pockets, or abnormal pulmonary infiltrates were detected. 
                    Both right and left lung fields exhibit normal radiographic transparency.
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_heat:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        col_th1, col_th2 = st.columns([1.6, 1])
        with col_th1:
            st.image(
                thermal_overlay,
                caption="Radiological Thermal Intensity Overlay",
                use_container_width=True,
            )
        with col_th2:
            st.markdown('<div class="viewer-header">🌡️ How to Read This Heatmap</div>', unsafe_allow_html=True)
            st.write(
                "This thermal color map visualizes the concentration of radiological evidence:\n\n"
                "- 🔴 **Red & Yellow Areas:** Highest concentration of pneumonia-like density patterns.\n"
                "- 🔵 **Blue & Cool Areas:** Normal, clear air-filled lung spaces."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_orig:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        col_or1, col_or2 = st.columns([1.6, 1])
        with col_or1:
            st.image(
                display_img,
                caption=f"Original Uploaded Radiograph ({active_name})",
                use_container_width=True,
            )
        with col_or2:
            st.markdown('<div class="viewer-header">📋 Scan Details</div>', unsafe_allow_html=True)
            st.write("**File Name:**", active_name)
            st.write("**Scan Format:**", meta.get("format", "Radiograph"))
            st.write("**Native Dimensions:**", meta.get("dimensions", "N/A"))
            st.write("**Patient ID:**", meta.get("patient_id", "Anonymous"))
            st.write("**Projection:**", meta.get("view_position", "Standard Chest PA"))
        st.markdown('</div>', unsafe_allow_html=True)

    # 11. Personalized Doctor Advice & Visiting Recommendations
    st.markdown(
        f"""
        <div class="advice-card">
            <div class="advice-header">
                <span class="advice-icon">🩺</span>
                <h3 class="advice-title">Personalized Doctor & Care Recommendations</h3>
                <span class="advice-timeline-badge {guidance['badge_class']}">{guidance['timeline_text']}</span>
            </div>
            
            <div class="advice-section">
                <div class="advice-sec-title"><span>🏥</span> Clinical Consultation Guidance</div>
                <p class="advice-sec-text">{guidance['doctor_advice']}</p>
            </div>
            
            <div class="advice-section">
                <div class="advice-sec-title"><span>🍵</span> Home Care & Immediate Management</div>
                <ul class="advice-sec-text" style="padding-left: 1.25rem; margin-top: 0.35rem;">
                    {''.join(f'<li style="margin-bottom:0.25rem;">{item}</li>' for item in guidance['home_care'])}
                </ul>
            </div>
            
            <div class="advice-section">
                <div class="advice-sec-title"><span>🔬</span> Diagnostic Tests Your Doctor May Recommend</div>
                <p class="advice-sec-text">{guidance['tests_expected']}</p>
            </div>

            <div class="advice-section" style="margin-bottom: 0.5rem;">
                <div class="advice-sec-title"><span>📝</span> Questions to Ask Your Doctor During Your Visit</div>
                <ul class="advice-sec-text" style="padding-left: 1.25rem; margin-top: 0.35rem;">
                    {''.join(f'<li style="margin-bottom:0.25rem;">"{q}"</li>' for q in guidance['questions_to_ask'])}
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 12. Red Flag Warnings Card (Always visible for safety)
    st.markdown("""
    <div class="red-flag-box">
        <div class="red-flag-title">⚠️ When to Seek Immediate Emergency Medical Care</div>
        <div style="font-size: 0.84rem; color: #881337; margin-bottom: 0.35rem;">
            Do not wait for a scheduled appointment if you or the patient experience any of the following critical warning signs:
        </div>
        <ul class="red-flag-list">
            <li><b>Severe shortness of breath</b> or struggling to breathe while resting.</li>
            <li><b>Bluish color</b> appearing on the lips, skin, or fingernail beds (cyanosis).</li>
            <li><b>Oxygen saturation (SpO2) falling below 92%</b> on a pulse oximeter.</li>
            <li><b>Sharp or stabbing chest pain</b> that worsens when taking a deep breath or coughing.</li>
            <li><b>Confusion, severe dizziness, or extreme lethargy</b>, particularly in elderly individuals.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 13. Patient Trust & Educational Disclaimer Footer
    st.markdown("""
    <div class="trust-footer">
        <b>Educational Health Screening Tool:</b> PneumoVision provides preliminary chest radiograph analysis 
        to support informed discussions between patients and their physicians. This software is not a replacement for 
        a formal clinical diagnosis by a licensed radiologist or medical practitioner. Always consult a healthcare 
        provider for personalized clinical care and prescription treatments.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
