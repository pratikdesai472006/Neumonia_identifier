"""PneumoVision — AI Chest X-Ray Pneumonia Screening.

Professional Computer Vision / Medical Research Streamlit Application
Powered by locally vendored TorchXRayVision DenseNet-121 RSNA.
"""

import io
from pathlib import Path
import sys
from typing import Optional, Tuple

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
# Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoVision — Chest Radiograph Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Clean, medical/research background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Hide default Streamlit decoration header */
    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.9);
        backdrop-filter: blur(8px);
    }

    /* Header / Hero Section */
    .hero-container {
        padding: 1rem 0 1.25rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0B192C;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-subtitle-primary {
        font-size: 1.12rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 0.35rem;
    }
    .hero-subtitle-desc {
        font-size: 0.95rem;
        font-weight: 400;
        color: #475569;
        margin-top: 0.25rem;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Subtle 3-Step Workflow Indicator */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 1rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        margin-bottom: 1.5rem;
    }
    .wf-step {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    .wf-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #0284C7;
        background-color: #E0F2FE;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    .wf-arrow {
        color: #94A3B8;
        font-size: 0.85rem;
        user-select: none;
    }

    /* Sidebar: Model Information Specification Cards */
    .sidebar-section-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #0B192C;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .spec-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0284C7;
        border-radius: 6px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.5rem;
    }
    .spec-card-title {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .spec-card-value {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.1rem;
    }

    /* Upload Panel */
    .upload-panel-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem 0.85rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .upload-panel-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.35rem;
    }
    .upload-icon {
        font-size: 1.4rem;
        line-height: 1;
    }
    .upload-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0B192C;
    }
    .upload-formats {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.75rem;
    }
    .upload-formats code {
        background-color: #F1F5F9;
        color: #0369A1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Empty State Container (Before Analysis) */
    .empty-state-card {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 3.5rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .empty-icon {
        font-size: 3.2rem;
        margin-bottom: 0.6rem;
    }
    .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0B192C;
        margin-bottom: 0.3rem;
    }
    .empty-desc {
        font-size: 0.92rem;
        color: #64748B;
        max-width: 420px;
        margin: 0 auto;
    }

    /* Results Dashboard */
    .results-container {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .result-banner-pos {
        background-color: #FFFFFF;
        border: 2px solid #E11D48;
        border-left: 8px solid #E11D48;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        height: 100%;
    }
    .result-banner-neg {
        background-color: #FFFFFF;
        border: 2px solid #16A34A;
        border-left: 8px solid #16A34A;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        height: 100%;
    }
    .result-eyebrow {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .result-val-pos {
        font-size: 2.15rem;
        font-weight: 800;
        color: #BE123C;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-val-neg {
        font-size: 2.15rem;
        font-weight: 800;
        color: #15803D;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-subtitle {
        font-size: 0.88rem;
        color: #475569;
        margin-top: 0.35rem;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #0B192C;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Section Headers */
    .section-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0B192C;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }
    .section-header-desc {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 1rem;
        font-style: italic;
    }

    /* Side-by-side Image Panels */
    .image-panel-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem;
        text-align: center;
    }
    .image-panel-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0B192C;
        margin-bottom: 0.6rem;
    }

    /* Intentional Research Prototype Disclaimer */
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin: 2rem 0 1.5rem 0;
    }
    .disclaimer-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #92400E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .disclaimer-text {
        font-size: 0.86rem;
        color: #78350F;
        margin: 0;
        line-height: 1.45;
    }

    /* Footer */
    .footer-box {
        border-top: 1px solid #E2E8F0;
        padding: 1.5rem 0;
        margin-top: 2.5rem;
        text-align: center;
    }
    .footer-primary {
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
    }
    .footer-secondary {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader (Single-Process Initialization)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DenseNet-121 RSNA weights...")
def get_model() -> PneumoniaModel:
    """Load and cache the PneumoniaModel wrapper."""
    cache_path = ROOT_DIR / "checkpoints"
    return PneumoniaModel(
        weights="densenet121-res224-rsna",
        cache_dir=str(cache_path),
    )


# -----------------------------------------------------------------------------
# Input Preprocessing Functions (Exact Validated Logic Preserved)
# -----------------------------------------------------------------------------
def preprocess_dicom_bytes(file_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Parse and normalize raw DICOM bytes using official XRV pipeline."""
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

    # Invert MONOCHROME1 to match standard MONOCHROME2
    if photo_interp == "MONOCHROME1":
        data = max_val - data

    # Ensure square aspect ratio via symmetric padding to preserve anatomy
    orig_h, orig_w = data.shape
    if orig_h != orig_w:
        from src.data.dicom_utils import pad_to_square_np
        data, _ = pad_to_square_np(data, pad_value=float(data.min()))

    # Scale to [-1024, 1024] as required by TorchXRayVision
    norm_data = xrv.utils.normalize(data, max_val)

    # (1, 1, H, W) float tensor
    tensor = torch.from_numpy(norm_data[None, None, ...]).float()

    # 8-bit display representation
    d_min, d_max = data.min(), data.max()
    display_uint8 = np.uint8(255.0 * (data - d_min) / (d_max - d_min + 1e-6))
    display_pil = Image.fromarray(display_uint8, mode="L").convert("RGB")

    metadata = {
        "format": "DICOM (.dcm)",
        "dimensions": f"{orig_h} × {orig_w}" if orig_h == orig_w else f"{orig_h} × {orig_w} (Square-padded)",
        "patient_id": getattr(ds, "PatientID", "ANONYMOUS"),
        "view_position": getattr(ds, "ViewPosition", "N/A"),
        "photometric_interpretation": photo_interp,
    }
    return tensor, display_pil, metadata


def preprocess_image_bytes(file_bytes: bytes, filename: str) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Load and normalize standard PNG/JPG/JPEG image bytes with non-square padding."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as e:
        raise ValueError(f"Invalid or unreadable image file: {e}")

    img_arr = np.array(pil_img, dtype=np.float32)
    orig_h, orig_w = img_arr.shape

    # Symmetrically pad non-square images to square
    if orig_h != orig_w:
        from src.data.dicom_utils import pad_to_square_np, pad_to_square_pil
        img_arr, _ = pad_to_square_np(img_arr, pad_value=0.0)
        display_pil, _ = pad_to_square_pil(pil_img.convert("RGB"), bg_color=0)
    else:
        display_pil = pil_img.convert("RGB")

    # Normalize [0..255] to [-1024, 1024] via XRV standard
    norm_data = xrv.utils.normalize(img_arr, 255)

    tensor = torch.from_numpy(norm_data[None, None, ...]).float()

    metadata = {
        "format": f"Standard Image ({Path(filename).suffix.upper().replace('.', '')})",
        "dimensions": f"{orig_h} × {orig_w}" if orig_h == orig_w else f"{orig_h} × {orig_w} (Square-padded)",
        "patient_id": Path(filename).stem,
        "view_position": "Standard Grayscale",
        "photometric_interpretation": "Grayscale 8-bit",
    }
    return tensor, display_pil, metadata


# -----------------------------------------------------------------------------
# Main Application Interface
# -----------------------------------------------------------------------------
def main():
    # 1. Header / Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🫁 PneumoVision</h1>
        <div class="hero-subtitle-primary">Automated Pneumonia Screening from Chest Radiographs</div>
        <div class="hero-subtitle-desc">
            Research-oriented chest X-ray analysis using a pretrained DenseNet-121 model trained on the RSNA Pneumonia Detection Challenge.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Model Initialization
    try:
        model_wrapper = get_model()
    except Exception as e:
        st.error(f"Failed to initialize DenseNet-121 model: {e}")
        return

    # 3. Sidebar: Model Information (Compact Specification Cards)
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Model Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="spec-card">
            <div class="spec-card-title">Architecture</div>
            <div class="spec-card-value">DenseNet-121</div>
        </div>
        <div class="spec-card">
            <div class="spec-card-title">Pretraining</div>
            <div class="spec-card-value">RSNA Pneumonia Detection Challenge</div>
        </div>
        <div class="spec-card">
            <div class="spec-card-title">Input</div>
            <div class="spec-card-value">224 × 224</div>
        </div>
        <div class="spec-card">
            <div class="spec-card-title">Target</div>
            <div class="spec-card-value">Pneumonia</div>
        </div>
        <div class="spec-card">
            <div class="spec-card-title">Decision Threshold</div>
            <div class="spec-card-value">0.3000</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">Demonstration Samples</div>', unsafe_allow_html=True)
        sample_mode = st.radio(
            "Select input source:",
            options=["Upload Custom X-Ray", "Sample 1: RSNA DICOM (Pneumonia)", "Sample 2: Converted PNG"],
            index=0,
            label_visibility="collapsed",
        )

    # 4. Workflow Indicator
    st.markdown("""
    <div class="workflow-container">
        <div class="wf-step"><span class="wf-badge">01</span> Upload X-Ray</div>
        <div class="wf-arrow">→</div>
        <div class="wf-step"><span class="wf-badge">02</span> DenseNet-121 Screening</div>
        <div class="wf-arrow">→</div>
        <div class="wf-step"><span class="wf-badge">03</span> Review Grad-CAM</div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Acquire Input File (Upload Section as Visual Focus)
    uploaded_file = None
    sample_file_bytes = None
    sample_file_name = None

    if sample_mode == "Upload Custom X-Ray":
        st.markdown("""
        <div class="upload-panel-box">
            <div class="upload-panel-header">
                <span class="upload-icon">📂</span>
                <span class="upload-title">Upload Chest X-Ray</span>
            </div>
            <div class="upload-formats">
                Supported formats: <code>.DICOM</code> &nbsp;•&nbsp; <code>.PNG</code> &nbsp;•&nbsp; <code>.JPG</code> &nbsp;•&nbsp; <code>.JPEG</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray file or drag and drop here",
            type=["dcm", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="Supported: .DICOM, .PNG, .JPG, .JPEG. Files are processed strictly in-memory.",
        )
    elif sample_mode == "Sample 1: RSNA DICOM (Pneumonia)":
        sample_path = ROOT_DIR / "dataset" / "stage_2_train_images" / "0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm"
        if sample_path.exists():
            sample_file_bytes = sample_path.read_bytes()
            sample_file_name = sample_path.name
            st.info(f"Loaded Demonstration DICOM: `{sample_file_name}`")
        else:
            st.error("Demonstration DICOM not found on disk.")
    elif sample_mode == "Sample 2: Converted PNG":
        sample_path = ROOT_DIR / "dataset" / "raw" / "images" / "0004cfab-14fd-4e49-80ba-63a80b6bddd6.png"
        if sample_path.exists():
            sample_file_bytes = sample_path.read_bytes()
            sample_file_name = sample_path.name
            st.info(f"Loaded Demonstration PNG: `{sample_file_name}`")
        else:
            st.error("Demonstration PNG not found on disk.")

    active_bytes = uploaded_file.getvalue() if uploaded_file else sample_file_bytes
    active_name = uploaded_file.name if uploaded_file else sample_file_name

    # 6. Before Analysis: Empty-State Display (Disappears once uploaded)
    if active_bytes is None:
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-icon">🩻</div>
            <div class="empty-title">Ready for analysis</div>
            <div class="empty-desc">Upload a chest X-ray to begin screening.</div>
        </div>
        """, unsafe_allow_html=True)

        # Subtle Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <div class="disclaimer-title">Research Prototype</div>
            <p class="disclaimer-text">Automated screening output, not a medical diagnosis. Clinical interpretation by a qualified healthcare professional is required.</p>
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div class="footer-box">
            <div class="footer-primary">PneumoVision — Computer Vision Open Elective Project</div>
            <div class="footer-secondary">B.Tech Computer Engineering</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 7. Process Input Bytes
    try:
        file_ext = Path(active_name).suffix.lower()
        if file_ext == ".dcm":
            input_tensor, display_img, meta = preprocess_dicom_bytes(active_bytes)
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            input_tensor, display_img, meta = preprocess_image_bytes(active_bytes, active_name)
        else:
            st.error(f"Unsupported format: '{file_ext}'. Supported formats: .dcm, .png, .jpg, .jpeg")
            return
    except Exception as err:
        st.error(f"Image Preprocessing Error: {err}")
        return

    # 8. Inference & Grad-CAM Execution
    with st.spinner("Analyzing radiograph with DenseNet-121 and generating Grad-CAM..."):
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
            overlay_img, _ = gradcam.overlay_heatmap(
                original_image=display_img,
                heatmap=heatmap,
                alpha=0.45,
                colormap=cv2.COLORMAP_JET,
            )
            gradcam.remove_hooks()
        except Exception as e:
            st.error(f"Inference / Explainability Error: {e}")
            return

    # 9. Results Section (Clear Results Dashboard)
    st.markdown("---")
    col_result, col_score, col_threshold = st.columns([1.5, 1, 1])

    with col_result:
        if prediction.is_positive:
            st.markdown(
                f"""
                <div class="result-banner-pos">
                    <div class="result-eyebrow">SCREENING RESULT</div>
                    <div class="result-val-pos">🔴 {prediction.prediction}</div>
                    <div class="result-subtitle">Model score exceeds calibrated operational cutoff.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-banner-neg">
                    <div class="result-eyebrow">SCREENING RESULT</div>
                    <div class="result-val-neg">🟢 {prediction.prediction}</div>
                    <div class="result-subtitle">Model score remains below calibrated operational cutoff.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_score:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Score</div>
                <div class="metric-val">{prediction.raw_score:.4f}</div>
                <div class="metric-sub">Raw model output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_threshold:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Decision Threshold</div>
                <div class="metric-val">{prediction.threshold:.4f}</div>
                <div class="metric-sub">RSNA operational threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("ℹ️ **Model Score Clarification:** The model score reflects the operational output of the RSNA-trained DenseNet-121. It is compared directly to the decision threshold (0.3000) to determine the binary assessment. It is not an uncalibrated clinical probability.")

    # 10. Explainability Section
    st.markdown('<div class="section-header-title">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header-desc">Grad-CAM highlights image regions that contributed to the model\'s Pneumonia output.</div>', unsafe_allow_html=True)

    col_orig, col_cam = st.columns(2)
    with col_orig:
        st.markdown('<div class="image-panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-panel-title">Original X-Ray</div>', unsafe_allow_html=True)
        st.image(
            display_img,
            caption=f"Input Radiograph ({meta['dimensions']})",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="image-panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-panel-title">Grad-CAM</div>', unsafe_allow_html=True)
        st.image(
            overlay_img,
            caption="Grad-CAM Activation Heatmap Overlay",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 11. Technical Analysis Details (Expandable)
    with st.expander("🔍 Radiograph & Model Metadata"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**File Name:**", active_name)
            st.write("**Image Format:**", meta["format"])
            st.write("**Dimensions:**", meta["dimensions"])
            st.write("**Photometric Mode:**", meta["photometric_interpretation"])
            st.write("**Patient ID:**", meta["patient_id"])
        with col_m2:
            st.write("**Model Architecture:**", "DenseNet-121")
            st.write("**Pretrained Weights:**", "densenet121-res224-rsna.pt")
            st.write("**Input Resolution:**", "224 × 224 pixels")
            st.write("**Target Node:**", f"{prediction.target_name} (Index {prediction.target_index})")
            st.write("**Grad-CAM Target Layer:**", "features.denseblock4 (1,024 channels)")
            st.write("**Calibrated Threshold:**", f"{prediction.threshold:.6f}")

    # 12. Intentional Research Prototype Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <div class="disclaimer-title">Research Prototype</div>
        <p class="disclaimer-text">Automated screening output, not a medical diagnosis. Clinical interpretation by a qualified healthcare professional is required.</p>
    </div>
    """, unsafe_allow_html=True)

    # 13. Footer
    st.markdown("""
    <div class="footer-box">
        <div class="footer-primary">PneumoVision — Computer Vision Open Elective Project</div>
        <div class="footer-secondary">B.Tech Computer Engineering</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
