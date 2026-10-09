"""Verification Script: End-to-End RSNA Inference & Real Grad-CAM Visualization.

Loads:
- Locally vendored TorchXRayVision DenseNet-121 RSNA
- Pretrained weights: checkpoints/densenet121-res224-rsna.pt
- Test DICOM: dataset/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm

Executes:
1. Official XRV DICOM preprocessing (MONOCHROME correction + [-1024, 1024] normalization)
2. PneumoniaModel inference wrapper
3. Grad-CAM generation for the Pneumonia target node
4. Blended heatmap overlay saved to results/gradcam_test/
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

# Ensure local project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "torchxrayvision"))

import torchxrayvision as xrv
from src.models.pneumonia_model import PneumoniaModel
from src.explainability.gradcam import GradCAM

# 1. Inputs & Outputs Configuration
TEST_DICOM = ROOT_DIR / "dataset" / "stage_2_train_images" / "0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm"
OUTPUT_DIR = ROOT_DIR / "results" / "gradcam_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "0004cfab-14fd-4e49-80ba-63a80b6bddd6_gradcam.png"


def main():
    if not TEST_DICOM.exists():
        raise FileNotFoundError(f"Test DICOM not found at: {TEST_DICOM.resolve()}")

    # 2. Instantiate PneumoniaModel wrapper
    wrapper = PneumoniaModel(
        weights="densenet121-res224-rsna",
        cache_dir=str(ROOT_DIR / "checkpoints"),
    )

    # 3. Read and preprocess DICOM via official TorchXRayVision pipeline
    # Returns 2D float array in [-1024, 1024]
    img_np = xrv.utils.read_xray_dcm(str(TEST_DICOM))
    orig_h, orig_w = img_np.shape

    # For original image visualization, convert [-1024, 1024] to [0, 255] uint8
    norm_vis = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    orig_uint8 = (norm_vis * 255.0).astype(np.uint8)

    # Shape tensor for model input: (1, 1, H, W)
    input_tensor = torch.from_numpy(img_np[None, None, ...]).float()

    # 4. Run inference via our PneumoniaModel wrapper
    prediction_result = wrapper.predict(input_tensor)

    # 5. Initialize Grad-CAM targeting denseblock4
    gradcam = GradCAM(
        model=wrapper.raw_model,
        target_layer=wrapper.get_gradcam_target_layer(),
        target_index=prediction_result.target_index,
    )

    # 6. Generate heatmap for the Pneumonia output node
    heatmap = gradcam.generate_heatmap(
        input_tensor=input_tensor,
        target_index=prediction_result.target_index,
    )

    # 7. Create blended overlay
    overlay_img, resized_cam = gradcam.overlay_heatmap(
        original_image=orig_uint8,
        heatmap=heatmap,
        alpha=0.45,
    )

    # Clean up hooks
    gradcam.remove_hooks()

    # 8. Save visualization (side-by-side: Original | Grad-CAM Overlay)
    side_by_side = Image.new("RGB", (orig_w * 2, orig_h))
    orig_pil = Image.fromarray(orig_uint8, mode="L").convert("RGB")
    side_by_side.paste(orig_pil, (0, 0))
    side_by_side.paste(overlay_img, (orig_w, 0))
    side_by_side.save(OUTPUT_FILE)

    # 9. Print strict report format
    print(f"MODEL: DenseNet-121 RSNA")
    print(f"TARGET: {prediction_result.target_name}")
    print(f"TARGET_INDEX: {prediction_result.target_index}")
    print(f"RAW_SCORE: {prediction_result.raw_score:.6f}")
    print(f"THRESHOLD: {prediction_result.threshold:.8f}")
    print(f"PREDICTION: {prediction_result.prediction}")
    print(f"GRADCAM: PASS")
    print(f"GRADCAM_SHAPE: {list(heatmap.shape)}")
    print(f"OUTPUT_FILE: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
