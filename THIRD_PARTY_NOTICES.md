# Third-Party Software & Model Notices

This project (`PneumoVision`) integrates and locally vendors open-source components from **TorchXRayVision** under the terms of the Apache License 2.0.

---

### Project Attribution: TorchXRayVision

- **Original Project**: TorchXRayVision
- **Official Repository**: https://github.com/mlmed/torchxrayvision
- **Official Publication**:
  Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Parsa Torabian, Matteo Guarrera, Matthew P. Lungren, Akshay Chaudhari, Rupert Brooks, Mohammad Hashir, Hao-Ren Jia.
  *TorchXRayVision: An Open-Source Software Library for Chest X-Ray Research*.
  Conference on Medical Imaging with Deep Learning (MIDL), 2022.
- **Vendored Location**: `third_party/torchxrayvision/`
- **License**: Apache License 2.0 (see `third_party/torchxrayvision/LICENSE`)

---

### Pretrained Model & Weights: DenseNet-121 RSNA Pneumonia

- **Architecture**: DenseNet-121 (Huang et al., 2017)
- **Input Resolution**: 224 × 224 pixels, single-channel (grayscale), normalized to [-1024, 1024]
- **Target Output Class**: `"Pneumonia"` (Class index 8 in default pathologies)
- **Official Calibrated Operating Threshold**: `0.13486601` (~0.135)
- **Pretrained Weights Identifier**: `densenet121-res224-rsna`
- **Weight Checkpoint File**: `checkpoints/densenet121-res224-rsna.pt` (27.07 MB)
- **Primary Source**: https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
- **Mirror Source**: https://huggingface.co/torchxrayvision/densenet121-res224-rsna

---

### Our Project Contribution & Scope

The core model architecture and pretrained weights were created by the TorchXRayVision project team and trained on the RSNA Pneumonia Detection Challenge dataset.

**Our engineering contributions in PneumoVision comprise**:
1. Integration and local encapsulation via our custom interface (`src/models/pneumonia_model.py`).
2. Dual-modality input pipeline supporting raw medical DICOM (`.dcm`) and standard images (`.png`, `.jpg`).
3. Correct medical grayscale processing (`MONOCHROME1` dynamic range inversion, windowing, and XRV normalization).
4. Grad-CAM visual interpretability pipeline targeting DenseNet-121's final feature extractor layer (`features.denseblock4.denselayer16.conv2`).
5. Unified Streamlit web diagnostic application (`app/app.py`).
6. End-to-end evaluation, testing, and comprehensive technical documentation.
