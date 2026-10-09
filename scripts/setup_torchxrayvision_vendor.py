"""Vendor official TorchXRayVision library files into third_party/torchxrayvision/
and download the official densenet121-res224-rsna.pt pretrained weights.

Official Repository: https://github.com/mlmed/torchxrayvision
Official Weights: https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
Mirror Weights: https://huggingface.co/torchxrayvision/densenet121-res224-rsna/resolve/main/model.pt
License: Apache License 2.0
"""

import os
import shutil
import sys
import time
from pathlib import Path
import urllib.request
from tqdm import tqdm

RAW_BASE = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master"
THIRD_PARTY_DIR = Path("third_party/torchxrayvision")
CHECKPOINT_DIR = Path("checkpoints")
WEIGHTS_DEST = CHECKPOINT_DIR / "densenet121-res224-rsna.pt"

WEIGHTS_URLS = [
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt",
    "https://huggingface.co/torchxrayvision/densenet121-res224-rsna/resolve/main/model.pt",
]

CORE_FILES = [
    ("LICENSE", "LICENSE"),
    ("README.md", "README.md"),
    ("torchxrayvision/__init__.py", "torchxrayvision/__init__.py"),
    ("torchxrayvision/_version.py", "torchxrayvision/_version.py"),
    ("torchxrayvision/models.py", "torchxrayvision/models.py"),
    ("torchxrayvision/utils.py", "torchxrayvision/utils.py"),
    ("torchxrayvision/datasets.py", "torchxrayvision/datasets.py"),
]


def download_file(url: str, dest: Path, max_retries: int = 4):
    """Download a file with retries and progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                total_size = int(resp.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as pbar:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

            if temp_path.exists():
                if dest.exists():
                    dest.unlink()
                temp_path.rename(dest)
            return True
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Failed to download {url}: {e}")


def vendor_source_files():
    print("=" * 70)
    print("1. VENDORING OFFICIAL TORCHXRAYVISION SOURCE CODE")
    print("=" * 70)
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)

    for rel_src, rel_dest in CORE_FILES:
        url = f"{RAW_BASE}/{rel_src}"
        dest_file = THIRD_PARTY_DIR / rel_dest
        if not dest_file.exists():
            print(f"[*] Downloading {rel_dest}...")
            download_file(url, dest_file)
        print(f"    [OK] {rel_dest} ({dest_file.stat().st_size:,} bytes)")


def download_rsna_weights():
    print("\n" + "=" * 70)
    print("2. DOWNLOADING OFFICIAL DENSENET-121 RSNA PRETRAINED WEIGHTS")
    print("=" * 70)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if WEIGHTS_DEST.exists() and WEIGHTS_DEST.stat().st_size > 1024 * 1024 * 20:
        print(f"[*] Pretrained weights already exist: {WEIGHTS_DEST.resolve()} ({WEIGHTS_DEST.stat().st_size / (1024*1024):.2f} MB)")
        return

    download_success = False
    for url in WEIGHTS_URLS:
        try:
            print(f"[*] Attempting download from: {url}")
            download_file(url, WEIGHTS_DEST)
            print(f"[+] Download successful: {WEIGHTS_DEST.resolve()} ({WEIGHTS_DEST.stat().st_size / (1024*1024):.2f} MB)")
            download_success = True
            break
        except Exception as e:
            print(f"[!] Warning: Failed from {url}: {e}. Trying next mirror...")

    if not download_success:
        raise RuntimeError("Failed to download densenet121-res224-rsna.pt from all mirrors.")


def create_third_party_notices():
    notices_file = Path("THIRD_PARTY_NOTICES.md")
    content = """# Third-Party Software & Model Notices

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
"""
    notices_file.write_text(content, encoding="utf-8")
    print(f"[+] Written third-party notices to: {notices_file.resolve()}")


def main():
    vendor_source_files()
    download_rsna_weights()
    create_third_party_notices()
    print("\n[+] TorchXRayVision vendoring and weights download completed successfully!")


if __name__ == "__main__":
    main()
