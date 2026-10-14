# PneumoVision: Standalone All-In-One Pneumonia Screening & Statistical Evaluation

This directory is a **self-contained, standalone bundle** containing the entire PneumoVision pneumonia screening and statistical pipeline in **both** an interactive Jupyter Notebook (`.ipynb`) and a single standalone Python script (`.py`).

---

## ⚡ Quickstart: Automated Dependency Installation

### 1-Command Terminal Installation
Install all required libraries in your active Python or virtual environment with a single command:
```bash
pip install -r ../requirements.txt
```
*(Or from the repository root: `pip install -r requirements.txt`)*

### 🤖 Automatic Pre-Run Verification & Auto-Installer
Both `pneumonia_complete_pipeline.ipynb` and `pneumonia_complete_pipeline.py` include an **automated dependency self-checker**:
1. When launched, the environment checks every required library (`torch`, `torchvision`, `pydicom`, `opencv-python`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `kagglehub`).
2. If any library is missing, it **automatically downloads and installs it via pip** before running!
3. Users and evaluators can clone and run without worrying about missing packages or manual setup steps.

---

## 📁 Directory Structure

```
pneumonia_all_in_one/
│
├── pneumonia_complete_pipeline.ipynb      # Complete All-In-One Jupyter Notebook
├── pneumonia_complete_pipeline.py         # Complete All-In-One Standalone Python Script
├── run_pipeline.bat                       # 1-Click Windows Batch Launcher
│
├── data/                                  # Datasets & Metadata
│   ├── stage_2_train_labels.csv           # Raw RSNA Challenge Dataset Labels
│   ├── train.csv                          # Train Split (70%, 18,678 images)
│   ├── val.csv                            # Validation Split (15%, 4,003 images)
│   ├── test.csv                           # Test Split (15%, 4,003 images)
│   ├── split_metadata.json                # Patient-level split isolation metadata
│   └── sample_images/                     # Pre-packaged test radiographs (DICOM & PNG)
│       ├── pneumonia/                     # Confirmed pneumonia cases
│       └── normal/                        # Confirmed healthy cases
│
├── checkpoints/                           # Model Weights Directory
│   ├── densenet121-res224-rsna.pt         # Pretrained DenseNet-121 Checkpoint
│   └── kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
│
├── third_party/                           # Vendored Dependencies
│   └── torchxrayvision/                   # Standalone TorchXRayVision library
│
└── results/                               # Output Plots & Tables
    ├── statistical_analysis_results.csv   # Comprehensive metrics across all thresholds
    ├── roc_and_pr_curves_all_in_one.png   # ROC & Precision-Recall curves
    └── ...
```

---

## 🚀 How to Run

### Option 1: Run the Interactive Jupyter Notebook (`.ipynb`)
Open `pneumonia_complete_pipeline.ipynb` in:
- **VS Code**: Open the notebook and select your Python kernel.
- **Jupyter Lab / Notebook**:
  ```bash
  jupyter notebook pneumonia_complete_pipeline.ipynb
  ```
- **Google Colab / Kaggle**: Upload the notebook and run Cell 1 to auto-install all dependencies.

### Option 2: Run the Standalone Python Script (`.py`)
Run the script directly in your terminal:
```bash
python pneumonia_all_in_one/pneumonia_complete_pipeline.py
```
Or from inside this directory:
```bash
python pneumonia_complete_pipeline.py
```

### Option 3: Double-Click Batch File (Windows)
Simply double-click `run_pipeline.bat`. It will automatically check dependencies and run the complete pipeline end-to-end.

---

## 📊 Pipeline Outputs

When executed, the pipeline performs:
1. **Raw RSNA Dataset Exploration**: Class counts, bounding box dimensions, and prevalence.
2. **Train/Val/Test Isolation Audit**: Confirms zero patient overlap across splits.
3. **DenseNet-121 Model Loading**: RSNA weights and feature reuse architecture.
4. **Multi-Threshold Evaluation**: Evaluates 12 clinical metrics across cutoffs (0.20, 0.30, 0.50).
5. **Diagnostic Visualizations**: Generates ROC curves, PR curves, and confusion matrices into `results/`.
6. **Grad-CAM Explainability**: Generates spatial attention heatmaps for sample X-rays.
