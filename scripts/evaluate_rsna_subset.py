"""Test 20 known-label RSNA test images and evaluate metrics.

Evaluates:
- 10 known Pneumonia (label=1) cases
- 10 known Normal / No Pneumonia (label=0) cases
From dataset/splits/test.csv and dataset/stage_2_train_images/.
"""

import sys
from pathlib import Path

# Add project root and vendored package to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "torchxrayvision"))

import numpy as np
import pandas as pd
import pydicom
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torchxrayvision as xrv
from src.models.pneumonia_model import PneumoniaModel


def main():
    print("=" * 80)
    print("STEP 2: TESTING 20 KNOWN-LABEL RSNA TEST IMAGES")
    print("=" * 80)

    # 1. Load test split CSV
    test_csv = ROOT_DIR / "dataset" / "splits" / "test.csv"
    dcm_dir = ROOT_DIR / "dataset" / "stage_2_train_images"

    if not test_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    print(f"Loaded test split: {len(df)} images.")

    # 2. Select 10 positive and 10 negative cases where DICOM exists on disk
    pos_candidates = df[df["label"] == 1]["patientId"].tolist()
    neg_candidates = df[df["label"] == 0]["patientId"].tolist()

    selected_pos = []
    for pid in pos_candidates:
        if (dcm_dir / f"{pid}.dcm").exists():
            selected_pos.append(pid)
            if len(selected_pos) == 10:
                break

    selected_neg = []
    for pid in neg_candidates:
        if (dcm_dir / f"{pid}.dcm").exists():
            selected_neg.append(pid)
            if len(selected_neg) == 10:
                break

    print(f"Selected {len(selected_pos)} positive cases and {len(selected_neg)} negative cases.")

    # 3. Load Model
    model = PneumoniaModel(weights="densenet121-res224-rsna", cache_dir=str(ROOT_DIR / "checkpoints"))
    op_thresh_orig = model.threshold  # 0.134866
    print(f"Model target index: {model.pneumonia_index}")
    print(f"Model configured threshold: {op_thresh_orig:.6f}")

    results = []

    all_cases = [(pid, 1) for pid in selected_pos] + [(pid, 0) for pid in selected_neg]

    for idx, (pid, true_label) in enumerate(all_cases, start=1):
        dcm_path = dcm_dir / f"{pid}.dcm"
        ds = pydicom.dcmread(str(dcm_path))

        photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        max_val = 2 ** getattr(ds, "BitsStored", 16) - 1
        data = ds.pixel_array.astype(np.float32)

        if photo_interp == "MONOCHROME1":
            data = max_val - data

        # Pad to square if needed
        if data.shape[0] != data.shape[1]:
            from src.data.dicom_utils import pad_to_square_np
            data, _ = pad_to_square_np(data, pad_value=float(data.min()))

        norm_data = xrv.utils.normalize(data, max_val)
        tensor = torch.from_numpy(norm_data[None, None, ...]).float()

        # Run model
        pred = model.predict(tensor)
        score = pred.raw_score

        # Also get raw un-op_norm sigmoid to understand calibration
        with torch.no_grad():
            t_in = tensor.to(model.device)
            # fix_resolution
            t_res = xrv.utils.fix_resolution(t_in, 224, model.raw_model)
            feat = model.raw_model.features2(t_res)
            logits = model.raw_model.classifier(feat)
            raw_sigmoid = float(torch.sigmoid(logits)[0, model.pneumonia_index].item())

        results.append({
            "index": idx,
            "patient_id": pid[:8] + "...",
            "full_id": pid,
            "true_label": true_label,
            "true_class": "Pneumonia" if true_label == 1 else "Normal",
            "model_score": score,
            "raw_sigmoid": raw_sigmoid,
            "pred_at_01349": 1 if score >= op_thresh_orig else 0,
            "pred_at_05000": 1 if score >= 0.50 else 0,
            "pred_sigmoid_at_01349": 1 if raw_sigmoid >= op_thresh_orig else 0,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 105)
    print(f"{'Idx':<4} {'PatientID':<14} {'Ground Truth':<14} {'ModelScore':<12} {'RawSigmoid':<12} {'Pred (Th=0.1349)':<18} {'Pred (Th=0.50)':<16}")
    print("-" * 105)
    for _, r in results_df.iterrows():
        p1 = "PNEUMONIA (FP)" if r["pred_at_01349"] == 1 and r["true_label"] == 0 else ("PNEUMONIA (TP)" if r["pred_at_01349"] == 1 else "NORMAL (TN)")
        p2 = "PNEUMONIA (FP)" if r["pred_at_05000"] == 1 and r["true_label"] == 0 else ("PNEUMONIA (TP)" if r["pred_at_05000"] == 1 else ("NORMAL (TN)" if r["pred_at_05000"] == 0 and r["true_label"] == 0 else "NORMAL (FN)"))
        print(f"{r['index']:<4} {r['patient_id']:<14} {r['true_class']:<14} {r['model_score']:<12.4f} {r['raw_sigmoid']:<12.4f} {p1:<18} {p2:<16}")

    # =========================================================================
    # STEP 3: EVALUATE METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: METRICS EVALUATION")
    print("=" * 80)

    y_true = results_df["true_label"].values

    def compute_metrics(y_pred, name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, zero_division=0)  # Recall
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "Setting": name,
            "Accuracy": f"{acc:.1%}",
            "Sensitivity (Recall)": f"{sens:.1%}",
            "Specificity": f"{spec:.1%}",
            "Precision": f"{prec:.1%}",
            "F1-Score": f"{f1:.3f}",
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        }

    m1 = compute_metrics(results_df["pred_at_01349"].values, "Current: ModelScore >= 0.134866")
    m2 = compute_metrics(results_df["pred_at_05000"].values, "Calibrated: ModelScore >= 0.500000 (op_norm cutoff)")
    m3 = compute_metrics(results_df["pred_sigmoid_at_01349"].values, "Raw Sigmoid: Sigmoid >= 0.134866")

    summary_df = pd.DataFrame([m1, m2, m3])
    print(summary_df.to_string(index=False))

    # Calculate ROC AUC
    auc_score = roc_auc_score(y_true, results_df["model_score"].values)
    print(f"\nROC AUC Score: {auc_score:.4f}")

    # Save results to disk
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "20_sample_test_results.csv", index=False)
    print(f"\nDetailed results saved to: {out_dir / '20_sample_test_results.csv'}")


if __name__ == "__main__":
    main()
