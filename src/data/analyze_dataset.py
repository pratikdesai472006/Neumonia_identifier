"""Dataset Analysis Script for RSNA Pneumonia Detection Challenge.

This script performs Phase 1 data exploration:
- Analyzes annotation-level vs image-level labels
- Checks for duplicate/multiple bounding box annotations per patient
- Computes class balance and patient counts
- Checks disk capacity and estimates storage requirements
- Generates reproducible summaries and visualization figures in results/
"""

import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_dataset(
    csv_path: str = "dataset/stage_2_train_labels.csv",
    output_dir: str = "results",
    drive_path: str = "E:",
) -> dict:
    """Perform comprehensive, reproducible exploratory data analysis on the RSNA CSV."""
    csv_file = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_file.resolve()}")

    print(f"[*] Reading dataset annotations from: {csv_file.resolve()}")
    df = pd.read_csv(csv_file)

    # 1. Annotation-level metrics
    total_annotation_rows = len(df)
    columns = list(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    row_target_counts = df["Target"].value_counts().to_dict()

    # 2. Patient / Image-level aggregation
    unique_patients = int(df["patientId"].nunique())

    # Check for consistency of targets per patientId
    patient_grp = df.groupby("patientId")["Target"]
    patient_min = patient_grp.min()
    patient_max = patient_grp.max()
    has_conflicting_targets = bool((patient_min != patient_max).any())

    # Image-level ground truth: 1 if ANY row is 1, else 0
    image_level_target = patient_max
    positive_images = int((image_level_target == 1).sum())
    negative_images = int((image_level_target == 0).sum())
    pos_percentage = (positive_images / unique_patients) * 100
    neg_percentage = (negative_images / unique_patients) * 100
    imbalance_ratio = negative_images / positive_images if positive_images > 0 else 0

    # 3. Duplicate and Multiple Annotation Analysis
    exact_duplicate_rows = int(df.duplicated().sum())

    # Bounding boxes per patient (overall)
    boxes_per_patient_all = df.groupby("patientId").size().value_counts().to_dict()
    # Sort keys
    boxes_per_patient_all = {int(k): int(v) for k, v in sorted(boxes_per_patient_all.items())}

    # Bounding boxes for positive patients
    pos_df = df[df["Target"] == 1].copy()
    boxes_per_pos_patient = pos_df.groupby("patientId").size().value_counts().to_dict()
    boxes_per_pos_patient = {int(k): int(v) for k, v in sorted(boxes_per_pos_patient.items())}

    # Bounding box geometry stats (for positive cases)
    pos_df["area"] = pos_df["width"] * pos_df["height"]
    pos_df["aspect_ratio"] = pos_df["width"] / pos_df["height"]

    bbox_stats = {
        "total_bounding_boxes": len(pos_df),
        "mean_width": float(pos_df["width"].mean()),
        "std_width": float(pos_df["width"].std()),
        "min_width": float(pos_df["width"].min()),
        "max_width": float(pos_df["width"].max()),
        "mean_height": float(pos_df["height"].mean()),
        "std_height": float(pos_df["height"].std()),
        "min_height": float(pos_df["height"].min()),
        "max_height": float(pos_df["height"].max()),
        "mean_area": float(pos_df["area"].mean()),
        "median_area": float(pos_df["area"].median()),
    }

    # 4. Storage and Disk Space Analysis
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(drive_path)
    total_gb = total_bytes / (1024**3)
    used_gb = used_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)

    # RSNA DICOM images are ~150 KB on average (typical range 100 KB - 200 KB)
    avg_dicom_size_kb = 150.0
    est_raw_dicom_mb = (unique_patients * avg_dicom_size_kb) / 1024.0
    est_raw_dicom_gb = est_raw_dicom_mb / 1024.0

    analysis_results = {
        "dataset_csv": str(csv_file.resolve()),
        "total_annotation_rows": total_annotation_rows,
        "columns": columns,
        "missing_values_per_column": missing_counts,
        "row_level_distribution": {
            "negative_rows (Target=0)": row_target_counts.get(0, 0),
            "positive_rows (Target=1)": row_target_counts.get(1, 0),
        },
        "image_level_distribution": {
            "total_unique_images": unique_patients,
            "positive_images (Pneumonia)": positive_images,
            "negative_images (No Pneumonia)": negative_images,
            "positive_percentage": round(pos_percentage, 2),
            "negative_percentage": round(neg_percentage, 2),
            "class_imbalance_ratio (Negative : Positive)": f"{imbalance_ratio:.2f} : 1",
            "has_conflicting_targets_per_patient": has_conflicting_targets,
        },
        "annotation_multiplicity": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "boxes_per_patient_distribution_all": boxes_per_patient_all,
            "boxes_per_positive_patient_distribution": boxes_per_pos_patient,
            "bbox_geometry_stats": bbox_stats,
        },
        "storage_and_system": {
            "drive": drive_path,
            "drive_total_gb": round(total_gb, 2),
            "drive_used_gb": round(used_gb, 2),
            "drive_free_gb": round(free_gb, 2),
            "required_training_dicom_files": unique_patients,
            "estimated_avg_file_size_kb": avg_dicom_size_kb,
            "estimated_dataset_size_gb": round(est_raw_dicom_gb, 2),
            "disk_space_sufficient": free_gb > (est_raw_dicom_gb * 2),
        },
    }

    # Save JSON report
    json_path = out_dir / "dataset_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"[+] Saved structured analysis to: {json_path}")

    # Generate human-readable text report
    txt_path = out_dir / "dataset_analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RSNA PNEUMONIA DETECTION CHALLENGE - PHASE 1 DATASET ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset CSV: {csv_file.resolve()}\n")
        f.write(f"Total Annotation Rows: {total_annotation_rows:,}\n")
        f.write(f"Columns: {', '.join(columns)}\n\n")

        f.write("-" * 50 + "\n")
        f.write("1. IMAGE-LEVEL BINARY CLASSIFICATION METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Unique Patient / Image IDs: {unique_patients:,}\n")
        f.write(f"Positive Images (Target=1, Pneumonia):    {positive_images:,} ({pos_percentage:.2f}%)\n")
        f.write(f"Negative Images (Target=0, No Pneumonia): {negative_images:,} ({neg_percentage:.2f}%)\n")
        f.write(f"Class Imbalance Ratio (Neg : Pos):        {imbalance_ratio:.2f} : 1\n")
        f.write(f"Conflicting Target Rows per Patient:       {has_conflicting_targets}\n\n")

        f.write("-" * 50 + "\n")
        f.write("2. ANNOTATION MULTIPLICITY & BOUNDING BOX BEHAVIOR\n")
        f.write("-" * 50 + "\n")
        f.write(f"Exact Duplicate Rows in CSV: {exact_duplicate_rows}\n")
        f.write("Bounding boxes per positive patient:\n")
        for num_boxes, count in boxes_per_pos_patient.items():
            pct = (count / positive_images) * 100
            f.write(f"  - {num_boxes} box(es): {count:,} patients ({pct:.2f}%)\n")
        f.write(f"Total positive bounding box annotations: {bbox_stats['total_bounding_boxes']:,}\n")
        f.write(f"Average box size: width={bbox_stats['mean_width']:.1f}px, height={bbox_stats['mean_height']:.1f}px\n\n")

        f.write("-" * 50 + "\n")
        f.write("3. STORAGE & DISK CAPACITY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Target Drive: {drive_path}\n")
        f.write(f"Available Free Space on {drive_path}: {free_gb:.2f} GB\n")
        f.write(f"Required Training DICOM Files:  {unique_patients:,} files\n")
        f.write(f"Estimated Dataset Size:         ~{est_raw_dicom_gb:.2f} GB (~{est_raw_dicom_mb:.1f} MB)\n")
        f.write(f"Storage Status:                 {'SUFFICIENT (AMPLE SPACE)' if free_gb > est_raw_dicom_gb * 2 else 'LOW DISK SPACE'}\n")
        f.write("=" * 70 + "\n")

    print(f"[+] Saved text report to: {txt_path}")

    # Generate Visualizations
    generate_analysis_plots(analysis_results, pos_df, out_dir)

    return analysis_results


def generate_analysis_plots(results: dict, pos_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate visual exploratory charts for class distribution and bounding box stats."""
    # Plot 1: Class Distribution (Row level vs Image level)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Row level pie
    row_labels = ["Negative Rows (Target=0)", "Positive Rows (Target=1)"]
    row_sizes = [
        results["row_level_distribution"]["negative_rows (Target=0)"],
        results["row_level_distribution"]["positive_rows (Target=1)"],
    ]
    colors = ["#4A90E2", "#E94E77"]
    axes[0].pie(row_sizes, labels=row_labels, autopct="%1.1f%%", startangle=140, colors=colors, explode=(0, 0.05))
    axes[0].set_title(f"Row-Level Annotations (Total: {results['total_annotation_rows']:,})", fontsize=12, fontweight="bold")

    # Image level pie
    img_labels = ["No Pneumonia (Target=0)", "Pneumonia (Target=1)"]
    img_sizes = [
        results["image_level_distribution"]["negative_images (No Pneumonia)"],
        results["image_level_distribution"]["positive_images (Pneumonia)"],
    ]
    axes[1].pie(img_sizes, labels=img_labels, autopct="%1.1f%%", startangle=140, colors=colors, explode=(0, 0.05))
    axes[1].set_title(f"Unique Patient Images (Total: {results['image_level_distribution']['total_unique_images']:,})", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart_path1 = out_dir / "class_distribution.png"
    plt.savefig(chart_path1, dpi=300)
    plt.close()
    print(f"[+] Saved class distribution plot to: {chart_path1}")

    # Plot 2: Bounding Box Multiplicity & Box Dimension Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box counts per positive patient
    box_dist = results["annotation_multiplicity"]["boxes_per_positive_patient_distribution"]
    boxes = list(box_dist.keys())
    counts = list(box_dist.values())
    bars = axes[0].bar([f"{b} Box{'es' if b > 1 else ''}" for b in boxes], counts, color="#2ECC71", edgecolor="#27AE60")
    axes[0].set_title("Bounding Boxes per Positive Patient (Target=1)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Patient Count")
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", fontsize=10)

    # Box Width & Height histogram
    axes[1].hist(pos_df["width"], bins=30, alpha=0.6, label="Width", color="#3498DB", edgecolor="black")
    axes[1].hist(pos_df["height"], bins=30, alpha=0.6, label="Height", color="#E67E22", edgecolor="black")
    axes[1].set_title("Pneumonia Opacity Bounding Box Dimensions", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Pixels (Image Size: 1024x1024)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    chart_path2 = out_dir / "bounding_box_analysis.png"
    plt.savefig(chart_path2, dpi=300)
    plt.close()
    print(f"[+] Saved bounding box analysis plot to: {chart_path2}")


if __name__ == "__main__":
    analyze_dataset()
