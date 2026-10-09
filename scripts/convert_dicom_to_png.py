"""DICOM to PNG Batch Converter & Metadata Generator for RSNA CXR.

Features:
- Reads DICOM via pydicom
- Automatically inverts MONOCHROME1 to MONOCHROME2
- Applies RescaleSlope and RescaleIntercept
- Performs robust min-max normalization to 8-bit unsigned integer (0-255)
- Saves lossless PNG images to dataset/raw/images/{patientId}.png
- Skips already converted valid PNGs for seamless resume
- Preserves original .dcm files without modification
- Builds dataset/raw/metadata.csv linking patientId, dicom_filename, png_filename, label, split
- Multi-threaded parallel processing for fast conversion
- Supports --limit N for fast test runs and progress tracking via tqdm
"""

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

# Ensure local src/ is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dicom_utils import dicom_to_uint8, read_dicom_raw

DEFAULT_DICOM_DIR = Path("dataset/stage_2_train_images")
DEFAULT_PNG_DIR = Path("dataset/raw/images")
DEFAULT_LABELS_CSV = Path("dataset/stage_2_train_labels.csv")
DEFAULT_SPLITS_DIR = Path("dataset/splits")
DEFAULT_METADATA_CSV = Path("dataset/raw/metadata.csv")


def convert_single_dicom(
    dcm_path: Path,
    png_dir: Path,
    patient_id: str,
) -> Tuple[str, bool, str, int]:
    """Convert a single DICOM file to PNG.
    
    Returns:
        (patient_id, success, message, file_size_bytes)
    """
    target_png = png_dir / f"{patient_id}.png"

    # Skip if already exists and is non-empty
    if target_png.exists() and target_png.stat().st_size > 1024:
        return (patient_id, True, "SKIPPED_EXISTING", target_png.stat().st_size)

    try:
        pixel_array, meta = read_dicom_raw(dcm_path)
        uint8_arr = dicom_to_uint8(pixel_array)

        # Save using OpenCV / Pillow
        pil_img = Image.fromarray(uint8_arr, mode="L")
        pil_img.save(target_png, format="PNG", optimize=False)

        return (patient_id, True, "CONVERTED", target_png.stat().st_size)
    except Exception as e:
        return (patient_id, False, f"ERROR: {str(e)}", 0)


def build_metadata(
    labels_csv_path: Path,
    splits_dir: Path,
    png_dir: Path,
    dcm_dir: Path,
    output_metadata_csv: Path,
) -> pd.DataFrame:
    """Generate master dataset/raw/metadata.csv mapping patientId, dicom, png, label, and split."""
    print(f"\n[*] Generating metadata mapping from: {labels_csv_path.resolve()}")
    df = pd.read_csv(labels_csv_path)

    # Patient-level binary label
    patient_labels = (
        df.groupby("patientId")["Target"]
        .max()
        .reset_index()
        .rename(columns={"Target": "label"})
    )
    patient_labels["label"] = patient_labels["label"].astype(int)

    # Load splits if available to preserve exact split mapping
    split_map = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.csv"
        if split_file.exists():
            s_df = pd.read_csv(split_file)
            for pid in s_df["patientId"]:
                split_map[pid] = split_name

    records = []
    for _, row in patient_labels.iterrows():
        pid = row["patientId"]
        dcm_name = f"{pid}.dcm"
        png_name = f"{pid}.png"
        dcm_exists = (dcm_dir / dcm_name).exists()
        png_exists = (png_dir / png_name).exists()
        split = split_map.get(pid, "unassigned")

        records.append({
            "patientId": pid,
            "dicom_filename": dcm_name,
            "png_filename": png_name,
            "label": row["label"],
            "split": split,
            "dicom_available": dcm_exists,
            "png_available": png_exists,
        })

    meta_df = pd.DataFrame(records)
    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(output_metadata_csv, index=False)
    print(f"[+] Metadata CSV successfully written: {output_metadata_csv.resolve()} ({len(meta_df):,} rows)")
    return meta_df


def run_conversion(
    dcm_dir: Path = DEFAULT_DICOM_DIR,
    png_dir: Path = DEFAULT_PNG_DIR,
    labels_csv: Path = DEFAULT_LABELS_CSV,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    limit: Optional[int] = None,
    num_workers: int = 8,
) -> dict:
    """Execute batch DICOM to PNG conversion."""
    dcm_dir = Path(dcm_dir)
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PNEUMOVISION — DICOM TO PNG CONVERTER & METADATA BUILDER")
    print("=" * 70)
    print(f"DICOM Source:   {dcm_dir.resolve()}")
    print(f"PNG Target:     {png_dir.resolve()}")
    print(f"Workers:        {num_workers}")

    # Gather patient list
    df_labels = pd.read_csv(labels_csv)
    unique_pids = df_labels["patientId"].drop_duplicates().tolist()
    print(f"[*] Total unique patients in annotations: {len(unique_pids):,}")

    if limit is not None and limit > 0:
        unique_pids = unique_pids[:limit]
        print(f"[*] Applying limit: Converting first {len(unique_pids):,} files only.")

    results = {
        "total_requested": len(unique_pids),
        "converted": 0,
        "skipped_existing": 0,
        "failed": 0,
        "failed_list": [],
    }

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_pid = {
            executor.submit(
                convert_single_dicom,
                dcm_dir / f"{pid}.dcm",
                png_dir,
                pid,
            ): pid
            for pid in unique_pids
        }

        with tqdm(total=len(unique_pids), desc="Converting to PNG", unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    patient_id, success, status, _ = future.result()
                    if success:
                        if status == "SKIPPED_EXISTING":
                            results["skipped_existing"] += 1
                        else:
                            results["converted"] += 1
                    else:
                        results["failed"] += 1
                        results["failed_list"].append({"patient_id": patient_id, "error": status})
                except Exception as exc:
                    results["failed"] += 1
                    results["failed_list"].append({"patient_id": pid, "error": str(exc)})
                pbar.update(1)

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)

    # Build and refresh metadata CSV
    meta_df = build_metadata(labels_csv, splits_dir, png_dir, dcm_dir, metadata_csv)

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total Processed:    {results['total_requested']:,}")
    print(f"Newly Converted:    {results['converted']:,}")
    print(f"Skipped (Existing): {results['skipped_existing']:,}")
    print(f"Failed:             {results['failed']:,}")
    print(f"Elapsed Time:       {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"PNGs in Target Dir: {len(list(png_dir.glob('*.png'))):,}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert RSNA DICOM chest radiographs to PNG.")
    parser.add_argument("--dcm-dir", default=str(DEFAULT_DICOM_DIR), help="Path to DICOM directory")
    parser.add_argument("--png-dir", default=str(DEFAULT_PNG_DIR), help="Path to output PNG directory")
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV), help="Path to stage_2_train_labels.csv")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR), help="Path to dataset/splits directory")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV), help="Output metadata CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to convert (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads")

    args = parser.parse_args()
    run_conversion(
        dcm_dir=Path(args.dcm_dir),
        png_dir=Path(args.png_dir),
        labels_csv=Path(args.labels_csv),
        splits_dir=Path(args.splits_dir),
        metadata_csv=Path(args.metadata_csv),
        limit=args.limit,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
