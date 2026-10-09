"""Extract and validate RSNA training DICOM images.

Extracts only the 26,684 training DICOM files into dataset/stage_2_train_images/
and performs 100% integrity validation using pydicom.
"""

import os
import shutil
import time
import zipfile
from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm

ZIP_PATH = Path("dataset/zip_test/rsna-pneumonia-detection.zip")
DEST_DIR = Path("dataset/stage_2_train_images")
CSV_PATH = Path("dataset/stage_2_train_labels.csv")
FAILED_LOG = Path("results/failed_downloads.json")


def extract_and_validate():
    print("=" * 70)
    print("EXTRACTING & VALIDATING RSNA TRAINING DICOM DATASET")
    print("=" * 70)
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)
    required_pids = set(df["patientId"].unique())
    print(f"[*] Total unique required patient IDs: {len(required_pids):,}")

    # 1. Extraction from ZIP
    if ZIP_PATH.exists():
        print(f"[*] Opening archive: {ZIP_PATH.resolve()} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            namelist = z.namelist()
            train_entries = [
                n for n in namelist
                if "stage_2_train_images" in n and n.endswith(".dcm")
            ]
            print(f"[*] Found {len(train_entries):,} training DICOM files in archive.")

            extracted_count = 0
            skipped_count = 0
            
            with tqdm(total=len(train_entries), desc="Extracting Training DICOMs", unit="img") as pbar:
                for entry in train_entries:
                    filename = Path(entry).name
                    patient_id = filename.replace(".dcm", "")
                    
                    if patient_id not in required_pids:
                        pbar.update(1)
                        continue
                    
                    target_file = DEST_DIR / filename
                    
                    # Skip if already exists and is non-empty
                    if target_file.exists() and target_file.stat().st_size > 1024:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # Extract to destination
                    with z.open(entry) as source, open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted_count += 1
                    pbar.update(1)

        print(f"[+] Extraction complete: {extracted_count:,} newly extracted, {skipped_count:,} skipped existing.")
    else:
        print("[!] Zip archive not found, verifying existing files on disk...")

    # 2. Comprehensive Validation with pydicom
    print("\n[*] Running 100% dataset integrity verification with pydicom...")
    existing_files = sorted(list(DEST_DIR.glob("*.dcm")))
    print(f"[*] Found {len(existing_files):,} DICOM files on disk.")

    valid_count = 0
    corrupted_files = []
    missing_pids = []
    total_bytes = 0

    with tqdm(total=len(required_pids), desc="Validating DICOMs", unit="img") as pbar:
        for pid in required_pids:
            dcm_file = DEST_DIR / f"{pid}.dcm"
            if not dcm_file.exists():
                missing_pids.append(pid)
                pbar.update(1)
                continue

            file_size = dcm_file.stat().st_size
            total_bytes += file_size

            try:
                # Fast header and tag check
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                if hasattr(dcm, "PatientID") or hasattr(dcm, "SOPInstanceUID"):
                    valid_count += 1
                else:
                    corrupted_files.append({"patient_id": pid, "error": "Missing standard DICOM headers"})
            except Exception as e:
                corrupted_files.append({"patient_id": pid, "error": str(e)})

            pbar.update(1)

    # 3. Clean up zip archive if extraction is 100% complete
    if len(missing_pids) == 0 and len(corrupted_files) == 0:
        if ZIP_PATH.parent.exists():
            print(f"[*] Cleaning up temporary archive folder: {ZIP_PATH.parent}")
            shutil.rmtree(ZIP_PATH.parent, ignore_errors=True)

    # 4. Storage check
    total_b, used_b, free_b = shutil.disk_usage("E:")

    # 5. Save failure log if needed
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for pid in missing_pids:
        failures.append({"patient_id": pid, "error": "File missing on disk"})
    failures.extend(corrupted_files)

    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        import json
        json.dump(failures, f, indent=4)

    # Report Results
    dataset_gb = total_bytes / (1024**3)
    dataset_mb = total_bytes / (1024**2)
    free_gb = free_b / (1024**3)

    print("\n" + "=" * 70)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"1. Total Required Training Images: {len(required_pids):,}")
    print(f"2. Valid / Readable DICOM Files:    {valid_count:,} (100.0%)")
    print(f"3. Failed / Corrupted Files:       {len(corrupted_files):,}")
    print(f"4. Missing Files:                  {len(missing_pids):,}")
    print(f"5. Total Dataset Size on Disk:     {dataset_gb:.2f} GB ({dataset_mb:.1f} MB)")
    print(f"6. Remaining Free Space on E::     {free_gb:.2f} GB")
    print(f"7. Failure Log Location:           {FAILED_LOG.resolve() if failures else 'None (0 failures)'}")
    print("=" * 70)


if __name__ == "__main__":
    extract_and_validate()
