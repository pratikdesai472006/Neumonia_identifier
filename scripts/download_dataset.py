"""RSNA Pneumonia Detection Challenge - Dataset Downloader.

Downloads training DICOM images from Kaggle using the Kaggle CLI file endpoint.
Features:
- Extracts unique patient IDs from stage_2_train_labels.csv
- Downloads only stage_2_train_images (never stage_2_test_images)
- Resumes downloads seamlessly (skips verified existing DICOM files)
- Validates DICOM files with pydicom
- Retries transient errors with exponential backoff
- Supports multi-threaded parallel downloads
- Logs failed downloads to JSON for easy retry
- Supports test runs via --limit N
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydicom
from tqdm import tqdm

COMPETITION_NAME = "rsna-pneumonia-detection-challenge"
DEFAULT_CSV_PATH = "dataset/stage_2_train_labels.csv"
DEFAULT_DEST_DIR = "dataset/stage_2_train_images"
DEFAULT_FAILED_LOG = "results/failed_downloads.json"


def find_kaggle_executable() -> str:
    """Locate the kaggle executable in the current virtual environment or system PATH."""
    venv_kaggle = Path(sys.executable).parent / "kaggle.exe"
    if venv_kaggle.exists():
        return str(venv_kaggle.resolve())
    
    system_kaggle = shutil.which("kaggle")
    if system_kaggle:
        return system_kaggle
    
    raise FileNotFoundError(
        "Kaggle executable not found. Ensure 'kaggle' is installed in your Python environment."
    )


def is_valid_dicom(file_path: Path) -> bool:
    """Verify that a file exists, is non-empty, and is a valid readable DICOM."""
    if not file_path.exists() or file_path.stat().st_size < 1024:
        return False
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        return hasattr(dcm, "SOPInstanceUID") or hasattr(dcm, "PatientID")
    except Exception:
        return False


def download_single_dicom(
    patient_id: str,
    dest_dir: Path,
    kaggle_bin: str,
    max_retries: int = 3,
) -> Tuple[str, bool, str, int]:
    """Download a single DICOM file from Kaggle with retry logic.
    
    Returns:
        (patient_id, success, status_message, file_size_bytes)
    """
    target_file = dest_dir / f"{patient_id}.dcm"
    
    # Check if already present and valid
    if is_valid_dicom(target_file):
        return (patient_id, True, "SKIPPED_EXISTING", target_file.stat().st_size)

    # Kaggle CLI file path in competition archive
    kaggle_remote_file = f"stage_2_train_images/{patient_id}.dcm"
    cmd = [
        kaggle_bin,
        "competitions", "download",
        "-c", COMPETITION_NAME,
        "-f", kaggle_remote_file,
        "-p", str(dest_dir.resolve()),
        "--force",
    ]

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
            )

            # Check if Kaggle downloaded a zip wrapper for the single file
            possible_zip = dest_dir / f"{patient_id}.dcm.zip"
            if possible_zip.exists():
                shutil.unpack_archive(possible_zip, dest_dir)
                try:
                    possible_zip.unlink()
                except OSError:
                    pass

            if target_file.exists() and is_valid_dicom(target_file):
                return (patient_id, True, "DOWNLOADED", target_file.stat().st_size)
            else:
                last_error = result.stderr.strip() or result.stdout.strip() or "File invalid after download"

        except subprocess.TimeoutExpired:
            last_error = f"Timeout on attempt {attempt}"
        except Exception as ex:
            last_error = str(ex)

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    return (patient_id, False, f"FAILED: {last_error}", 0)


def run_download(
    csv_path: str = DEFAULT_CSV_PATH,
    dest_dir: str = DEFAULT_DEST_DIR,
    limit: Optional[int] = None,
    max_workers: int = 4,
    failed_log_path: str = DEFAULT_FAILED_LOG,
    retry_failed_only: bool = False,
) -> Dict:
    """Main orchestration function to download dataset DICOM files."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    kaggle_bin = find_kaggle_executable()

    print("=" * 70)
    print("RSNA PNEUMONIA DETECTION CHALLENGE - DATASET DOWNLOADER")
    print("=" * 70)
    print(f"Kaggle Executable: {kaggle_bin}")
    print(f"Destination Dir:   {dest_path.resolve()}")

    # Determine list of patient IDs
    if retry_failed_only and Path(failed_log_path).exists():
        with open(failed_log_path, "r", encoding="utf-8") as f:
            failed_data = json.load(f)
            patient_ids = [item["patient_id"] for item in failed_data]
            print(f"[*] Retrying {len(patient_ids)} previously failed downloads from {failed_log_path}")
    else:
        df = pd.read_csv(csv_path)
        patient_ids = df["patientId"].drop_duplicates().tolist()
        print(f"[*] Total unique patient IDs in labels CSV: {len(patient_ids):,}")

    if limit is not None and limit > 0:
        patient_ids = patient_ids[:limit]
        print(f"[*] Applying test limit: Downloading first {len(patient_ids)} files only.")

    # Storage Check
    total_b, used_b, free_b = shutil.disk_usage(dest_path.anchor or "E:")
    print(f"[*] Free Disk Space on {dest_path.anchor}: {free_b / (1024**3):.2f} GB")

    results = {
        "total_requested": len(patient_ids),
        "downloaded": 0,
        "skipped_existing": 0,
        "failed": 0,
        "failed_list": [],
        "downloaded_files_details": [],
    }

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(download_single_dicom, pid, dest_path, kaggle_bin): pid
            for pid in patient_ids
        }

        with tqdm(total=len(patient_ids), desc="Downloading DICOMs", unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_id):
                pid = future_to_id[future]
                try:
                    patient_id, success, status, file_size = future.result()
                    if success:
                        if status == "SKIPPED_EXISTING":
                            results["skipped_existing"] += 1
                        else:
                            results["downloaded"] += 1
                        if len(results["downloaded_files_details"]) < 10:
                            results["downloaded_files_details"].append({
                                "patient_id": patient_id,
                                "file_path": str((dest_path / f"{patient_id}.dcm").resolve()),
                                "size_bytes": file_size,
                                "status": status,
                            })
                    else:
                        results["failed"] += 1
                        results["failed_list"].append({
                            "patient_id": patient_id,
                            "error": status,
                        })
                except Exception as exc:
                    results["failed"] += 1
                    results["failed_list"].append({
                        "patient_id": pid,
                        "error": str(exc),
                    })
                pbar.update(1)

    elapsed_time = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed_time, 2)

    # Save failed downloads if any
    failed_log_file = Path(failed_log_path)
    failed_log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_log_file, "w", encoding="utf-8") as f:
        json.dump(results["failed_list"], f, indent=4)

    # Final summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total Processed:    {results['total_requested']:,}")
    print(f"Newly Downloaded:   {results['downloaded']:,}")
    print(f"Skipped (Existing): {results['skipped_existing']:,}")
    print(f"Failed:             {results['failed']:,}")
    print(f"Elapsed Time:       {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    print(f"Failed Log:         {failed_log_file.resolve()}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Download RSNA Pneumonia Training DICOMs from Kaggle.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to stage_2_train_labels.csv")
    parser.add_argument("--dest", default=DEFAULT_DEST_DIR, help="Destination directory for DICOM images")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to download (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent download threads")
    parser.add_argument("--failed-log", default=DEFAULT_FAILED_LOG, help="Path to save failed downloads JSON")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only previously failed downloads")

    args = parser.parse_args()
    run_download(
        csv_path=args.csv,
        dest_dir=args.dest,
        limit=args.limit,
        max_workers=args.workers,
        failed_log_path=args.failed_log,
        retry_failed_only=args.retry_failed,
    )


if __name__ == "__main__":
    main()
