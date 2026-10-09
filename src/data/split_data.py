"""Patient-level stratified train/val/test split for RSNA Pneumonia Detection.

Generates a reproducible 70 / 15 / 15 split at the *patient* level (never
at the annotation-row level). Stratification preserves the
~22.5 % / 77.5 % Pneumonia / No-Pneumonia class ratio in all three splits.

IMPORTANT — test-set lock policy
---------------------------------
Once created, the three split CSV files must NEVER be regenerated unless all
model results are invalidated and the project is restarted from scratch.
Regeneration requires the explicit ``--overwrite`` flag, which also prints a
prominent warning. The test split must remain unseen until the very final
evaluation step.

EXPERIMENTAL DESIGN NOTE
--------------------------
Primary model:   torchvision DenseNet-121 (ImageNet pretrained)
                 → independently evaluated; test-set metrics are the primary
                   reported result.
Reference model: TorchXRayVision densenet121-res224-rsna
                 → non-independent reference only; its backbone was pretrained
                   on the full RSNA dataset (all 26,684 patients), which
                   overlaps with every split produced here. XRV results must
                   be reported separately with an explicit disclosure note and
                   must not be compared directly to the primary test-set AUC.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = Path("dataset/stage_2_train_labels.csv")
SPLITS_DIR = Path("dataset/splits")

RANDOM_SEED: int = 42   # immutable — changing this invalidates all results
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15   # = 1.0 - TRAIN_RATIO - VAL_RATIO

# Proportion of the (train + val) pool that becomes the validation set
# i.e. val / (train + val) = 0.15 / 0.85 ≈ 0.17647
_VAL_OF_TRAINVAL: float = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def create_splits(
    csv_path: Path = CSV_PATH,
    splits_dir: Path = SPLITS_DIR,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> dict:
    """Generate and persist patient-level stratified 70 / 15 / 15 splits.

    Args:
        csv_path:   Path to ``stage_2_train_labels.csv``.
        splits_dir: Output directory for split CSV files and metadata.
        seed:       Random seed for reproducibility (default: 42).
        overwrite:  If ``True``, regenerate existing splits.
                    Defaults to ``False`` to prevent accidental re-randomisation.

    Returns:
        dict with keys ``'train'``, ``'val'``, ``'test'`` (DataFrames) and
        ``'metadata'`` (the dict written to ``split_metadata.json``).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        RuntimeError:      If splits already exist and ``overwrite=False``.
    """
    csv_path = Path(csv_path)
    splits_dir = Path(splits_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path.resolve()}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Guard against accidental re-randomisation
    sentinel_files = [splits_dir / n for n in ("train.csv", "val.csv", "test.csv")]
    if any(p.exists() for p in sentinel_files) and not overwrite:
        raise RuntimeError(
            f"\nSplit files already exist in: {splits_dir.resolve()}\n"
            "Regenerating the split would invalidate all existing results.\n"
            "Pass --overwrite only if you are intentionally restarting the project."
        )
    if overwrite:
        print("[!] WARNING: --overwrite is set. Existing splits will be replaced.")
        print("[!] All previously computed model results are now invalidated.")

    print("=" * 70)
    print("RSNA PNEUMONIA DETECTION — PATIENT-LEVEL DATASET SPLITTING")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load and aggregate to patient level
    # -----------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df):,} annotation rows from: {csv_path.resolve()}")

    # Patient-level binary label: 1 if ANY annotation row for that patient has
    # Target=1. This is identical to the logic in analyze_dataset.py (patient_max).
    patient_df = (
        df.groupby("patientId")["Target"]
        .max()
        .reset_index()
        .rename(columns={"Target": "label"})
    )
    patient_df["label"] = patient_df["label"].astype(int)

    total = len(patient_df)
    n_pos = int(patient_df["label"].sum())
    n_neg = total - n_pos

    print(f"[*] Total unique patients : {total:,}")
    print(f"[*] Positive (Pneumonia)  : {n_pos:,}  ({n_pos / total * 100:.2f}%)")
    print(f"[*] Negative (No Pneumonia): {n_neg:,} ({n_neg / total * 100:.2f}%)")
    print(f"[*] Random seed           : {seed}")
    print(f"[*] Ratios                : Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")

    # -----------------------------------------------------------------------
    # 2. Two-stage stratified split
    # -----------------------------------------------------------------------
    # Stage A: carve off the test set (15 % of total)
    trainval_df, test_df = train_test_split(
        patient_df,
        test_size=TEST_RATIO,
        stratify=patient_df["label"],
        random_state=seed,
    )

    # Stage B: split the remainder into train (70 %) and val (15 %)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=_VAL_OF_TRAINVAL,
        stratify=trainval_df["label"],
        random_state=seed,
    )

    # Reset indices for clean CSV output
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 3. Persist split CSVs
    # -----------------------------------------------------------------------
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Build and save metadata
    # -----------------------------------------------------------------------
    def _stats(df: pd.DataFrame) -> dict:
        pos = int(df["label"].sum())
        neg = len(df) - pos
        return {
            "n_patients": len(df),
            "n_positive": pos,
            "n_negative": neg,
            "pct_positive": round(pos / len(df) * 100, 4),
            "pct_negative": round(neg / len(df) * 100, 4),
        }

    metadata = {
        "random_seed": seed,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "total_patients": total,
        "splits": {
            "train": _stats(train_df),
            "val": _stats(val_df),
            "test": _stats(test_df),
        },
        "csv_source": str(csv_path.resolve()),
        "test_set_policy": (
            "TEST SET IS LOCKED. Do not use test.csv for any model selection, "
            "hyperparameter tuning, or exploratory analysis decision. "
            "The test set is used exactly once, at final evaluation."
        ),
        "xrv_disclosure": (
            "The TorchXRayVision densenet121-res224-rsna backbone was pretrained "
            "on the full RSNA Pneumonia Detection Challenge dataset (26,684 patients), "
            "which overlaps completely with all three splits produced here. "
            "XRV model evaluations are non-independent reference benchmarks and "
            "must not be compared directly to the primary ImageNet-pretrained model results."
        ),
    }

    with open(splits_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # -----------------------------------------------------------------------
    # 5. Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    header = f"{'Split':<10} {'Patients':>10} {'Positive':>10} {'Negative':>10} {'Pos%':>8}"
    print(header)
    print("-" * len(header))
    for name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = int(df_split["label"].sum())
        neg = len(df_split) - pos
        pct = pos / len(df_split) * 100
        print(f"{name:<10} {len(df_split):>10,} {pos:>10,} {neg:>10,} {pct:>7.2f}%")
    print("-" * len(header))
    total_check = len(train_df) + len(val_df) + len(test_df)
    print(f"{'TOTAL':<10} {total_check:>10,}")

    # Integrity check — no patient may appear in more than one split
    assert total_check == total, (
        f"Split total {total_check} != expected {total}. Data integrity error."
    )
    all_ids = set(train_df["patientId"]) | set(val_df["patientId"]) | set(test_df["patientId"])
    overlap_train_val = set(train_df["patientId"]) & set(val_df["patientId"])
    overlap_train_test = set(train_df["patientId"]) & set(test_df["patientId"])
    overlap_val_test = set(val_df["patientId"]) & set(test_df["patientId"])
    assert not overlap_train_val, f"Patient overlap between train and val: {overlap_train_val}"
    assert not overlap_train_test, f"Patient overlap between train and test: {overlap_train_test}"
    assert not overlap_val_test, f"Patient overlap between val and test: {overlap_val_test}"
    assert len(all_ids) == total, "Some patients are missing from all splits combined."

    print(f"\n[+] Leakage check passed — 0 patients appear in multiple splits.")
    print(f"[+] Splits saved to    : {splits_dir.resolve()}")
    print(f"[!] TEST SET LOCKED    : {(splits_dir / 'test.csv').resolve()}")
    print("=" * 70)

    return {"train": train_df, "val": val_df, "test": test_df, "metadata": metadata}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patient-level 70/15/15 train/val/test split for RSNA Pneumonia Detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "IMPORTANT: Run this script ONCE only. "
            "Regenerating the split invalidates all existing model results."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to stage_2_train_labels.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Output directory for split CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed (default: %(default)s). Do not change after first run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate splits even if they already exist. "
            "WARNING: this invalidates all existing model results."
        ),
    )
    args = parser.parse_args()
    create_splits(
        csv_path=args.csv,
        splits_dir=args.splits_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
