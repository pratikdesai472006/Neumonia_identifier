"""PyTorch Dataset implementation for RSNA Pneumonia Detection.

Features:
- Transparently loads from either DICOM (.dcm) or converted PNG (.png) files
- Uses patient-level labels (0 = No Pneumonia / Normal, 1 = Pneumonia)
- Supports train/val/test split CSVs or dataset/raw/metadata.csv
- Configurable spatial resizing (default: 224x224) and ImageNet normalization
- Optional data augmentations for training/validation
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.dicom_utils import dicom_to_pil

# Standard ImageNet normalization parameters used by pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> transforms.Compose:
    """Get standardized PyTorch transforms for chest radiograph classification."""
    transform_list = [
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ]

    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    transform_list.extend([
        transforms.ToTensor(),  # Converts PIL [0, 255] to FloatTensor [0.0, 1.0]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


class RSNAPneumoniaDataset(Dataset):
    """PyTorch Dataset for RSNA Pneumonia Detection Challenge images."""

    def __init__(
        self,
        csv_path: Union[str, Path] = "dataset/splits/train.csv",
        dcm_dir: Union[str, Path] = "dataset/stage_2_train_images",
        png_dir: Union[str, Path] = "dataset/raw/images",
        prefer_png: bool = True,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.dcm_dir = Path(dcm_dir)
        self.png_dir = Path(png_dir)
        self.prefer_png = prefer_png
        self.target_size = target_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Annotation/Split CSV not found: {self.csv_path.resolve()}")

        df = pd.read_csv(self.csv_path)
        # Standardize column names
        if "label" not in df.columns and "Target" in df.columns:
            # Group by patientId if raw stage_2_train_labels.csv is passed
            df = df.groupby("patientId")["Target"].max().reset_index().rename(columns={"Target": "label"})

        if limit is not None and limit > 0:
            df = df.iloc[:limit].copy()

        self.df = df.reset_index(drop=True)
        self.transform = transform or get_default_transforms(target_size=target_size, is_training=False)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, patient_id: str) -> Image.Image:
        """Load image as RGB PIL Image, trying PNG first if requested, then fallback to DICOM."""
        png_file = self.png_dir / f"{patient_id}.png"
        dcm_file = self.dcm_dir / f"{patient_id}.dcm"

        if self.prefer_png and png_file.exists():
            return Image.open(png_file).convert("RGB")

        if dcm_file.exists():
            pil_img, _ = dicom_to_pil(dcm_file)
            return pil_img

        if png_file.exists():
            return Image.open(png_file).convert("RGB")

        raise FileNotFoundError(f"Neither PNG nor DICOM file found for patient: {patient_id}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        patient_id = str(row["patientId"])
        label = int(row["label"])

        pil_img = self._load_image(patient_id)
        tensor = self.transform(pil_img)

        return tensor, label, patient_id
