"""DICOM Processing and Utility Functions for RSNA Chest Radiographs.

Handles:
- Loading DICOM files using pydicom
- Photometric Interpretation correction (inverting MONOCHROME1 to match MONOCHROME2)
- Rescale slope and intercept processing
- Robust min-max windowing and dynamic range normalization
- Conversion to uint8 NumPy array, PIL Image, and PyTorch FloatTensor
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pydicom
from PIL import Image
import torch


def read_dicom_raw(dicom_source: Union[str, Path, bytes, pydicom.dataset.FileDataset]) -> Tuple[np.ndarray, dict]:
    """Read a DICOM file from file path, raw bytes, or existing pydicom dataset.
    
    Returns:
        (pixel_array as float32, metadata_dict)
    """
    if isinstance(dicom_source, pydicom.dataset.FileDataset):
        dcm = dicom_source
    elif isinstance(dicom_source, (str, Path)):
        path = Path(dicom_source)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path.resolve()}")
        dcm = pydicom.dcmread(str(path))
    elif isinstance(dicom_source, bytes):
        import io
        dcm = pydicom.dcmread(io.BytesIO(dicom_source))
    else:
        raise TypeError(f"Unsupported dicom_source type: {type(dicom_source)}")

    metadata = {
        "PatientID": getattr(dcm, "PatientID", "UNKNOWN"),
        "PatientAge": getattr(dcm, "PatientAge", "N/A"),
        "PatientSex": getattr(dcm, "PatientSex", "N/A"),
        "ViewPosition": getattr(dcm, "ViewPosition", "N/A"),
        "Modality": getattr(dcm, "Modality", "CR/DX"),
        "PhotometricInterpretation": getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"),
        "Rows": int(getattr(dcm, "Rows", 1024)),
        "Columns": int(getattr(dcm, "Columns", 1024)),
    }

    # Extract pixel array
    pixel_array = dcm.pixel_array.astype(np.float32)

    # 1. Apply Rescale Slope and Intercept if present in DICOM tags
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # 2. Correct MONOCHROME1 -> MONOCHROME2
    # In MONOCHROME1: 0 is white (dense) and max is dark (air).
    # In MONOCHROME2: 0 is black (air) and max is white (bone/consolidation).
    # To standardize for CV backbones, invert MONOCHROME1 so it matches MONOCHROME2.
    if metadata["PhotometricInterpretation"] == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    return pixel_array, metadata


def normalize_pixel_array(pixel_array: np.ndarray) -> np.ndarray:
    """Safely normalize a 2D float pixel array to [0.0, 1.0].
    
    Guards against division-by-zero on uniform or corrupted arrays.
    """
    p_min = np.min(pixel_array)
    p_max = np.max(pixel_array)
    if p_max - p_min > 1e-6:
        normalized = (pixel_array - p_min) / (p_max - p_min)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.float32)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def dicom_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    """Normalize pixel array and convert to standard 8-bit unsigned integer [0, 255]."""
    norm = normalize_pixel_array(pixel_array)
    return (norm * 255.0).round().astype(np.uint8)


def dicom_to_pil(dicom_source: Union[str, Path, bytes, pydicom.dataset.FileDataset]) -> Tuple[Image.Image, dict]:
    """Convert DICOM source directly to a standard 8-bit RGB PIL Image.
    
    Returns:
        (PIL Image in RGB mode, metadata_dict)
    """
    pixel_array, metadata = read_dicom_raw(dicom_source)
    uint8_img = dicom_to_uint8(pixel_array)
    pil_img = Image.fromarray(uint8_img, mode="L").convert("RGB")
    return pil_img, metadata


def pad_to_square_np(
    arr: np.ndarray,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad a 2D or 3D numpy array symmetrically to square (S, S) where S = max(H, W).
    
    Preserves anatomical aspect ratio without distortion or cropping.
    
    Returns:
        (padded_array, (pad_top, pad_bottom, pad_left, pad_right))
    """
    h, w = arr.shape[:2]
    if h == w:
        return arr, (0, 0, 0, 0)

    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if arr.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif arr.ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    padded = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def pad_to_square_pil(
    pil_img: Image.Image,
    bg_color: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Pad a PIL Image symmetrically to square of size max(W, H) x max(W, H).
    
    Preserves anatomical aspect ratio without distortion or cropping.
    
    Returns:
        (padded_pil_image, (pad_top, pad_bottom, pad_left, pad_right))
    """
    w, h = pil_img.size
    if w == h:
        return pil_img, (0, 0, 0, 0)

    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top

    if pil_img.mode == "RGB" and isinstance(bg_color, int):
        bg_color = (bg_color, bg_color, bg_color)

    new_img = Image.new(pil_img.mode, (max_dim, max_dim), bg_color)
    new_img.paste(pil_img, (pad_left, pad_top))
    return new_img, (pad_top, pad_bottom, pad_left, pad_right)


def pad_to_square_tensor(
    x: torch.Tensor,
    pad_value: float = -1024.0,
) -> torch.Tensor:
    """Pad a PyTorch tensor (..., H, W) symmetrically to square (..., S, S).
    
    Preserves anatomical aspect ratio without distortion or cropping.
    """
    h, w = x.shape[-2], x.shape[-1]
    if h == w:
        return x

    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad expects (left, right, top, bottom)
    return torch.nn.functional.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
    )


def dicom_to_tensor(
    dicom_source: Union[str, Path, bytes, pydicom.dataset.FileDataset],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[torch.Tensor, dict]:
    """Convert DICOM directly to a preprocessed PyTorch FloatTensor suitable for CNN input.
    
    Returns:
        (torch.Tensor shape (1, 3, H, W) in [0.0, 1.0], metadata_dict)
    """
    pil_img, metadata = dicom_to_pil(dicom_source)
    pil_img, _ = pad_to_square_pil(pil_img, bg_color=0)
    if target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor, metadata

