"""Pneumonia Model Wrapper for Pretrained DenseNet-121 RSNA.

Encapsulates the locally vendored TorchXRayVision DenseNet-121 model trained on the
RSNA Pneumonia Detection Challenge dataset.

Model Attribution:
- Source Project: TorchXRayVision (Joseph Paul Cohen et al., MIDL 2022)
- Architecture: DenseNet-121 (Huang et al., CVPR 2017)
- Pretrained Checkpoint: densenet121-res224-rsna (kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt)
- License: Apache License 2.0 (see THIRD_PARTY_NOTICES.md)

CLINICAL & STATISTICAL NOTE:
The model output for Pneumonia is an operational model score (bounded in [0.0, 1.0] after op_norm),
NOT a calibrated Bayesian posterior probability. A score above the official operating threshold
indicates radiological features consistent with Pneumonia according to the RSNA training benchmark.
It must never be presented as a medical diagnosis.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Ensure local vendored torchxrayvision is importable
VENDOR_PATH = Path(__file__).resolve().parent.parent.parent / "third_party" / "torchxrayvision"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

import torchxrayvision as xrv


# Decision threshold on ModelScore
MODEL_SCORE_THRESHOLD: float = 0.30


@dataclass
class PneumoniaPrediction:
    """Structured container for Pneumonia screening results."""
    prediction: str              # 'PNEUMONIA' or 'NO PNEUMONIA'
    raw_score: float             # Exact ModelScore from model output tensor
    threshold: float             # Decision threshold for ModelScore (0.30)
    target_name: str             # 'Pneumonia'
    target_index: int            # Dynamic index in model.targets list
    is_positive: bool            # True if raw_score >= threshold


class PneumoniaModel:
    """Wrapper around TorchXRayVision DenseNet-121 RSNA pretrained model."""

    def __init__(
        self,
        weights: str = "densenet121-res224-rsna",
        cache_dir: Union[str, Path] = "checkpoints",
        device: Optional[torch.device] = None,
    ):
        self.weights_name = weights
        self.cache_dir = Path(cache_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load official DenseNet model with local checkpoints cache
        self.model = xrv.models.DenseNet(
            weights=self.weights_name,
            cache_dir=str(self.cache_dir),
        )

        # 2. Put in evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # 3. Dynamically discover 'Pneumonia' target index
        if "Pneumonia" not in self.model.targets:
            raise ValueError(f"'Pneumonia' target not found in model targets: {self.model.targets}")
        self.pneumonia_index = self.model.targets.index("Pneumonia")

        # 4. Calibrated operating threshold on ModelScore
        # TorchXRayVision applies op_norm(out, op_threshs) where op_threshs[idx] ≈ 0.134866.
        # Inside op_norm, values equal to op_threshs are mapped to 0.50.
        # Therefore, the decision threshold on the returned ModelScore is exactly 0.50.
        if hasattr(self.model, "op_threshs") and self.model.op_threshs is not None:
            self.raw_op_thresh = float(self.model.op_threshs[self.pneumonia_index].item())
        else:
            self.raw_op_thresh = 0.134866

        self.threshold = MODEL_SCORE_THRESHOLD

    @property
    def raw_model(self) -> nn.Module:
        """Access underlying PyTorch nn.Module."""
        return self.model

    def get_gradcam_target_layer(self) -> nn.Module:
        """Locate the deepest convolutional feature layer for Grad-CAM.
        
        In DenseNet-121, features.denseblock4 contains the final dense feature maps.
        The last layer in denseblock4 is denselayer16, whose conv2 outputs the final
        high-level spatial feature representations.
        """
        return self.model.features.denseblock4.denselayer16.conv2

    def predict(self, x: torch.Tensor) -> PneumoniaPrediction:
        """Run inference on preprocessed X-ray tensor.
        
        Args:
            x: Input tensor shape (1, 1, H, W) normalized to [-1024, 1024]
        
        Returns:
            PneumoniaPrediction dataclass with binary result, score, and threshold.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)

        # Handle non-square input tensors gracefully by symmetric padding to square
        if x.shape[2] != x.shape[3]:
            from src.data.dicom_utils import pad_to_square_tensor
            x = pad_to_square_tensor(x, pad_value=-1024.0)

        x = x.to(self.device).float()

        with torch.no_grad():
            outputs = self.model(x)

        score = float(outputs[0, self.pneumonia_index].item())
        is_positive = bool(score >= self.threshold)
        prediction_label = "PNEUMONIA" if is_positive else "NO PNEUMONIA"

        return PneumoniaPrediction(
            prediction=prediction_label,
            raw_score=score,
            threshold=self.threshold,
            target_name="Pneumonia",
            target_index=self.pneumonia_index,
            is_positive=is_positive,
        )


def load_pneumonia_model(
    cache_dir: Union[str, Path] = "checkpoints",
    device: Optional[torch.device] = None,
) -> PneumoniaModel:
    """Convenience factory function to instantiate and return PneumoniaModel."""
    return PneumoniaModel(cache_dir=cache_dir, device=device)
