"""Grad-CAM (Gradient-weighted Class Activation Mapping) for DenseNet-121 RSNA.

Calculates visual interpretability heatmaps highlighting spatial regions
in chest radiographs that drive the model's Pneumonia prediction.

Methodology:
1. Registers forward hook on the final dense block (model.features.denseblock4)
   capturing all 1,024 high-level feature maps.
2. Registers backward hook capturing gradients of the target class score.
3. Computes global-average-pooled gradient weights across spatial dimensions.
4. Calculates the ReLU-rectified weighted combination of activation maps.
5. Bilinearly upsamples the heatmap to the original image dimensions and
   generates a color overlay.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM implementation for TorchXRayVision DenseNet-121."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_index: Optional[int] = None,
    ):
        self.model = model
        self.model.eval()

        # Target final dense feature block (denseblock4) by default
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            self.target_layer = model.features.denseblock4
        else:
            raise ValueError("Could not automatically locate target convolutional layer in model.")

        # Dynamically discover Pneumonia index if not specified
        if target_index is not None:
            self.target_index = target_index
        elif hasattr(model, "targets") and "Pneumonia" in model.targets:
            self.target_index = model.targets.index("Pneumonia")
        else:
            self.target_index = 8  # fallback verified RSNA index

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where the first element is the gradient tensor
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Cleanly unregister PyTorch hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_index: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a 2D float heatmap in [0.0, 1.0] for the Pneumonia output.
        
        Args:
            input_tensor: (1, 1, H, W) preprocessed tensor.
            target_index: Index of output class (defaults to Pneumonia target index).
        
        Returns:
            2D numpy array (H, W) normalized to [0.0, 1.0].
        """
        if target_index is None:
            target_index = self.target_index

        # Ensure tensor is 4D and squared for the underlying DenseNet backbone
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[2] != input_tensor.shape[3]:
            from src.data.dicom_utils import pad_to_square_tensor
            input_tensor = pad_to_square_tensor(input_tensor, pad_value=-1024.0)

        # Ensure tensor requires grad for backward propagation
        tensor_in = input_tensor.clone()
        if not tensor_in.requires_grad:
            tensor_in.requires_grad_(True)

        # Forward pass
        outputs = self.model(tensor_in)

        # Target score for Pneumonia output
        score = outputs[0, target_index]

        # Backward pass
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Compute importance weights via Global Average Pooling of gradients: shape (C, 1, 1)
        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)  # (C, 1, 1)
        activations = self.activations[0]                                   # (C, H_feat, W_feat)

        # Weighted combination: sum_c (w_c * A_c)
        cam = torch.sum(weights * activations, dim=0)                       # (H_feat, W_feat)

        # Apply ReLU: only positive evidence contributes to Pneumonia presence
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Normalize to [0.0, 1.0]
        c_min, c_max = np.min(cam_np), np.max(cam_np)
        if c_max - c_min > 1e-7:
            cam_norm = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_np, dtype=np.float32)

        return cam_norm

    def overlay_heatmap(
        self,
        original_image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[Image.Image, np.ndarray]:
        """Blend heatmap onto original image.
        
        Args:
            original_image: PIL Image or uint8 NumPy array (H, W) or (H, W, 3)
            heatmap: 2D float array in [0.0, 1.0]
            alpha: Heatmap blend opacity (0.0 = only original, 1.0 = only heatmap)
            colormap: OpenCV colormap enum
        
        Returns:
            (PIL.Image blended_overlay, 2D float resized_heatmap)
        """
        if isinstance(original_image, Image.Image):
            orig_rgb = np.array(original_image.convert("RGB"))
        elif isinstance(original_image, np.ndarray):
            if original_image.ndim == 2:
                orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 1:
                orig_rgb = cv2.cvtColor(original_image.squeeze(2), cv2.COLOR_GRAY2RGB)
            else:
                orig_rgb = original_image.copy()
        else:
            raise TypeError(f"Unsupported original_image type: {type(original_image)}")

        h_orig, w_orig = orig_rgb.shape[0], orig_rgb.shape[1]

        # Bilinear interpolation upsampling to match input dimensions
        resized_cam = cv2.resize(heatmap, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        resized_cam = np.clip(resized_cam, 0.0, 1.0)

        # Colorize
        cam_uint8 = np.uint8(255 * resized_cam)
        colored_bgr = cv2.applyColorMap(cam_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend
        blended = (1.0 - alpha) * orig_rgb.astype(np.float32) + alpha * colored_rgb.astype(np.float32)
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_uint8), resized_cam
