"""Semantic segmentation inference with optional augmentation.

Supports multiple models and runs on CUDA or MPS (Apple Silicon).
"""

from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_mmcv_ops_stub():
    """Inject stub mmcv.ops so mmseg can load without compiled mmcv._ext (e.g. on Mac)."""
    if "mmcv.ops" in sys.modules:
        return
    import types
    ops = types.ModuleType("mmcv.ops")

    def point_sample(input: torch.Tensor, point_coords: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
        # input (B, C, H, W), point_coords (B, N, 2) in [0,1]; return (B, C, N). Pure PyTorch fallback.
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0).expand(input.size(0), -1, -1)
        B, N, _ = point_coords.shape
        grid = point_coords.view(B, 1, N, 2) * 2.0 - 1.0  # [0,1] -> [-1,1] for grid_sample
        out = torch.nn.functional.grid_sample(
            input, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
        )
        return out.squeeze(2)

    def sigmoid_focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        # Focal loss (PyTorch fallback when mmcv._ext not available). pred/target (N, C).
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(gamma)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
        if weight is not None:
            loss = loss * weight
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        return loss.sum()

    class CrissCrossAttention(torch.nn.Module):
        """Stub for mmcv.ops.CrissCrossAttention (identity when _ext not available)."""

        def __init__(self, in_channels: int):
            super().__init__()
            self.in_channels = in_channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    ops.point_sample = point_sample
    ops.sigmoid_focal_loss = sigmoid_focal_loss
    ops.CrissCrossAttention = CrissCrossAttention
    sys.modules["mmcv.ops"] = ops


_ensure_mmcv_ops_stub()


def get_device(prefer: Optional[str] = None) -> str:
    """Return best available device: cuda, mps, or cpu."""
    if prefer:
        if prefer == 'cuda' and torch.cuda.is_available():
            return 'cuda:0'
        if prefer == 'mps':
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                return 'mps'
            if torch.cuda.is_available():
                return 'cuda:0'
        return prefer
    if torch.cuda.is_available():
        return 'cuda:0'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _load_mmseg_model(config_path: str, checkpoint_path: str, device: str):
    """Load mmseg model from config and checkpoint."""
    from mmseg.apis import init_model
    return init_model(config_path, checkpoint_path, device=device)


def _run_mmseg_inference(model: Any, image: np.ndarray, device: str) -> np.ndarray:
    """Run mmseg model on image (H,W,3) BGR or RGB; return pred mask (H,W) int."""
    from mmseg.apis import inference_model as mmseg_inference

    result = mmseg_inference(model, image)
    pred = result.pred_sem_seg.data
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    else:
        pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    return pred.astype(np.int32)


def _device_for_inference(device: str) -> str:
    """Use CPU when MPS is requested to avoid 'Adaptive pool MPS: input sizes must be divisible by output sizes' (e.g. PSPNet)."""
    if device == 'mps':
        import warnings
        warnings.warn(
            'Using CPU for inference instead of MPS to avoid adaptive pooling limitation '
            '(see https://github.com/pytorch/pytorch/issues/96056). Set --device cpu to silence.'
        )
        return 'cpu'
    return device


class Segmentor:
    """Single segmentation model wrapper (mmseg)."""

    def __init__(
        self,
        name: str,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.name = name
        self.device = device or get_device()
        self.device = _device_for_inference(self.device)
        self.model = _load_mmseg_model(config_path, checkpoint_path, self.device)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return _run_mmseg_inference(self.model, image, self.device)


def run_inference(
    image: np.ndarray,
    segmentor: Segmentor,
    augmentation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Run one model on image, optionally after augmentation. Returns (H,W) int pred."""
    if augmentation is not None:
        image = augmentation(image)
    return segmentor(image)


def run_inference_multi(
    image: np.ndarray,
    segmentors: List[Segmentor],
    augmentations: Optional[List[Tuple[Optional[str], Optional[Callable[[np.ndarray], np.ndarray]]]]] = None,
) -> List[Tuple[str, Optional[str], np.ndarray]]:
    """
    Run multiple models and optionally multiple augmentation conditions.
    augmentations: list of (aug_name, aug_fn). Use (None, None) for clean. If None, only clean.
    Returns list of (model_name, aug_name_or_None, pred_mask).
    """
    if augmentations is None:
        augmentations = [(None, None)]
    results = []
    for seg in segmentors:
        for aug_name, aug_fn in augmentations:
            img = aug_fn(image) if aug_fn is not None else image
            pred = seg(img)
            results.append((seg.name, aug_name, pred))
    return results
