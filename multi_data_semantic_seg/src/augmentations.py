"""Optional image augmentations for semantic segmentation inference.

Provides slight and severe variants of:
  - contrast_shift (contrast/brightness)
  - motion_blur
"""

from typing import Callable, List, Tuple

import cv2
import numpy as np

# (alpha, beta) for contrast/brightness; (kernel_size,) for motion blur
AUGMENT_CONFIG = {
    'contrast_shift_slight': (0.85, -25),
    'contrast_shift_severe': (0.5, -80),
    'motion_blur_slight': (7,),
    'motion_blur_severe': (21,),
}


def apply_contrast_shift(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Adjust contrast (alpha) and brightness (beta)."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply horizontal motion blur."""
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def get_augmentation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that takes BGR or RGB image (H,W,3) and returns augmented image."""
    if name not in AUGMENT_CONFIG:
        raise KeyError(f"Unknown augmentation: {name}. Choose from {list(AUGMENT_CONFIG.keys())}")
    args = AUGMENT_CONFIG[name]
    if 'contrast_shift' in name:
        alpha, beta = args
        return lambda img: apply_contrast_shift(img, alpha, beta)
    if 'motion_blur' in name:
        (k,) = args
        return lambda img: apply_motion_blur(img, k)
    raise KeyError(name)


def list_augmentations() -> List[str]:
    return list(AUGMENT_CONFIG.keys())
