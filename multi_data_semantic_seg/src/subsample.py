"""Subsample GT and prediction into two types for semantic segmentation.

Types (no instance distinction for semantic seg):
  - 1CnI: 1 class as COI — manually sample one class; filter model response to this class only.
  - nCnI: N classes — retain all classes (no filtering).
"""

from typing import List, Tuple

import numpy as np
from scipy import ndimage

# Type labels: 1C = one class COI, nC = multiple classes
INSTANCE_TYPE_1CNI = '1CnI'
INSTANCE_TYPE_NCNI = 'nCnI'


def _connected_components_per_class(mask: np.ndarray, ignore_index: int = 255) -> List[Tuple[int, int]]:
    """For each non-ignore class id, count connected components. Returns [(class_id, num_components), ...]."""
    out = []
    valid = (mask >= 0) & (mask != ignore_index)
    if not np.any(valid):
        return out
    unique_classes = np.unique(mask[valid])
    for c in unique_classes:
        if c == ignore_index or c < 0:
            continue
        binary = (mask == c).astype(np.uint8)
        labeled, num = ndimage.label(binary)
        out.append((int(c), int(num)))
    return out


def get_instance_type(
    gt_mask: np.ndarray,
    ignore_index: int = 255,
) -> str:
    """
    Classify sample by number of classes. For semantic seg there is no 1I/NI distinction.
    Returns 1CnI when exactly one class is present, else nCnI.
    """
    per_class = _connected_components_per_class(gt_mask, ignore_index)
    if not per_class:
        return INSTANCE_TYPE_NCNI
    num_classes = len(per_class)
    if num_classes == 1:
        return INSTANCE_TYPE_1CNI
    return INSTANCE_TYPE_NCNI


def get_classes_and_instance_counts(
    gt_mask: np.ndarray,
    ignore_index: int = 255,
) -> Tuple[List[int], List[int], int]:
    """
    Returns (class_ids, instance_count_per_class, total_instances).
    """
    per_class = _connected_components_per_class(gt_mask, ignore_index)
    class_ids = [c for c, _ in per_class]
    counts = [n for _, n in per_class]
    total = sum(counts)
    return class_ids, counts, total


def subsample_mask_by_coi(
    mask: np.ndarray,
    class_ids: List[int],
    ignore_index: int = 255,
) -> np.ndarray:
    """Return mask with only the given class IDs; others set to ignore_index."""
    out = np.full_like(mask, ignore_index)
    for c in class_ids:
        out[mask == c] = c
    return out


def filter_predictions_by_instance_type(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    instance_type: str,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (gt_sub, pred_sub) restricted by type.
    - 1CnI: one class as COI — keep only that class in gt and pred.
    - nCnI: retain all classes in COI.
    """
    class_ids, _, _ = get_classes_and_instance_counts(gt_mask, ignore_index)
    if not class_ids:
        return gt_mask, pred_mask

    if instance_type == INSTANCE_TYPE_NCNI:
        gt_sub = subsample_mask_by_coi(gt_mask, class_ids, ignore_index)
        pred_sub = np.where(np.isin(gt_mask, class_ids), pred_mask, ignore_index).astype(pred_mask.dtype)
        pred_sub[gt_mask == ignore_index] = ignore_index
        return gt_sub, pred_sub

    if instance_type == INSTANCE_TYPE_1CNI and len(class_ids) == 1:
        c = class_ids[0]
        gt_sub = np.full_like(gt_mask, ignore_index)
        gt_sub[gt_mask == c] = c
        pred_sub = np.full_like(pred_mask, ignore_index)
        pred_sub[gt_mask == c] = pred_mask[gt_mask == c]
        return gt_sub, pred_sub

    return gt_mask, pred_mask
