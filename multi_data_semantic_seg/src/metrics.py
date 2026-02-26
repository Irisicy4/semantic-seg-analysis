"""Standard metrics for semantic segmentation: mIoU, accuracy, etc."""

from typing import Dict, List, Optional

import numpy as np


def intersect_and_union(
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple:
    """Compute intersection and union per class. Returns (intersect, union, pred_area, label_area)."""
    pred = np.asarray(pred, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    # Only consider pixels where GT is valid and pred is in valid class range
    # (so pred==ignore_index or pred>=num_classes don't corrupt histogram bins)
    valid = (
        (label != ignore_index)
        & (label >= 0)
        & (pred != ignore_index)
        & (pred >= 0)
        & (pred < num_classes)
    )
    pred = pred[valid]
    label = label[valid]

    intersect = pred[pred == label]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes))[0]
    area_pred = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes))[0]
    area_union = area_pred + area_label - area_intersect
    return area_intersect.astype(np.float64), area_union.astype(np.float64), area_pred, area_label


def compute_iou(
    area_intersect: np.ndarray,
    area_union: np.ndarray,
    nan_to_num: Optional[float] = None,
) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(area_union > 0, area_intersect / area_union, np.nan)
    if nan_to_num is not None:
        iou = np.nan_to_num(iou, nan=nan_to_num)
    return iou


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    nan_to_num: float = 0.0,
) -> Dict[str, float]:
    """
    Compute mIoU, mDice, mean accuracy, and overall accuracy.
    Returns dict with keys: mean_iou, mean_dice, mean_acc, overall_acc, and per-class IoU if useful.
    """
    area_intersect, area_union, area_pred, area_label = intersect_and_union(
        pred, gt, num_classes, ignore_index
    )
    iou = compute_iou(area_intersect, area_union, nan_to_num=nan_to_num)
    valid = area_union > 0
    mean_iou = float(np.nanmean(iou[valid])) if np.any(valid) else 0.0

    # Dice: 2*intersect / (pred + label)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = np.where(
            area_pred + area_label > 0,
            2 * area_intersect / (area_pred + area_label),
            np.nan,
        )
    mean_dice = float(np.nanmean(dice[valid])) if np.any(valid) else 0.0

    # Per-class accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(area_label > 0, area_intersect / area_label, np.nan)
    mean_acc = float(np.nanmean(acc[valid])) if np.any(valid) else 0.0

    # Overall pixel accuracy
    mask = (gt != ignore_index) & (gt >= 0)
    if np.any(mask):
        overall_acc = float(np.mean(pred[mask] == gt[mask]))
    else:
        overall_acc = 0.0

    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'mean_acc': mean_acc,
        'overall_acc': overall_acc,
        'num_classes': num_classes,
        'valid_classes': int(np.sum(valid)),
    }


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics over multiple samples."""
    if not results:
        return {}
    out = {}
    for k in results[0]:
        if k in ('num_classes', 'valid_classes'):
            continue
        vals = [r[k] for r in results if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
