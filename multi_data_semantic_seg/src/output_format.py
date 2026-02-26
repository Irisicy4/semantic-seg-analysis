"""Build output JSON in the format of sample_json_edited.json (semantic segmentation variant)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def build_image_entry(
    image_id: str,
    file_path: str,
    height: int,
    width: int,
    groundtruth_class: List[str],
    groundtruth_class_id: List[int],
    data_source: str = "local",
    scene: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": image_id,
        "file_path": file_path,
        "data_source": data_source,
        "height": height,
        "width": width,
        "scene": scene or "unknown",
        "groundtruth_class": groundtruth_class,
        "groundtruth_class_id": groundtruth_class_id,
    }


def build_annotation_entry(
    annotation_id: str,
    image_id: str,
    task: str,
    coi: List[str],
    instance_type: str,
    model_name: str,
    augmentation: Optional[str],
    final_score: float,
    other_scores: Dict[str, Any],
    predictions_type: str = "image",
    predictions: Optional[List[Dict[str, Any]]] = None,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one annotation (one model + one aug + one instance_type)."""
    entry = {
        "id": annotation_id,
        "task": task,
        "image_id": image_id,
        "coi": coi,
        "instance_type": instance_type,
        "model_name": model_name,
        "error_type": error_type,
    }
    if augmentation:
        entry["augmentation"] = augmentation
    entry["final_score"] = final_score
    entry["other_scores"] = other_scores
    entry["predictions_type"] = predictions_type
    entry["predictions"] = predictions or []
    return entry


def build_predictions_semantic(
    pred_mask: np.ndarray,
    class_ids: List[int],
    class_names: List[str],
    height: int,
    width: int,
    output_dir: Optional[Path] = None,
    image_id: str = "",
    model_name: str = "",
    aug_name: Optional[str] = None,
    instance_type: str = "nCnI",
    class_id_to_hex: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build predictions list for semantic segmentation.
    Optionally save mask to output_dir and reference by path.
    If class_id_to_hex is provided, each prediction gets a color_hex field.
    """
    predictions = []
    for i, (c_id, c_name) in enumerate(zip(class_ids, class_names)):
        pred = {
            "label": c_name,
            "class_id": c_id,
        }
        if class_id_to_hex is not None:
            pred["color_hex"] = class_id_to_hex.get(int(c_id), "#404040")
        if output_dir and image_id:
            rel_path = f"{image_id}_{model_name}"
            if aug_name:
                rel_path += f"_{aug_name}"
            rel_path += f"_{instance_type}_mask.png"
            pred["file_path"] = rel_path
        predictions.append(pred)
    if not predictions and class_ids:
        p = {
            "label": class_names[0] if class_names else "unknown",
            "class_id": int(class_ids[0]) if class_ids else 0,
        }
        if class_id_to_hex is not None:
            p["color_hex"] = class_id_to_hex.get(int(class_ids[0]), "#404040")
        predictions.append(p)
    return predictions


def build_output_json(
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Final output dict matching sample_json_edited structure."""
    return {
        "images": images,
        "annotations": annotations,
    }
