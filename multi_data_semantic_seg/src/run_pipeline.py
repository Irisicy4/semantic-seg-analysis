#!/usr/bin/env python3
"""
Semantic segmentation inference pipeline.

- Loads ADE20K and/or Cityscapes with ground truth
- Runs multiple models with optional augmentations (contrast_shift / motion_blur, slight & severe)
- Subsamples GT + prediction into 2 types: 1CnI (one class COI, filter model to that class), nCnI (retain all classes)
- Computes metrics and writes JSON in sample_json_edited format

Usage:
  python -m multi_data_semantic_seg.src.run_pipeline --help
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root for imports when run as script (before other local imports)
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from PIL import Image

from multi_data_semantic_seg.src.augmentations import get_augmentation, list_augmentations
from multi_data_semantic_seg.src.datasets import ADE20KDataset, CityscapesDataset
from multi_data_semantic_seg.src.inference import get_device, run_inference_multi, Segmentor
from multi_data_semantic_seg.src.metrics import compute_metrics
from multi_data_semantic_seg.src.output_format import (
    build_annotation_entry,
    build_image_entry,
    build_output_json,
    build_predictions_semantic,
)
from multi_data_semantic_seg.src.subsample import (
    filter_predictions_by_instance_type,
    get_classes_and_instance_counts,
    subsample_mask_by_coi,
    INSTANCE_TYPE_1CNI,
    INSTANCE_TYPE_NCNI,
)

try:
    from multi_data_semantic_seg.dataset_models import get_models_for_dataset, DEFAULT_CHECKPOINT_DIR
except ImportError:
    get_models_for_dataset = None  # type: ignore
    DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "mmsegmentation" / "checkpoints"

# Ignore index in masks
IGNORE_INDEX = 255

# One hex color per class index (0, 1, 2, ...) for reproducible, distinguishable masks.
# First 19 = Cityscapes; then extended for ADE20K and others. Ignore (255) = #404040.
CLASS_COLOR_HEX = [
    "#804080", "#F423E8", "#464646", "#66669C", "#BE9999", "#999999",
    "#FAAA1E", "#DCDC00", "#6B8E23", "#98FB98", "#4682B4", "#DC143C",
    "#FF0000", "#00008E", "#000046", "#003C64", "#005064", "#770B20", "#0000E6",
    "#2E8B57", "#FF6347", "#9370DB", "#FFD700", "#20B2AA", "#FF69B4", "#00CED1",
    "#FF4500", "#32CD32", "#BA55D3", "#ADFF2F", "#FF1493", "#00BFFF", "#696969",
    "#556B2F", "#8B4513", "#483D8B", "#2F4F4F", "#CD853F", "#8B008B", "#B22222",
    "#5F9EA0", "#D2691E", "#6495ED", "#DC143C", "#000080", "#008080", "#808080",
    "#FF00FF", "#800000", "#008000", "#800080", "#008080", "#C0C0C0", "#FFA07A",
    "#7CFC00", "#DDA0DD", "#F0E68C", "#E6E6FA", "#BC8F8F", "#4169E1", "#8B4513",
    "#A0522D", "#CD5C5C", "#4B0082", "#F0FFF0", "#FFF0F5", "#F5F5DC", "#FFE4B5",
    "#E0FFFF", "#FAF0E6", "#FFE4E1", "#F5DEB3", "#FFF8DC", "#FFFAF0", "#F0F8FF",
    "#F8F8FF", "#F5F5F5", "#FFEFD5", "#FAEBD7", "#FFE4C4", "#FFDEAD", "#FFDAB9",
    "#FFC0CB", "#FFB6C1", "#FFA07A", "#FF7F50", "#FF6347", "#FF4500", "#FFD700",
    "#FFA500", "#FF8C00", "#B8860B", "#DAA520", "#D2691E", "#8B4513", "#A0522D",
    "#CD853F", "#DEB887", "#F5DEB3", "#D2B48C", "#BC8F8F", "#F4A460", "#DA70D6",
    "#EE82EE", "#FF00FF", "#BA55D3", "#9370DB", "#8A2BE2", "#9400D3", "#9932CC",
    "#8B008B", "#4B0082", "#6A5ACD", "#483D8B", "#7B68EE", "#00CED1", "#48D1CC",
    "#40E0D0", "#00FA9A", "#3CB371", "#2E8B57", "#228B22", "#006400", "#9ACD32",
    "#6B8E23", "#556B2F", "#ADFF2F", "#7FFF00", "#7CFC00", "#00FF00", "#32CD32",
    "#98FB98", "#90EE90", "#00FF7F", "#00FA9A", "#8FBC8F", "#3CB371", "#2E8B57",
    "#228B22", "#006400", "#9ACD32", "#6B8E23", "#808000", "#BDB76B", "#EEE8AA",
    "#FAFAD2", "#FFFFF0", "#FFFFE0", "#FFFACD", "#FAF0E6", "#FFEFD5", "#FFE4B5",
    "#FFDAB9", "#FFDEAD", "#FFE4C4", "#FFEFD5", "#FFF8DC", "#FFFAF0", "#FFFF00",
    "#FFD700", "#FFA500", "#FF8C00", "#FF7F50", "#FF6347", "#FF4500", "#DC143C",
    "#B22222", "#8B0000", "#A52A2A", "#CD5C5C", "#F08080", "#FFA07A", "#FA8072",
    "#E9967A", "#F4A460", "#FF7F50", "#FF6347", "#FF4500", "#EE82EE", "#FF69B4",
    "#FF1493", "#FF00FF", "#DB7093", "#C71585", "#8B008B", "#9370DB", "#8A2BE2",
    "#9400D3", "#9932CC", "#BA55D3", "#DDA0DD", "#EE82EE", "#DA70D6", "#FF00FF",
    "#00BFFF", "#1E90FF", "#4169E1", "#6495ED", "#87CEEB", "#87CEFA", "#4682B4",
    "#5F9EA0", "#00CED1", "#48D1CC", "#40E0D0", "#00FFFF", "#7FFFD4", "#F0FFFF",
    "#E0FFFF", "#AFEEEE", "#B0E0E6", "#ADD8E6", "#87CEEB", "#87CEFA", "#00BFFF",
    "#1E90FF", "#4169E1", "#6495ED", "#708090", "#778899", "#2F4F4F", "#696969",
    "#808080", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#DCDCDC", "#F5F5F5", "#FFFAFA",
    "#F0F0F0", "#FFFFFF",
]
IGNORE_COLOR_HEX = "#404040"


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' to (r, g, b) uint8."""
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def _palette_from_hex(max_label: int) -> np.ndarray:
    """Build (256, 3) RGB palette from CLASS_COLOR_HEX; index 255 = IGNORE_COLOR_HEX."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[IGNORE_INDEX] = _hex_to_rgb(IGNORE_COLOR_HEX)
    for i in range(min(max_label + 1, 255)):
        if i == IGNORE_INDEX:
            continue
        hex_color = CLASS_COLOR_HEX[i % len(CLASS_COLOR_HEX)]
        palette[i] = _hex_to_rgb(hex_color)
    return palette


def _colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert (H,W) label mask to (H,W,3) RGB using CLASS_COLOR_HEX per class."""
    mask = np.asarray(mask, dtype=np.int32)
    max_label = int(max(mask.max(), 0))
    if ignore_index <= max_label:
        max_label = max(max_label, 19)
    palette = _palette_from_hex(max_label)
    h, w = mask.shape
    flat = np.clip(mask.ravel(), 0, 255)
    rgb = palette[flat].reshape((h, w, 3))
    return rgb


def _sample_pred_to_coi_for_1c(
    pred_sub: np.ndarray, gt_sub: np.ndarray, coi_class_id: int, ignore_index: int = 255
) -> np.ndarray:
    """For 1CnI: sample prediction down to the single COI class (filter model response to this class). Pixels where model predicted coi_class_id stay as coi_class_id, all others become ignore."""
    out = np.full_like(pred_sub, ignore_index)
    in_coi_region = (gt_sub >= 0) & (gt_sub != ignore_index)
    predicted_coi = (pred_sub == coi_class_id) & in_coi_region
    out[predicted_coi] = coi_class_id
    return out


def get_class_id_to_hex(max_class_id: int = 255) -> Dict[int, str]:
    """Return mapping class_id -> hex for use in outputs (e.g. JSON or legend)."""
    out = {}
    out[IGNORE_INDEX] = IGNORE_COLOR_HEX
    for i in range(min(max_class_id + 1, 255)):
        if i == IGNORE_INDEX:
            continue
        out[i] = CLASS_COLOR_HEX[i % len(CLASS_COLOR_HEX)]
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Semantic segmentation inference pipeline")
    p.add_argument("--ade20k-root", type=str, default=None, help="Path to ADE20K_2021_17_01_val")
    p.add_argument("--cityscapes-root", type=str, default=None, help="Path to gtFine_trainvaltest (Cityscapes)")
    p.add_argument("--cityscapes-image-root", type=str, default=None, help="Path to leftImg8bit if not under cityscapes-root")
    p.add_argument("--models", nargs="+", default=[], help="List of (model_name, config_path, checkpoint_path) as name:config:ckpt (ignored if --use-dataset-models)")
    p.add_argument("--use-dataset-models", action="store_true", help="Use 3–4 models per dataset (ade20k / cityscapes only); run only applicable models on each dataset")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory for --use-dataset-models (default: mmsegmentation/checkpoints)")
    p.add_argument("--augmentations", nargs="*", default=None, help="Aug names: contrast_shift_slight, contrast_shift_severe, motion_blur_slight, motion_blur_severe. Empty = none, omit = all four")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Device (default: auto)")
    p.add_argument("--output-dir", type=str, default="multi_data_semantic_seg/output", help="Output directory")
    p.add_argument("--output-json", type=str, default="results.json", help="Output JSON filename")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    p.add_argument("--save-masks", action="store_true", help="Save prediction masks as PNG")
    p.add_argument("--split", type=str, default="val", help="Split: val, train, or test (Cityscapes); validation (ADE20K)")
    return p.parse_args()


def collect_augmentations(aug_names: Optional[List[str]]) -> List[Tuple[Optional[str], Optional[Any]]]:
    if aug_names is None:
        aug_names = list_augmentations()
    if not aug_names:
        return [(None, None)]
    out = [(None, None)]
    for name in aug_names:
        try:
            fn = get_augmentation(name)
            out.append((name, fn))
        except KeyError:
            pass
    return out


def main():
    args = parse_args()
    if not args.ade20k_root and not args.cityscapes_root:
        print("Provide at least one of --ade20k-root or --cityscapes-root", file=sys.stderr)
        sys.exit(1)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Datasets
    datasets = []
    if args.ade20k_root:
        ade = ADE20KDataset(
            args.ade20k_root,
            split="validation",
            max_samples=args.max_samples,
        )
        datasets.append(("ade20k", ade))
    if args.cityscapes_root:
        cs = CityscapesDataset(
            args.cityscapes_root,
            image_root=args.cityscapes_image_root,
            split=args.split,
            max_samples=args.max_samples,
        )
        datasets.append(("cityscapes", cs))

    # Models: either per-dataset (--use-dataset-models) or single list (--models)
    segmentors_by_dataset: Dict[str, List[Segmentor]] = {}
    if args.use_dataset_models:
        if not get_models_for_dataset:
            print("--use-dataset-models requires multi_data_semantic_seg.dataset_models", file=sys.stderr)
            sys.exit(1)
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        for dset_name, dset in datasets:
            specs = get_models_for_dataset(dset_name, ckpt_dir)
            segmentors = []
            for name, config_path, ckpt_path in specs:
                if not ckpt_path.exists():
                    print(f"Checkpoint missing for {dset_name}/{name}: {ckpt_path}. Run: bash multi_data_semantic_seg/download_models.sh", file=sys.stderr)
                    continue
                segmentors.append(Segmentor(name, config_path, str(ckpt_path), device=device))
            segmentors_by_dataset[dset_name] = segmentors
        for dset_name, segs in segmentors_by_dataset.items():
            if not segs:
                print(f"No models loaded for {dset_name}. Download checkpoints: bash multi_data_semantic_seg/download_models.sh", file=sys.stderr)
                sys.exit(1)
    else:
        segmentors: List[Segmentor] = []
        for spec in args.models:
            parts = spec.split(":")
            if len(parts) != 3:
                print(f"Invalid model spec (use name:config:checkpoint): {spec}", file=sys.stderr)
                continue
            name, config_path, ckpt_path = parts
            segmentors.append(Segmentor(name, config_path, ckpt_path, device=device))
        if not segmentors:
            print("No models loaded. Provide --models name:config:ckpt ... or --use-dataset-models", file=sys.stderr)
            sys.exit(1)
        for dset_name, _ in datasets:
            segmentors_by_dataset[dset_name] = segmentors

    augmentations = collect_augmentations(args.augmentations)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks" if args.save_masks else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)

    images_out: List[Dict[str, Any]] = []
    annotations_out: List[Dict[str, Any]] = []
    ann_id = 0

    for dset_name, dset in datasets:
        if len(dset) == 0:
            print(
                f"No samples found for {dset_name}. " + (
                    "Cityscapes needs leftImg8bit/{split}/{city}/*_leftImg8bit.png under the cityscapes root."
                    if dset_name == "cityscapes" else "Check dataset paths and structure."
                ),
                file=sys.stderr,
            )
            continue
        class_names = dset.get_class_names()
        if hasattr(dset, "ignore_index"):
            ignore_index = dset.ignore_index
        else:
            ignore_index = 255

        for idx in tqdm(range(len(dset)), desc=dset_name, unit="img"):
            sample = dset[idx]
            image = sample["image"]
            gt = sample["gt"]
            sample_id = sample["sample_id"]
            image_path = sample["image_path"]
            h, w = sample["height"], sample["width"]

            class_ids, counts, total_instances = get_classes_and_instance_counts(gt, ignore_index)
            coi_names = [class_names[c] if c < len(class_names) else str(c) for c in class_ids]

            image_id = f"{dset_name}_{sample_id}"
            images_out.append(
                build_image_entry(
                    image_id=image_id,
                    file_path=Path(image_path).name,
                    height=h,
                    width=w,
                    groundtruth_class=coi_names,
                    groundtruth_class_id=class_ids,
                    data_source=dset_name,
                )
            )

            # Inference: multiple models × (clean + augmentations) — only models for this dataset
            segmentors = segmentors_by_dataset[dset_name]
            try:
                results = run_inference_multi(image, segmentors, augmentations)
            except Exception as e:
                print(f"Inference failed for {image_id}: {e}", file=sys.stderr)
                continue

            # Class ID -> hex for prediction labels (same palette as saved masks)
            class_id_to_hex = get_class_id_to_hex(254)

            # --- nCnI: always emit annotations with all COI classes ---
            itype_ncni = INSTANCE_TYPE_NCNI
            gt_sub_ncni, _ = filter_predictions_by_instance_type(gt, gt, itype_ncni, ignore_index)

            for model_name, aug_name, pred in results:
                gt_sub, pred_sub = filter_predictions_by_instance_type(gt, pred, itype_ncni, ignore_index)
                num_classes = min(256, max(int(gt_sub.max()) + 1, int(pred_sub.max()) + 1, len(class_ids) + 1))
                met = compute_metrics(pred_sub, gt_sub, num_classes, ignore_index=ignore_index)

                # Warn when IoU is 0 and pred/gt label ranges suggest cross-dataset mismatch
                if met["mean_iou"] == 0.0 and np.any((gt_sub >= 0) & (gt_sub != ignore_index)):
                    valid_region = (gt_sub >= 0) & (gt_sub != ignore_index)
                    pred_in_region = pred_sub[valid_region]
                    pred_class_vals = pred_in_region[
                        (pred_in_region >= 0) & (pred_in_region != ignore_index) & (pred_in_region < num_classes)
                    ]
                    pred_max = int(pred_class_vals.max()) if pred_class_vals.size else -1
                    gt_class_vals = gt_sub[valid_region]
                    gt_max = int(gt_class_vals.max()) if gt_class_vals.size else -1
                    if pred_max <= 20 and gt_max > 20:
                        print(
                            f"Warning: IoU=0 likely due to label space mismatch "
                            f"(pred max={pred_max}, GT max={gt_max}). "
                            "Use a model trained on the same dataset for meaningful IoU.",
                            file=sys.stderr,
                        )

                other_scores = {
                    "mean_iou": met["mean_iou"],
                    "mean_dice": met["mean_dice"],
                    "mean_acc": met["mean_acc"],
                    "overall_acc": met["overall_acc"],
                    "num_classes_gt": len(class_ids),
                    "total_instances_gt": total_instances,
                    "instance_type": itype_ncni,
                }

                pred_list = build_predictions_semantic(
                    pred_sub,
                    class_ids,
                    coi_names,
                    h,
                    w,
                    output_dir=mask_dir,
                    image_id=image_id,
                    model_name=model_name,
                    aug_name=aug_name,
                    instance_type=itype_ncni,
                    class_id_to_hex=class_id_to_hex,
                )

                ann_id_str = f"seg_{ann_id}"
                ann_id += 1
                annotations_out.append(
                    build_annotation_entry(
                        annotation_id=ann_id_str,
                        image_id=image_id,
                        task="semantic_segmentation",
                        coi=coi_names,
                        instance_type=itype_ncni,
                        model_name=model_name,
                        augmentation=aug_name,
                        final_score=met["mean_iou"],
                        other_scores=other_scores,
                        predictions_type="image",
                        predictions=pred_list,
                    )
                )

                if args.save_masks and mask_dir:
                    mean_iou = met["mean_iou"]
                    fname = f"{image_id}_{model_name}"
                    if aug_name:
                        fname += f"_{aug_name}"
                    fname += f"_{itype_ncni}_iou{mean_iou:.4f}.png"
                    out_path = mask_dir / fname
                    rgb = _colorize_mask(pred_sub, ignore_index)
                    Image.fromarray(rgb).save(out_path)

            # nCnI GT entry
            if mask_dir:
                gt_mask_fname = f"{image_id}_gt_{itype_ncni}_mask.png"
                gt_mask_path = mask_dir / gt_mask_fname
                Image.fromarray(_colorize_mask(gt_sub_ncni, ignore_index)).save(gt_mask_path)
            gt_pred_list = build_predictions_semantic(
                gt_sub_ncni,
                class_ids,
                coi_names,
                h,
                w,
                output_dir=mask_dir,
                image_id=image_id,
                model_name="gt",
                aug_name=None,
                instance_type=itype_ncni,
                class_id_to_hex=class_id_to_hex,
            )
            ann_id_str = f"seg_{ann_id}"
            ann_id += 1
            annotations_out.append(
                build_annotation_entry(
                    annotation_id=ann_id_str,
                    image_id=image_id,
                    task="semantic_segmentation",
                    coi=coi_names,
                    instance_type=itype_ncni,
                    model_name="gt",
                    error_type="gt",
                    augmentation=None,
                    final_score=1.0,
                    other_scores={
                        "mean_iou": 1.0,
                        "mean_dice": 1.0,
                        "mean_acc": 1.0,
                        "overall_acc": 1.0,
                        "num_classes_gt": len(class_ids),
                        "total_instances_gt": total_instances,
                        "instance_type": itype_ncni,
                    },
                    predictions_type="image",
                    predictions=gt_pred_list,
                )
            )

            # --- 1CnI: rule-based random sample of one class; trim GT and pred to that class ---
            if class_ids:
                rng = random.Random(hash(image_id) % (2**32))
                coi_1c = int(rng.choice(class_ids))
                coi_names_1c = [class_names[coi_1c] if coi_1c < len(class_names) else str(coi_1c)]
                class_ids_1c = [coi_1c]
                gt_sub_1c = subsample_mask_by_coi(gt, class_ids_1c, ignore_index)
                if mask_dir:
                    gt_mask_fname_1c = f"{image_id}_gt_{INSTANCE_TYPE_1CNI}_mask.png"
                    Image.fromarray(_colorize_mask(gt_sub_1c, ignore_index)).save(mask_dir / gt_mask_fname_1c)

                for model_name, aug_name, pred in results:
                    pred_1c = _sample_pred_to_coi_for_1c(pred, gt_sub_1c, coi_1c, ignore_index)
                    num_classes_1c = 2  # class + ignore
                    met_1c = compute_metrics(pred_1c, gt_sub_1c, num_classes_1c, ignore_index=ignore_index)
                    other_scores_1c = {
                        "mean_iou": met_1c["mean_iou"],
                        "mean_dice": met_1c["mean_dice"],
                        "mean_acc": met_1c["mean_acc"],
                        "overall_acc": met_1c["overall_acc"],
                        "num_classes_gt": 1,
                        "total_instances_gt": total_instances,
                        "instance_type": INSTANCE_TYPE_1CNI,
                    }
                    pred_list_1c = build_predictions_semantic(
                        pred_1c,
                        class_ids_1c,
                        coi_names_1c,
                        h,
                        w,
                        output_dir=mask_dir,
                        image_id=image_id,
                        model_name=model_name,
                        aug_name=aug_name,
                        instance_type=INSTANCE_TYPE_1CNI,
                        class_id_to_hex=class_id_to_hex,
                    )
                    ann_id_str = f"seg_{ann_id}"
                    ann_id += 1
                    annotations_out.append(
                        build_annotation_entry(
                            annotation_id=ann_id_str,
                            image_id=image_id,
                            task="semantic_segmentation",
                            coi=coi_names_1c,
                            instance_type=INSTANCE_TYPE_1CNI,
                            model_name=model_name,
                            augmentation=aug_name,
                            final_score=met_1c["mean_iou"],
                            other_scores=other_scores_1c,
                            predictions_type="image",
                            predictions=pred_list_1c,
                        )
                    )
                    if args.save_masks and mask_dir:
                        fname = f"{image_id}_{model_name}"
                        if aug_name:
                            fname += f"_{aug_name}"
                        fname += f"_{INSTANCE_TYPE_1CNI}_iou{met_1c['mean_iou']:.4f}.png"
                        Image.fromarray(_colorize_mask(pred_1c, ignore_index)).save(mask_dir / fname)

                # 1CnI GT entry
                gt_pred_list_1c = build_predictions_semantic(
                    gt_sub_1c,
                    class_ids_1c,
                    coi_names_1c,
                    h,
                    w,
                    output_dir=mask_dir,
                    image_id=image_id,
                    model_name="gt",
                    aug_name=None,
                    instance_type=INSTANCE_TYPE_1CNI,
                    class_id_to_hex=class_id_to_hex,
                )
                ann_id_str = f"seg_{ann_id}"
                ann_id += 1
                annotations_out.append(
                    build_annotation_entry(
                        annotation_id=ann_id_str,
                        image_id=image_id,
                        task="semantic_segmentation",
                        coi=coi_names_1c,
                        instance_type=INSTANCE_TYPE_1CNI,
                        model_name="gt",
                        error_type="gt",
                        augmentation=None,
                        final_score=1.0,
                        other_scores={
                            "mean_iou": 1.0,
                            "mean_dice": 1.0,
                            "mean_acc": 1.0,
                            "overall_acc": 1.0,
                            "num_classes_gt": 1,
                            "total_instances_gt": total_instances,
                            "instance_type": INSTANCE_TYPE_1CNI,
                        },
                        predictions_type="image",
                        predictions=gt_pred_list_1c,
                    )
                )

    if not images_out:
        print("No predictions produced (no samples in any dataset). Check paths and see messages above.", file=sys.stderr)
        sys.exit(1)
    out_json = build_output_json(images_out, annotations_out)
    out_path = out_dir / args.output_json
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Wrote {out_path}")

    # Write class_id -> color hex for mask legend (same palette used for saved masks)
    palette_path = out_dir / "palette_class_id_to_hex.json"
    id_to_hex = get_class_id_to_hex(254)
    with open(palette_path, "w") as f:
        json.dump({str(k): v for k, v in id_to_hex.items()}, f, indent=2)
    print(f"Wrote {palette_path} (class_id -> color hex for mask visualization)")


if __name__ == "__main__":
    main()
