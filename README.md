# Multi-Data Semantic Segmentation Inference Pipeline

Run semantic segmentation (ADE20K and/or Cityscapes) with multiple models and optional image augmentations; output JSON and optional mask PNGs with metrics.

---

## Setup variation list

| Variation | CLI / env | Default | Description |
|-----------|-----------|---------|-------------|
| **Datasets** | `--ade20k-root`, `--cityscapes-root` | (none) | At least one required. Paths to ADE20K_2021_17_01_val and/or gtFine_trainvaltest. |
| **Cityscapes images** | `--cityscapes-image-root` or `CITYSCAPES_IMAGE_ROOT` | same as cityscapes-root | Folder that contains `leftImg8bit/` if not under cityscapes-root. |
| **Models** | `--use-dataset-models` or `--models name:config:ckpt ...` | (none) | **Dataset-specific**: 4 ADE20K + 4 Cityscapes models (see below). **Custom**: list of `name:config:ckpt` (same models on all datasets). |
| **Checkpoints** | `--checkpoint-dir` or `CHECKPOINT_DIR` | `mmsegmentation/checkpoints` | Used only with `--use-dataset-models`. |
| **Augmentations** | `--augmentations [names]` | all four | `contrast_shift_slight`, `contrast_shift_severe`, `motion_blur_slight`, `motion_blur_severe`. Omit = all four; pass empty `[]` = clean only. |
| **Device** | `--device` or `DEVICE` | auto (cuda → mps → cpu) | `cuda`, `mps`, or `cpu`. |
| **Split** | `--split` | `val` | Cityscapes: `val` / `train` / `test`. ADE20K always validation. |
| **Max samples** | `--max-samples` or `MAX_SAMPLES` | no limit | Cap per dataset (for quick runs). |
| **Output dir** | `--output-dir` | `multi_data_semantic_seg/output` | Directory for JSON and (if requested) `masks/`. |
| **Output JSON** | `--output-json` | `results.json` | Filename of results JSON inside output-dir. |
| **Save masks** | `--save-masks` | off | Save colorized prediction mask PNGs (filename includes IoU). |

**Dataset-specific models (with `--use-dataset-models`):**

- **ADE20K**: pspnet_r50, pspnet_r101, deeplabv3plus_r50, segformer_mit-b2 (150 classes).
- **Cityscapes**: pspnet_r50, deeplabv3plus_r50, deeplabv3plus_r101, segformer_mit-b2 (19 train IDs).

Checkpoints are downloaded once with `bash multi_data_semantic_seg/download_models.sh`.

---

## Metrics

Computed **per (image, model, augmentation, instance_type)** over the evaluated region (COI only for nCnI/1CnI):

| Metric | Definition | In output |
|--------|------------|-----------|
| **mean_iou** | Mean over classes (with union > 0) of IoU = intersection / union. | `final_score`, `other_scores.mean_iou` |
| **mean_dice** | Mean over classes of Dice = 2×intersection / (pred_area + label_area). | `other_scores.mean_dice` |
| **mean_acc** | Mean over classes of per-class accuracy = intersection / label_area. | `other_scores.mean_acc` |
| **overall_acc** | Pixel accuracy: fraction of (valid GT) pixels where pred == GT. | `other_scores.overall_acc` |

- **final_score** in the JSON (and in mask filenames) is **mean_iou**.
- Ignore index is 255; only pixels with valid GT and valid pred class index are included in the histograms.

---

## How to run

All commands from the **repo root** (parent of `multi_data_semantic_seg`).

### 1. Download checkpoints (once)

```bash
bash multi_data_semantic_seg/download_models.sh
```

### 2. Scripts (recommended)

**Both ADE20K and Cityscapes** (dataset-specific models, slight augmentations):

```bash
bash multi_data_semantic_seg/run_pipeline.sh
```

Options via env or CLI:

```bash
# CPU and/or limit samples
bash multi_data_semantic_seg/run_pipeline.sh --device cpu
DEVICE=cpu MAX_SAMPLES=10 bash multi_data_semantic_seg/run_pipeline.sh
```

**Cityscapes only** (writes to `output/run_cityscapes/`, `results_cityscapes.json`):

```bash
bash multi_data_semantic_seg/run_pipeline_cityscapes.sh
DEVICE=cpu MAX_SAMPLES=5 bash multi_data_semantic_seg/run_pipeline_cityscapes.sh
```

Override Cityscapes paths if needed:

```bash
CITYSCAPES_ROOT=/path/to/gtFine_trainvaltest \
CITYSCAPES_IMAGE_ROOT=/path/to/folder/containing/leftImg8bit \
bash multi_data_semantic_seg/run_pipeline_cityscapes.sh
```

### 3. Manual Python

```bash
export PYTHONPATH=/path/to/202509-vllm-tool-judge-seg:$PYTHONPATH

# ADE20K + Cityscapes, dataset-specific models, slight augs, CPU, save masks
python multi_data_semantic_seg/src/run_pipeline.py \
  --ade20k-root multi_data_semantic_seg/data/ADE20K_2021_17_01_val \
  --cityscapes-root multi_data_semantic_seg/data/gtFine_trainvaltest \
  --use-dataset-models \
  --augmentations contrast_shift_slight motion_blur_slight \
  --device cpu \
  --max-samples 5 \
  --save-masks \
  --output-dir multi_data_semantic_seg/output/run \
  --output-json results.json

# Cityscapes only
python multi_data_semantic_seg/src/run_pipeline.py \
  --cityscapes-root multi_data_semantic_seg/data/gtFine_trainvaltest \
  --use-dataset-models \
  --augmentations contrast_shift_slight motion_blur_slight \
  --device cpu --save-masks \
  --output-dir multi_data_semantic_seg/output/run_cityscapes \
  --output-json results_cityscapes.json

# Custom models (no --use-dataset-models)
python multi_data_semantic_seg/src/run_pipeline.py \
  --ade20k-root multi_data_semantic_seg/data/ADE20K_2021_17_01_val \
  --models mymodel:mmseg/config.py:mmseg/checkpoints/ckpt.pth \
  --max-samples 2 --output-json results.json
```

---

## Data layout

- **ADE20K**: `{ade20k-root}/images/ADE/validation/...` with `{stem}.jpg`, `{stem}_seg.png`, and optional `objects.txt` at root (needed for correct 150-class GT from RGB-encoded _seg.png).
- **Cityscapes**: You need **both** labels and input images or the run will produce 0 predictions. Labels: `{cityscapes-root}/gtFine/val/{city}/*_gtFine_labelIds.png`. Images: `{image-root}/leftImg8bit/val/{city}/*_leftImg8bit.png` (image-root defaults to cityscapes-root). If images live elsewhere, set `--cityscapes-image-root` (or `CITYSCAPES_IMAGE_ROOT`) to the folder that contains `leftImg8bit/`. Fallback: images in the same folder as the GT with name `{stem}_leftImg8bit.png` or `{stem}.png` are also found.

---

## Output

- **JSON**: `output-dir/{output-json}` — `images` (id, file_path, groundtruth_class, data_source, …) and `annotations` (task, image_id, coi, instance_type, model_name, augmentation, final_score, other_scores, predictions).
- **Masks** (if `--save-masks`): `output-dir/masks/` — one PNG per (image, model, augmentation, instance_type); filename includes mean IoU.
- **Palette**: `output-dir/palette_class_id_to_hex.json` — class ID → hex color used for mask visualization.

---

## Dependencies

- Python 3, numpy, Pillow, scipy, torch, **opencv-python** (cv2)
- **mmsegmentation** (mmengine, mmcv) for inference
