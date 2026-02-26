#!/usr/bin/env bash
# Run the pipeline for Cityscapes only (dataset-specific models: PSPNet, DeepLabV3+, SegFormer).
#
# From repo root:
#   bash multi_data_semantic_seg/run_pipeline_cityscapes.sh
#   DEVICE=cpu MAX_SAMPLES=10 bash multi_data_semantic_seg/run_pipeline_cityscapes.sh
#
# Required: input images must be found for each GT file. Either:
#   - Put leftImg8bit inside cityscapes root: gtFine_trainvaltest/leftImg8bit/val/{city}/*_leftImg8bit.png
#   - Or set CITYSCAPES_IMAGE_ROOT to the folder that contains leftImg8bit/ (e.g. parent of leftImg8bit).
# GT: gtFine_trainvaltest/gtFine/val/{city}/*_gtFine_labelIds.png

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-$REPO_ROOT/multi_data_semantic_seg/data/gtFine_trainvaltest}"
OUTPUT_DIR="${CITYSCAPES_OUTPUT_DIR:-multi_data_semantic_seg/output/run_cityscapes}"
OUTPUT_JSON="${CITYSCAPES_OUTPUT_JSON:-results_cityscapes.json}"

# Download checkpoints if missing
bash multi_data_semantic_seg/download_models.sh

CMD=(
  python multi_data_semantic_seg/src/run_pipeline.py
  --cityscapes-root "$CITYSCAPES_ROOT"
  --use-dataset-models
  --augmentations contrast_shift_slight motion_blur_slight
  --device "${DEVICE:-mps}"
  --max-samples "${MAX_SAMPLES:-5}"
  --save-masks
  --output-dir "$OUTPUT_DIR"
  --output-json "$OUTPUT_JSON"
)
[[ -n "${CITYSCAPES_IMAGE_ROOT:-}" ]] && CMD+=(--cityscapes-image-root "$CITYSCAPES_IMAGE_ROOT")

"${CMD[@]}"

echo "Done. Results: $OUTPUT_DIR/$OUTPUT_JSON"
