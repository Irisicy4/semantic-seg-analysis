#!/usr/bin/env bash
# Run Cityscapes pipeline on 1000 images (max). Uses leftImg8bit under data root only.
#
# From repo root:
#   bash multi_data_semantic_seg/run_1000_cityscapes.sh
#
# Requires: multi_data_semantic_seg/data/gtFine_trainvaltest/leftImg8bit/val/{city}/*_leftImg8bit.png

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CITYSCAPES_ROOT="${REPO_ROOT}/multi_data_semantic_seg/data/gtFine_trainvaltest"
OUTPUT_DIR="${REPO_ROOT}/multi_data_semantic_seg/output/run_cityscapes"
OUTPUT_JSON="results_cityscapes.json"

bash multi_data_semantic_seg/download_models.sh

python multi_data_semantic_seg/src/run_pipeline.py \
  --cityscapes-root "$CITYSCAPES_ROOT" \
  --use-dataset-models \
  --augmentations contrast_shift_slight motion_blur_slight \
  --device "${DEVICE:-mps}" \
  --max-samples 1000 \
  --save-masks \
  --output-dir "$OUTPUT_DIR" \
  --output-json "$OUTPUT_JSON"

echo "Done. Results: $OUTPUT_DIR/$OUTPUT_JSON"
