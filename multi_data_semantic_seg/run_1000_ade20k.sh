#!/usr/bin/env bash
# Run ADE20K pipeline on 1000 images (max). Same options as run_1000_cityscapes.sh.
#
# From repo root:
#   bash multi_data_semantic_seg/run_1000_ade20k.sh
#   bash multi_data_semantic_seg/run_1000_ade20k.sh --parallel-jobs 10
#
# Requires: multi_data_semantic_seg/data/ADE20K_2021_17_01_val (images + annotations)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

ADE20K_ROOT="${ADE20K_ROOT:-$REPO_ROOT/multi_data_semantic_seg/data/ADE20K_2021_17_01_val}"
OUTPUT_DIR="${ADE20K_OUTPUT_DIR:-$REPO_ROOT/multi_data_semantic_seg/output/run_ade20k}"
OUTPUT_JSON="${ADE20K_OUTPUT_JSON:-results_ade20k.json}"

bash multi_data_semantic_seg/download_models.sh

python multi_data_semantic_seg/src/run_pipeline.py \
  --ade20k-root "$ADE20K_ROOT" \
  --use-dataset-models \
  --augmentations contrast_shift_slight motion_blur_slight \
  --device "${DEVICE:-mps}" \
  --max-samples 500 \
  --save-masks \
  --parallel-jobs "${PARALLEL_JOBS:-5}" \
  --output-dir "$OUTPUT_DIR" \
  --output-json "$OUTPUT_JSON" \
  "$@"

echo "Done. Results: $OUTPUT_DIR/$OUTPUT_JSON"
