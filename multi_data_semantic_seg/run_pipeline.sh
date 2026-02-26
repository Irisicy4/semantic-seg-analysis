#!/usr/bin/env bash
# Run the multi_data_semantic_seg inference pipeline from repo root.
# Uses dataset-specific models only (ADE20K models on ADE20K, Cityscapes on Cityscapes).
#
# From 202509-vllm-tool-judge-seg:
#   bash multi_data_semantic_seg/download_models.sh   # once, to fetch checkpoints
#   bash multi_data_semantic_seg/run_pipeline.sh
#
# Optional: CHECKPOINT_DIR=... DEVICE=cuda MAX_SAMPLES=100 bash multi_data_semantic_seg/run_pipeline.sh
# Or: bash multi_data_semantic_seg/run_pipeline.sh --device cpu

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Allow --device / --max-samples from command line
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)   export DEVICE="$2";   shift 2 ;;
    --max-samples) export MAX_SAMPLES="$2"; shift 2 ;;
    *) break ;;
  esac
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Download checkpoints if missing (for --use-dataset-models)
bash multi_data_semantic_seg/download_models.sh

python multi_data_semantic_seg/src/run_pipeline.py \
  --ade20k-root multi_data_semantic_seg/data/ADE20K_2021_17_01_val \
  --cityscapes-root multi_data_semantic_seg/data/gtFine_trainvaltest \
  --use-dataset-models \
  --augmentations contrast_shift_slight motion_blur_slight \
  --device "${DEVICE:-mps}" \
  --max-samples "${MAX_SAMPLES:-5}" \
  --save-masks \
  --output-dir multi_data_semantic_seg/output/run \
  --output-json results.json

echo "Done. Results: multi_data_semantic_seg/output/run/results.json"
