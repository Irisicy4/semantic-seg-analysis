#!/usr/bin/env bash
# Download checkpoints for dataset-specific models (ADE20K and Cityscapes).
# Run from repo root: bash multi_data_semantic_seg/download_models.sh
# Checkpoints are saved under mmsegmentation/checkpoints/.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/mmsegmentation/checkpoints}"
mkdir -p "$CHECKPOINT_DIR"

while IFS=$'\t' read -r url path; do
  if [[ -f "$path" ]]; then
    echo "Skip (exists): $path"
  else
    echo "Download: $url -> $path"
    wget -q -O "$path" "$url" || { echo "Failed: $url"; exit 1; }
  fi
done < <(python3 - "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from multi_data_semantic_seg.dataset_models import DATASET_MODELS, get_checkpoint_path

checkpoint_dir = Path(sys.argv[1])
for dset, models in DATASET_MODELS.items():
    for name, config_rel, url in models:
        path = get_checkpoint_path(url, checkpoint_dir)
        print(f"{url}\t{path}")
PY
)

echo "Done. Checkpoints in: $CHECKPOINT_DIR"
