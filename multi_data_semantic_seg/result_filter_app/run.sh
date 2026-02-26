#!/usr/bin/env bash
# Run the result filter app. From repo root or from result_filter_app/.
#
# Usage:
#   bash multi_data_semantic_seg/result_filter_app/run.sh [results.json] [analysis.json]
#
# Example:
#   bash multi_data_semantic_seg/result_filter_app/run.sh ../output/run/results.json ../output/run/analysis.json
#   bash multi_data_semantic_seg/result_filter_app/run.sh ../output/run_cityscapes/results_cityscapes.json

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS="${1:-../output/run/results.json}"
ANALYSIS="${2:-}"
pip install -q -r requirements.txt 2>/dev/null || true
if [[ -n "$ANALYSIS" ]]; then
  python app.py "$RESULTS" --analysis "$ANALYSIS" --port 5000
else
  python app.py "$RESULTS" --port 5000
fi
