#!/usr/bin/env bash
# Run analysis on results and open the result filter UI for review.
#
# From repo root:
#   bash multi_data_semantic_seg/review_cityscapes.sh [RESULTS_JSON]
#
# Examples:
#   bash multi_data_semantic_seg/review_cityscapes.sh
#   bash multi_data_semantic_seg/review_cityscapes.sh multi_data_semantic_seg/output0/run/results.json
#
# 1) Runs result_analyzer → writes analysis JSON next to results
# 2) Starts the filter app (Flask) and opens http://127.0.0.1:5001 (default port; avoids macOS AirPlay on 5000)
#
# Let others on your LAN connect (judgments save to judgments JSON next to results):
#   HOST=0.0.0.0 bash multi_data_semantic_seg/review_cityscapes.sh [RESULTS_JSON]

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

DEFAULT_RESULTS="${REPO_ROOT}/multi_data_semantic_seg/output/run_cityscapes/results_cityscapes.json"
RESULTS_JSON="${1:-$DEFAULT_RESULTS}"
# Resolve to absolute path so dirname works when path is relative
RESULTS_JSON="$(cd "$(dirname "$RESULTS_JSON")" && pwd)/$(basename "$RESULTS_JSON")"
OUTPUT_DIR="$(dirname "$RESULTS_JSON")"
BASE="$(basename "$RESULTS_JSON" .json)"
if [[ "$BASE" == "results" ]]; then
  ANALYSIS_JSON="${OUTPUT_DIR}/analysis.json"
else
  ANALYSIS_JSON="${OUTPUT_DIR}/analysis_${BASE#results_}.json"
fi
PORT="${PORT:-5001}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -f "$RESULTS_JSON" ]]; then
  echo "Error: $RESULTS_JSON not found." >&2
  exit 1
fi

echo "Running analysis..."
python multi_data_semantic_seg/src/result_analyzer.py "$RESULTS_JSON" -o "$ANALYSIS_JSON"
if [[ "$HOST" = "0.0.0.0" ]]; then
  MY_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
  [[ -z "$MY_IP" ]] && MY_IP=$(hostname -s 2>/dev/null || echo "this-machine")
  echo "Review UI (LAN): http://${MY_IP}:$PORT"
  echo "  — Others: open the URL above in the browser (include :$PORT)."
  echo "  — If connection times out: on this Mac open System Settings → Network → Firewall → Options and allow Python (or turn off firewall temporarily)."
else
  echo "Starting review UI at http://127.0.0.1:$PORT (open in browser if it doesn't open automatically)"
  (sleep 2 && (open "http://127.0.0.1:$PORT" 2>/dev/null || xdg-open "http://127.0.0.1:$PORT" 2>/dev/null || true)) &
fi
pip install -q -r multi_data_semantic_seg/result_filter_app/requirements.txt 2>/dev/null || true
python multi_data_semantic_seg/result_filter_app/app.py "$RESULTS_JSON" --analysis "$ANALYSIS_JSON" --port "$PORT" --host "$HOST"
