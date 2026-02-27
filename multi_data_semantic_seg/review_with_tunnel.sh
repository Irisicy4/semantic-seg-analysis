#!/usr/bin/env bash
# Run the review UI and expose it via ngrok so anyone with the link can access (no same-LAN needed).
#
# Prerequisites:
#   brew install ngrok
#   ngrok config add-authtoken YOUR_TOKEN   # get token from https://ngrok.com
#
# From repo root:
#   bash multi_data_semantic_seg/review_with_tunnel.sh [RESULTS_JSON]
#
# Share the https://xxxx.ngrok-free.app URL with reviewers. Judgments still save to your local judgments JSON.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

DEFAULT_RESULTS="${REPO_ROOT}/multi_data_semantic_seg/output/run_cityscapes/results_cityscapes.json"
RESULTS_JSON="${1:-$DEFAULT_RESULTS}"
RESULTS_JSON="$(cd "$(dirname "$RESULTS_JSON")" && pwd)/$(basename "$RESULTS_JSON")"
OUTPUT_DIR="$(dirname "$RESULTS_JSON")"
BASE="$(basename "$RESULTS_JSON" .json)"
if [[ "$BASE" == "results" ]]; then
  ANALYSIS_JSON="${OUTPUT_DIR}/analysis.json"
else
  ANALYSIS_JSON="${OUTPUT_DIR}/analysis_${BASE#results_}.json"
fi
PORT="${PORT:-5002}"

if [[ ! -f "$RESULTS_JSON" ]]; then
  echo "Error: $RESULTS_JSON not found." >&2
  exit 1
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "Error: ngrok not found. Install with: brew install ngrok" >&2
  echo "Then add your authtoken: ngrok config add-authtoken YOUR_TOKEN" >&2
  exit 1
fi

echo "Running analysis..."
python multi_data_semantic_seg/src/result_analyzer.py "$RESULTS_JSON" -o "$ANALYSIS_JSON"
pip install -q -r multi_data_semantic_seg/result_filter_app/requirements.txt 2>/dev/null || true

echo "Starting review UI on port $PORT (background)..."
python multi_data_semantic_seg/result_filter_app/app.py "$RESULTS_JSON" --analysis "$ANALYSIS_JSON" --port "$PORT" --host "0.0.0.0" &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null; exit' INT TERM

sleep 2
if ! kill -0 $APP_PID 2>/dev/null; then
  echo "Error: app failed to start." >&2
  exit 1
fi

echo "Starting ngrok tunnel. Share the HTTPS URL below with reviewers."
echo "Judgments will save to your local judgments file. Press Ctrl+C to stop."
ngrok http "$PORT"
