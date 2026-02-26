#!/usr/bin/env python3
"""
Result filter app: serve results JSON, mask images, and accept pass/fail judgments.
Updates results JSON in real time with user_judgment key.
"""

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")

# Set at startup
RESULTS_PATH = None
OUTPUT_DIR = None
ANALYSIS_PATH = None


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save_results(data):
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/api/images")
def api_images():
    data = load_results()
    images = data.get("images", [])
    return jsonify([{"id": im["id"], "file_path": im.get("file_path"), "data_source": im.get("data_source")} for im in images])


@app.route("/api/annotations")
def api_annotations():
    image_id = request.args.get("image_id")
    instance_type = request.args.get("instance_type", "nCnI")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400
    data = load_results()
    anns = data.get("annotations", [])
    preds = [a for a in anns if a.get("image_id") == image_id and a.get("instance_type") == instance_type and a.get("model_name") != "gt"]
    gt_ann = next((a for a in anns if a.get("image_id") == image_id and a.get("instance_type") == instance_type and a.get("model_name") == "gt"), None)
    # Rank by final_score desc
    preds.sort(key=lambda a: a.get("final_score") or 0, reverse=True)
    return jsonify({"predictions": preds, "gt": gt_ann})


@app.route("/api/image_info/<image_id>")
def api_image_info(image_id):
    data = load_results()
    im = next((i for i in data.get("images", []) if i["id"] == image_id), None)
    if not im:
        return jsonify({"error": "image not found"}), 404
    return jsonify(im)


@app.route("/api/judge", methods=["POST"])
def api_judge():
    body = request.get_json() or {}
    ann_id = body.get("annotation_id")
    judgment = body.get("user_judgment")
    if judgment not in ("pass", "fail"):
        return jsonify({"error": "user_judgment must be 'pass' or 'fail'"}), 400
    if not ann_id:
        return jsonify({"error": "annotation_id required"}), 400
    data = load_results()
    for a in data.get("annotations", []):
        if a.get("id") == ann_id:
            a["user_judgment"] = judgment
            save_results(data)
            return jsonify({"ok": True, "id": ann_id, "user_judgment": judgment})
    return jsonify({"error": "annotation not found"}), 404


@app.route("/api/analysis")
def api_analysis():
    if not ANALYSIS_PATH or not Path(ANALYSIS_PATH).exists():
        return jsonify({"by_image_instance": []})
    with open(ANALYSIS_PATH) as f:
        data = json.load(f)
    return jsonify(data.get("by_image_instance", []))


@app.route("/files/masks/<path:filename>")
def files_masks(filename):
    if not OUTPUT_DIR:
        return "No output dir", 404
    masks_dir = Path(OUTPUT_DIR) / "masks"
    if not masks_dir.exists():
        return "Masks dir not found", 404
    return send_from_directory(masks_dir, filename)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Result filter app: compare predictions, pass/fail, realtime JSON update")
    p.add_argument("results_json", type=str, help="Path to results.json (or results_cityscapes.json)")
    p.add_argument("--analysis", type=str, default=None, help="Optional path to analysis JSON from result_analyzer")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()

    global RESULTS_PATH, OUTPUT_DIR, ANALYSIS_PATH
    RESULTS_PATH = os.path.abspath(args.results_json)
    OUTPUT_DIR = str(Path(RESULTS_PATH).parent)
    ANALYSIS_PATH = os.path.abspath(args.analysis) if args.analysis else None

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
