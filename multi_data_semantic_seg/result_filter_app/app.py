#!/usr/bin/env python3
"""
Result filter app: serve results JSON (read-only), mask images, and accept pass/fail judgments.
Judgments are stored in a separate file (e.g. judgments_cityscapes.json), not written back to results.
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
JUDGMENTS_PATH = None


def load_results():
    """Read results JSON only (no repair, no filtering). Run fix_results_json.py first if file is corrupted."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def load_judgments():
    """Load { annotation_id: 'pass'|'fail', ... } from judgments file. Read-only for results."""
    if not JUDGMENTS_PATH or not Path(JUDGMENTS_PATH).exists():
        return {}
    with open(JUDGMENTS_PATH) as f:
        return json.load(f)


def save_judgments(judgments: dict):
    """Persist judgments to the judgments file only (results JSON is never written)."""
    with open(JUDGMENTS_PATH, "w") as f:
        json.dump(judgments, f, indent=2)


@app.route("/api/images")
def api_images():
    data = load_results()
    judgments = load_judgments()
    images = data.get("images", [])
    anns = data.get("annotations", [])
    # Count passes per image: nCnI only (≥5 shows green dot), 1CnI only (≥5 shows blue dot)
    approved_nCnI_count = {}
    approved_1CnI_count = {}
    for a in anns:
        if a.get("model_name") == "gt":
            continue
        iid = a.get("image_id")
        if judgments.get(a.get("id")) != "pass":
            continue
        it = a.get("instance_type")
        if it == "1CnI":
            approved_1CnI_count[iid] = approved_1CnI_count.get(iid, 0) + 1
        else:
            approved_nCnI_count[iid] = approved_nCnI_count.get(iid, 0) + 1
    out = []
    for im in images:
        iid = im["id"]
        out.append({
            "id": iid,
            "file_path": im.get("file_path"),
            "data_source": im.get("data_source"),
            "approved_count": int(approved_nCnI_count.get(iid, 0)),
            "approved_1CnI_count": int(approved_1CnI_count.get(iid, 0)),
        })
    return jsonify(out)


@app.route("/api/palette")
def api_palette():
    """Return unique label + color_hex from annotations' predictions for the legend."""
    data = load_results()
    seen = {}
    for ann in data.get("annotations", []):
        for p in ann.get("predictions") or []:
            if isinstance(p, dict):
                label = p.get("label")
                color = p.get("color_hex")
                if label is not None and color:
                    seen[label] = color
    items = [{"label": k, "color_hex": v} for k, v in sorted(seen.items())]
    return jsonify(items)


@app.route("/api/annotations")
def api_annotations():
    image_id = request.args.get("image_id")
    instance_type = request.args.get("instance_type", "nCnI")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400
    data = load_results()
    judgments = load_judgments()
    anns = data.get("annotations", [])
    preds = [a for a in anns if a.get("image_id") == image_id and a.get("instance_type") == instance_type and a.get("model_name") != "gt"]
    gt_ann = next((a for a in anns if a.get("image_id") == image_id and a.get("instance_type") == instance_type and a.get("model_name") == "gt"), None)
    for a in preds:
        if a.get("id") in judgments:
            a["user_judgment"] = judgments[a["id"]]
    # Rank: 1CnI by overall_acc, else by final_score (desc)
    if instance_type == "1CnI":
        preds.sort(key=lambda a: (a.get("other_scores") or {}).get("overall_acc") or 0, reverse=True)
    else:
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
    if not any(a.get("id") == ann_id for a in data.get("annotations", [])):
        return jsonify({"error": "annotation not found"}), 404
    judgments = load_judgments()
    judgments[ann_id] = judgment
    save_judgments(judgments)
    return jsonify({"ok": True, "id": ann_id, "user_judgment": judgment})


@app.route("/api/save_judgments", methods=["POST"])
def api_save_judgments():
    """Force-write judgments to disk (read current file and write back) so user can ensure updates are persisted."""
    judgments = load_judgments()
    save_judgments(judgments)
    return jsonify({"ok": True, "count": len(judgments)})


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
    path = masks_dir / filename
    if path.exists():
        return send_from_directory(masks_dir, filename)
    # Pipeline saves as *_iou0.xxxx.png but JSON references *_mask.png — serve iou variant
    if filename.endswith("_mask.png"):
        prefix = filename[:-len("_mask.png")]
        for f in masks_dir.iterdir():
            if f.name.startswith(prefix + "_iou") and f.name.endswith(".png"):
                return send_from_directory(masks_dir, f.name)
    return "Not found", 404


def main():
    import argparse
    p = argparse.ArgumentParser(description="Result filter app: compare predictions, pass/fail, realtime JSON update")
    p.add_argument("results_json", type=str, help="Path to results.json (or results_cityscapes.json)")
    p.add_argument("--analysis", type=str, default=None, help="Optional path to analysis JSON from result_analyzer")
    p.add_argument("--port", type=int, default=5001, help="Port (default 5001 avoids macOS AirPlay on 5000)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind address: 127.0.0.1 = local only; 0.0.0.0 = allow LAN (others can connect)")
    args = p.parse_args()

    global RESULTS_PATH, OUTPUT_DIR, ANALYSIS_PATH, JUDGMENTS_PATH
    RESULTS_PATH = os.path.abspath(args.results_json)
    OUTPUT_DIR = str(Path(RESULTS_PATH).parent)
    ANALYSIS_PATH = os.path.abspath(args.analysis) if args.analysis else None
    # Judgments stored alongside results, e.g. results_cityscapes.json -> judgments_cityscapes.json
    base = Path(RESULTS_PATH).stem
    if base.startswith("results_"):
        JUDGMENTS_PATH = str(Path(RESULTS_PATH).parent / f"judgments_{base[8:]}.json")
    else:
        JUDGMENTS_PATH = str(Path(RESULTS_PATH).parent / "judgments.json")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
