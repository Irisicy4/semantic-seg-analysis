#!/usr/bin/env python3
"""
Analyze results JSON: group by (image_id, instance_type), compute variance of
final_score and whether rankings of other_scores agree; output composite score
and sort so best variance (highest performance variance across predictions) is first.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Metrics in other_scores used for ranking agreement
RANK_METRICS = ("mean_iou", "mean_dice", "mean_acc", "overall_acc")


def _rank_order(values: List[float], descending: bool = True) -> List[int]:
    """Return rank (0-based) for each value; ties get same rank (min rank in group)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def ranking_agrees(anns: List[Dict[str, Any]]) -> bool:
    """
    True if ordering of annotations by final_score matches ordering by each of
    mean_iou, mean_dice, mean_acc, overall_acc (all descending = higher is better).
    """
    if len(anns) <= 1:
        return True
    scores = [a.get("final_score") for a in anns]
    if any(s is None for s in scores):
        return False
    ref_ranks = _rank_order(scores, descending=True)
    for key in RANK_METRICS:
        vals = []
        for a in anns:
            os = a.get("other_scores") or {}
            vals.append(os.get(key))
        if None in vals:
            continue
        ranks = _rank_order(vals, descending=True)
        if ranks != ref_ranks:
            return False
    return True


def variance_final_score(anns: List[Dict[str, Any]]) -> float:
    """Population variance of final_score (excluding gt)."""
    scores = [a["final_score"] for a in anns if a.get("final_score") is not None]
    if len(scores) <= 1:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    return sum((x - mean) ** 2 for x in scores) / n


def analyze_results(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load results JSON, group annotations by (image_id, instance_type),
    exclude model_name=="gt", compute variance and ranking agreement, composite, sort.
    """
    with open(json_path) as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    # Exclude GT entries so we measure variance across model×aug predictions only
    pred_anns = [a for a in annotations if a.get("model_name") != "gt"]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for a in pred_anns:
        image_id = a.get("image_id")
        instance_type = a.get("instance_type", "nCnI")
        if not image_id:
            continue
        grouped[(image_id, instance_type)].append(a)

    rows = []
    for (image_id, instance_type), anns in grouped.items():
        if len(anns) < 2:
            var = 0.0
            agrees = True
        else:
            var = variance_final_score(anns)
            agrees = ranking_agrees(anns)
        # Composite: prefer higher variance (more spread = interesting); break ties by ranking agreement
        composite = var + (0.5 if agrees else 0.0)
        rows.append({
            "image_id": image_id,
            "instance_type": instance_type,
            "variance_final_score": round(var, 8),
            "ranking_agrees": agrees,
            "composite_score": round(composite, 8),
            "n_annotations": len(anns),
        })

    # Sort: best variance in front (descending composite = variance first, then agreement)
    rows.sort(key=lambda r: (r["variance_final_score"], r["ranking_agrees"]), reverse=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="Analyze results JSON: variance and ranking agreement per image×instance_type")
    p.add_argument("results_json", type=str, help="Path to results JSON (e.g. results.json or results_cityscapes.json)")
    p.add_argument("-o", "--output", type=str, default=None, help="Output JSON path; default stdout")
    p.add_argument("--top", type=int, default=None, help="Emit only top N by composite score")
    args = p.parse_args()

    json_path = Path(args.results_json)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = analyze_results(json_path)
    if args.top is not None:
        rows = rows[: args.top]

    out = {"by_image_instance": rows}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
