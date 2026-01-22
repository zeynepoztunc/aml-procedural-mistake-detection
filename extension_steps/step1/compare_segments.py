from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _segments_from_record(record: dict[str, Any]) -> list[tuple[float, float]]:
    segs = record.get("segments") or []
    out: list[tuple[float, float]] = []
    for s in segs:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            out.append((float(s[0]), float(s[1])))
    return out


def _boundaries_from_segments(segments: list[tuple[float, float]]) -> list[float]:
    b: list[float] = []
    for s0, s1 in segments:
        b.append(float(s0))
        b.append(float(s1))
    return sorted(b)


def _boundary_f1(
    *,
    pred_boundaries: list[float],
    gt_boundaries: list[float],
    tolerance_sec: float,
) -> BoundaryMetrics:
    if not pred_boundaries and not gt_boundaries:
        return BoundaryMetrics(precision=1.0, recall=1.0, f1=1.0)
    if not pred_boundaries:
        return BoundaryMetrics(precision=0.0, recall=0.0, f1=0.0)
    if not gt_boundaries:
        return BoundaryMetrics(precision=0.0, recall=0.0, f1=0.0)

    used_gt = [False] * len(gt_boundaries)
    tp = 0
    for pb in pred_boundaries:
        best_j = -1
        best_d = float("inf")
        for j, gb in enumerate(gt_boundaries):
            if used_gt[j]:
                continue
            d = abs(pb - gb)
            if d <= tolerance_sec and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used_gt[best_j] = True
            tp += 1

    fp = len(pred_boundaries) - tp
    fn = len(gt_boundaries) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return BoundaryMetrics(precision=float(precision), recall=float(recall), f1=float(f1))


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-8, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def _greedy_match_iou(
    *,
    pred_segments: list[tuple[float, float]],
    gt_segments: list[tuple[float, float]],
) -> list[float]:
    """
    Greedy 1-1 matching by IoU (highest first). Returns matched IoUs.
    This is NOT mAP, but is a simple, interpretable overlap score.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, ps in enumerate(pred_segments):
        for j, gs in enumerate(gt_segments):
            pairs.append((_iou_1d(ps, gs), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_p = set()
    used_g = set()
    ious: list[float] = []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(float(iou))
    return ious


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Step 1 predicted segments vs GT segments.")
    p.add_argument(
        "--gt-pkl",
        default=str(Path("extension_data") / "step_embeddings_gt.pkl"),
        help="GT/oracle Step 1 output (.pkl).",
    )
    p.add_argument(
        "--pred-pkl",
        default=str(Path("extension_data") / "step_embeddings_actionformer.pkl"),
        help="Predicted Step 1 output (.pkl).",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to evaluate on (uses the GT pkl splits).",
    )
    p.add_argument(
        "--boundary-tol-sec",
        type=float,
        default=2.0,
        help="Boundary match tolerance in seconds for precision/recall/F1.",
    )
    p.add_argument(
        "--boundary-tols-sec",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional list of boundary tolerances (seconds). "
            "If provided, reports metrics for each tolerance."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gt = _load_pkl(Path(args.gt_pkl))
    pred = _load_pkl(Path(args.pred_pkl))

    gt_data: dict[str, Any] = gt.get("data") or {}
    pred_data: dict[str, Any] = pred.get("data") or {}

    splits: dict[str, Any] = gt.get("splits") or {}
    if args.split == "all":
        rec_ids = sorted(set(gt_data.keys()) & set(pred_data.keys()))
    else:
        split_ids = splits.get(args.split) or []
        rec_ids = [rid for rid in split_ids if rid in gt_data and rid in pred_data]

    if not rec_ids:
        print("No overlapping recordings to compare for the requested split.")
        return 2

    matched_ious: list[float] = []
    gt_counts: list[int] = []
    pred_counts: list[int] = []

    gt_boundaries_by_rid: dict[str, list[float]] = {}
    pred_boundaries_by_rid: dict[str, list[float]] = {}

    for rid in rec_ids:
        gt_segs = _segments_from_record(gt_data[rid])
        pr_segs = _segments_from_record(pred_data[rid])
        gt_counts.append(len(gt_segs))
        pred_counts.append(len(pr_segs))

        gt_boundaries_by_rid[rid] = _boundaries_from_segments(gt_segs)
        pred_boundaries_by_rid[rid] = _boundaries_from_segments(pr_segs)

        matched_ious.extend(_greedy_match_iou(pred_segments=pr_segs, gt_segments=gt_segs))

    tolerances = args.boundary_tols_sec if args.boundary_tols_sec else [float(args.boundary_tol_sec)]

    print("=== Step 1 segment comparison ===")
    print(f"GT:   {Path(args.gt_pkl)}")
    print(f"Pred: {Path(args.pred_pkl)}")
    print(f"Split: {args.split}  (n={len(rec_ids)})")
    print("")
    print(f"Avg #segments  GT={_mean(gt_counts):.2f}  Pred={_mean(pred_counts):.2f}")

    print("Boundary F1 (avg over recordings):")
    for tol in tolerances:
        metrics: list[BoundaryMetrics] = []
        for rid in rec_ids:
            metrics.append(
                _boundary_f1(
                    pred_boundaries=pred_boundaries_by_rid[rid],
                    gt_boundaries=gt_boundaries_by_rid[rid],
                    tolerance_sec=float(tol),
                )
            )
        print(
            f"  tol={tol:.2f}s  "
            f"P={_mean([m.precision for m in metrics]):.3f} "
            f"R={_mean([m.recall for m in metrics]):.3f} "
            f"F1={_mean([m.f1 for m in metrics]):.3f}"
        )

    print(f"Matched-segment IoU (greedy 1-1): mean={_mean(matched_ious):.3f}  n_pairs={len(matched_ious)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
