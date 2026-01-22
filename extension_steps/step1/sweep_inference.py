from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

def _ensure_import_root() -> Path:
    # Allow running directly:
    #   python extension_steps/step1/sweep_inference.py ...
    pkg_root = Path(__file__).resolve().parents[2]  # .../aml-procedural-mistake-detection
    sys.path.insert(0, str(pkg_root))
    return pkg_root


REPO_ROOT = _ensure_import_root()

from extension_steps.step1.actionformer.data import load_features  # noqa: E402
from extension_steps.step1.actionformer.infer import boundary_probs, segments_from_probs  # noqa: E402
from extension_steps.step1.actionformer.model import ActionFormer  # noqa: E402


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
    if not pred_boundaries or not gt_boundaries:
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


def _parse_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Step 1 inference parameters and write a CSV summary.")
    p.add_argument(
        "--checkpoint-path",
        default=str(Path("extension_data") / "actionformer_best.pt"),
        help="Checkpoint produced by `extension_steps/step1/actionformer.py`.",
    )
    p.add_argument(
        "--gt-pkl",
        default=str(Path("extension_data") / "step_embeddings_gt.pkl"),
        help="GT/oracle Step 1 output (.pkl) used for ground-truth segments + split ids.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="val",
        help="Which split to tune against (recommend: val).",
    )
    p.add_argument(
        "--egovlp-feature-dir",
        default=str(Path("data") / "features" / "egovlp"),
        help="Directory containing EgoVLP feature .npz files.",
    )
    p.add_argument(
        "--feature-fps",
        type=float,
        default=0.5,
        help="Feature sampling rate used to convert indices -> seconds.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        default=["0.3", "0.4", "0.5", "0.6", "0.7"],
        help="Boundary probability thresholds to sweep (space or comma separated).",
    )
    p.add_argument(
        "--min-seg-lens",
        nargs="+",
        default=["8", "15", "25"],
        help="Minimum segment length in feature frames (space or comma separated).",
    )
    p.add_argument(
        "--smooth-windows",
        nargs="+",
        default=["1"],
        help="Smoothing window sizes (feature frames) to sweep (space or comma separated). Use 1 to disable.",
    )
    p.add_argument(
        "--smooth-types",
        nargs="+",
        default=["box"],
        choices=["box", "gaussian"],
        help="Smoothing types to sweep.",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (frames) when smooth-type is gaussian (default: window/6).",
    )
    p.add_argument(
        "--peak-distances",
        nargs="+",
        default=["0"],
        help="Peak-distance values (feature frames) to sweep (space or comma separated). Use 0 to disable.",
    )
    p.add_argument(
        "--segment-modes",
        nargs="+",
        default=["threshold"],
        choices=["threshold", "topk"],
        help="Segmentation modes to sweep (thresholded peaks vs top-k peaks).",
    )
    p.add_argument(
        "--target-num-segments",
        type=int,
        default=None,
        help="When segment-mode is topk: target number of segments per recording.",
    )
    p.add_argument(
        "--min-peak-prob",
        type=float,
        default=0.0,
        help="When segment-mode is topk: ignore local maxima below this prob.",
    )
    p.add_argument(
        "--min-peak-prominences",
        nargs="+",
        default=["0"],
        help="Optional peak-prominence thresholds to sweep (probability units). Use 0 to disable.",
    )
    p.add_argument(
        "--prominence-windows",
        nargs="+",
        default=["0"],
        help="Prominence window sizes (feature frames) to sweep. Use 0 to disable prominence filtering.",
    )
    p.add_argument(
        "--max-seg-lens",
        nargs="+",
        default=["0"],
        help="Optional max segment length values (feature frames) to sweep. Use 0 to disable.",
    )
    p.add_argument(
        "--min-split-peak-prob",
        type=float,
        default=None,
        help="When max-seg-len is enabled: minimum peak prob used to split long segments (default: threshold).",
    )
    p.add_argument(
        "--tolerances",
        nargs="+",
        default=["1", "2", "5"],
        help="Boundary tolerance values (seconds) to report.",
    )
    p.add_argument(
        "--drop-first-last-boundary",
        action="store_true",
        help="Drop the first/last boundaries (0 and end) from predictions before boundary metrics.",
    )
    p.add_argument(
        "--score-target-avg-segments",
        type=float,
        default=None,
        help="Target avg #segments for the composite score (default: use avg_gt_segments from the evaluation split).",
    )
    p.add_argument(
        "--score-count-penalty",
        type=float,
        default=0.02,
        help="Penalty weight for segment count mismatch in the composite score: score = iou_mean - w*abs(avg_pred-avg_gt).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Print the top-N configurations by the chosen metric after writing CSV.",
    )
    p.add_argument(
        "--sort-metric",
        default="iou_mean",
        help="Metric column to sort by for the console top-N summary (e.g., iou_mean, f1@2s, f1@5s).",
    )
    p.add_argument(
        "--out-csv",
        default=str(Path("extension_data") / "sweeps" / "inference_sweep.csv"),
        help="CSV output path.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.chdir(REPO_ROOT)

    thresholds = _parse_float_list(args.thresholds)
    min_seg_lens = _parse_int_list(args.min_seg_lens)
    smooth_windows = _parse_int_list(args.smooth_windows)
    smooth_types = list(args.smooth_types)
    peak_distances = _parse_int_list(args.peak_distances)
    segment_modes = list(args.segment_modes)
    min_peak_prominences = _parse_float_list(args.min_peak_prominences)
    prominence_windows = _parse_int_list(args.prominence_windows)
    max_seg_lens = _parse_int_list(args.max_seg_lens)
    tolerances = _parse_float_list(args.tolerances)

    gt = _load_pkl(Path(args.gt_pkl))
    gt_data: dict[str, Any] = gt.get("data") or {}
    splits: dict[str, Any] = gt.get("splits") or {}

    if args.split == "all":
        rec_ids = sorted(gt_data.keys())
    else:
        split_ids = splits.get(args.split) or []
        rec_ids = [rid for rid in split_ids if rid in gt_data]

    if not rec_ids:
        print("No recordings found for requested split in GT pkl.")
        return 2

    egovlp_dir = Path(args.egovlp_feature_dir)
    if not egovlp_dir.is_absolute():
        egovlp_dir = REPO_ROOT / egovlp_dir

    # Pre-load GT segments/boundaries for speed.
    gt_segments_by_rid: dict[str, list[tuple[float, float]]] = {}
    gt_boundaries_by_rid: dict[str, list[float]] = {}
    for rid in rec_ids:
        segs = _segments_from_record(gt_data[rid])
        gt_segments_by_rid[rid] = segs
        gt_boundaries_by_rid[rid] = _boundaries_from_segments(segs)

    # Load model checkpoint once, compute boundary probabilities once per recording.
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActionFormer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probs_by_rid: dict[str, np.ndarray] = {}
    missing_features: list[str] = []
    for rid in tqdm(rec_ids, desc="computing boundary probs"):
        feats = load_features(rid, egovlp_dir)
        if feats is None:
            missing_features.append(rid)
            continue
        probs_by_rid[rid] = boundary_probs(model, feats, device)

    eval_ids = [rid for rid in rec_ids if rid in probs_by_rid]
    if not eval_ids:
        print("No recordings with features found for the requested split.")
        return 2

    out_path = Path(args.out_csv)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "n_recordings",
        "segment_mode",
        "target_num_segments",
        "min_peak_prob",
        "min_peak_prominence",
        "prominence_window",
        "threshold",
        "min_seg_len",
        "smooth_window",
        "smooth_type",
        "smooth_sigma",
        "peak_distance",
        "max_seg_len",
        "min_split_peak_prob",
        "avg_gt_segments",
        "avg_pred_segments",
        "iou_mean",
        "iou_pairs",
        "score_iou_count",
        "score_target_avg_segments",
        "score_count_penalty",
    ]
    for tol in tolerances:
        fieldnames.extend([f"p@{tol:g}s", f"r@{tol:g}s", f"f1@{tol:g}s"])

    rows_written: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for thr in thresholds:
            for min_len in min_seg_lens:
                for sm in smooth_windows:
                    for st in smooth_types:
                        for pd in peak_distances:
                            for max_len in max_seg_lens:
                                for prom in min_peak_prominences:
                                    for prom_w in prominence_windows:
                                        for mode in segment_modes:
                                            boundary_metrics_by_tol: dict[float, list[BoundaryMetrics]] = {
                                                tol: [] for tol in tolerances
                                            }
                                            matched_ious: list[float] = []
                                            gt_counts: list[int] = []
                                            pred_counts: list[int] = []

                                            for rid in eval_ids:
                                                probs = probs_by_rid[rid]
                                                seg_frames = segments_from_probs(
                                                    probs=probs,
                                                    threshold=float(thr),
                                                    min_segment_len=int(min_len),
                                                    smooth_window=int(sm),
                                                    smooth_type=str(st),
                                                    smooth_sigma=args.smooth_sigma,
                                                    peak_distance=int(pd),
                                                    segment_mode=str(mode),
                                                    target_num_segments=args.target_num_segments,
                                                    min_peak_prob=float(args.min_peak_prob),
                                                    max_segment_len=int(max_len) if int(max_len) > 0 else None,
                                                    min_split_peak_prob=args.min_split_peak_prob,
                                                    min_peak_prominence=float(prom),
                                                    prominence_window=int(prom_w),
                                                )
                                                pred_segments = [
                                                    (sf / float(args.feature_fps), ef / float(args.feature_fps))
                                                    for sf, ef in seg_frames
                                                ]

                                                gt_segments = gt_segments_by_rid[rid]
                                                gt_counts.append(len(gt_segments))
                                                pred_counts.append(len(pred_segments))

                                                pred_boundaries = _boundaries_from_segments(pred_segments)
                                                if args.drop_first_last_boundary and len(pred_boundaries) >= 2:
                                                    pred_boundaries = pred_boundaries[1:-1]

                                                for tol in tolerances:
                                                    boundary_metrics_by_tol[tol].append(
                                                        _boundary_f1(
                                                            pred_boundaries=pred_boundaries,
                                                            gt_boundaries=gt_boundaries_by_rid[rid],
                                                            tolerance_sec=float(tol),
                                                        )
                                                    )

                                                matched_ious.extend(
                                                    _greedy_match_iou(pred_segments=pred_segments, gt_segments=gt_segments)
                                                )

                                            row: dict[str, Any] = {
                                                "split": args.split,
                                                "n_recordings": len(eval_ids),
                                                "segment_mode": str(mode),
                                                "target_num_segments": int(args.target_num_segments)
                                                if args.target_num_segments is not None
                                                else "",
                                                "min_peak_prob": float(args.min_peak_prob),
                                                "min_peak_prominence": float(prom),
                                                "prominence_window": int(prom_w),
                                                "threshold": float(thr),
                                                "min_seg_len": int(min_len),
                                                "smooth_window": int(sm),
                                                "smooth_type": str(st),
                                                "smooth_sigma": float(args.smooth_sigma) if args.smooth_sigma is not None else "",
                                                "peak_distance": int(pd),
                                                "max_seg_len": int(max_len) if int(max_len) > 0 else "",
                                                "min_split_peak_prob": float(args.min_split_peak_prob)
                                                if args.min_split_peak_prob is not None
                                                else "",
                                                "avg_gt_segments": _mean(gt_counts),
                                                "avg_pred_segments": _mean(pred_counts),
                                                "iou_mean": _mean(matched_ious),
                                                "iou_pairs": len(matched_ious),
                                            }
                                            avg_gt = float(row["avg_gt_segments"])
                                            avg_pred = float(row["avg_pred_segments"])
                                            score_target = (
                                                float(args.score_target_avg_segments)
                                                if args.score_target_avg_segments is not None
                                                else avg_gt
                                            )
                                            w = float(args.score_count_penalty)
                                            row["score_iou_count"] = float(row["iou_mean"]) - w * abs(avg_pred - score_target)
                                            row["score_target_avg_segments"] = score_target
                                            row["score_count_penalty"] = w
                                            for tol in tolerances:
                                                m = boundary_metrics_by_tol[tol]
                                                row[f"p@{tol:g}s"] = _mean([x.precision for x in m])
                                                row[f"r@{tol:g}s"] = _mean([x.recall for x in m])
                                                row[f"f1@{tol:g}s"] = _mean([x.f1 for x in m])

                                            writer.writerow(row)
                                            rows_written.append(row)

    print(f"Wrote: {out_path}")
    if missing_features:
        print(f"Skipped {len(missing_features)} recordings missing features.")

    # Console summary: top-N rows by a metric.
    sort_key = str(args.sort_metric)
    if rows_written and sort_key in rows_written[0]:
        top_n = max(0, int(args.top_n))
        rows_sorted = sorted(rows_written, key=lambda r: float(r.get(sort_key, -1e9)), reverse=True)
        if top_n:
            print(f"\nTop {top_n} by {sort_key}:")
            for r in rows_sorted[:top_n]:
                maybe_f1_2s = ""
                if "f1@2s" in r:
                    maybe_f1_2s = f" f1@2s={float(r['f1@2s']):.4f}"
                # Avoid printing the same metric twice when sorting by IoU.
                maybe_iou = ""
                if sort_key != "iou_mean" and "iou_mean" in r:
                    maybe_iou = f" iou_mean={float(r['iou_mean']):.4f}"
                maybe_score = ""
                if sort_key != "score_iou_count" and "score_iou_count" in r:
                    maybe_score = f" score={float(r['score_iou_count']):.4f}"
                print(
                    f"  thr={r['threshold']:.3g} min={int(r['min_seg_len'])} "
                    f"mode={r['segment_mode']} sm={int(r['smooth_window'])}({r['smooth_type']}) pd={int(r['peak_distance'])} "
                    f"prom={float(r.get('min_peak_prominence', 0.0)):.3g}@{int(r.get('prominence_window', 0))} "
                    f"max={r.get('max_seg_len', '')} "
                    f"{sort_key}={float(r[sort_key]):.4f}{maybe_iou}{maybe_score}{maybe_f1_2s} avg_pred={float(r['avg_pred_segments']):.2f}"
                )
    else:
        print(f"\nNote: sort metric '{sort_key}' not found in CSV columns.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
