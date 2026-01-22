from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig, Paths, TrainConfig, repo_root_from_file
from .data import StepLocalizationDataset, load_features, load_json, collate_fn
from .export import build_step_embeddings_actionformer, save_step_embeddings_pkl
from .infer import boundary_probs, segments_from_probs, predict_segments
from .model import ActionFormer, count_parameters
from .train import cosine_with_warmup, evaluate, train_one_epoch


def _set_seed(seed: int) -> None:
    # Pipeline helper: make training/inference deterministic-ish across runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    # Pipeline entry: CLI for wiring paths/hyperparameters to the end-to-end Step 1 ActionFormer run.
    parser = argparse.ArgumentParser(description="Step 1: ActionFormer step localization (refactored).")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "train+infer"],
        default="train+infer",
        help="Run mode: train only, infer only (requires checkpoint), or both.",
    )
    parser.add_argument("--egovlp-feature-dir", default=None, help="Path to data/features/egovlp")
    parser.add_argument("--annotation-path", default=None, help="Path to annotations/.../step_annotations.json")
    parser.add_argument("--split-path", default=None, help="Path to er_annotations/recordings_combined_splits.json")
    parser.add_argument("--out-dir", default="extension_data", help="Output directory (relative to repo root)")
    parser.add_argument("--out-pkl", default="step_embeddings_actionformer.pkl", help="Output pkl filename under out-dir")
    parser.add_argument(
        "--checkpoint-path",
        default=str(Path("extension_data") / "actionformer_best.pt"),
        help="Path to save/load the best model checkpoint (relative to repo root unless absolute).",
    )
    parser.add_argument("--align-to-gt", action="store_true", help="Assign description/labels by best IoU with GT steps.")

    parser.add_argument("--feature-fps", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument(
        "--loss-type",
        choices=["bce", "focal"],
        default="bce",
        help="Training loss for boundary logits (BCE-with-logits or focal loss).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (only used when --loss-type focal).",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional focal loss alpha for class balancing (only used when --loss-type focal).",
    )
    parser.add_argument("--num-epochs", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--boundary-label-mode",
        choices=["hard", "gaussian"],
        default="hard",
        help="How to build boundary labels for training (hard window vs soft Gaussian bump).",
    )
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=3,
        help="Boundary label window radius in feature frames (hard labels or Gaussian truncation window).",
    )
    parser.add_argument(
        "--boundary-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma (in feature frames) when --boundary-label-mode gaussian.",
    )

    parser.add_argument(
        "--select-best-by",
        choices=["val_f1", "score_iou_count"],
        default="val_f1",
        help="Which validation metric to use for selecting/saving the best checkpoint.",
    )
    parser.add_argument(
        "--val-seg-eval-every",
        type=int,
        default=1,
        help="Run val segmentation IoU/count evaluation every N epochs (only used when selecting by score_iou_count).",
    )
    parser.add_argument(
        "--score-count-penalty",
        type=float,
        default=0.02,
        help="Penalty weight for segment-count mismatch in score_iou_count: iou_mean - w*abs(avg_pred-avg_gt).",
    )

    parser.add_argument("--threshold", type=float, default=0.5, help="Boundary threshold for inference")
    parser.add_argument("--min-seg-len", type=int, default=15, help="Min segment length in feature frames")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average smoothing window (in feature frames) for boundary probs.",
    )
    parser.add_argument(
        "--smooth-type",
        choices=["box", "gaussian"],
        default="box",
        help="Smoothing kernel type applied before peak picking.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (in frames) when --smooth-type gaussian (default: window/6).",
    )
    parser.add_argument(
        "--peak-distance",
        type=int,
        default=0,
        help="Optional minimum distance (in feature frames) between boundary peaks (peak NMS).",
    )
    parser.add_argument(
        "--min-peak-prominence",
        type=float,
        default=0.0,
        help="Optional peak prominence threshold (probability units). Higher filters spurious boundaries.",
    )
    parser.add_argument(
        "--prominence-window",
        type=int,
        default=0,
        help="Window (in feature frames) used to estimate prominence (0 disables prominence filtering).",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["threshold", "topk"],
        default="threshold",
        help="How to convert boundary probabilities into segments.",
    )
    parser.add_argument(
        "--target-num-segments",
        type=int,
        default=None,
        help="When --segment-mode topk: target number of segments per recording (uses top-(N-1) peaks).",
    )
    parser.add_argument(
        "--min-peak-prob",
        type=float,
        default=0.0,
        help="When --segment-mode topk: ignore local maxima with prob below this value.",
    )
    parser.add_argument(
        "--max-seg-len",
        type=int,
        default=None,
        help="Optional max segment length (in feature frames). If set, long segments are split using strong peaks.",
    )
    parser.add_argument(
        "--min-split-peak-prob",
        type=float,
        default=None,
        help="When --max-seg-len is set: minimum prob for a peak to be used for splitting (default: threshold).",
    )
    return parser.parse_args()


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


@torch.no_grad()
def evaluate_val_segments(
    *,
    model: torch.nn.Module,
    recording_ids: list[str],
    annotations: dict[str, dict],
    feature_dir: Path,
    device: torch.device,
    feature_fps: float,
    infer_threshold: float,
    min_seg_len: int,
    smooth_window: int,
    smooth_type: str,
    smooth_sigma: float | None,
    peak_distance: int,
    segment_mode: str,
    target_num_segments: int | None,
    min_peak_prob: float,
    max_seg_len: int | None,
    min_split_peak_prob: float | None,
    min_peak_prominence: float,
    prominence_window: int,
    score_count_penalty: float,
) -> tuple[float, float, float, float]:
    """
    Validation metric aligned with Step 1 objectives:
    - iou_mean: greedy 1-1 IoU mean across all matched pairs
    - avg_gt_segments, avg_pred_segments
    - score_iou_count = iou_mean - w*abs(avg_pred - avg_gt)
    """
    model.eval()

    matched_ious: list[float] = []
    gt_counts: list[int] = []
    pred_counts: list[int] = []

    for rid in recording_ids:
        feats = load_features(rid, feature_dir)
        if feats is None:
            continue

        probs = boundary_probs(model, feats, device)
        seg_frames = segments_from_probs(
            probs=probs,
            threshold=float(infer_threshold),
            min_segment_len=int(min_seg_len),
            smooth_window=int(smooth_window),
            smooth_type=str(smooth_type),
            smooth_sigma=smooth_sigma,
            peak_distance=int(peak_distance),
            segment_mode=str(segment_mode),
            target_num_segments=target_num_segments,
            min_peak_prob=float(min_peak_prob),
            max_segment_len=int(max_seg_len) if max_seg_len is not None else None,
            min_split_peak_prob=min_split_peak_prob,
            min_peak_prominence=float(min_peak_prominence),
            prominence_window=int(prominence_window),
        )
        pred_segments = [(sf / float(feature_fps), ef / float(feature_fps)) for sf, ef in seg_frames]

        steps = (annotations.get(rid) or {}).get("steps") or []
        gt_segments = [(float(s["start_time"]), float(s["end_time"])) for s in steps]

        gt_counts.append(len(gt_segments))
        pred_counts.append(len(pred_segments))
        matched_ious.extend(_greedy_match_iou(pred_segments=pred_segments, gt_segments=gt_segments))

    iou_mean = _mean(matched_ious)
    avg_gt = _mean(gt_counts)
    avg_pred = _mean(pred_counts)
    score = float(iou_mean) - float(score_count_penalty) * abs(float(avg_pred) - float(avg_gt))
    return float(iou_mean), float(avg_gt), float(avg_pred), float(score)


def main() -> int:
    # Pipeline entrypoint:
    # 1) load JSON + features
    # 2) train boundary detector
    # 3) infer segments for each recording
    # 4) pool segments -> step embeddings
    # 5) save Step-2-compatible .pkl
    args = parse_args()

    repo_root = repo_root_from_file(Path(__file__))
    os.chdir(repo_root)

    paths = Paths(
        repo_root=repo_root,
        egovlp_feature_dir=Path(args.egovlp_feature_dir) if args.egovlp_feature_dir else repo_root / "data" / "features" / "egovlp",
        annotation_path=Path(args.annotation_path) if args.annotation_path else repo_root / "annotations" / "annotation_json" / "step_annotations.json",
        split_path=Path(args.split_path) if args.split_path else repo_root / "er_annotations" / "recordings_combined_splits.json",
        out_dir=(repo_root / args.out_dir),
    )
    train_cfg = TrainConfig(
        feature_fps=args.feature_fps,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
    )
    model_cfg = ModelConfig()

    _set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")

    annotations = load_json(paths.annotation_path)
    splits = load_json(paths.split_path)

    # Filter recordings to those with available features
    available = []
    for rec_id in annotations.keys():
        if load_features(rec_id, paths.egovlp_feature_dir) is not None:
            available.append(rec_id)

    train_ids = [r for r in splits["train"] if r in available]
    val_ids = [r for r in splits["val"] if r in available]
    test_ids = [r for r in splits["test"] if r in available]

    train_ds = StepLocalizationDataset(
        recording_ids=train_ids,
        annotations=annotations,
        feature_dir=paths.egovlp_feature_dir,
        max_len=train_cfg.max_len,
        feature_fps=train_cfg.feature_fps,
        boundary_window=args.boundary_window,
        boundary_label_mode=args.boundary_label_mode,
        boundary_sigma=args.boundary_sigma,
    )
    val_ds = StepLocalizationDataset(
        recording_ids=val_ids,
        annotations=annotations,
        feature_dir=paths.egovlp_feature_dir,
        max_len=train_cfg.max_len,
        feature_fps=train_cfg.feature_fps,
        boundary_window=args.boundary_window,
        boundary_label_mode=args.boundary_label_mode,
        boundary_sigma=args.boundary_sigma,
    )
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ActionFormer(
        input_dim=model_cfg.input_dim,
        embd_dim=model_cfg.embd_dim,
        n_head=model_cfg.n_head,
        arch=model_cfg.arch,
        mha_win_size=model_cfg.mha_win_size,
        scale_factor=model_cfg.scale_factor,
        fpn_dim=model_cfg.fpn_dim,
        head_dim=model_cfg.head_dim,
        num_classes=1,
        attn_pdrop=model_cfg.attn_pdrop,
        proj_pdrop=model_cfg.proj_pdrop,
        path_pdrop=model_cfg.path_pdrop,
    ).to(device)
    print(f"Model params: {count_parameters(model):,}")

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path

    if args.mode in ("train", "train+infer"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

        best_val_f1 = -1.0
        best_val_score = -1e9
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(train_cfg.num_epochs):
            lr_now = cosine_with_warmup(
                base_lr=train_cfg.lr,
                epoch=epoch,
                num_epochs=train_cfg.num_epochs,
                warmup_epochs=train_cfg.warmup_epochs,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                device=device,
                pos_weight=train_cfg.pos_weight,
                optimizer=optimizer,
                loss_type=str(args.loss_type),
                focal_gamma=float(args.focal_gamma),
                focal_alpha=args.focal_alpha,
            )
            stats = evaluate(model=model, loader=val_loader, device=device, pos_weight=train_cfg.pos_weight, threshold=0.5)
            if stats.val_f1 > best_val_f1:
                best_val_f1 = stats.val_f1

            line = (
                f"[epoch {epoch+1:02d}/{train_cfg.num_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={stats.val_loss:.4f} val_f1={stats.val_f1:.4f} lr={lr_now:.2e}"
            )

            val_score = None
            if args.select_best_by == "score_iou_count" and (epoch % max(1, int(args.val_seg_eval_every)) == 0):
                val_iou, val_avg_gt, val_avg_pred, val_score = evaluate_val_segments(
                    model=model,
                    recording_ids=val_ids,
                    annotations=annotations,
                    feature_dir=paths.egovlp_feature_dir,
                    device=device,
                    feature_fps=train_cfg.feature_fps,
                    infer_threshold=args.threshold,
                    min_seg_len=args.min_seg_len,
                    smooth_window=args.smooth_window,
                    smooth_type=args.smooth_type,
                    smooth_sigma=args.smooth_sigma,
                    peak_distance=args.peak_distance,
                    segment_mode=args.segment_mode,
                    target_num_segments=args.target_num_segments,
                    min_peak_prob=args.min_peak_prob,
                    max_seg_len=args.max_seg_len,
                    min_split_peak_prob=args.min_split_peak_prob,
                    min_peak_prominence=args.min_peak_prominence,
                    prominence_window=args.prominence_window,
                    score_count_penalty=args.score_count_penalty,
                )
                line += (
                    f" val_iou={float(val_iou):.4f} "
                    f"val_avg_segs(gt/pred)={float(val_avg_gt):.2f}/{float(val_avg_pred):.2f} "
                    f"val_score={float(val_score):.4f}"
                )

            print(line)

            if args.select_best_by == "val_f1":
                # Save the best-by-val-F1 checkpoint (ties go to the latest).
                if stats.val_f1 >= best_val_f1:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                if val_score is not None and float(val_score) > best_val_score:
                    best_val_score = float(val_score)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_cfg.__dict__,
                "train_config": train_cfg.__dict__,
                "best_val_f1": float(best_val_f1),
                "best_val_score_iou_count": float(best_val_score),
                "select_best_by": str(args.select_best_by),
            },
            str(checkpoint_path),
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    if args.mode == "infer":
        if not checkpoint_path.exists():
            raise SystemExit(
                f"Missing checkpoint: {checkpoint_path}\n"
                "Run training once (mode train or train+infer) to create it."
            )
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")

    if args.mode == "train":
        return 0

    # Inference: predict segments for all available recordings
    segments_by_rec: dict[str, list[tuple[int, int]]] = {}
    model.eval()
    for rec_id in tqdm(available, desc="infer"):
        feats = load_features(rec_id, paths.egovlp_feature_dir)
        if feats is None:
            continue
        res = predict_segments(
            model=model,
            features=feats,
            device=device,
            threshold=args.threshold,
            min_segment_len=args.min_seg_len,
            smooth_window=args.smooth_window,
            smooth_type=args.smooth_type,
            smooth_sigma=args.smooth_sigma,
            peak_distance=args.peak_distance,
            segment_mode=args.segment_mode,
            target_num_segments=args.target_num_segments,
            min_peak_prob=args.min_peak_prob,
            max_segment_len=args.max_seg_len,
            min_split_peak_prob=args.min_split_peak_prob,
            min_peak_prominence=args.min_peak_prominence,
            prominence_window=args.prominence_window,
        )
        segments_by_rec[rec_id] = res.segments_frames

    # Export embeddings in the same structure Step 2 expects
    processed = build_step_embeddings_actionformer(
        recording_ids=available,
        annotations=annotations,
        feature_dir=paths.egovlp_feature_dir,
        segments_by_rec_frames=segments_by_rec,
        feature_fps=train_cfg.feature_fps,
        align_to_gt=args.align_to_gt,
    )
    feature_dim = model_cfg.input_dim
    out_path = paths.out_dir / args.out_pkl
    save_step_embeddings_pkl(out_path=out_path, method="actionformer", feature_dim=feature_dim, data=processed, splits=splits)
    print(f"Saved: {out_path}")

    print(f"Train/val/test recordings with features: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Exported recordings: {len(processed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
