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
from .infer import predict_segments
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
    parser.add_argument("--num-epochs", type=int, default=13)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)

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
    )
    val_ds = StepLocalizationDataset(
        recording_ids=val_ids,
        annotations=annotations,
        feature_dir=paths.egovlp_feature_dir,
        max_len=train_cfg.max_len,
        feature_fps=train_cfg.feature_fps,
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

        best_f1 = -1.0
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
            )
            stats = evaluate(model=model, loader=val_loader, device=device, pos_weight=train_cfg.pos_weight, threshold=0.5)
            print(
                f"[epoch {epoch+1:02d}/{train_cfg.num_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={stats.val_loss:.4f} val_f1={stats.val_f1:.4f} lr={lr_now:.2e}"
            )
            if stats.val_f1 > best_f1:
                best_f1 = stats.val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_cfg.__dict__,
                "train_config": train_cfg.__dict__,
                "best_val_f1": float(best_f1),
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
