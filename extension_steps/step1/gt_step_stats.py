from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute GT step duration statistics from step_annotations.json.")
    p.add_argument(
        "--annotation-path",
        default=str(Path("annotations") / "annotation_json" / "step_annotations.json"),
        help="Path to step_annotations.json (relative to repo root).",
    )
    p.add_argument(
        "--feature-fps",
        type=float,
        default=0.5,
        help="Feature sampling rate used in Step 1 (seconds -> feature frames).",
    )
    p.add_argument(
        "--percentiles",
        default="5,10,25,50,75,90,95",
        help="Comma-separated percentiles to print.",
    )
    return p.parse_args()


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.2f}"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    ann_path = Path(args.annotation_path)
    if not ann_path.is_absolute():
        ann_path = repo_root / ann_path

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    durations_sec: list[float] = []
    durations_frames: list[float] = []

    for rec in data.values():
        for step in rec.get("steps", []):
            try:
                start = float(step["start_time"])
                end = float(step["end_time"])
            except Exception:
                continue
            dur = max(0.0, end - start)
            durations_sec.append(dur)
            durations_frames.append(dur * float(args.feature_fps))

    if not durations_sec:
        print("No step durations found.")
        return 2

    sec = np.array(durations_sec, dtype=np.float64)
    frames = np.array(durations_frames, dtype=np.float64)

    pct = [float(x.strip()) for x in str(args.percentiles).split(",") if x.strip()]

    print("=== GT step duration stats ===")
    print(f"Annotations: {ann_path}")
    print(f"Steps counted: {len(sec)}")
    print(f"feature_fps: {args.feature_fps}")
    print("")
    print("Duration (seconds):")
    print(f"  mean={_fmt(float(sec.mean()))}  median={_fmt(float(np.median(sec)))}  std={_fmt(float(sec.std()))}")
    for p in pct:
        print(f"  p{int(p):02d}={_fmt(float(np.percentile(sec, p)))}")
    print("")
    print("Implied length (feature frames = seconds * feature_fps):")
    print(f"  mean={_fmt(float(frames.mean()))}  median={_fmt(float(np.median(frames)))}  std={_fmt(float(frames.std()))}")
    for p in pct:
        print(f"  p{int(p):02d}={_fmt(float(np.percentile(frames, p)))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

