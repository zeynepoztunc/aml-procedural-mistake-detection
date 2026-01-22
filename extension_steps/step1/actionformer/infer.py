from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SegmentResult:
    segments_frames: list[tuple[int, int]]
    probs: np.ndarray


def smooth_probs(probs: np.ndarray, window: int) -> np.ndarray:
    # Pipeline stage (inference): optional smoothing to reduce jittery peak picking.
    if window <= 1:
        return probs.astype(np.float32, copy=False)
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(probs.astype(np.float32, copy=False), kernel, mode="same")


def smooth_probs_gaussian(probs: np.ndarray, window: int, *, sigma: float | None = None) -> np.ndarray:
    # Pipeline stage (inference): Gaussian smoothing (often preserves peak location better than box smoothing).
    if window <= 1:
        return probs.astype(np.float32, copy=False)
    w = int(window)
    if w % 2 == 0:
        w += 1
    # If not provided, a common heuristic is sigma ~= window / 6 (covers ~+-3σ inside the window).
    sig = float(sigma) if sigma is not None else float(w) / 6.0
    sig = max(1e-6, sig)
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    kernel = np.exp(-0.5 * (x / sig) ** 2).astype(np.float32)
    kernel = kernel / float(kernel.sum() + 1e-8)
    return np.convolve(probs.astype(np.float32, copy=False), kernel, mode="same")


def _peak_prominence_fixed_window(probs: np.ndarray, idx: int, window: int) -> float:
    # Pipeline helper: approximate "peak prominence" using a fixed left/right window.
    #
    # True prominence searches until a higher peak is reached (SciPy has an implementation), but for our
    # use case a fixed-window baseline works well: if a peak does not rise above nearby valleys, it is
    # likely a spurious boundary.
    if window <= 0:
        return float("inf")
    i = int(idx)
    w = int(window)
    left0 = max(0, i - w)
    right1 = min(len(probs), i + w + 1)
    if i <= 0 or i >= len(probs) - 1:
        return 0.0

    left_slice = probs[left0:i]
    right_slice = probs[i + 1 : right1]
    if left_slice.size == 0 or right_slice.size == 0:
        return 0.0

    left_min = float(np.min(left_slice))
    right_min = float(np.min(right_slice))
    baseline = max(left_min, right_min)
    return float(probs[i]) - baseline


def select_peaks(
    *,
    probs: np.ndarray,
    threshold: float,
    min_distance: int,
    min_prominence: float = 0.0,
    prominence_window: int = 0,
) -> list[int]:
    # Pipeline stage (inference): pick boundary peak indices using local maxima + simple NMS by distance.
    if probs.size == 0:
        return []

    # Local maxima above threshold (exclude endpoints).
    candidates: list[int] = []
    for i in range(1, len(probs) - 1):
        if probs[i] > threshold and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
            if float(min_prominence) > 0.0:
                prom = _peak_prominence_fixed_window(probs, i, int(prominence_window))
                if prom < float(min_prominence):
                    continue
            candidates.append(i)

    if not candidates:
        return []

    # Non-maximum suppression on peaks using a minimum distance in indices.
    if min_distance <= 0:
        return sorted(candidates)

    order = sorted(candidates, key=lambda i: float(probs[i]), reverse=True)
    keep: list[int] = []
    for idx in order:
        if all(abs(idx - k) >= min_distance for k in keep):
            keep.append(idx)
    return sorted(keep)


def _enforce_min_segment_len(
    *,
    boundaries: list[int],
    end: int,
    min_segment_len: int,
    boundary_scores: dict[int, float],
) -> list[int]:
    if min_segment_len <= 0:
        return boundaries

    b = [0, *[x for x in boundaries if 0 < x < end], end]
    b = sorted(set(int(x) for x in b))
    if len(b) < 2:
        return [0, end]

    # Endpoints should never be removed.
    boundary_scores = dict(boundary_scores)
    boundary_scores[0] = float("inf")
    boundary_scores[end] = float("inf")

    # Iteratively remove boundaries that create too-short segments.
    changed = True
    while changed and len(b) > 2:
        changed = False
        gaps = [b[i + 1] - b[i] for i in range(len(b) - 1)]
        min_gap = min(gaps) if gaps else min_segment_len
        if min_gap >= min_segment_len:
            break
        i = gaps.index(min_gap)
        left = b[i]
        right = b[i + 1]
        # Remove the weaker boundary among the two that define the short segment.
        # If one is an endpoint, remove the other.
        if left == 0:
            drop = right
        elif right == end:
            drop = left
        else:
            drop = left if boundary_scores.get(left, 0.0) <= boundary_scores.get(right, 0.0) else right
        if drop in (0, end):
            break
        b.remove(drop)
        changed = True

    return b


def _enforce_max_segment_len(
    *,
    boundaries: list[int],
    end: int,
    max_segment_len: int,
    min_segment_len: int,
    candidate_peaks: list[int],
    peak_scores: dict[int, float],
    peak_distance: int,
) -> list[int]:
    """
    Pipeline stage (inference): optional duration prior.

    If any segment becomes too long, try to insert additional boundaries by choosing the strongest
    local-maximum peak within that segment, respecting `min_segment_len` and optional `peak_distance`.
    """
    if max_segment_len <= 0:
        return boundaries

    b = [0, *[x for x in boundaries if 0 < x < end], end]
    b = sorted(set(int(x) for x in b))
    if len(b) < 2:
        return [0, end]

    while True:
        lengths = [b[i + 1] - b[i] for i in range(len(b) - 1)]
        if not lengths:
            break
        worst_len = max(lengths)
        if worst_len <= max_segment_len:
            break

        i = lengths.index(worst_len)
        s = b[i]
        e = b[i + 1]

        lo = s + max(1, int(min_segment_len))
        hi = e - max(1, int(min_segment_len))
        if hi <= lo:
            break

        best_idx = -1
        best_score = -1.0
        for p in candidate_peaks:
            if p <= lo or p >= hi:
                continue
            if peak_distance > 0 and any(abs(p - bb) < peak_distance for bb in b if bb not in (0, end)):
                continue
            score = float(peak_scores.get(int(p), 0.0))
            if score > best_score:
                best_score = score
                best_idx = int(p)

        if best_idx < 0:
            break

        b.append(best_idx)
        b = sorted(set(b))

    return b


@torch.no_grad()
def boundary_probs(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    # Pipeline stage (inference): run model on one recording and return per-timestep boundary probabilities.
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, C)
    out_cls, _, _, _ = model(x)
    cls_logits = out_cls[0].squeeze(0).squeeze(0)  # (T')
    probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
    if probs.shape[0] != features.shape[0]:
        # 1D linear interpolation to match original length (avoid SciPy dependency).
        src_x = np.linspace(0.0, 1.0, num=probs.shape[0], endpoint=True)
        dst_x = np.linspace(0.0, 1.0, num=features.shape[0], endpoint=True)
        probs = np.interp(dst_x, src_x, probs).astype(np.float32)
    return probs


def segments_from_probs(
    *,
    probs: np.ndarray,
    threshold: float,
    min_segment_len: int,
    smooth_window: int = 1,
    smooth_type: str = "box",
    smooth_sigma: float | None = None,
    peak_distance: int = 0,
    segment_mode: str = "threshold",
    target_num_segments: int | None = None,
    min_peak_prob: float = 0.0,
    max_segment_len: int | None = None,
    min_split_peak_prob: float | None = None,
    min_peak_prominence: float = 0.0,
    prominence_window: int = 0,
) -> list[tuple[int, int]]:
    # Pipeline stage (inference): convert boundary probabilities into segments using smoothing + peak NMS.
    if probs.size == 0:
        return []

    if smooth_type == "gaussian":
        probs_s = smooth_probs_gaussian(probs, smooth_window, sigma=smooth_sigma)
    else:
        probs_s = smooth_probs(probs, smooth_window)

    end = int(len(probs_s))
    boundary_scores: dict[int, float] = {}

    # Candidate local maxima used for optional max-seg-length splitting (duration prior).
    split_gate = float(min_split_peak_prob) if min_split_peak_prob is not None else float(threshold)
    candidate_peaks: list[int] = []
    for i in range(1, len(probs_s) - 1):
        if probs_s[i] >= split_gate and probs_s[i] >= probs_s[i - 1] and probs_s[i] >= probs_s[i + 1]:
            if float(min_peak_prominence) > 0.0:
                prom = _peak_prominence_fixed_window(probs_s, i, int(prominence_window))
                if prom < float(min_peak_prominence):
                    continue
            candidate_peaks.append(int(i))
            boundary_scores[int(i)] = float(probs_s[int(i)])

    if segment_mode == "topk":
        # Local maxima candidates (optionally gated by min_peak_prob)
        candidates: list[int] = []
        for i in range(1, len(probs_s) - 1):
            if probs_s[i] >= min_peak_prob and probs_s[i] >= probs_s[i - 1] and probs_s[i] >= probs_s[i + 1]:
                if float(min_peak_prominence) > 0.0:
                    prom = _peak_prominence_fixed_window(probs_s, i, int(prominence_window))
                    if prom < float(min_peak_prominence):
                        continue
                candidates.append(i)
                boundary_scores[i] = float(probs_s[i])

        k = max(0, int(target_num_segments or 0) - 1)
        if k <= 0 or not candidates:
            boundaries = [0, end]
        else:
            # Pick the top-k peaks by score, applying a distance-based NMS.
            order = sorted(candidates, key=lambda i: float(probs_s[i]), reverse=True)
            keep: list[int] = []
            for idx in order:
                if len(keep) >= k:
                    break
                if peak_distance > 0 and any(abs(idx - j) < peak_distance for j in keep):
                    continue
                keep.append(int(idx))
            boundaries = _enforce_min_segment_len(
                boundaries=keep,
                end=end,
                min_segment_len=min_segment_len,
                boundary_scores=boundary_scores,
            )
    else:
        # Default: thresholded local maxima + optional peak-distance NMS (select_peaks).
        peaks = select_peaks(
            probs=probs_s,
            threshold=threshold,
            min_distance=peak_distance,
            min_prominence=float(min_peak_prominence),
            prominence_window=int(prominence_window),
        )
        for p in peaks:
            boundary_scores[int(p)] = float(probs_s[int(p)])
        boundaries = _enforce_min_segment_len(
            boundaries=peaks,
            end=end,
            min_segment_len=min_segment_len,
            boundary_scores=boundary_scores,
        )

    if max_segment_len is not None and int(max_segment_len) > 0:
        boundaries = _enforce_max_segment_len(
            boundaries=[x for x in boundaries if x not in (0, end)],
            end=end,
            max_segment_len=int(max_segment_len),
            min_segment_len=int(min_segment_len),
            candidate_peaks=candidate_peaks,
            peak_scores=boundary_scores,
            peak_distance=int(peak_distance),
        )

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def predict_segments(
    *,
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    min_segment_len: int = 15,
    smooth_window: int = 1,
    smooth_type: str = "box",
    smooth_sigma: float | None = None,
    peak_distance: int = 0,
    segment_mode: str = "threshold",
    target_num_segments: int | None = None,
    min_peak_prob: float = 0.0,
    max_segment_len: int | None = None,
    min_split_peak_prob: float | None = None,
    min_peak_prominence: float = 0.0,
    prominence_window: int = 0,
) -> SegmentResult:
    # Pipeline stage (inference): convenience wrapper = boundary_probs(...) + segments_from_probs(...).
    probs = boundary_probs(model, features, device)
    segs = segments_from_probs(
        probs=probs,
        threshold=threshold,
        min_segment_len=min_segment_len,
        smooth_window=smooth_window,
        smooth_type=smooth_type,
        smooth_sigma=smooth_sigma,
        peak_distance=peak_distance,
        segment_mode=segment_mode,
        target_num_segments=target_num_segments,
        min_peak_prob=min_peak_prob,
        max_segment_len=max_segment_len,
        min_split_peak_prob=min_split_peak_prob,
        min_peak_prominence=min_peak_prominence,
        prominence_window=prominence_window,
    )
    return SegmentResult(segments_frames=segs, probs=probs)


def frames_to_time(frame: int, fps: float) -> float:
    # Pipeline helper: convert segment frame indices to seconds using feature FPS.
    return float(frame) / float(fps)
