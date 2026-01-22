from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2D arrays for similarity computation.")
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


@dataclass(frozen=True)
class MatchResult:
    matches: list[tuple[int, int, float]]  # (visual_idx, text_idx, similarity)
    unmatched_visual: list[int]
    unmatched_text: list[int]


def hungarian_matching(similarity: np.ndarray) -> MatchResult:
    """Max-similarity 1–1 assignment using Hungarian algorithm (SciPy), with a greedy fallback."""
    sim = np.asarray(similarity, dtype=np.float32)
    n_vis, n_txt = sim.shape

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-sim)
        pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        # Greedy fallback: repeatedly take the best remaining similarity.
        pairs = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
        for r, c in flat.tolist():
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
            if len(used_r) == n_vis or len(used_c) == n_txt:
                break

    matches = [(r, c, float(sim[r, c])) for r, c in pairs]
    matched_r = {r for r, _, _ in matches}
    matched_c = {c for _, c, _ in matches}
    unmatched_visual = [i for i in range(n_vis) if i not in matched_r]
    unmatched_text = [j for j in range(n_txt) if j not in matched_c]

    return MatchResult(matches=matches, unmatched_visual=unmatched_visual, unmatched_text=unmatched_text)

