from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from extension_steps.step3.matching import cosine_similarity_matrix, hungarian_matching


@dataclass(frozen=True)
class RealizeConfig:
    alpha_visual: float = 0.5
    min_match_similarity: float = 0.0
    unmatched_node_features: str = "text"  # "text" or "zero"


def create_realized_graph(
    *,
    visual_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    edges: list[dict[str, int]],
    cfg: RealizeConfig = RealizeConfig(),
) -> dict:
    """Create a realized task graph by matching visual steps to graph nodes."""
    visual_emb = np.asarray(visual_embeddings, dtype=np.float32)
    text_emb = np.asarray(text_embeddings, dtype=np.float32)

    if visual_emb.ndim != 2 or text_emb.ndim != 2:
        raise ValueError("Expected 2D arrays for visual/text embeddings.")
    if visual_emb.shape[1] != text_emb.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: visual {visual_emb.shape[1]} vs text {text_emb.shape[1]}"
        )

    sim = cosine_similarity_matrix(visual_emb, text_emb)
    m = hungarian_matching(sim)

    num_nodes = text_emb.shape[0]
    node_features = np.copy(text_emb)
    node_matched = np.zeros(num_nodes, dtype=bool)
    match_similarities = np.zeros(num_nodes, dtype=np.float32)

    a = float(cfg.alpha_visual)
    a = min(max(a, 0.0), 1.0)
    min_sim = float(cfg.min_match_similarity)

    unmatched_mode = str(cfg.unmatched_node_features).lower().strip()
    if unmatched_mode not in ("text", "zero"):
        raise ValueError(f"unmatched_node_features must be 'text' or 'zero' (got {cfg.unmatched_node_features!r})")
    if unmatched_mode == "zero":
        node_features[:] = 0.0

    for v_idx, t_idx, s in m.matches:
        # Always record the similarity for analysis.
        match_similarities[t_idx] = np.float32(s)

        # But only treat this as a "valid" match if it clears a minimum similarity threshold.
        # This avoids forced low-quality assignments from Hungarian being counted as matched nodes.
        if float(s) >= min_sim:
            node_matched[t_idx] = True
            node_features[t_idx] = a * visual_emb[v_idx] + (1.0 - a) * text_emb[t_idx]

    if len(edges) > 0:
        edge_src = [int(e["from"]) for e in edges]
        edge_dst = [int(e["to"]) for e in edges]
        edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "node_matched": node_matched,
        "match_similarities": match_similarities,
        "num_matches": len(m.matches),
        "num_valid_matches": int(node_matched.sum()),
        "num_nodes": int(num_nodes),
        "num_visual_steps": int(visual_emb.shape[0]),
    }
