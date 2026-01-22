from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _ensure_import_root() -> None:
    # Allow running directly:
    #   python extension_steps/step3/summarize_realized_graphs.py
    pkg_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(pkg_root))


_ensure_import_root()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Step 3 realized-task-graph matching statistics.")
    p.add_argument(
        "--pkl",
        type=str,
        default="extension_data/realized_task_graphs.pkl",
        help="Path to realized_task_graphs.pkl",
    )
    p.add_argument("--top-n", type=int, default=10, help="Top-N worst/best examples to print")
    p.add_argument(
        "--zero-eps",
        type=float,
        default=1e-12,
        help="Treat min_sim <= eps as 'zero' for the min_sim==0 diagnostic",
    )
    return p.parse_args(argv)


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _summarize(payload: dict, *, top_n: int, zero_eps: float) -> None:
    realized_graphs: dict[str, dict] = payload.get("realized_graphs", {})
    if not realized_graphs:
        raise ValueError("No realized_graphs found in pickle payload.")

    rows: list[dict[str, object]] = []
    for rid, g in realized_graphs.items():
        sims = np.asarray(g.get("match_similarities", []), dtype=float)
        node_matched = g.get("node_matched", None)
        if node_matched is None:
            node_matched_arr = None
        else:
            node_matched_arr = np.asarray(node_matched).astype(bool)

        num_nodes = int(g.get("num_nodes") or (len(node_matched_arr) if node_matched_arr is not None else sims.size))
        num_valid_matches = int(g.get("num_valid_matches") or (int(node_matched_arr.sum()) if node_matched_arr is not None else num_nodes))
        num_unmatched = int(num_nodes - num_valid_matches)

        mean_sim = _safe_float(sims.mean()) if sims.size else float("nan")
        min_sim = _safe_float(sims.min()) if sims.size else float("nan")
        max_sim = _safe_float(sims.max()) if sims.size else float("nan")

        rows.append(
            {
                "rid": str(rid),
                "label": int(g.get("recipe_label", 0)),
                "num_nodes": num_nodes,
                "num_valid_matches": num_valid_matches,
                "num_unmatched": num_unmatched,
                "mean_sim": mean_sim,
                "min_sim": min_sim,
                "max_sim": max_sim,
                "min_is_zero": bool(np.isfinite(min_sim) and min_sim <= float(zero_eps)),
            }
        )

    def _avg(key: str, *, label: int | None = None) -> float:
        vals = [r[key] for r in rows if label is None or r["label"] == label]
        vals_f = [float(v) for v in vals if np.isfinite(float(v))]
        return float(np.mean(vals_f)) if vals_f else float("nan")

    def _frac(key: str, *, label: int | None = None) -> float:
        vals = [r[key] for r in rows if label is None or r["label"] == label]
        if not vals:
            return float("nan")
        return float(np.mean([bool(v) for v in vals]))

    def _sum(key: str, *, label: int | None = None) -> int:
        vals = [r[key] for r in rows if label is None or r["label"] == label]
        return int(np.sum([int(v) for v in vals]))

    n = len(rows)
    n_err = sum(1 for r in rows if r["label"] == 1)
    n_ok = n - n_err
    cfg = payload.get("config", {})

    print("=== Step 3 realized graphs summary ===")
    print(f"pkl config: {cfg}")
    print(f"n_recordings={n}  error={n_err}  no_error={n_ok}")
    print("")

    print("Similarity stats (all recordings):")
    print(f"  mean(mean_sim)={_avg('mean_sim'):.4f}  mean(min_sim)={_avg('min_sim'):.4f}")
    print(f"  total_unmatched_nodes={_sum('num_unmatched')}  mean_unmatched_per_rec={_avg('num_unmatched'):.2f}")
    print("")

    print("Similarity stats split by label:")
    print(f"  mean(min_sim) error=1: {_avg('min_sim', label=1):.4f}  error=0: {_avg('min_sim', label=0):.4f}")
    print(
        f"  frac(min_sim<=eps) error=1: {_frac('min_is_zero', label=1):.4f}  error=0: {_frac('min_is_zero', label=0):.4f} (eps={zero_eps})"
    )
    print("")

    worst_by_mean = sorted(rows, key=lambda r: float(r["mean_sim"]))[: top_n]
    worst_by_min = sorted(rows, key=lambda r: float(r["min_sim"]))[: top_n]
    worst_by_unmatched = sorted(rows, key=lambda r: int(r["num_unmatched"]), reverse=True)[: top_n]

    def _print_rows(title: str, items: list[dict[str, object]]) -> None:
        print(title)
        for r in items:
            print(
                f"  {r['rid']}: label={r['label']} nodes={r['num_nodes']} valid={r['num_valid_matches']} "
                f"unmatched={r['num_unmatched']} min/mean/max={float(r['min_sim']):.3f}/{float(r['mean_sim']):.3f}/{float(r['max_sim']):.3f}"
            )
        print("")

    _print_rows(f"Worst {top_n} by mean_sim:", worst_by_mean)
    _print_rows(f"Worst {top_n} by min_sim:", worst_by_min)
    _print_rows(f"Top {top_n} by #unmatched nodes:", worst_by_unmatched)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pkl_path = Path(args.pkl)
    payload = pickle.load(open(pkl_path, "rb"))
    _summarize(payload, top_n=int(args.top_n), zero_eps=float(args.zero_eps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

