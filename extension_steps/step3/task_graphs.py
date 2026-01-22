from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskGraph:
    recipe_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, int]]


def build_task_graphs_from_step_annotations(step_annotations: dict[str, Any]) -> dict[str, TaskGraph]:
    """Build simple linear task-graph templates from step annotations.

    Each recipe type (recipe_id) has a canonical ordered set of steps (by step_id).
    """
    recipe_steps: dict[str, dict[int, str]] = defaultdict(dict)

    for recording_id, ann in step_annotations.items():
        recipe_id = str(recording_id).split("_")[0] if "_" in str(recording_id) else "unknown"

        for step in ann.get("steps", []):
            step_id = step.get("step_id")
            description = step.get("description", "")
            if step_id is None or not description:
                continue
            if int(step_id) not in recipe_steps[recipe_id]:
                recipe_steps[recipe_id][int(step_id)] = str(description)

    graphs: dict[str, TaskGraph] = {}
    for recipe_id, steps in recipe_steps.items():
        sorted_steps = sorted(steps.items(), key=lambda x: x[0])
        nodes = [
            {"step_id": sid, "description": desc, "index": i}
            for i, (sid, desc) in enumerate(sorted_steps)
        ]
        edges = [{"from": i, "to": i + 1} for i in range(max(0, len(nodes) - 1))]
        graphs[recipe_id] = TaskGraph(recipe_id=recipe_id, nodes=nodes, edges=edges)

    return graphs


def normalize_task_graphs(raw: dict[str, Any]) -> dict[str, TaskGraph]:
    out: dict[str, TaskGraph] = {}
    for recipe_id, graph in raw.items():
        rid = str(graph.get("recipe_id", recipe_id))
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        out[str(recipe_id)] = TaskGraph(recipe_id=rid, nodes=nodes, edges=edges)
    return out

