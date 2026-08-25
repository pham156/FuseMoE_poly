#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

from common import (
    INTERACTION_COLUMNS,
    MECH_ROOT,
    dominant_expert_from_sparse,
    ensure_week32_dirs,
    load_completed_runs,
    read_csv,
    try_float,
    write_csv,
)


def kmeans(vectors: list[list[float]], k: int, max_iter: int = 50) -> tuple[list[int], list[list[float]]]:
    if len(vectors) < k:
        raise ValueError(f"need at least {k} vectors, got {len(vectors)}")
    centroids = [list(vec) for vec in vectors[:k]]
    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        changed = False
        for idx, vec in enumerate(vectors):
            best_cluster = min(
                range(k),
                key=lambda cluster_id: sum((value - centroids[cluster_id][dim]) ** 2 for dim, value in enumerate(vec)),
            )
            if assignments[idx] != best_cluster:
                assignments[idx] = best_cluster
                changed = True
        new_centroids = [[0.0] * len(vectors[0]) for _ in range(k)]
        counts = [0] * k
        for cluster_id, vec in zip(assignments, vectors):
            counts[cluster_id] += 1
            for dim, value in enumerate(vec):
                new_centroids[cluster_id][dim] += value
        for cluster_id in range(k):
            if counts[cluster_id] == 0:
                new_centroids[cluster_id] = list(vectors[random.randrange(len(vectors))])
            else:
                new_centroids[cluster_id] = [value / counts[cluster_id] for value in new_centroids[cluster_id]]
        centroids = new_centroids
        if not changed:
            break
    return assignments, centroids


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            out -= prob * math.log(prob)
    return out


def mutual_information(xs: list[int], ys: list[int]) -> float:
    total = len(xs)
    cx = Counter(xs)
    cy = Counter(ys)
    cxy = Counter(zip(xs, ys))
    out = 0.0
    for (xval, yval), joint in cxy.items():
        pxy = joint / total
        px = cx[xval] / total
        py = cy[yval] / total
        out += pxy * math.log(pxy / (px * py))
    return out


def normalized_mutual_information(xs: list[int], ys: list[int]) -> float:
    hx = entropy(Counter(xs))
    hy = entropy(Counter(ys))
    if hx == 0.0 or hy == 0.0:
        return 0.0
    return mutual_information(xs, ys) / math.sqrt(hx * hy)


def adjusted_rand_index(xs: list[int], ys: list[int]) -> float:
    from math import comb

    n = len(xs)
    if n < 2:
        return 0.0
    cx = Counter(xs)
    cy = Counter(ys)
    cxy = Counter(zip(xs, ys))
    sum_xy = sum(comb(count, 2) for count in cxy.values())
    sum_x = sum(comb(count, 2) for count in cx.values())
    sum_y = sum(comb(count, 2) for count in cy.values())
    total_pairs = comb(n, 2)
    expected = (sum_x * sum_y) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_x + sum_y)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return (sum_xy - expected) / denom


def load_sample_rows(run: dict) -> list[dict]:
    diag_path = Path(run["output_dir"]) / "router_diagnostics_test.csv"
    rows = read_csv(diag_path)
    if not rows or any(col not in rows[0] for col in INTERACTION_COLUMNS):
        return []
    sample_rows = []
    for row in rows:
        vector = []
        for col in INTERACTION_COLUMNS:
            value = try_float(row.get(col))
            if value is None:
                vector = []
                break
            vector.append(value)
        if not vector:
            continue
        expert_id = dominant_expert_from_sparse(row.get("topk_expert_ids_weights", ""))
        if expert_id is None:
            continue
        sample_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "expert_id": expert_id,
                "vector": vector,
            }
        )
    return sample_rows


def main() -> None:
    ensure_week32_dirs()
    random.seed(0)
    runs = load_completed_runs()
    out_rows = []
    for run in runs:
        sample_rows = load_sample_rows(run)
        if len(sample_rows) < 32:
            continue
        vectors = [row["vector"] for row in sample_rows]
        expert_ids = [row["expert_id"] for row in sample_rows]
        cluster_ids, centroids = kmeans(vectors, 4)
        nmi = normalized_mutual_information(cluster_ids, expert_ids)
        ari = adjusted_rand_index(cluster_ids, expert_ids)
        cluster_sizes = Counter(cluster_ids)
        expert_sizes = Counter(expert_ids)
        out_rows.append(
            {
                "config_id": run["config_id"],
                "architecture": run["architecture"],
                "router_family": run["router_family"],
                "init_strategy": run["init_strategy"],
                "gating_function": run["gating_function"],
                "task_short": run["task_short"],
                "seed": run["seed"],
                "n_samples": len(sample_rows),
                "nmi_interaction_cluster_vs_expert": nmi,
                "ari_interaction_cluster_vs_expert": ari,
                "cluster_sizes": ";".join(f"c{idx}:{cluster_sizes.get(idx, 0)}" for idx in range(4)),
                "expert_sizes": ";".join(f"e{idx}:{expert_sizes.get(idx, 0)}" for idx in range(4)),
                "centroid_summary": ";".join(
                    f"c{idx}:" + ",".join(f"{value:.4f}" for value in centroid[:2]) for idx, centroid in enumerate(centroids)
                ),
            }
        )

    write_csv(
        MECH_ROOT / "interaction_cluster_alignment.csv",
        out_rows,
        [
            "config_id",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "task_short",
            "seed",
            "n_samples",
            "nmi_interaction_cluster_vs_expert",
            "ari_interaction_cluster_vs_expert",
            "cluster_sizes",
            "expert_sizes",
            "centroid_summary",
        ],
    )

    family_scores: dict[tuple[str, str], list[float]] = {}
    for row in out_rows:
        key = (row["router_family"], row["gating_function"])
        family_scores.setdefault(key, []).append(float(row["nmi_interaction_cluster_vs_expert"]))

    lines = [
        "# Week32 Interaction Cluster vs Expert Alignment",
        "",
        "Paper inspiration: MMOE and Guiding Mixture-of-Experts with Temporal Multimodal Interactions (2025).",
        "Adaptation: ICU patients are clustered by six logged multimodal interaction features and aligned against dominant expert assignments from corrected Week31 checkpoints.",
        "Novel contribution: Quantifies whether healthcare experts recover interaction regimes rather than only modality or task partitions.",
        "",
        f"- runs with usable interaction diagnostics: {len(out_rows)}",
        "",
        "## Mean NMI by family",
    ]
    for (router_family, gating_function), values in sorted(family_scores.items()):
        lines.append(f"- `{router_family}/{gating_function}`: {sum(values) / len(values):.4f}")
    (MECH_ROOT / "interaction_cluster_alignment_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
