#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

from common import MECH_ROOT, parse_sparse_distribution, read_csv, read_tsv, summarize, write_csv

MANIFEST_PATH = MECH_ROOT / "week32_masked_eval_manifest.tsv"


def kl_div(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return total


def js_div(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m, eps=eps) + 0.5 * kl_div(q, m, eps=eps)


def l1_dist(p: list[float], q: list[float]) -> float:
    return sum(abs(pi - qi) for pi, qi in zip(p, q))


def build_full_lookup(checkpoint_path: str) -> dict[tuple[str, str], list[float]]:
    full_csv = Path(checkpoint_path).parent / "router_diagnostics_test.csv"
    rows = read_csv(full_csv)
    lookup = {}
    for row in rows:
        key = (str(row.get("sample_id", "")), str(row.get("task_name", "")))
        lookup[key] = parse_sparse_distribution(row.get("topk_expert_ids_weights", ""))
    return lookup


def main() -> None:
    manifest_rows = read_tsv(MANIFEST_PATH)
    out_rows = []
    sample_rows = []
    cache: dict[str, dict[tuple[str, str], list[float]]] = {}
    for row in manifest_rows:
        eval_csv = Path(row["output_dir"]) / "router_diagnostics_eval.csv"
        if not eval_csv.exists():
            continue
        checkpoint_path = row["checkpoint_path"]
        if checkpoint_path not in cache:
            cache[checkpoint_path] = build_full_lookup(checkpoint_path)
        full_lookup = cache[checkpoint_path]
        kl_values = []
        js_values = []
        l1_values = []
        eval_rows = read_csv(eval_csv)
        for eval_row in eval_rows:
            key = (str(eval_row.get("sample_id", "")), str(eval_row.get("task_name", "")))
            full = full_lookup.get(key)
            if full is None:
                continue
            masked = parse_sparse_distribution(eval_row.get("topk_expert_ids_weights", ""))
            kl = kl_div(full, masked)
            js = js_div(full, masked)
            l1 = l1_dist(full, masked)
            kl_values.append(kl)
            js_values.append(js)
            l1_values.append(l1)
            sample_rows.append(
                {
                    "config_id": row["config_id"],
                    "task_short": row["task_short"],
                    "architecture": row["architecture"],
                    "router_family": row["router_family"],
                    "init_strategy": row["init_strategy"],
                    "gating_function": row["gating_function"],
                    "seed": row["seed"],
                    "mask_name": row["mask_name"],
                    "sample_id": key[0],
                    "task_name": key[1],
                    "kl_full_to_masked": kl,
                    "js_full_masked": js,
                    "l1_full_masked": l1,
                }
            )
        kl_mean, kl_std = summarize(kl_values)
        js_mean, js_std = summarize(js_values)
        l1_mean, l1_std = summarize(l1_values)
        out_rows.append(
            {
                "config_id": row["config_id"],
                "task_short": row["task_short"],
                "architecture": row["architecture"],
                "router_family": row["router_family"],
                "init_strategy": row["init_strategy"],
                "gating_function": row["gating_function"],
                "seed": row["seed"],
                "mask_name": row["mask_name"],
                "n_samples": len(kl_values),
                "kl_full_to_masked": kl_mean,
                "kl_full_to_masked_std": kl_std,
                "js_full_masked": js_mean,
                "js_full_masked_std": js_std,
                "l1_full_masked": l1_mean,
                "l1_full_masked_std": l1_std,
            }
        )

    write_csv(
        MECH_ROOT / "routing_stability.csv",
        out_rows,
        [
            "config_id",
            "task_short",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "seed",
            "mask_name",
            "n_samples",
            "kl_full_to_masked",
            "kl_full_to_masked_std",
            "js_full_masked",
            "js_full_masked_std",
            "l1_full_masked",
            "l1_full_masked_std",
        ],
    )
    write_csv(
        MECH_ROOT / "routing_stability_per_sample.csv",
        sample_rows,
        [
            "config_id",
            "task_short",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "seed",
            "mask_name",
            "sample_id",
            "task_name",
            "kl_full_to_masked",
            "js_full_masked",
            "l1_full_masked",
        ],
    )

    report_lines = [
        "# Week32 Routing Stability",
        "",
        "Paper inspiration: No direct precedent in the surveyed literature.",
        "Adaptation: Full versus masked routing distributions are compared on the same patient after replaying corrected Week31 checkpoints under fixed missing-modality masks.",
        "Novel contribution: Routing stability as an explicit robustness statistic for multimodal healthcare MoE.",
        "",
        f"- completed eval rows aggregated: {len(out_rows)}",
        f"- per-sample comparisons: {len(sample_rows)}",
        "- metrics: KL(full || masked), JS(full, masked), L1(full, masked)",
    ]
    (MECH_ROOT / "routing_stability_report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
