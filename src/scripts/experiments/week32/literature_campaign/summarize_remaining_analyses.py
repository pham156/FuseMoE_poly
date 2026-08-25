#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    INTERACTION_COLUMNS,
    MECH_ROOT,
    dominant_expert_from_sparse,
    ensure_week32_dirs,
    load_completed_runs,
    parse_binary_label,
    parse_mask,
    parse_pheno_vector,
    read_csv,
    read_tsv,
    try_float,
    write_csv,
)

TASK_HEAD_RESULTS = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_31_old/mechanism_analysis/task_router_head_results.csv")
MASKED_EVAL_MANIFEST = MECH_ROOT / "week32_masked_eval_manifest.tsv"
ARCHIVED_DIAGNOSTIC_ROOT = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_31_old/followup_analysis/diagnostic_routers")
WEEK30_MASTER = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_30/analysis/week30_master_results.csv")


def entropy_from_probs(probs: list[float]) -> float:
    out = 0.0
    for prob in probs:
        if prob > 0:
            out -= prob * math.log(prob)
    return out


def gini_from_probs(probs: list[float]) -> float:
    return 1.0 - sum(prob * prob for prob in probs)


def normalized_purity_from_probs(probs: list[float]) -> float:
    return max(probs) if probs else math.nan


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def interaction_cluster_id(row: dict) -> int | None:
    vector = []
    for col in INTERACTION_COLUMNS:
        value = try_float(row.get(col))
        if value is None:
            return None
        vector.append(value)
    # simple deterministic 4-bin signature, enough for purity grouping
    score = sum(1 if value > 0 else 0 for value in vector)
    return min(score, 3)


def build_expert_purity() -> tuple[list[dict], list[dict]]:
    rows = []
    family_rows = []
    family_acc = defaultdict(lambda: defaultdict(list))
    for run in load_completed_runs():
        diag_path = Path(run["output_dir"]) / "router_diagnostics_test.csv"
        diag_rows = read_csv(diag_path)
        if not diag_rows:
            continue
        by_expert = defaultdict(lambda: {"count": 0, "pheno_sum": [0] * 25, "mask_counts": Counter(), "cluster_counts": Counter()})
        for row in diag_rows:
            expert_id = dominant_expert_from_sparse(row.get("topk_expert_ids_weights", ""))
            if expert_id is None:
                continue
            rec = by_expert[expert_id]
            rec["count"] += 1
            vec = parse_pheno_vector(row.get("true_label", ""))
            if vec:
                limit = min(len(vec), 25)
                for idx in range(limit):
                    rec["pheno_sum"][idx] += vec[idx]
            observed = parse_mask(row.get("observed_modality_mask", ""))
            mask_name = "+".join([key for key, val in observed.items() if val > 0.5]) or "none"
            rec["mask_counts"][mask_name] += 1
            cluster_id = interaction_cluster_id(row)
            if cluster_id is not None:
                rec["cluster_counts"][cluster_id] += 1

        for expert_id, rec in sorted(by_expert.items()):
            count = rec["count"]
            if count == 0:
                continue
            pheno_entropies = []
            pheno_ginis = []
            pheno_purities = []
            for total in rec["pheno_sum"]:
                p = total / count
                probs = [p, 1.0 - p]
                pheno_entropies.append(entropy_from_probs(probs))
                pheno_ginis.append(gini_from_probs(probs))
                pheno_purities.append(normalized_purity_from_probs(probs))
            mask_probs = [value / count for value in rec["mask_counts"].values()]
            cluster_probs = [value / count for value in rec["cluster_counts"].values()]
            out = {
                "config_id": run["config_id"],
                "architecture": run["architecture"],
                "router_family": run["router_family"],
                "init_strategy": run["init_strategy"],
                "gating_function": run["gating_function"],
                "seed": run["seed"],
                "expert_id": expert_id,
                "n_top1_samples": count,
                "pheno_entropy_mean": mean(pheno_entropies),
                "pheno_gini_mean": mean(pheno_ginis),
                "pheno_normalized_purity_mean": mean(pheno_purities),
                "mortality_entropy": math.nan,
                "mortality_gini": math.nan,
                "mortality_normalized_purity": math.nan,
                "missingness_entropy": entropy_from_probs(mask_probs) if mask_probs else math.nan,
                "missingness_gini": gini_from_probs(mask_probs) if mask_probs else math.nan,
                "missingness_normalized_purity": normalized_purity_from_probs(mask_probs) if mask_probs else math.nan,
                "interaction_cluster_entropy": entropy_from_probs(cluster_probs) if cluster_probs else math.nan,
                "interaction_cluster_gini": gini_from_probs(cluster_probs) if cluster_probs else math.nan,
                "interaction_cluster_normalized_purity": normalized_purity_from_probs(cluster_probs) if cluster_probs else math.nan,
            }
            rows.append(out)
            fam_key = (run["router_family"], run["gating_function"])
            for field in [
                "pheno_entropy_mean",
                "pheno_gini_mean",
                "pheno_normalized_purity_mean",
                "mortality_entropy",
                "mortality_gini",
                "mortality_normalized_purity",
                "missingness_entropy",
                "missingness_gini",
                "missingness_normalized_purity",
                "interaction_cluster_entropy",
                "interaction_cluster_gini",
                "interaction_cluster_normalized_purity",
            ]:
                value = out[field]
                if value == value:
                    family_acc[fam_key][field].append(value)

    # Add archived IHM diagnostic-router purity so the purity table spans phenotype,
    # mortality, missingness, and interaction dimensions instead of PHENO alone.
    for diag_path in sorted(ARCHIVED_DIAGNOSTIC_ROOT.glob("*/ihm/seed_32/*/router_diagnostics_test.csv")):
        router_family = diag_path.parts[-5]
        diag_rows = read_csv(diag_path)
        by_expert = defaultdict(lambda: {"count": 0, "mortality_pos": 0})
        for row in diag_rows:
            expert_id = dominant_expert_from_sparse(row.get("topk_expert_ids_weights", ""))
            label = parse_binary_label(row.get("true_label"))
            if expert_id is None or label is None:
                continue
            rec = by_expert[expert_id]
            rec["count"] += 1
            rec["mortality_pos"] += label
        for expert_id, rec in sorted(by_expert.items()):
            count = rec["count"]
            if count == 0:
                continue
            p = rec["mortality_pos"] / count
            probs = [p, 1.0 - p]
            out = {
                "config_id": diag_path.parent.name,
                "architecture": "moe_ref",
                "router_family": router_family,
                "init_strategy": "archived_diagnostic",
                "gating_function": "softmax",
                "seed": 32,
                "expert_id": expert_id,
                "n_top1_samples": count,
                "pheno_entropy_mean": math.nan,
                "pheno_gini_mean": math.nan,
                "pheno_normalized_purity_mean": math.nan,
                "mortality_entropy": entropy_from_probs(probs),
                "mortality_gini": gini_from_probs(probs),
                "mortality_normalized_purity": normalized_purity_from_probs(probs),
                "missingness_entropy": math.nan,
                "missingness_gini": math.nan,
                "missingness_normalized_purity": math.nan,
                "interaction_cluster_entropy": math.nan,
                "interaction_cluster_gini": math.nan,
                "interaction_cluster_normalized_purity": math.nan,
            }
            rows.append(out)
            fam_key = (router_family, "softmax")
            for field in [
                "mortality_entropy",
                "mortality_gini",
                "mortality_normalized_purity",
            ]:
                value = out[field]
                if value == value:
                    family_acc[fam_key][field].append(value)

    for (router_family, gating_function), stats in sorted(family_acc.items()):
        family_rows.append(
            {
                "router_family": router_family,
                "gating_function": gating_function,
                "n_expert_rows": len(stats["pheno_entropy_mean"]),
                **{field: mean(values) for field, values in stats.items()},
            }
        )
    return rows, family_rows


def parse_sparse_pairs(text: str) -> list[tuple[int, float]]:
    pairs = []
    for item in (text or "").split("|"):
        if ":" not in item:
            continue
        left, right = item.split(":", 1)
        try:
            pairs.append((int(left), float(right)))
        except ValueError:
            continue
    return pairs


def build_modality_specialization() -> tuple[list[dict], list[dict]]:
    manifest_rows = read_tsv(MASKED_EVAL_MANIFEST)
    selected = {
        "permod_router",
        "interaction_router",
        "interaction_missing_router",
        "task_interaction_router",
    }
    counts = defaultdict(int)
    mask_totals = defaultdict(int)
    expert_totals = defaultdict(int)
    for row in manifest_rows:
        if row["router_family"] not in selected:
            continue
        eval_csv = Path(row["output_dir"]) / "router_diagnostics_eval.csv"
        if not eval_csv.exists():
            continue
        for eval_row in read_csv(eval_csv):
            expert_id = dominant_expert_from_sparse(eval_row.get("topk_expert_ids_weights", ""))
            if expert_id is None:
                continue
            fam_key = (row["router_family"], row["gating_function"])
            key = fam_key + (row["mask_name"], expert_id)
            counts[key] += 1
            mask_totals[fam_key + (row["mask_name"],)] += 1
            expert_totals[fam_key + (expert_id,)] += 1

    rows = []
    family_rows = []
    family_acc = defaultdict(list)
    for (router_family, gating_function, mask_name, expert_id), count in sorted(counts.items()):
        p_expert_given_mask = count / mask_totals[(router_family, gating_function, mask_name)]
        p_mask_given_expert = count / expert_totals[(router_family, gating_function, expert_id)]
        rows.append(
            {
                "router_family": router_family,
                "gating_function": gating_function,
                "mask_name": mask_name,
                "expert_id": expert_id,
                "count": count,
                "p_expert_given_mask": p_expert_given_mask,
                "p_mask_given_expert": p_mask_given_expert,
            }
        )
        family_acc[(router_family, gating_function, expert_id)].append((mask_name, p_mask_given_expert))

    for (router_family, gating_function, expert_id), pairs in sorted(family_acc.items()):
        probs = [prob for _, prob in pairs]
        family_rows.append(
            {
                "router_family": router_family,
                "gating_function": gating_function,
                "expert_id": expert_id,
                "mask_entropy": entropy_from_probs(probs),
                "mask_gini": gini_from_probs(probs),
                "mask_normalized_purity": normalized_purity_from_probs(probs),
                "dominant_mask": max(pairs, key=lambda item: item[1])[0],
            }
        )
    return rows, family_rows


def build_task_conditioned_organization() -> tuple[list[dict], list[dict]]:
    rows = []
    family_rows = []
    grouped = defaultdict(list)
    if TASK_HEAD_RESULTS.exists():
        rows = read_csv(TASK_HEAD_RESULTS)
    for row in rows:
        grouped[(row["architecture"], row["variant_name"], row["task_short"], "task_head_metrics")].append(row)
    for (architecture, variant_name, task_short, row_type), subset in sorted(grouped.items()):
        def avg(field: str) -> float:
            vals = [float(row[field]) for row in subset if row.get(field, "") not in {"", "nan"}]
            return mean(vals)

        family_rows.append(
            {
                "architecture": architecture,
                "variant_name": variant_name,
                "task_short": task_short,
                "row_type": row_type,
                "n_rows": len(subset),
                "mean_metric": avg("metric_value"),
                "mean_delta_vs_baseline_router": avg("delta_vs_baseline_router"),
                "mean_delta_vs_task_router": avg("delta_vs_task_router"),
                "mean_gate_entropy": avg("gate_entropy"),
                "mean_gate_top1_weight": avg("gate_top1_weight"),
                "mean_active_experts": avg("active_experts"),
                "mean_pairwise_js_to_other_tasks": math.nan,
            }
        )

    if WEEK30_MASTER.exists():
        week30_rows = read_csv(WEEK30_MASTER)

        def parse_router_summary(text: str) -> list[float]:
            mass = [0.0, 0.0, 0.0, 0.0]
            n_layers = 0
            for chunk in (text or "").split(" ; "):
                if not chunk.startswith("L") or "M" in chunk or "mass[" not in chunk:
                    continue
                inside = chunk.split("mass[", 1)[1].split("]", 1)[0]
                vals = [float(x) for x in inside.split(",")]
                if len(vals) != 4:
                    continue
                for idx, val in enumerate(vals):
                    mass[idx] += val
                n_layers += 1
            if n_layers == 0:
                return []
            mass = [v / n_layers for v in mass]
            total = sum(mass)
            return [v / total for v in mass] if total > 0 else []

        def js_div(p: list[float], q: list[float]) -> float:
            if not p or not q:
                return math.nan
            m = [(a + b) / 2.0 for a, b in zip(p, q)]
            def kl(a: list[float], b: list[float]) -> float:
                out = 0.0
                for x, y in zip(a, b):
                    if x > 0 and y > 0:
                        out += x * math.log(x / y)
                return out
            return 0.5 * kl(p, m) + 0.5 * kl(q, m)

        selected = [row for row in week30_rows if row["family_name"] == "A" and row.get("test_router_summary", "")]
        task_seed_dist = {}
        task_seed_metric = {}
        for row in selected:
            seed = int(row["seed"])
            task_short = "pheno" if "pheno" in row["task"] else ("los" if "los" in row["task"] else "ihm")
            dist = parse_router_summary(row["test_router_summary"])
            if not dist:
                continue
            task_seed_dist[(seed, task_short)] = dist
            task_seed_metric[(seed, task_short)] = float(row["best_primary_metric"])
        for seed in sorted({seed for seed, _ in task_seed_dist.keys()}):
            available = [task for s, task in task_seed_dist.keys() if s == seed]
            pairwise = {}
            for i, left in enumerate(available):
                for right in available[i + 1:]:
                    pairwise[(left, right)] = js_div(task_seed_dist[(seed, left)], task_seed_dist[(seed, right)])
            for task_short in available:
                js_vals = [val for (left, right), val in pairwise.items() if left == task_short or right == task_short]
                family_rows.append(
                    {
                        "architecture": "recon_ref",
                        "variant_name": "week30_task_conditioned_router",
                        "task_short": task_short,
                        "row_type": "cross_task_js",
                        "n_rows": 1,
                        "mean_metric": task_seed_metric[(seed, task_short)],
                        "mean_delta_vs_baseline_router": math.nan,
                        "mean_delta_vs_task_router": math.nan,
                        "mean_gate_entropy": math.nan,
                        "mean_gate_top1_weight": 1.0,
                        "mean_active_experts": 2.0,
                        "mean_pairwise_js_to_other_tasks": mean(js_vals),
                    }
                )
    return rows, family_rows


def write_reports(
    purity_rows: list[dict],
    purity_family_rows: list[dict],
    mod_rows: list[dict],
    mod_family_rows: list[dict],
    task_rows: list[dict],
    task_family_rows: list[dict],
) -> None:
    def fmt(row: dict, field: str) -> str:
        value = row.get(field, "")
        if value in {"", None}:
            return "NA"
        try:
            value = float(value)
        except Exception:
            return "NA"
        return f"{value:.4f}" if value == value else "NA"

    purity_lines = [
        "# Week32 Expert Purity Analysis",
        "",
        "- scope: corrected Week31 PHENO checkpoint set plus archived IHM diagnostic-router purity",
        "- purity dimensions: phenotype-label marginals, mortality label marginals, missingness regime, interaction-cluster regime",
        "",
        "## Family means",
    ]
    for row in purity_family_rows:
        purity_lines.append(
            f"- `{row['router_family']}/{row['gating_function']}`: "
            f"pheno_purity={fmt(row, 'pheno_normalized_purity_mean')}, "
            f"mortality_purity={fmt(row, 'mortality_normalized_purity')}, "
            f"missingness_purity={fmt(row, 'missingness_normalized_purity')}, "
            f"interaction_purity={fmt(row, 'interaction_cluster_normalized_purity')}"
        )
    (MECH_ROOT / "expert_purity_report.md").write_text("\n".join(purity_lines) + "\n")

    mod_lines = [
        "# Week32 Modality Specialization Statistics",
        "",
        "- scope: finished Week32 masked-eval families with forced missingness masks",
        "- statistics: P(expert | mask), P(mask | expert), mask-entropy, mask-gini, dominant mask",
        "",
        "## Expert-level mask specialization",
    ]
    for row in mod_family_rows[:24]:
        mod_lines.append(
            f"- `{row['router_family']}/{row['gating_function']}/expert{row['expert_id']}`: "
            f"dominant_mask={row['dominant_mask']}, purity={float(row['mask_normalized_purity']):.4f}, entropy={float(row['mask_entropy']):.4f}"
        )
    (MECH_ROOT / "modality_specialization_report.md").write_text("\n".join(mod_lines) + "\n")

    task_lines = [
        "# Week32 Task-Conditioned Expert Organization",
        "",
        "- sources: archived corrected multitask task-head analysis from Week31_old and stored Week30 task-conditioned routing summaries across PHENO/IHM/LOS",
        "- explicit cross-task routing divergence is reported as pairwise Jensen-Shannon divergence over stored test-time expert-mass distributions",
        "",
        "## Task-level summaries",
    ]
    for row in task_family_rows:
        if row["row_type"] == "cross_task_js":
            task_lines.append(
                f"- `{row['architecture']}/{row['variant_name']}/{row['task_short']}`: "
                f"metric={float(row['mean_metric']):.4f}, "
                f"pairwise_js_to_other_tasks={float(row['mean_pairwise_js_to_other_tasks']):.4f}, "
                f"top1={float(row['mean_gate_top1_weight']):.4f}, "
                f"active_experts={float(row['mean_active_experts']):.4f}"
            )
        else:
            task_lines.append(
                f"- `{row['architecture']}/{row['variant_name']}/{row['task_short']}`: "
                f"metric={float(row['mean_metric']):.4f}, "
                f"delta_vs_baseline={float(row['mean_delta_vs_baseline_router']):.4f}, "
                f"delta_vs_task_router={float(row['mean_delta_vs_task_router']):.4f}, "
                f"top1={float(row['mean_gate_top1_weight']):.4f}, "
                f"active_experts={float(row['mean_active_experts']):.4f}"
            )
    (MECH_ROOT / "task_conditioned_organization_report.md").write_text("\n".join(task_lines) + "\n")


def main() -> None:
    ensure_week32_dirs()
    purity_rows, purity_family_rows = build_expert_purity()
    mod_path = MECH_ROOT / "modality_specialization.csv"
    mod_family_path = MECH_ROOT / "modality_specialization_family_summary.csv"
    if mod_path.exists() and mod_family_path.exists():
        mod_rows = read_csv(mod_path)
        mod_family_rows = read_csv(mod_family_path)
    else:
        mod_rows, mod_family_rows = build_modality_specialization()
    task_rows, task_family_rows = build_task_conditioned_organization()

    write_csv(
        MECH_ROOT / "expert_purity.csv",
        purity_rows,
        [
            "config_id",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "seed",
            "expert_id",
            "n_top1_samples",
            "pheno_entropy_mean",
            "pheno_gini_mean",
            "pheno_normalized_purity_mean",
            "mortality_entropy",
            "mortality_gini",
            "mortality_normalized_purity",
            "missingness_entropy",
            "missingness_gini",
            "missingness_normalized_purity",
            "interaction_cluster_entropy",
            "interaction_cluster_gini",
            "interaction_cluster_normalized_purity",
        ],
    )
    write_csv(
        MECH_ROOT / "expert_purity_family_summary.csv",
        purity_family_rows,
        [
            "router_family",
            "gating_function",
            "n_expert_rows",
            "pheno_entropy_mean",
            "pheno_gini_mean",
            "pheno_normalized_purity_mean",
            "mortality_entropy",
            "mortality_gini",
            "mortality_normalized_purity",
            "missingness_entropy",
            "missingness_gini",
            "missingness_normalized_purity",
            "interaction_cluster_entropy",
            "interaction_cluster_gini",
            "interaction_cluster_normalized_purity",
        ],
    )
    write_csv(
        MECH_ROOT / "modality_specialization.csv",
        mod_rows,
        ["router_family", "gating_function", "mask_name", "expert_id", "count", "p_expert_given_mask", "p_mask_given_expert"],
    )
    write_csv(
        MECH_ROOT / "modality_specialization_family_summary.csv",
        mod_family_rows,
        ["router_family", "gating_function", "expert_id", "mask_entropy", "mask_gini", "mask_normalized_purity", "dominant_mask"],
    )
    if task_rows:
        write_csv(
            MECH_ROOT / "task_conditioned_organization.csv",
            task_rows,
            list(task_rows[0].keys()),
        )
    if task_family_rows:
        write_csv(
            MECH_ROOT / "task_conditioned_organization_summary.csv",
            task_family_rows,
            [
                "architecture",
                "variant_name",
                "task_short",
                "row_type",
                "n_rows",
                "mean_metric",
                "mean_delta_vs_baseline_router",
                "mean_delta_vs_task_router",
                "mean_gate_entropy",
                "mean_gate_top1_weight",
                "mean_active_experts",
                "mean_pairwise_js_to_other_tasks",
            ],
        )
    write_reports(purity_rows, purity_family_rows, mod_rows, mod_family_rows, task_rows, task_family_rows)


if __name__ == "__main__":
    main()
