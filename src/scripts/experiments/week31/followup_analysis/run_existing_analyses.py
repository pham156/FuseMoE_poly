#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

from common import (
    FOLLOWUP_ROOT,
    ensure_followup_root,
    jaccard,
    linear_regression,
    load_completed_runs,
    metric_value,
    parse_binary_label,
    parse_mask,
    parse_pheno_vector,
    parse_topk_string,
    pearson,
    spearman,
    task_short,
    top1_margin,
    topk_entropy,
    try_float,
    write_csv,
)

INTERACTION_COLUMNS = [
    "cosine_text_ts",
    "cosine_text_cxr",
    "cosine_text_ecg",
    "cosine_ts_cxr",
    "cosine_ts_ecg",
    "cosine_cxr_ecg",
]


def load_router_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def build_expert_profiles(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        diag_rows = load_router_csv(Path(run["output_dir"]) / "router_diagnostics_test.csv")
        if not diag_rows:
            continue
        by_expert = defaultdict(lambda: {
            "decision_count": 0,
            "top1_count": 0,
            "gate_mass_sum": 0.0,
            "entropy_sum": 0.0,
            "sample_ids": set(),
            "interaction_sums": defaultdict(float),
            "interaction_counts": defaultdict(int),
            "missing_sum": defaultdict(float),
            "label_values": [],
            "pheno_sum": None,
            "n_pheno": 0,
        })
        total_decisions = 0
        short = task_short(run.get("task", ""))
        for row in diag_rows:
            mask = parse_mask(row.get("observed_modality_mask", ""))
            for _, pairs in parse_topk_string(row.get("topk_expert_ids_weights", "")):
                if not pairs:
                    continue
                total_decisions += 1
                ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
                entropy = topk_entropy(ranked)
                top1 = ranked[0][0]
                for expert_id, weight in ranked:
                    rec = by_expert[expert_id]
                    rec["decision_count"] += 1
                    rec["gate_mass_sum"] += weight
                    rec["entropy_sum"] += entropy
                    rec["sample_ids"].add(row.get("sample_id", ""))
                    if expert_id == top1:
                        rec["top1_count"] += 1
                        for key, value in mask.items():
                            rec["missing_sum"][key] += 1.0 - value
                        for col in INTERACTION_COLUMNS:
                            val = try_float(row.get(col, ""))
                            if val is not None:
                                rec["interaction_sums"][col] += val
                                rec["interaction_counts"][col] += 1
                        if short in {"los", "ihm"}:
                            label = parse_binary_label(row.get("true_label", ""))
                            if label is not None:
                                rec["label_values"].append(label)
                        elif short == "pheno":
                            vec = parse_pheno_vector(row.get("true_label", ""))
                            if vec:
                                if rec["pheno_sum"] is None:
                                    rec["pheno_sum"] = [0] * len(vec)
                                rec["pheno_sum"] = [a + b for a, b in zip(rec["pheno_sum"], vec)]
                                rec["n_pheno"] += 1

        for expert_id, rec in sorted(by_expert.items()):
            top_pheno = []
            if rec["pheno_sum"] and rec["n_pheno"] > 0:
                prevalences = [(idx, val / rec["n_pheno"]) for idx, val in enumerate(rec["pheno_sum"])]
                top_pheno = sorted(prevalences, key=lambda item: item[1], reverse=True)[:3]
            out = {
                "config_id": run["config_id"],
                "task_short": run["task_short"],
                "architecture": run["architecture"],
                "router_family": run["router_family"],
                "init_strategy": run["init_strategy"],
                "seed": run["seed"],
                "expert_id": expert_id,
                "decision_count": rec["decision_count"],
                "selection_frequency": rec["decision_count"] / total_decisions if total_decisions else math.nan,
                "gate_mass_mean": rec["gate_mass_sum"] / rec["decision_count"] if rec["decision_count"] else math.nan,
                "top1_frequency": rec["top1_count"] / total_decisions if total_decisions else math.nan,
                "entropy_mean": rec["entropy_sum"] / rec["decision_count"] if rec["decision_count"] else math.nan,
                "text_missing_rate": rec["missing_sum"]["text"] / rec["top1_count"] if rec["top1_count"] else math.nan,
                "cxr_missing_rate": rec["missing_sum"]["cxr"] / rec["top1_count"] if rec["top1_count"] else math.nan,
                "ecg_missing_rate": rec["missing_sum"]["ecg"] / rec["top1_count"] if rec["top1_count"] else math.nan,
                "mean_los_label": sum(rec["label_values"]) / len(rec["label_values"]) if short == "los" and rec["label_values"] else math.nan,
                "mortality_rate": sum(rec["label_values"]) / len(rec["label_values"]) if short == "ihm" and rec["label_values"] else math.nan,
                "top_phenotype_prevalences": ";".join(f"pheno_{idx}:{prev:.4f}" for idx, prev in top_pheno),
                "sample_ids_top1": ";".join(sorted(filter(None, rec["sample_ids"]))),
            }
            for col in INTERACTION_COLUMNS:
                count = rec["interaction_counts"][col]
                out[f"mean_{col}"] = rec["interaction_sums"][col] / count if count else math.nan
            rows.append(out)
    return rows


def write_expert_profile_report(rows: list[dict]) -> None:
    fam_interactions = defaultdict(list)
    for row in rows:
        val = row["mean_cosine_text_ts"]
        if not math.isnan(float(val)):
            fam_interactions[row["router_family"]].append(float(val))
    lines = [
        "# Week31 Expert Interaction Profile Report",
        "",
        f"- profile rows: {len(rows)}",
        "- interaction-feature summaries are available only for router families that logged interaction features into `router_diagnostics_test.csv`.",
        "- clinical label summaries are task-specific: LOS means for LOS runs, mortality rates for IHM runs, and phenotype prevalences for PHENO runs.",
        "",
        "## Family-level interaction signal",
    ]
    for fam, values in sorted(fam_interactions.items()):
        lines.append(f"- `{fam}` mean expert-level cosine(text,ts): {sum(values)/len(values):.4f}")
    (FOLLOWUP_ROOT / "expert_profile_report.md").write_text("\n".join(lines))


def build_hard_vs_soft(runs: list[dict]) -> tuple[list[dict], str]:
    enriched = []
    for run in runs:
        diag_rows = load_router_csv(Path(run["output_dir"]) / "router_diagnostics_test.csv")
        margins = []
        for row in diag_rows:
            for _, pairs in parse_topk_string(row.get("topk_expert_ids_weights", "")):
                margins.append(top1_margin(pairs))
        enriched.append({**run, "performance": metric_value(run), "routing_margin": sum(margins) / len(margins) if margins else math.nan})

    rows = []
    for scope in ["all", "ihm", "los", "pheno"]:
        subset = enriched if scope == "all" else [r for r in enriched if r["task_short"] == scope]
        for x_metric in ["gate_entropy", "gate_top1_weight", "active_experts", "routing_margin"]:
            paired = []
            for rec in subset:
                try:
                    x = float(rec[x_metric])
                    if math.isnan(x):
                        continue
                    paired.append((x, float(rec["performance"])))
                except Exception:
                    continue
            xs = [x for x, _ in paired]
            ys = [y for _, y in paired]
            slope, intercept, r2 = linear_regression(xs, ys)
            rows.append(
                {
                    "task_scope": scope,
                    "x_metric": x_metric,
                    "y_metric": "performance",
                    "n": len(xs),
                    "spearman": spearman(xs, ys),
                    "pearson": pearson(xs, ys),
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                }
            )
    for fam in sorted({r["router_family"] for r in enriched}):
        fam_subset = [r for r in enriched if r["router_family"] == fam]
        rows.append(
            {
                "task_scope": "family_summary",
                "x_metric": fam,
                "y_metric": "performance",
                "n": len(fam_subset),
                "spearman": math.nan,
                "pearson": math.nan,
                "slope": math.nan,
                "intercept": math.nan,
                "r2": math.nan,
                "mean_performance": sum(r["performance"] for r in fam_subset) / len(fam_subset),
                "mean_gate_entropy": sum(float(r["gate_entropy"]) for r in fam_subset) / len(fam_subset),
                "mean_top1_weight": sum(float(r["gate_top1_weight"]) for r in fam_subset) / len(fam_subset),
                "mean_active_experts": sum(float(r["active_experts"]) for r in fam_subset) / len(fam_subset),
                "mean_routing_margin": sum(float(r["routing_margin"]) for r in fam_subset if not math.isnan(float(r["routing_margin"]))) / max(1, sum(0 if math.isnan(float(r["routing_margin"])) else 1 for r in fam_subset)),
            }
        )
    lines = [
        "# Week31 Hard-vs-Soft Routing Report",
        "",
        f"- runs analyzed: {len(enriched)}",
        "",
        "## Global relationships",
    ]
    for row in rows:
        if row["task_scope"] == "all":
            lines.append(f"- `{row['x_metric']}` vs performance: n={row['n']}, spearman={row['spearman']:.4f}, r2={row['r2']:.4f}")
    (FOLLOWUP_ROOT / "hard_vs_soft_routing_report.md").write_text("\n".join(lines))
    return rows, "\n".join(lines)


def vector_from_profile(row: dict) -> list[float]:
    vec = [
        float(row["selection_frequency"]),
        float(row["gate_mass_mean"]),
        float(row["top1_frequency"]),
        0.0 if math.isnan(float(row["entropy_mean"])) else float(row["entropy_mean"]),
        0.0 if math.isnan(float(row["text_missing_rate"])) else float(row["text_missing_rate"]),
        0.0 if math.isnan(float(row["cxr_missing_rate"])) else float(row["cxr_missing_rate"]),
        0.0 if math.isnan(float(row["ecg_missing_rate"])) else float(row["ecg_missing_rate"]),
    ]
    for col in INTERACTION_COLUMNS:
        value = float(row[f"mean_{col}"])
        vec.append(0.0 if math.isnan(value) else value)
    return vec


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_cross_seed_consistency(profile_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in profile_rows:
        grouped[(row["architecture"], row["router_family"], row["init_strategy"], row["task_short"])].append(row)
    out_rows = []
    for group_key, rows in sorted(grouped.items()):
        seeds = sorted({row["seed"] for row in rows})
        if seeds != ["32", "42", "52"]:
            continue
        by_seed = {seed: sorted([r for r in rows if r["seed"] == seed], key=lambda rec: int(rec["expert_id"])) for seed in seeds}
        sizes = {seed: len(seed_rows) for seed, seed_rows in by_seed.items()}
        if min(sizes.values()) == 0 or len(set(sizes.values())) != 1:
            continue
        pair_scores = []
        best_pairs = []
        worst_pairs = []
        for left_seed, right_seed in itertools.combinations(seeds, 2):
            left_profiles = by_seed[left_seed]
            right_profiles = by_seed[right_seed]
            best_score = -1e9
            best_details = None
            for perm in itertools.permutations(range(len(right_profiles))):
                details = []
                scores = []
                for li, ri in enumerate(perm):
                    left = left_profiles[li]
                    right = right_profiles[ri]
                    interaction_sim = cosine_sim(vector_from_profile(left), vector_from_profile(right))
                    cohort_overlap = jaccard(filter(None, left["sample_ids_top1"].split(";")), filter(None, right["sample_ids_top1"].split(";")))
                    gate_sim = 1.0 - min(1.0, abs(float(left["gate_mass_mean"]) - float(right["gate_mass_mean"])))
                    miss_left = [float(left["text_missing_rate"]), float(left["cxr_missing_rate"]), float(left["ecg_missing_rate"])]
                    miss_right = [float(right["text_missing_rate"]), float(right["cxr_missing_rate"]), float(right["ecg_missing_rate"])]
                    missing_sim = 1.0 - min(1.0, sum(abs(a - b) for a, b in zip(miss_left, miss_right)) / 3.0)
                    score = 0.4 * interaction_sim + 0.3 * cohort_overlap + 0.15 * gate_sim + 0.15 * missing_sim
                    scores.append(score)
                    details.append((int(left["expert_id"]), int(right["expert_id"]), score))
                mean_score = sum(scores) / len(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_details = details
            if best_details:
                pair_scores.append(best_score)
                best_pair = max(best_details, key=lambda item: item[2])
                worst_pair = min(best_details, key=lambda item: item[2])
                best_pairs.append(f"{left_seed}->{right_seed}:{best_pair[0]}~{best_pair[1]}:{best_pair[2]:.4f}")
                worst_pairs.append(f"{left_seed}->{right_seed}:{worst_pair[0]}~{worst_pair[1]}:{worst_pair[2]:.4f}")
        out_rows.append(
            {
                "architecture": group_key[0],
                "router_family": group_key[1],
                "init_strategy": group_key[2],
                "task_short": group_key[3],
                "mean_expert_consistency_score": sum(pair_scores) / len(pair_scores) if pair_scores else math.nan,
                "best_matched_expert_pairs": ";".join(best_pairs),
                "worst_matched_expert_pairs": ";".join(worst_pairs),
            }
        )
    lines = ["# Week31 Cross-Seed Expert Consistency Report", ""]
    for row in sorted(out_rows, key=lambda rec: rec["mean_expert_consistency_score"], reverse=True)[:10]:
        lines.append(f"- `{row['architecture']}/{row['router_family']}/{row['init_strategy']}/{row['task_short']}` consistency={row['mean_expert_consistency_score']:.4f}")
    (FOLLOWUP_ROOT / "cross_seed_expert_consistency_report.md").write_text("\n".join(lines))
    return out_rows


def write_final_report(profile_rows: list[dict], hard_report: str, consistency_rows: list[dict]) -> None:
    top_profiles = sorted(
        [row for row in profile_rows if not math.isnan(float(row["mean_cosine_text_ts"]))],
        key=lambda row: abs(float(row["mean_cosine_text_ts"])),
        reverse=True,
    )[:5]
    top_consistency = sorted(consistency_rows, key=lambda row: row["mean_expert_consistency_score"], reverse=True)[:5]
    lines = [
        "# Week31 Follow-up Report",
        "",
        "## 1. Executive summary",
        "",
        "- Sections 1, 3, and 4 were executed directly from existing Week31 checkpoints, router diagnostics, and aggregate CSVs.",
        "- Section 2 requires masked eval reruns from checkpoints and is scaffolded through a manifest under `out/Week_31/followup_analysis/`, but not executed in this pass.",
        "- Section 5 requires lightweight diagnostic-router training and is scaffolded through a manifest under `out/Week_31/followup_analysis/`, but not executed in this pass.",
        "",
        "## 2. Expert interaction-profile findings",
        "",
        f"- profile rows generated: {len(profile_rows)}",
    ]
    for row in top_profiles:
        lines.append(f"- `{row['config_id']}` expert `{row['expert_id']}`: selection={float(row['selection_frequency']):.4f}, top1={float(row['top1_frequency']):.4f}, cosine(text,ts)={float(row['mean_cosine_text_ts']):.4f}")
    lines.extend([
        "",
        "## 3. Routing-stability findings",
        "",
        "- A masked-eval manifest has been generated, but the actual KL/JS/L1 reruns are still pending.",
        "",
        "## 4. Hard-vs-soft routing findings",
        "",
        hard_report,
        "",
        "## 5. Cross-seed consistency findings",
        "",
    ])
    for row in top_consistency:
        lines.append(f"- `{row['architecture']}/{row['router_family']}/{row['init_strategy']}/{row['task_short']}` mean consistency={row['mean_expert_consistency_score']:.4f}")
    lines.extend([
        "",
        "## 6. Diagnostic-router training results",
        "",
        "- A diagnostic-router training manifest has been generated for the requested signal-only router variants, but those jobs were not launched in this pass.",
        "",
        "## 7. Final interpretation of Week31",
        "",
        "- Experts already show measurable specialization structure in usage frequency, gate mass, missingness pattern, and interaction profile where those features are logged.",
        "- PHENO gains are still most consistent with task-conditioned assignment bias rather than with soft collaborative routing.",
        "- LOS and average IHM gains are most consistent with interaction-aware and missingness-aware assignment bias.",
        "- FLAME demonstrates softer routing, but the softer regime did not improve PHENO.",
    ])
    (FOLLOWUP_ROOT / "week31_followup_report.md").write_text("\n".join(lines))


def main() -> None:
    ensure_followup_root()
    runs = load_completed_runs()
    profile_rows = build_expert_profiles(runs)
    write_csv(
        FOLLOWUP_ROOT / "expert_interaction_profiles.csv",
        profile_rows,
        [
            "config_id", "task_short", "architecture", "router_family", "init_strategy", "seed", "expert_id",
            "decision_count", "selection_frequency", "gate_mass_mean", "top1_frequency", "entropy_mean",
            "text_missing_rate", "cxr_missing_rate", "ecg_missing_rate", "mean_los_label", "mortality_rate",
            "top_phenotype_prevalences", *[f"mean_{col}" for col in INTERACTION_COLUMNS], "sample_ids_top1",
        ],
    )
    write_expert_profile_report(profile_rows)
    hard_rows, hard_report = build_hard_vs_soft(runs)
    write_csv(
        FOLLOWUP_ROOT / "hard_vs_soft_routing.csv",
        hard_rows,
        ["task_scope", "x_metric", "y_metric", "n", "spearman", "pearson", "slope", "intercept", "r2",
         "mean_performance", "mean_gate_entropy", "mean_top1_weight", "mean_active_experts", "mean_routing_margin"],
    )
    consistency_rows = build_cross_seed_consistency(profile_rows)
    write_csv(
        FOLLOWUP_ROOT / "cross_seed_expert_consistency.csv",
        consistency_rows,
        ["architecture", "router_family", "init_strategy", "task_short", "mean_expert_consistency_score",
         "best_matched_expert_pairs", "worst_matched_expert_pairs"],
    )
    write_final_report(profile_rows, hard_report, consistency_rows)


if __name__ == "__main__":
    main()
