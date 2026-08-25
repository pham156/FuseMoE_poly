#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path

from common import (
    INTERACTION_COLUMNS,
    MECH_ROOT,
    cosine_sim,
    ensure_week32_dirs,
    jaccard,
    load_completed_runs,
    mean,
    parse_binary_label,
    parse_mask,
    parse_pheno_vector,
    parse_topk_string,
    read_csv,
    task_short,
    topk_entropy,
    try_float,
    write_csv,
)


def build_expert_profiles(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        diag_path = Path(run["output_dir"]) / "router_diagnostics_test.csv"
        diag_rows = read_csv(diag_path)
        if not diag_rows:
            continue
        by_expert = defaultdict(
            lambda: {
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
            }
        )
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
                "gating_function": run["gating_function"],
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


def build_cross_seed_reproducibility(profile_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in profile_rows:
        grouped[
            (
                row["architecture"],
                row["router_family"],
                row["init_strategy"],
                row["gating_function"],
                row["task_short"],
            )
        ].append(row)

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
        if pair_scores:
            out_rows.append(
                {
                    "architecture": group_key[0],
                    "router_family": group_key[1],
                    "init_strategy": group_key[2],
                    "gating_function": group_key[3],
                    "task_short": group_key[4],
                    "mean_expert_reproducibility_score": mean(pair_scores),
                    "best_matched_expert_pairs": ";".join(best_pairs),
                    "worst_matched_expert_pairs": ";".join(worst_pairs),
                }
            )
    return out_rows


def write_report(profile_rows: list[dict], repro_rows: list[dict]) -> None:
    top_rows = sorted(repro_rows, key=lambda row: row["mean_expert_reproducibility_score"], reverse=True)[:15]
    lines = [
        "# Week32 Cross-Seed Expert Reproducibility",
        "",
        "Paper inspiration: No direct precedent in surveyed multimodal MoE literature.",
        "Adaptation: Expert profiles are built from corrected Week31 router diagnostics, then matched across seeds by profile similarity and cohort overlap.",
        "Novel contribution: Cross-seed reproducibility analysis for multimodal ICU experts under different router families and gating functions.",
        "",
        f"- expert profile rows: {len(profile_rows)}",
        f"- reproducibility groups with complete seeds: {len(repro_rows)}",
        "",
        "## Top reproducible settings",
    ]
    for row in top_rows:
        lines.append(
            f"- `{row['architecture']}/{row['router_family']}/{row['init_strategy']}/{row['gating_function']}/{row['task_short']}`: "
            f"{row['mean_expert_reproducibility_score']:.4f}"
        )
    (MECH_ROOT / "cross_seed_expert_reproducibility_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_week32_dirs()
    runs = load_completed_runs()
    profile_rows = build_expert_profiles(runs)
    write_csv(
        MECH_ROOT / "expert_interaction_profiles.csv",
        profile_rows,
        [
            "config_id",
            "task_short",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "seed",
            "expert_id",
            "decision_count",
            "selection_frequency",
            "gate_mass_mean",
            "top1_frequency",
            "entropy_mean",
            "text_missing_rate",
            "cxr_missing_rate",
            "ecg_missing_rate",
            "mean_los_label",
            "mortality_rate",
            "top_phenotype_prevalences",
            *[f"mean_{col}" for col in INTERACTION_COLUMNS],
            "sample_ids_top1",
        ],
    )
    repro_rows = build_cross_seed_reproducibility(profile_rows)
    write_csv(
        MECH_ROOT / "cross_seed_expert_reproducibility.csv",
        repro_rows,
        [
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "task_short",
            "mean_expert_reproducibility_score",
            "best_matched_expert_pairs",
            "worst_matched_expert_pairs",
        ],
    )
    write_report(profile_rows, repro_rows)


if __name__ == "__main__":
    main()
