#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from common import MECH_ROOT, REPO_ROOT, WEEK31_ROUTING, WEEK31_SUMMARY, mean, read_csv, write_csv


HISTORICAL_ROUTING_FORMULATION_ROWS = [
    {
        "source": "historical_no_shared_sweep",
        "router_mode": "joint",
        "xmoe": "no",
        "gate_entropy": 0.003,
        "top1_weight_pct": 99.86,
        "top1_margin_pct": 99.72,
        "mass_range_pct": 95.78,
        "mean_best_val_metric_pct": 59.21,
        "mean_test_auc_pct": 81.56,
        "ihm_f1_pct": 47.04,
        "los_f1_pct": 73.62,
    },
    {
        "source": "historical_no_shared_sweep",
        "router_mode": "joint",
        "xmoe": "yes",
        "gate_entropy": 0.509,
        "top1_weight_pct": 78.80,
        "top1_margin_pct": 57.59,
        "mass_range_pct": 78.79,
        "mean_best_val_metric_pct": 59.55,
        "mean_test_auc_pct": 81.76,
        "ihm_f1_pct": 46.39,
        "los_f1_pct": 73.47,
    },
    {
        "source": "historical_no_shared_sweep",
        "router_mode": "permod",
        "xmoe": "no",
        "gate_entropy": 0.645,
        "top1_weight_pct": 62.12,
        "top1_margin_pct": 24.25,
        "mass_range_pct": 2.15,
        "mean_best_val_metric_pct": 59.66,
        "mean_test_auc_pct": 81.26,
        "ihm_f1_pct": 46.19,
        "los_f1_pct": 74.35,
    },
    {
        "source": "historical_no_shared_sweep",
        "router_mode": "permod",
        "xmoe": "yes",
        "gate_entropy": 0.593,
        "top1_weight_pct": 68.83,
        "top1_margin_pct": 37.65,
        "mass_range_pct": 61.99,
        "mean_best_val_metric_pct": 59.95,
        "mean_test_auc_pct": 81.69,
        "ihm_f1_pct": 44.05,
        "los_f1_pct": 74.36,
    },
]

ARCHIVED_INTERACTION_ONLY_ROOT = REPO_ROOT / "out" / "Week_31_old" / "mechanism_analysis" / "masked_eval" / "moe_ref" / "interaction_only_router"
CURRENT_INTERACTION_ONLY = REPO_ROOT / "out" / "Week_31" / "aggregates" / "interaction_only_router_pheno_summary.csv"


def summarize_feature_vs_interaction() -> list[dict]:
    summary_rows = {row["config_id"]: row for row in read_csv(WEEK31_SUMMARY)}
    routing_rows = read_csv(WEEK31_ROUTING)
    stability_rows = read_csv(MECH_ROOT / "routing_stability.csv")
    current_interaction_only_rows = read_csv(CURRENT_INTERACTION_ONLY)

    families = {"permod_router", "interaction_router", "task_interaction_router"}
    by_group: dict[tuple[str, str], dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "macro_f1": [],
            "mean_top1_weight": [],
            "mean_topk_entropy": [],
            "usage_entropy": [],
            "js_full_masked": [],
            "kl_full_to_masked": [],
            "l1_full_masked": [],
            "n_runs": 0,
            "n_mask_rows": 0,
        }
    )

    for row in routing_rows:
        family = row["router_family"]
        if family not in families:
            continue
        key = (family, row["gating_function"])
        acc = by_group[key]
        acc["n_runs"] += 1
        if row["config_id"] in summary_rows and summary_rows[row["config_id"]].get("macro_f1", "") not in {"", "nan"}:
            acc["macro_f1"].append(float(summary_rows[row["config_id"]]["macro_f1"]))
        for field in ("mean_top1_weight", "mean_topk_entropy", "usage_entropy"):
            if row.get(field, "") not in {"", "nan"}:
                acc[field].append(float(row[field]))

    for row in current_interaction_only_rows:
        if row.get("architecture") != "moe_ref":
            continue
        key = ("interaction_only_router", row["gating_function"])
        acc = by_group[key]
        acc["n_runs"] += 1
        if row.get("macro_f1", "") not in {"", "nan"}:
            acc["macro_f1"].append(float(row["macro_f1"]))
        for field, src in (
            ("mean_top1_weight", "gate_top1_weight"),
            ("mean_topk_entropy", "gate_entropy"),
        ):
            if row.get(src, "") not in {"", "nan"}:
                acc[field].append(float(row[src]))

    for row in stability_rows:
        family = row["router_family"]
        if family not in families:
            continue
        key = (family, row["gating_function"])
        acc = by_group[key]
        acc["n_mask_rows"] += 1
        for field in ("js_full_masked", "kl_full_to_masked", "l1_full_masked"):
            if row.get(field, "") not in {"", "nan"}:
                acc[field].append(float(row[field]))

    out = []
    for (family, gating), acc in sorted(by_group.items()):
        macro_vals = acc["macro_f1"]
        out.append(
            {
                "router_family": family,
                "gating_function": gating,
                "n_runs": acc["n_runs"],
                "n_mask_rows": acc["n_mask_rows"],
                "mean_macro_f1": mean(macro_vals),
                "best_macro_f1": max(macro_vals) if macro_vals else math.nan,
                "mean_top1_weight": mean(acc["mean_top1_weight"]),
                "mean_topk_entropy": mean(acc["mean_topk_entropy"]),
                "mean_usage_entropy": mean(acc["usage_entropy"]),
                "mean_js_full_masked": mean(acc["js_full_masked"]),
                "mean_kl_full_to_masked": mean(acc["kl_full_to_masked"]),
                "mean_l1_full_masked": mean(acc["l1_full_masked"]),
            }
        )
    return out


def summarize_single_vs_permod() -> list[dict]:
    out = list(HISTORICAL_ROUTING_FORMULATION_ROWS)
    current_rows = read_csv(WEEK31_ROUTING)
    pheno_rows = [
        row
        for row in current_rows
        if row["router_family"] == "permod_router" and row.get("macro_f1", "") not in {"", "nan"}
    ]
    if pheno_rows:
        out.append(
            {
                "source": "week31_corrected_pheno_slice",
                "router_mode": "permod",
                "xmoe": "mixed_inits",
                "gate_entropy": mean([float(row["mean_topk_entropy"]) for row in pheno_rows]),
                "top1_weight_pct": 100.0 * mean([float(row["mean_top1_weight"]) for row in pheno_rows]),
                "top1_margin_pct": math.nan,
                "mass_range_pct": math.nan,
                "mean_best_val_metric_pct": math.nan,
                "mean_test_auc_pct": math.nan,
                "ihm_f1_pct": math.nan,
                "los_f1_pct": math.nan,
            }
        )
    return out


def format_float(value: float) -> str:
    return "NA" if value != value else f"{value:.4f}"


def write_reports(feature_rows: list[dict], routing_rows: list[dict]) -> None:
    feature_lines = [
        "# Week32 Feature vs Interaction Router Packaging",
        "",
        "- scope: corrected Week31 PHENO checkpoint set plus finished Week32 masked-eval replay",
        "- comparison set: `permod_router` (feature-only), `interaction_only_router` (interaction-only), `interaction_router` (feature + interaction), `task_interaction_router` (feature + interaction + task conditioning)",
        "- caveat: the corrected `interaction_only_router` PHENO rerun is now included for train/test performance, but it does not yet have matched current masked-eval replay rows, so routing-drift fields remain `NA` for that branch in this packaged summary",
        "",
        "## Family-by-gate summary",
    ]
    for row in feature_rows:
        feature_lines.append(
            f"- `{row['router_family']}/{row['gating_function']}`: "
            f"mean_macro_f1={format_float(row['mean_macro_f1'])}, "
            f"best_macro_f1={format_float(row['best_macro_f1'])}, "
            f"top1={format_float(row['mean_top1_weight'])}, "
            f"entropy={format_float(row['mean_topk_entropy'])}, "
            f"usage_entropy={format_float(row['mean_usage_entropy'])}, "
            f"js_drift={format_float(row['mean_js_full_masked'])}"
        )
    feature_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- In the corrected PHENO slice, interaction-aware routing does not deliver a universal metric advantage over feature-only per-modality routing.",
            "- `task_interaction_router` still has the highest family-level mean macro-F1 in the completed current slice, but `interaction_only_router` now provides the strongest single PHENO run.",
            "- `interaction_only_router` therefore strengthens the claim that interaction signals can be useful, but its mean performance remains in the same narrow band as the surrounding families.",
            "- `interaction_router` and `task_interaction_router` are slightly more stable than `permod_router` under missingness replay by mean JS drift, but the effect size is small.",
            f"- Archived masked-eval coverage for `interaction_only_router` still exists at `{ARCHIVED_INTERACTION_ONLY_ROOT}` if a historical robustness cross-check is needed.",
        ]
    )
    (MECH_ROOT / "feature_vs_interaction_router_report.md").write_text("\n".join(feature_lines) + "\n")

    routing_lines = [
        "# Week32 Single Router vs Per-Modality Router Packaging",
        "",
        "- scope: historical matched no-shared routing-formulation sweep plus current corrected per-modality PHENO slice",
        "- naming: `single router` here refers to historical joint/global routing",
        "",
        "## Historical matched formulation table",
    ]
    for row in routing_rows[:4]:
        label = f"{row['router_mode']}" + (" + X-MoE" if row["xmoe"] == "yes" else "")
        routing_lines.append(
            f"- `{label}`: gate_entropy={format_float(row['gate_entropy'])}, "
            f"top1_weight_pct={format_float(row['top1_weight_pct'])}, "
            f"mass_range_pct={format_float(row['mass_range_pct'])}, "
            f"mean_test_auc_pct={format_float(row['mean_test_auc_pct'])}, "
            f"ihm_f1_pct={format_float(row['ihm_f1_pct'])}, "
            f"los_f1_pct={format_float(row['los_f1_pct'])}"
        )
    routing_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Historical joint/global routing was sharply collapsed, with gate entropy near zero and top-1 weight near one.",
            "- Historical per-modality routing removed that collapse while keeping downstream IHM/LOS metrics in the same range and slightly improving LOS F1 in the matched table.",
            "- The corrected Week31 PHENO campaign therefore concentrated on `permod_router`; it does not provide a fresh matched joint rerun, but it does provide a modern per-modality baseline with mean gate entropy near the two-expert uniform value and mean top-1 weight around 0.52.",
            "- The strongest defensible single-vs-permod conclusion is structural rather than purely metric: per-modality routing is the only routing mode in the current repository with a stable specialization story, while joint routing mainly serves as a collapse control.",
        ]
    )
    (MECH_ROOT / "single_vs_permod_router_report.md").write_text("\n".join(routing_lines) + "\n")


def main() -> None:
    feature_rows = summarize_feature_vs_interaction()
    routing_rows = summarize_single_vs_permod()
    write_csv(
        MECH_ROOT / "feature_vs_interaction_router_summary.csv",
        feature_rows,
        [
            "router_family",
            "gating_function",
            "n_runs",
            "n_mask_rows",
            "mean_macro_f1",
            "best_macro_f1",
            "mean_top1_weight",
            "mean_topk_entropy",
            "mean_usage_entropy",
            "mean_js_full_masked",
            "mean_kl_full_to_masked",
            "mean_l1_full_masked",
        ],
    )
    write_csv(
        MECH_ROOT / "single_vs_permod_router_summary.csv",
        routing_rows,
        [
            "source",
            "router_mode",
            "xmoe",
            "gate_entropy",
            "top1_weight_pct",
            "top1_margin_pct",
            "mass_range_pct",
            "mean_best_val_metric_pct",
            "mean_test_auc_pct",
            "ihm_f1_pct",
            "los_f1_pct",
        ],
    )
    write_reports(feature_rows, routing_rows)


if __name__ == "__main__":
    main()
