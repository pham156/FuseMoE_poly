#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from common import MECH_ROOT, WEEK31_SUMMARY, read_csv, write_csv


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(ordered):
        j = idx
        while j < len(ordered) and ordered[j][0] == ordered[idx][0]:
            j += 1
        avg_rank = 0.5 * (idx + j - 1) + 1.0
        for _, original_idx in ordered[idx:j]:
            ranks[original_idx] = avg_rank
        idx = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return math.nan
    return num / (denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    return pearson(rankdata(xs), rankdata(ys))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def build_merged_rows() -> list[dict]:
    perf_rows = {row["config_id"]: row for row in read_csv(WEEK31_SUMMARY)}
    align_rows = {row["config_id"]: row for row in read_csv(MECH_ROOT / "interaction_cluster_alignment.csv")}

    repro_rows = {}
    for row in read_csv(MECH_ROOT / "cross_seed_expert_reproducibility.csv"):
        key = (
            row["architecture"],
            row["router_family"],
            row["init_strategy"],
            row["gating_function"],
            row["task_short"],
        )
        repro_rows[key] = row

    stability_by_cfg: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(MECH_ROOT / "routing_stability.csv"):
        stability_by_cfg[row["config_id"]].append(row)

    merged = []
    cfg_ids = set(perf_rows)
    for cfg in sorted(cfg_ids):
        perf = perf_rows[cfg]
        stab_rows = stability_by_cfg.get(cfg, [])
        align = align_rows.get(cfg)
        repro_key = (
            perf["architecture"],
            perf["router_family"],
            perf["init_strategy"],
            perf["gating_function"],
            "pheno",
        )
        repro = repro_rows.get(repro_key)
        by_mask = {row["mask_name"]: row for row in stab_rows}
        js_vals = [float(r["js_full_masked"]) for r in stab_rows]
        kl_vals = [float(r["kl_full_to_masked"]) for r in stab_rows]
        l1_vals = [float(r["l1_full_masked"]) for r in stab_rows]
        merged.append(
            {
                "config_id": cfg,
                "architecture": perf["architecture"],
                "router_family": perf["router_family"],
                "init_strategy": perf["init_strategy"],
                "gating_function": perf["gating_function"],
                "seed": perf["seed"],
                "macro_f1": perf["macro_f1"],
                "micro_f1": perf["micro_f1"],
                "nmi_interaction_cluster_vs_expert": align["nmi_interaction_cluster_vs_expert"] if align else "",
                "ari_interaction_cluster_vs_expert": align["ari_interaction_cluster_vs_expert"] if align else "",
                "mean_expert_reproducibility_score": repro["mean_expert_reproducibility_score"] if repro else "",
                "js_full_masked_mean": f"{mean(js_vals):.12f}" if js_vals else "",
                "kl_full_to_masked_mean": f"{mean(kl_vals):.12f}" if kl_vals else "",
                "l1_full_masked_mean": f"{mean(l1_vals):.12f}" if l1_vals else "",
                "js_no_text": by_mask.get("no_text", {}).get("js_full_masked", ""),
                "js_no_cxr": by_mask.get("no_cxr", {}).get("js_full_masked", ""),
                "js_no_ecg": by_mask.get("no_ecg", {}).get("js_full_masked", ""),
                "js_only_ts": by_mask.get("only_ts", {}).get("js_full_masked", ""),
            }
        )
    return merged


def correlation_rows(merged: list[dict]) -> list[dict]:
    metric_names = [
        "nmi_interaction_cluster_vs_expert",
        "ari_interaction_cluster_vs_expert",
        "mean_expert_reproducibility_score",
        "js_full_masked_mean",
        "kl_full_to_masked_mean",
        "l1_full_masked_mean",
        "js_no_text",
        "js_no_cxr",
        "js_no_ecg",
        "js_only_ts",
    ]
    rows = []
    for metric in metric_names:
        paired = []
        for row in merged:
            if row.get(metric, "") == "":
                continue
            paired.append((float(row[metric]), float(row["macro_f1"])))
        xs = [x for x, _ in paired]
        ys = [y for _, y in paired]
        rows.append(
            {
                "x_metric": metric,
                "y_metric": "macro_f1",
                "n": len(xs),
                "spearman": spearman(xs, ys),
                "pearson": pearson(xs, ys),
            }
        )
    return rows


def family_alignment_rows(merged: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in merged:
        if row["nmi_interaction_cluster_vs_expert"] == "":
            continue
        grouped[(row["router_family"], row["gating_function"])].append(row)
    rows = []
    for (router_family, gating_function), subset in sorted(grouped.items()):
        rows.append(
            {
                "router_family": router_family,
                "gating_function": gating_function,
                "n_configs": len(subset),
                "mean_macro_f1": mean([float(r["macro_f1"]) for r in subset]),
                "mean_nmi": mean([float(r["nmi_interaction_cluster_vs_expert"]) for r in subset]),
                "mean_ari": mean([float(r["ari_interaction_cluster_vs_expert"]) for r in subset]),
            }
        )
    return rows


def family_rows(merged: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in merged:
        grouped[(row["router_family"], row["gating_function"])].append(row)
    rows = []
    for (router_family, gating_function), subset in sorted(grouped.items()):
        def maybe_mean(field: str) -> float:
            vals = [float(r[field]) for r in subset if r.get(field, "") != ""]
            return mean(vals)

        rows.append(
            {
                "router_family": router_family,
                "gating_function": gating_function,
                "n_configs": len(subset),
                "mean_macro_f1": maybe_mean("macro_f1"),
                "mean_nmi": maybe_mean("nmi_interaction_cluster_vs_expert"),
                "mean_reproducibility": maybe_mean("mean_expert_reproducibility_score"),
                "mean_js_full_masked": maybe_mean("js_full_masked_mean"),
            }
        )
    return rows


def write_report(merged: list[dict], corr: list[dict], fam: list[dict], align_fam: list[dict]) -> None:
    top_perf = sorted(merged, key=lambda row: float(row["macro_f1"]), reverse=True)[:10]
    top_align = sorted(
        [row for row in merged if row["nmi_interaction_cluster_vs_expert"] != ""],
        key=lambda row: float(row["nmi_interaction_cluster_vs_expert"]),
        reverse=True,
    )[:10]
    strongest = sorted(corr, key=lambda row: abs(float(row["spearman"])) if row["spearman"] == row["spearman"] else -1.0, reverse=True)
    lines = [
        "# Week32 Mechanism Signal Summary",
        "",
        f"- merged checkpoint rows: {len(merged)}",
        f"- correlation rows: {len(corr)}",
        "",
        "## Strongest correlations with PHENO macro-F1",
    ]
    for row in strongest[:6]:
        lines.append(f"- `{row['x_metric']}`: n={row['n']}, spearman={float(row['spearman']):.4f}, pearson={float(row['pearson']):.4f}")
    lines.extend([
        "",
        "## Family means",
    ])
    for row in fam:
        lines.append(
            f"- `{row['router_family']}/{row['gating_function']}`: "
            f"macro-F1={float(row['mean_macro_f1']):.4f}, "
            f"NMI={float(row['mean_nmi']) if row['mean_nmi']==row['mean_nmi'] else float('nan'):.4f}, "
            f"repro={float(row['mean_reproducibility']) if row['mean_reproducibility']==row['mean_reproducibility'] else float('nan'):.4f}, "
            f"JS={float(row['mean_js_full_masked']):.4f}"
        )
    if align_fam:
        lines.extend([
            "",
            "## Alignment-family means",
        ])
        for row in align_fam:
            lines.append(
                f"- `{row['router_family']}/{row['gating_function']}`: "
                f"macro-F1={float(row['mean_macro_f1']):.4f}, "
                f"NMI={float(row['mean_nmi']):.4f}, "
                f"ARI={float(row['mean_ari']):.4f}"
            )
    lines.extend([
        "",
        "## Top performance rows",
    ])
    for row in top_perf:
        js_text = f"{float(row['js_full_masked_mean']):.4f}" if row["js_full_masked_mean"] != "" else "NA"
        lines.append(f"- `{row['config_id']}`: macro-F1={float(row['macro_f1']):.4f}, JS={js_text}")
    lines.extend([
        "",
        "## Top interaction-alignment rows",
    ])
    for row in top_align:
        lines.append(
            f"- `{row['config_id']}`: NMI={float(row['nmi_interaction_cluster_vs_expert']):.4f}, "
            f"ARI={float(row['ari_interaction_cluster_vs_expert']):.4f}, macro-F1={float(row['macro_f1']):.4f}"
        )
    (MECH_ROOT / "mechanism_signal_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    merged = build_merged_rows()
    corr = correlation_rows(merged)
    fam = family_rows(merged)
    align_fam = family_alignment_rows(merged)
    write_csv(
        MECH_ROOT / "mechanism_signal_merged.csv",
        merged,
        [
            "config_id",
            "architecture",
            "router_family",
            "init_strategy",
            "gating_function",
            "seed",
            "macro_f1",
            "micro_f1",
            "nmi_interaction_cluster_vs_expert",
            "ari_interaction_cluster_vs_expert",
            "mean_expert_reproducibility_score",
            "js_full_masked_mean",
            "kl_full_to_masked_mean",
            "l1_full_masked_mean",
            "js_no_text",
            "js_no_cxr",
            "js_no_ecg",
            "js_only_ts",
        ],
    )
    write_csv(
        MECH_ROOT / "mechanism_signal_correlations.csv",
        corr,
        ["x_metric", "y_metric", "n", "spearman", "pearson"],
    )
    write_csv(
        MECH_ROOT / "mechanism_signal_family_summary.csv",
        fam,
        ["router_family", "gating_function", "n_configs", "mean_macro_f1", "mean_nmi", "mean_reproducibility", "mean_js_full_masked"],
    )
    write_csv(
        MECH_ROOT / "mechanism_signal_alignment_family_summary.csv",
        align_fam,
        ["router_family", "gating_function", "n_configs", "mean_macro_f1", "mean_nmi", "mean_ari"],
    )
    write_report(merged, corr, fam, align_fam)


if __name__ == "__main__":
    main()
