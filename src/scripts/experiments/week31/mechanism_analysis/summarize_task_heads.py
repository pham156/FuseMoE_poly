#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
WEEK31_ROOT = REPO_ROOT / "out" / "Week_31"
MECH_ROOT = WEEK31_ROOT / "mechanism_analysis"
LOG_ROOT = WEEK31_ROOT / "logs"
MANIFEST_PATH = WEEK31_ROOT / "manifests" / "week31_targeted_followup_manifest.tsv"
AGG_PATH = WEEK31_ROOT / "aggregates" / "week31_results_run_level.csv"

BEST_VALIDATION_RE = re.compile(r"Best validation \(([^)]+)\):\s*([0-9eE+\-.]+)")
SPECIALIZATION_RE = re.compile(
    r"Test specialization diagnostics avg active_experts_per_sample:(?P<active>[0-9eE+\-.]+), "
    r"gate_batch_variance:(?P<batch_var>[0-9eE+\-.]+), "
    r"gate_entropy:(?P<entropy>[0-9eE+\-.]+), "
    r"gate_top12_margin:(?P<margin>[0-9eE+\-.]+), "
    r"gate_top1_weight:(?P<top1>[0-9eE+\-.]+)"
)
METRIC_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.+?)\s*$")


def read_tsv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_from_aggregate(row: dict) -> float | None:
    task_short = row["task_short"]
    if task_short == "pheno":
        return safe_float(row.get("pheno_macro_f1"))
    return safe_float(row.get("binary_f1"))


def parse_task_head_log(path: Path) -> dict:
    lines = path.read_text(errors="ignore").splitlines()
    best_metric = None
    best_value = None
    spec = None
    in_final = False
    current_task = None
    task_metrics: dict[str, dict[str, float]] = {}
    for line in lines:
        match = BEST_VALIDATION_RE.search(line)
        if match:
            best_metric = match.group(1)
            best_value = float(match.group(2))
        match = SPECIALIZATION_RE.search(line)
        if match:
            spec = {
                "active_experts": float(match.group("active")),
                "gate_batch_variance": float(match.group("batch_var")),
                "gate_entropy": float(match.group("entropy")),
                "gate_top12_margin": float(match.group("margin")),
                "gate_top1_weight": float(match.group("top1")),
            }
        if "===== FINAL TEST RESULTS =====" in line:
            in_final = True
            current_task = None
            continue
        if not in_final:
            continue
        stripped = line.strip()
        if stripped in {"[ihm]", "[los]", "[pheno]"}:
            current_task = stripped.strip("[]")
            task_metrics[current_task] = {}
            continue
        if current_task is None:
            continue
        metric_match = METRIC_RE.match(stripped)
        if metric_match:
            key, value = metric_match.groups()
            scalar = safe_float(value)
            if scalar is not None:
                task_metrics[current_task][key] = scalar
    return {
        "best_validation_metric": best_metric,
        "best_validation_value": best_value,
        "specialization": spec or {},
        "task_metrics": task_metrics,
    }


def summarize(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def fmt(value: float | None) -> str:
    return "NA" if value is None or math.isnan(value) else f"{value:.4f}"


def main() -> None:
    MECH_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [r for r in read_tsv(MANIFEST_PATH) if r.get("experiment_group") == "task_head"]
    agg_rows = read_csv(AGG_PATH)

    baseline_lookup = {}
    task_router_lookup = {}
    for row in agg_rows:
        key = (row["architecture"], row["task_short"], str(row["seed"]))
        if row["router_family"] == "joint_router" and row["init_strategy"] == "random":
            baseline_lookup[key] = row
        if row["router_family"] == "task_router" and row["init_strategy"] == "random":
            task_router_lookup[key] = row

    run_rows: list[dict] = []
    for row in manifest_rows:
        config_id = row["config_id"]
        logs = sorted(LOG_ROOT.glob(f"*{config_id}*.out"))
        if not logs:
            continue
        parsed = parse_task_head_log(logs[-1])
        spec = parsed["specialization"]
        for task_short, primary_metric_key in [("ihm", "f1"), ("los", "f1"), ("pheno", "macro_f1")]:
            metrics = parsed["task_metrics"].get(task_short, {})
            primary_value = safe_float(metrics.get(primary_metric_key))
            compare_key = (row["architecture"], task_short, row["seed"])
            baseline_value = metric_from_aggregate(baseline_lookup.get(compare_key, {})) if compare_key in baseline_lookup else None
            task_router_value = metric_from_aggregate(task_router_lookup.get(compare_key, {})) if compare_key in task_router_lookup else None
            run_rows.append(
                {
                    "config_id": config_id,
                    "variant_name": row["variant_name"],
                    "comparison_family": "task_head",
                    "architecture": row["architecture"],
                    "seed": int(row["seed"]),
                    "task_short": task_short,
                    "metric_name": primary_metric_key,
                    "metric_value": primary_value,
                    "auc": safe_float(metrics.get("auc")),
                    "auprc": safe_float(metrics.get("auprc")),
                    "pheno_auc_micro": safe_float(metrics.get("ave_auc_micro")),
                    "pheno_auc_macro": safe_float(metrics.get("ave_auc_macro")),
                    "pheno_auc_weighted": safe_float(metrics.get("ave_auc_weighted")),
                    "best_validation_metric": parsed["best_validation_metric"],
                    "best_validation_value": parsed["best_validation_value"],
                    "active_experts": spec.get("active_experts"),
                    "gate_batch_variance": spec.get("gate_batch_variance"),
                    "gate_entropy": spec.get("gate_entropy"),
                    "gate_top12_margin": spec.get("gate_top12_margin"),
                    "gate_top1_weight": spec.get("gate_top1_weight"),
                    "baseline_router_metric": baseline_value,
                    "task_router_metric": task_router_value,
                    "delta_vs_baseline_router": None if primary_value is None or baseline_value is None else primary_value - baseline_value,
                    "delta_vs_task_router": None if primary_value is None or task_router_value is None else primary_value - task_router_value,
                    "log_path": str(logs[-1]),
                    "output_dir": row["output_dir"],
                }
            )

    write_csv(
        MECH_ROOT / "task_router_head_results.csv",
        run_rows,
        list(run_rows[0].keys()) if run_rows else [],
    )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["architecture"], row["task_short"], row["variant_name"])].append(row)

    summary_rows = []
    for (architecture, task_short, variant_name), rows in sorted(grouped.items()):
        metric_mean, metric_std = summarize([safe_float(r["metric_value"]) for r in rows])
        delta_base_mean, delta_base_std = summarize([safe_float(r["delta_vs_baseline_router"]) for r in rows])
        delta_task_mean, delta_task_std = summarize([safe_float(r["delta_vs_task_router"]) for r in rows])
        entropy_mean, entropy_std = summarize([safe_float(r["gate_entropy"]) for r in rows])
        top1_mean, top1_std = summarize([safe_float(r["gate_top1_weight"]) for r in rows])
        active_mean, active_std = summarize([safe_float(r["active_experts"]) for r in rows])
        summary_rows.append(
            {
                "architecture": architecture,
                "task_short": task_short,
                "variant_name": variant_name,
                "n_seeds": len(rows),
                "metric_mean": metric_mean,
                "metric_std": metric_std,
                "delta_vs_baseline_router_mean": delta_base_mean,
                "delta_vs_baseline_router_std": delta_base_std,
                "delta_vs_task_router_mean": delta_task_mean,
                "delta_vs_task_router_std": delta_task_std,
                "gate_entropy_mean": entropy_mean,
                "gate_entropy_std": entropy_std,
                "gate_top1_weight_mean": top1_mean,
                "gate_top1_weight_std": top1_std,
                "active_experts_mean": active_mean,
                "active_experts_std": active_std,
            }
        )

    write_csv(
        MECH_ROOT / "task_router_head_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )

    lines = [
        "# Week31 Task Router Head Summary",
        "",
        "- Current date: 2026-06-25.",
        "- Scope: corrected multitask task-head reruns only.",
        "- Comparison baselines: single-task `joint_router` random baseline and single-task Week31 `task_router` random runs from `week31_results_run_level.csv`.",
        "- Caveat: the new task-head runs are multitask shared-trunk models, so the comparison against single-task baselines isolates routing-policy changes imperfectly.",
        "",
        "## Mean primary-metric deltas across seeds",
        "",
        "| Architecture | Task | Variant | Mean Metric | Delta vs Baseline | Delta vs Task Router | Gate Entropy | Top1 Weight | Active Experts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['architecture']} | {row['task_short']} | {row['variant_name']} | "
            f"{fmt(row['metric_mean'])} | {fmt(row['delta_vs_baseline_router_mean'])} | "
            f"{fmt(row['delta_vs_task_router_mean'])} | {fmt(row['gate_entropy_mean'])} | "
            f"{fmt(row['gate_top1_weight_mean'])} | {fmt(row['active_experts_mean'])} |"
        )

    MECH_ROOT.joinpath("task_router_head_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
