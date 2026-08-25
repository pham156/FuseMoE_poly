#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


BEST_VALIDATION_RE = re.compile(r"Best validation \(([^)]+)\):\s*([0-9eE+\-.]+)")
FINAL_METRIC_RE = re.compile(r"^([A-Za-z0-9_]+):\s*([0-9eE+\-.]+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _safe_mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _safe_std(values):
    values = [v for v in values if v is not None]
    return pstdev(values) if len(values) > 1 else 0.0 if values else None


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


def _parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    best_metric = None
    best_value = None
    for match in BEST_VALIDATION_RE.finditer(text):
        best_metric = match.group(1)
        best_value = float(match.group(2))
    final_metrics = {}
    lines = text.splitlines()
    in_final = False
    for line in lines:
        if "===== FINAL TEST RESULTS =====" in line:
            in_final = True
            continue
        if in_final:
            match = FINAL_METRIC_RE.match(line.strip())
            if match:
                final_metrics[match.group(1)] = float(match.group(2))
            elif final_metrics and not line.strip():
                break
    return {
        "best_validation_metric": best_metric,
        "best_validation_value": best_value,
        "final_metrics": final_metrics,
    }


def _parse_router_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    retained = []
    conf = []
    shared = []
    for row in rows:
        if row.get("expert_init_retained", "") != "":
            retained.append(float(row["expert_init_retained"]))
        if row.get("expert_init_confidence", "") != "":
            conf.append(float(row["expert_init_confidence"]))
        if row.get("shared_contribution_ratio", "") != "":
            shared.append(float(row["shared_contribution_ratio"]))
    return {
        "cohort_retention_mean": _safe_mean(retained),
        "confidence_mean": _safe_mean(conf),
        "confidence_std": _safe_std(conf),
        "routed_off_proxy_delta": None if not shared else 1.0 - _safe_mean(shared),
    }


def main() -> None:
    args = parse_args()
    manifest_rows = list(csv.DictReader(Path(args.manifest).open(), delimiter="\t"))
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for row in manifest_rows:
        if str(row.get("should_submit", "")).lower() != "true":
            continue
        logs = sorted(log_dir.glob(f"*{row['config_id']}*.out"))
        if not logs:
            continue
        log_info = _parse_log(logs[-1])
        router_info = _parse_router_csv(Path(row["output_dir"]) / "router_diagnostics_test.csv")
        final_metrics = log_info["final_metrics"]
        results.append(
            {
                "teacher_model": row["teacher_model"],
                "teacher_family": row["teacher_family"],
                "architecture": row["architecture"],
                "task": row["task"],
                "seed": int(row["seed"]),
                "metric": final_metrics.get(row["primary_metric"]),
                "best_validation_value": log_info["best_validation_value"],
                **router_info,
            }
        )

    none_by_arch_seed = {(r["architecture"], r["seed"]): r for r in results if r["teacher_model"] == "none"}
    random_by_arch_seed = {(r["architecture"], r["seed"]): r for r in results if r["teacher_model"] == "random"}
    for row in results:
        none_ref = none_by_arch_seed.get((row["architecture"], row["seed"]))
        random_ref = random_by_arch_seed.get((row["architecture"], row["seed"]))
        row["delta_vs_none"] = None if not none_ref or row["metric"] is None or none_ref["metric"] is None else row["metric"] - none_ref["metric"]
        row["delta_vs_random"] = None if not random_ref or row["metric"] is None or random_ref["metric"] is None else row["metric"] - random_ref["metric"]

    grouped = defaultdict(list)
    for row in results:
        grouped[(row["teacher_model"], row["teacher_family"], row["architecture"], row["task"])].append(row)

    perf_rows = []
    retention_rows = []
    robust_rows = []
    report_rows = []
    interpretation_map = {}
    for (teacher_model, teacher_family, architecture, task), rows in sorted(grouped.items()):
        mean_metric = _safe_mean([r["metric"] for r in rows])
        perf_rows.append(
            {
                "teacher_model": teacher_model,
                "teacher_family": teacher_family,
                "architecture": architecture,
                "task": task,
                "n_seeds": len(rows),
                "mean_metric": mean_metric,
                "std_metric": _safe_std([r["metric"] for r in rows]),
                "delta_vs_none": _safe_mean([r["delta_vs_none"] for r in rows]),
                "delta_vs_random": _safe_mean([r["delta_vs_random"] for r in rows]),
            }
        )
        retention_rows.append(
            {
                "teacher_model": teacher_model,
                "architecture": architecture,
                "task": task,
                "cohort_retention_mean": _safe_mean([r.get("cohort_retention_mean") for r in rows]),
                "cohort_retention_std": _safe_std([r.get("cohort_retention_mean") for r in rows]),
                "confidence_mean": _safe_mean([r.get("confidence_mean") for r in rows]),
                "confidence_std": _safe_std([r.get("confidence_mean") for r in rows]),
            }
        )
        robust_rows.append(
            {
                "teacher_model": teacher_model,
                "architecture": architecture,
                "task": task,
                "mask_condition": "full_only",
                "mean_metric": mean_metric,
                "delta_vs_full": 0.0 if mean_metric is not None else None,
                "delta_vs_none_masked": None,
            }
        )
        d_none = _safe_mean([r["delta_vs_none"] for r in rows])
        d_rand = _safe_mean([r["delta_vs_random"] for r in rows])
        if teacher_model == "random":
            interpretation = "random symmetry-breaking control"
        elif teacher_model == "none":
            interpretation = "baseline"
        elif d_rand is not None and d_rand > 0.005:
            interpretation = "beats random control"
        elif d_none is not None and d_none > 0:
            interpretation = "improves over no-teacher only"
        else:
            interpretation = "no clear advantage"
        interpretation_map[(teacher_model, architecture, task)] = interpretation

    report_index = {(r["teacher_model"], r["architecture"], r["task"]): r for r in perf_rows}
    retention_index = {(r["teacher_model"], r["architecture"], r["task"]): r for r in retention_rows}
    for teacher_model in ["none", "random", "qwen25", "biomistral", "meditron", "me_llama", "medgemma", "lingshu"]:
        for architecture in ["moe_ref", "recon_ref"]:
            key = (teacher_model, architecture, "pheno-all-cxr-notes-ecg")
            perf = report_index.get(key)
            ret = retention_index.get(key, {})
            report_rows.append(
                {
                    "teacher_model": teacher_model,
                    "teacher_family": perf["teacher_family"] if perf is not None else "",
                    "architecture": architecture,
                    "PHENO mean": perf["mean_metric"] if perf is not None else None,
                    "PHENO delta vs none": perf["delta_vs_none"] if perf is not None else None,
                    "PHENO delta vs random": perf["delta_vs_random"] if perf is not None else None,
                    "IHM mean if available": None,
                    "LOS mean if available": None,
                    "cohort retention": ret.get("cohort_retention_mean"),
                    "routed-off delta": None,
                    "best masked robustness delta": None,
                    "interpretation": interpretation_map.get(key, "unavailable or not run"),
                }
            )

    def _write(path: Path, rows: list[dict]):
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write(out_dir / "teacher_sweep_performance.csv", perf_rows)
    _write(out_dir / "teacher_sweep_retention.csv", retention_rows)
    _write(out_dir / "teacher_sweep_robustness.csv", robust_rows)
    _write(out_dir / "teacher_sweep_report_table.csv", report_rows)

    perf_by_teacher = {r["teacher_model"]: r for r in report_rows if r["architecture"] == "recon_ref"}
    random_row = perf_by_teacher.get("random")
    qwen_row = perf_by_teacher.get("qwen25")
    biomistral_row = perf_by_teacher.get("biomistral")
    meditron_row = perf_by_teacher.get("meditron")
    medgemma_row = perf_by_teacher.get("medgemma")
    lingshu_row = perf_by_teacher.get("lingshu")

    findings = []
    findings.append("# Teacher Sweep Findings")
    findings.append("")
    if random_row and lingshu_row and lingshu_row["PHENO delta vs none"] is not None and random_row["PHENO delta vs none"] is not None:
        if abs((lingshu_row["PHENO delta vs none"] or 0.0) - (random_row["PHENO delta vs none"] or 0.0)) <= 0.005:
            findings.append("- Random warm-start performs similarly to the medical teacher arm, so the current gain is more consistent with symmetry breaking than with uniquely medical supervision.")
    if qwen_row and lingshu_row and qwen_row["PHENO mean"] is not None and lingshu_row["PHENO mean"] is not None:
        if abs(qwen_row["PHENO mean"] - lingshu_row["PHENO mean"]) <= 0.005:
            findings.append("- Qwen2.5 is close to the medical teacher arm, which would support the generic-capacity explanation.")
    if biomistral_row and qwen_row and biomistral_row["PHENO mean"] is not None and qwen_row["PHENO mean"] is not None and biomistral_row["PHENO mean"] > qwen_row["PHENO mean"]:
        findings.append("- BioMistral exceeds Qwen2.5, which would support a biomedical-pretraining effect.")
    if meditron_row and biomistral_row and meditron_row["PHENO mean"] is not None and biomistral_row["PHENO mean"] is not None and meditron_row["PHENO mean"] > biomistral_row["PHENO mean"]:
        findings.append("- Meditron exceeds BioMistral, which would support a clinical-domain effect beyond biomedical literature pretraining.")
    if medgemma_row and lingshu_row and medgemma_row["PHENO mean"] is not None and lingshu_row["PHENO mean"] is not None and medgemma_row["PHENO mean"] >= lingshu_row["PHENO mean"]:
        findings.append("- MedGemma matches or exceeds Lingshu, so the previous Lingshu result would not look model-specific.")
    if not any(line.startswith("-") for line in findings[2:]):
        findings.append("- Only available and completed teacher arms can support a strong conclusion. Unavailable teachers remain explicitly skipped rather than silently omitted.")

    (out_dir / "teacher_sweep_findings.md").write_text("\n".join(findings) + "\n")


if __name__ == "__main__":
    main()
