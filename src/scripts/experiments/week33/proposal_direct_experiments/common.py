from __future__ import annotations

import csv
import re
from pathlib import Path


FINAL_METRIC_PATTERN = re.compile(r"^([A-Za-z0-9_]+):\s+([-+eE0-9\.]+)\s*$")


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_short(task_name: str) -> str:
    text = (task_name or "").lower()
    if "pheno" in text:
        return "pheno"
    if "los" in text:
        return "los"
    return "ihm"


def parse_final_metrics(log_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    capture = False
    if not log_path.exists():
        return metrics
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("===== FINAL TEST RESULTS =====") or line.startswith("===== EVAL RESULTS"):
            capture = True
            continue
        if capture and line.startswith("==="):
            break
        if not capture:
            continue
        match = FINAL_METRIC_PATTERN.match(line)
        if not match:
            continue
        key, value = match.groups()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def parse_router_summary(log_path: Path) -> dict[str, float]:
    summary = {
        "gate_entropy": None,
        "top1_weight": None,
    }
    if not log_path.exists():
        return summary
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if "specialization diagnostics avg" not in line:
            continue
        for chunk in line.split(","):
            chunk = chunk.strip()
            if "mean_topk_entropy:" in chunk:
                try:
                    summary["gate_entropy"] = float(chunk.split(":", 1)[1])
                except ValueError:
                    pass
            if "mean_top1_weight:" in chunk:
                try:
                    summary["top1_weight"] = float(chunk.split(":", 1)[1])
                except ValueError:
                    pass
    return summary

