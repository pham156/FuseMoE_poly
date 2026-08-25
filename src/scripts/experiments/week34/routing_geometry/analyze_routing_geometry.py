#!/usr/bin/env python3
"""Analyze whether router geometry explains expert assignment.

This script supports two input levels:
1. Legacy router diagnostics with only top-k gates. It reports routing balance
   and entropy.
2. Geometry-rich diagnostics produced with --log_router_geometry. It also
   measures router-input centroid separation by expert and the alignment
   between clean router logits and selected experts.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


GATE_RE = re.compile(r"(?:m(?P<modality>\d+)=)?(?:e)?(?P<expert>\d+):(?P<weight>[-+0-9.eE]+)")


def parse_gate_string(value: str) -> dict[int, dict[int, float]]:
    by_modality: dict[int, dict[int, float]] = defaultdict(dict)
    if not isinstance(value, str) or not value:
        return by_modality
    current_modality = 0
    for part in value.split(";"):
        if "=" in part:
            prefix, payload = part.split("=", 1)
            if prefix.startswith("m") and prefix[1:].isdigit():
                current_modality = int(prefix[1:])
            part = payload
        for token in part.split("|"):
            match = GATE_RE.search(token.strip())
            if not match:
                continue
            modality = match.group("modality")
            mod_idx = int(modality) if modality is not None else current_modality
            by_modality[mod_idx][int(match.group("expert"))] = float(match.group("weight"))
    return by_modality


def entropy(weights: list[float]) -> float:
    arr = np.asarray(weights, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return float("nan")
    arr = arr / arr.sum()
    return float(-(arr * np.log(arr + 1e-12)).sum())


def infer_run_metadata(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    meta = {
        "csv_path": str(path),
        "family": parts[0] if len(parts) > 0 else "",
        "variant": "",
        "task": "",
        "seed": "",
        "run_id": path.parent.name,
    }
    for i, part in enumerate(parts):
        if part in {"pheno", "ihm", "los"}:
            meta["task"] = part
            if i > 0:
                meta["variant"] = parts[i - 1]
        if part.startswith("seed_"):
            meta["seed"] = part.replace("seed_", "")
    return meta


def gate_summary(df: pd.DataFrame) -> dict[str, float]:
    top1_counts: Counter[tuple[int, int]] = Counter()
    entropies: list[float] = []
    total = 0
    for value in df.get("topk_expert_ids_weights", []):
        parsed = parse_gate_string(value)
        for modality_idx, weights in parsed.items():
            if not weights:
                continue
            top_expert = max(weights.items(), key=lambda item: item[1])[0]
            top1_counts[(modality_idx, top_expert)] += 1
            entropies.append(entropy(list(weights.values())))
            total += 1
    expert_counts = Counter()
    modality_counts = Counter()
    for (modality_idx, expert_idx), count in top1_counts.items():
        expert_counts[expert_idx] += count
        modality_counts[modality_idx] += count
    if total == 0:
        return {
            "gate_records": 0,
            "mean_gate_entropy": float("nan"),
            "top1_max_frequency": float("nan"),
            "top1_min_frequency": float("nan"),
            "top1_imbalance": float("nan"),
        }
    freqs = np.asarray([expert_counts[i] / total for i in sorted(expert_counts)], dtype=float)
    return {
        "gate_records": total,
        "mean_gate_entropy": float(np.nanmean(entropies)),
        "top1_max_frequency": float(freqs.max()),
        "top1_min_frequency": float(freqs.min()),
        "top1_imbalance": float(freqs.max() - freqs.min()),
    }


def geometry_summary(df: pd.DataFrame) -> dict[str, float | str]:
    input_cols = [col for col in df.columns if col.startswith("router_input_m")]
    logit_cols = [col for col in df.columns if col.startswith("router_clean_logit_m")]
    if not input_cols:
        return {
            "has_router_geometry": 0,
            "centroid_cosine_gap": float("nan"),
            "within_centroid_distance": float("nan"),
            "logit_top1_match_rate": float("nan"),
        }

    input_by_modality: dict[int, list[str]] = defaultdict(list)
    for col in input_cols:
        match = re.match(r"router_input_m(\d+)_d(\d+)", col)
        if match:
            input_by_modality[int(match.group(1))].append(col)
    for cols in input_by_modality.values():
        cols.sort(key=lambda col: int(col.rsplit("_d", 1)[1]))

    logit_by_modality: dict[int, list[str]] = defaultdict(list)
    for col in logit_cols:
        match = re.match(r"router_clean_logit_m(\d+)_e(\d+)", col)
        if match:
            logit_by_modality[int(match.group(1))].append(col)
    for cols in logit_by_modality.values():
        cols.sort(key=lambda col: int(col.rsplit("_e", 1)[1]))

    centroid_gaps = []
    within_distances = []
    logit_matches = []
    for modality_idx, cols in input_by_modality.items():
        assignments = []
        for value in df.get("topk_expert_ids_weights", []):
            parsed = parse_gate_string(value).get(modality_idx, {})
            if not parsed:
                assignments.append(None)
            else:
                assignments.append(max(parsed.items(), key=lambda item: item[1])[0])
        matrix = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        assignment_mask = np.asarray([a is not None for a in assignments], dtype=bool)
        valid = np.isfinite(matrix).all(axis=1) & assignment_mask
        if valid.sum() < 4:
            continue
        matrix = matrix[valid]
        labels = np.asarray([int(a) if a is not None else -1 for a in assignments], dtype=int)[valid]
        centroids = []
        for expert in sorted(set(labels.tolist())):
            rows = matrix[labels == expert]
            if rows.shape[0] < 2:
                continue
            centroid = rows.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
            within_distances.append(float(np.linalg.norm(rows - rows.mean(axis=0), axis=1).mean()))
        if len(centroids) >= 2:
            sims = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    sims.append(float(np.dot(centroids[i], centroids[j])))
            centroid_gaps.append(float(1.0 - np.mean(sims)))

        lcols = logit_by_modality.get(modality_idx, [])
        if lcols:
            logits = df[lcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            logits = logits[valid]
            if logits.size:
                pred = np.nanargmax(logits, axis=1)
                logit_matches.extend((pred == labels).astype(float).tolist())

    return {
        "has_router_geometry": 1,
        "centroid_cosine_gap": float(np.nanmean(centroid_gaps)) if centroid_gaps else float("nan"),
        "within_centroid_distance": float(np.nanmean(within_distances)) if within_distances else float("nan"),
        "logit_top1_match_rate": float(np.nanmean(logit_matches)) if logit_matches else float("nan"),
    }


def analyze_file(path: Path, root: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    row: dict[str, object] = infer_run_metadata(path, root)
    row["n_samples"] = int(len(df))
    row.update(gate_summary(df))
    row.update(geometry_summary(df))
    return row


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir / "routing_geometry_report.md"
    n_files = len(summary)
    geometry_files = int(summary["has_router_geometry"].fillna(0).sum()) if n_files else 0
    task_table = (
        summary.groupby("task", dropna=False)
        .agg(
            runs=("csv_path", "count"),
            mean_gate_entropy=("mean_gate_entropy", "mean"),
            mean_top1_imbalance=("top1_imbalance", "mean"),
            mean_centroid_cosine_gap=("centroid_cosine_gap", "mean"),
            mean_logit_top1_match_rate=("logit_top1_match_rate", "mean"),
        )
        .reset_index()
    )
    with report_path.open("w") as f:
        f.write("# Routing Geometry Diagnostic\n\n")
        f.write(f"Analyzed `{n_files}` router diagnostic files.\n\n")
        f.write(f"Files with router-input geometry columns: `{geometry_files}`.\n\n")
        if geometry_files == 0:
            f.write(
                "Existing diagnostics contain gate assignments but not router input vectors. "
                "They support routing-balance analysis, but not a direct test of hidden-state "
                "geometry versus specialization. Use `--log_router_geometry` in a future eval "
                "or training run to populate the geometry columns.\n\n"
            )
        f.write("## Task Summary\n\n")
        f.write("| task | runs | mean_gate_entropy | mean_top1_imbalance | mean_centroid_cosine_gap | mean_logit_top1_match_rate |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in task_table.iterrows():
            f.write(
                f"| {row['task']} | {int(row['runs'])} | "
                f"{float(row['mean_gate_entropy']):.6f} | "
                f"{float(row['mean_top1_imbalance']):.6f} | "
                f"{float(row['mean_centroid_cosine_gap']):.6f} | "
                f"{float(row['mean_logit_top1_match_rate']):.6f} |\n"
            )
        f.write("\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "`mean_top1_imbalance` is the difference between the most and least selected expert "
            "top-1 frequencies within a run. `centroid_cosine_gap` is available only when router "
            "input vectors are logged; larger values indicate stronger separation between expert "
            "cohort centroids in router-input space.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--pattern", default="**/router_diagnostics_test.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob(args.pattern))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [analyze_file(path, args.input_root) for path in paths]
    summary = pd.DataFrame(rows)
    out_csv = args.output_dir / "routing_geometry_summary.csv"
    summary.to_csv(out_csv, index=False)
    if not summary.empty:
        write_report(summary, args.output_dir)
    print(f"wrote {out_csv}")
    print(f"diagnostic_files={len(paths)}")


if __name__ == "__main__":
    main()
