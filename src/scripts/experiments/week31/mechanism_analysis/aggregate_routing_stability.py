#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
MECH_ROOT = REPO_ROOT / "out" / "Week_31" / "mechanism_analysis"
MANIFEST_PATH = MECH_ROOT / "week31_mechanism_masked_eval_manifest.tsv"


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_sparse_distribution(text: str, num_experts: int = 4) -> list[float]:
    dist = [0.0] * num_experts
    if not text:
        return dist
    total = 0.0
    for item in text.split("|"):
        if ":" not in item:
            continue
        idx_text, weight_text = item.split(":", 1)
        try:
            idx = int(idx_text)
            weight = float(weight_text)
        except ValueError:
            continue
        if 0 <= idx < num_experts:
            dist[idx] += max(weight, 0.0)
            total += max(weight, 0.0)
    if total > 0:
        dist = [v / total for v in dist]
    return dist


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


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


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

    summary_path = MECH_ROOT / "routing_stability.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)
    sample_path = MECH_ROOT / "routing_stability_per_sample.csv"
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()) if sample_rows else [])
        if sample_rows:
            writer.writeheader()
            writer.writerows(sample_rows)

    report_lines = [
        "# Week31 Routing Stability",
        "",
        f"- completed eval rows aggregated: {len(out_rows)}",
        f"- per-sample comparisons: {len(sample_rows)}",
        "- metrics: KL(full || masked), JS(full, masked), L1(full, masked)",
    ]
    (MECH_ROOT / "routing_stability_report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
