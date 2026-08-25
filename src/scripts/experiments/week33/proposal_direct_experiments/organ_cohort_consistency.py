#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

from common import read_tsv, write_csv
from layout import AGGREGATE_ROOT, MANIFEST_ROOT, REPORT_ROOT, ensure_week33_dirs


ORGANS = ["cardiovascular", "respiratory", "renal_metabolic", "neurological"]


def _parse_labels(text: str) -> set[str]:
    return {token for token in (text or "").split("|") if token}


def _parse_distribution(text: str, num_experts: int = 4) -> list[float]:
    dist = [0.0] * num_experts
    for item in (text or "").split("|"):
        if ":" not in item:
            continue
        expert_text, weight_text = item.split(":", 1)
        try:
            expert_idx = int(expert_text)
            weight = float(weight_text)
        except ValueError:
            continue
        if 0 <= expert_idx < num_experts:
            dist[expert_idx] = weight
    return dist


def main() -> None:
    ensure_week33_dirs()
    manifest_rows = read_tsv(MANIFEST_ROOT / "week33_train_manifest.tsv")
    result_rows = []
    for row in manifest_rows:
        if row.get("status") != "ready":
            continue
        diag_path = Path(row["output_dir"]) / "router_diagnostics_test.csv"
        if not diag_path.exists():
            continue
        per_organ_positive = defaultdict(list)
        per_organ_negative = defaultdict(list)
        with diag_path.open() as handle:
            reader = csv.DictReader(handle)
            for sample in reader:
                labels = _parse_labels(sample.get("weak_organ_labels", ""))
                dist = _parse_distribution(sample.get("topk_expert_ids_weights", ""))
                for organ_idx, organ_name in enumerate(ORGANS):
                    target = 1 if organ_name in labels else 0
                    gate_mass = dist[organ_idx]
                    if target == 1:
                        per_organ_positive[organ_name].append(gate_mass)
                    else:
                        per_organ_negative[organ_name].append(gate_mass)
        for organ_name in ORGANS:
            pos = per_organ_positive.get(organ_name, [])
            neg = per_organ_negative.get(organ_name, [])
            labels = [1] * len(pos) + [0] * len(neg)
            scores = pos + neg
            auroc = ""
            if pos and neg and len(set(labels)) == 2:
                try:
                    auroc = roc_auc_score(labels, scores)
                except ValueError:
                    auroc = ""
            result_rows.append(
                {
                    "config_id": row["config_id"],
                    "experiment_group": row["experiment_group"],
                    "variant_name": row["variant_name"],
                    "task_short": row["task_short"],
                    "architecture": row["architecture"],
                    "seed": row["seed"],
                    "organ": organ_name,
                    "p_expert_given_positive": sum(pos) / max(len(pos), 1) if pos else "",
                    "p_expert_given_negative": sum(neg) / max(len(neg), 1) if neg else "",
                    "delta_expert_mass": (sum(pos) / max(len(pos), 1) - sum(neg) / max(len(neg), 1)) if pos and neg else "",
                    "auroc_gate_mass": auroc,
                    "n_positive": len(pos),
                    "n_negative": len(neg),
                }
            )
    write_csv(
        AGGREGATE_ROOT / "organ_cohort_consistency.csv",
        result_rows,
        [
            "config_id", "experiment_group", "variant_name", "task_short", "architecture", "seed",
            "organ", "p_expert_given_positive", "p_expert_given_negative", "delta_expert_mass",
            "auroc_gate_mass", "n_positive", "n_negative",
        ],
    )
    (REPORT_ROOT / "organ_cohort_report.md").write_text(
        "\n".join(
            [
                "# Week33 Organ-Cohort Consistency",
                "",
                f"- rows: {len(result_rows)}",
                "- weak organ tags are derived from note-keyword matches already logged into router diagnostics.",
            ]
        ) + "\n"
    )


if __name__ == "__main__":
    main()

