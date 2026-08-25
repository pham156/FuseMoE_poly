#!/usr/bin/env python3
from __future__ import annotations

import csv

from common import FOLLOWUP_ROOT, MANIFEST_ROOT, ensure_followup_root, read_tsv

SCRATCH_ROOT = "/scratch/gilbreth/pham156/MoE/FuseMoE_poly/out/Week_31/followup_analysis/diagnostic_routers"


def main() -> None:
    ensure_followup_root()
    base_rows = {}
    for row in read_tsv(MANIFEST_ROOT / "week31_train_manifest.tsv"):
        if row["architecture"] == "moe_ref" and row["router_family"] == "joint_router" and row["init_strategy"] == "random":
            base_rows[(row["task"], row["seed"])] = row

    variants = [
        ("missingness_only_router", {"use_modality_mask_condition_router": "True", "mask_condition_stats": "True", "router_zero_input": "True"}),
        ("interaction_only_router", {"use_interaction_router": "True", "interaction_router_only": "True"}),
        ("task_only_router", {"use_task_condition_router": "True", "task_condition_stats": "True", "router_zero_input": "True"}),
        ("interaction_missing_only_router", {"use_interaction_router": "True", "use_modality_mask_condition_router": "True", "mask_condition_stats": "True", "router_zero_input": "True"}),
    ]

    rows = []
    for (task, seed), base in sorted(base_rows.items()):
        task_short = "pheno" if "pheno" in task else ("los" if "los" in task else "ihm")
        for variant_name, flags in variants:
            row = dict(base)
            row.update(flags)
            row["config_id"] = f"w31diag_{variant_name}_{task_short}_s{seed}"
            row["router_family"] = variant_name
            row["week31_router_family"] = variant_name
            row["week31_variant_id"] = variant_name
            row["output_dir"] = f"{SCRATCH_ROOT}/{variant_name}/{task_short}/seed_{seed}/{row['config_id']}"
            row["disable_run_folder_save"] = "True"
            row["status"] = "ready"
            row["skip_reason"] = ""
            rows.append(row)

    out_path = FOLLOWUP_ROOT / "week31_diagnostic_router_manifest.tsv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [], delimiter="\t")
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    result_csv = FOLLOWUP_ROOT / "conditioning_signal_ablation.csv"
    with result_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_id", "task_short", "architecture", "router_family", "seed",
                "primary_metric", "primary_value", "active_experts", "gate_entropy", "gate_top1_weight",
            ],
        )
        writer.writeheader()
    (FOLLOWUP_ROOT / "conditioning_signal_ablation_report.md").write_text(
        "\n".join(
            [
                "# Week31 Conditioning Signal Ablation Report",
                "",
                f"- diagnostic training rows prepared: {len(rows)}",
                "- `conditioning_signal_ablation.csv` has been initialized with headers only.",
                "- These rows define the requested signal-only router variants under `moe_ref` for PHENO, IHM, and LOS with seeds `32/42/52`.",
                "- Training was not launched in this pass.",
            ]
        )
    )


if __name__ == "__main__":
    main()
