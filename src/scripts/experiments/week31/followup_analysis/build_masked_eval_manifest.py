#!/usr/bin/env python3
from __future__ import annotations

import csv

from common import FOLLOWUP_ROOT, ensure_followup_root, load_completed_runs

MASKS = {
    "no_text": "text",
    "no_cxr": "cxr",
    "no_ecg": "ecg",
    "only_ts": "text,cxr,ecg",
}

PASSTHROUGH_FIELDS = [
    "modeltype",
    "num_modalities",
    "router_type",
    "router_init",
    "router_init_std",
    "use_task_condition_router",
    "task_condition_stats",
    "use_modality_mask_condition_router",
    "mask_condition_stats",
    "use_interaction_router",
    "interaction_router_only",
    "interaction_router_scale",
    "shared_semantic_memory_mode",
    "shared_semantic_memory_slots",
    "shared_semantic_memory_heads",
    "expert_init_strategy",
    "expert_init_target_path",
    "expert_init_epochs",
    "expert_init_soft_targets",
    "expert_init_freeze_router",
    "expert_init_release_schedule",
    "teacher_sweep",
    "teacher_model",
    "log_router_diagnostics",
    "log_expert_output_diagnostics",
    "log_interaction_features",
    "week31_variant_id",
    "week31_router_family",
    "router_zero_input",
]


def main() -> None:
    ensure_followup_root()
    rows = []
    for run in load_completed_runs():
        if not run.get("checkpoint_path"):
            continue
        for mask_name, forced_missing in MASKS.items():
            row = {
                "eval_id": f"{run['config_id']}__{mask_name}",
                "config_id": run["config_id"],
                "task": run["task"],
                "task_short": run["task_short"],
                "architecture": run["architecture"],
                "router_family": run["router_family"],
                "init_strategy": run["init_strategy"],
                "seed": run["seed"],
                "checkpoint_path": run["checkpoint_path"],
                "forced_missing": forced_missing,
                "mask_name": mask_name,
                "output_dir": str(
                    FOLLOWUP_ROOT
                    / "masked_eval"
                    / run["architecture"]
                    / run["router_family"]
                    / run["init_strategy"]
                    / run["task_short"]
                    / f"seed_{run['seed']}"
                    / f"{run['config_id']}__{mask_name}"
                ),
            }
            for field in PASSTHROUGH_FIELDS:
                row[field] = run.get(field, "")
            rows.append(row)
    out_path = FOLLOWUP_ROOT / "week31_masked_eval_manifest.tsv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [], delimiter="\t")
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    stability_csv = FOLLOWUP_ROOT / "routing_stability.csv"
    with stability_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_id",
                "task_short",
                "architecture",
                "router_family",
                "init_strategy",
                "seed",
                "mask_name",
                "kl_full_to_masked",
                "js_full_masked",
                "l1_full_masked",
            ],
        )
        writer.writeheader()
    (FOLLOWUP_ROOT / "routing_stability_report.md").write_text(
        "\n".join(
            [
                "# Week31 Routing Stability Report",
                "",
                f"- masked eval rows prepared: {len(rows)}",
                "- `routing_stability.csv` has been initialized with headers only.",
                "- The actual KL/JS/L1 routing-stability table requires executing these masked eval rows against the saved Week31 checkpoints.",
            ]
        )
    )


if __name__ == "__main__":
    main()
