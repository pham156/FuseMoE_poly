#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from common import MECH_ROOT, ensure_week32_dirs, load_completed_runs

MASKS = {
    "no_text": "text",
    "no_cxr": "cxr",
    "no_ecg": "ecg",
    "only_ts": "text,cxr,ecg",
}

FIELDS = [
    "modeltype",
    "num_modalities",
    "gating_function",
    "runtime_gating_function",
    "poly_power",
    "runtime_poly_power",
    "router_type",
    "router_init",
    "router_init_std",
    "use_task_condition_router",
    "task_condition_stats",
    "use_task_specific_router_heads",
    "multitask_shared_moe_trunk",
    "use_modality_mask_condition_router",
    "mask_condition_stats",
    "use_interaction_router",
    "interaction_router_only",
    "interaction_router_scale",
    "shared_semantic_memory_mode",
    "shared_semantic_memory_slots",
    "shared_semantic_memory_heads",
    "use_missing_modality_proxies",
    "use_cross_modal_missing_proxies",
    "missing_modality_proxy_init_std",
    "cross_modal_proxy_hidden",
    "cross_modal_proxy_dropout",
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


def build_rows() -> list[dict]:
    rows = []
    for run in load_completed_runs():
        if not run.get("checkpoint_path") or not Path(run["checkpoint_path"]).exists():
            continue
        for mask_name, forced_missing in MASKS.items():
            output_dir = (
                MECH_ROOT
                / "masked_eval"
                / run["architecture"]
                / run["router_family"]
                / run["init_strategy"]
                / run["task_short"]
                / run["gating_function"]
                / f"seed_{run['seed']}"
                / f"{run['config_id']}__{mask_name}"
            )
            rec = {
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
                "output_dir": str(output_dir),
            }
            for field in FIELDS:
                rec[field] = run.get(field, "")
            if not rec["runtime_gating_function"]:
                rec["runtime_gating_function"] = rec["gating_function"]
            if not rec["runtime_poly_power"]:
                rec["runtime_poly_power"] = rec["poly_power"]
            rows.append(rec)
    return rows


def main() -> None:
    ensure_week32_dirs()
    rows = build_rows()
    manifest_path = MECH_ROOT / "week32_masked_eval_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [], delimiter="\t")
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Week32 Masked Eval Manifest",
        "",
        "Paper inspiration: FuseMoE (NeurIPS 2024) for stress testing, extended here with routing-stability analysis.",
        "Adaptation: Corrected Week31 checkpoints are replayed under fixed missing-modality masks while preserving their original routing settings.",
        "Novel contribution: Enables per-checkpoint routing drift measurement under modality removal.",
        "",
        f"- manifest: `{manifest_path}`",
        f"- eval rows: {len(rows)}",
        f"- unique checkpoints: {len({row['config_id'] for row in rows})}",
        "- masks: `no_text`, `no_cxr`, `no_ecg`, `only_ts`",
    ]
    (MECH_ROOT / "week32_masked_eval_manifest.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
