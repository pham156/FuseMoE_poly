#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
WEEK31_ROOT = REPO_ROOT / "out" / "Week_31"
MECH_ROOT = WEEK31_ROOT / "mechanism_analysis"
AGG_ROOT = WEEK31_ROOT / "aggregates"
MANIFEST_ROOT = WEEK31_ROOT / "manifests"

MASKS = {
    "no_text": "text",
    "no_cxr": "cxr",
    "no_ecg": "ecg",
    "only_ts": "text,cxr,ecg",
}

FIELDS = [
    "modeltype",
    "num_modalities",
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def task_short(task_name: str) -> str:
    text = (task_name or "").lower()
    if "pheno" in text:
        return "pheno"
    if "los" in text:
        return "los"
    return "ihm"


def resolve_checkpoint_path(output_dir: str) -> str:
    if not output_dir:
        return ""
    path = Path(output_dir) / "best_model.pth.tar"
    return str(path) if path.exists() else ""


def load_base_runs() -> list[dict]:
    manifest_lookup = {
        row["config_id"]: row
        for row in read_tsv(MANIFEST_ROOT / "week31_train_manifest.tsv")
    }
    rows = []
    for row in read_csv(AGG_ROOT / "week31_results_run_level.csv"):
        manifest = manifest_lookup.get(row["config_id"], {})
        output_dir = manifest.get("output_dir", "")
        checkpoint_path = resolve_checkpoint_path(output_dir)
        if not checkpoint_path:
            continue
        rec = {
            **manifest,
            **row,
            "task_short": row.get("task_short") or task_short(row.get("task", "")),
            "output_dir": output_dir,
            "checkpoint_path": checkpoint_path,
        }
        rows.append(rec)
    return rows


def load_flame_runs(existing_ids: set[str]) -> list[dict]:
    manifest_lookup = {
        row["config_id"]: row
        for row in read_tsv(MANIFEST_ROOT / "week31_train_manifest.tsv")
    }
    flame_jobmap = {
        row["config_id"]: row
        for row in read_tsv(MANIFEST_ROOT / "week31_flame_confirmation_jobs.tsv")
    }
    rows = []
    for row in read_csv(AGG_ROOT / "week31_flame_confirmation_results.csv"):
        cfg = row["config_id"]
        if cfg in existing_ids:
            continue
        source = flame_jobmap.get(cfg, manifest_lookup.get(cfg, {}))
        output_dir = source.get("output_dir", manifest_lookup.get(cfg, {}).get("output_dir", ""))
        checkpoint_path = resolve_checkpoint_path(output_dir)
        if not checkpoint_path:
            continue
        rows.append(
            {
                **manifest_lookup.get(cfg, {}),
                "config_id": cfg,
                "job_id": row.get("job_id", source.get("job_id", "")),
                "task": row.get("task", "pheno-all-cxr-notes-ecg"),
                "task_short": "pheno",
                "architecture": row["architecture"],
                "router_family": row.get("router_family", "flame_router"),
                "init_strategy": row["init_strategy"],
                "seed": row["seed"],
                "output_dir": output_dir,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def load_targeted_followup_runs(existing_ids: set[str]) -> list[dict]:
    rows = []
    for row in read_tsv(MANIFEST_ROOT / "week31_targeted_followup_manifest.tsv"):
        cfg = row["config_id"]
        if cfg in existing_ids:
            continue
        checkpoint_path = resolve_checkpoint_path(row.get("output_dir", ""))
        if not checkpoint_path:
            continue
        rows.append(
            {
                **row,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def build_rows() -> list[dict]:
    base_runs = load_base_runs()
    existing_ids = {row["config_id"] for row in base_runs}
    flame_runs = load_flame_runs(existing_ids)
    existing_ids |= {row["config_id"] for row in flame_runs}
    followup_runs = load_targeted_followup_runs(existing_ids)
    all_runs = base_runs + flame_runs + followup_runs

    rows = []
    for run in all_runs:
        for mask_name, forced_missing in MASKS.items():
            output_dir = (
                MECH_ROOT
                / "masked_eval"
                / run["architecture"]
                / run["router_family"]
                / run["init_strategy"]
                / run["task_short"]
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
            rows.append(rec)
    return rows


def main() -> None:
    MECH_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    manifest_path = MECH_ROOT / "week31_mechanism_masked_eval_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [], delimiter="\t")
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    report_lines = [
        "# Week31 Mechanism Masked-Eval Manifest",
        "",
        f"- manifest: `{manifest_path}`",
        f"- eval rows: {len(rows)}",
        f"- unique checkpoints: {len({row['config_id'] for row in rows})}",
        "- masks: `no_text`, `no_cxr`, `no_ecg`, `only_ts`",
        "- output root: `out/Week_31/mechanism_analysis/masked_eval/`",
    ]
    (MECH_ROOT / "week31_mechanism_masked_eval_manifest.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
