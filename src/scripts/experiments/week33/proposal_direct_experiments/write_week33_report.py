#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from layout import AGGREGATE_ROOT, REPORT_ROOT, ensure_week33_dirs


IDEA_ROWS = [
    ("semantic organ routing", "X-MoE", "semantic_router"),
    ("organ-cohort consistency", "X-MoE / proposal note", "semantic_router"),
    ("missing-modality recovery", "SMIL / MissModal", "missing_recovery"),
    ("low-rank/LoRA-like experts", "MoLE / MixLoRA", "low_rank_experts"),
    ("cross-modal initialization", "proposal note", "cross_modal_init"),
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    ensure_week33_dirs()
    results = _read_csv(AGGREGATE_ROOT / "week33_results.csv")
    grouped = defaultdict(list)
    for row in results:
        grouped[row.get("experiment_group", "")].append(row)

    lines = [
        "# Week33 Proposal-Direct Report",
        "",
        "## 1. Which proposal-note idea was tested?",
        "",
        "This campaign tests semantic organ routing, organ-cohort consistency, missing-modality recovery, low-rank expert parameterization, and cross-modal initialization within the current per-modality FuseMoE pipeline.",
        "",
        "## 2. Which prior paper motivated it?",
        "",
        "Semantic organ routing is tied to X-MoE-style semantic routing, missing-modality recovery is tied to SMIL and MissModal-style recovery objectives, and low-rank experts are tied to MoLE / MixLoRA-style lightweight expert adapters.",
        "",
        "## 3. What was implemented?",
        "",
        "The Week33 manifests instantiate proposal-directed train and masked-eval runs under `out/Week_33/proposal_direct_experiments/`. Semantic routing reuses the existing semantic-profile router path with weak, medium, strong, semantic-only, shuffled-profile, and random-profile controls. Missing-modality recovery reuses the current proxy-token, cross-modal proxy, and recovery-loss paths with explicit training-time modality dropout. Low-rank experts reuse the existing `lora` and `residual_lora` expert types. Cross-modal initialization uses a lightweight target-generation script that trains tiny cross-modal regressors and emits expert-init target CSVs for the existing warm-start path.",
        "",
        "## 4. What was not implemented?",
        "",
        "The campaign does not import new foundation models and does not implement direct weight transplantation from cross-modal predictors into expert FFN weights. The cross-modal initialization arm is a lightweight warm-start approximation built on generated expert-init targets.",
        "",
        "## 5. What worked?",
        "",
        f"Current parsed result rows: {len(results)}.",
        "",
        "## 6. What failed?",
        "",
        "This report script does not infer failure causes. It only summarizes the currently parsed Week33 outputs.",
        "",
        "## 7. What can be claimed safely?",
        "",
        "Claims should remain tied to completed Week33 rows. Until the train and masked-eval manifests are executed, the safe conclusion is only that the proposal-note ideas have been operationalized into directly runnable FuseMoE experiments.",
        "",
        "## Final summary table",
        "",
        "| Idea | Paper inspiration | Implementation | Result | Safe conclusion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idea_name, paper_name, group_name in IDEA_ROWS:
        n_rows = len(grouped.get(group_name, []))
        result_text = f"{n_rows} parsed rows" if n_rows else "No parsed results yet"
        conclusion = "inconclusive" if n_rows == 0 else "supports further analysis"
        lines.append(f"| {idea_name} | {paper_name} | {group_name} manifest + analysis scripts | {result_text} | {conclusion} |")
    (REPORT_ROOT / "week33_proposal_direct_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
