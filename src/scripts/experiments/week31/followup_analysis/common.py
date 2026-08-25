from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
WEEK31_ROOT = REPO_ROOT / "out" / "Week_31"
FOLLOWUP_ROOT = WEEK31_ROOT / "followup_analysis"
AGG_ROOT = WEEK31_ROOT / "aggregates"
MANIFEST_ROOT = WEEK31_ROOT / "manifests"


def ensure_followup_root() -> Path:
    FOLLOWUP_ROOT.mkdir(parents=True, exist_ok=True)
    return FOLLOWUP_ROOT


def read_tsv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
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


def metric_value(row: dict) -> float:
    short = row.get("task_short") or task_short(row.get("task", ""))
    if short == "pheno":
        return float(row.get("pheno_macro_f1") or row.get("pheno_macro_f1_mean") or 0.0)
    return float(row.get("binary_f1") or row.get("binary_f1_mean") or 0.0)


def load_manifest_lookup() -> Dict[str, dict]:
    return {row["config_id"]: row for row in read_tsv(MANIFEST_ROOT / "week31_train_manifest.tsv")}


def resolve_checkpoint_path(output_dir: str) -> Optional[str]:
    if not output_dir:
        return None
    path = Path(output_dir) / "best_model.pth.tar"
    return str(path) if path.exists() else None


def load_completed_runs() -> List[dict]:
    manifest_lookup = load_manifest_lookup()
    completed: List[dict] = []
    for row in read_csv(AGG_ROOT / "week31_results_run_level.csv"):
        manifest = manifest_lookup.get(row["config_id"], {})
        out_dir = manifest.get("output_dir", "")
        completed.append(
            {
                **manifest,
                **row,
                "task_short": row.get("task_short") or task_short(row.get("task", "")),
                "output_dir": out_dir,
                "checkpoint_path": resolve_checkpoint_path(out_dir) or "",
                "is_flame": "False",
            }
        )

    existing_ids = {row["config_id"] for row in completed}
    flame_jobmap = {r["config_id"]: r for r in read_tsv(MANIFEST_ROOT / "week31_flame_confirmation_jobs.tsv")}
    for row in read_csv(AGG_ROOT / "week31_flame_confirmation_results.csv"):
        cfg = row["config_id"]
        if cfg in existing_ids:
            continue
        source = flame_jobmap.get(cfg, manifest_lookup.get(cfg, {}))
        manifest = manifest_lookup.get(cfg, {})
        out_dir = source.get("output_dir", manifest.get("output_dir", ""))
        completed.append(
            {
                **manifest,
                "config_id": cfg,
                "job_id": row.get("job_id", source.get("job_id", "")),
                "task": row.get("task", "pheno-all-cxr-notes-ecg"),
                "task_short": "pheno",
                "architecture": row["architecture"],
                "router_family": row.get("router_family", "flame_router"),
                "init_strategy": row["init_strategy"],
                "seed": row["seed"],
                "pheno_macro_f1": row["macro_f1"],
                "pheno_auc_macro": row["auc_macro"],
                "pheno_auc_micro": row["auc_micro"],
                "pheno_auc_weighted": row["auc_weighted"],
                "binary_auc": "",
                "binary_auprc": "",
                "binary_f1": row["macro_f1"],
                "active_experts": row["active_experts"],
                "gate_entropy": row["gate_entropy"],
                "gate_top1_weight": row["gate_top1_weight"],
                "output_dir": out_dir,
                "checkpoint_path": resolve_checkpoint_path(out_dir) or "",
                "is_flame": "True",
            }
        )
    return completed


def parse_topk_string(text: str) -> List[Tuple[Optional[int], List[Tuple[int, float]]]]:
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    decisions: List[Tuple[Optional[int], List[Tuple[int, float]]]] = []
    if "m0=" in text:
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            prefix, payload = chunk.split("=", 1)
            modality_idx = int(prefix[1:]) if prefix.startswith("m") and prefix[1:].isdigit() else None
            pairs = []
            for item in payload.split("|"):
                if ":" not in item:
                    continue
                expert, weight = item.split(":", 1)
                try:
                    pairs.append((int(expert), float(weight)))
                except ValueError:
                    continue
            if pairs:
                decisions.append((modality_idx, pairs))
        return decisions
    pairs = []
    for item in text.split("|"):
        if ":" not in item:
            continue
        expert, weight = item.split(":", 1)
        try:
            pairs.append((int(expert), float(weight)))
        except ValueError:
            continue
    if pairs:
        decisions.append((None, pairs))
    return decisions


def topk_entropy(pairs: List[Tuple[int, float]]) -> float:
    total = sum(max(weight, 0.0) for _, weight in pairs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for _, weight in pairs:
        p = max(weight, 0.0) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def top1_margin(pairs: List[Tuple[int, float]]) -> float:
    if not pairs:
        return 0.0
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    return ordered[0][1] if len(ordered) == 1 else ordered[0][1] - ordered[1][1]


def parse_mask(mask_text: str) -> Dict[str, float]:
    observed = {key: 0.0 for key in ("text", "ts", "cxr", "ecg")}
    for token in (mask_text or "").split("+"):
        token = token.strip()
        if token in observed:
            observed[token] = 1.0
    return observed


def parse_pheno_vector(text: str) -> List[int]:
    if not text:
        return []
    return [int(float(x)) for x in text.split("|")]


def parse_binary_label(text: str) -> Optional[int]:
    if text is None or text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def try_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def rankdata(values: List[float]) -> List[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(ordered):
        j = idx
        while j < len(ordered) and ordered[j][0] == ordered[idx][0]:
            j += 1
        avg_rank = 0.5 * (idx + j - 1) + 1.0
        for _, original_idx in ordered[idx:j]:
            ranks[original_idx] = avg_rank
        idx = j
    return ranks


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return math.nan
    return num / (denx * deny)


def spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return math.nan
    return pearson(rankdata(xs), rankdata(ys))


def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan, math.nan, math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return math.nan, math.nan, math.nan
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    preds = [intercept + slope * x for x in xs]
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - p) ** 2 for y, p in zip(ys, preds))
    r2 = math.nan if sst == 0 else 1.0 - (sse / sst)
    return slope, intercept, r2


def jaccard(xs: List[int], ys: List[int]) -> float:
    sx = set(xs)
    sy = set(ys)
    denom = len(sx | sy)
    if denom == 0:
        return math.nan
    return len(sx & sy) / denom
