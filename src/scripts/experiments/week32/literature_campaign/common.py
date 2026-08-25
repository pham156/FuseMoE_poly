from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
WEEK31_ROOT = REPO_ROOT / "out" / "Week_31"
WEEK32_ROOT = REPO_ROOT / "out" / "Week_32"
MECH_ROOT = WEEK32_ROOT / "mechanism_analysis"
LOG_ROOT = WEEK32_ROOT / "logs"

WEEK31_MANIFEST = WEEK31_ROOT / "manifests" / "week31_train_manifest.tsv"
WEEK31_SUMMARY = WEEK31_ROOT / "aggregates" / "week31_completed278_pheno_summary.csv"
WEEK31_ROUTING = WEEK31_ROOT / "aggregates" / "week31_completed278_routing_summary.csv"

INTERACTION_COLUMNS = [
    "cosine_text_ts",
    "cosine_text_cxr",
    "cosine_text_ecg",
    "cosine_ts_cxr",
    "cosine_ts_ecg",
    "cosine_cxr_ecg",
]


def ensure_week32_dirs() -> None:
    MECH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


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


def load_completed_runs() -> list[dict]:
    summary_rows = read_csv(WEEK31_SUMMARY)
    routing_rows = {row["config_id"]: row for row in read_csv(WEEK31_ROUTING)}
    manifest_rows = {row["config_id"]: row for row in read_tsv(WEEK31_MANIFEST)}
    runs = []
    for row in summary_rows:
        cfg = row["config_id"]
        manifest = manifest_rows.get(cfg, {})
        routing = routing_rows.get(cfg, {})
        output_dir = row.get("output_dir") or manifest.get("output_dir", "")
        checkpoint_path = str(Path(output_dir) / "best_model.pth.tar") if output_dir else ""
        runtime_gating_function = manifest.get("gating_function", "") or row.get("gating_function", "")
        runtime_poly_power = manifest.get("poly_power", "")
        report_gating_function = row.get("gating_function", "") or runtime_gating_function
        runs.append(
            {
                **manifest,
                **row,
                **routing,
                "task_short": task_short(row.get("task", "")),
                "output_dir": output_dir,
                "checkpoint_path": checkpoint_path,
                "gating_function": report_gating_function,
                "runtime_gating_function": runtime_gating_function,
                "runtime_poly_power": runtime_poly_power,
            }
        )
    return runs


def parse_topk_string(text: str) -> list[tuple[int | None, list[tuple[int, float]]]]:
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    decisions: list[tuple[int | None, list[tuple[int, float]]]] = []
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


def try_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def topk_entropy(pairs: list[tuple[int, float]]) -> float:
    total = sum(max(weight, 0.0) for _, weight in pairs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for _, weight in pairs:
        p = max(weight, 0.0) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def top1_margin(pairs: list[tuple[int, float]]) -> float:
    if not pairs:
        return 0.0
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    if len(ordered) == 1:
        return ordered[0][1]
    return ordered[0][1] - ordered[1][1]


def parse_mask(mask_text: str) -> dict[str, float]:
    observed = {key: 0.0 for key in ("text", "ts", "cxr", "ecg")}
    for token in (mask_text or "").split("+"):
        token = token.strip()
        if token in observed:
            observed[token] = 1.0
    return observed


def parse_binary_label(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_pheno_vector(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(float(x)) for x in text.split("|")]


def dominant_expert_from_sparse(text: str) -> int | None:
    decisions = parse_topk_string(text)
    if not decisions:
        return None
    vote_count: dict[int, int] = {}
    vote_weight: dict[int, float] = {}
    for _, pairs in decisions:
        ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
        if not ranked:
            continue
        expert_id, weight = ranked[0]
        vote_count[expert_id] = vote_count.get(expert_id, 0) + 1
        vote_weight[expert_id] = vote_weight.get(expert_id, 0.0) + weight
    if not vote_count:
        return None
    return sorted(vote_count, key=lambda expert_id: (vote_count[expert_id], vote_weight.get(expert_id, 0.0), -expert_id), reverse=True)[0]


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
        dist = [value / total for value in dist]
    return dist


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    set_left = set(left)
    set_right = set(right)
    union = set_left | set_right
    if not union:
        return 0.0
    return len(set_left & set_right) / len(union)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    avg = sum(values) / len(values)
    if len(values) == 1:
        return avg, 0.0
    var = sum((value - avg) ** 2 for value in values) / len(values)
    return avg, math.sqrt(var)
