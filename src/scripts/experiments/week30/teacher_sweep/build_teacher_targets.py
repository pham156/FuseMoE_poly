#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from core.lingshu_pseudotoken import ORGAN_EXPERT_PROFILE_TEXT
from jobs.week29_semantic_init_full.target_utils import (
    EXPERT_PROFILE_NAMES,
    build_structured_summary,
    load_stays,
    mean_cxr_embedding,
    mean_ecg_embedding,
    mean_reg_ts,
    mean_text_embedding,
    normalize_scores,
    note_texts,
    sample_id,
)
from scripts.experiments.week30.teacher_sweep.teacher_registry import (
    get_teacher_spec,
    iter_teacher_specs,
    resolve_teacher_path,
    teacher_availability_record,
)


DEFAULT_HOME_ROOT = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_30/teacher_sweep")
DEFAULT_DATA_ROOT = Path("/home/pham156/MoE/FuseMoE_poly/data/MIMIC-IV")
WEEK29_TARGET_ROOT = Path("/scratch/gilbreth/pham156/MoE/FuseMoE_poly/out/Week_29/semantic_init_pheno_first/targets")
MODEL_ORDER = ["none", "random", "qwen25", "biomistral", "meditron", "me_llama", "medgemma", "lingshu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="all", choices=["all", *MODEL_ORDER])
    parser.add_argument("--task", default="pheno-all-cxr-notes-ecg")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_HOME_ROOT / "targets"))
    parser.add_argument("--status-file", default=str(DEFAULT_HOME_ROOT / "manifests" / "teacher_target_status.tsv"))
    parser.add_argument("--teacher_num_experts", type=int, default=4)
    parser.add_argument("--teacher_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--teacher_low_confidence_policy", choices=["skip", "uniform"], default="uniform")
    parser.add_argument("--teacher_freeze_backbone", action="store_true")
    parser.add_argument("--teacher_use_lora", action="store_true")
    parser.add_argument("--teacher_lora_rank", type=int, default=8)
    parser.add_argument("--teacher_max_samples", type=int, default=None)
    parser.add_argument("--teacher_batch_size", type=int, default=16)
    parser.add_argument("--teacher_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--teacher_cache_embeddings", action="store_true")
    parser.add_argument("--seeds", default="32,42,52")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def split_paths(data_root: Path, task: str) -> dict[str, Path]:
    return {
        split: data_root / f"{split}_{task}_stays.pkl"
        for split in ("train", "val", "test")
    }


def _seed_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _hidden_size_from_config(config) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to infer hidden size from teacher config")


_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def _modality_embedding(modality_name: str, hidden_size: int, device, dtype):
    key = ("modality", modality_name, hidden_size, str(device), str(dtype))
    if key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_from_text(f"modality::{modality_name}::{hidden_size}"))
    vec = torch.randn(hidden_size, generator=gen, dtype=torch.float32)
    vec = F.normalize(vec, dim=0)
    result = vec.to(device=device, dtype=dtype)
    _TENSOR_CACHE[key] = result
    return result


def _projection_matrix(tag: str, in_dim: int, out_dim: int, device, dtype):
    key = ("projection", tag, in_dim, out_dim, str(device), str(dtype))
    if key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_from_text(f"projection::{tag}::{in_dim}::{out_dim}"))
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32) / math.sqrt(max(in_dim, 1))
    result = weight.to(device=device, dtype=dtype)
    _TENSOR_CACHE[key] = result
    return result


def _as_tensor(arr: np.ndarray, device, dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device=device, dtype=dtype)


def _missing_flags(example: dict) -> np.ndarray:
    return np.asarray(
        [
            0.0,
            float(bool(example.get("text_missing", 0))),
            float(bool(example.get("cxr_missing", 0))),
            float(bool(example.get("ecg_missing", 0))),
        ],
        dtype=np.float32,
    )


def _patient_tokens(example: dict, hidden_size: int, device, dtype) -> torch.Tensor:
    features = {
        "ts": mean_reg_ts(example),
        "text": mean_text_embedding(example),
        "cxr": mean_cxr_embedding(example),
        "ecg": mean_ecg_embedding(example),
        "mask": _missing_flags(example),
    }
    tokens = []
    for name, values in features.items():
        proj = _projection_matrix(name, int(values.shape[0]), hidden_size, device=device, dtype=dtype)
        token = _as_tensor(values, device=device, dtype=dtype) @ proj
        token = token + _modality_embedding(name, hidden_size, device=device, dtype=dtype)
        tokens.append(token)
    return torch.stack(tokens, dim=0)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, dim=-1)


def encode_text_strings(backbone, tokenizer, texts, device, max_length=512, batch_size=4):
    pooled = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            raw = backbone(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = raw.hidden_states[-1] if getattr(raw, "hidden_states", None) is not None else raw.last_hidden_state
        pooled.append(mean_pool(hidden.float(), encoded["attention_mask"]).cpu())
    return torch.cat(pooled, dim=0)


def encode_pseudotokens(backbone, config, examples, device, batch_size=16):
    hidden_size = _hidden_size_from_config(config)
    backbone_dtype = next(backbone.parameters()).dtype
    outputs = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs_embeds = torch.stack(
            [_patient_tokens(example, hidden_size, device=device, dtype=backbone_dtype) for example in batch],
            dim=0,
        )
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            raw = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = raw.hidden_states[-1] if getattr(raw, "hidden_states", None) is not None else raw.last_hidden_state
        outputs.append(mean_pool(hidden.float(), attention_mask).cpu())
    return torch.cat(outputs, dim=0)


def patient_text(example: dict, max_notes: int = 5) -> str:
    texts = note_texts(example)
    note_blob = " ".join(texts[-max_notes:]) if texts else ""
    structured = build_structured_summary(example)
    return f"Clinical notes: {note_blob}\nStructured summary: {structured}" if note_blob else f"Structured summary: {structured}"


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "target_e0",
                "target_e1",
                "target_e2",
                "target_e3",
                "confidence",
                "source",
                "teacher_model",
                "teacher_family",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_rows(rows: list[dict], num_experts: int) -> tuple[bool, str]:
    for row in rows:
        values = [float(row[f"target_e{i}"]) for i in range(num_experts)]
        total = sum(values)
        if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
            return False, f"invalid target sum for sample_id={row['sample_id']}: {total}"
    return True, ""


def _teacher_rows_from_examples(
    spec_key: str,
    examples: list[dict],
    backbone,
    config,
    tokenizer,
    device,
    batch_size: int,
) -> list[dict]:
    spec = get_teacher_spec(spec_key)
    profile_texts = [ORGAN_EXPERT_PROFILE_TEXT[name] for name in EXPERT_PROFILE_NAMES]
    profile_embeddings = encode_text_strings(backbone, tokenizer, profile_texts, device=device)
    patient_embeddings = encode_pseudotokens(
        backbone,
        config,
        examples,
        device=device,
        batch_size=batch_size,
    )
    probs = torch.softmax(patient_embeddings @ profile_embeddings.t(), dim=-1).cpu().numpy()

    rows = []
    for example, p in zip(examples, probs):
        p = normalize_scores(p.astype(np.float32))
        rows.append(
            {
                "sample_id": sample_id(example),
                "target_e0": float(p[0]),
                "target_e1": float(p[1]),
                "target_e2": float(p[2]),
                "target_e3": float(p[3]),
                "confidence": float(np.max(p)),
                "source": spec.source,
                "teacher_model": spec.key,
                "teacher_family": spec.teacher_family,
            }
        )
    return rows


def _build_random_rows(examples: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    rows = []
    for rank, idx in enumerate(order):
        target = [0.0, 0.0, 0.0, 0.0]
        target[rank % 4] = 1.0
        rows.append(
            {
                "sample_id": sample_id(examples[idx]),
                "target_e0": target[0],
                "target_e1": target[1],
                "target_e2": target[2],
                "target_e3": target[3],
                "confidence": 1.0,
                "source": "random",
                "teacher_model": "random",
                "teacher_family": "random_control",
            }
        )
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def _build_random_rows_from_ids(sample_ids: list[str], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)
    rows = []
    for rank, sid in enumerate(shuffled):
        target = [0.0, 0.0, 0.0, 0.0]
        target[rank % 4] = 1.0
        rows.append(
            {
                "sample_id": sid,
                "target_e0": target[0],
                "target_e1": target[1],
                "target_e2": target[2],
                "target_e3": target[3],
                "confidence": 1.0,
                "source": "random",
                "teacher_model": "random",
                "teacher_family": "random_control",
            }
        )
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def _subset_examples(examples: list[dict], max_samples: int | None) -> list[dict]:
    return examples if not max_samples else examples[: max_samples]


def _week29_sample_id_file(split: str) -> Path:
    return WEEK29_TARGET_ROOT / f"base_{split}_sample_ids.txt"


def _load_sample_ids_for_split(split: str, max_samples: int | None) -> list[str]:
    path = _week29_sample_id_file(split)
    if path.exists():
        sample_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return sample_ids if not max_samples else sample_ids[:max_samples]
    return []


def _record_status(status_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        merged = {}
        if path.exists():
            with path.open() as handle:
                merged = {row["teacher_model"]: row for row in csv.DictReader(handle, delimiter="\t")}
        for row in status_rows:
            merged[row["teacher_model"]] = row
        ordered_rows = [merged[key] for key in MODEL_ORDER if key in merged]
        tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        with tmp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "teacher_model",
                    "source",
                    "teacher_family",
                    "display_name",
                    "available",
                    "resolved_path",
                    "reason",
                    "status",
                    "task",
                    "output_dir",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(ordered_rows)
        os.replace(tmp_path, path)


def _teacher_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def _load_teacher_backbone(spec_key: str, trust_remote_code: bool, device: torch.device, dtype_name: str):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    spec = get_teacher_spec(spec_key)
    model_path, reason = resolve_teacher_path(spec)
    if model_path is None:
        raise FileNotFoundError(reason)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    dtype = _teacher_dtype(dtype_name, device)
    load_kwargs = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    last_error = None
    backbone = None
    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            backbone = loader.from_pretrained(str(model_path), **load_kwargs)
            break
        except Exception as exc:
            last_error = exc
    if backbone is None:
        raise RuntimeError(f"Unable to load teacher backbone from {model_path}: {last_error}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return backbone, config, tokenizer, model_path


def build_one_teacher(args: argparse.Namespace, spec_key: str) -> dict:
    record = teacher_availability_record(spec_key)
    record["task"] = args.task
    record["output_dir"] = str(Path(args.output_root) / spec_key)
    if spec_key == "none":
        record["status"] = "control_no_targets"
        return record

    output_dir = Path(record["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if spec_key == "random":
        for seed_text in [x.strip() for x in args.seeds.split(",") if x.strip()]:
            seed = int(seed_text)
            seed_dir = output_dir / f"seed_{seed}"
            for split in ("train", "val", "test"):
                sample_ids = _load_sample_ids_for_split(split, args.teacher_max_samples)
                if sample_ids:
                    rows = _build_random_rows_from_ids(sample_ids, seed=seed)
                else:
                    examples = _subset_examples(load_stays(split_paths(Path(args.data_root), args.task)[split]), args.teacher_max_samples)
                    rows = _build_random_rows(examples, seed=seed)
                _write_rows(seed_dir / f"{split}_random_targets.csv", rows)
        record["status"] = "ready"
        return record

    if not record["available"]:
        record["status"] = "skipped_unavailable"
        return record

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    try:
        backbone, config, tokenizer, model_path = _load_teacher_backbone(
            spec_key,
            trust_remote_code=args.trust_remote_code,
            device=device,
            dtype_name=args.teacher_dtype,
        )
        if device.type == "cpu":
            backbone.to(device)
        backbone.eval()
        record["resolved_path"] = str(model_path)
        for split, path in split_paths(Path(args.data_root), args.task).items():
            examples = _subset_examples(load_stays(path), args.teacher_max_samples)
            rows = _teacher_rows_from_examples(
                spec_key,
                examples,
                backbone,
                config,
                tokenizer,
                device,
                batch_size=args.teacher_batch_size,
            )
            valid, message = _validate_rows(rows, args.teacher_num_experts)
            if not valid:
                raise ValueError(message)
            _write_rows(output_dir / f"{split}_{spec_key}_targets.csv", rows)
            del examples, rows
        record["status"] = "ready"
    except Exception as exc:
        record["status"] = "skipped_error"
        record["reason"] = f"{record['reason']}; runtime_error={type(exc).__name__}: {exc}"
    return record


def main() -> None:
    args = parse_args()
    status_rows = []
    selected = MODEL_ORDER if args.teacher_model == "all" else [args.teacher_model]
    for key in selected:
        status_rows.append(build_one_teacher(args, key))
    _record_status(status_rows, Path(args.status_file))
    print(json.dumps(status_rows, indent=2))


if __name__ == "__main__":
    main()
