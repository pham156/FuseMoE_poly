#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.train import _split_mimic_batch
from preprocessing.data_mimiciv import TSNote_Irg, TextTSIrgcollate_fn
from scripts.main_mimiciv import _build_text_backbone
from utils.util import parse_args, set_seed
from layout import TARGET_ROOT, ensure_week33_dirs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORGANS = ["cardiovascular", "respiratory", "renal_metabolic", "neuro_general"]


class TinyRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = min(512, max(128, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pool_sequence(x: torch.Tensor, batch_size: int) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 2:
        return x.float()
    if x.ndim == 3:
        if x.shape[0] == batch_size:
            return x.float().mean(dim=1)
        if x.shape[1] == batch_size:
            return x.float().mean(dim=0)
    if x.ndim >= 4:
        x = x.float().reshape(batch_size, -1, x.shape[-1])
        return x.mean(dim=1)
    return None


def _pool_ts(reg_ts: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if reg_ts is None:
        return None
    if reg_ts.ndim == 3:
        if reg_ts.shape[0] == batch_size:
            return reg_ts.float().mean(dim=1)
        if reg_ts.shape[1] == batch_size:
            return reg_ts.float().mean(dim=0)
    if reg_ts.ndim == 2:
        return reg_ts.float()
    return None


def _to_bool(x: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    if x is None:
        return torch.zeros(batch_size, dtype=torch.bool)
    return x.detach().cpu().bool().view(-1)


def _collect_split(dataset: TSNote_Irg, split: str) -> list[dict]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=TextTSIrgcollate_fn, num_workers=0)
    rows: list[dict] = []
    for batch in loader:
        batch_tensors, metadata = _split_mimic_batch(batch)
        (
            _ts_input_sequences,
            _ts_mask_sequences,
            _ts_tt,
            reg_ts,
            _input_ids_sequences,
            _attn_mask_sequences,
            text_emb,
            _note_time,
            _note_time_mask,
            cxr_feats,
            _cxr_time,
            _cxr_time_mask,
            ecg_feats,
            _ecg_time,
            _ecg_time_mask,
            _label,
            cxr_missing,
            text_missing,
            ecg_missing,
        ) = batch_tensors
        sample_ids = list(metadata.get("sample_ids", []))
        batch_size = len(sample_ids)
        pooled = {
            "ts": _pool_ts(reg_ts, batch_size),
            "text": _pool_sequence(text_emb, batch_size),
            "cxr": _pool_sequence(cxr_feats, batch_size),
            "ecg": _pool_sequence(ecg_feats, batch_size),
        }
        missing = {
            "ts": torch.zeros(batch_size, dtype=torch.bool),
            "text": _to_bool(text_missing, batch_size),
            "cxr": _to_bool(cxr_missing, batch_size),
            "ecg": _to_bool(ecg_missing, batch_size),
        }
        for i, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "split": split,
                    "sample_id": str(sample_id),
                    "ts": None if pooled["ts"] is None else pooled["ts"][i].clone(),
                    "text": None if pooled["text"] is None else pooled["text"][i].clone(),
                    "cxr": None if pooled["cxr"] is None else pooled["cxr"][i].clone(),
                    "ecg": None if pooled["ecg"] is None else pooled["ecg"][i].clone(),
                    "text_missing": bool(missing["text"][i].item()),
                    "cxr_missing": bool(missing["cxr"][i].item()),
                    "ecg_missing": bool(missing["ecg"][i].item()),
                }
            )
    return rows


def _source_vector(row: dict, names: list[str]) -> torch.Tensor | None:
    tensors = []
    for name in names:
        value = row.get(name)
        if value is None:
            return None
        if name != "ts" and bool(row.get(f"{name}_missing", False)):
            return None
        tensors.append(value.float())
    return torch.cat(tensors, dim=0)


def _target_vector(row: dict, name: str) -> torch.Tensor | None:
    value = row.get(name)
    if value is None:
        return None
    if name != "ts" and bool(row.get(f"{name}_missing", False)):
        return None
    return value.float()


def _task_specs() -> dict[str, tuple[list[str], str]]:
    return {
        "cardiovascular": (["text", "ts"], "ecg"),
        "respiratory": (["text", "ts"], "cxr"),
        "renal_metabolic": (["text", "cxr", "ecg"], "ts"),
        "neuro_general": (["ts", "cxr", "ecg"], "text"),
    }


def _build_examples(rows: list[dict], source_names: list[str], target_name: str) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    examples = []
    for row in rows:
        source = _source_vector(row, source_names)
        target = _target_vector(row, target_name)
        if source is None or target is None:
            continue
        examples.append((row["sample_id"], source, target))
    return examples


def _train_predictor(examples: list[tuple[str, torch.Tensor, torch.Tensor]], epochs: int = 5) -> tuple[TinyRegressor, dict[str, float]]:
    inputs = torch.stack([src for _, src, _ in examples], dim=0).to(DEVICE)
    targets = torch.stack([tgt for _, _, tgt in examples], dim=0).to(DEVICE)
    model = TinyRegressor(inputs.size(1), targets.size(1)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(inputs)
        mse = F.mse_loss(pred, targets)
        cosine = 1.0 - F.cosine_similarity(pred, targets, dim=1).mean()
        loss = 0.5 * (mse + cosine)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = model(inputs)
        mse = F.mse_loss(pred, targets).item()
        cosine = F.cosine_similarity(pred, targets, dim=1).mean().item()
    return model, {"train_mse": mse, "train_cosine": cosine, "n_examples": len(examples)}


def _per_sample_error(model: TinyRegressor, examples: list[tuple[str, torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    errors = {}
    model.eval()
    with torch.no_grad():
        for sample_id, source, target in examples:
            pred = model(source.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            mse = F.mse_loss(pred, target, reduction="mean").item()
            cosine = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0), dim=1).item()
            errors[sample_id] = 0.5 * (mse + (1.0 - cosine))
    return errors


def main() -> None:
    ensure_week33_dirs()
    args = parse_args()
    set_seed(args.seed)
    BioBert, tokenizer = _build_text_backbone(args, DEVICE)

    split_rows = {
        split: _collect_split(TSNote_Irg(args, split, tokenizer), split)
        for split in ("train", "val", "test")
    }
    specs = _task_specs()
    split_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    metric_rows = []

    for organ_name, (source_names, target_name) in specs.items():
        train_examples = _build_examples(split_rows["train"], source_names, target_name)
        if not train_examples:
            continue
        model, metrics = _train_predictor(train_examples)
        torch.save(model.state_dict(), TARGET_ROOT / "cross_modal_init" / f"{organ_name}_predictor.pt")
        metric_rows.append({
            "organ": organ_name,
            "split": "train",
            **metrics,
        })
        for split_name, rows in split_rows.items():
            examples = _build_examples(rows, source_names, target_name)
            if not examples:
                continue
            per_sample = _per_sample_error(model, examples)
            split_scores[split_name][organ_name] = per_sample
            if split_name != "train":
                metric_rows.append({
                    "organ": organ_name,
                    "split": split_name,
                    "train_mse": sum(per_sample.values()) / max(len(per_sample), 1),
                    "train_cosine": "",
                    "n_examples": len(per_sample),
                })

    out_dir = TARGET_ROOT / "cross_modal_init"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        output_rows = []
        for row in rows:
            sample_id = row["sample_id"]
            losses = []
            for organ_name in ORGANS:
                value = split_scores.get(split_name, {}).get(organ_name, {}).get(sample_id)
                losses.append(None if value is None else float(value))
            valid = [loss for loss in losses if loss is not None]
            if not valid:
                weights = [0.25, 0.25, 0.25, 0.25]
                confidence = 0.25
            else:
                logits = torch.tensor(
                    [(-loss if loss is not None else -1e6) for loss in losses],
                    dtype=torch.float32,
                )
                probs = torch.softmax(logits, dim=0).tolist()
                weights = probs
                confidence = float(max(probs))
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "target_e0": weights[0],
                    "target_e1": weights[1],
                    "target_e2": weights[2],
                    "target_e3": weights[3],
                    "confidence": confidence,
                    "source": "cross_modal_init",
                }
            )
        with (out_dir / f"{split_name}_cross_modal_init_targets.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "target_e0", "target_e1", "target_e2", "target_e3", "confidence", "source"])
            writer.writeheader()
            writer.writerows(output_rows)

    with (out_dir / "cross_modal_init_pretrain_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["organ", "split", "train_mse", "train_cosine", "n_examples"])
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Wrote targets under {out_dir}")


if __name__ == "__main__":
    main()

