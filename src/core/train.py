import os
import pickle
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import csv
import hashlib
# import wandb
import copy
from pathlib import Path

from utils.checkpoint import *
from utils.util import *
from tqdm import tqdm
from core.r2t2 import R2T2ReferenceCollector
# from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize


ORGAN_KEYWORDS = {
    "cardiovascular": [
        "blood pressure", "hypotension", "hypertension", "heart rate", "vasopressor",
        "norepinephrine", "epinephrine", "shock", "cardiac", "heart failure",
        "arrest", "hemodynamic", "pressor",
    ],
    "respiratory": [
        "oxygen", "ventilation", "ventilator", "spo2", "saturation", "lung",
        "pneumonia", "respiratory failure", "intubated", "intubation",
        "pulmonary", "edema", "cxr",
    ],
    "renal_metabolic": [
        "creatinine", "bun", "urea nitrogen", "urine output", "renal",
        "kidney", "dialysis", "electrolyte", "sodium", "potassium",
        "bicarbonate", "acidosis", "alkalosis", "anion gap", "glucose",
        "lactate", "metabolic",
    ],
    "neurological": [
        "gcs", "glasgow", "coma", "sedation", "sedated", "delirium",
        "mental status", "altered mental", "consciousness", "neurologic",
        "neurological", "seizure", "stroke", "pupil",
    ],
}

EXPERT_PROFILE_NAMES = ["cardiovascular", "respiratory", "renal_metabolic", "neurological"]


def _split_mimic_batch(batch):
    metadata = batch[19] if len(batch) > 19 else {}
    return batch[:19], metadata


def _weak_organ_labels(note_texts):
    joined = " ".join(str(t) for t in note_texts).lower()
    labels = []
    for organ, keywords in ORGAN_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            labels.append(organ)
    return labels


def _weak_organ_target_tensor(metadata, device):
    note_texts = metadata.get("note_texts", [])
    targets = []
    for sample_notes in note_texts:
        if isinstance(sample_notes, str):
            sample_notes = [sample_notes]
        labels = set(_weak_organ_labels(sample_notes))
        targets.append([1.0 if name in labels else 0.0 for name in EXPERT_PROFILE_NAMES])
    if not targets:
        return None
    return torch.tensor(targets, dtype=torch.float32, device=device)


def prepare_expert_init_targets(args):
    args.expert_init_targets_by_split = {}
    if getattr(args, "expert_init_strategy", "none") != "fixed_cohort":
        return
    target_path = getattr(args, "expert_init_target_path", None)
    if not target_path:
        raise ValueError("--expert_init_target_path is required when --expert_init_strategy fixed_cohort")
    for split in ("train", "val", "test"):
        split_path = str(target_path).format(split=split)
        if not os.path.exists(split_path):
            if split == "train":
                raise FileNotFoundError(f"Missing expert-init target file for train split: {split_path}")
            continue
        table = {}
        with open(split_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    continue
                table[sample_id] = {
                    "target": torch.tensor(
                        [
                            float(row.get("target_e0", 0.0) or 0.0),
                            float(row.get("target_e1", 0.0) or 0.0),
                            float(row.get("target_e2", 0.0) or 0.0),
                            float(row.get("target_e3", 0.0) or 0.0),
                        ],
                        dtype=torch.float32,
                    ),
                    "confidence": float(row.get("confidence", 1.0) or 1.0),
                    "source": str(row.get("source", "") or ""),
                }
        args.expert_init_targets_by_split[split] = table


def _resolve_teacher_split_path(base_path, split):
    if not base_path:
        return None
    if "{split}" in str(base_path):
        return str(base_path).format(split=split)
    if os.path.isdir(base_path):
        return os.path.join(base_path, f"{split}_unimodal_teacher_targets.csv")
    return str(base_path)


def prepare_unimodal_teacher_targets(args):
    args.unimodal_teacher_targets_by_split = {}
    if not bool(getattr(args, "use_unimodal_kd", False)):
        return
    base_path = getattr(args, "unimodal_teacher_dir", "")
    if not base_path:
        raise ValueError("--unimodal_teacher_dir is required when --use_unimodal_kd is enabled")

    modalities = [item.strip().lower() for item in str(getattr(args, "unimodal_kd_modalities", "")).split(",") if item.strip()]
    output_dim = 25 if "pheno" in getattr(args, "task", "") else 2
    for split in ("train", "val", "test"):
        split_path = _resolve_teacher_split_path(base_path, split)
        if split_path is None or not os.path.exists(split_path):
            if split == "train":
                raise FileNotFoundError(f"Missing unimodal teacher target file for train split: {split_path}")
            continue
        table = {}
        with open(split_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            modality_fields = {
                modality: [f"{modality}_{idx}" for idx in range(output_dim)]
                for modality in modalities
            }
            for row in reader:
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    continue
                entry = {}
                for modality, fields in modality_fields.items():
                    if not all(field in fieldnames for field in fields):
                        continue
                    entry[modality] = torch.tensor(
                        [float(row.get(field, 0.0) or 0.0) for field in fields],
                        dtype=torch.float32,
                    )
                if entry:
                    table[sample_id] = entry
        args.unimodal_teacher_targets_by_split[split] = table


def _resolve_unimodal_kd_batch(metadata, args, split):
    target_tables = getattr(args, "unimodal_teacher_targets_by_split", {})
    table = target_tables.get(split)
    if not table:
        return None
    sample_ids = [str(sample_id) for sample_id in metadata.get("sample_ids", [])]
    if not sample_ids:
        return None
    modalities = [item.strip().lower() for item in str(getattr(args, "unimodal_kd_modalities", "")).split(",") if item.strip()]
    output_dim = 25 if "pheno" in getattr(args, "task", "") else 2
    result = {}
    found_any = False
    for modality in modalities:
        rows = []
        available = []
        for sample_id in sample_ids:
            row = table.get(sample_id, {})
            if modality in row:
                rows.append(row[modality].clone())
                available.append(1.0)
                found_any = True
            else:
                rows.append(torch.zeros(output_dim, dtype=torch.float32))
                available.append(0.0)
        result[modality] = {
            "targets": torch.stack(rows, dim=0),
            "available": torch.tensor(available, dtype=torch.float32),
        }
    return result if found_any else None


def _resolve_expert_init_batch(metadata, args, split):
    target_tables = getattr(args, "expert_init_targets_by_split", {})
    table = target_tables.get(split)
    if not table:
        return None
    sample_ids = [str(sample_id) for sample_id in metadata.get("sample_ids", [])]
    if not sample_ids:
        return None
    targets = []
    confidence = []
    source = []
    found_any = False
    for sample_id in sample_ids:
        row = table.get(sample_id)
        if row is None:
            targets.append(torch.zeros(4, dtype=torch.float32))
            confidence.append(0.0)
            source.append("")
            continue
        found_any = True
        targets.append(row["target"].clone())
        confidence.append(float(row["confidence"]))
        source.append(row["source"])
    if not found_any:
        return None
    metadata["expert_init_targets"] = [target.tolist() for target in targets]
    metadata["expert_init_confidence"] = confidence
    metadata["expert_init_source"] = source
    return {
        "targets": torch.stack(targets, dim=0),
        "confidence": torch.tensor(confidence, dtype=torch.float32),
        "source": source,
    }


def _set_expert_init_batch(model, args, metadata, split, active):
    batch_info = _resolve_expert_init_batch(metadata, args, split)
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "set_expert_init_batch"):
            module.set_expert_init_batch(batch_info, active=active)


def _configure_expert_init_trainability(model, args, epoch):
    strategy = getattr(args, "expert_init_strategy", "none")
    warm_epochs = int(getattr(args, "expert_init_epochs", 0) or 0)
    active = strategy == "fixed_cohort" and warm_epochs > 0
    if active:
        if getattr(args, "expert_init_release_schedule", "hard") == "hard":
            active = epoch < warm_epochs
        else:
            release_epochs = max(1, int(np.ceil(warm_epochs * 0.25)))
            active = epoch < (warm_epochs + release_epochs)

    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "set_expert_init_router_frozen"):
            module.set_expert_init_router_frozen(active and bool(getattr(args, "expert_init_freeze_router", False)))

    if getattr(args, "expert_init_freeze_backbone", False):
        for name, param in unwrapped.named_parameters():
            if ".moe." in name or name.startswith("moe."):
                continue
            param.requires_grad = not active


def _apply_train_modality_dropout(args, cxr_missing, text_missing, ecg_missing, reference_tensor):
    cxr_prob = float(getattr(args, "train_cxr_modality_dropout", 0.0) or 0.0)
    text_prob = float(getattr(args, "train_text_modality_dropout", 0.0) or 0.0)
    ecg_prob = float(getattr(args, "train_ecg_modality_dropout", 0.0) or 0.0)
    if cxr_prob <= 0.0 and text_prob <= 0.0 and ecg_prob <= 0.0:
        return cxr_missing, text_missing, ecg_missing

    batch_size = reference_tensor.shape[0]
    device = reference_tensor.device

    def apply_one(existing_missing, prob):
        if existing_missing is None:
            missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            missing = existing_missing.to(device=device).bool().clone()
        if prob <= 0.0:
            return missing
        observed = ~missing
        dropped = (torch.rand(batch_size, device=device) < prob) & observed
        return missing | dropped

    return (
        apply_one(cxr_missing, cxr_prob),
        apply_one(text_missing, text_prob),
        apply_one(ecg_missing, ecg_prob),
    )


def _apply_consistency_train_mask(args, cxr_missing, text_missing, ecg_missing, reference_tensor):
    if not bool(getattr(args, "use_full_partial_consistency", False)):
        return cxr_missing, text_missing, ecg_missing

    batch_size = reference_tensor.shape[0]
    device = reference_tensor.device
    mode = str(getattr(args, "consistency_mask_mode", "single_random"))
    level = float(getattr(args, "consistency_missingness_level", 0.25) or 0.0)

    def as_missing(existing):
        if existing is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return existing.to(device=device).bool().clone()

    cxr = as_missing(cxr_missing)
    txt = as_missing(text_missing)
    ecg = as_missing(ecg_missing)

    if mode == "progressive":
        def drop(existing):
            observed = ~existing
            extra = (torch.rand(batch_size, device=device) < level) & observed
            return existing | extra
        return drop(cxr), drop(txt), drop(ecg)

    observed_matrix = torch.stack([~txt, ~cxr, ~ecg], dim=1)
    choice = torch.multinomial(
        observed_matrix.float() + 1e-8,
        num_samples=1,
        replacement=True,
    ).squeeze(1)
    has_any = observed_matrix.any(dim=1)
    txt = txt | (has_any & (choice == 0) & observed_matrix[:, 0])
    cxr = cxr | (has_any & (choice == 1) & observed_matrix[:, 1])
    ecg = ecg | (has_any & (choice == 2) & observed_matrix[:, 2])
    return cxr, txt, ecg


def _find_router_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_router_diagnostics", None)
        if diagnostics is not None:
            found.append(diagnostics)
    return found[-1] if found else None


def _find_all_router_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_router_diagnostics", None)
        if diagnostics is not None:
            found.append(diagnostics)
    return found


def _iter_r2t2_router_modules(model):
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "set_r2t2_adaptor") and hasattr(module, "clear_r2t2_adaptor"):
            yield module


def _set_r2t2_adaptor(model, adaptor):
    for layer_idx, module in enumerate(_iter_r2t2_router_modules(model)):
        module.set_r2t2_adaptor(adaptor, layer_idx)


def _clear_r2t2_adaptor(model):
    for module in _iter_r2t2_router_modules(model):
        module.clear_r2t2_adaptor()


def build_r2t2_adaptor(args, device, val_dataloader, model, dataset="mimic"):
    _clear_r2t2_adaptor(model)
    collector = R2T2ReferenceCollector(
        correct_reference_only=getattr(args, "r2t2_correct_reference_only", False),
    )
    evaluate_irg(
        args=args,
        device=device,
        data_loader=val_dataloader,
        model=model,
        mode="val",
        dataset=dataset,
        r2t2_collector=collector,
    )
    adaptor = collector.build(
        num_neighbors=getattr(args, "r2t2_num_neighbors", 32),
        blend=getattr(args, "r2t2_blend", 0.5),
        kernel_sigma=getattr(args, "r2t2_kernel_sigma", 1.0),
    )
    if adaptor.num_reference_groups == 0:
        raise RuntimeError(
            "R2T2 reference bank is empty. Check that validation evaluation produced router diagnostics."
        )
    _set_r2t2_adaptor(model, adaptor)
    print(
        "R2T2 reference bank: "
        f"samples_seen={adaptor.num_samples_seen} "
        f"samples_kept={adaptor.num_samples_kept} "
        f"groups={adaptor.num_reference_groups} "
        f"vectors={adaptor.num_reference_vectors} "
        f"k={adaptor.num_neighbors} blend={adaptor.blend:.4g} sigma={adaptor.kernel_sigma:.4g}"
    )
    return adaptor


def _find_all_router_aux_components(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        components = getattr(module, "last_router_aux_components", None)
        if components is not None:
            found.append(components)
    return found


def _find_all_specialization_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_specialization_diagnostics", None)
        if diagnostics:
            found.append(diagnostics)
    return found


def _find_all_expert_output_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_expert_output_diagnostics", None)
        if diagnostics:
            found.append(diagnostics)
    return found


def _find_all_contribution_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_contribution_diagnostics", None)
        if diagnostics is not None:
            found.append(diagnostics)
    return found


def _find_interaction_feature_output(model):
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        output = getattr(module, "last_interaction_feature_output", None)
        if output is not None:
            return output
    return None


def _find_all_router_temperatures(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        log_tau = getattr(module, "log_tau", None)
        if log_tau is not None:
            found.append(float(log_tau.detach().exp().cpu().item()))
    return found


def _find_missing_recon_metrics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    metrics = {}
    loss_value = getattr(unwrapped, "last_missing_recon_loss", None)
    if loss_value is not None:
        metrics["missing_recon_loss"] = loss_value
    details = getattr(unwrapped, "last_missing_recon_details", {}) or {}
    metrics.update(details)
    return metrics


def _find_ts_patch_recon_metrics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    metrics = {}
    loss_value = getattr(unwrapped, "last_ts_patch_recon_loss", None)
    if loss_value is not None:
        metrics["ts_patch_recon_loss"] = loss_value
    details = getattr(unwrapped, "last_ts_patch_recon_details", {}) or {}
    metrics.update(details)
    return metrics


def _router_usage_str(values):
    return "[" + ",".join(f"{float(v):.3f}" for v in values) + "]"


def _accumulate_metric_dict(metric_sums, metric_counts, metric_dicts):
    for metric_dict in metric_dicts:
        for name, value in metric_dict.items():
            try:
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    metric_value = float(value.detach().cpu().item())
                else:
                    metric_value = float(value)
            except Exception:
                continue
            metric_sums[name] = metric_sums.get(name, 0.0) + metric_value
            metric_counts[name] = metric_counts.get(name, 0) + 1


def _format_metric_sums(metric_sums, metric_counts):
    return ", ".join(
        f"{name}:{metric_sums[name] / max(metric_counts.get(name, 0), 1):.6f}"
        for name in sorted(metric_sums)
    )


def _accumulate_expert_output_diagnostics(sums, counts, diagnostics_by_layer):
    for layer_idx, records in enumerate(diagnostics_by_layer):
        while len(sums) <= layer_idx:
            sums.append({})
            counts.append({})
        for record in records:
            mod_idx = record.get("modality_idx")
            key = "joint" if mod_idx is None else f"m{mod_idx}"
            sim = record.get("sim_matrix")
            offdiag = record.get("offdiag_cos")
            norms = record.get("expert_norms")
            if sim is None or offdiag is None:
                continue
            if key not in sums[layer_idx]:
                sums[layer_idx][key] = {
                    "sim": sim.clone().float(),
                    "offdiag": float(offdiag),
                    "norms": norms.clone().float() if norms is not None else None,
                }
                counts[layer_idx][key] = 1
            else:
                sums[layer_idx][key]["sim"] += sim.float()
                sums[layer_idx][key]["offdiag"] += float(offdiag)
                if norms is not None and sums[layer_idx][key]["norms"] is not None:
                    sums[layer_idx][key]["norms"] += norms.float()
                counts[layer_idx][key] += 1


def _format_expert_output_diagnostics(sums, counts):
    parts = []
    for layer_idx, layer_sums in enumerate(sums):
        for key in sorted(layer_sums):
            count = max(counts[layer_idx].get(key, 0), 1)
            offdiag = layer_sums[key]["offdiag"] / count
            sim = layer_sums[key]["sim"] / count
            norms = layer_sums[key]["norms"]
            sim_flat = ",".join(f"{float(v):.3f}" for v in sim.flatten())
            if norms is not None:
                norm_flat = ",".join(f"{float(v):.3f}" for v in (norms / count))
                parts.append(f"L{layer_idx}{key}:offdiag={offdiag:.3f} sim=[{sim_flat}] norm=[{norm_flat}]")
            else:
                parts.append(f"L{layer_idx}{key}:offdiag={offdiag:.3f} sim=[{sim_flat}]")
    return " ; ".join(parts)


def _parse_csv_choice_set(raw_value, valid_values):
    if not raw_value:
        return set()
    values = set()
    for piece in str(raw_value).split(","):
        item = piece.strip().lower()
        if not item:
            continue
        if item not in valid_values:
            raise ValueError(f"Unsupported value '{item}'. Expected one of {sorted(valid_values)}")
        values.add(item)
    return values


def _parse_expert_ablation_set(raw_value):
    if not raw_value:
        return set()
    indices = set()
    for piece in str(raw_value).split(","):
        item = piece.strip()
        if not item:
            continue
        indices.add(int(item))
    return indices


def _configure_eval_controls(model, args):
    expert_indices = _parse_expert_ablation_set(getattr(args, "eval_ablate_experts", ""))
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "eval_ablate_expert_indices"):
            module.eval_ablate_expert_indices = set(expert_indices)
        if hasattr(module, "eval_ablate_shared_path"):
            module.eval_ablate_shared_path = bool(getattr(args, "eval_ablate_shared_path", False))
        if hasattr(module, "eval_ablate_routed_path"):
            module.eval_ablate_routed_path = bool(getattr(args, "eval_ablate_routed_path", False))


def _apply_eval_modality_mask(args, cxr_missing, text_missing, ecg_missing, reference_tensor):
    forced = _parse_csv_choice_set(
        getattr(args, "eval_force_missing_modalities", ""),
        {"text", "cxr", "ecg"},
    )
    missingness_level = float(getattr(args, "eval_missingness_level", 0.0) or 0.0)
    use_progressive = bool(getattr(args, "eval_progressive_missingness", False)) and missingness_level > 0.0
    if not forced and not use_progressive:
        return cxr_missing, text_missing, ecg_missing

    batch_size = reference_tensor.shape[0]
    device = reference_tensor.device
    sample_ids = getattr(args, "_current_eval_sample_ids", [""] * batch_size)
    eval_seed = int(getattr(args, "eval_missingness_seed", 0) or 0)

    def progressive_mask(existing_missing, enabled, modality_name):
        if existing_missing is None:
            missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            missing = existing_missing.to(device=device).bool().clone()
        if enabled:
            missing = missing | torch.ones(batch_size, dtype=torch.bool, device=device)
        if use_progressive and missingness_level > 0.0:
            extra = []
            for sample_id in sample_ids:
                digest = hashlib.md5(f"{sample_id}|{modality_name}|{missingness_level:.4f}|{eval_seed}".encode("utf-8")).hexdigest()
                value = int(digest[:8], 16) / float(16 ** 8)
                extra.append(value < missingness_level)
            extra_mask = torch.tensor(extra, dtype=torch.bool, device=device)
            missing = missing | extra_mask
        return missing

    return (
        progressive_mask(cxr_missing, "cxr" in forced, "cxr"),
        progressive_mask(text_missing, "text" in forced, "text"),
        progressive_mask(ecg_missing, "ecg" in forced, "ecg"),
    )


def _metadata_task_name(metadata, args):
    task_name = metadata.get("task_name") if isinstance(metadata, dict) else None
    if task_name:
        return task_name
    task = getattr(args, "task", "")
    if "pheno" in task:
        return "pheno"
    if "los" in task:
        return "los"
    return "ihm"


def _append_semantic_retention_history(args, epoch, split, metric_sums, metric_counts):
    output_path = getattr(args, "semantic_retention_history_path", None)
    if not output_path or not metric_sums:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "epoch": int(epoch),
        "split": split,
    }
    for name, total in sorted(metric_sums.items()):
        row[name] = total / max(metric_counts.get(name, 0), 1)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _accumulate_cohort_diagnostics(sums, counts, diagnostics_by_layer, labels, cxr_missing=None, text_missing=None, ecg_missing=None):
    labels = labels.detach().float().cpu()
    labels = labels.mean(dim=1) if labels.ndim > 1 else labels.view(-1)
    missing_items = {
        "label": labels,
        "cxr_missing": cxr_missing.detach().float().cpu().view(-1) if cxr_missing is not None else None,
        "text_missing": text_missing.detach().float().cpu().view(-1) if text_missing is not None else None,
        "ecg_missing": ecg_missing.detach().float().cpu().view(-1) if ecg_missing is not None else None,
    }
    for layer_idx, diagnostics in enumerate(diagnostics_by_layer):
        gates = diagnostics.get("gates")
        if gates is None:
            continue
        gates_list = gates if isinstance(gates, list) else [gates]
        while len(sums) <= layer_idx:
            sums.append({})
            counts.append({})
        for mod_idx, gate_tensor in enumerate(gates_list):
            key_prefix = f"m{mod_idx}" if isinstance(gates, list) else "joint"
            selected = gate_tensor.detach().float().cpu() > 0
            for expert_idx in range(selected.size(1)):
                mask = selected[:, expert_idx]
                n = int(mask.sum().item())
                if n == 0:
                    continue
                key = f"{key_prefix}e{expert_idx}"
                if key not in sums[layer_idx]:
                    sums[layer_idx][key] = {name: 0.0 for name in missing_items if missing_items[name] is not None}
                    counts[layer_idx][key] = 0
                counts[layer_idx][key] += n
                for name, values in missing_items.items():
                    if values is not None:
                        sums[layer_idx][key][name] += float(values[mask].sum().item())


def _format_cohort_diagnostics(sums, counts):
    parts = []
    for layer_idx, layer_sums in enumerate(sums):
        for key in sorted(layer_sums):
            n = max(counts[layer_idx].get(key, 0), 1)
            stats = ",".join(f"{name}={value / n:.3f}" for name, value in sorted(layer_sums[key].items()))
            parts.append(f"L{layer_idx}{key}:n={counts[layer_idx].get(key, 0)} {stats}")
    return " ; ".join(parts)


def _append_router_summary(summary_parts, prefix, layer_idx, gate_sum, select_sum, gate_count, modality_idx=None):
    if gate_sum is None or select_sum is None or gate_count <= 0:
        return
    usage = gate_sum / max(gate_count, 1)
    selection = select_sum / max(gate_count, 1)
    label = f"L{layer_idx}" if modality_idx is None else f"L{layer_idx}M{modality_idx}"
    summary_parts.append(f"{label}:mass{_router_usage_str(usage)} sel{_router_usage_str(selection)}")


def _set_moe_epoch(model, epoch):
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "set_current_epoch"):
            module.set_current_epoch(epoch)


def _find_all_expert_init_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    found = []
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_expert_init_diagnostics", None)
        if diagnostics:
            found.append(diagnostics)
    return found


def _find_collaboration_diagnostics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        diagnostics = getattr(module, "last_collaboration_diagnostics", None)
        if diagnostics is not None:
            return diagnostics
    return None


def _find_week34_aux_metrics(model):
    unwrapped = model.module if hasattr(model, "module") else model
    metrics = {}
    kd_loss = getattr(unwrapped, "last_unimodal_kd_loss", None)
    if kd_loss is not None:
        metrics["unimodal_kd_loss"] = kd_loss
    kd_details = getattr(unwrapped, "last_unimodal_kd_details", {}) or {}
    metrics.update(kd_details)
    consistency = getattr(unwrapped, "last_consistency_loss", None)
    if consistency is not None:
        metrics["consistency_loss"] = consistency
    ts_patch_metrics = _find_ts_patch_recon_metrics(unwrapped)
    metrics.update(ts_patch_metrics)
    ts_aux_loss = getattr(unwrapped, "last_ts_aux_loss", None)
    if ts_aux_loss is not None:
        metrics["ts_aux_loss"] = ts_aux_loss
    ts_aux_details = getattr(unwrapped, "last_ts_aux_details", {}) or {}
    metrics.update(ts_aux_details)
    return metrics


def _compute_consistency_penalty(full_latent, masked_latent, full_logits, masked_logits, args, task_name):
    if full_latent is None or masked_latent is None:
        return None
    loss_type = str(getattr(args, "consistency_loss_type", "mse"))
    latent_shapes_match = tuple(masked_latent.shape) == tuple(full_latent.shape)

    def _logit_fallback():
        if full_logits is None or masked_logits is None:
            return None
        if tuple(masked_logits.shape) != tuple(full_logits.shape):
            return None
        if task_name == "pheno":
            return F.mse_loss(masked_logits, full_logits.detach())
        return F.mse_loss(masked_logits, full_logits.detach())

    if loss_type == "mse":
        if not latent_shapes_match:
            return _logit_fallback()
        return F.mse_loss(masked_latent, full_latent.detach())
    if loss_type == "smooth_l1":
        if not latent_shapes_match:
            return _logit_fallback()
        return F.smooth_l1_loss(masked_latent, full_latent.detach())
    if loss_type == "cosine":
        if not latent_shapes_match:
            return _logit_fallback()
        return (1.0 - F.cosine_similarity(masked_latent, full_latent.detach(), dim=-1)).mean()
    if loss_type == "js":
        if full_logits is None or masked_logits is None:
            if not latent_shapes_match:
                return None
            return F.mse_loss(masked_latent, full_latent.detach())
        if task_name == "pheno":
            p = torch.sigmoid(full_logits.detach()).clamp(1e-6, 1 - 1e-6)
            q = torch.sigmoid(masked_logits).clamp(1e-6, 1 - 1e-6)
            m = 0.5 * (p + q)
            js = 0.5 * (
                F.kl_div(torch.log(q), m, reduction="batchmean")
                + F.kl_div(torch.log(p), m, reduction="batchmean")
            )
            return js
        p = torch.softmax(full_logits.detach(), dim=-1).clamp_min(1e-6)
        q = torch.softmax(masked_logits, dim=-1).clamp_min(1e-6)
        m = 0.5 * (p + q)
        return 0.5 * (
            F.kl_div(torch.log(q), m, reduction="batchmean")
            + F.kl_div(torch.log(p), m, reduction="batchmean")
        )
    raise ValueError(f"Unsupported consistency_loss_type: {loss_type}")


def _format_gate_row(gates, sample_idx):
    if isinstance(gates, list):
        parts = []
        for modality_idx, gate in enumerate(gates):
            row = gate[sample_idx]
            nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
            pieces = [f"{expert}:{float(row[expert]):.6g}" for expert in nonzero]
            parts.append(f"m{modality_idx}=" + "|".join(pieces))
        return ";".join(parts)

    row = gates[sample_idx]
    nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
    return "|".join(f"{expert}:{float(row[expert]):.6g}" for expert in nonzero)


def _format_expert_names(gates, sample_idx):
    if isinstance(gates, list):
        parts = []
        for modality_idx, gate in enumerate(gates):
            row = gate[sample_idx]
            nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
            names = [EXPERT_PROFILE_NAMES[expert % len(EXPERT_PROFILE_NAMES)] for expert in nonzero]
            parts.append(f"m{modality_idx}=" + "|".join(names))
        return ";".join(parts)

    row = gates[sample_idx]
    nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
    return "|".join(EXPERT_PROFILE_NAMES[expert % len(EXPERT_PROFILE_NAMES)] for expert in nonzero)


def _format_expert_indices(gates, sample_idx):
    if isinstance(gates, list):
        parts = []
        for modality_idx, gate in enumerate(gates):
            row = gate[sample_idx]
            nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
            parts.append(f"m{modality_idx}=" + "|".join(f"e{expert}" for expert in nonzero))
        return ";".join(parts)

    row = gates[sample_idx]
    nonzero = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
    return "|".join(f"e{expert}" for expert in nonzero)


def _write_router_diagnostics(args, metadata, labels, logits, model):
    if not getattr(args, "log_router_diagnostics", False):
        return

    if getattr(args, "router_diagnostics_layers", "last") == "all":
        diagnostics_list = _find_all_router_diagnostics(model)
    else:
        diagnostics = _find_router_diagnostics(model)
        diagnostics_list = [diagnostics] if diagnostics is not None else []
    if not diagnostics_list:
        return
    semantic_first_only = (
        getattr(args, "use_semantic_expert_profiles", False)
        and getattr(args, "semantic_profile_layers", "all") == "first"
    )

    output_path = args.router_diagnostics_path
    if output_path in (None, ""):
        base_output_dir = args.output_dir if getattr(args, "output_dir", None) else "."
        output_path = os.path.join(base_output_dir, "router_diagnostics_test.csv")
    output_dirname = os.path.dirname(output_path) or "."
    os.makedirs(output_dirname, exist_ok=True)

    sample_ids = metadata.get("sample_ids", [""] * len(labels))
    note_texts = metadata.get("note_texts", [[] for _ in range(len(labels))])
    expert_init_targets = metadata.get("expert_init_targets", [[] for _ in range(len(labels))])
    expert_init_confidence = metadata.get("expert_init_confidence", [0.0 for _ in range(len(labels))])
    expert_init_source = metadata.get("expert_init_source", ["" for _ in range(len(labels))])
    contribution_diagnostics = _find_all_contribution_diagnostics(model)
    primary_contrib = contribution_diagnostics[-1] if contribution_diagnostics else None
    collab_diagnostics = _find_collaboration_diagnostics(model)
    interaction_output = _find_interaction_feature_output(model)
    write_header = not os.path.exists(output_path)
    max_chars = getattr(args, "router_diagnostics_max_text_chars", 500)

    def _format_array_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            scalar = arr.reshape(-1)[0]
            if np.issubdtype(arr.dtype, np.integer):
                return str(int(scalar))
            return f"{float(scalar):.8g}"
        return "|".join(f"{float(x):.8g}" for x in arr.reshape(-1))

    def _prediction_fields(value):
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            score = float(arr.reshape(-1)[0])
            return f"{score:.8g}", str(int(score > 0.5))
        flat = arr.reshape(-1)
        if flat.size > 2:
            pred = (flat > 0.5).astype(int)
        else:
            pred = np.zeros_like(flat, dtype=int)
            pred[int(np.argmax(flat))] = 1
        return "|".join(f"{float(x):.8g}" for x in flat), "|".join(str(int(x)) for x in pred)

    def _retained_target(gates, sample_idx, target_values):
        if not target_values:
            return ""
        arr = np.asarray(target_values, dtype=float).reshape(-1)
        if arr.size == 0 or float(arr.sum()) <= 0:
            return ""
        row = gates[0][sample_idx] if isinstance(gates, list) else gates[sample_idx]
        pred_expert = int(torch.argmax(row).item())
        max_target = float(arr.max())
        target_top = {idx for idx, val in enumerate(arr) if abs(val - max_target) <= 1e-12}
        return str(int(pred_expert in target_top))

    def _as_tensor_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if item is not None]
        return [value]

    def _geometry_fieldnames(diagnostics):
        if not getattr(args, "log_router_geometry", False):
            return []
        max_dims = int(getattr(args, "router_geometry_max_dims", 256) or 0)
        fields = []
        router_inputs = _as_tensor_list(diagnostics.get("router_inputs"))
        clean_logits = _as_tensor_list(diagnostics.get("clean_logits"))
        for modality_idx, values in enumerate(router_inputs):
            if not isinstance(values, torch.Tensor) or values.ndim < 2:
                continue
            width = values.shape[1] if max_dims <= 0 else min(values.shape[1], max_dims)
            fields.extend(f"router_input_m{modality_idx}_d{dim_idx}" for dim_idx in range(width))
        for modality_idx, values in enumerate(clean_logits):
            if not isinstance(values, torch.Tensor) or values.ndim < 2:
                continue
            fields.extend(f"router_clean_logit_m{modality_idx}_e{expert_idx}" for expert_idx in range(values.shape[1]))
        return fields

    def _fill_geometry_fields(row, diagnostics, sample_idx):
        if not getattr(args, "log_router_geometry", False):
            return
        max_dims = int(getattr(args, "router_geometry_max_dims", 256) or 0)
        router_inputs = _as_tensor_list(diagnostics.get("router_inputs"))
        clean_logits = _as_tensor_list(diagnostics.get("clean_logits"))
        for modality_idx, values in enumerate(router_inputs):
            if not isinstance(values, torch.Tensor) or values.ndim < 2 or sample_idx >= values.shape[0]:
                continue
            width = values.shape[1] if max_dims <= 0 else min(values.shape[1], max_dims)
            for dim_idx, value in enumerate(values[sample_idx, :width].tolist()):
                row[f"router_input_m{modality_idx}_d{dim_idx}"] = f"{float(value):.8g}"
        for modality_idx, values in enumerate(clean_logits):
            if not isinstance(values, torch.Tensor) or values.ndim < 2 or sample_idx >= values.shape[0]:
                continue
            for expert_idx, value in enumerate(values[sample_idx].tolist()):
                row[f"router_clean_logit_m{modality_idx}_e{expert_idx}"] = f"{float(value):.8g}"

    layer_fields = []
    for layer_idx in range(len(diagnostics_list)):
        layer_fields.append(f"layer{layer_idx}_topk_expert_ids_weights")
        if semantic_first_only and layer_idx > 0:
            layer_fields.append(f"layer{layer_idx}_topk_expert_indices")
        else:
            layer_fields.append(f"layer{layer_idx}_topk_expert_profile_names")

    with open(output_path, "a", newline="") as f:
        fieldnames = [
            "sample_id",
            "true_label",
            "pred_score",
            "pred_label",
            "weak_organ_labels",
            "expert_init_target",
            "expert_init_confidence",
            "expert_init_source",
            "expert_init_retained",
            "task_name",
            "task_id",
            "observed_modality_mask",
            "observed_modality_mask_id",
            "topk_expert_ids_weights",
            "topk_expert_profile_names",
            *layer_fields,
            "note_text_snippet",
        ]
        fieldnames.extend(_geometry_fieldnames(diagnostics_list[-1]))
        if getattr(args, "log_per_sample_contributions", False):
            fieldnames.extend([
                "shared_output_norm",
                "routed_output_norm",
                "shared_contribution_ratio",
            ])
        if getattr(args, "log_collaboration_diagnostics", False):
            fieldnames.extend([
                "branch_agreement",
                "fusion_weight_moe",
                "fusion_weight_dense",
                "moe_pred_score",
                "moe_pred_label",
                "dense_pred_score",
                "dense_pred_label",
                "moe_correct",
                "dense_correct",
                "branch_case",
            ])
        if getattr(args, "log_interaction_features", False) and interaction_output is not None:
            fieldnames.extend(interaction_output.feature_names)
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        if write_header:
            writer.writeheader()

        for i in range(len(labels)):
            sample_notes = note_texts[i] if i < len(note_texts) else []
            if isinstance(sample_notes, str):
                sample_notes = [sample_notes]
            snippet = " ".join(str(t).replace("\n", " ") for t in sample_notes)[:max_chars]
            score, pred_label = _prediction_fields(logits[i])
            primary_gates = diagnostics_list[-1]["gates"]
            row = {
                "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                "true_label": _format_array_value(labels[i]),
                "pred_score": score,
                "pred_label": pred_label,
                "weak_organ_labels": "|".join(_weak_organ_labels(sample_notes)),
                "expert_init_target": _format_array_value(expert_init_targets[i]) if i < len(expert_init_targets) and expert_init_targets[i] else "",
                "expert_init_confidence": f"{float(expert_init_confidence[i]):.8g}" if i < len(expert_init_confidence) else "",
                "expert_init_source": expert_init_source[i] if i < len(expert_init_source) else "",
                "expert_init_retained": _retained_target(primary_gates, i, expert_init_targets[i] if i < len(expert_init_targets) else []),
                "task_name": diagnostics_list[-1].get("task_name", ""),
                "task_id": "" if diagnostics_list[-1].get("task_id", None) is None else _format_array_value(diagnostics_list[-1].get("task_id")[i] if isinstance(diagnostics_list[-1].get("task_id"), torch.Tensor) else diagnostics_list[-1].get("task_id")),
                "observed_modality_mask": diagnostics_list[-1].get("modality_mask_strings", [""] * len(labels))[i] if diagnostics_list[-1].get("modality_mask_strings") is not None else "",
                "observed_modality_mask_id": "" if diagnostics_list[-1].get("modality_mask_ids", None) is None else _format_array_value(diagnostics_list[-1].get("modality_mask_ids")[i] if hasattr(diagnostics_list[-1].get("modality_mask_ids"), "__getitem__") else diagnostics_list[-1].get("modality_mask_ids")),
                "topk_expert_ids_weights": _format_gate_row(primary_gates, i),
                "topk_expert_profile_names": _format_expert_names(primary_gates, i),
                "note_text_snippet": snippet,
            }
            _fill_geometry_fields(row, diagnostics_list[-1], i)
            if getattr(args, "log_per_sample_contributions", False) and primary_contrib is not None:
                row["shared_output_norm"] = f"{float(primary_contrib['shared_output_norm_per_sample'][i]):.8g}"
                row["routed_output_norm"] = f"{float(primary_contrib['routed_output_norm_per_sample'][i]):.8g}"
                row["shared_contribution_ratio"] = f"{float(primary_contrib['shared_contribution_ratio_per_sample'][i]):.8g}"
            if getattr(args, "log_collaboration_diagnostics", False) and collab_diagnostics is not None:
                row["branch_agreement"] = f"{float(collab_diagnostics['branch_agreement'][i]):.8g}"
                row["fusion_weight_moe"] = f"{float(collab_diagnostics['fusion_weight_moe'][i]):.8g}"
                row["fusion_weight_dense"] = f"{float(collab_diagnostics['fusion_weight_dense'][i]):.8g}"
                moe_score, moe_pred = _prediction_fields(collab_diagnostics["moe_logits"][i])
                dense_score, dense_pred = _prediction_fields(collab_diagnostics["dense_logits"][i])
                row["moe_pred_score"] = moe_score
                row["moe_pred_label"] = moe_pred
                row["dense_pred_score"] = dense_score
                row["dense_pred_label"] = dense_pred
                true_label = row["true_label"]
                if "|" in true_label or "|" in moe_pred:
                    truth = np.asarray([int(float(x)) for x in true_label.split("|")], dtype=int)
                    moe_arr = np.asarray([int(x) for x in moe_pred.split("|")], dtype=int)
                    dense_arr = np.asarray([int(x) for x in dense_pred.split("|")], dtype=int)
                    moe_correct = int(np.array_equal(truth, moe_arr))
                    dense_correct = int(np.array_equal(truth, dense_arr))
                else:
                    truth = int(float(true_label))
                    moe_correct = int(int(moe_pred) == truth)
                    dense_correct = int(int(dense_pred) == truth)
                row["moe_correct"] = str(moe_correct)
                row["dense_correct"] = str(dense_correct)
                if moe_correct and not dense_correct:
                    row["branch_case"] = "moe_only"
                elif dense_correct and not moe_correct:
                    row["branch_case"] = "dense_only"
                elif moe_correct and dense_correct:
                    row["branch_case"] = "both_correct"
                else:
                    row["branch_case"] = "both_wrong"
            if getattr(args, "log_interaction_features", False) and interaction_output is not None:
                for feature_name in interaction_output.feature_names:
                    values = interaction_output.feature_dict.get(feature_name)
                    row[feature_name] = _format_array_value(values[i]) if values is not None else ""
            for layer_idx, diagnostics in enumerate(diagnostics_list):
                gates = diagnostics["gates"]
                row[f"layer{layer_idx}_topk_expert_ids_weights"] = _format_gate_row(gates, i)
                if semantic_first_only and layer_idx > 0:
                    row[f"layer{layer_idx}_topk_expert_indices"] = _format_expert_indices(gates, i)
                else:
                    row[f"layer{layer_idx}_topk_expert_profile_names"] = _format_expert_names(gates, i)
            writer.writerow(row)

# def eval_test(args, model, test_data_loader, device):
#     model.eval()
#     rootdir = args.ck_file_path

#     os.makedirs(rootdir, exist_ok=True)

#     try:
#         result_dict = pickle.load(open(rootdir + "result.pkl", "rb"))
#     except:
#         result_dict = {}

#     if args.generate_data:
#         seed = str(args.datagereate_seed) + "_" + str(args.seed)
#     else:
#         seed = args.seed
#     result_dict[seed] = {}

#     for subdir, dirs, files in os.walk(rootdir):
#         if len(files) == 0 or "pkl" in files[0]:
#             continue
#         substr = subdir.split("/")[-1]
#         if substr == "model":
#             continue

#         file = str(seed) + ".pth.tar"
#         file_path = os.path.join(subdir, file)
#         print(file_path)
#         checkpoint = torch.load(file_path, map_location=device)
#         model.load_state_dict(checkpoint["network"])
#         test_val = evaluate_irg(args=args, device=device, data_loader=test_data_loader, model=model, mode="test")

#         for eval_type, val in test_val.items():
#             result_dict[seed][eval_type] = {}
#             result_dict[seed][eval_type]["val"] = checkpoint["best_val"][eval_type]
#             result_dict[seed][eval_type]["test"] = test_val[eval_type]

#             # wandb logging for test vs best-val
#             wandb.log({
#                 f"test/{eval_type}_val": checkpoint["best_val"][eval_type],
#                 f"test/{eval_type}_test": test_val[eval_type],
#             })

#     with open(rootdir + "/result.pkl", "wb") as f:
#         pickle.dump(result_dict, f)


def trainer_irg(
    model,
    args,
    accelerator,
    train_dataloader,
    dev_dataloader,
    test_data_loader,
    device,
    optimizer,
    pretrain_epoch=None,
    writer=None,
    scheduler=None,
    dataset="mimic",
):
    count = 0
    global_step = 0
    best_evals = {}
    primary_metric = args.primary_metric
    best_model_score = -float("inf")
    best_model_state = None
    total = len(train_dataloader)
    unwrapped_model = accelerator.unwrap_model(model)


    for epoch in tqdm(range(args.num_train_epochs)):
        args._current_epoch = epoch
        model.train()
        _set_moe_epoch(model, epoch)
        _configure_expert_init_trainability(model, args, epoch)
        if getattr(args, "dense_warmup_epochs", 0) > 0:
            phase = "dense" if epoch < args.dense_warmup_epochs else "topk"
            print(f"Router phase epoch {epoch}: {phase}")
        balance_loss_sum = 0.0
        balance_loss_count = 0
        router_gate_sum = None
        router_select_sum = None
        router_gate_count = 0
        router_layer_gate_sums = []
        router_layer_select_sums = []
        router_layer_counts = []
        router_layer_mod_gate_sums = []
        router_layer_mod_select_sums = []
        router_layer_mod_counts = []
        router_aux_sums = {}
        router_aux_count = 0
        specialization_metric_sums = {}
        specialization_metric_counts = {}
        expert_output_diag_sums = []
        expert_output_diag_counts = []
        cohort_diag_sums = []
        cohort_diag_counts = []
        expert_init_metric_sums = {}
        expert_init_metric_counts = {}
        if dataset == "mimic" and "Text" in args.modeltype and hasattr(model, "bertrep"):
            if (
                args.num_update_bert_epochs < args.num_train_epochs
                and (epoch) % args.num_update_bert_epochs == 0
                and count < args.bertcount
            ):
                count += 1
                print("bert update at epoch " + str(epoch))
                for param in model.bertrep.parameters():
                    param.requires_grad = True
            else:
                for param in model.bertrep.parameters():
                    param.requires_grad = False

            for param in model.bertrep.parameters():
                print(epoch, param.requires_grad)
                break

        none_count = 0
        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=total,
            miniters=max(1, total // 20),
            mininterval=1.5
        ):
            if batch is None:
                none_count += 1
                continue
            global_step += 1

            if dataset == "pam":
                # PAM data format can be (modality_list, labels) or
                # (modality_list, labels, subject_ids) for subject-wise evaluation.
                if len(batch) == 3:
                    modality_list, labels, _ = batch
                else:
                    modality_list, labels = batch
                # Move to device
                modality_list = [mod.to(device, non_blocking=True) for mod in modality_list]
                labels = labels.to(device, non_blocking=True)
                result = model(modality_list, labels=labels)
                if isinstance(result, tuple):
                    loss, balance_loss = result
                else:
                    loss = result
                    balance_loss = None
            else:
                batch_tensors, metadata = _split_mimic_batch(batch)
                _set_expert_init_batch(model, args, metadata, split="train", active=True)
                metadata_task_name = _metadata_task_name(metadata, args)
                router_organ_targets = (
                    _weak_organ_target_tensor(metadata, device)
                    if getattr(args, "use_router_organ_supervision", False)
                    else None
                )
                (
                    ts_input_sequences,
                    ts_mask_sequences,
                    ts_tt,
                    reg_ts,
                    input_ids_sequences,
                    attn_mask_sequences,
                    text_emb,
                    note_time,
                    note_time_mask,
                    cxr_feats,
                    cxr_time,
                    cxr_time_mask,
                    ecg_feats,
                    ecg_time,
                    ecg_time_mask,
                    label,
                    cxr_missing,
                    text_missing,
                    ecg_missing,
                ) = batch_tensors
                cxr_missing, text_missing, ecg_missing = _apply_train_modality_dropout(
                    args,
                    cxr_missing,
                    text_missing,
                    ecg_missing,
                    label,
                )
                teacher_targets = _resolve_unimodal_kd_batch(metadata, args, split="train")
                common_kwargs = {
                    "labels": label,
                    "task_name_override": metadata_task_name,
                    "unimodal_teacher_targets": teacher_targets,
                }

                if args.modeltype == "TS_Text":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        reg_ts=reg_ts,
                        router_organ_targets=router_organ_targets,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "TS_CXR":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        reg_ts=reg_ts,
                        router_organ_targets=router_organ_targets,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "TS_CXR_Text":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        router_organ_targets=router_organ_targets,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "TS_CXR_Text_ECG":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats,
                        ecg_time=ecg_time,
                        ecg_time_mask=ecg_time_mask,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        ecg_missing=ecg_missing,
                        router_organ_targets=router_organ_targets,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "Text_MOE":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "TS_MOE":
                    result = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        reg_ts=reg_ts,
                        **common_kwargs,
                    )
                    if isinstance(result, tuple):
                        loss, balance_loss = result
                    else:
                        loss = result
                        balance_loss = None
                elif args.modeltype == "TS":
                    loss = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        labels=label,
                        reg_ts=reg_ts,
                    )
                    balance_loss = None
                elif args.modeltype == "Text":
                    loss = model(
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        labels=label,
                    )
                    balance_loss = None

                if bool(getattr(args, "use_full_partial_consistency", False)):
                    full_latent = getattr(unwrapped_model, "last_fused_latent", None)
                    full_logits = getattr(unwrapped_model, "last_output_logits", None)
                    masked_cxr_missing, masked_text_missing, masked_ecg_missing = _apply_consistency_train_mask(
                        args,
                        cxr_missing,
                        text_missing,
                        ecg_missing,
                        label,
                    )
                    if args.modeltype == "TS_CXR_Text":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            input_ids_sequences=input_ids_sequences,
                            attn_mask_sequences=attn_mask_sequences,
                            text_emb=text_emb,
                            note_time_list=note_time,
                            note_time_mask_list=note_time_mask,
                            cxr_feats=cxr_feats,
                            cxr_time=cxr_time,
                            cxr_time_mask=cxr_time_mask,
                            reg_ts=reg_ts,
                            cxr_missing=masked_cxr_missing,
                            text_missing=masked_text_missing,
                            router_organ_targets=router_organ_targets,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    elif args.modeltype == "TS_CXR_Text_ECG":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            input_ids_sequences=input_ids_sequences,
                            attn_mask_sequences=attn_mask_sequences,
                            text_emb=text_emb,
                            note_time_list=note_time,
                            note_time_mask_list=note_time_mask,
                            cxr_feats=cxr_feats,
                            cxr_time=cxr_time,
                            cxr_time_mask=cxr_time_mask,
                            ecg_feats=ecg_feats,
                            ecg_time=ecg_time,
                            ecg_time_mask=ecg_time_mask,
                            reg_ts=reg_ts,
                            cxr_missing=masked_cxr_missing,
                            text_missing=masked_text_missing,
                            ecg_missing=masked_ecg_missing,
                            router_organ_targets=router_organ_targets,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    elif args.modeltype == "TS_Text":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            input_ids_sequences=input_ids_sequences,
                            attn_mask_sequences=attn_mask_sequences,
                            text_emb=text_emb,
                            note_time_list=note_time,
                            note_time_mask_list=note_time_mask,
                            reg_ts=reg_ts,
                            text_missing=masked_text_missing,
                            router_organ_targets=router_organ_targets,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    elif args.modeltype == "TS_CXR":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            cxr_feats=cxr_feats,
                            cxr_time=cxr_time,
                            cxr_time_mask=cxr_time_mask,
                            reg_ts=reg_ts,
                            cxr_missing=masked_cxr_missing,
                            router_organ_targets=router_organ_targets,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    elif args.modeltype == "TS_MOE":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            reg_ts=reg_ts,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    elif args.modeltype == "Text_MOE":
                        _ = model(
                            x_ts=ts_input_sequences,
                            x_ts_mask=ts_mask_sequences,
                            ts_tt_list=ts_tt,
                            input_ids_sequences=input_ids_sequences,
                            attn_mask_sequences=attn_mask_sequences,
                            text_emb=text_emb,
                            note_time_list=note_time,
                            note_time_mask_list=note_time_mask,
                            text_missing=masked_text_missing,
                            task_name_override=metadata_task_name,
                            unimodal_teacher_targets=None,
                        )
                    else:
                        _ = None

                    masked_latent = getattr(unwrapped_model, "last_fused_latent", None)
                    masked_logits = getattr(unwrapped_model, "last_output_logits", None)
                    consistency_penalty = _compute_consistency_penalty(
                        full_latent,
                        masked_latent,
                        full_logits,
                        masked_logits,
                        args,
                        metadata_task_name,
                    )
                    if consistency_penalty is not None:
                        setattr(unwrapped_model, "last_consistency_loss", consistency_penalty.detach())
                        if loss is not None:
                            loss = loss + float(getattr(args, "consistency_weight", 0.0)) * consistency_penalty
                    else:
                        setattr(unwrapped_model, "last_consistency_loss", None)
                else:
                    setattr(unwrapped_model, "last_consistency_loss", None)

            if loss is None:
                warnings.warn("loss is None!")
                continue

            # Incorporate balance_loss if enabled and available
            if hasattr(args, "use_balance_loss") and args.use_balance_loss and balance_loss is not None:
                if getattr(args, "specialization_loss_mode", "legacy") == "paper":
                    total_loss = loss + balance_loss
                else:
                    total_loss = loss + args.balance_loss_coef * balance_loss
            else:
                total_loss = loss

            total_loss = total_loss / args.gradient_accumulation_steps
            accelerator.backward(total_loss)

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()

            # tensorboard (existing)
            if writer is not None:
                writer.add_scalar("training/train_loss", loss, global_step)
                if balance_loss is not None:
                    writer.add_scalar("training/balance_loss", balance_loss, global_step)
                if hasattr(args, "use_balance_loss") and args.use_balance_loss and balance_loss is not None:
                    writer.add_scalar("training/total_loss", total_loss, global_step)

            # wandb logging
            log_dict = {
                "train/loss": float(loss.detach().cpu().item()),
                "train/total_loss": float(total_loss.detach().cpu().item()),
                "train/global_step": global_step,
            }
            if balance_loss is not None:
                try:
                    balance_value = float(balance_loss.detach().cpu().item())
                    log_dict["train/balance_loss"] = balance_value
                    balance_loss_sum += balance_value
                    balance_loss_count += 1
                    aux_components = _find_all_router_aux_components(unwrapped_model)
                    if aux_components:
                        step_components = {}
                        for components in aux_components:
                            for name, value in components.items():
                                try:
                                    component_value = float(value.detach().cpu().item())
                                except Exception:
                                    component_value = float(value)
                                step_components[name] = step_components.get(name, 0.0) + component_value
                        for name, value in step_components.items():
                            router_aux_sums[name] = router_aux_sums.get(name, 0.0) + value
                        router_aux_count += 1
                    specialization_diagnostics = _find_all_specialization_diagnostics(unwrapped_model)
                    _accumulate_metric_dict(
                        specialization_metric_sums,
                        specialization_metric_counts,
                        specialization_diagnostics,
                    )
                    _accumulate_metric_dict(
                        expert_init_metric_sums,
                        expert_init_metric_counts,
                        _find_all_expert_init_diagnostics(unwrapped_model),
                    )
                except Exception:
                    pass
            week34_metrics = _find_week34_aux_metrics(unwrapped_model)
            for name, value in week34_metrics.items():
                try:
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            continue
                        scalar = float(value.detach().cpu().item())
                    else:
                        scalar = float(value)
                    log_dict[f"train/{name}"] = scalar
                except Exception:
                    continue
            try:
                unwrapped_model = accelerator.unwrap_model(model)
                missing_recon_loss = getattr(unwrapped_model, "last_missing_recon_loss", None)
                if missing_recon_loss is not None:
                    log_dict["train/missing_recon_loss"] = float(missing_recon_loss.detach().cpu().item())
                    details = getattr(unwrapped_model, "last_missing_recon_details", {})
                    for target_name, target_value in details.items():
                        log_dict[f"train/missing_recon_{target_name}"] = float(target_value)
                last_router_gates = getattr(unwrapped_model, "last_router_gates", None)
                if last_router_gates is not None:
                    detached_gates = last_router_gates.detach().float()
                    gate_sum = detached_gates.sum(dim=0).cpu()
                    select_sum = (detached_gates > 0).float().sum(dim=0).cpu()
                    router_gate_sum = gate_sum if router_gate_sum is None else router_gate_sum + gate_sum
                    router_select_sum = select_sum if router_select_sum is None else router_select_sum + select_sum
                    router_gate_count += int(last_router_gates.size(0))
                else:
                    all_diagnostics = _find_all_router_diagnostics(unwrapped_model)
                    for layer_idx, diagnostics in enumerate(all_diagnostics):
                        gates = diagnostics.get("gates")
                        if isinstance(gates, list):
                            detached_gates = [g.detach().float() for g in gates]
                            gate_sum = sum(g.sum(dim=0).cpu() for g in detached_gates)
                            select_sum = sum((g > 0).float().sum(dim=0).cpu() for g in detached_gates)
                            gate_count = sum(int(g.size(0)) for g in gates)
                            while len(router_layer_mod_gate_sums) <= layer_idx:
                                router_layer_mod_gate_sums.append([])
                                router_layer_mod_select_sums.append([])
                                router_layer_mod_counts.append([])
                            for mod_idx, mod_gates in enumerate(detached_gates):
                                while len(router_layer_mod_gate_sums[layer_idx]) <= mod_idx:
                                    router_layer_mod_gate_sums[layer_idx].append(None)
                                    router_layer_mod_select_sums[layer_idx].append(None)
                                    router_layer_mod_counts[layer_idx].append(0)
                                mod_gate_sum = mod_gates.sum(dim=0).cpu()
                                mod_select_sum = (mod_gates > 0).float().sum(dim=0).cpu()
                                router_layer_mod_gate_sums[layer_idx][mod_idx] = (
                                    mod_gate_sum
                                    if router_layer_mod_gate_sums[layer_idx][mod_idx] is None
                                    else router_layer_mod_gate_sums[layer_idx][mod_idx] + mod_gate_sum
                                )
                                router_layer_mod_select_sums[layer_idx][mod_idx] = (
                                    mod_select_sum
                                    if router_layer_mod_select_sums[layer_idx][mod_idx] is None
                                    else router_layer_mod_select_sums[layer_idx][mod_idx] + mod_select_sum
                                )
                                router_layer_mod_counts[layer_idx][mod_idx] += int(mod_gates.size(0))
                        else:
                            detached_gates = gates.detach().float()
                            gate_sum = detached_gates.sum(dim=0).cpu()
                            select_sum = (detached_gates > 0).float().sum(dim=0).cpu()
                            gate_count = int(gates.size(0))

                        while len(router_layer_gate_sums) <= layer_idx:
                            router_layer_gate_sums.append(None)
                            router_layer_select_sums.append(None)
                            router_layer_counts.append(0)

                        router_layer_gate_sums[layer_idx] = (
                            gate_sum
                            if router_layer_gate_sums[layer_idx] is None
                            else router_layer_gate_sums[layer_idx] + gate_sum
                        )
                        router_layer_select_sums[layer_idx] = (
                            select_sum
                            if router_layer_select_sums[layer_idx] is None
                            else router_layer_select_sums[layer_idx] + select_sum
                        )
                        router_layer_counts[layer_idx] += gate_count
                    if getattr(args, "log_expert_output_diagnostics", False):
                        _accumulate_expert_output_diagnostics(
                            expert_output_diag_sums,
                            expert_output_diag_counts,
                            _find_all_expert_output_diagnostics(unwrapped_model),
                        )
                        if dataset == "mimic":
                            _accumulate_cohort_diagnostics(
                                cohort_diag_sums,
                                cohort_diag_counts,
                                all_diagnostics,
                                label,
                                cxr_missing=cxr_missing,
                                text_missing=text_missing,
                                ecg_missing=ecg_missing,
                            )
            except Exception:
                pass
            # learning rate(s)
            try:
                lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                if len(lrs) == 1:
                    log_dict["train/lr"] = lrs[0]
                else:
                    for i, lr in enumerate(lrs):
                        log_dict[f"train/lr_group_{i}"] = lr
            except Exception:
                pass
            # wandb.log(log_dict)

        if none_count > 0:
            print("none_count", none_count)
            if none_count == total:
                print("warning: all training batches returned None; check modeltype/MoE path for invalid tensors")
        router_print_mode = getattr(args, "router_print_mode", "verbose")
        if balance_loss_count > 0 and router_print_mode != "none":
            print("Train router_aux_loss avg", balance_loss_sum / balance_loss_count)
        if router_aux_count > 0 and router_print_mode != "none":
            component_str = ", ".join(
                f"{name}:{router_aux_sums[name] / router_aux_count:.6f}"
                for name in sorted(router_aux_sums)
            )
            print("Train router aux components avg", component_str)
        if specialization_metric_sums and router_print_mode != "none":
            print("Train specialization diagnostics avg", _format_metric_sums(specialization_metric_sums, specialization_metric_counts))
            _append_semantic_retention_history(args, epoch, "train", specialization_metric_sums, specialization_metric_counts)
        if expert_init_metric_sums and router_print_mode != "none":
            print("Train expert-init diagnostics avg", _format_metric_sums(expert_init_metric_sums, expert_init_metric_counts))
        if getattr(args, "log_expert_output_diagnostics", False) and expert_output_diag_sums and router_print_mode != "none":
            print("Train expert output similarity", _format_expert_output_diagnostics(expert_output_diag_sums, expert_output_diag_counts))
        if getattr(args, "log_expert_output_diagnostics", False) and cohort_diag_sums and router_print_mode != "none":
            print("Train expert cohort stats", _format_cohort_diagnostics(cohort_diag_sums, cohort_diag_counts))
        router_temperatures = _find_all_router_temperatures(unwrapped_model)
        if router_temperatures and router_print_mode != "none":
            temp_str = ", ".join(f"tau{i}:{v:.6f}" for i, v in enumerate(router_temperatures))
            print("Train router learned temperatures", temp_str)
        train_router_summary = []
        if router_gate_sum is not None and router_select_sum is not None and router_gate_count > 0:
            _append_router_summary(train_router_summary, "Train", "last", router_gate_sum, router_select_sum, router_gate_count)
        for layer_idx, (gate_sum, select_sum, gate_count) in enumerate(
            zip(router_layer_gate_sums, router_layer_select_sums, router_layer_counts)
        ):
            if router_print_mode == "concise":
                _append_router_summary(train_router_summary, "Train", layer_idx, gate_sum, select_sum, gate_count)
            elif router_print_mode == "verbose":
                if gate_sum is None or select_sum is None or gate_count <= 0:
                    continue
                usage = gate_sum / gate_count
                usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
                selection = select_sum / gate_count
                selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
                print(f"Train router layer{layer_idx} gate mass avg", usage_str)
                print(f"Train router layer{layer_idx} selection freq avg", selection_str)
        for layer_idx, (mod_gate_sums, mod_select_sums, mod_counts) in enumerate(
            zip(router_layer_mod_gate_sums, router_layer_mod_select_sums, router_layer_mod_counts)
        ):
            for mod_idx, (gate_sum, select_sum, gate_count) in enumerate(
                zip(mod_gate_sums, mod_select_sums, mod_counts)
            ):
                if router_print_mode == "concise":
                    _append_router_summary(train_router_summary, "Train", layer_idx, gate_sum, select_sum, gate_count, modality_idx=mod_idx)
                elif router_print_mode == "verbose":
                    if gate_sum is None or select_sum is None or gate_count <= 0:
                        continue
                    usage = gate_sum / gate_count
                    usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
                    selection = select_sum / gate_count
                    selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
                    print(f"Train router layer{layer_idx} modality{mod_idx} gate mass avg", usage_str)
                    print(f"Train router layer{layer_idx} modality{mod_idx} selection freq avg", selection_str)
        if router_print_mode == "concise" and train_router_summary:
            print("Train router summary", " ; ".join(train_router_summary))

        # --- Evaluate on dev set ---
        if isinstance(dev_dataloader, dict):
            eval_vals = {}
            multitask_scores = []
            for task_name, task_loader in dev_dataloader.items():
                task_args = copy.copy(args)
                task_args.task = task_name
                task_args.num_labels = 25 if task_name == "pheno" else 2
                task_eval = evaluate_irg(task_args, device, task_loader, model, mode="val", dataset=dataset)
                for key, value in task_eval.items():
                    eval_vals[f"{task_name}_{key}"] = value
                primary_key = "macro_f1" if task_name == "pheno" else "f1"
                if primary_key in task_eval:
                    multitask_scores.append(float(task_eval[primary_key]))
            if multitask_scores:
                eval_vals["multitask_mean"] = float(np.mean(multitask_scores))
        else:
            eval_vals = evaluate_irg(args, device, dev_dataloader, model, mode="val", dataset=dataset)
        for k, v in eval_vals.items():
            if k == "auc_scores":
                continue
            try:
                scalar_v = float(v)
            except (TypeError, ValueError):
                print("Current " + k, "[non-scalar metric omitted from best-tracking]")
                continue
            if writer is not None:
                writer.add_scalar("dev/" + k, scalar_v, epoch + 1)

            best_eval = best_evals.get(k, 0)
            if scalar_v > best_eval:
                best_eval = scalar_v
                best_evals[k] = best_eval
            print("Current " + k, scalar_v)
            print("Best " + k, best_eval)

            # wandb logging for dev metrics and best-so-far
            # wandb.log({f"dev/{k}": v, f"dev/best_{k}": best_eval, "epoch": epoch + 1})
        
        score_key = primary_metric
        if score_key not in eval_vals:
            multitask_fallbacks = [f"pheno_{primary_metric}", "multitask_mean"]
            for candidate in multitask_fallbacks:
                if candidate in eval_vals:
                    score_key = candidate
                    break
        if score_key not in eval_vals:
            suffix_matches = [
                key for key, value in eval_vals.items()
                if key.endswith(f"_{primary_metric}") and np.isscalar(value)
            ]
            suffix_matches.sort(key=lambda key: (0 if key.startswith("pheno_") else 1, key))
            if suffix_matches:
                score_key = suffix_matches[0]
        if score_key in eval_vals:
            current_score = float(eval_vals[score_key])
            if current_score > best_model_score:
                best_model_score = current_score
                best_model_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())

        if writer is not None:
            writer.close()
    # --- End of training loop ---
    return best_model_state, best_model_score



def evaluate_irg(args, device, data_loader, model, mode=None, dataset="mimic", r2t2_collector=None):
    model.eval()
    _configure_eval_controls(model, args)
    eval_logits = []
    eval_example = []
    eval_subjects = []
    none_count = 0
    eval_router_layer_gate_sums = []
    eval_router_layer_select_sums = []
    eval_router_layer_counts = []
    eval_router_layer_mod_gate_sums = []
    eval_router_layer_mod_select_sums = []
    eval_router_layer_mod_counts = []
    eval_specialization_metric_sums = {}
    eval_specialization_metric_counts = {}
    eval_expert_init_metric_sums = {}
    eval_expert_init_metric_counts = {}
    eval_missing_recon_metric_sums = {}
    eval_missing_recon_metric_counts = {}
    eval_expert_output_diag_sums = []
    eval_expert_output_diag_counts = []
    eval_cohort_diag_sums = []
    eval_cohort_diag_counts = []
    total = len(data_loader)
    for idx, batch in enumerate(
        tqdm(
            data_loader,
            total=total,
            miniters=max(1, total // 20),
            mininterval=1.5
        )
    ):
        if batch is None:
            none_count += 1
            continue

        if dataset == "pam":
            if len(batch) == 3:
                modality_list, labels, subject_ids = batch
            else:
                modality_list, labels = batch
                subject_ids = None
            modality_list = [mod.to(device) for mod in modality_list]
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(modality_list)
            if logits is None:
                warnings.warn("logits is None!")
                continue
            if torch.isnan(logits).any():
                warnings.warn("logits is nan!")
                continue
            logits = logits.cpu().numpy()
            label_ids = labels.cpu().numpy()
            eval_logits += logits.tolist()
            eval_example += label_ids.tolist()
            if subject_ids is not None:
                eval_subjects += subject_ids.cpu().numpy().tolist()
        else:
            batch_tensors, metadata = _split_mimic_batch(batch)
            args._current_eval_sample_ids = list(metadata.get("sample_ids", []))
            metadata_task_name = _metadata_task_name(metadata, args)
            if mode == "test":
                eval_split_name = "test"
            elif getattr(args, "mode", "train") == "eval":
                eval_split_name = getattr(args, "eval_split", "val")
            else:
                eval_split_name = "val"
            if eval_split_name not in {"train", "val", "test"}:
                eval_split_name = "val"
            _set_expert_init_batch(model, args, metadata, split=eval_split_name, active=False)
            (
                ts_input_sequences,
                ts_mask_sequences,
                ts_tt,
                reg_ts,
                input_ids_sequences,
                attn_mask_sequences,
                text_emb,
                note_time,
                note_time_mask,
                cxr_feats,
                cxr_time,
                cxr_time_mask,
                ecg_feats,
                ecg_time,
                ecg_time_mask,
                label,
                cxr_missing,
                text_missing,
                ecg_missing,
            ) = batch_tensors
            with torch.no_grad():
                if args.modeltype == "TS_Text":
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        reg_ts=reg_ts,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "TS_CXR":
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        reg_ts=reg_ts,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "TS_CXR_Text":
                    cxr_missing, text_missing, ecg_missing = _apply_eval_modality_mask(
                        args, cxr_missing, text_missing, ecg_missing, label
                    )
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "TS_CXR_Text_ECG":
                    cxr_missing, text_missing, ecg_missing = _apply_eval_modality_mask(
                        args, cxr_missing, text_missing, ecg_missing, label
                    )
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats,
                        ecg_time=ecg_time,
                        ecg_time_mask=ecg_time_mask,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        ecg_missing=ecg_missing,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "Text_MOE":
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "TS_MOE":
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        reg_ts=reg_ts,
                        task_name_override=metadata_task_name,
                    )
                elif args.modeltype == "TS":
                    logits = model(
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        reg_ts=reg_ts,
                    )
                elif args.modeltype == "Text":
                    logits = model(
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                    )

            if logits is None:
                warnings.warn("logits is None!")
                continue
            if logits.numel() == 0 or logits.size(0) == 0:
                warnings.warn(
                    f"logits is empty in mode={mode} task={args.task} batch_idx={idx} "
                    f"sample_ids={metadata.get('sample_ids', [])[:3]}"
                )
                continue
            if not torch.isfinite(logits).all():
                finite_ratio = float(torch.isfinite(logits).float().mean().item())
                warnings.warn(
                    f"logits is non-finite in mode={mode} task={args.task} batch_idx={idx} "
                    f"finite_ratio={finite_ratio:.4f} sample_ids={metadata.get('sample_ids', [])[:3]}"
                )
                continue

            logits_np = logits.cpu().numpy()
            label_ids = label.cpu().numpy()
            if mode == "test":
                _write_router_diagnostics(args, metadata, label_ids, logits_np, model)
            if r2t2_collector is not None:
                all_diagnostics_for_r2t2 = [
                    d for d in _find_all_router_diagnostics(model) if d is not None
                ]
                r2t2_collector.add_batch(all_diagnostics_for_r2t2, logits_np, label_ids)
            if mode in {"val", "test"}:
                specialization_diagnostics = _find_all_specialization_diagnostics(model)
                _accumulate_metric_dict(
                    eval_specialization_metric_sums,
                    eval_specialization_metric_counts,
                    specialization_diagnostics,
                )
                _accumulate_metric_dict(
                    eval_expert_init_metric_sums,
                    eval_expert_init_metric_counts,
                    _find_all_expert_init_diagnostics(model),
                )
                _accumulate_metric_dict(
                    eval_missing_recon_metric_sums,
                    eval_missing_recon_metric_counts,
                    [_find_missing_recon_metrics(model)],
                )
                all_diagnostics = _find_all_router_diagnostics(model)
                all_diagnostics = [d for d in all_diagnostics if d is not None]
                if getattr(args, "log_expert_output_diagnostics", False):
                    _accumulate_expert_output_diagnostics(
                        eval_expert_output_diag_sums,
                        eval_expert_output_diag_counts,
                        _find_all_expert_output_diagnostics(model),
                    )
                    if dataset == "mimic":
                        _accumulate_cohort_diagnostics(
                            eval_cohort_diag_sums,
                            eval_cohort_diag_counts,
                            all_diagnostics,
                            label,
                            cxr_missing=cxr_missing,
                            text_missing=text_missing,
                            ecg_missing=ecg_missing,
                        )
                for layer_idx, diagnostics in enumerate(all_diagnostics):
                    gates = diagnostics.get("gates")
                    if gates is None:
                        continue
                    if isinstance(gates, list):
                        detached_gates = [g.detach().float() for g in gates]
                        gate_sum = sum(g.sum(dim=0).cpu() for g in detached_gates)
                        select_sum = sum((g > 0).float().sum(dim=0).cpu() for g in detached_gates)
                        gate_count = sum(int(g.size(0)) for g in gates)
                        while len(eval_router_layer_mod_gate_sums) <= layer_idx:
                            eval_router_layer_mod_gate_sums.append([])
                            eval_router_layer_mod_select_sums.append([])
                            eval_router_layer_mod_counts.append([])
                        for mod_idx, mod_gates in enumerate(detached_gates):
                            while len(eval_router_layer_mod_gate_sums[layer_idx]) <= mod_idx:
                                eval_router_layer_mod_gate_sums[layer_idx].append(None)
                                eval_router_layer_mod_select_sums[layer_idx].append(None)
                                eval_router_layer_mod_counts[layer_idx].append(0)
                            mod_gate_sum = mod_gates.sum(dim=0).cpu()
                            mod_select_sum = (mod_gates > 0).float().sum(dim=0).cpu()
                            eval_router_layer_mod_gate_sums[layer_idx][mod_idx] = (
                                mod_gate_sum
                                if eval_router_layer_mod_gate_sums[layer_idx][mod_idx] is None
                                else eval_router_layer_mod_gate_sums[layer_idx][mod_idx] + mod_gate_sum
                            )
                            eval_router_layer_mod_select_sums[layer_idx][mod_idx] = (
                                mod_select_sum
                                if eval_router_layer_mod_select_sums[layer_idx][mod_idx] is None
                                else eval_router_layer_mod_select_sums[layer_idx][mod_idx] + mod_select_sum
                            )
                            eval_router_layer_mod_counts[layer_idx][mod_idx] += int(mod_gates.size(0))
                    else:
                        detached_gates = gates.detach().float()
                        gate_sum = detached_gates.sum(dim=0).cpu()
                        select_sum = (detached_gates > 0).float().sum(dim=0).cpu()
                        gate_count = int(gates.size(0))

                    while len(eval_router_layer_gate_sums) <= layer_idx:
                        eval_router_layer_gate_sums.append(None)
                        eval_router_layer_select_sums.append(None)
                        eval_router_layer_counts.append(0)

                    eval_router_layer_gate_sums[layer_idx] = (
                        gate_sum
                        if eval_router_layer_gate_sums[layer_idx] is None
                        else eval_router_layer_gate_sums[layer_idx] + gate_sum
                    )
                    eval_router_layer_select_sums[layer_idx] = (
                        select_sum
                        if eval_router_layer_select_sums[layer_idx] is None
                        else eval_router_layer_select_sums[layer_idx] + select_sum
                    )
                    eval_router_layer_counts[layer_idx] += gate_count
            eval_logits += logits_np.tolist()
            eval_example += label_ids.tolist()

    if none_count > 0:
        print("none_count", none_count)
    router_print_mode = getattr(args, "router_print_mode", "verbose")
    if mode in {"val", "test"} and eval_router_layer_gate_sums and router_print_mode != "none":
        mode_label = "Validation" if mode == "val" else "Test"
        eval_router_summary = []
        for layer_idx, (gate_sum, select_sum, gate_count) in enumerate(
            zip(eval_router_layer_gate_sums, eval_router_layer_select_sums, eval_router_layer_counts)
        ):
            if router_print_mode == "concise":
                _append_router_summary(eval_router_summary, mode_label, layer_idx, gate_sum, select_sum, gate_count)
            else:
                if gate_sum is None or gate_count <= 0:
                    continue
                usage = gate_sum / max(gate_count, 1)
                usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
                print(f"{mode_label} router layer{layer_idx} gate mass avg", usage_str)
                selection = select_sum / max(gate_count, 1)
                selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
                print(f"{mode_label} router layer{layer_idx} selection freq avg", selection_str)
        for layer_idx, (mod_gate_sums, mod_select_sums, mod_counts) in enumerate(
            zip(eval_router_layer_mod_gate_sums, eval_router_layer_mod_select_sums, eval_router_layer_mod_counts)
        ):
            for mod_idx, (gate_sum, select_sum, gate_count) in enumerate(
                zip(mod_gate_sums, mod_select_sums, mod_counts)
            ):
                if router_print_mode == "concise":
                    _append_router_summary(eval_router_summary, mode_label, layer_idx, gate_sum, select_sum, gate_count, modality_idx=mod_idx)
                else:
                    if gate_sum is None or select_sum is None or gate_count <= 0:
                        continue
                    usage = gate_sum / max(gate_count, 1)
                    usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
                    print(f"{mode_label} router layer{layer_idx} modality{mod_idx} gate mass avg", usage_str)
                    selection = select_sum / max(gate_count, 1)
                    selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
                    print(f"{mode_label} router layer{layer_idx} modality{mod_idx} selection freq avg", selection_str)
        if router_print_mode == "concise" and eval_router_summary:
            print(f"{mode_label} router summary", " ; ".join(eval_router_summary))
        if eval_specialization_metric_sums:
            print(f"{mode_label} specialization diagnostics avg", _format_metric_sums(eval_specialization_metric_sums, eval_specialization_metric_counts))
            _append_semantic_retention_history(args, getattr(args, "_current_epoch", -1), mode, eval_specialization_metric_sums, eval_specialization_metric_counts)
        if eval_expert_init_metric_sums:
            print(f"{mode_label} expert-init diagnostics avg", _format_metric_sums(eval_expert_init_metric_sums, eval_expert_init_metric_counts))
        if eval_missing_recon_metric_sums:
            print(f"{mode_label} missing-recovery diagnostics avg", _format_metric_sums(eval_missing_recon_metric_sums, eval_missing_recon_metric_counts))
        if getattr(args, "log_expert_output_diagnostics", False) and eval_expert_output_diag_sums:
            print(f"{mode_label} expert output similarity", _format_expert_output_diagnostics(eval_expert_output_diag_sums, eval_expert_output_diag_counts))
        if getattr(args, "log_expert_output_diagnostics", False) and eval_cohort_diag_sums:
            print(f"{mode_label} expert cohort stats", _format_cohort_diagnostics(eval_cohort_diag_sums, eval_cohort_diag_counts))

    eval_vals = {}
    all_logits = np.array(eval_logits)
    all_label = np.array(eval_example)
    if all_logits.size == 0 or all_label.size == 0:
        raise RuntimeError(
            f"Empty evaluation set in mode={mode} task={args.task}. "
            f"Validation/test batches produced no finite logits. "
            f"none_count={none_count}, collected_logits={len(eval_logits)}, collected_labels={len(eval_example)}"
        )
    all_pred = np.where(all_logits > 0.5, 1, 0)

    if args.dataset == "pam":
        # PAM: multiclass classification
        # Softmax to get probabilities
        exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
        all_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_pred = np.argmax(all_probs, axis=1)

        n_classes = all_probs.shape[1]
        y_onehot = label_binarize(all_label, classes=range(n_classes))

        eval_vals["acc"] = accuracy_score(all_label, all_pred)
        eval_vals["macro_f1"] = f1_score(all_label, all_pred, average='macro')
        try:
            eval_vals["auc_macro"] = roc_auc_score(y_onehot, all_probs, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Error computing AUC: {e}")
            eval_vals["auc_macro"] = 0.0

        if mode == "test" and len(eval_subjects) == len(all_label):
            subject_ids = sorted(set(eval_subjects))
            for subject_id in subject_ids:
                subject_mask = np.array(eval_subjects) == subject_id
                subject_labels = all_label[subject_mask]
                subject_probs = all_probs[subject_mask]
                subject_pred = np.argmax(subject_probs, axis=1)
                eval_vals[f"subject_{subject_id}_acc"] = accuracy_score(subject_labels, subject_pred)
                eval_vals[f"subject_{subject_id}_macro_f1"] = f1_score(subject_labels, subject_pred, average='macro')

    elif "pheno" in args.task:
        eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
        eval_vals["macro_f1"] = f1_score(all_label, all_pred, average="macro")

        # if mode is None:
        #     check_point(eval_vals, model, eval_logits, args, "macro_f1")

    elif "ihm" in args.task or "los" in args.task:
        eval_val = roc_auc_score(np.array(eval_example), np.array(eval_logits))
        eval_vals["auc"] = eval_val
        (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_example), np.array(eval_logits))
        eval_val = auc(recalls, precisions)
        eval_vals["auprc"] = eval_val
        eval_val = f1_score(np.array(eval_example), all_pred)
        eval_vals["f1"] = eval_val
        # if mode is None:
        #     check_point(eval_vals, model, eval_logits, args, "f1")

    if hasattr(args, "_current_eval_sample_ids"):
        args._current_eval_sample_ids = []
    for name, total in sorted(eval_missing_recon_metric_sums.items()):
        eval_vals[name] = total / max(eval_missing_recon_metric_counts.get(name, 0), 1)
    return eval_vals
