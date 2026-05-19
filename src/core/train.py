import os
import pickle
import warnings
import numpy as np
import torch
import csv
# import wandb
import copy

from utils.checkpoint import *
from utils.util import *
from tqdm import tqdm
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
    "infection": [
        "sepsis", "septic", "fever", "antibiotic", "wbc", "white blood",
        "infection", "infectious", "culture", "inflammation",
    ],
    "renal_metabolic": [
        "creatinine", "bun", "urea nitrogen", "urine output", "renal",
        "kidney", "dialysis", "electrolyte", "sodium", "potassium",
        "bicarbonate", "acidosis", "alkalosis", "anion gap", "glucose",
        "lactate", "metabolic",
    ],
}

EXPERT_PROFILE_NAMES = ["cardiovascular", "respiratory", "renal_metabolic", "infection"]


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


def _set_moe_epoch(model, epoch):
    unwrapped = model.module if hasattr(model, "module") else model
    for module in unwrapped.modules():
        if hasattr(module, "set_current_epoch"):
            module.set_current_epoch(epoch)


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
    if output_path is None:
        output_path = os.path.join(args.output_dir, "router_diagnostics_test.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample_ids = metadata.get("sample_ids", [""] * len(labels))
    note_texts = metadata.get("note_texts", [[] for _ in range(len(labels))])
    write_header = not os.path.exists(output_path)
    max_chars = getattr(args, "router_diagnostics_max_text_chars", 500)
    layer_fields = []
    for layer_idx in range(len(diagnostics_list)):
        layer_fields.append(f"layer{layer_idx}_topk_expert_ids_weights")
        if semantic_first_only and layer_idx > 0:
            layer_fields.append(f"layer{layer_idx}_topk_expert_indices")
        else:
            layer_fields.append(f"layer{layer_idx}_topk_expert_profile_names")

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "true_label",
                "pred_score",
                "pred_label",
                "weak_organ_labels",
                "topk_expert_ids_weights",
                "topk_expert_profile_names",
                *layer_fields,
                "note_text_snippet",
            ],
        )
        if write_header:
            writer.writeheader()

        for i in range(len(labels)):
            sample_notes = note_texts[i] if i < len(note_texts) else []
            if isinstance(sample_notes, str):
                sample_notes = [sample_notes]
            snippet = " ".join(str(t).replace("\n", " ") for t in sample_notes)[:max_chars]
            score = float(logits[i])
            primary_gates = diagnostics_list[-1]["gates"]
            row = {
                "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                "true_label": int(labels[i]),
                "pred_score": score,
                "pred_label": int(score > 0.5),
                "weak_organ_labels": "|".join(_weak_organ_labels(sample_notes)),
                "topk_expert_ids_weights": _format_gate_row(primary_gates, i),
                "topk_expert_profile_names": _format_expert_names(primary_gates, i),
                "note_text_snippet": snippet,
            }
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


    for epoch in tqdm(range(args.num_train_epochs)):
        model.train()
        _set_moe_epoch(model, epoch)
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
                        labels=label,
                        reg_ts=reg_ts,
                        router_organ_targets=router_organ_targets,
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
                        labels=label,
                        reg_ts=reg_ts,
                        router_organ_targets=router_organ_targets,
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
                        labels=label,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        router_organ_targets=router_organ_targets,
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
                        labels=label,
                        reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        ecg_missing=ecg_missing,
                        router_organ_targets=router_organ_targets,
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

            if loss is None:
                warnings.warn("loss is None!")
                continue

            # Incorporate balance_loss if enabled and available
            if hasattr(args, "use_balance_loss") and args.use_balance_loss and balance_loss is not None:
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
                except Exception:
                    pass
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
        if balance_loss_count > 0:
            print("Train balance_loss avg", balance_loss_sum / balance_loss_count)
        if router_gate_sum is not None and router_gate_count > 0:
            usage = router_gate_sum / max(router_gate_count, 1)
            usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
            print("Train router gate mass avg", usage_str)
        if router_select_sum is not None and router_gate_count > 0:
            selection = router_select_sum / max(router_gate_count, 1)
            selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
            print("Train router selection freq avg", selection_str)
        for layer_idx, (gate_sum, select_sum, gate_count) in enumerate(
            zip(router_layer_gate_sums, router_layer_select_sums, router_layer_counts)
        ):
            if gate_sum is None or select_sum is None or gate_count <= 0:
                continue
            usage = gate_sum / gate_count
            usage_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(usage))
            selection = select_sum / gate_count
            selection_str = ", ".join(f"e{i}:{float(v):.4f}" for i, v in enumerate(selection))
            print(f"Train router layer{layer_idx} gate mass avg", usage_str)
            print(f"Train router layer{layer_idx} selection freq avg", selection_str)

        # --- Evaluate on dev set ---
        eval_vals = evaluate_irg(args, device, dev_dataloader, model, dataset=dataset)
        for k, v in eval_vals.items():
            if k == "auc_scores":
                continue
            if writer is not None:
                writer.add_scalar("dev/" + k, v, epoch + 1)

            best_eval = best_evals.get(k, 0)
            if v > best_eval:
                best_eval = v
                best_evals[k] = best_eval
            print("Current " + k, v)
            print("Best " + k, best_eval)

            # wandb logging for dev metrics and best-so-far
            # wandb.log({f"dev/{k}": v, f"dev/best_{k}": best_eval, "epoch": epoch + 1})
        
        if primary_metric in eval_vals:
            current_score = eval_vals[primary_metric]
            if current_score > best_model_score:
                best_model_score = current_score
                best_model_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())

        if writer is not None:
            writer.close()
    # --- End of training loop ---
    return best_model_state, best_model_score



def evaluate_irg(args, device, data_loader, model, mode=None, dataset="mimic"):
    model.eval()
    eval_logits = []
    eval_example = []
    eval_subjects = []
    none_count = 0
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
                    )
                elif args.modeltype == "TS_CXR_Text":
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
                    )
                elif args.modeltype == "TS_CXR_Text_ECG":
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
            if torch.isnan(logits).any():
                warnings.warn("logits is nan!")
                continue

            logits_np = logits.cpu().numpy()
            label_ids = label.cpu().numpy()
            if mode == "test":
                _write_router_diagnostics(args, metadata, label_ids, logits_np, model)
            eval_logits += logits_np.tolist()
            eval_example += label_ids.tolist()

    if none_count > 0:
        print("none_count", none_count)

    eval_vals = {}
    all_logits = np.array(eval_logits)
    all_label = np.array(eval_example)
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

    return eval_vals
