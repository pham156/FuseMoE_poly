import torch
import torch.nn as nn
import torch.nn.functional as F


ORGAN_EXPERT_PROFILE_TEXT = {
    "cardiovascular": "Specialized in hemodynamics, blood pressure, heart rate, cardiac function, shock, and vasopressors.",
    "respiratory": "Specialized in ventilation, oxygen saturation, chest radiographs, respiratory failure, and blood gas analysis.",
    "renal_metabolic": "Specialized in kidney function, creatinine, urine output, acid base status, glucose, and electrolyte balance.",
    "neurological": "Specialized in mental status, Glasgow Coma Scale, sedation, delirium, seizures, and neurological assessment.",
}


def _load_lingshu_backbone(model_path, trust_remote_code=True):
    if model_path is None:
        raise ValueError(
            "--lingshu_model_path is required for the Lingshu pseudo-token model. "
            "Use a local scratch path or Hugging Face id for Lingshu."
        )

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    loaders = [AutoModelForCausalLM, AutoModel]
    last_error = None
    for loader in loaders:
        try:
            model = loader.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            return model, config
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load Lingshu backbone from {model_path}: {last_error}")


def _attach_peft_lora(model, args):
    if not args.lingshu_use_peft_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for --lingshu_use_peft_lora True. "
            "Install peft in the fusemoe environment, or pass --lingshu_use_peft_lora False."
        ) from exc

    peft_config = LoraConfig(
        r=args.lingshu_lora_r,
        lora_alpha=args.lingshu_lora_alpha,
        lora_dropout=args.lingshu_lora_dropout,
        target_modules=args.lingshu_lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, peft_config)


def _make_lora_config(args):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lingshu_lora_r,
        lora_alpha=args.lingshu_lora_alpha,
        lora_dropout=args.lingshu_lora_dropout,
        target_modules=args.lingshu_lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def _attach_peft_lora_experts(model, args, expert_names):
    if not args.lingshu_use_peft_lora:
        raise ValueError("organ_lora_moe requires --lingshu_use_peft_lora True.")
    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for --lingshu_architecture organ_lora_moe. "
            "Install peft in the fusemoe environment."
        ) from exc

    if len(expert_names) < 1:
        raise ValueError("--lingshu_moe_expert_names must contain at least one expert.")

    first_name = expert_names[0]
    model = get_peft_model(model, _make_lora_config(args), adapter_name=first_name)
    for name in expert_names[1:]:
        model.add_adapter(name, _make_lora_config(args))

    for param_name, param in model.named_parameters():
        if "lora_" in param_name:
            param.requires_grad = True
    model.set_adapter(first_name)
    return model


def _backbone_forward_hidden(backbone, inputs_embeds, attention_mask):
    outputs = backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise RuntimeError("Lingshu backbone did not return hidden states.")


class LingshuPseudoTokenModel(nn.Module):
    """Lingshu/Qwen-style frozen backbone over current preprocessed ICU features.

    This is intentionally separate from FuseMoE's normal MoE path. It reuses the
    existing dataloader outputs, projects each modality feature to the Lingshu
    hidden size as pseudo-token embeddings, and trains only projectors, LoRA
    adapters, and the task head when the backbone is frozen.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.task = args.task
        self.num_labels = args.num_labels
        self.pooling = args.lingshu_pooling
        self.max_ts_tokens = args.lingshu_max_ts_tokens

        self.backbone, self.backbone_config = _load_lingshu_backbone(
            args.lingshu_model_path,
            trust_remote_code=args.lingshu_trust_remote_code,
        )

        if args.lingshu_freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone = _attach_peft_lora(self.backbone, args)

        hidden_size = getattr(self.backbone_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone_config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer Lingshu hidden size from config.")
        self.hidden_size = hidden_size

        self.ts_proj = nn.Linear(60, hidden_size)
        self.text_proj = nn.Linear(768, hidden_size)
        self.cxr_proj = nn.Linear(1024, hidden_size)
        self.ecg_proj = nn.Linear(256, hidden_size)
        self.modality_embeddings = nn.Embedding(4, hidden_size)
        self.missing_modality_embeddings = nn.Embedding(4, hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out_layer = nn.Linear(hidden_size, args.num_labels)

        if "pheno" in self.task:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def _as_token_sequence(self, features, expected_dim, modality_name):
        if features is None:
            return None
        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.dim() != 3:
            raise ValueError(
                f"{modality_name} features must be [batch, dim] or [batch, tokens, dim], "
                f"got shape {tuple(features.shape)}."
            )
        if features.size(-1) != expected_dim:
            raise ValueError(
                f"{modality_name} feature dim mismatch: expected {expected_dim}, "
                f"got {features.size(-1)} from shape {tuple(features.shape)}."
            )
        return features

    def _missing_mask(self, missing, batch_size, device):
        if missing is None:
            return None
        missing = missing.to(device=device).bool()
        if missing.dim() > 1:
            missing = missing.view(batch_size, -1).any(dim=1)
        return missing

    def _append_tokens(self, token_chunks, mask_chunks, features, projector, modality_id, modality_name, missing=None):
        if features is None:
            return

        projected = projector(features.to(dtype=projector.weight.dtype))
        projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        modality_embedding = self.modality_embeddings.weight[modality_id].to(dtype=projected.dtype)
        projected = projected + modality_embedding

        missing = self._missing_mask(missing, projected.size(0), projected.device)
        if missing is not None and missing.any():
            missing_embedding = (
                self.missing_modality_embeddings.weight[modality_id].to(dtype=projected.dtype)
                + modality_embedding
            )
            projected = projected.clone()
            projected[missing] = missing_embedding.view(1, 1, -1)

        token_chunks.append(projected)
        mask_chunks.append(torch.ones(projected.shape[:2], dtype=torch.long, device=projected.device))

    def _build_inputs(self, reg_ts=None, text_emb=None, cxr_feats=None, ecg_feats=None, cxr_missing=None, text_missing=None, ecg_missing=None):
        token_chunks = []
        mask_chunks = []

        if reg_ts is not None:
            ts_features = self._as_token_sequence(reg_ts, 60, "time-series")
            ts_features = ts_features[:, : self.max_ts_tokens]
            self._append_tokens(token_chunks, mask_chunks, ts_features, self.ts_proj, 0, "time-series")

        text_emb = self._as_token_sequence(text_emb, 768, "text")
        cxr_feats = self._as_token_sequence(cxr_feats, 1024, "cxr")
        ecg_feats = self._as_token_sequence(ecg_feats, 256, "ecg")

        self._append_tokens(token_chunks, mask_chunks, text_emb, self.text_proj, 1, "text", missing=text_missing)
        self._append_tokens(token_chunks, mask_chunks, cxr_feats, self.cxr_proj, 2, "cxr", missing=cxr_missing)
        self._append_tokens(token_chunks, mask_chunks, ecg_feats, self.ecg_proj, 3, "ecg", missing=ecg_missing)

        if not token_chunks:
            raise ValueError("No modality features were provided to LingshuPseudoTokenModel.")

        inputs_embeds = torch.cat(token_chunks, dim=1)
        attention_mask = torch.cat(mask_chunks, dim=1)
        backbone_dtype = next(self.backbone.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=backbone_dtype)
        inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        return self.dropout(inputs_embeds), attention_mask

    def _pool(self, hidden, attention_mask):
        if self.pooling == "last":
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        x_ts=None,
        x_ts_mask=None,
        ts_tt_list=None,
        cxr_missing=None,
        text_missing=None,
        ecg_missing=None,
        input_ids_sequences=None,
        attn_mask_sequences=None,
        text_emb=None,
        note_time_list=None,
        note_time_mask_list=None,
        labels=None,
        reg_ts=None,
        cxr_feats=None,
        cxr_time=None,
        cxr_time_mask=None,
        ecg_feats=None,
        ecg_time=None,
        ecg_time_mask=None,
        router_organ_targets=None,
        **unused_kwargs,
    ):
        inputs_embeds, attention_mask = self._build_inputs(
            reg_ts=reg_ts,
            text_emb=text_emb,
            cxr_feats=cxr_feats,
            ecg_feats=ecg_feats,
            cxr_missing=cxr_missing,
            text_missing=text_missing,
            ecg_missing=ecg_missing,
        )
        hidden = _backbone_forward_hidden(self.backbone, inputs_embeds, attention_mask)
        pooled = self._pool(hidden, attention_mask)
        logits = self.out_layer(pooled)

        if labels is None:
            if "pheno" in self.task:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=-1)[:, 1] if self.num_labels == 2 else logits

        if "pheno" in self.task:
            return self.loss_fct(logits, labels.float())
        return self.loss_fct(logits, labels)


class LingshuOrganLoraMoEModel(LingshuPseudoTokenModel):
    """Mentor-style prototype: organ LoRA adapters are the experts.

    The existing preprocessed FuseMoE features are still converted into pseudo
    tokens, but routing now selects among named organ-system LoRA adapters on a
    shared frozen Lingshu backbone. This keeps the original FuseMoE path untouched
    and makes the Lingshu variant opt-in via --lingshu_architecture organ_lora_moe.
    """

    def __init__(self, args, device):
        nn.Module.__init__(self)
        self.args = args
        self.device = device
        self.task = args.task
        self.num_labels = args.num_labels
        self.pooling = args.lingshu_pooling
        self.max_ts_tokens = args.lingshu_max_ts_tokens
        self.expert_names = list(args.lingshu_moe_expert_names)
        self.num_experts = len(self.expert_names)
        self.top_k = min(args.lingshu_moe_top_k, self.num_experts)
        self.router_type = args.lingshu_moe_router
        self.temperature = max(float(args.lingshu_moe_temperature), 1e-6)

        self.backbone, self.backbone_config = _load_lingshu_backbone(
            args.lingshu_model_path,
            trust_remote_code=args.lingshu_trust_remote_code,
        )

        if args.lingshu_freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = getattr(self.backbone_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone_config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer Lingshu hidden size from config.")
        self.hidden_size = hidden_size

        self.backbone = _attach_peft_lora_experts(self.backbone, args, self.expert_names)

        self.ts_proj = nn.Linear(60, hidden_size)
        self.text_proj = nn.Linear(768, hidden_size)
        self.cxr_proj = nn.Linear(1024, hidden_size)
        self.ecg_proj = nn.Linear(256, hidden_size)
        self.modality_embeddings = nn.Embedding(4, hidden_size)
        self.missing_modality_embeddings = nn.Embedding(4, hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out_layer = nn.Linear(hidden_size, args.num_labels)
        self.router = nn.Linear(hidden_size, self.num_experts)
        self.semantic_router_proj = nn.Linear(hidden_size, hidden_size)
        self.register_buffer(
            "expert_profile_embeddings",
            self._build_profile_embeddings(args, hidden_size),
        )
        self.last_router_gates = None
        self.last_balance_loss = None

        if "pheno" in self.task:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def _build_profile_embeddings(self, args, hidden_size):
        profiles = []
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.lingshu_model_path,
                trust_remote_code=args.lingshu_trust_remote_code,
            )
            embeddings = self.backbone.get_input_embeddings().weight.detach().float().cpu()
            for name in self.expert_names:
                text = ORGAN_EXPERT_PROFILE_TEXT.get(name, name.replace("_", " "))
                encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                token_ids = encoded["input_ids"].view(-1)
                token_ids = token_ids[(token_ids >= 0) & (token_ids < embeddings.size(0))]
                if token_ids.numel() == 0:
                    raise ValueError(f"No valid profile tokens for expert {name}")
                profiles.append(embeddings[token_ids].mean(dim=0))
            return torch.stack(profiles, dim=0)
        except Exception:
            generator = torch.Generator()
            generator.manual_seed(17)
            return torch.randn(self.num_experts, hidden_size, generator=generator) * 0.02

    def _route(self, inputs_embeds, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
        pooled_input = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_input = pooled_input.float()

        if self.router_type == "linear":
            logits = self.router(pooled_input)
        else:
            projected = F.normalize(self.semantic_router_proj(pooled_input), dim=-1)
            profiles = F.normalize(
                self.expert_profile_embeddings.to(device=pooled_input.device, dtype=pooled_input.dtype),
                dim=-1,
            )
            logits = projected @ profiles.t()

        logits = logits / self.temperature
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_gates = torch.softmax(top_values, dim=-1)
        top_gates = torch.nan_to_num(top_gates, nan=1.0 / self.top_k, posinf=1.0 / self.top_k, neginf=0.0)
        gates = torch.zeros_like(logits, dtype=top_gates.dtype)
        gates.scatter_(1, top_indices, top_gates)
        self.last_router_gates = gates.detach()
        return gates

    def _balance_loss(self, gates):
        importance = gates.float().sum(dim=0)
        if importance.numel() <= 1:
            return importance.new_tensor(0.0)
        mean = importance.mean().clamp_min(1e-6)
        loss = ((importance - mean) ** 2).mean() / (mean ** 2)
        self.last_balance_loss = loss.detach()
        return loss

    def _expert_hidden(self, adapter_name, inputs_embeds, attention_mask):
        self.backbone.set_adapter(adapter_name)
        hidden = _backbone_forward_hidden(self.backbone, inputs_embeds, attention_mask)
        return self._pool(hidden, attention_mask)

    def forward(
        self,
        x_ts=None,
        x_ts_mask=None,
        ts_tt_list=None,
        cxr_missing=None,
        text_missing=None,
        ecg_missing=None,
        input_ids_sequences=None,
        attn_mask_sequences=None,
        text_emb=None,
        note_time_list=None,
        note_time_mask_list=None,
        labels=None,
        reg_ts=None,
        cxr_feats=None,
        cxr_time=None,
        cxr_time_mask=None,
        ecg_feats=None,
        ecg_time=None,
        ecg_time_mask=None,
        router_organ_targets=None,
        **unused_kwargs,
    ):
        inputs_embeds, attention_mask = self._build_inputs(
            reg_ts=reg_ts,
            text_emb=text_emb,
            cxr_feats=cxr_feats,
            ecg_feats=ecg_feats,
            cxr_missing=cxr_missing,
            text_missing=text_missing,
            ecg_missing=ecg_missing,
        )
        gates = self._route(inputs_embeds, attention_mask)
        balance_loss = self._balance_loss(gates)

        pooled = None
        selected_experts = torch.nonzero(gates.sum(dim=0) > 0, as_tuple=False).flatten().tolist()
        for expert_idx in selected_experts:
            expert_hidden = self._expert_hidden(self.expert_names[expert_idx], inputs_embeds, attention_mask)
            weighted_hidden = expert_hidden * gates[:, expert_idx].to(dtype=expert_hidden.dtype).unsqueeze(-1)
            pooled = weighted_hidden if pooled is None else pooled + weighted_hidden
        if pooled is None:
            raise RuntimeError("Lingshu organ-LoRA-MoE router selected no experts.")
        logits = self.out_layer(pooled)

        if labels is None:
            if "pheno" in self.task:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=-1)[:, 1] if self.num_labels == 2 else logits

        if "pheno" in self.task:
            task_loss = self.loss_fct(logits, labels.float())
        else:
            task_loss = self.loss_fct(logits, labels)
        return task_loss, balance_loss
