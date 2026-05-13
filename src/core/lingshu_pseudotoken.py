import torch
import torch.nn as nn


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
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


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
