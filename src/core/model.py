import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from core.module import *
from core.interp import *
from scripts.experiments.week31.multimodal_moe_campaign.components import (
    InteractionFeatureExtractor,
    MissingnessEncoder,
)
import copy
import pdb


class LightweightStateSpaceBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        state = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(seq_len):
            token = x[:, step, :]
            delta = torch.sigmoid(self.delta_proj(token))
            candidate = torch.tanh(self.in_proj(token))
            state = (1.0 - delta) * state + delta * candidate
            outputs.append(state)
        stacked = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(self.out_proj(stacked)))


class MaestroTokenBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_modalities,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        missing_init_std=0.02,
        use_cross_modal_encoder=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.use_cross_modal_encoder = use_cross_modal_encoder
        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        self.observed_embeddings = nn.Embedding(2, embed_dim)
        self.missing_tokens = nn.Parameter(torch.empty(num_modalities, embed_dim))
        nn.init.normal_(self.missing_tokens, std=missing_init_std)
        if self.use_cross_modal_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=max(1, min(num_heads, embed_dim)),
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        else:
            self.encoder = None
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pooled_tokens, missing_mask, modality_ids):
        batch_size, num_tokens, dim = pooled_tokens.shape
        ids = modality_ids.to(device=pooled_tokens.device, dtype=torch.long).clamp(min=0, max=self.num_modalities - 1)
        observed = (~missing_mask.bool()).long()
        missing_replacements = self.missing_tokens[ids].unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.where(missing_mask.bool().unsqueeze(-1), missing_replacements, pooled_tokens)
        tokens = tokens + self.modality_embeddings(ids).unsqueeze(0) + self.observed_embeddings(observed)
        encoded = self.encoder(tokens) if self.encoder is not None else tokens
        return self.norm(encoded)


def default_router_instruction(task, modeltype):
    modalities = []
    if "TS" in modeltype:
        modalities.append("time-series clinical measurements for acute physiological trends")
    if "CXR" in modeltype:
        modalities.append("chest X-ray features for cardiopulmonary imaging evidence")
    if "Text" in modeltype:
        modalities.append("clinical notes for diagnoses, comorbidities, and clinical context")
    if "ECG" in modeltype:
        modalities.append("ECG features for abnormal cardiac patterns")

    modality_text = "; ".join(modalities) if modalities else "available clinical modalities"
    if "ihm" in task:
        target = "in-hospital mortality"
        evidence = "acute physiological instability, severe cardiopulmonary findings, comorbidities, and abnormal cardiac patterns"
    elif "los" in task:
        target = "prolonged length of stay"
        evidence = "disease severity, complications, delayed recovery, abnormal imaging findings, unstable physiological trends, and comorbidities"
    elif "pheno" in task:
        target = "clinical phenotypes"
        evidence = "diagnosis-specific evidence, abnormal measurements, imaging findings, and relevant clinical context"
    else:
        target = "the clinical prediction task"
        evidence = "task-relevant evidence across the available modalities"

    return (
        f"Predict {target}. Use {modality_text}. "
        f"Prioritize {evidence} when selecting useful experts."
    )


def default_expert_profiles(profile_set):
    if profile_set == "icu_organ_system":
        return [
            (
                "Cardiovascular and hemodynamic expert. Focuses on heart and circulation "
                "problems including blood pressure, heart rate, shock, vasopressors, "
                "poor perfusion, cardiac arrest, and heart failure."
            ),
            (
                "Respiratory and pulmonary expert. Focuses on breathing and lung problems "
                "including oxygen saturation, ventilation, respiratory rate, pneumonia, "
                "pulmonary edema, respiratory failure, and chest X-ray lung findings."
            ),
            (
                "Renal and metabolic expert. Focuses on kidney function and metabolic "
                "instability including creatinine, blood urea nitrogen, urine output, "
                "electrolyte imbalance, acidosis, alkalosis, glucose, and lactate."
            ),
            (
                "Neurological expert. Focuses on Glasgow Coma Scale, consciousness, "
                "sedation level, delirium, neurological assessments, seizure, stroke, "
                "and acute changes in mental status."
            ),
        ]
    if profile_set == "random_clinical_control":
        return [
            (
                "Infectious disease documentation specialist. Focuses on antimicrobial exposure, "
                "culture follow-up, source control, isolation precautions, and fever workups."
            ),
            (
                "Nutrition and gastrointestinal specialist. Focuses on tube feeds, bowel regimen, "
                "abdominal symptoms, liver function, nausea, emesis, and stool output."
            ),
            (
                "Procedural and device management specialist. Focuses on line placement, access issues, "
                "catheter care, wound management, dressing changes, and bedside procedures."
            ),
            (
                "Disposition and care-planning specialist. Focuses on rehabilitation planning, social history, "
                "discharge barriers, family communication, and placement coordination."
            ),
        ]
    raise ValueError(f"Unknown semantic_profile_set: {profile_set}")


def encode_expert_profiles(args, biobert, tokenizer, device):
    if biobert is None or tokenizer is None:
        raise ValueError("--use_semantic_expert_profiles requires a text encoder/tokenizer, so use a modeltype that includes Text.")

    profiles = default_expert_profiles(args.semantic_profile_set)
    encoded = tokenizer(
        profiles,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    was_training = biobert.training
    biobert.eval()
    with torch.no_grad():
        outputs = biobert(**encoded)
        profile_embeddings = outputs[0][:, 0, :].detach()
    if getattr(args, "semantic_profile_shuffle_assignments", False):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(args, "seed", 0)) + 1879)
        permutation = torch.randperm(profile_embeddings.size(0), generator=generator)
        profile_embeddings = profile_embeddings[permutation.to(profile_embeddings.device)]
    if was_training:
        biobert.train()
    return profile_embeddings


def encode_router_instruction(args, biobert, tokenizer, device, modeltype):
    if biobert is None or tokenizer is None:
        raise ValueError("--use_instruction_router requires a text encoder/tokenizer, so use a modeltype that includes Text.")

    instruction = args.router_instruction or default_router_instruction(args.task, modeltype)
    encoded = tokenizer(
        instruction,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding="max_length" if args.pad_to_max_length else False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    was_training = biobert.training
    biobert.eval()
    with torch.no_grad():
        outputs = biobert(**encoded)
        instruction_embedding = outputs[0][:, 0, :].detach()
    if was_training:
        biobert.train()
    return instruction_embedding.squeeze(0)


class BertForRepresentation(nn.Module):
    """
    This class represents a BERT model for text representation.

    Args:
        args (object): The arguments for the model.
        BioBert (object): The BioBERT model.

    Attributes:
        bert (object): The BioBERT model.
        dropout (object): The dropout layer.
        model_name (str): The name of the model.
    """
    
    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        """
        Forward pass of the model.

        Args:
            input_ids_sequence (List[Tensor]): List of input token IDs for each sequence.
            attention_mask_sequence (List[Tensor]): List of attention masks for each sequence.
            sent_idx_list (List[int], optional): List of sentence indices. Defaults to None.
            doc_idx_list (List[int], optional): List of document indices. Defaults to None.

        Returns:
            Tensor: Text embeddings for each sequence.
        """
        txt_arr = []

        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr


class TextModel(nn.Module):
    def __init__(self,args,device,orig_d_txt=768,Biobert=None):
        """
        Construct a TextModel.
        """
        super(TextModel, self).__init__()

        self.device=device
        self.task=args.task
        # self.agg_type=args.agg_type
        self.out_dropout = args.dropout
        self.orig_d_txt=orig_d_txt
        self.d_txt= args.embed_dim
        self.bertrep=BertForRepresentation(args,Biobert)

        self.proj_txt =nn.Linear(self.orig_d_txt, self.d_txt)

        output_dim = args.num_labels
        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if 'ihm' in self.task:
            self.loss_fct1=CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")


    def forward(self, input_ids_sequences,
                attn_mask_sequences,labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)
        x_txt=torch.mean(x_txt,dim=1)
        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task:
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,orig_d_txt=None,ts_seq_num=None,text_seq_num=None, Biobert=None, tokenizer=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        self.num_modalities = args.num_modalities
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)
        self.use_maestro_tokens = bool(getattr(args, "use_maestro_tokens", False))
        self.maestro_sparse_topk = int(getattr(args, "maestro_sparse_topk", 2))
        self.maestro_residual_weight = float(getattr(args, "maestro_residual_weight", 1.0))
        self.maestro_missing_token_mode = str(getattr(args, "maestro_missing_token_mode", "learned"))
        self.maestro_use_cross_modal_encoder = bool(getattr(args, "maestro_use_cross_modal_encoder", True))
        self.last_maestro_token_details = {}
        if self.use_maestro_tokens:
            self.maestro_token_block = MaestroTokenBlock(
                embed_dim=args.embed_dim,
                num_modalities=args.num_modalities,
                num_layers=int(getattr(args, "maestro_token_layers", 1)),
                num_heads=int(getattr(args, "maestro_token_heads", 4)),
                dropout=args.dropout,
                missing_init_std=float(getattr(args, "maestro_missing_init_std", 0.02)),
                use_cross_modal_encoder=self.maestro_use_cross_modal_encoder,
            )
        else:
            self.maestro_token_block = None
        self.use_missing_modality_proxies = getattr(args, "use_missing_modality_proxies", False)
        if self.use_missing_modality_proxies:
            self.missing_modality_proxy_tokens = nn.Parameter(
                torch.empty(args.num_modalities, args.tt_max, args.embed_dim)
            )
            nn.init.normal_(
                self.missing_modality_proxy_tokens,
                std=float(getattr(args, "missing_modality_proxy_init_std", 0.02)),
            )
        else:
            self.missing_modality_proxy_tokens = None
        self.use_cross_modal_missing_proxies = getattr(args, "use_cross_modal_missing_proxies", False)
        if self.use_cross_modal_missing_proxies:
            proxy_hidden = int(getattr(args, "cross_modal_proxy_hidden", 256))
            proxy_dropout = float(getattr(args, "cross_modal_proxy_dropout", args.dropout))
            self.cross_modal_missing_proxy_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(args.embed_dim, proxy_hidden),
                    nn.ReLU(),
                    nn.Dropout(proxy_dropout),
                    nn.Linear(proxy_hidden, args.tt_max * args.embed_dim),
                )
                for name in ("text", "cxr", "ecg")
            })
        else:
            self.cross_modal_missing_proxy_heads = nn.ModuleDict()
        self.use_learned_missing_embeddings = getattr(args, "use_learned_missing_embeddings", False)
        if self.use_learned_missing_embeddings:
            self.missing_modality_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)
            nn.init.zeros_(self.missing_modality_embeddings.weight)
        else:
            self.missing_modality_embeddings = None
        self.use_instruction_router = args.use_instruction_router
        if self.use_instruction_router:
            instruction_embedding = encode_router_instruction(args, Biobert, tokenizer, device, self.modeltype)
            self.register_buffer("router_instruction_embedding", instruction_embedding)
        else:
            self.router_instruction_embedding = None

        if args.use_semantic_expert_profiles:
            if getattr(args, "semantic_profile_embedding_source", "biolongformer") == "random":
                profile_dim = args.semantic_project_dim or getattr(Biobert.config, "hidden_size", args.embed_dim) if Biobert is not None else args.embed_dim
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.seed) + 7919)
                profile_embeddings = torch.randn(4, profile_dim, generator=generator, dtype=torch.float32).to(device)
                if getattr(args, "semantic_profile_shuffle_assignments", False):
                    permutation = torch.randperm(profile_embeddings.size(0), generator=generator)
                    profile_embeddings = profile_embeddings[permutation.to(profile_embeddings.device)]
            else:
                profile_embeddings = encode_expert_profiles(args, Biobert, tokenizer, device)
            self.register_buffer("semantic_expert_profile_embeddings", profile_embeddings)
            args.semantic_profile_embeddings = profile_embeddings
        else:
            self.semantic_expert_profile_embeddings = None
            args.semantic_profile_embeddings = None

        self.semantic_profile_source = args.semantic_profile_source
        self.semantic_profile_note_pooling = args.semantic_profile_note_pooling
        self.task_to_id = {"ihm": 0, "los": 1, "pheno": 2}
        self.task_name = "pheno" if "pheno" in self.task else ("los" if "los" in self.task else "ihm")
        self.task_id_value = self.task_to_id[self.task_name]
        self.multitask_shared_moe_trunk = bool(getattr(args, "multitask_shared_moe_trunk", False))
        self.multitask_head_output_dims = {"ihm": 2, "los": 2, "pheno": 25}

        self.use_task_condition_router = getattr(args, "use_task_condition_router", False)
        self.use_task_condition_expert_modulation = getattr(args, "use_task_condition_expert_modulation", False)
        if self.use_task_condition_router or self.use_task_condition_expert_modulation or self.multitask_shared_moe_trunk:
            self.task_router_embeddings = nn.Embedding(len(self.task_to_id), args.task_router_dim)
        else:
            self.task_router_embeddings = None

        self.use_modality_mask_condition_router = getattr(args, "use_modality_mask_condition_router", False)
        self.use_interaction_router = bool(getattr(args, "use_interaction_router", False))
        self.interaction_router_only = bool(getattr(args, "interaction_router_only", False))
        self.log_interaction_features = bool(getattr(args, "log_interaction_features", False))
        self.last_interaction_feature_output = None
        self.interaction_feature_extractor = InteractionFeatureExtractor(
            include_l2=bool(getattr(args, "interaction_include_l2", False)),
            include_dot=True,
            include_norm_ratio=bool(getattr(args, "interaction_include_norm_ratio", False)),
        )
        args.interaction_feature_dim = (
            len(self.interaction_feature_extractor(
                pooled_embeddings={
                    "text": torch.zeros(1, args.embed_dim),
                    "ts": torch.zeros(1, args.embed_dim),
                    "cxr": torch.zeros(1, args.embed_dim),
                    "ecg": torch.zeros(1, args.embed_dim),
                },
                missing_flags={
                    "text": torch.zeros(1, dtype=torch.bool),
                    "ts": torch.zeros(1, dtype=torch.bool),
                    "cxr": torch.zeros(1, dtype=torch.bool),
                    "ecg": torch.zeros(1, dtype=torch.bool),
                },
            ).feature_names)
        )
        if self.use_modality_mask_condition_router:
            self.modality_mask_router_embeddings = MissingnessEncoder(
                output_dim=args.modality_mask_router_dim,
                encoder_type=getattr(args, "missingness_encoder_type", "lookup"),
                num_modalities=4,
                hidden_dim=args.modality_mask_router_dim,
            )
        else:
            self.modality_mask_router_embeddings = None

        self.use_collaboration_branch = getattr(args, "use_collaboration_branch", False)
        self.collaboration_fusion = getattr(args, "collaboration_fusion", "fixed_average")
        self.collaboration_fixed_weight = float(getattr(args, "collaboration_fixed_weight", 0.5))
        self.log_collaboration_diagnostics = bool(getattr(args, "log_collaboration_diagnostics", False))
        self.last_collaboration_diagnostics = None
        self.use_unimodal_kd = bool(getattr(args, "use_unimodal_kd", False))
        self.unimodal_kd_modalities = [
            item.strip().lower()
            for item in str(getattr(args, "unimodal_kd_modalities", "ts,text,cxr,ecg")).split(",")
            if item.strip()
        ]
        self.unimodal_kd_weight = float(getattr(args, "unimodal_kd_weight", 1.0))
        self.unimodal_kd_temperature = float(getattr(args, "unimodal_kd_temperature", 1.0))
        self.last_unimodal_kd_loss = None
        self.last_unimodal_kd_details = {}
        self.last_consistency_loss = None
        self.last_fused_latent = None
        self.last_output_logits = None
        self.use_ts_variable_tokens = bool(getattr(args, "use_ts_variable_tokens", False))
        self.ts_variable_token_layers = int(getattr(args, "ts_variable_token_layers", 1))
        self.ts_variable_token_heads = int(getattr(args, "ts_variable_token_heads", max(1, min(args.num_heads, 4))))
        self.use_ts_patch_encoder = bool(getattr(args, "use_ts_patch_encoder", False))
        self.ts_patch_size = int(getattr(args, "ts_patch_size", 4))
        self.ts_patch_layers = int(getattr(args, "ts_patch_layers", 1))
        self.ts_patch_mask_ratio = float(getattr(args, "ts_patch_mask_ratio", 0.3))
        self.ts_patch_masking_mode = str(getattr(args, "ts_patch_masking_mode", "patch"))
        self.ts_patch_recon_weight = float(getattr(args, "ts_patch_recon_weight", 0.0))
        self.use_ts_state_space_encoder = bool(getattr(args, "use_ts_state_space_encoder", False))
        self.ts_state_space_layers = int(getattr(args, "ts_state_space_layers", 1))
        self.use_ts_shared_private = bool(getattr(args, "use_ts_shared_private", False))
        self.ts_shared_private_weight = float(getattr(args, "ts_shared_private_weight", 0.05))
        self.ts_shared_private_ortho_coef = float(getattr(args, "ts_shared_private_ortho_coef", 1.0))
        self.ts_shared_private_align_coef = float(getattr(args, "ts_shared_private_align_coef", 1.0))
        self.use_ts_confidence_fusion = bool(getattr(args, "use_ts_confidence_fusion", False))
        self.ts_confidence_hidden = int(getattr(args, "ts_confidence_hidden", 64))
        self.use_ts_cross_modal_distill = bool(getattr(args, "use_ts_cross_modal_distill", False))
        self.ts_cross_modal_distill_weight = float(getattr(args, "ts_cross_modal_distill_weight", 0.05))
        self.last_ts_patch_recon_loss = None
        self.last_ts_patch_recon_details = {}
        self.last_ts_aux_loss = None
        self.last_ts_aux_details = {}
        self.use_labelwise_fusion = bool(getattr(args, "use_labelwise_fusion", False))
        self.labelwise_fusion_source = str(getattr(args, "labelwise_fusion_source", "modalities"))
        self.labelwise_hidden_dim = int(getattr(args, "labelwise_hidden_dim", 128))
        self.labelwise_rank_loss_weight = float(getattr(args, "labelwise_rank_loss_weight", 0.0))

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time - 1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
            if self.use_ts_variable_tokens:
                self.ts_variable_value_proj = nn.Linear(1, self.d_ts)
                self.ts_variable_id_embeddings = nn.Embedding(self.orig_d_ts, self.d_ts)
                self.ts_variable_observed_embeddings = nn.Embedding(2, self.d_ts)
                self.ts_variable_time_proj = nn.Linear(args.embed_time, self.d_ts)
                self.ts_variable_query_proj = nn.Linear(args.embed_time, self.d_ts)
                self.ts_variable_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.d_ts,
                        nhead=max(1, min(self.ts_variable_token_heads, self.d_ts)),
                        dim_feedforward=max(self.d_ts * 4, args.hidden_size),
                        dropout=args.dropout,
                        batch_first=True,
                    ),
                    num_layers=self.ts_variable_token_layers,
                )
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
                if self.use_ts_state_space_encoder:
                    self.ts_state_space_input = nn.Linear(self.orig_reg_d_ts, self.d_ts)
                    self.ts_state_space_encoder = nn.ModuleList([
                        LightweightStateSpaceBlock(self.d_ts, dropout=args.dropout)
                        for _ in range(max(1, self.ts_state_space_layers))
                    ])
                else:
                    self.ts_state_space_input = None
                    self.ts_state_space_encoder = None
                if self.use_ts_patch_encoder:
                    if self.ts_patch_size <= 0:
                        raise ValueError("--ts_patch_size must be positive")
                    self.ts_patch_count = math.ceil(self.tt_max / self.ts_patch_size)
                    self.ts_patch_input_dim = self.orig_reg_d_ts * self.ts_patch_size
                    self.ts_patch_proj = nn.Linear(self.ts_patch_input_dim, self.d_ts)
                    self.ts_patch_pos_embeddings = nn.Embedding(self.ts_patch_count, self.d_ts)
                    self.ts_patch_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_ts))
                    nn.init.normal_(self.ts_patch_mask_token, std=0.02)
                    self.ts_patch_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=self.d_ts,
                            nhead=max(1, min(args.num_heads, self.d_ts)),
                            dim_feedforward=max(self.d_ts * 4, args.hidden_size),
                            dropout=args.dropout,
                            batch_first=True,
                        ),
                        num_layers=self.ts_patch_layers,
                    )
                    self.ts_patch_recon_head = nn.Linear(self.d_ts, self.ts_patch_input_dim)
                if self.use_ts_shared_private:
                    self.ts_shared_proj = nn.Linear(self.d_ts, self.d_ts)
                    self.ts_private_proj = nn.Linear(self.d_ts, self.d_ts)
                else:
                    self.ts_shared_proj = None
                    self.ts_private_proj = None
                if self.use_ts_confidence_fusion:
                    self.ts_confidence_head = nn.Sequential(
                        nn.Linear(self.d_ts + 2, self.ts_confidence_hidden),
                        nn.ReLU(),
                        nn.Dropout(args.dropout),
                        nn.Linear(self.ts_confidence_hidden, 1),
                    )
                else:
                    self.ts_confidence_head = None
                if self.use_ts_cross_modal_distill:
                    self.ts_distill_student_proj = nn.Linear(self.d_ts, self.d_ts)
                    self.ts_distill_teacher_proj = nn.Linear(self.d_ts, self.d_ts)
                else:
                    self.ts_distill_student_proj = None
                    self.ts_distill_teacher_proj = None

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        # if self.modeltype == "TS_CXR":
        if "CXR" in self.modeltype:
            self.orig_d_cxr = 1024
            self.d_cxr = args.embed_dim
            self.cxr_seq_num = 5

            if self.irregular_learn_emb_cxr:
                self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
            else:
                self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.orig_d_ecg = 256
            self.d_ecg = args.embed_dim
            self.ecg_seq_num = 5

            if self.irregular_learn_emb_ecg:
                self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
            else:
                self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels if not self.multitask_shared_moe_trunk else self.multitask_head_output_dims[self.task_name]
        # if self.modeltype=="TS_Text":
        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.cross_layers)
            dim = 0
            if "TS" in self.modeltype:
                dim += self.d_ts
            if "Text" in self.modeltype:
                dim += self.d_txt
            if "CXR" in self.modeltype:
                dim += self.d_cxr
            if "ECG" in self.modeltype:
                dim += self.d_ecg            

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)
        else:
            # baseline fusion methods
            self.d_txt = args.embed_dim
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
            self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

            if self.cross_method=="MulT":
                self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
                self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
                self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
            elif self.cross_method=="MAGGate":
                self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            elif self.cross_method=="Outer":
                self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            else:
                self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)

        if self.multitask_shared_moe_trunk:
            trunk_dim = self.out_layer.in_features
            self.multitask_heads = nn.ModuleDict({
                task_name: nn.Linear(trunk_dim, out_dim)
                for task_name, out_dim in self.multitask_head_output_dims.items()
            })
            self.multitask_loss_fns = {
                "ihm": nn.CrossEntropyLoss(),
                "los": nn.CrossEntropyLoss(),
                "pheno": nn.BCEWithLogitsLoss(),
            }
        else:
            self.multitask_heads = None
            self.multitask_loss_fns = None

        self.available_modality_names = []
        if "TS" in self.modeltype:
            self.available_modality_names.append("ts")
        if "Text" in self.modeltype:
            self.available_modality_names.append("text")
        if "CXR" in self.modeltype:
            self.available_modality_names.append("cxr")
        if "ECG" in self.modeltype:
            self.available_modality_names.append("ecg")

        self.unimodal_kd_weight_map = self._parse_modality_weight_map(
            getattr(args, "unimodal_kd_weights", ""),
            self.unimodal_kd_modalities,
        )
        if self.use_unimodal_kd:
            self.unimodal_heads = nn.ModuleDict({
                modality: nn.Sequential(
                    nn.Linear(args.embed_dim, args.embed_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.embed_dim, output_dim),
                )
                for modality in self.unimodal_kd_modalities
                if modality in self.available_modality_names
            })
        else:
            self.unimodal_heads = nn.ModuleDict()

        if self.use_labelwise_fusion:
            self.labelwise_source_proj = nn.Linear(args.embed_dim, self.labelwise_hidden_dim)
            self.labelwise_fused_proj = nn.Linear(self.out_layer.in_features, self.labelwise_hidden_dim)
            if self.multitask_shared_moe_trunk:
                self.labelwise_query = nn.ParameterDict({
                    task_name: nn.Parameter(torch.randn(out_dim, self.labelwise_hidden_dim) * 0.02)
                    for task_name, out_dim in self.multitask_head_output_dims.items()
                })
                self.labelwise_output = nn.ParameterDict({
                    task_name: nn.Parameter(torch.randn(out_dim, self.labelwise_hidden_dim) * 0.02)
                    for task_name, out_dim in self.multitask_head_output_dims.items()
                })
                self.labelwise_bias = nn.ParameterDict({
                    task_name: nn.Parameter(torch.zeros(out_dim))
                    for task_name, out_dim in self.multitask_head_output_dims.items()
                })
            else:
                self.labelwise_query = nn.Parameter(torch.randn(output_dim, self.labelwise_hidden_dim) * 0.02)
                self.labelwise_output = nn.Parameter(torch.randn(output_dim, self.labelwise_hidden_dim) * 0.02)
                self.labelwise_bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.labelwise_source_proj = None
            self.labelwise_fused_proj = None
            self.labelwise_query = None
            self.labelwise_output = None
            self.labelwise_bias = None

        # TODO: add baseline fusion methods for TS_CXR
        # if self.modeltype == "TS_CXR":
        #     if self.cross_method in ["self_cross", "moe", "moe_cross"]:
        #         self.trans_self_cross_ts_txt=self.get_cross_network(args, layers=args.cross_layers)
        #         self.proj1 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.proj2 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.out_layer = nn.Linear(self.d_ts+self.d_cxr, output_dim)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

        if self.use_collaboration_branch:
            collab_hidden = int(getattr(args, "collaboration_hidden_size", args.hidden_size))
            collab_input_dim = self.out_layer.in_features
            if getattr(args, "collaboration_branch_type", "mlp") == "transformer":
                self.collaboration_transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=args.embed_dim,
                        nhead=max(1, min(args.num_heads, args.embed_dim)),
                        dim_feedforward=collab_hidden,
                        dropout=args.dropout,
                        batch_first=True,
                    ),
                    num_layers=1,
                )
                self.collaboration_proj1 = None
                self.collaboration_proj2 = None
                self.collaboration_out = nn.Linear(args.embed_dim * self.num_modalities, output_dim)
            else:
                self.collaboration_transformer = None
                self.collaboration_proj1 = nn.Linear(collab_input_dim, collab_hidden)
                self.collaboration_proj2 = nn.Linear(collab_hidden, collab_input_dim)
                self.collaboration_out = nn.Linear(collab_input_dim, output_dim)
            if self.collaboration_fusion == "learnable_scalar":
                init_logit = math.log(
                    max(self.collaboration_fixed_weight, 1e-6)
                    / max(1.0 - self.collaboration_fixed_weight, 1e-6)
                )
                self.collaboration_fusion_logit = nn.Parameter(torch.tensor(float(init_logit)))
            else:
                self.collaboration_fusion_logit = None
        else:
            self.collaboration_transformer = None
            self.collaboration_proj1 = None
            self.collaboration_proj2 = None
            self.collaboration_out = None
            self.collaboration_fusion_logit = None

        self.use_missing_modality_recon = getattr(args, "use_missing_modality_recon", False)
        self.missing_modality_recon_coef = getattr(args, "missing_modality_recon_coef", 0.1)
        self.missing_modality_recon_targets = [
            target.strip().lower()
            for target in getattr(args, "missing_modality_recon_targets", "cxr,ecg").split(",")
            if target.strip()
        ]
        self.last_missing_recon_loss = None
        self.last_missing_recon_details = {}
        if self.use_missing_modality_recon:
            recon_hidden = getattr(args, "missing_modality_recon_hidden", 256)
            valid_recon_targets = {"ts", "text", "txt", "cxr", "ecg"}
            unknown_targets = set(self.missing_modality_recon_targets) - valid_recon_targets
            if unknown_targets:
                raise ValueError(f"Unknown missing reconstruction targets: {sorted(unknown_targets)}")
            self.missing_recon_heads = nn.ModuleDict()
            for target in self.missing_modality_recon_targets:
                canonical_target = "text" if target == "txt" else target
                self.missing_recon_heads[canonical_target] = nn.Sequential(
                    nn.Linear(args.embed_dim, recon_hidden),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(recon_hidden, args.embed_dim),
                )
        else:
            self.missing_recon_heads = nn.ModuleDict()

    def _encode_ts_variable_tokens(self, x_ts, x_ts_mask, ts_tt_list):
        batch_size, seq_len, num_vars = x_ts.shape
        value_tokens = self.ts_variable_value_proj(x_ts.unsqueeze(-1))
        var_ids = torch.arange(num_vars, device=x_ts.device, dtype=torch.long).view(1, 1, num_vars)
        var_embed = self.ts_variable_id_embeddings(var_ids).expand(batch_size, seq_len, num_vars, -1)
        observed_embed = self.ts_variable_observed_embeddings(x_ts_mask.long().clamp(min=0, max=1))
        time_embed = self.learn_time_embedding(ts_tt_list).to(x_ts.device)
        time_embed = self.ts_variable_time_proj(time_embed).unsqueeze(2).expand(-1, -1, num_vars, -1)
        tokens = value_tokens + var_embed + observed_embed + time_embed
        tokens = tokens.reshape(batch_size, seq_len * num_vars, self.d_ts)

        time_valid = (x_ts_mask.sum(dim=-1) > 0)
        token_valid = time_valid.unsqueeze(-1).expand(-1, -1, num_vars).reshape(batch_size, seq_len * num_vars)
        encoded = self.ts_variable_encoder(tokens, src_key_padding_mask=~token_valid)

        query_time = self.learn_time_embedding(self.time_query.unsqueeze(0).expand(batch_size, -1)).to(x_ts.device)
        query = self.ts_variable_query_proj(query_time)
        scores = torch.matmul(query, encoded.transpose(1, 2)) / math.sqrt(self.d_ts)
        scores = scores.masked_fill(~token_valid.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, encoded)
        return pooled.transpose(0, 1).contiguous()

    def _encode_ts_patches(self, reg_ts):
        batch_size, total_steps, feat_dim = reg_ts.shape
        pad_steps = self.ts_patch_count * self.ts_patch_size - total_steps
        if pad_steps > 0:
            reg_ts = F.pad(reg_ts, (0, 0, 0, pad_steps))
        patches = reg_ts.reshape(batch_size, self.ts_patch_count, self.ts_patch_size * feat_dim)
        patch_inputs = patches
        feature_mask_ratio = 0.0

        feature_mask = None
        if self.training and self.ts_patch_recon_weight > 0.0 and self.ts_patch_masking_mode == "feature":
            feature_mask = torch.rand_like(patches) < self.ts_patch_mask_ratio
            feature_mask_ratio = float(feature_mask.float().mean().detach().cpu().item())
            if feature_mask.any():
                patch_inputs = patches.masked_fill(feature_mask, 0.0)

        patch_embeddings = self.ts_patch_proj(patch_inputs)
        patch_positions = torch.arange(self.ts_patch_count, device=reg_ts.device, dtype=torch.long).unsqueeze(0)
        patch_embeddings = patch_embeddings + self.ts_patch_pos_embeddings(patch_positions)

        recon_loss = None
        recon_details = {}
        if self.training and self.ts_patch_recon_weight > 0.0:
            if self.ts_patch_masking_mode == "feature":
                encoded = self.ts_patch_encoder(patch_embeddings)
                recon = self.ts_patch_recon_head(encoded)
                if feature_mask is not None and feature_mask.any():
                    recon_loss = F.mse_loss(recon[feature_mask], patches[feature_mask])
                    recon_details = {
                        "ts_patch_recon_loss": float(recon_loss.detach().cpu().item()),
                        "ts_patch_recon_mask_ratio": feature_mask_ratio,
                        "ts_patch_masking_mode": self.ts_patch_masking_mode,
                    }
            else:
                mask = torch.rand(batch_size, self.ts_patch_count, device=reg_ts.device) < self.ts_patch_mask_ratio
                if mask.any():
                    masked_embeddings = patch_embeddings.clone()
                    masked_embeddings[mask] = self.ts_patch_mask_token.to(device=reg_ts.device, dtype=patch_embeddings.dtype)
                    encoded = self.ts_patch_encoder(masked_embeddings)
                    recon = self.ts_patch_recon_head(encoded[mask])
                    target = patches[mask]
                    recon_loss = F.mse_loss(recon, target)
                    recon_details = {
                        "ts_patch_recon_loss": float(recon_loss.detach().cpu().item()),
                        "ts_patch_recon_mask_ratio": float(mask.float().mean().detach().cpu().item()),
                        "ts_patch_masking_mode": self.ts_patch_masking_mode,
                    }
                else:
                    encoded = self.ts_patch_encoder(patch_embeddings)
        else:
            encoded = self.ts_patch_encoder(patch_embeddings)

        expanded = encoded.unsqueeze(2).expand(-1, -1, self.ts_patch_size, -1).reshape(batch_size, self.ts_patch_count * self.ts_patch_size, self.d_ts)
        expanded = expanded[:, :total_steps, :]
        return expanded.transpose(0, 1).contiguous(), recon_loss, recon_details

    def _encode_ts_state_space(self, reg_ts):
        seq = self.ts_state_space_input(reg_ts)
        for layer in self.ts_state_space_encoder:
            seq = layer(seq)
        return seq.transpose(0, 1).contiguous()

    def _compute_ts_shared_private_loss(self, proj_x_ts, modality_embeddings, missing_masks):
        if not self.use_ts_shared_private or proj_x_ts is None:
            return proj_x_ts, None, {}

        seq = proj_x_ts.transpose(0, 1)
        shared = self.ts_shared_proj(seq)
        private = self.ts_private_proj(seq)
        fused = 0.5 * (shared + private)

        shared_pool = shared.mean(dim=1)
        private_pool = private.mean(dim=1)
        ortho = (F.cosine_similarity(shared_pool, private_pool, dim=-1) ** 2).mean()

        align = shared_pool.new_zeros(())
        teacher_terms = []
        batch_size = shared_pool.size(0)
        device = shared_pool.device
        for name in ("text", "cxr", "ecg"):
            value = modality_embeddings.get(name)
            if value is None:
                continue
            observed = self._observed_mask(missing_masks.get(name), batch_size, device)
            pooled = self._pool_modality_embedding(value)
            if observed.any():
                teacher_terms.append((pooled, observed))
        if teacher_terms:
            teacher = shared_pool.new_zeros(shared_pool.shape)
            weight = shared_pool.new_zeros((batch_size, 1))
            for pooled, observed in teacher_terms:
                obs = observed.float().unsqueeze(1)
                teacher = teacher + pooled * obs
                weight = weight + obs
            valid = weight.squeeze(1) > 0
            if valid.any():
                teacher = teacher / weight.clamp_min(1.0)
                align = F.mse_loss(shared_pool[valid], teacher[valid].detach())

        total = self.ts_shared_private_weight * (
            self.ts_shared_private_ortho_coef * ortho
            + self.ts_shared_private_align_coef * align
        )
        details = {
            "ts_shared_private_loss": float(total.detach().cpu().item()),
            "ts_shared_private_ortho": float(ortho.detach().cpu().item()),
            "ts_shared_private_align": float(align.detach().cpu().item()),
        }
        return fused.transpose(0, 1).contiguous(), total, details

    def _apply_ts_confidence_fusion(self, proj_x_ts, x_ts_mask):
        if not self.use_ts_confidence_fusion or proj_x_ts is None:
            return proj_x_ts, {}
        pooled = self._pool_modality_embedding(proj_x_ts)
        observed_ratio = x_ts_mask.float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        active_timesteps = (x_ts_mask.sum(dim=-1) > 0).float().mean(dim=1, keepdim=True)
        features = torch.cat([pooled, observed_ratio, active_timesteps], dim=1)
        confidence = torch.sigmoid(self.ts_confidence_head(features))
        scaled = proj_x_ts * confidence.unsqueeze(0)
        details = {
            "ts_confidence_mean": float(confidence.mean().detach().cpu().item()),
            "ts_confidence_min": float(confidence.min().detach().cpu().item()),
            "ts_confidence_max": float(confidence.max().detach().cpu().item()),
        }
        return scaled, details

    def _compute_ts_cross_modal_distill_loss(self, proj_x_ts, modality_embeddings, missing_masks):
        if not self.use_ts_cross_modal_distill or proj_x_ts is None:
            return None, {}
        student = self.ts_distill_student_proj(self._pool_modality_embedding(proj_x_ts))
        batch_size = student.size(0)
        device = student.device
        teacher = student.new_zeros(student.shape)
        weight = student.new_zeros((batch_size, 1))
        for name in ("text", "cxr", "ecg"):
            value = modality_embeddings.get(name)
            if value is None:
                continue
            observed = self._observed_mask(missing_masks.get(name), batch_size, device)
            pooled = self.ts_distill_teacher_proj(self._pool_modality_embedding(value))
            obs = observed.float().unsqueeze(1)
            teacher = teacher + pooled * obs
            weight = weight + obs
        valid = weight.squeeze(1) > 0
        if not valid.any():
            return None, {}
        teacher = teacher / weight.clamp_min(1.0)
        loss = self.ts_cross_modal_distill_weight * F.mse_loss(student[valid], teacher[valid].detach())
        details = {
            "ts_cross_modal_distill_loss": float(loss.detach().cpu().item()),
            "ts_cross_modal_teacher_count": float(weight[valid].mean().detach().cpu().item()),
        }
        return loss, details

    def _parse_modality_weight_map(self, raw_value, default_modalities):
        weights = {modality: 1.0 for modality in default_modalities}
        if not raw_value:
            return weights
        for piece in str(raw_value).split(","):
            item = piece.strip()
            if not item or "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip().lower()
            if not name:
                continue
            try:
                weights[name] = float(value)
            except ValueError:
                continue
        return weights

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.ts_seq_num, None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num

        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len)

    def get_cross_network(self, args, layers=-1):
        if hasattr(self, "d_ts"):
            embed_dim = self.d_ts
        elif hasattr(self, "d_txt"):
            embed_dim = self.d_txt
        elif hasattr(self, "d_cxr"):
            embed_dim = self.d_cxr
        elif hasattr(self, "d_ecg"):
            embed_dim = self.d_ecg
        else:
            raise ValueError("No modality embedding dimension found for cross network")
        q_seq_len = self.tt_max
        return TransformerCrossEncoder(args=args,
                                        embed_dim=embed_dim,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        device=self.device,
                                        attn_dropout=self.dropout,
                                        relu_dropout=self.dropout,
                                        res_dropout=self.dropout,
                                        embed_dropout=self.dropout,
                                        attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len,
                                        num_modalities=self.num_modalities)

    def learn_time_embedding(self, tt):
        '''
        Time2Vec Module
        '''
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        # only two dimension?
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx), device=missing_idx.device)
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool, device=missing_idx.device)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def _missing_modality_fill(self, modality_id, num_missing, device, dtype):
        if num_missing == 0:
            return torch.zeros(
                (self.args.tt_max, num_missing, self.args.embed_dim),
                dtype=dtype,
                device=device,
            )
        if self.missing_modality_proxy_tokens is not None:
            proxy = self.missing_modality_proxy_tokens[int(modality_id)].to(device=device, dtype=dtype)
            return proxy.unsqueeze(1).expand(-1, num_missing, -1)
        if self.missing_modality_embeddings is None:
            return torch.zeros(
                (self.args.tt_max, num_missing, self.args.embed_dim),
                dtype=dtype,
                device=device,
            )
        ids = torch.full(
            (self.args.tt_max, num_missing),
            int(modality_id),
            dtype=torch.long,
            device=device,
        )
        return self.missing_modality_embeddings(ids).to(dtype=dtype)

    def _apply_maestro_tokens(self, modality_embeddings, missing_masks, modality_order):
        self.last_maestro_token_details = {}
        if self.maestro_token_block is None:
            return modality_embeddings
        available = [name for name in modality_order if name in modality_embeddings and modality_embeddings[name] is not None]
        if not available:
            return modality_embeddings

        pooled = []
        masks = []
        for name in available:
            seq = modality_embeddings[name]
            pooled.append(seq.mean(dim=0))
            missing = missing_masks.get(name)
            if missing is None:
                missing = torch.zeros(seq.size(1), device=seq.device, dtype=torch.bool)
            else:
                missing = missing.to(device=seq.device).bool()
            masks.append(missing)

        pooled_tokens = torch.stack(pooled, dim=1)
        missing_mask = torch.stack(masks, dim=1).to(device=pooled_tokens.device)
        modality_ids = torch.arange(len(available), device=pooled_tokens.device)
        encoded_tokens = self.maestro_token_block(pooled_tokens, missing_mask, modality_ids)
        delta = encoded_tokens - pooled_tokens

        if 0 < self.maestro_sparse_topk < len(available):
            scores = delta.detach().float().norm(dim=-1)
            keep = torch.zeros_like(scores, dtype=torch.bool)
            topk = scores.topk(k=self.maestro_sparse_topk, dim=1).indices
            keep.scatter_(1, topk, True)
            delta = delta * keep.unsqueeze(-1).to(delta.dtype)

        updated = dict(modality_embeddings)
        for idx, name in enumerate(available):
            seq = modality_embeddings[name]
            updated[name] = seq + self.maestro_residual_weight * delta[:, idx, :].unsqueeze(0).to(
                device=seq.device,
                dtype=seq.dtype,
            )

        with torch.no_grad():
            self.last_maestro_token_details = {
                "maestro_token_count": float(len(available)),
                "maestro_missing_rate": float(missing_mask.float().mean().detach().cpu().item()),
                "maestro_delta_norm": float(delta.detach().float().norm(dim=-1).mean().cpu().item()),
            }
        return updated

    def _compute_note_semantic_profile_logits(self, text_emb, note_time_mask_list):
        if (
            not self.args.use_semantic_expert_profiles
            or self.semantic_profile_source != "note"
            or text_emb is None
            or self.semantic_expert_profile_embeddings is None
        ):
            return None

        profiles = self.semantic_expert_profile_embeddings.to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        note_embeddings = F.normalize(text_emb, dim=-1)
        profile_embeddings = F.normalize(profiles, dim=-1)
        note_scores = note_embeddings @ profile_embeddings.t()

        if note_time_mask_list is not None:
            valid_notes = note_time_mask_list.to(device=text_emb.device).bool()
        else:
            valid_notes = torch.ones(note_scores.shape[:2], dtype=torch.bool, device=text_emb.device)

        if self.semantic_profile_note_pooling == "mean":
            masked_scores = note_scores.masked_fill(~valid_notes.unsqueeze(-1), 0.0)
            denom = valid_notes.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=text_emb.dtype)
            return masked_scores.sum(dim=1) / denom

        masked_scores = note_scores.masked_fill(~valid_notes.unsqueeze(-1), -1e4)
        pooled = masked_scores.max(dim=1).values
        no_valid_notes = valid_notes.sum(dim=1) == 0
        if no_valid_notes.any():
            pooled[no_valid_notes] = 0.0
        return pooled

    def _pool_modality_embedding(self, value):
        return value.float().mean(dim=0)

    def _labelwise_parameters(self, active_task_name):
        if self.multitask_shared_moe_trunk:
            return (
                self.labelwise_query[active_task_name],
                self.labelwise_output[active_task_name],
                self.labelwise_bias[active_task_name],
            )
        return self.labelwise_query, self.labelwise_output, self.labelwise_bias

    def _observed_mask(self, missing, batch_size, device):
        if missing is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return ~missing.to(device=device).bool()

    def _resolve_task_name(self, task_name_override=None):
        if task_name_override:
            task_name = str(task_name_override).lower()
            if "pheno" in task_name:
                return "pheno"
            if "los" in task_name:
                return "los"
            return "ihm"
        return self.task_name

    def _task_router_embedding(self, batch_size, device, task_name_override=None):
        if self.task_router_embeddings is None:
            return None, None, self._resolve_task_name(task_name_override)
        task_name = self._resolve_task_name(task_name_override)
        task_id_value = self.task_to_id[task_name]
        task_ids = torch.full(
            (batch_size,),
            int(task_id_value),
            dtype=torch.long,
            device=device,
        )
        return self.task_router_embeddings(task_ids), task_ids, task_name

    def _observed_modality_mask_features(self, batch_size, device, text_missing=None, cxr_missing=None, ecg_missing=None):
        observed_ts = torch.ones(batch_size, dtype=torch.long, device=device) if "TS" in self.modeltype else torch.zeros(batch_size, dtype=torch.long, device=device)
        observed_text = self._observed_mask(text_missing, batch_size, device).long() if "Text" in self.modeltype else torch.zeros(batch_size, dtype=torch.long, device=device)
        observed_cxr = self._observed_mask(cxr_missing, batch_size, device).long() if "CXR" in self.modeltype else torch.zeros(batch_size, dtype=torch.long, device=device)
        observed_ecg = self._observed_mask(ecg_missing, batch_size, device).long() if "ECG" in self.modeltype else torch.zeros(batch_size, dtype=torch.long, device=device)
        mask_ids = (
            observed_text
            + (observed_cxr * 2)
            + (observed_ecg * 4)
            + (observed_ts * 8)
        ).long()
        mask_strings = []
        for idx in range(batch_size):
            names = []
            if int(observed_text[idx].item()) == 1:
                names.append("text")
            if int(observed_cxr[idx].item()) == 1:
                names.append("cxr")
            if int(observed_ecg[idx].item()) == 1:
                names.append("ecg")
            if int(observed_ts[idx].item()) == 1:
                names.append("ts")
            mask_strings.append("+".join(names) if names else "none")
        return mask_ids, mask_strings

    def _modality_mask_router_embedding(self, batch_size, device, text_missing=None, cxr_missing=None, ecg_missing=None):
        mask_ids, mask_strings = self._observed_modality_mask_features(
            batch_size=batch_size,
            device=device,
            text_missing=text_missing,
            cxr_missing=cxr_missing,
            ecg_missing=ecg_missing,
        )
        if self.modality_mask_router_embeddings is None:
            return None, mask_ids, mask_strings
        observed_flags = torch.stack(
            [
                torch.ones_like(mask_ids, dtype=torch.float32, device=device),
                self._observed_mask(text_missing, batch_size, device).float(),
                self._observed_mask(cxr_missing, batch_size, device).float(),
                self._observed_mask(ecg_missing, batch_size, device).float(),
            ],
            dim=1,
        )
        mask_embedding, mask_ids = self.modality_mask_router_embeddings(observed_flags)
        return mask_embedding, mask_ids, mask_strings

    def _compute_interaction_features(self, batch_size, device, text_missing=None, cxr_missing=None, ecg_missing=None, proj_x_ts=None, proj_x_txt=None, proj_x_cxr=None, proj_x_ecg=None):
        pooled = {
            "ts": self._pool_modality_embedding(proj_x_ts) if proj_x_ts is not None else None,
            "text": self._pool_modality_embedding(proj_x_txt) if proj_x_txt is not None else None,
            "cxr": self._pool_modality_embedding(proj_x_cxr) if proj_x_cxr is not None else None,
            "ecg": self._pool_modality_embedding(proj_x_ecg) if proj_x_ecg is not None else None,
        }
        missing = {
            "ts": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "text": text_missing.to(device=device).bool() if text_missing is not None else None,
            "cxr": cxr_missing.to(device=device).bool() if cxr_missing is not None else None,
            "ecg": ecg_missing.to(device=device).bool() if ecg_missing is not None else None,
        }
        output = self.interaction_feature_extractor(pooled, missing)
        self.last_interaction_feature_output = output
        return output.features, output.feature_dict

    def _modality_token_id_map(self):
        mapping = {}
        token_id = 0
        if "TS" in self.modeltype:
            mapping["ts"] = token_id
            token_id += 1
        if "Text" in self.modeltype:
            mapping["text"] = token_id
            token_id += 1
        if "CXR" in self.modeltype:
            mapping["cxr"] = token_id
            token_id += 1
        if "ECG" in self.modeltype:
            mapping["ecg"] = token_id
            token_id += 1
        return mapping

    def _cross_modal_proxy_sequence(self, target_name, modality_embeddings, missing_masks, batch_size, device, dtype):
        if (
            not self.use_cross_modal_missing_proxies
            or target_name not in self.cross_modal_missing_proxy_heads
            or target_name not in modality_embeddings
        ):
            return None

        source_terms = []
        source_masks = []
        for source_name, source_value in modality_embeddings.items():
            if source_name == target_name or source_value is None:
                continue
            source_terms.append(self._pool_modality_embedding(source_value))
            source_missing = missing_masks.get(source_name)
            source_masks.append(self._observed_mask(source_missing, batch_size, device).float().unsqueeze(1))
        if not source_terms:
            return None

        stacked_sources = torch.stack(source_terms, dim=0)
        stacked_masks = torch.stack(source_masks, dim=0)
        source_count = stacked_masks.sum(dim=0).clamp_min(1.0)
        pooled_source = (stacked_sources * stacked_masks).sum(dim=0) / source_count
        pooled_source = pooled_source.to(dtype=dtype)

        proxy = self.cross_modal_missing_proxy_heads[target_name](pooled_source)
        proxy = proxy.view(batch_size, self.args.tt_max, self.args.embed_dim).permute(1, 0, 2).contiguous()
        token_id = self._modality_token_id_map()[target_name]
        token_ids = torch.full(
            (self.args.tt_max, batch_size),
            int(token_id),
            dtype=torch.long,
            device=device,
        )
        proxy = proxy + self.token_type_embeddings(token_ids).to(dtype=dtype)
        return proxy

    def _combine_collaboration_logits(self, moe_logits, dense_logits):
        if not self.use_collaboration_branch or dense_logits is None:
            self.last_collaboration_diagnostics = None
            return moe_logits

        fusion_mode = self.collaboration_fusion
        if fusion_mode == "input_dependent":
            fusion_mode = "confidence_based"
        if fusion_mode == "fixed_average":
            moe_weight = float(self.collaboration_fixed_weight)
            weights = moe_logits.new_full((moe_logits.size(0), 1), moe_weight)
        elif fusion_mode == "learnable_scalar":
            moe_weight = torch.sigmoid(self.collaboration_fusion_logit).to(device=moe_logits.device, dtype=moe_logits.dtype)
            weights = moe_weight.view(1, 1).expand(moe_logits.size(0), 1)
        else:
            if 'pheno' in self.task:
                moe_prob = torch.sigmoid(moe_logits)
                dense_prob = torch.sigmoid(dense_logits)
                moe_conf = (moe_prob - 0.5).abs().mean(dim=-1, keepdim=True)
                dense_conf = (dense_prob - 0.5).abs().mean(dim=-1, keepdim=True)
            elif moe_logits.size(1) == 1:
                moe_conf = torch.abs(torch.sigmoid(moe_logits) - 0.5)
                dense_conf = torch.abs(torch.sigmoid(dense_logits) - 0.5)
            else:
                moe_conf = torch.softmax(moe_logits, dim=-1).max(dim=-1, keepdim=True).values
                dense_conf = torch.softmax(dense_logits, dim=-1).max(dim=-1, keepdim=True).values
            weights = moe_conf / (moe_conf + dense_conf + 1e-12)

        fused = weights * moe_logits + (1.0 - weights) * dense_logits
        with torch.no_grad():
            if 'pheno' in self.task:
                moe_pred = (torch.sigmoid(moe_logits) > 0.5).float()
                dense_pred = (torch.sigmoid(dense_logits) > 0.5).float()
                agreement = (moe_pred == dense_pred).float().mean(dim=1)
            elif moe_logits.size(1) == 1:
                moe_pred = (torch.sigmoid(moe_logits) > 0.5).float()
                dense_pred = (torch.sigmoid(dense_logits) > 0.5).float()
                agreement = (moe_pred == dense_pred).float().mean(dim=1)
            else:
                agreement = (moe_logits.argmax(dim=-1) == dense_logits.argmax(dim=-1)).float()
            self.last_collaboration_diagnostics = {
                "fusion_weight_moe": weights.detach().cpu(),
                "fusion_weight_dense": (1.0 - weights).detach().cpu(),
                "branch_agreement": agreement.detach().cpu(),
                "moe_logits": moe_logits.detach().cpu(),
                "dense_logits": dense_logits.detach().cpu(),
            }
        return fused

    def _compute_unimodal_kd_loss(self, modality_embeddings, missing_masks, unimodal_teacher_targets, active_task_name):
        self.last_unimodal_kd_loss = None
        self.last_unimodal_kd_details = {}
        if not self.use_unimodal_kd or not unimodal_teacher_targets:
            return None

        kd_terms = []
        temperature = max(self.unimodal_kd_temperature, 1e-6)
        for modality, head in self.unimodal_heads.items():
            if modality not in modality_embeddings or modality not in unimodal_teacher_targets:
                continue
            teacher_entry = unimodal_teacher_targets[modality]
            teacher_targets = teacher_entry["targets"].to(
                device=modality_embeddings[modality].device,
                dtype=modality_embeddings[modality].dtype,
            )
            teacher_available = teacher_entry["available"].to(device=teacher_targets.device).bool()
            if modality != "ts":
                teacher_available = teacher_available & self._observed_mask(
                    missing_masks.get(modality),
                    teacher_targets.shape[0],
                    teacher_targets.device,
                )
            if not teacher_available.any():
                continue

            pooled = self._pool_modality_embedding(modality_embeddings[modality])
            student_logits = head(pooled)
            student_valid = student_logits[teacher_available]
            teacher_valid = teacher_targets[teacher_available]
            if active_task_name == "pheno":
                teacher_prob = torch.sigmoid(teacher_valid / temperature)
                kd_loss = F.binary_cross_entropy_with_logits(
                    student_valid / temperature,
                    teacher_prob,
                ) * (temperature ** 2)
            else:
                kd_loss = F.kl_div(
                    F.log_softmax(student_valid / temperature, dim=-1),
                    F.softmax(teacher_valid / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
            weight = float(self.unimodal_kd_weight_map.get(modality, 1.0))
            kd_terms.append(weight * kd_loss)
            self.last_unimodal_kd_details[f"unimodal_kd_{modality}"] = float(kd_loss.detach().cpu().item())
            self.last_unimodal_kd_details[f"unimodal_kd_{modality}_count"] = float(teacher_available.sum().detach().cpu().item())

        if not kd_terms:
            return None

        total_kd = self.unimodal_kd_weight * torch.stack(kd_terms).sum()
        self.last_unimodal_kd_loss = total_kd.detach()
        return total_kd

    def _compute_labelwise_logits(self, modality_embeddings, missing_masks, fused_latent, active_task_name):
        if not self.use_labelwise_fusion:
            return None

        source_vectors = []
        source_masks = []
        batch_size = fused_latent.shape[0]
        device = fused_latent.device

        if self.labelwise_fusion_source == "branches":
            source_vectors.append(self.labelwise_fused_proj(fused_latent))
            source_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))

        for modality_name in self.available_modality_names:
            value = modality_embeddings.get(modality_name)
            if value is None:
                continue
            source_vectors.append(self.labelwise_source_proj(self._pool_modality_embedding(value)))
            source_masks.append(self._observed_mask(missing_masks.get(modality_name), batch_size, device))

        if not source_vectors:
            return None

        stacked = torch.stack(source_vectors, dim=1)
        valid = torch.stack(source_masks, dim=1)
        query, output_weight, bias = self._labelwise_parameters(active_task_name)
        scores = torch.einsum("bsh,lh->bls", stacked, query.to(device=stacked.device, dtype=stacked.dtype))
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bls,bsh->blh", weights, stacked)
        return (context * output_weight.to(device=context.device, dtype=context.dtype).unsqueeze(0)).sum(dim=-1) + bias.to(device=context.device, dtype=context.dtype).unsqueeze(0)

    def _select_output_head(self, last_hs_proj, active_task_name):
        if not self.multitask_shared_moe_trunk:
            return self.out_layer(last_hs_proj)
        return self.multitask_heads[active_task_name](last_hs_proj)

    def _task_loss_and_prediction(self, output, labels, active_task_name, missing_recon_loss=None, ts_patch_recon_loss=None, ts_aux_loss=None):
        if active_task_name in {"ihm", "los"}:
            if labels is not None:
                labels = labels.to(output.device)
                task_loss = self.multitask_loss_fns[active_task_name](output, labels) if self.multitask_shared_moe_trunk else self.loss_fct1(output, labels)
                if missing_recon_loss is not None:
                    task_loss = task_loss + self.missing_modality_recon_coef * missing_recon_loss
                if ts_patch_recon_loss is not None:
                    task_loss = task_loss + self.ts_patch_recon_weight * ts_patch_recon_loss
                if ts_aux_loss is not None:
                    task_loss = task_loss + ts_aux_loss
                return task_loss
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]
        if labels is not None:
            labels = labels.to(output.device).float()
            task_loss = self.multitask_loss_fns["pheno"](output, labels) if self.multitask_shared_moe_trunk else self.loss_fct1(output, labels)
            if missing_recon_loss is not None:
                task_loss = task_loss + self.missing_modality_recon_coef * missing_recon_loss
            if ts_patch_recon_loss is not None:
                task_loss = task_loss + self.ts_patch_recon_weight * ts_patch_recon_loss
            if ts_aux_loss is not None:
                task_loss = task_loss + ts_aux_loss
            return task_loss
        return torch.nn.functional.sigmoid(output)

    def _compute_missing_modality_recon_loss(self, modality_embeddings, missing_masks):
        if not self.use_missing_modality_recon:
            self.last_missing_recon_loss = None
            self.last_missing_recon_details = {}
            return None

        available = {
            name: self._pool_modality_embedding(value)
            for name, value in modality_embeddings.items()
            if value is not None
        }
        if len(available) < 2:
            self.last_missing_recon_loss = None
            self.last_missing_recon_details = {}
            return None

        batch_size = next(iter(available.values())).shape[0]
        device = next(iter(available.values())).device
        observed_masks = {
            name: self._observed_mask(missing_masks.get(name), batch_size, device)
            for name in available
        }

        losses = []
        details = {}
        for raw_target in self.missing_modality_recon_targets:
            target = "text" if raw_target == "txt" else raw_target
            if target not in available or target not in self.missing_recon_heads:
                continue

            source_terms = []
            source_masks = []
            for source_name, source_value in available.items():
                if source_name == target:
                    continue
                source_terms.append(source_value)
                source_masks.append(observed_masks[source_name].float().unsqueeze(1))
            if not source_terms:
                continue

            stacked_sources = torch.stack(source_terms, dim=0)
            stacked_masks = torch.stack(source_masks, dim=0)
            source_count = stacked_masks.sum(dim=0).clamp_min(1.0)
            source_embedding = (stacked_sources * stacked_masks).sum(dim=0) / source_count

            target_mask = observed_masks[target]
            if not target_mask.any():
                continue

            pred = self.missing_recon_heads[target](source_embedding)
            target_embedding = available[target].detach()
            pred_valid = pred[target_mask]
            target_valid = target_embedding[target_mask]
            mse_per_sample = F.mse_loss(pred_valid, target_valid, reduction="none").mean(dim=1)
            cosine_per_sample = 1.0 - F.cosine_similarity(pred_valid, target_valid, dim=1)
            target_loss = 0.5 * (mse_per_sample.mean() + cosine_per_sample.mean())
            losses.append(target_loss)
            with torch.no_grad():
                sim = pred_valid @ target_valid.t()
                topk = min(5, sim.size(1))
                topk_indices = sim.topk(k=topk, dim=1).indices
                gold = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
                retrieval_at1 = (topk_indices[:, :1] == gold).any(dim=1).float().mean()
                retrieval_at5 = (topk_indices == gold).any(dim=1).float().mean()
            details[f"{target}_loss"] = float(target_loss.detach().cpu().item())
            details[f"{target}_mse"] = float(mse_per_sample.mean().detach().cpu().item())
            details[f"{target}_cosine"] = float((1.0 - cosine_per_sample.mean()).detach().cpu().item())
            details[f"{target}_retrieval_at1"] = float(retrieval_at1.detach().cpu().item())
            details[f"{target}_retrieval_at5"] = float(retrieval_at5.detach().cpu().item())

        if not losses:
            self.last_missing_recon_loss = None
            self.last_missing_recon_details = {}
            return None

        recon_loss = torch.stack(losses).mean()
        self.last_missing_recon_loss = recon_loss.detach()
        self.last_missing_recon_details = details
        return recon_loss

    def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
                attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
                labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
                ecg_time=None, ecg_time_mask=None, router_organ_targets=None, task_name_override=None,
                unimodal_teacher_targets=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        ts_patch_recon_loss = None
        self.last_ts_patch_recon_loss = None
        self.last_ts_patch_recon_details = {}
        self.last_ts_aux_loss = None
        self.last_ts_aux_details = {}
        shared_time_query = None
        if "TS" in self.modeltype:
            # mTAND module part
            if self.irregular_learn_emb_ts:
                shared_time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                if self.use_ts_variable_tokens:
                    proj_x_ts_irg = self._encode_ts_variable_tokens(x_ts, x_ts_mask, ts_tt_list)
                else:
                    time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                    x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                    x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
                    proj_x_ts_irg=self.time_attn_ts(shared_time_query, time_key_ts, x_ts_irg, x_ts_mask)
                    proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                if self.use_ts_patch_encoder:
                    proj_x_ts_reg, ts_patch_recon_loss, ts_patch_recon_details = self._encode_ts_patches(reg_ts.to(self.device))
                    if ts_patch_recon_loss is not None:
                        self.last_ts_patch_recon_loss = ts_patch_recon_loss.detach()
                        self.last_ts_patch_recon_details = ts_patch_recon_details
                elif self.use_ts_state_space_encoder:
                    proj_x_ts_reg = self._encode_ts_state_space(reg_ts.to(self.device))
                else:
                    x_ts_reg = reg_ts.transpose(1, 2).to(self.device)
                    proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                    proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg, dim=0).values
                    g_reg =torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate=torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                # print('moe_gate', torch.isnan(moe_gate).any())
                mixup_rate = self.moe(moe_gate)
                # print('mixup_rate', torch.isnan(mixup_rate).any())
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
            proj_x_ts += self.token_type_embeddings(
                torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=proj_x_ts.device)
            )
            proj_x_ts, ts_confidence_details = self._apply_ts_confidence_fusion(proj_x_ts, x_ts_mask)
            self.last_ts_aux_details.update(ts_confidence_details)

        missing_recon_loss = None
        mod_count = 1 if "TS" in self.modeltype else 0
        if "Text" in self.modeltype:
            # compute irregular clinical notes attention
            # if text_missing is None or torch.all(text_missing == 0):
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
            text_batch_size = x_txt.shape[0]
            text_device = x_txt.device

            semantic_profile_logits = self._compute_note_semantic_profile_logits(
                x_txt,
                note_time_mask_list,
            )
            if semantic_profile_logits is not None and text_missing is not None:
                semantic_profile_logits = semantic_profile_logits.masked_fill(
                    text_missing.to(device=semantic_profile_logits.device).bool().unsqueeze(1),
                    0.0,
                )

            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if "TS" not in self.modeltype or not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                else:
                    time_query = shared_time_query
                proj_x_txt=self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, text_batch_size), dtype=torch.long, device=proj_x_txt.device)
                )
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=proj_x_txt.device)
                )
                if self.use_cross_modal_missing_proxies:
                    proj_x_txt[:, missing_indices, :] = 0.0
                else:
                    proj_x_txt[:, missing_indices, :] = self._missing_modality_fill(
                        mod_count,
                        len(missing_indices),
                        proj_x_txt.device,
                        proj_x_txt.dtype,
                    )
            mod_count += 1

        if "CXR" in self.modeltype:
            # compute irregular clinical notes attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                if "TS" not in self.modeltype or not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                else:
                    time_query = shared_time_query

                proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr=proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=proj_x_cxr.device)
                )
            elif not torch.all(cxr_missing == 0):
                # proj_x_cxr = None
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=proj_x_cxr.device)
                )
                if self.use_cross_modal_missing_proxies:
                    proj_x_cxr[:, missing_indices, :] = 0.0
                else:
                    proj_x_cxr[:, missing_indices, :] = self._missing_modality_fill(
                        mod_count,
                        len(missing_indices),
                        proj_x_cxr.device,
                        proj_x_cxr.dtype,
                    )
            mod_count += 1

        if "ECG" in self.modeltype:
            # compute irregular ECG attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                if "TS" not in self.modeltype or not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                else:
                    time_query = shared_time_query

                proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg=proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=proj_x_ecg.device)
                )
            elif not torch.all(ecg_missing == 0):
                # proj_x_ecg = None
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(
                    mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=proj_x_ecg.device)
                )
                if self.use_cross_modal_missing_proxies:
                    proj_x_ecg[:, missing_indices, :] = 0.0
                else:
                    proj_x_ecg[:, missing_indices, :] = self._missing_modality_fill(
                        mod_count,
                        len(missing_indices),
                        proj_x_ecg.device,
                        proj_x_ecg.dtype,
                    )
            mod_count += 1

        if self.use_cross_modal_missing_proxies:
            modality_embeddings = {}
            missing_masks = {}
            if "TS" in self.modeltype:
                modality_embeddings["ts"] = proj_x_ts
                missing_masks["ts"] = None
            if "Text" in self.modeltype:
                modality_embeddings["text"] = proj_x_txt
                missing_masks["text"] = text_missing
            if "CXR" in self.modeltype:
                modality_embeddings["cxr"] = proj_x_cxr
                missing_masks["cxr"] = cxr_missing
            if "ECG" in self.modeltype:
                modality_embeddings["ecg"] = proj_x_ecg
                missing_masks["ecg"] = ecg_missing
            for target_name, missing_mask in (
                ("text", text_missing),
                ("cxr", cxr_missing),
                ("ecg", ecg_missing),
            ):
                if target_name not in modality_embeddings or missing_mask is None or not torch.any(missing_mask != 0):
                    continue
                proxy_sequence = self._cross_modal_proxy_sequence(
                    target_name=target_name,
                    modality_embeddings=modality_embeddings,
                    missing_masks=missing_masks,
                    batch_size=x_ts.shape[0],
                    device=modality_embeddings[target_name].device,
                    dtype=modality_embeddings[target_name].dtype,
                )
                if proxy_sequence is None:
                    continue
                missing_indices, _ = self._missing_indices(missing_mask)
                modality_embeddings[target_name][:, missing_indices, :] = proxy_sequence[:, missing_indices, :]

        modality_embeddings = {}
        missing_masks = {}
        if "TS" in self.modeltype:
            modality_embeddings["ts"] = proj_x_ts
            missing_masks["ts"] = None
        if "Text" in self.modeltype:
            modality_embeddings["text"] = proj_x_txt
            missing_masks["text"] = text_missing
        if "CXR" in self.modeltype:
            modality_embeddings["cxr"] = proj_x_cxr
            missing_masks["cxr"] = cxr_missing
        if "ECG" in self.modeltype:
            modality_embeddings["ecg"] = proj_x_ecg
            missing_masks["ecg"] = ecg_missing

        modality_order = []
        if "TS" in self.modeltype:
            modality_order.append("ts")
        if "CXR" in self.modeltype:
            modality_order.append("cxr")
        if "Text" in self.modeltype:
            modality_order.append("text")
        if "ECG" in self.modeltype:
            modality_order.append("ecg")
        modality_embeddings = self._apply_maestro_tokens(
            modality_embeddings=modality_embeddings,
            missing_masks=missing_masks,
            modality_order=modality_order,
        )
        if "TS" in modality_embeddings:
            proj_x_ts = modality_embeddings["ts"]
        if "Text" in self.modeltype and "text" in modality_embeddings:
            proj_x_txt = modality_embeddings["text"]
        if "CXR" in self.modeltype and "cxr" in modality_embeddings:
            proj_x_cxr = modality_embeddings["cxr"]
        if "ECG" in self.modeltype and "ecg" in modality_embeddings:
            proj_x_ecg = modality_embeddings["ecg"]

        ts_aux_loss = None
        if "TS" in self.modeltype:
            proj_x_ts, ts_shared_private_loss, ts_shared_private_details = self._compute_ts_shared_private_loss(
                proj_x_ts,
                modality_embeddings,
                missing_masks,
            )
            modality_embeddings["ts"] = proj_x_ts
            self.last_ts_aux_details.update(ts_shared_private_details)
            if ts_shared_private_loss is not None:
                ts_aux_loss = ts_shared_private_loss if ts_aux_loss is None else ts_aux_loss + ts_shared_private_loss

            ts_distill_loss, ts_distill_details = self._compute_ts_cross_modal_distill_loss(
                proj_x_ts,
                modality_embeddings,
                missing_masks,
            )
            self.last_ts_aux_details.update(ts_distill_details)
            if ts_distill_loss is not None:
                ts_aux_loss = ts_distill_loss if ts_aux_loss is None else ts_aux_loss + ts_distill_loss
            if ts_aux_loss is not None:
                self.last_ts_aux_loss = ts_aux_loss.detach()

        if self.use_missing_modality_recon:
            missing_recon_loss = self._compute_missing_modality_recon_loss(
                modality_embeddings,
                missing_masks,
            )
        else:
            self.last_missing_recon_loss = None
            self.last_missing_recon_details = {}

        router_task_embedding = None
        router_task_ids = None
        router_mask_embedding = None
        router_mask_ids = None
        router_mask_strings = None
        interaction_features = None
        interaction_feature_dict = None
        batch_size = x_ts.shape[0]
        routing_device = (
            proj_x_ts.device if "TS" in self.modeltype else
            proj_x_txt.device if "Text" in self.modeltype else
            proj_x_cxr.device if "CXR" in self.modeltype else
            proj_x_ecg.device
        )
        active_task_name = self._resolve_task_name(task_name_override)
        if self.task_router_embeddings is not None:
            router_task_embedding, router_task_ids, active_task_name = self._task_router_embedding(batch_size, routing_device, task_name_override=task_name_override)
        if self.use_modality_mask_condition_router:
            router_mask_embedding, router_mask_ids, router_mask_strings = self._modality_mask_router_embedding(
                batch_size=batch_size,
                device=routing_device,
                text_missing=text_missing,
                cxr_missing=cxr_missing,
                ecg_missing=ecg_missing,
            )
        else:
            _, router_mask_ids, router_mask_strings = self._modality_mask_router_embedding(
                batch_size=batch_size,
                device=routing_device,
                text_missing=text_missing,
                cxr_missing=cxr_missing,
                ecg_missing=ecg_missing,
            )

        if self.use_interaction_router:
            interaction_features, interaction_feature_dict = self._compute_interaction_features(
                batch_size=batch_size,
                device=routing_device,
                text_missing=text_missing,
                cxr_missing=cxr_missing,
                ecg_missing=ecg_missing,
                proj_x_ts=proj_x_ts if "TS" in self.modeltype else None,
                proj_x_txt=proj_x_txt if "Text" in self.modeltype else None,
                proj_x_cxr=proj_x_cxr if "CXR" in self.modeltype else None,
                proj_x_ecg=proj_x_ecg if "ECG" in self.modeltype else None,
            )

        balance_loss = None
        if self.cross_method in ["self_cross", "moe", "hme"]:
            if "Text" not in self.modeltype:
                semantic_profile_logits = None
            if self.modeltype == "TS_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_txt, proj_x_ts],
                    ['txt', 'ts'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding if (self.use_task_condition_router or self.use_task_condition_expert_modulation or self.multitask_shared_moe_trunk) else None,
                    task_name=active_task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )
            elif self.modeltype == "Text_MOE":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_txt],
                    ['txt'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding,
                    task_name=self.task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )
            elif self.modeltype == "TS_MOE":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_ts],
                    ['ts'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding,
                    task_name=self.task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )
            elif self.modeltype == "TS_CXR":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_cxr, proj_x_ts],
                    ['cxr', 'ts'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding,
                    task_name=self.task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )
            elif self.modeltype == "TS_CXR_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_ts, proj_x_cxr, proj_x_txt],
                    ['ts', 'cxr', 'txt'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding,
                    task_name=self.task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg],
                    ['ts', 'cxr', 'txt', 'ecg'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                    task_embedding=router_task_embedding,
                    task_name=self.task_name,
                    task_id=router_task_ids,
                    modality_mask_embedding=router_mask_embedding,
                    modality_mask_ids=router_mask_ids,
                    modality_mask_strings=router_mask_strings,
                    interaction_features=interaction_features,
                )

            if hiddens is None:
                return None
            # h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
            # last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        else:
            if 'CXR' in self.modeltype:
                proj_x_txt = proj_x_cxr
            if self.cross_method=="MulT":
                # ts --> txt
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                # txt --> ts
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                last_h_ts=proj_x_ts[-1]
                last_h_txt=proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method=="MAGGate":
                    last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
                elif self.cross_method=="Outer":
                    last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self._select_output_head(last_hs_proj, active_task_name)
        labelwise_logits = self._compute_labelwise_logits(
            modality_embeddings=modality_embeddings,
            missing_masks=missing_masks,
            fused_latent=last_hs_proj,
            active_task_name=active_task_name,
        )
        if labelwise_logits is not None:
            output = labelwise_logits
        if self.use_collaboration_branch:
            if self.collaboration_transformer is not None and self.cross_method in ["self_cross", "moe", "hme"]:
                branch_tokens = torch.stack([hid[-1] for hid in hiddens], dim=1)
                dense_hidden = self.collaboration_transformer(branch_tokens).reshape(branch_tokens.size(0), -1)
                dense_logits = self.collaboration_out(dense_hidden)
            else:
                dense_hidden = F.relu(self.collaboration_proj1(last_hs))
                dense_hidden = F.dropout(dense_hidden, p=self.dropout, training=self.training)
                dense_hidden = self.collaboration_proj2(dense_hidden) + last_hs
                dense_logits = self.collaboration_out(dense_hidden)
            if self.multitask_shared_moe_trunk:
                dense_logits = self.multitask_heads[active_task_name](dense_hidden if self.collaboration_transformer is None else dense_hidden)
            output = self._combine_collaboration_logits(output, dense_logits)
        else:
            self.last_collaboration_diagnostics = None

        self.last_fused_latent = last_hs_proj
        self.last_output_logits = output
        unimodal_kd_loss = self._compute_unimodal_kd_loss(
            modality_embeddings=modality_embeddings,
            missing_masks=missing_masks,
            unimodal_teacher_targets=unimodal_teacher_targets,
            active_task_name=active_task_name,
        )

        if labels is not None:
            task_loss = self._task_loss_and_prediction(output, labels, active_task_name, missing_recon_loss=missing_recon_loss, ts_patch_recon_loss=ts_patch_recon_loss, ts_aux_loss=ts_aux_loss)
            if unimodal_kd_loss is not None:
                task_loss = task_loss + unimodal_kd_loss
            return task_loss, balance_loss
        return self._task_loss_and_prediction(output, labels, active_task_name, missing_recon_loss=missing_recon_loss, ts_patch_recon_loss=ts_patch_recon_loss, ts_aux_loss=ts_aux_loss)


class PAMPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used only for the lightweight PAM branch."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# class FlexiModalMULTCrossModel(nn.Module):
#     """
#     Model for PAMAP2 that accepts a list of modality tensors.
#     Uses the same cross encoder and final layers as the original MULTCrossModel.
#     """
#     def __init__(self, args, device, modality_dims):
#         super().__init__()
#         self.args = args
#         self.device = device
#         self.num_modalities = len(modality_dims)
#         self.d_model = args.embed_dim
#         self.dropout = args.dropout
#         self.cross_method = args.cross_method

#         # Give each modality its own local temporal encoder before cross-modal fusion.
#         self.modality_proj = nn.ModuleList([
#             nn.Conv1d(
#                 dim,
#                 self.d_model,
#                 kernel_size=args.kernel_size,
#                 padding=math.floor((args.kernel_size - 1) / 2),
#                 bias=False,
#             )
#             for dim in modality_dims
#         ])
#         self.modality_temporal = nn.ModuleList([
#             TransformerEncoder(
#                 embed_dim=self.d_model,
#                 num_heads=args.num_heads,
#                 layers=args.layers,
#                 device=self.device,
#                 attn_dropout=self.dropout,
#                 relu_dropout=self.dropout,
#                 res_dropout=self.dropout,
#                 embed_dropout=self.dropout,
#                 attn_mask=False,
#                 q_seq_len=args.tt_max,
#                 kv_seq_len=None,
#             )
#             for _ in modality_dims
#         ])
#         self.modality_positional = nn.ModuleList([
#             PAMPositionalEncoding(self.d_model, dropout=self.dropout, max_len=args.tt_max + 8)
#             for _ in modality_dims
#         ])
#         self.modality_self_attn = nn.ModuleList([
#             nn.MultiheadAttention(
#                 self.d_model,
#                 args.num_heads,
#                 dropout=self.dropout,
#                 batch_first=True,
#             )
#             for _ in modality_dims
#         ])
#         self.modality_attn_norm = nn.ModuleList([
#             nn.LayerNorm(self.d_model) for _ in modality_dims
#         ])
#         self.modality_ffn = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.d_model, self.d_model * 2),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout),
#                 nn.Linear(self.d_model * 2, self.d_model),
#             )
#             for _ in modality_dims
#         ])
#         self.modality_ffn_norm = nn.ModuleList([
#             nn.LayerNorm(self.d_model) for _ in modality_dims
#         ])
#         self.modality_norm = nn.ModuleList([
#             nn.LayerNorm(self.d_model) for _ in modality_dims
#         ])
#         self.token_type_embeddings = nn.Embedding(self.num_modalities, self.d_model)
#         # Lightweight attention blocks keep the PAM model simple while letting it
#         # reweight noisy channels and sensors before MoE fusion.
#         hidden_gate = max(16, self.d_model // 4)
#         self.modality_channel_gate = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.d_model, hidden_gate),
#                 nn.ReLU(),
#                 nn.Linear(hidden_gate, self.d_model),
#                 nn.Sigmoid(),
#             )
#             for _ in modality_dims
#         ])
#         self.temporal_pool = nn.ModuleList([
#             nn.Linear(self.d_model, 1) for _ in modality_dims
#         ])
#         self.modality_score = nn.Sequential(
#             nn.Linear(self.d_model, hidden_gate),
#             nn.ReLU(),
#             nn.Linear(hidden_gate, 1),
#         )

#         # Cross encoder (same as in original)
#         self.trans_self_cross_ts_txt = self._get_cross_network(args)

#         # Final layers
#         total_dim = self.d_model * self.num_modalities
#         self.proj1 = nn.Linear(total_dim, total_dim)
#         self.proj2 = nn.Linear(total_dim, total_dim)
#         self.out_layer = nn.Linear(total_dim, args.num_labels)
#         self.loss_fct = nn.CrossEntropyLoss()

#     def _get_cross_network(self, args):
#         from core.module import TransformerCrossEncoder
#         return TransformerCrossEncoder(
#             args=args,
#             embed_dim=self.d_model,
#             num_heads=args.num_heads,
#             layers=args.cross_layers,
#             device=self.device,
#             attn_dropout=self.dropout,
#             relu_dropout=self.dropout,
#             res_dropout=self.dropout,
#             embed_dropout=self.dropout,
#             attn_mask=False,
#             q_seq_len_1=args.tt_max,
#             num_modalities=self.num_modalities
#         )

#     def forward(self, modality_list, labels=None):
#         B, T = modality_list[0].shape[:2]

#         projected = []
#         for mod_idx, (
#             proj,
#             temporal_encoder,
#             pos_enc,
#             self_attn,
#             attn_norm,
#             ffn,
#             ffn_norm,
#             norm,
#             ch_gate,
#             mod,
#         ) in enumerate(
#             zip(
#                 self.modality_proj,
#                 self.modality_temporal,
#                 self.modality_positional,
#                 self.modality_self_attn,
#                 self.modality_attn_norm,
#                 self.modality_ffn,
#                 self.modality_ffn_norm,
#                 self.modality_norm,
#                 self.modality_channel_gate,
#                 modality_list,
#             )
#         ):
#             x = proj(mod.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
#             modality_ids = torch.full((B, T), mod_idx, dtype=torch.long, device=mod.device)
#             x = norm(x + self.token_type_embeddings(modality_ids))
#             x = pos_enc(x)
#             attn_out, _ = self_attn(x, x, x, need_weights=False)
#             x = attn_norm(x + F.dropout(attn_out, p=self.dropout, training=self.training))
#             x = ffn_norm(x + ffn(x))
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = temporal_encoder(x.permute(1, 0, 2))
#             gate = ch_gate(x.mean(dim=0)).unsqueeze(0)
#             x = x * gate
#             projected.append(x)

#         hiddens, balance_loss = self.trans_self_cross_ts_txt(
#             projected, [f"mod_{i}" for i in range(self.num_modalities)]
#         )
#         pooled = []
#         for hid, pool in zip(hiddens, self.temporal_pool):
#             hid_bt = hid.transpose(0, 1)
#             attn = torch.softmax(pool(hid_bt).squeeze(-1), dim=1).unsqueeze(-1)
#             pooled.append((hid_bt * attn).sum(dim=1))

#         modality_logits = torch.cat([self.modality_score(p) for p in pooled], dim=1)
#         modality_weights = torch.softmax(modality_logits, dim=1)
#         weighted_pooled = [
#             pooled[i] * modality_weights[:, i].unsqueeze(-1)
#             for i in range(self.num_modalities)
#         ]
#         pooled_hs = torch.cat(weighted_pooled, dim=1)

#         out = F.relu(self.proj1(pooled_hs))
#         out = F.dropout(out, p=self.dropout, training=self.training)
#         out = self.proj2(out) + pooled_hs
#         logits = self.out_layer(out)

#         if labels is not None:
#             loss = self.loss_fct(logits, labels)
#             return loss, balance_loss
#         else:
#             return logits

class FlexiModalMULTCrossModel(nn.Module):
    """
    Model for PAMAP2 that accepts a list of modality tensors.
    Uses the same cross encoder and final layers as the original MULTCrossModel.
    """
    def __init__(self, args, device, modality_dims):
        super().__init__()
        self.args = args
        self.device = device
        self.num_modalities = len(modality_dims)
        self.d_model = args.embed_dim
        self.dropout = args.dropout
        self.cross_method = args.cross_method

        # Project each modality to d_model
        self.modality_proj = nn.ModuleList([
            nn.Linear(dim, self.d_model) for dim in modality_dims
        ])

        # Cross encoder (same as in original)
        self.trans_self_cross_ts_txt = self._get_cross_network(args)

        # Final layers
        total_dim = self.d_model * self.num_modalities
        self.proj1 = nn.Linear(total_dim, total_dim)
        self.proj2 = nn.Linear(total_dim, total_dim)
        self.out_layer = nn.Linear(total_dim, args.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def _get_cross_network(self, args):
        from core.module import TransformerCrossEncoder
        return TransformerCrossEncoder(
            args=args,
            embed_dim=self.d_model,
            num_heads=args.num_heads,
            layers=args.cross_layers,
            device=self.device,
            attn_dropout=self.dropout,
            relu_dropout=self.dropout,
            res_dropout=self.dropout,
            embed_dropout=self.dropout,
            attn_mask=False,
            q_seq_len_1=args.tt_max,
            num_modalities=self.num_modalities
        )

    def forward(self, modality_list, labels=None):
        B, T = modality_list[0].shape[:2]

        # projected = [proj(mod) for proj, mod in zip(self.modality_proj, modality_list)]
        projected = []
        for proj, mod in zip(self.modality_proj, modality_list):
            x = proj(mod)                     # (B, T, d_model)
            x = x.permute(1, 0, 2)           # (T, B, d_model)
            projected.append(x)

        hiddens, balance_loss = self.trans_self_cross_ts_txt(
            projected, [f"mod_{i}" for i in range(self.num_modalities)]
        )
        last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)

        out = F.relu(self.proj1(last_hs))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.proj2(out) + last_hs
        logits = self.out_layer(out)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, balance_loss
        else:
            return logits
        
class TSMixed(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,ts_seq_num=None):

        super(TSMixed, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads

        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.Interp=args.Interp
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.TS_model=args.TS_model
        self.tt_max=args.tt_max

        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_ts=orig_d_ts
        self.d_ts=args.embed_dim
        self.ts_seq_num=ts_seq_num

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts=orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level=='batch':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq_feature':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

                # self.moe = nn.Linear(self.d_ts*self.tt_max*2, 1)
        if self.TS_model=='LSTM':
            self.trans_ts_mem=nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts, num_layers=args.layers,dropout=self.dropout,bidirectional=True)

        elif self.TS_model=='CNN':
            self.trans_ts_mem=TimeSeriesCnnModel(input_size=self.d_ts,n_filters=self.d_ts,filter_size=self.kernel_size,\
            dropout=self.dropout,length=self.tt_max,n_neurons=self.d_ts,layers=args.layers)
        elif self.TS_model=='Atten':
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
        
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer= nn.Linear(self.d_ts, output_dim)

        if 'ihm' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        embed_dim=self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts :
                q_seq_len= self.tt_max
            else:
                q_seq_len= self.ts_seq_num

        return TransformerEncoder(embed_dim=embed_dim,
                                    num_heads=self.num_heads,
                                    layers=layers,
                                    device=self.device,
                                    attn_dropout=self.dropout,
                                    relu_dropout=self.dropout,
                                    res_dropout=self.dropout,
                                    embed_dropout=self.dropout,
                                    attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                    kv_seq_len=None)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype :

            if self.Interp:
                x_ts_mask_interp=copy.deepcopy(x_ts_mask)
                x_ts_interp=copy.deepcopy(x_ts)
                recon_m=hold_out(x_ts_mask_interp)
                recon_m=torch.Tensor(recon_m).to(self.device)
                proj_x_ts_interp=self.proj_ts_intp(self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m))) #dimension [batch_size,  n_features,seq_len]
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp=self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m, reconstruction=True),reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2).to(self.device)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg=proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts :
                    proj_x_ts_reg=proj_x_ts_interp
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                # for name, parameter in self.moe.named_parameters():
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")


            if self.TS_model=='CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)

            elif self.TS_model=='LSTM':
                    _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            if  self.TS_model!='CNN':
                last_h_ts=proj_x_ts[-1]

            else:
                last_h_ts=proj_x_ts

 
            if self.modeltype=="TS" :
                last_hs=last_h_ts
            else:
                raise ValueError("Unknown model type")
                       
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_ts)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        if self.Interp:
            reconloss_interp=recon_loss(x_ts_interp,x_ts_mask_interp,recon_m,recon_interp,self.d_ts)

        if 'ihm' in self.task:
            if labels!=None:
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)
