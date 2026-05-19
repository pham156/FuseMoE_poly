import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from core.module import *
from core.interp import *
import copy
import pdb


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
    if profile_set != "icu_organ_system":
        raise ValueError(f"Unknown semantic_profile_set: {profile_set}")

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
            "Infection and inflammation expert. Focuses on sepsis, fever, antibiotics, "
            "white blood cell count, infection source, inflammatory response, and "
            "clinical deterioration from infection."
        ),
    ]


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
        self.use_instruction_router = args.use_instruction_router
        if self.use_instruction_router:
            instruction_embedding = encode_router_instruction(args, Biobert, tokenizer, device, self.modeltype)
            self.register_buffer("router_instruction_embedding", instruction_embedding)
        else:
            self.router_instruction_embedding = None

        if args.use_semantic_expert_profiles:
            profile_embeddings = encode_expert_profiles(args, Biobert, tokenizer, device)
            self.register_buffer("semantic_expert_profile_embeddings", profile_embeddings)
            args.semantic_profile_embeddings = profile_embeddings
        else:
            self.semantic_expert_profile_embeddings = None
            args.semantic_profile_embeddings = None

        self.semantic_profile_source = args.semantic_profile_source
        self.semantic_profile_note_pooling = args.semantic_profile_note_pooling

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
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

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

        output_dim = args.num_labels
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
        embed_dim, q_seq_len = self.d_ts, self.tt_max
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
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

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

    def _observed_mask(self, missing, batch_size, device):
        if missing is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return ~missing.to(device=device).bool()

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
            per_sample = F.mse_loss(pred, target_embedding, reduction="none").mean(dim=1)
            target_loss = per_sample[target_mask].mean()
            losses.append(target_loss)
            details[target] = float(target_loss.detach().cpu().item())

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
                ecg_time=None, ecg_time_mask=None, router_organ_targets=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype:
            # mTAND module part
            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                # print('proj_x_ts_reg', torch.isnan(proj_x_ts_reg).any())
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
            proj_x_ts += self.token_type_embeddings(torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))

        missing_recon_loss = None
        mod_count = 1
        if "Text" in self.modeltype:
            # compute irregular clinical notes attention
            # if text_missing is None or torch.all(text_missing == 0):
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)

            semantic_profile_logits = self._compute_note_semantic_profile_logits(
                x_txt,
                note_time_mask_list,
            )

            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_txt=self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "CXR" in self.modeltype:
            # compute irregular clinical notes attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr=proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(cxr_missing == 0):
                # proj_x_cxr = None
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "ECG" in self.modeltype:
            # compute irregular ECG attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg=proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(ecg_missing == 0):
                # proj_x_ecg = None
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if self.use_missing_modality_recon:
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
            missing_recon_loss = self._compute_missing_modality_recon_loss(
                modality_embeddings,
                missing_masks,
            )
        else:
            self.last_missing_recon_loss = None
            self.last_missing_recon_details = {}

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
                )
            elif self.modeltype == "TS_CXR":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_cxr, proj_x_ts],
                    ['cxr', 'ts'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                )
            elif self.modeltype == "TS_CXR_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_ts, proj_x_cxr, proj_x_txt],
                    ['ts', 'cxr', 'txt'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
                )
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens, balance_loss = self.trans_self_cross_ts_txt(
                    [proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg],
                    ['ts', 'cxr', 'txt', 'ecg'],
                    instruction_embedding=self.router_instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    router_organ_targets=router_organ_targets,
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
        output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task or 'los' in self.task:
            if labels!=None:
                task_loss = self.loss_fct1(output, labels)
                if missing_recon_loss is not None:
                    task_loss = task_loss + self.missing_modality_recon_coef * missing_recon_loss
                return task_loss, balance_loss
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                task_loss = self.loss_fct1(output, labels)
                if missing_recon_loss is not None:
                    task_loss = task_loss + self.missing_modality_recon_coef * missing_recon_loss
                return task_loss, balance_loss
            return torch.nn.functional.sigmoid(output)


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
                x_ts_reg = reg_ts.transpose(1, 2)
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
