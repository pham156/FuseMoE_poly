from email import parser
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../TS/mimic3-benchmarks')
sys.path.insert(0, '../ClinicalNotes_TimeSeries/models')
import pickle
import re
import numpy as np
import json
from preprocessing.data import *
import statistics as stat
logger = None
import  argparse
import pickle
from accelerate import Accelerator
from sklearn import metrics
import pdb
from torch.optim import AdamW
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel,
                          LongformerConfig,
                          LongformerModel,
                          LongformerTokenizer,
                         )
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
def parse_args():
    parser = argparse.ArgumentParser(description="Alignment text and ts data")
    parser.add_argument(
            "--task", type=str, default="ihm"
        )
    parser.add_argument(
        "--file_path", type=str, default="Data", help="A path to dataset folder"
    )
    parser.add_argument("--output_dir", type=str, default="Checkpoints", help="Where to store the final model.")
    parser.add_argument("--disable_run_folder_save", action="store_true", help="Do not create run_folder/checkpoint output directories for diagnostic runs.")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--modeltype", type=str, default="TS_Text", help="TS, Text or TS_Text")
    parser.add_argument("--eval_score", default=['auc', 'auprc', 'f1'], type=list)
    parser.add_argument("--primary_metric", type=str, default="f1", choices=["auc", "auprc", "f1", "macro_f1", "acc"], help="Metric used to select the best validation model")

    parser.add_argument("--dataset", type=str, default="mimic", choices=["mimic", "pam"], help="Choose dataset: mimic (original MIMIC) or pam (PAMAP2)")
    parser.add_argument("--pam_train_subjects", nargs='*', type=int, default=[1, 2, 3, 4, 5, 6], help="PAMAP2 subject IDs used for training")
    parser.add_argument("--pam_val_subjects", nargs='*', type=int, default=[7], help="PAMAP2 subject IDs used for validation")
    parser.add_argument("--pam_test_subjects", nargs='*', type=int, default=[8, 9], help="PAMAP2 subject IDs used for testing")
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated," " sequences shorter will be padded if `--pad_to_max_lengh` is passed."),)
    parser.add_argument( "--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument( "--model_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                    help="Fraction of training data to use (0 < ratio <= 1)")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size  for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument("--num_update_bert_epochs", type=int, default=10, help="Number of per training epochs update the bert model.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--txt_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for Txt self-attention and Bert to use.",
    )

    parser.add_argument(
        "--ts_learning_rate",
        type=float,
        default=0.0004,
        help="Initial learning rate for TS self-attention to use.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument( "--pt_mask_ratio",default=0.15, type=float, help="mask rate for pretrain .",
    )
    parser.add_argument( "--mean_mask_length",default=3, type=int, help="mean mask length for pretrain .",
    )

    parser.add_argument('--chunk', action='store_true')
    parser.add_argument("--chunk_type", default='sent_doc_pos', type=str, help="How to chunk the text. sent_doc_pos: sentence level position + doc level position")
    parser.add_argument("--warmup_proportion", default=0.10, type=float, help="proportion for the warmup in the lr scheduler.")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for CNN.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer encoder layer.")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of transformer cross encoder layer.")
    parser.add_argument("--embed_dim", default=30, type=int, help="attention embedding dim.")

    parser.add_argument("--irregular_learn_emb_ts", action='store_true')
    parser.add_argument("--irregular_learn_emb_text", action='store_true')
    parser.add_argument("--irregular_learn_emb_cxr", action='store_true')
    parser.add_argument("--irregular_learn_emb_ecg", action='store_true')
    parser.add_argument("--reg_ts", action='store_true')
    parser.add_argument("--tt_max", default=48, type=int, help="max time for irregular time series.")
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument('--ts_to_txt', action='store_true')
    parser.add_argument('--txt_to_ts', action='store_true')

    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument("--model_name", default='BioBert', type=str, help="model for text")
    parser.add_argument('--num_of_notes', help='Number of notes to include for a patient input 0 for all the notes', type=int, default=5)
    parser.add_argument('--notes_order', help='Should we get notes from beginning of the admission time or from end of it, options are: 1. First: pick first notes 2. Last: pick last notes', default=None)
    parser.add_argument('--ratio_notes_order', help='The parameter of a bernulli distribution on whether take notes from First or Last, 1-Last, 0-First',type=float, default=None)

    parser.add_argument('--bertcount',type=int, default=3,help='number of count update bert in total')
    parser.add_argument('--first_n_item', help='Top n item in val seeds', type=int, default=3)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--self_cross', action='store_true')
    parser.add_argument('--TS_mixup', action='store_true', help='mix up reg and irg data')
    parser.add_argument("--mixup_level", default=None, type=str, help="mixedup level for two time series data, choose: 'batch', batch_seq' or 'batch_seq_feature'. ")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--FTLSTM', action='store_true')
    parser.add_argument('--Interp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--datagereate_seed", type=int, default=42, help="A seed for reproducible data generation .")
    parser.add_argument("--TS_model", type=str, default='Atten', help="LSTM, CNN, Atten")

    parser.add_argument("--cross_method", default='moe', type=str, help="all fusion methods: moe, hme, moe_cross, self_cross, MAGGate, MulT, Outer, concat")
    parser.add_argument("--hidden_size", default=512, type=int, help="hidden size of MLP second layer")
    parser.add_argument("--gating_function", nargs='*', type=str, help="all gating functions: softmax, laplace, gaussian, enter at least one")
    parser.add_argument("--poly_power", default=None, type=float, help="power of polynomial gating")
    # parser.add_argument("--poly_powers", type=int, nargs="+", default=[2], help="powers of polynomial gating, only used when gating_function includes 'polynomial'")

    parser.add_argument("--student_degree", default=None, type=float, help="degree of student-t gating")

    parser.add_argument("--num_of_experts", nargs='*', type=int, help="number of MLPs in MoE, for HME need to specify each level")
    parser.add_argument("--top_k", nargs='*', type=int, help="the number of experts finally combined together for joint and permod routers")
    parser.add_argument("--disjoint_top_k", default=2, type=int, help="the number of experts finally combined together for disjoint routers")
    parser.add_argument("--num_modalities", default=2, type=int, help="the number of input modalities used to train transformer")
    parser.add_argument("--use_pt_text_embeddings", action='store_true', help="Option to use pre-extracted text embeddings")
    parser.add_argument("--router_type", default='joint', type=str, help="all router types: joint, permod, disjoint")
    parser.add_argument("--use_balance_loss", action='store_true', help="Whether to include balance_loss term in total loss (only for MoE/HME fusion methods)")
    parser.add_argument("--balance_loss_coef", default=0.01, type=float, help="Coefficient for balance_loss term in total loss")
    parser.add_argument("--noisy_gating", type=str2bool, default=True, help="Enable or disable noisy gating (True/False).")
    parser.add_argument("--normalized", type=str2bool, default=True, help="Enable or disable normalization (True/False).")
    parser.add_argument('--use_bias', type=str2bool, default=False, help='Add a learnable bias per expert in polynomial/student‑t gating')
    parser.add_argument('--router_bias_mode', default='mul', choices=['mul', 'add'], help='How --use_bias affects router logits. mul preserves old FuseMoE behavior; add is closer to DeepSeek-V3 selection bias.')
    parser.add_argument('--gate_normalization', default='selected', choices=['selected', 'full', 'full_renorm'], help='selected normalizes selected top-k logits. full uses full softmax scores before top-k. full_renorm renormalizes gathered full-softmax top-k weights.')
    parser.add_argument('--router_topk_mode', default='k_plus_1', choices=['k_plus_1', 'k'], help='k_plus_1 preserves Shazeer/FuseMoE noisy-load threshold. k uses exact top-k selection like DeepSeek when noisy load estimation is not needed.')
    parser.add_argument('--moe_mixing_space', default='logprob', choices=['logprob', 'linear'], help='logprob preserves FuseMoE log-space expert mixing. linear mixes ordinary hidden feature outputs like transformer MoE FFN experts.')
    parser.add_argument('--load_balance_mode', default='cv', choices=['cv', 'deepseek_aux'], help='cv preserves FuseMoE CV importance/load loss. deepseek_aux uses count/probability-style auxiliary routing loss.')
    parser.add_argument('--shared_experts', type=int, default=0, help='Number of shared experts (always active). Default 0 = no shared experts.')
    parser.add_argument('--enable_shared_expert', action='store_true', help='Alias for --shared_experts 1. Keeps the shared FFN always active.')
    parser.add_argument('--shared_expert_weight', default=1.0, type=float, help='Scale applied to the always-active shared FFN output before combining with routed experts.')
    parser.add_argument('--freeze_shared_ffn', action='store_true', help='Freeze shared FFN/shared expert parameters after initialization.')
    parser.add_argument('--use_temp', type=str2bool, default=False, help='Temperature parameter for softmax gating function.')
    parser.add_argument('--expert_type', default='mlp', choices=['mlp', 'lora', 'residual_lora'], help='Expert module type inside MoE. residual_lora is used with --staged_shared_lora.')
    parser.add_argument('--lora_rank', default=8, type=int, help='Rank of each LoRA expert delta.')
    parser.add_argument('--lora_alpha', default=16.0, type=float, help='LoRA scaling alpha; effective scale is alpha / rank.')
    parser.add_argument('--lora_dropout', default=0.0, type=float, help='Dropout applied before the LoRA expert delta.')
    parser.add_argument('--freeze_expert_base', type=str2bool, default=True, help='Freeze the base projection in LoRA experts and train only low-rank deltas.')
    parser.add_argument('--staged_shared_lora', action='store_true', help='Stage 1 trains only the shared FFN path; Stage 2 freezes it and trains router plus residual LoRA experts.')
    parser.add_argument('--staged_shared_lora_warmup_epochs', default=8, type=int, help='Number of initial epochs used for shared-FFN pretraining before residual LoRA routing.')
    parser.add_argument('--expert_orth_coef', default=0.0, type=float, help='Coefficient for routed expert diversity/orthogonality loss. Default 0 keeps old FuseMoE behavior.')
    parser.add_argument('--use_instruction_router', action='store_true', help='Use a task instruction embedding to condition the MoE router.')
    parser.add_argument('--router_instruction', default=None, type=str, help='Clinical/task instruction text used when --use_instruction_router is enabled. If unset, a default task/modality instruction is used.')
    parser.add_argument('--instruction_router_scale', default=1.0, type=float, help='Scale applied to instruction-derived router logits.')
    parser.add_argument(
        '--instruction_router_fusion',
        default='logit_bias',
        choices=['logit_bias', 'input_add', 'both'],
        help='How to fuse instruction into the router: logit_bias adds instruction-derived logits; input_add adds a projected instruction vector to router input; both uses both.'
    )
    parser.add_argument('--use_semantic_expert_profiles', action='store_true', help='Use fixed clinical expert profile embeddings as semantic router anchors.')
    parser.add_argument('--use_semantic_logit_bias', action='store_true', help='Mentor-style semantic routing: add cosine profile logits as a soft bias to learned router logits.')
    parser.add_argument('--semantic_bias_scale', default=None, type=float, help='Alias for --semantic_profile_scale when using semantic logit bias.')
    parser.add_argument('--semantic_project_dim', default=None, type=int, help='Router-space dimension for semantic cosine projection W_sem x.')
    parser.add_argument('--semantic_only_router', action='store_true', help='Ablation: replace learned router logits with semantic logits only.')
    parser.add_argument(
        '--semantic_profile_embedding_source',
        default='biolongformer',
        choices=['biolongformer', 'random'],
        help='How to initialize fixed semantic expert profile embeddings.'
    )
    parser.add_argument(
        '--semantic_profile_set',
        default='icu_organ_system',
        choices=['icu_organ_system'],
        help='Built-in expert profile set used when --use_semantic_expert_profiles is enabled.'
    )
    parser.add_argument('--semantic_profile_scale', default=1.0, type=float, help='Scale applied to semantic profile router logits.')
    parser.add_argument(
        '--semantic_profile_fusion',
        default='add',
        choices=['add', 'replace'],
        help='How semantic profile logits affect router logits: add combines with learned router logits; replace uses semantic logits only.'
    )
    parser.add_argument(
        '--semantic_profile_source',
        default='patient',
        choices=['patient', 'note'],
        help='patient uses the MoE patient/modality representation; note uses note-level text embeddings to produce semantic profile logits.'
    )
    parser.add_argument(
        '--semantic_profile_note_pooling',
        default='max',
        choices=['max', 'mean'],
        help='How to pool note-level profile similarities into patient-level profile logits.'
    )
    parser.add_argument(
        '--semantic_profile_modalities',
        nargs='*',
        default=['txt'],
        help='For per-modality routers with --semantic_profile_source note, modalities that receive note-derived semantic logits.'
    )
    parser.add_argument(
        '--semantic_profile_layers',
        default='all',
        choices=['all', 'first'],
        help='MoE layers that receive semantic profile router bias. first applies it only to layer 0.'
    )
    parser.add_argument('--use_prototype_router', action='store_true', help='Use learnable router-space expert prototypes instead of the linear router matrix.')
    parser.add_argument('--prototype_router_dim', default=256, type=int, help='Latent dimension for learnable prototype routing.')
    parser.add_argument('--prototype_router_temperature', default=1.0, type=float, help='Temperature used by prototype router softmax/top-k gates.')
    parser.add_argument('--prototype_router_dense', action='store_true', help='Use dense prototype routing over all experts instead of top-k masking.')
    parser.add_argument('--prototype_router_orth_coef', default=0.0, type=float, help='Coefficient for prototype orthogonality regularization inside MoE balance loss.')
    parser.add_argument('--use_router_organ_supervision', action='store_true', help='Use weak note-derived organ labels to supervise router gate mass.')
    parser.add_argument('--router_organ_supervision_coef', default=1.0, type=float, help='Coefficient for weak organ-router supervision inside MoE auxiliary loss.')
    parser.add_argument('--router_organ_supervision_layers', default='all', choices=['all', 'first'], help='MoE layers that receive weak organ-router supervision.')
    parser.add_argument('--router_organ_supervision_class_balanced', action='store_true', help='Apply inverse-frequency batch balancing to weak organ-router targets.')
    parser.add_argument('--use_missing_modality_recon', action='store_true', help='Add embedding-level missing-modality reconstruction loss from observed modality embeddings.')
    parser.add_argument('--use_learned_missing_embeddings', action='store_true', help='Replace zero-filled missing modality representations with learned per-modality missing embeddings.')
    parser.add_argument('--missing_modality_recon_coef', default=0.1, type=float, help='Coefficient for embedding-level missing-modality reconstruction loss.')
    parser.add_argument('--missing_modality_recon_targets', default='cxr,ecg', type=str, help='Comma-separated modality targets reconstructed from the other observed modalities, e.g. cxr,ecg,text.')
    parser.add_argument('--missing_modality_recon_hidden', default=256, type=int, help='Hidden size for lightweight modality reconstruction heads.')
    parser.add_argument('--router_z_loss_coef', default=0.0, type=float, help='Coefficient for router z-loss inside the MoE auxiliary loss. Default 0 keeps old behavior.')
    parser.add_argument('--z_loss_weight', default=None, type=float, help='Alias for --router_z_loss_coef.')
    parser.add_argument('--router_z_loss_type', default='logsumexp', choices=['logsumexp', 'squared_logits'], help='Router z-loss variant.')
    parser.add_argument('--router_entropy_coef', default=0.0, type=float, help='Coefficient for negative router entropy regularization inside the MoE auxiliary loss. Positive values encourage softer gates.')
    parser.add_argument('--router_variance_coef', default=0.0, type=float, help='Coefficient for Advancing Expert Specialization routing variance loss. Positive values encourage routing scores to vary across samples.')
    parser.add_argument('--router_variance_loss_weight', default=None, type=float, help='Alias for --router_variance_coef.')
    parser.add_argument('--output_orth_coef', default=0.0, type=float, help='Coefficient for selected expert output orthogonality loss. Positive values discourage selected experts from producing overlapping outputs.')
    parser.add_argument('--orthogonal_loss_weight', default=None, type=float, help='Alias for --output_orth_coef.')
    parser.add_argument('--specialization_loss_mode', default='legacy', choices=['legacy', 'paper'], help='legacy keeps old MoE auxiliary scaling; paper uses alpha*Laux + beta*Lo + gamma*Lv from Advancing Expert Specialization directly.')
    parser.add_argument('--specialization_aux_coef', default=1e-3, type=float, help='Alpha coefficient for Laux when --specialization_loss_mode paper.')
    parser.add_argument('--dense_warmup_epochs', default=0, type=int, help='Use dense all-expert routing for the first N training epochs before switching back to top-k. Default 0 keeps old behavior.')
    parser.add_argument('--router_temperature', default=1.0, type=float, help='Fixed router temperature applied to softmax/sigmoid logits when --use_temp is false. Values >1 soften routing.')
    parser.add_argument('--router_noise_scale', default=1.0, type=float, help='Multiplier for standard noisy top-k router noise. Values below 1 reduce random exploration.')
    parser.add_argument('--router_noise_final_scale', default=None, type=float, help='If set, linearly decay standard router noise from --router_noise_scale to this value.')
    parser.add_argument('--router_noise_decay_epochs', default=0, type=int, help='Epochs over which standard router noise decays from initial to final scale.')
    parser.add_argument('--router_init', default='zero', choices=['zero', 'normal', 'xavier', 'kaiming'], help='Initialization for non-XMoE router w_gate. zero keeps original FuseMoE; random modes test DeepSeek-style non-tied router starts.')
    parser.add_argument('--router_init_std', default=0.02, type=float, help='Stddev for --router_init normal.')
    parser.add_argument('--use_xmoe_router', action='store_true', help='Use X-MOE-style low-dimensional L2-normalized hypersphere routing while keeping full features for expert inputs.')
    parser.add_argument('--xmoe_router_dim', default=128, type=int, help='Low-dimensional routing space for --use_xmoe_router.')
    parser.add_argument('--xmoe_router_init_norm', default=0.1, type=float, help='Initial/fixed norm scale for X-MOE expert embeddings before L2-normalized scoring.')
    parser.add_argument('--xmoe_noise_scale', default=1.0, type=float, help='Multiplier for X-MOE router noise stddev. Lower values reduce near-uniform random exploration.')
    parser.add_argument('--log_expert_output_diagnostics', action='store_true', help='Print expert output similarity and routed cohort diagnostics for MoE layers.')
    parser.add_argument('--log_router_diagnostics', action='store_true', help='Write per-sample router top-k diagnostics during test evaluation.')
    parser.add_argument('--router_diagnostics_path', default=None, type=str, help='CSV path for router diagnostics. Defaults to output_dir/router_diagnostics_test.csv.')
    parser.add_argument('--router_diagnostics_layers', default='last', choices=['last', 'all'], help='Write only the last MoE layer or all MoE layers in router diagnostics CSV.')
    parser.add_argument('--router_diagnostics_max_text_chars', default=500, type=int, help='Maximum raw note characters stored per sample in router diagnostics CSV.')
    parser.add_argument('--router_print_mode', default='verbose', choices=['verbose', 'concise', 'none'], help='Router usage print style: verbose keeps existing lines, concise prints one compact line per split, none suppresses router usage prints.')
    parser.add_argument('--lingshu_model_path', default=None, type=str, help='Path or Hugging Face id for Lingshu/Qwen-VL backbone used by the separate Lingshu pseudo-token model.')
    parser.add_argument('--lingshu_trust_remote_code', type=str2bool, default=True, help='Pass trust_remote_code to the Lingshu backbone loader.')
    parser.add_argument('--lingshu_freeze_backbone', type=str2bool, default=True, help='Freeze Lingshu backbone parameters before attaching/trainining LoRA adapters.')
    parser.add_argument('--lingshu_use_peft_lora', type=str2bool, default=True, help='Use PEFT LoRA adapters on the Lingshu backbone if peft is installed.')
    parser.add_argument('--lingshu_lora_r', default=8, type=int, help='PEFT LoRA rank for Lingshu adapters.')
    parser.add_argument('--lingshu_lora_alpha', default=16, type=int, help='PEFT LoRA alpha for Lingshu adapters.')
    parser.add_argument('--lingshu_lora_dropout', default=0.05, type=float, help='PEFT LoRA dropout for Lingshu adapters.')
    parser.add_argument('--lingshu_lora_target_modules', nargs='*', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'], help='Target module names for PEFT LoRA on the Lingshu backbone.')
    parser.add_argument('--lingshu_max_ts_tokens', default=48, type=int, help='Maximum time-series pseudo tokens passed to Lingshu.')
    parser.add_argument('--lingshu_pooling', default='mean', choices=['mean', 'last'], help='Pooling strategy over Lingshu hidden states for classification.')
    parser.add_argument('--lingshu_architecture', default='pseudotoken', choices=['pseudotoken', 'organ_lora_moe'], help='Lingshu model variant. pseudotoken uses one shared LoRA adapter; organ_lora_moe uses organ-specific LoRA adapters as experts.')
    parser.add_argument('--lingshu_moe_router', default='semantic_profile', choices=['semantic_profile', 'linear'], help='Router used by --lingshu_architecture organ_lora_moe.')
    parser.add_argument('--lingshu_moe_top_k', default=2, type=int, help='Number of organ LoRA experts selected by the Lingshu MoE router.')
    parser.add_argument('--lingshu_moe_expert_names', nargs='*', default=['cardiovascular', 'respiratory', 'renal_metabolic', 'neurological'], help='Names for organ-specific Lingshu LoRA expert adapters.')
    parser.add_argument('--lingshu_moe_temperature', default=1.0, type=float, help='Temperature for Lingshu organ expert routing.')

    args = parser.parse_args()

    if args.enable_shared_expert:
        args.shared_experts = max(args.shared_experts, 1)

    if args.use_semantic_logit_bias:
        args.use_semantic_expert_profiles = True
        args.semantic_profile_fusion = "add"
    if args.semantic_only_router:
        args.use_semantic_expert_profiles = True
        args.semantic_profile_fusion = "replace"
    if args.semantic_bias_scale is not None:
        args.semantic_profile_scale = args.semantic_bias_scale

    if args.z_loss_weight is not None:
        args.router_z_loss_coef = args.z_loss_weight
    if args.router_variance_loss_weight is not None:
        args.router_variance_coef = args.router_variance_loss_weight
    if args.orthogonal_loss_weight is not None:
        args.output_orth_coef = args.orthogonal_loss_weight
    return args

def loadBert(args,device):
    if args.model_name!=None:
        if args.model_name== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            BioBert=AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.model_name=="bioRoberta":
            config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        elif args.model_name== "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")
        elif args.model_name== "bioLongformer":
            tok_path = os.getenv("CLIN_LONGFORMER_DIR", "yikuan8/Clinical-Longformer")
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            BioBert= AutoModel.from_pretrained("yikuan8/Clinical-Longformer")

        else:
            raise ValueError("model_name should be BioBert,bioRoberta,bioLongformer or Bert")
    else:
        if args.model_path!=None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            BioBert = AutoModel.from_pretrained(args.model_path)
        else:
            raise ValueError("provide either model_name or model_path")

    BioBert = BioBert.to(device)
    BioBertConfig = BioBert.config
    return BioBert, BioBertConfig,tokenizer


def data_generate(args):
    dataPath = os.path.join(args.file_path,  'all_data_p2x_data.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if args.debug:
                data=data[:100]

    data=np.array(data)
    total_num=len(data)
    idx=np.arange(total_num)

    np.random.seed(args.seed)
    np.random.shuffle(idx)

    train= data[idx[:int(len(idx)*0.8)]]
    print(train[0]['data_names'])
    val=data[idx[int(len(idx)*0.8):int(len(idx)*0.9)]]
    test=data[idx[int(len(idx)*0.9):]]

    train=train.tolist()
    val=val.tolist()
    test=test.tolist()
    return train, val, test


def metrics_multilabel(y_true, predictions, verbose=1):
    # import pdb; pdb.set_trace()
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return{"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


def diff_float(time1, time2):
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0


def get_time_to_end_diffs(times, starttimes):

    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')
        for t in times:
            time = np.datetime64(t)
            dt = diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def change_data_form(file_path,mode,debug=False):
    dataPath = os.path.join(file_path, mode + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            # (data, _, _, _) = pickle.load(f)
            data = pickle.load(f)
            if debug:
                data=data[:500]

        data_X = data[0]
        data_y = data[1]
        data_text = data[2]
        data_names = data[3]
        start_times = data[4]
        timetoends = data[5]

        dataList=[]

        assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 


        assert  len(data_text[0])==len(timetoends[0])
        for x,y, text, name, start, end in zip(data_X,data_y,data_text, data_names,start_times,timetoends):
            if len(text)==0:
                continue
            new_text=[]
            for t in text:
                # import pdb;
                # pdb.set_trace()
                t=re.sub(r'\s([,;?.!:%"](?:\s|$))', r'\1', t)
                t=re.sub(r"\b\s+'\b", r"'", t)
                new_text.append(t.lower().strip())


            data_detail={"data_names":name,
                         "TS_data":x,
                         "text_data":new_text,
                        "label":y,
                         "adm_time":start,
                         "text_time_to_end":end
                        }
            dataList.append(data_detail)

    os.makedirs('Data',exist_ok=True)
    dataPath2 = os.path.join(file_path, mode + 'p2x_data.pkl')

    with open(dataPath2, 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump(dataList, f)

    return dataList

def data_replace(file_path1,file_path2,mode,debug=False):
    dataPath1 = os.path.join(file_path2, mode + '.pkl')
    dataPath2 = os.path.join(file_path1, mode + 'p2x_data.pkl')
    if os.path.isfile(dataPath1):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath1)
        with open(dataPath1, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data=data[:500]

    with open(dataPath2, 'rb') as f:
            data_r=pickle.load(f)
    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    data_names = data[3]
    start_times = data[4]
    timetoends = data[5]
    data_dict={}

    assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 
    assert  len(data_text[0])==len(timetoends[0])
    for x,name in zip(data_X, data_names):

        data_dict[name]=x
    for idx, data_detail in enumerate(data_r):
        new_x=data_dict[data_detail['data_names']]
        data_detail['TS_data']=new_x

    dataPath3=os.path.join(file_path2, mode + 'p2x_data.pkl')
    with open(dataPath3, 'wb') as f:
        pickle.dump(data_r, f)


def merge_reg_irg(dataPath_reg, dataPath_irg):
    with open(dataPath_irg, 'rb') as f:
        data_irg=pickle.load(f)

    with open(dataPath_reg, 'rb') as f:
        data_reg=pickle.load(f)

    for idx, data_dict in enumerate(data_reg):
        irg_dict=data_irg[data_dict['data_names']]
        data_dict['ts_tt']=irg_dict['ts_tt']
        data_dict['irg_ts']=irg_dict['irg_ts']
        data_dict['irg_ts_mask']=irg_dict['irg_ts_mask']

        assert (data_dict['label']==irg_dict['label']).all()

    with open(dataPath_reg, 'wb') as f:
        pickle.dump(data_reg,f)
