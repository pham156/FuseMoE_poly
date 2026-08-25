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
import argparse
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


def normalize_gating_function_arg(gating_function):
    if gating_function is None:
        return None
    if isinstance(gating_function, str):
        return [gating_function]
    return list(gating_function)


def primary_gating_function(args_or_gating):
    gating_values = getattr(args_or_gating, "gating_function", args_or_gating)
    gating_values = normalize_gating_function_arg(gating_values)
    if not gating_values:
        return None
    return gating_values[0]


def gating_descriptor(args_or_gating):
    gating_values = getattr(args_or_gating, "gating_function", args_or_gating)
    gating_values = normalize_gating_function_arg(gating_values)
    if not gating_values:
        return "nogate"
    return "_".join(str(value) for value in gating_values)


def build_run_name(args, prefix="fusemoe_new"):
    run_name = f"{prefix}_{args.cross_method}_{args.modeltype}"
    gating_name = primary_gating_function(args)
    gating_values = normalize_gating_function_arg(getattr(args, "gating_function", None))

    if args.cross_method == "hme":
        gating_label = gating_descriptor(args)
        run_name = f"{prefix}_{gating_label}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"
        if gating_values and "poly" in gating_values:
            run_name = f"{prefix}_{gating_label}_{args.poly_power}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"
        elif gating_values and "student_t" in gating_values:
            run_name = f"{prefix}_{gating_label}_{args.student_degree}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"
    elif args.cross_method == "moe" and gating_name is not None:
        run_name = f"{prefix}_{gating_name}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"
        if gating_values and "poly" in gating_values:
            run_name = f"{prefix}_{gating_name}_{args.poly_power}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"
        elif gating_values and "student_t" in gating_values:
            run_name = f"{prefix}_{gating_name}_{args.student_degree}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"

    noise_tag = "noise" if args.noisy_gating else "clean"
    return f"{run_name}_{noise_tag}"


def _apply_alias_args(args):
    if args.teacher_sweep or args.teacher_target_path or args.teacher_model != 'none':
        if args.teacher_target_path and not args.expert_init_target_path:
            args.expert_init_target_path = args.teacher_target_path
        if args.teacher_output_dir and args.output_dir == "Checkpoints":
            args.output_dir = args.teacher_output_dir
        if args.teacher_soft_targets and not args.expert_init_soft_targets:
            args.expert_init_soft_targets = True
        if args.teacher_confidence_threshold and args.expert_init_confidence_threshold == 0.0:
            args.expert_init_confidence_threshold = args.teacher_confidence_threshold
        if args.teacher_low_confidence_policy == "skip":
            args.expert_init_skip_low_confidence = True
        elif args.teacher_low_confidence_policy == "uniform":
            args.expert_init_uniform_fallback = True

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


def _validate_overlapping_args(args, parser):
    missing_fill_flags = [
        ("--use_learned_missing_embeddings", bool(args.use_learned_missing_embeddings)),
        ("--use_missing_modality_proxies", bool(args.use_missing_modality_proxies)),
        ("--use_cross_modal_missing_proxies", bool(args.use_cross_modal_missing_proxies)),
        ("--use_maestro_tokens", bool(args.use_maestro_tokens)),
    ]
    enabled_missing_fill_flags = [name for name, enabled in missing_fill_flags if enabled]
    if len(enabled_missing_fill_flags) > 1:
        parser.error(
            "Missing-modality fill options overlap. Choose only one of "
            + ", ".join(enabled_missing_fill_flags)
            + "."
        )

    if args.interaction_router_only and not args.use_interaction_router:
        parser.error("--interaction_router_only requires --use_interaction_router.")
    if args.use_interaction_experts:
        if args.router_type != "permod":
            parser.error("--use_interaction_experts currently requires --router_type permod.")
        if int(args.num_of_experts[0]) < 4:
            parser.error("--use_interaction_experts requires at least 4 experts.")
    if args.use_interaction_expert_reweighting:
        if not args.use_interaction_experts:
            parser.error("--use_interaction_expert_reweighting requires --use_interaction_experts.")
        if args.router_type != "permod":
            parser.error("--use_interaction_expert_reweighting currently requires --router_type permod.")

    if args.use_multihead_permod_router:
        if args.router_type != "permod":
            parser.error("--use_multihead_permod_router requires --router_type permod.")
        if args.multihead_router_heads < 2:
            parser.error("--multihead_router_heads must be >= 2 when --use_multihead_permod_router is enabled.")
    if args.use_moh_attention_head_experts:
        if args.moh_head_expert_top_k < 0:
            parser.error("--moh_head_expert_top_k must be >= 0.")
        if args.moh_head_expert_top_k > args.num_heads:
            parser.error("--moh_head_expert_top_k cannot exceed --num_heads.")
        if args.moh_head_expert_temperature <= 0:
            parser.error("--moh_head_expert_temperature must be positive.")
    if args.use_dynamic_top_k:
        if args.dynamic_top_k_min < 1:
            parser.error("--dynamic_top_k_min must be >= 1.")
        if args.dynamic_top_k_max < args.dynamic_top_k_min:
            parser.error("--dynamic_top_k_max must be >= --dynamic_top_k_min.")
        if args.dynamic_top_k_max > int(args.num_of_experts[0]):
            parser.error("--dynamic_top_k_max cannot exceed --num_of_experts.")

    gating_values = normalize_gating_function_arg(args.gating_function)
    if args.cross_method == "moe" and gating_values and len(gating_values) != 1:
        parser.error("--cross_method moe expects exactly one --gating_function value.")
    if args.cross_method == "hme" and gating_values and len(gating_values) != 2:
        parser.error("--cross_method hme expects exactly two --gating_function values.")


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
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path used for eval-only mode or explicit checkpoint reload.")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate in --mode eval.")
    parser.add_argument("--eval_force_missing_modalities", type=str, default="", help="Comma-separated modalities to force missing at eval time. Supported: text,cxr,ecg.")
    parser.add_argument("--eval_ablate_experts", type=str, default="", help="Comma-separated expert indices to ablate at eval time.")
    parser.add_argument("--eval_ablate_shared_path", action="store_true", help="Zero the shared-expert path at eval time.")
    parser.add_argument("--eval_ablate_routed_path", action="store_true", help="Zero the routed expert path at eval time.")
    parser.add_argument("--log_per_sample_contributions", action="store_true", help="Write per-sample shared/routed contribution fields into router diagnostics CSV.")
    parser.add_argument("--use_r2t2_rerouting", action="store_true", help="Build a validation router-state bank after training and reroute test gates by nearest validation router states.")
    parser.add_argument("--r2t2_num_neighbors", type=int, default=32, help="Number of validation router neighbors used by --use_r2t2_rerouting.")
    parser.add_argument("--r2t2_blend", type=float, default=0.5, help="Blend weight for validation-neighbor gates in --use_r2t2_rerouting.")
    parser.add_argument("--r2t2_kernel_sigma", type=float, default=1.0, help="RBF kernel sigma for validation-neighbor weighting in --use_r2t2_rerouting.")
    parser.add_argument("--r2t2_correct_reference_only", action="store_true", help="Restrict the R2T2 validation bank to samples predicted correctly by the selected model.")
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
    parser.add_argument("--use_ts_variable_tokens", action='store_true', help="Replace the default irregular TS attention input with variable-aware time-variable tokens.")
    parser.add_argument("--ts_variable_token_layers", default=1, type=int, help="Transformer layers for the variable-token irregular TS encoder.")
    parser.add_argument("--ts_variable_token_heads", default=4, type=int, help="Attention heads for the variable-token irregular TS encoder.")
    parser.add_argument("--use_ts_patch_encoder", action='store_true', help="Replace the default regular TS Conv1d branch with a patch-based encoder.")
    parser.add_argument("--ts_patch_size", default=4, type=int, help="Temporal patch size for the regular TS patch encoder.")
    parser.add_argument("--ts_patch_layers", default=1, type=int, help="Transformer layers for the regular TS patch encoder.")
    parser.add_argument("--ts_patch_mask_ratio", default=0.3, type=float, help="Mask ratio used by TS patch reconstruction when enabled.")
    parser.add_argument("--ts_patch_masking_mode", default="patch", choices=["patch", "feature"], help="Mask whole TS patches or feature-time elements inside each patch during TS auxiliary reconstruction.")
    parser.add_argument("--ts_patch_recon_weight", default=0.0, type=float, help="Auxiliary loss weight for masked TS patch reconstruction.")
    parser.add_argument("--use_ts_state_space_encoder", action='store_true', help="Replace the regular TS Conv1d branch with a lightweight state-space-style encoder.")
    parser.add_argument("--ts_state_space_layers", default=1, type=int, help="Number of lightweight state-space layers for the regular TS encoder.")
    parser.add_argument("--use_ts_shared_private", action='store_true', help="Factor TS representations into shared and private components with auxiliary regularization.")
    parser.add_argument("--ts_shared_private_weight", default=0.05, type=float, help="Total auxiliary weight for the TS shared/private decomposition losses.")
    parser.add_argument("--ts_shared_private_ortho_coef", default=1.0, type=float, help="Relative weight for shared/private orthogonality inside the TS decomposition loss.")
    parser.add_argument("--ts_shared_private_align_coef", default=1.0, type=float, help="Relative weight for aligning the TS shared component with observed non-TS modalities.")
    parser.add_argument("--use_ts_confidence_fusion", action='store_true', help="Scale the fused TS stream by a learned confidence score derived from TS coverage and embedding summaries.")
    parser.add_argument("--ts_confidence_hidden", default=64, type=int, help="Hidden size for the TS confidence head.")
    parser.add_argument("--use_ts_cross_modal_distill", action='store_true', help="Distill pooled text/CXR/ECG information into the TS representation.")
    parser.add_argument("--ts_cross_modal_distill_weight", default=0.05, type=float, help="Auxiliary weight for cross-modal distillation into TS.")
    parser.add_argument("--use_maestro_tokens", action='store_true', help="Apply MAESTRO-style missingness-aware cross-modal modality tokens before the MoE fusion block.")
    parser.add_argument("--maestro_token_layers", default=1, type=int, help="Transformer layers in the MAESTRO-style modality-token block.")
    parser.add_argument("--maestro_token_heads", default=4, type=int, help="Attention heads in the MAESTRO-style modality-token block.")
    parser.add_argument("--maestro_missing_token_mode", default="learned", choices=["learned"], help="Missing-modality token parameterization for the MAESTRO-style block.")
    parser.add_argument("--maestro_use_cross_modal_encoder", type=str2bool, default=True, help="Use cross-modal self-attention inside the MAESTRO-style token block. False tests missing-token encoding only.")
    parser.add_argument("--maestro_missing_init_std", default=0.02, type=float, help="Initialization std for learned MAESTRO missing-modality tokens.")
    parser.add_argument("--maestro_sparse_topk", default=2, type=int, help="Number of modality-token residual updates kept per sample. Use 0 to keep all updates.")
    parser.add_argument("--maestro_residual_weight", default=1.0, type=float, help="Scale for broadcasting MAESTRO token residuals back to modality sequences.")
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
    parser.add_argument('--semantic_profile_shuffle_assignments', action='store_true', help='Shuffle the mapping between semantic profile embeddings and expert ids while keeping profile texts fixed.')
    parser.add_argument(
        '--semantic_profile_set',
        default='icu_organ_system',
        choices=['icu_organ_system', 'random_clinical_control'],
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
    parser.add_argument('--use_missing_modality_proxies', action='store_true', help='Replace zero-filled missing modality representations with learned per-modality proxy token sequences.')
    parser.add_argument('--missing_modality_proxy_init_std', default=0.02, type=float, help='Initialization std for learned missing-modality proxy token sequences.')
    parser.add_argument('--use_cross_modal_missing_proxies', action='store_true', help='Predict missing modality token sequences from the observed modality embeddings instead of using static zero or learned proxy fills.')
    parser.add_argument('--cross_modal_proxy_hidden', default=256, type=int, help='Hidden size for cross-modal missing-proxy predictors.')
    parser.add_argument('--cross_modal_proxy_dropout', default=0.1, type=float, help='Dropout used inside cross-modal missing-proxy predictors.')
    parser.add_argument('--missing_modality_recon_coef', default=0.1, type=float, help='Coefficient for embedding-level missing-modality reconstruction loss.')
    parser.add_argument('--missing_modality_recon_targets', default='cxr,ecg', type=str, help='Comma-separated modality targets reconstructed from the other observed modalities, e.g. cxr,ecg,text.')
    parser.add_argument('--missing_modality_recon_hidden', default=256, type=int, help='Hidden size for lightweight modality reconstruction heads.')
    parser.add_argument('--train_cxr_modality_dropout', default=0.0, type=float, help='Training-only probability of marking observed CXR as missing.')
    parser.add_argument('--train_text_modality_dropout', default=0.0, type=float, help='Training-only probability of marking observed text as missing.')
    parser.add_argument('--train_ecg_modality_dropout', default=0.0, type=float, help='Training-only probability of marking observed ECG as missing.')
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
    parser.add_argument('--poly_noise_mode', default='distance', choices=['distance', 'score'], help='For polynomial gating only: distance preserves old behavior by adding noise before the kernel; score adds noise directly to the polynomial scores before top-k normalization.')
    parser.add_argument('--router_init', default='zero', choices=['zero', 'normal', 'xavier', 'kaiming'], help='Initialization for non-XMoE router w_gate. zero keeps original FuseMoE; random modes test DeepSeek-style non-tied router starts.')
    parser.add_argument('--router_init_std', default=0.02, type=float, help='Stddev for --router_init normal.')
    parser.add_argument('--use_dynamic_top_k', action='store_true', help='Choose sparse router top-k per sample from confidence/entropy instead of using fixed --top_k.')
    parser.add_argument('--dynamic_top_k_min', default=1, type=int, help='Minimum per-sample k when --use_dynamic_top_k is enabled.')
    parser.add_argument('--dynamic_top_k_max', default=3, type=int, help='Maximum per-sample k when --use_dynamic_top_k is enabled.')
    parser.add_argument('--dynamic_top_k_confidence_threshold', default=0.7, type=float, help='Use minimum k when full-router top-1 probability is at least this value.')
    parser.add_argument('--dynamic_top_k_entropy_threshold', default=0.8, type=float, help='Use maximum k when normalized full-router entropy is at least this value.')
    parser.add_argument('--use_multihead_permod_router', action='store_true', help='Use multiple independent router heads per modality for router_type=permod, then fuse head logits before top-k gating.')
    parser.add_argument('--multihead_router_heads', default=4, type=int, help='Number of heads per modality when --use_multihead_permod_router is enabled.')
    parser.add_argument('--multihead_router_fusion', default='mean', choices=['mean', 'learned'], help='How to fuse per-head router logits before top-k gating.')
    parser.add_argument('--use_moh_attention_head_experts', action='store_true', help='Use MoH-style attention heads as gated experts inside the cross-modal attention blocks.')
    parser.add_argument('--moh_head_expert_top_k', default=0, type=int, help='Sparse top-k attention heads to keep when --use_moh_attention_head_experts is enabled. 0 uses all heads with soft weights.')
    parser.add_argument('--moh_head_expert_temperature', default=1.0, type=float, help='Softmax temperature for MoH-style attention-head expert gates.')
    parser.add_argument('--use_xmoe_router', action='store_true', help='Use X-MOE-style low-dimensional L2-normalized hypersphere routing while keeping full features for expert inputs.')
    parser.add_argument('--xmoe_router_dim', default=128, type=int, help='Low-dimensional routing space for --use_xmoe_router.')
    parser.add_argument('--xmoe_router_init_norm', default=0.1, type=float, help='Initial/fixed norm scale for X-MOE expert embeddings before L2-normalized scoring.')
    parser.add_argument('--xmoe_noise_scale', default=1.0, type=float, help='Multiplier for X-MOE router noise stddev. Lower values reduce near-uniform random exploration.')
    parser.add_argument('--log_expert_output_diagnostics', action='store_true', help='Print expert output similarity and routed cohort diagnostics for MoE layers.')
    parser.add_argument('--log_router_diagnostics', action='store_true', help='Write per-sample router top-k diagnostics during test evaluation.')
    parser.add_argument('--router_diagnostics_path', default=None, type=str, help='CSV path for router diagnostics. Defaults to output_dir/router_diagnostics_test.csv.')
    parser.add_argument('--router_diagnostics_layers', default='last', choices=['last', 'all'], help='Write only the last MoE layer or all MoE layers in router diagnostics CSV.')
    parser.add_argument('--router_diagnostics_max_text_chars', default=500, type=int, help='Maximum raw note characters stored per sample in router diagnostics CSV.')
    parser.add_argument('--log_router_geometry', action='store_true', help='Add router input vectors and clean router logits to router diagnostics CSV for geometry-vs-routing analysis.')
    parser.add_argument('--router_geometry_max_dims', default=256, type=int, help='Maximum router input dimensions per modality written when --log_router_geometry is enabled.')
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
    parser.add_argument('--expert_init_strategy', default='none', choices=['none', 'fixed_cohort'], help='Optional Week30 expert initialization strategy.')
    parser.add_argument('--expert_init_target_path', default=None, type=str, help='Path to expert-init target CSV, or a template containing {split}.')
    parser.add_argument('--expert_init_epochs', default=0, type=int, help='Number of warm-start epochs using fixed cohort routing.')
    parser.add_argument('--expert_init_soft_targets', action='store_true', help='Treat target CSV rows as soft cohort weights instead of one-hot assignments.')
    parser.add_argument('--expert_init_freeze_router', action='store_true', help='Freeze MoE router parameters while expert-init warm-start is active.')
    parser.add_argument('--expert_init_freeze_backbone', action='store_true', help='Freeze non-MoE backbone parameters while expert-init warm-start is active.')
    parser.add_argument('--expert_init_release_schedule', default='hard', choices=['hard', 'linear'], help='How to release fixed cohort routing back to learned routing.')
    parser.add_argument('--expert_init_confidence_threshold', default=0.0, type=float, help='Minimum target confidence required to apply cohort warm-start for a sample.')
    parser.add_argument('--expert_init_uniform_fallback', action='store_true', help='Use uniform expert targets when a sample has no valid expert-init target.')
    parser.add_argument('--expert_init_skip_low_confidence', action='store_true', help='Ignore low-confidence target rows instead of routing them uniformly.')
    parser.add_argument('--teacher_sweep', action='store_true', help='Tag a run as part of the Week30 teacher-model sweep.')
    parser.add_argument('--teacher_model', default='none', choices=['none', 'random', 'qwen25', 'biomistral', 'meditron', 'me_llama', 'medgemma', 'lingshu'], help='Teacher arm identifier for Week30 teacher-sweep metadata.')
    parser.add_argument('--teacher_target_path', default=None, type=str, help='Optional alias for --expert_init_target_path used by teacher-sweep manifests.')
    parser.add_argument('--teacher_output_dir', default=None, type=str, help='Optional path for exported teacher targets or compact diagnostics.')
    parser.add_argument('--teacher_num_experts', default=4, type=int, help='Number of teacher-induced expert cohorts.')
    parser.add_argument('--teacher_soft_targets', action='store_true', help='Alias for soft cohort targets in teacher-sweep manifests.')
    parser.add_argument('--teacher_confidence_threshold', default=0.0, type=float, help='Minimum teacher confidence used by teacher-target exporters.')
    parser.add_argument('--teacher_low_confidence_policy', default='uniform', choices=['skip', 'uniform'], help='Fallback policy for low-confidence teacher targets.')
    parser.add_argument('--teacher_freeze_backbone', action='store_true', help='Freeze teacher backbones inside target-generation utilities when supported.')
    parser.add_argument('--teacher_use_lora', action='store_true', help='Enable optional LoRA adapters in teacher target-generation utilities.')
    parser.add_argument('--teacher_lora_rank', default=8, type=int, help='Optional LoRA rank for teacher target-generation utilities.')
    parser.add_argument('--teacher_max_samples', default=None, type=int, help='Optional cap used for teacher-target smoke generation.')
    parser.add_argument('--teacher_cache_embeddings', action='store_true', help='Cache intermediate teacher embeddings during target export when supported.')
    parser.add_argument('--log_cohort_retention', action='store_true', help='Explicitly tag that cohort-retention diagnostics should be captured for this run.')
    parser.add_argument('--use_task_condition_router', action='store_true', help='Week30 Family A1: inject a learned task embedding into the MoE router input.')
    parser.add_argument('--task_condition_stats', action='store_true', help='Week30 Family A: write task-conditioned routing fields for later aggregation.')
    parser.add_argument('--task_router_dim', default=128, type=int, help='Embedding dimension for the learned task-conditioning vector.')
    parser.add_argument('--use_task_specific_router_heads', action='store_true', help='Use separate router heads for PHENO, IHM, and LOS while keeping a shared expert pool.')
    parser.add_argument('--use_task_condition_expert_modulation', action='store_true', help='Week30 Family A3: modulate expert hidden/output states with learned task embeddings.')
    parser.add_argument('--task_expert_modulation_type', default='film', choices=['film', 'bias'], help='How task embeddings modulate expert outputs when --use_task_condition_expert_modulation is enabled.')
    parser.add_argument('--multitask_shared_moe_trunk', action='store_true', help='Week30 Family A4: train a shared MoE trunk jointly across IHM, LOS, and PHENO with separate prediction heads.')
    parser.add_argument('--multitask_tasks', default='ihm-48-cxr-notes-ecg,los-48-cxr-notes-ecg,pheno-all-cxr-notes-ecg', type=str, help='Comma-separated task list for Week30 Family A4 multitask runs.')
    parser.add_argument('--multitask_primary_metrics', default='ihm=f1,los=f1,pheno=macro_f1', type=str, help='Primary metric map for multitask model selection.')
    parser.add_argument('--use_modality_mask_condition_router', action='store_true', help='Week30 Family B1/B2: inject a learned observed-modality-mask embedding into the MoE router input.')
    parser.add_argument('--mask_condition_stats', action='store_true', help='Week30 Family B: write modality-mask-conditioned routing fields for later aggregation.')
    parser.add_argument('--modality_mask_router_dim', default=128, type=int, help='Embedding dimension for the observed-modality-mask conditioning vector.')
    parser.add_argument('--use_modality_subset_expert_priors', action='store_true', help='Week30 Family B3: combine patient-conditioned expert scores with learnable modality-subset expert priors.')
    parser.add_argument('--modality_subset_prior_weight', default=1.0, type=float, help='Initial scaling applied to modality-subset expert priors before fusion with patient-conditioned logits.')
    parser.add_argument('--use_collaboration_branch', action='store_true', help='Week30 Family C: add a dense multimodal collaboration branch on top of the same fused latent state.')
    parser.add_argument('--collaboration_branch_type', default='mlp', choices=['mlp', 'transformer'], help='Architecture for the dense collaboration branch.')
    parser.add_argument('--collaboration_hidden_size', default=512, type=int, help='Hidden size for the dense collaboration branch.')
    parser.add_argument('--collaboration_fusion', default='fixed_average', choices=['fixed_average', 'learnable_scalar', 'confidence_based', 'input_dependent'], help='How to fuse MoE and dense-branch logits.')
    parser.add_argument('--collaboration_fixed_weight', default=0.5, type=float, help='Weight on MoE logits when --collaboration_fusion fixed_average is used.')
    parser.add_argument('--log_collaboration_diagnostics', action='store_true', help='Write branch agreement and fusion-weight diagnostics to router_diagnostics_test.csv.')
    parser.add_argument('--use_unimodal_kd', action='store_true', help='Week34 A: distill modality-specific teacher targets into unimodal student heads attached to FuseMoE.')
    parser.add_argument('--unimodal_kd_modalities', default='ts,text,cxr,ecg', type=str, help='Comma-separated modalities that receive Week34 unimodal KD.')
    parser.add_argument('--unimodal_kd_weight', default=1.0, type=float, help='Global scale applied to the summed Week34 unimodal KD loss.')
    parser.add_argument('--unimodal_kd_weights', default='', type=str, help='Optional comma-separated modality weights such as ts=1.0,text=1.0,cxr=1.5,ecg=1.0.')
    parser.add_argument('--unimodal_teacher_dir', default='', type=str, help='Directory or split template for Week34 unimodal teacher targets. Expected file names default to {split}_unimodal_teacher_targets.csv when a directory is given.')
    parser.add_argument('--unimodal_kd_temperature', default=1.0, type=float, help='Temperature used by Week34 unimodal KD.')
    parser.add_argument('--use_full_partial_consistency', action='store_true', help='Week34 B: match masked fused representations to full fused representations during training.')
    parser.add_argument('--consistency_weight', default=0.0, type=float, help='Scale applied to the Week34 full-vs-partial consistency loss.')
    parser.add_argument('--consistency_loss_type', default='mse', choices=['mse', 'cosine', 'smooth_l1', 'js'], help='Loss used for Week34 full-vs-partial consistency.')
    parser.add_argument('--consistency_mask_mode', default='single_random', choices=['single_random', 'progressive'], help='How to construct masked views for Week34 consistency training.')
    parser.add_argument('--consistency_missingness_level', default=0.25, type=float, help='Independent drop probability used when --consistency_mask_mode progressive.')
    parser.add_argument('--use_labelwise_fusion', action='store_true', help='Week34 C: replace the default fused head with a label-wise adaptive fusion head.')
    parser.add_argument('--labelwise_fusion_source', default='modalities', choices=['modalities', 'branches'], help='Source set used by Week34 label-wise adaptive fusion.')
    parser.add_argument('--labelwise_hidden_dim', default=128, type=int, help='Hidden dimension used by Week34 label-wise adaptive fusion projections.')
    parser.add_argument('--labelwise_rank_loss_weight', default=0.0, type=float, help='Optional auxiliary weight reserved for future label-wise ranking regularization.')
    parser.add_argument('--semantic_guidance_coef', default=0.0, type=float, help='Week30 Family D2/D3: coefficient for persistent expert-target guidance beyond warm-start.')
    parser.add_argument('--semantic_guidance_target_path', default=None, type=str, help='Optional CSV path/template for semantic targets used by persistent guidance and retention tracking. Falls back to expert-init targets when unset.')
    parser.add_argument('--semantic_retention_history_path', default=None, type=str, help='Optional CSV path for per-epoch semantic retention logging (Week30 D4).')
    parser.add_argument('--week30_variant_id', default='', type=str, help='Optional Week30 variant identifier used by manifests, grouped outputs, and analysis utilities.')
    parser.add_argument('--week30_family_name', default='', type=str, help='Optional Week30 family name used by grouped output organization.')
    parser.add_argument('--results_home_root', default='', type=str, help='Optional home-root override for grouped Week30 light outputs.')
    parser.add_argument('--results_scratch_root', default='', type=str, help='Optional scratch-root override for grouped Week30 heavy outputs.')
    parser.add_argument('--eval_progressive_missingness', action='store_true', help='Week30 B4/C5: evaluate under probabilistic modality corruption instead of only exact forced masks.')
    parser.add_argument('--eval_missingness_level', default=0.0, type=float, help='Week30 B4/C5: corruption probability in [0,1] applied independently to observed optional modalities during eval.')
    parser.add_argument('--eval_missingness_seed', default=0, type=int, help='Week30 B4/C5: deterministic seed used for progressive missingness corruption.')
    parser.add_argument('--use_interaction_router', action='store_true', help='Week31: condition the router on pairwise multimodal interaction features.')
    parser.add_argument('--interaction_router_only', action='store_true', help='Week31 ablation: use only interaction-derived router conditioning instead of raw router input additions.')
    parser.add_argument('--router_zero_input', action='store_true', help='Week31 follow-up diagnostic: zero the base router input before adding conditioning signals.')
    parser.add_argument('--interaction_router_scale', default=1.0, type=float, help='Scale applied to projected interaction features before router fusion.')
    parser.add_argument('--interaction_include_l2', action='store_true', help='Include pairwise L2 distances in Week31 interaction features.')
    parser.add_argument('--interaction_include_norm_ratio', action='store_true', help='Include pairwise norm ratios in Week31 interaction features.')
    parser.add_argument('--use_interaction_experts', action='store_true', help='I2MoE-style expert inputs: give routed experts fixed unique/redundant/synergy/contrast interaction views.')
    parser.add_argument('--interaction_expert_mode', default='roles4', choices=['roles4'], help='Role template for --use_interaction_experts.')
    parser.add_argument('--use_interaction_expert_reweighting', action='store_true', help='Learn per-sample weights over named interaction experts before combining routed expert outputs.')
    parser.add_argument('--interaction_reweight_hidden', default=128, type=int, help='Hidden size for --use_interaction_expert_reweighting.')
    parser.add_argument('--use_mohave_group_router', action='store_true', help='MoHAVE-style two-level routing: predict a modality-group prior and convert it into expert-logit bias before sparse expert routing.')
    parser.add_argument('--mohave_num_groups', default=4, type=int, help='Number of modality groups used by --use_mohave_group_router.')
    parser.add_argument('--mohave_group_router_hidden', default=128, type=int, help='Hidden size for the MoHAVE-style group router.')
    parser.add_argument('--mohave_group_prior_weight', default=1.0, type=float, help='Scale applied to group-derived expert-logit priors.')
    parser.add_argument('--mohave_group_assignments', default='', type=str, help='Comma-separated expert-to-group ids. Empty assigns expert i to group i mod num_groups.')
    parser.add_argument('--mohave_use_interaction_group', action='store_true', help='Condition the MoHAVE-style group router on interaction features when available.')
    parser.add_argument('--missingness_encoder_type', default='lookup', choices=['lookup', 'linear', 'mlp'], help='Week31 missingness encoder architecture.')
    parser.add_argument('--shared_semantic_memory_mode', default='none', choices=['none', 'fixed', 'learnable'], help='Week31 FLAME-inspired shared semantic memory mode.')
    parser.add_argument('--shared_semantic_memory_slots', default=8, type=int, help='Number of shared semantic memory slots.')
    parser.add_argument('--shared_semantic_memory_heads', default=1, type=int, help='Number of attention heads used by shared semantic memory.')
    parser.add_argument('--log_interaction_features', action='store_true', help='Write per-sample interaction feature summaries into router diagnostics when available.')
    parser.add_argument('--week31_variant_id', default='', type=str, help='Optional Week31 variant identifier for manifests and grouped outputs.')
    parser.add_argument('--week31_router_family', default='', type=str, help='Optional Week31 router family label for manifests and summaries.')

    args = parser.parse_args()
    args.gating_function = normalize_gating_function_arg(args.gating_function)

    if args.collaboration_fusion == "input_dependent":
        args.collaboration_fusion = "confidence_based"

    _apply_alias_args(args)
    _validate_overlapping_args(args, parser)
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
