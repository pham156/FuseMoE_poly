# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import math
from core.activations import ACT2FN
from utils.config import MoEConfig
from scripts.experiments.week31.multimodal_moe_campaign.components import SharedSemanticMemory, TaskSpecificRouter
import pdb

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, router_type):
        """Create a SparseDispatcher."""
        # each forward pass initialize the sparse dispatcher once
        self._gates = gates
        self._num_experts = num_experts
        self._router_type = router_type
        positive_gates = gates > 0
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(positive_gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        # _batch_index: sample index inside a batch that is assigned to particular expert, concatenated
        self._batch_index = torch.nonzero(positive_gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = positive_gates.sum(0).tolist()
        # expand gates to match with self._batch_index
        # from e.g., [64, 4] -> [256, 4], collection of samples assigned to each gate
        gates_exp = gates[self._batch_index.flatten()]
        # difference between torch.nonzero(gates) and self._nonzero_gates
        # the first one is index and the second one is actural weights!
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        if isinstance(inp, list) and self._router_type == 'joint':
            inp = torch.concat(inp, dim=1)
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # concat all sample outputs from each expert
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        # this is the weighted combination step
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def combine_linear(self, expert_out, multiply_by_gates=True):
        """Weighted sum for experts that emit ordinary feature deltas."""
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
            dtype=stitched.dtype,
        )
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, config:MoEConfig, input_size:int, output_size:int, hidden_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out


class FeatureMLP(nn.Module):
    """MLP expert that emits hidden features instead of log probabilities."""

    def __init__(self, config:MoEConfig, input_size:int, output_size:int, hidden_size:int):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        return self.fc2(out)


class LoRAExpert(nn.Module):
    """Frozen-base low-rank expert delta.

    This implements the mentor's LoRA-style expert equation at the current MoE
    latent-feature level:

        y = base(x) + scale * ((x A) B)

    The current architecture routes encoded multimodal representations, not raw
    LLM token states, so the frozen base is a feature-space projection rather
    than a pretrained medical LLM FFN layer.
    """

    def __init__(
        self,
        config: MoEConfig,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ):
        super(LoRAExpert, self).__init__()
        rank = config.lora_rank
        if rank <= 0:
            raise ValueError("lora_rank must be positive")

        self.base = nn.Linear(input_size, output_size)
        self.lora_A = nn.Linear(input_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_size, bias=False)
        self.dropout = nn.Dropout(config.lora_dropout)
        self.scaling = config.lora_alpha / rank
        self.log_soft = nn.LogSoftmax(1)

        if input_size == output_size:
            nn.init.eye_(self.base.weight)
            nn.init.zeros_(self.base.bias)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        if config.freeze_expert_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        out = base_out + delta
        return self.log_soft(out)


class ResidualLoRAExpert(nn.Module):
    """Low-rank residual expert for staged shared-FFN adaptation.

    The shared FFN is trained first and then frozen. These experts only learn
    residual deltas that are routed on top of the frozen shared FFN output:

        y = SharedFFN_frozen(x) + sum_i gate_i * ((x A_i) B_i)
    """

    def __init__(
        self,
        config: MoEConfig,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ):
        super(ResidualLoRAExpert, self).__init__()
        rank = config.lora_rank
        if rank <= 0:
            raise ValueError("lora_rank must be positive")
        self.lora_A = nn.Linear(input_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_size, bias=False)
        self.dropout = nn.Dropout(config.lora_dropout)
        self.scaling = config.lora_alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def build_expert(config: MoEConfig, input_size: int, output_size: int, hidden_size: int):
    if config.expert_type == "mlp":
        if getattr(config, "moe_mixing_space", "logprob") == "linear":
            return FeatureMLP(config, input_size, output_size, hidden_size)
        return MLP(config, input_size, output_size, hidden_size)
    if config.expert_type == "lora":
        return LoRAExpert(config, input_size, output_size, hidden_size)
    if config.expert_type == "residual_lora":
        return ResidualLoRAExpert(config, input_size, output_size, hidden_size)
    raise ValueError(f"Unknown expert_type: {config.expert_type}")


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, config: MoEConfig):
        super(MoE, self).__init__()
        self.noisy_gating = config.noisy_gating
        # print(self.noisy_gating)
        self.num_experts = config.num_experts
        self.output_size = config.moe_output_size
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size
        self.k = config.top_k
        self.disjoint_k = config.disjoint_top_k
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.gating = config.gating
        self.poly_power = config.poly_power
        self.student_degree = config.student_degree
        self.normalized = config.normalized
        self.use_bias = config.use_bias
        self.router_bias_mode = getattr(config, "router_bias_mode", "mul")
        self.gate_normalization = getattr(config, "gate_normalization", "selected")
        self.router_topk_mode = getattr(config, "router_topk_mode", "k_plus_1")
        self.moe_mixing_space = getattr(config, "moe_mixing_space", "logprob")
        self.load_balance_mode = getattr(config, "load_balance_mode", "cv")
        self.shared_experts = config.shared_experts
        self.shared_expert_weight = float(getattr(config, "shared_expert_weight", 1.0))
        self.freeze_shared_ffn = bool(getattr(config, "freeze_shared_ffn", False))
        self.use_temp = config.use_temp
        self.expert_type = config.expert_type
        self.staged_shared_lora = getattr(config, "staged_shared_lora", False)
        self.staged_shared_lora_warmup_epochs = int(getattr(config, "staged_shared_lora_warmup_epochs", 8))
        self.staged_shared_lora_phase = "disabled"
        self.expert_orth_coef = getattr(config, "expert_orth_coef", 0.0)
        self.use_instruction_router = config.use_instruction_router
        self.router_instruction_dim = config.router_instruction_dim
        self.instruction_router_scale = config.instruction_router_scale
        self.instruction_router_fusion = config.instruction_router_fusion
        self.use_semantic_expert_profiles = config.use_semantic_expert_profiles
        self.semantic_profile_scale = config.semantic_profile_scale
        self.semantic_profile_fusion = config.semantic_profile_fusion
        self.semantic_profile_source = config.semantic_profile_source
        self.semantic_project_dim = getattr(config, "semantic_project_dim", None)
        self.semantic_profile_modalities = config.semantic_profile_modalities or ["txt"]
        self.use_prototype_router = config.use_prototype_router
        self.prototype_router_dim = config.prototype_router_dim
        self.prototype_router_temperature = config.prototype_router_temperature
        self.prototype_router_dense = config.prototype_router_dense
        self.prototype_router_orth_coef = config.prototype_router_orth_coef
        self.use_router_organ_supervision = config.use_router_organ_supervision
        self.router_organ_supervision_coef = config.router_organ_supervision_coef
        self.router_organ_supervision_class_balanced = config.router_organ_supervision_class_balanced
        self.router_z_loss_coef = getattr(config, "router_z_loss_coef", 0.0)
        self.router_z_loss_type = getattr(config, "router_z_loss_type", "logsumexp")
        self.router_entropy_coef = getattr(config, "router_entropy_coef", 0.0)
        self.router_variance_coef = getattr(config, "router_variance_coef", 0.0)
        self.output_orth_coef = getattr(config, "output_orth_coef", 0.0)
        self.specialization_loss_mode = getattr(config, "specialization_loss_mode", "legacy")
        self.specialization_aux_coef = float(getattr(config, "specialization_aux_coef", 1e-3))
        self.dense_warmup_epochs = getattr(config, "dense_warmup_epochs", 0)
        self.router_temperature = max(float(getattr(config, "router_temperature", 1.0)), 1e-6)
        self.router_noise_scale = max(float(getattr(config, "router_noise_scale", 1.0)), 0.0)
        router_noise_final_scale = getattr(config, "router_noise_final_scale", None)
        self.router_noise_final_scale = (
            None
            if router_noise_final_scale is None
            else max(float(router_noise_final_scale), 0.0)
        )
        self.router_noise_decay_epochs = max(int(getattr(config, "router_noise_decay_epochs", 0)), 0)
        self.poly_noise_mode = getattr(config, "poly_noise_mode", "distance")
        self.router_init = getattr(config, "router_init", "zero")
        self.router_init_std = float(getattr(config, "router_init_std", 0.02))
        self.use_dynamic_top_k = bool(getattr(config, "use_dynamic_top_k", False))
        self.dynamic_top_k_min = max(int(getattr(config, "dynamic_top_k_min", 1)), 1)
        self.dynamic_top_k_max = max(int(getattr(config, "dynamic_top_k_max", 3)), self.dynamic_top_k_min)
        self.dynamic_top_k_confidence_threshold = float(getattr(config, "dynamic_top_k_confidence_threshold", 0.7))
        self.dynamic_top_k_entropy_threshold = float(getattr(config, "dynamic_top_k_entropy_threshold", 0.8))
        self.use_multihead_permod_router = bool(getattr(config, "use_multihead_permod_router", False))
        self.multihead_router_heads = max(int(getattr(config, "multihead_router_heads", 4)), 1)
        self.multihead_router_fusion = getattr(config, "multihead_router_fusion", "mean")
        self.use_xmoe_router = getattr(config, "use_xmoe_router", False)
        self.xmoe_router_dim = getattr(config, "xmoe_router_dim", 128)
        self.xmoe_router_init_norm = getattr(config, "xmoe_router_init_norm", 0.1)
        self.xmoe_noise_scale = max(float(getattr(config, "xmoe_noise_scale", 1.0)), 0.0)
        self.log_expert_output_diagnostics = getattr(config, "log_expert_output_diagnostics", False)
        self.expert_init_strategy = getattr(config, "expert_init_strategy", "none")
        self.expert_init_epochs = max(int(getattr(config, "expert_init_epochs", 0)), 0)
        self.expert_init_soft_targets = bool(getattr(config, "expert_init_soft_targets", False))
        self.expert_init_release_schedule = getattr(config, "expert_init_release_schedule", "hard")
        self.expert_init_confidence_threshold = float(getattr(config, "expert_init_confidence_threshold", 0.0))
        self.expert_init_uniform_fallback = bool(getattr(config, "expert_init_uniform_fallback", False))
        self.expert_init_skip_low_confidence = bool(getattr(config, "expert_init_skip_low_confidence", False))
        self.use_task_condition_router = bool(getattr(config, "use_task_condition_router", False))
        self.task_condition_stats = bool(getattr(config, "task_condition_stats", False))
        self.task_router_dim = int(getattr(config, "task_router_dim", 128))
        self.use_task_specific_router_heads = bool(getattr(config, "use_task_specific_router_heads", False))
        self.use_task_condition_expert_modulation = bool(getattr(config, "use_task_condition_expert_modulation", False))
        self.task_expert_modulation_type = getattr(config, "task_expert_modulation_type", "film")
        self.use_modality_mask_condition_router = bool(getattr(config, "use_modality_mask_condition_router", False))
        self.mask_condition_stats = bool(getattr(config, "mask_condition_stats", False))
        self.modality_mask_router_dim = int(getattr(config, "modality_mask_router_dim", 128))
        self.use_modality_subset_expert_priors = bool(getattr(config, "use_modality_subset_expert_priors", False))
        self.modality_subset_prior_weight = float(getattr(config, "modality_subset_prior_weight", 1.0))
        self.semantic_guidance_coef = float(getattr(config, "semantic_guidance_coef", 0.0))
        self.use_interaction_router = bool(getattr(config, "use_interaction_router", False))
        self.interaction_router_only = bool(getattr(config, "interaction_router_only", False))
        self.router_zero_input = bool(getattr(config, "router_zero_input", False))
        self.interaction_feature_dim = int(getattr(config, "interaction_feature_dim", 0))
        self.interaction_router_scale = float(getattr(config, "interaction_router_scale", 1.0))
        self.use_interaction_experts = bool(getattr(config, "use_interaction_experts", False))
        self.interaction_expert_mode = getattr(config, "interaction_expert_mode", "roles4")
        self.use_interaction_expert_reweighting = bool(getattr(config, "use_interaction_expert_reweighting", False))
        self.interaction_reweight_hidden = int(getattr(config, "interaction_reweight_hidden", 128))
        self.use_mohave_group_router = bool(getattr(config, "use_mohave_group_router", False))
        self.mohave_num_groups = max(int(getattr(config, "mohave_num_groups", 4)), 1)
        self.mohave_group_router_hidden = max(int(getattr(config, "mohave_group_router_hidden", 128)), 1)
        self.mohave_group_prior_weight = float(getattr(config, "mohave_group_prior_weight", 1.0))
        self.mohave_group_assignments = str(getattr(config, "mohave_group_assignments", "") or "")
        self.mohave_use_interaction_group = bool(getattr(config, "mohave_use_interaction_group", False))
        self.shared_semantic_memory_mode = getattr(config, "shared_semantic_memory_mode", "none")
        self.shared_semantic_memory_slots = int(getattr(config, "shared_semantic_memory_slots", 8))
        self.shared_semantic_memory_heads = int(getattr(config, "shared_semantic_memory_heads", 1))
        self.current_epoch = 0
        self._last_router_logits_for_loss = None
        self._output_orth_losses = []
        self._paper_aux_scale_target = None
        self._specialization_metric_records = []
        self.last_specialization_diagnostics = None
        self.last_expert_output_diagnostics = None
        self.last_contribution_diagnostics = None
        self.eval_ablate_expert_indices = set()
        self.eval_ablate_shared_path = False
        self.eval_ablate_routed_path = False
        self.current_expert_init_batch = None
        self.current_expert_init_active = False
        self.last_expert_init_diagnostics = None
        self.current_task_name = None
        self.current_task_id = None
        self.current_task_embedding = None
        self.current_modality_mask_ids = None
        self.current_modality_mask_strings = None
        self.current_interaction_features = None
        self._last_router_inputs_for_diag = None
        self._last_clean_logits_for_diag = None
        self.r2t2_adaptor = None
        self.r2t2_layer_id = None

        if self.use_multihead_permod_router:
            if self.router_type != 'permod':
                raise ValueError("--use_multihead_permod_router requires router_type='permod'")
            if self.use_xmoe_router:
                raise ValueError("--use_multihead_permod_router is not compatible with --use_xmoe_router")
            if self.multihead_router_heads < 2:
                raise ValueError("multihead_router_heads must be >= 2")
            if self.multihead_router_fusion not in {"mean", "learned"}:
                raise ValueError(f"Unknown multihead_router_fusion={self.multihead_router_fusion}")
        if self.use_dynamic_top_k:
            if self.dynamic_top_k_max > self.num_experts:
                raise ValueError("dynamic_top_k_max cannot exceed num_experts")
            if self.dynamic_top_k_min > self.dynamic_top_k_max:
                raise ValueError("dynamic_top_k_min cannot exceed dynamic_top_k_max")
        if self.use_interaction_experts:
            if self.router_type != 'permod':
                raise ValueError("--use_interaction_experts currently requires router_type='permod'")
            if self.num_experts < 4:
                raise ValueError("--use_interaction_experts requires at least 4 experts")
            if self.moe_mixing_space == "linear":
                raise ValueError("--use_interaction_experts currently supports logprob MoE mixing only")
            if self.interaction_expert_mode != "roles4":
                raise ValueError(f"Unknown interaction_expert_mode={self.interaction_expert_mode}")
        if self.use_interaction_expert_reweighting and not self.use_interaction_experts:
            raise ValueError("--use_interaction_expert_reweighting requires --use_interaction_experts")
        if self.use_mohave_group_router:
            if self.router_type != 'permod':
                raise ValueError("--use_mohave_group_router currently requires router_type='permod'")
            if self.mohave_num_groups < 1:
                raise ValueError("mohave_num_groups must be >= 1")

        if self.use_task_specific_router_heads and self.gating != 'softmax':
            raise ValueError("--use_task_specific_router_heads currently requires --gating_function softmax")

        # instantiate experts
        if self.use_xmoe_router:
            if self.router_type == 'disjoint':
                self.xmoe_router_proj = nn.ModuleList([
                    nn.Linear(self.input_size//self.num_modalities, self.xmoe_router_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
                self.xmoe_noise_proj = nn.ModuleList([
                    nn.Linear(self.xmoe_router_dim, self.num_experts//self.num_modalities, bias=False)
                    for _ in range(self.num_modalities)
                ])
                self.xmoe_expert_embeddings = nn.ParameterList([
                    nn.Parameter(torch.empty(self.num_experts//self.num_modalities, self.xmoe_router_dim))
                    for _ in range(self.num_modalities)
                ])
            elif self.router_type == 'permod':
                self.xmoe_router_proj = nn.ModuleList([
                    nn.Linear(self.input_size//self.num_modalities, self.xmoe_router_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
                self.xmoe_noise_proj = nn.ModuleList([
                    nn.Linear(self.xmoe_router_dim, self.num_experts, bias=False)
                    for _ in range(self.num_modalities)
                ])
                self.xmoe_expert_embeddings = nn.ParameterList([
                    nn.Parameter(torch.empty(self.num_experts, self.xmoe_router_dim))
                    for _ in range(self.num_modalities)
                ])
            else:
                self.xmoe_router_proj = nn.Linear(self.input_size, self.xmoe_router_dim, bias=False)
                self.xmoe_noise_proj = nn.Linear(self.xmoe_router_dim, self.num_experts, bias=False)
                self.xmoe_expert_embeddings = nn.Parameter(torch.empty(self.num_experts, self.xmoe_router_dim))
            self._reset_xmoe_router()
        else:
            self.xmoe_router_proj = None
            self.xmoe_noise_proj = None
            self.xmoe_expert_embeddings = None

        if self.router_type == 'disjoint':
            self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
        elif self.router_type == 'permod':
            if self.use_multihead_permod_router:
                self.w_gate = [
                    nn.Parameter(
                        torch.zeros(self.multihead_router_heads, self.input_size//self.num_modalities, self.num_experts),
                        requires_grad=True,
                    )
                    for _ in range(self.num_modalities)
                ]
                self.w_noise = [
                    nn.Parameter(
                        torch.zeros(self.multihead_router_heads, self.input_size//self.num_modalities, self.num_experts),
                        requires_grad=True,
                    )
                    for _ in range(self.num_modalities)
                ]
                self.multihead_router_logits = nn.Parameter(torch.zeros(self.num_modalities, self.multihead_router_heads))
            else:
                self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
                self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
                self.multihead_router_logits = None
        else:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            self.multihead_router_logits = None

        self._reset_standard_router()

        if self.use_task_specific_router_heads:
            task_names = ["ihm", "los", "pheno"]
            if self.router_type == 'disjoint':
                input_dim = self.input_size // self.num_modalities
                sub_experts = self.num_experts // self.num_modalities
                self.task_specific_router = nn.ModuleList([
                    TaskSpecificRouter(input_dim=input_dim, num_experts=sub_experts, task_names=task_names)
                    for _ in range(self.num_modalities)
                ])
            elif self.router_type == 'permod':
                input_dim = self.input_size // self.num_modalities
                self.task_specific_router = nn.ModuleList([
                    TaskSpecificRouter(input_dim=input_dim, num_experts=self.num_experts, task_names=task_names)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.task_specific_router = TaskSpecificRouter(
                    input_dim=self.input_size,
                    num_experts=self.num_experts,
                    task_names=task_names,
                )
        else:
            self.task_specific_router = None

        if self.use_prototype_router:
            if self.router_type == 'disjoint':
                sub_experts = self.num_experts // self.num_modalities
                input_dim = self.input_size // self.num_modalities
                self.prototype_query_proj = nn.ModuleList([
                    self._build_prototype_query_proj(input_dim)
                    for _ in range(self.num_modalities)
                ])
                self.router_prototypes = nn.ParameterList([
                    nn.Parameter(torch.empty(sub_experts, self.prototype_router_dim))
                    for _ in range(self.num_modalities)
                ])
            elif self.router_type == 'permod':
                input_dim = self.input_size // self.num_modalities
                self.prototype_query_proj = nn.ModuleList([
                    self._build_prototype_query_proj(input_dim)
                    for _ in range(self.num_modalities)
                ])
                self.router_prototypes = nn.ParameterList([
                    nn.Parameter(torch.empty(self.num_experts, self.prototype_router_dim))
                    for _ in range(self.num_modalities)
                ])
            else:
                self.prototype_query_proj = self._build_prototype_query_proj(self.input_size)
                self.router_prototypes = nn.Parameter(torch.empty(self.num_experts, self.prototype_router_dim))
            self._reset_router_prototypes()
        else:
            self.prototype_query_proj = None
            self.router_prototypes = None

        if self.use_instruction_router:
            if self.router_instruction_dim is None:
                raise ValueError("router_instruction_dim must be set when use_instruction_router=True")
            use_logit_bias = self.instruction_router_fusion in ["logit_bias", "both"]
            use_input_add = self.instruction_router_fusion in ["input_add", "both"]

            if use_logit_bias:
                if self.router_type == 'disjoint':
                    sub_experts = self.num_experts // self.num_modalities
                    self.instruction_router = nn.ModuleList([
                        nn.Linear(self.router_instruction_dim, sub_experts, bias=False)
                        for _ in range(self.num_modalities)
                    ])
                elif self.router_type == 'permod':
                    self.instruction_router = nn.ModuleList([
                        nn.Linear(self.router_instruction_dim, self.num_experts, bias=False)
                        for _ in range(self.num_modalities)
                    ])
                else:
                    self.instruction_router = nn.Linear(self.router_instruction_dim, self.num_experts, bias=False)
                for layer in self.instruction_router.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
            else:
                self.instruction_router = None

            if use_input_add:
                if self.router_type in ['disjoint', 'permod']:
                    input_dim = self.input_size // self.num_modalities
                    self.instruction_input_proj = nn.ModuleList([
                        nn.Linear(self.router_instruction_dim, input_dim, bias=False)
                        for _ in range(self.num_modalities)
                    ])
                else:
                    self.instruction_input_proj = nn.Linear(self.router_instruction_dim, self.input_size, bias=False)
                for layer in self.instruction_input_proj.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
            else:
                self.instruction_input_proj = None
        else:
            self.instruction_router = None
            self.instruction_input_proj = None

        if self.use_task_condition_router:
            if self.router_type in ['disjoint', 'permod']:
                input_dim = self.input_size // self.num_modalities
                self.task_input_proj = nn.ModuleList([
                    nn.Linear(self.task_router_dim, input_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.task_input_proj = nn.Linear(self.task_router_dim, self.input_size, bias=False)
            for layer in self.task_input_proj.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
        else:
            self.task_input_proj = None

        if self.use_task_condition_expert_modulation:
            modulation_dim = self.output_size * (2 if self.task_expert_modulation_type == "film" else 1)
            if self.router_type == 'disjoint':
                sub_experts = self.num_experts // self.num_modalities
                self.task_expert_modulators = nn.ModuleList([
                    nn.ModuleList([
                        nn.Linear(self.task_router_dim, modulation_dim, bias=True)
                        for _ in range(sub_experts)
                    ])
                    for _ in range(self.num_modalities)
                ])
            elif self.router_type == 'permod':
                self.task_expert_modulators = nn.ModuleList([
                    nn.Linear(self.task_router_dim, modulation_dim, bias=True)
                    for _ in range(self.num_experts)
                ])
            else:
                self.task_expert_modulators = nn.ModuleList([
                    nn.Linear(self.task_router_dim, modulation_dim, bias=True)
                    for _ in range(self.num_experts)
                ])
            for layer in self.task_expert_modulators.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
        else:
            self.task_expert_modulators = None

        if self.use_modality_mask_condition_router:
            if self.router_type in ['disjoint', 'permod']:
                input_dim = self.input_size // self.num_modalities
                self.mask_input_proj = nn.ModuleList([
                    nn.Linear(self.modality_mask_router_dim, input_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.mask_input_proj = nn.Linear(self.modality_mask_router_dim, self.input_size, bias=False)
            for layer in self.mask_input_proj.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
        else:
            self.mask_input_proj = None

        if self.use_interaction_router:
            if self.interaction_feature_dim <= 0:
                raise ValueError("interaction_feature_dim must be positive when use_interaction_router=True")
            if self.router_type in ['disjoint', 'permod']:
                input_dim = self.input_size // self.num_modalities
                self.interaction_input_proj = nn.ModuleList([
                    nn.Linear(self.interaction_feature_dim, input_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.interaction_input_proj = nn.Linear(self.interaction_feature_dim, self.input_size, bias=False)
            for layer in self.interaction_input_proj.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
        else:
            self.interaction_input_proj = None

        if self.use_modality_subset_expert_priors:
            prior_weight = torch.tensor(float(self.modality_subset_prior_weight))
            self.modality_subset_prior_log_scale = nn.Parameter(prior_weight.log())
            if self.router_type == 'disjoint':
                sub_experts = self.num_experts // self.num_modalities
                self.mask_prior_proj = nn.ModuleList([
                    nn.Linear(self.modality_mask_router_dim, sub_experts, bias=False)
                    for _ in range(self.num_modalities)
                ])
            elif self.router_type == 'permod':
                self.mask_prior_proj = nn.ModuleList([
                    nn.Linear(self.modality_mask_router_dim, self.num_experts, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.mask_prior_proj = nn.Linear(self.modality_mask_router_dim, self.num_experts, bias=False)
            for layer in self.mask_prior_proj.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
        else:
            self.modality_subset_prior_log_scale = None
            self.mask_prior_proj = None

        if self.use_interaction_expert_reweighting:
            input_dim = self.input_size // self.num_modalities
            hidden_dim = max(int(self.interaction_reweight_hidden), 1)
            self.interaction_reweight_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(input_dim * 4),
                    nn.Linear(input_dim * 4, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.num_experts),
                )
                for _ in range(self.num_modalities)
            ])
            for head in self.interaction_reweight_heads:
                final = head[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.interaction_reweight_heads = None

        if self.use_mohave_group_router:
            input_dim = self.input_size // self.num_modalities
            group_input_dim = input_dim
            if self.mohave_use_interaction_group:
                group_input_dim += max(self.interaction_feature_dim, 0)
            self.mohave_group_router = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(group_input_dim),
                    nn.Linear(group_input_dim, self.mohave_group_router_hidden),
                    nn.GELU(),
                    nn.Linear(self.mohave_group_router_hidden, self.mohave_num_groups),
                )
                for _ in range(self.num_modalities)
            ])
            self.mohave_group_log_scale = nn.Parameter(torch.tensor(float(self.mohave_group_prior_weight)).clamp_min(1e-6).log())
            if self.mohave_group_assignments:
                assignments = [int(item.strip()) for item in self.mohave_group_assignments.split(",") if item.strip()]
                if len(assignments) != self.num_experts:
                    raise ValueError("mohave_group_assignments length must match num_experts")
            else:
                assignments = [idx % self.mohave_num_groups for idx in range(self.num_experts)]
            assignment_tensor = torch.tensor(assignments, dtype=torch.long).clamp(min=0, max=self.mohave_num_groups - 1)
            self.register_buffer("mohave_expert_group_ids", assignment_tensor)
            for head in self.mohave_group_router:
                final = head[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.mohave_group_router = None
            self.mohave_group_log_scale = None

        semantic_profiles = config.semantic_profile_embeddings
        if self.use_semantic_expert_profiles and self.semantic_profile_source == "patient":
            if semantic_profiles is None:
                raise ValueError("semantic_profile_embeddings must be set when use_semantic_expert_profiles=True")
            self.register_buffer("semantic_profile_embeddings", semantic_profiles.detach().float())
            profile_dim = semantic_profiles.size(-1)
            semantic_dim = int(self.semantic_project_dim) if self.semantic_project_dim else profile_dim
            if self.router_type in ['disjoint', 'permod']:
                input_dim = self.input_size // self.num_modalities
                self.semantic_input_proj = nn.ModuleList([
                    nn.Linear(input_dim, semantic_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.semantic_input_proj = nn.Linear(self.input_size, semantic_dim, bias=False)
            self.semantic_profile_proj = (
                nn.Linear(profile_dim, semantic_dim, bias=False)
                if semantic_dim != profile_dim
                else None
            )
        elif self.use_semantic_expert_profiles:
            if semantic_profiles is None:
                raise ValueError("semantic_profile_embeddings must be set when use_semantic_expert_profiles=True")
            self.register_buffer("semantic_profile_embeddings", semantic_profiles.detach().float())
            self.semantic_input_proj = None
            self.semantic_profile_proj = None
        else:
            self.semantic_profile_embeddings = None
            self.semantic_input_proj = None
            self.semantic_profile_proj = None

        if self.router_type == 'disjoint':
            self.experts = nn.ModuleList(
                nn.ModuleList([build_expert(config, self.input_size//self.num_modalities, self.output_size, self.hidden_size) for _ in range(self.num_experts//self.num_modalities)])
                for _ in range(self.num_modalities)
            )
        elif self.router_type == 'permod':
            self.experts = nn.ModuleList([build_expert(config, self.input_size//self.num_modalities, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([build_expert(config, self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        
        if self.use_bias:
            if self.router_type == 'joint':
                self.bias = nn.Parameter(torch.zeros(self.num_experts))
            elif self.router_type == 'permod':
                # Each modality has its own set of experts (num_experts per modality)
                self.bias = nn.ParameterList([nn.Parameter(torch.zeros(self.num_experts)) for _ in range(self.num_modalities)])
            elif self.router_type == 'disjoint':
                sub_experts = self.num_experts // self.num_modalities
                self.bias = nn.ParameterList([nn.Parameter(torch.zeros(sub_experts)) for _ in range(self.num_modalities)])
            else:
                self.bias = None
        else:
            self.bias = None
        
        if self.shared_experts > 0:
            # Shared experts operate on the original input (before dispatching)
            self.shared_mlps = nn.ModuleList([
                build_expert(config, self.input_size, self.output_size, self.hidden_size)
                for _ in range(self.shared_experts)
            ])
            if self.normalized:
                self.omega = nn.Parameter(torch.zeros(self.shared_experts))   # passed through softmax
            else:
                self.omega = nn.Parameter(torch.ones(self.shared_experts)) 
            if self.freeze_shared_ffn:
                for param in self.shared_mlps.parameters():
                    param.requires_grad = False
        else:
            self.shared_mlps = None
            self.omega = None

        if self.shared_semantic_memory_mode != "none":
            expert_input_dim = self.input_size // self.num_modalities if self.router_type in ['disjoint', 'permod'] else self.input_size
            self.shared_semantic_memory = SharedSemanticMemory(
                dim=expert_input_dim,
                num_slots=self.shared_semantic_memory_slots,
                mode=self.shared_semantic_memory_mode,
                num_heads=self.shared_semantic_memory_heads,
            )
        else:
            self.shared_semantic_memory = None

        if self.use_temp:
            self.log_tau = nn.Parameter(torch.zeros(1))  # log of temperature parameter for softmax gating
        else:
            self.log_tau = None

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        if self.staged_shared_lora:
            if self.expert_type != "residual_lora":
                raise ValueError("--staged_shared_lora requires --expert_type residual_lora")
            if self.shared_experts <= 0:
                raise ValueError("--staged_shared_lora requires --shared_experts >= 1")
            self._configure_staged_shared_lora_phase(0)

        assert(self.k <= self.num_experts)

    def set_current_epoch(self, epoch):
        self.current_epoch = int(epoch)
        if self.staged_shared_lora:
            self._configure_staged_shared_lora_phase(self.current_epoch)

    def set_expert_init_batch(self, batch_info=None, active=False):
        self.current_expert_init_batch = batch_info
        self.current_expert_init_active = bool(active)

    def set_expert_init_router_frozen(self, frozen):
        for param in self._iter_router_parameters():
            param.requires_grad = not frozen

    def _expert_init_release_epochs(self):
        if self.expert_init_epochs <= 0:
            return 0
        return max(1, int(math.ceil(self.expert_init_epochs * 0.25)))

    def _current_expert_init_alpha(self, train):
        if (
            not train
            or self.expert_init_strategy != "fixed_cohort"
            or self.current_expert_init_batch is None
            or not self.current_expert_init_active
            or self.expert_init_epochs <= 0
        ):
            return 0.0
        if self.expert_init_release_schedule == "hard":
            return 1.0 if self.current_epoch < self.expert_init_epochs else 0.0
        release_epochs = self._expert_init_release_epochs()
        if self.current_epoch < self.expert_init_epochs:
            return 1.0
        decay_epoch = self.current_epoch - self.expert_init_epochs
        if decay_epoch >= release_epochs:
            return 0.0
        return max(0.0, 1.0 - (decay_epoch / max(release_epochs, 1)))

    def _prepare_expert_init_targets(self, batch_size, device, dtype):
        batch = self.current_expert_init_batch or {}
        targets = batch.get("targets")
        if targets is None:
            return None, None, None, None
        targets = targets.to(device=device, dtype=dtype)
        if targets.dim() != 2 or targets.size(0) != batch_size or targets.size(1) != self.num_experts:
            return None, None, None, None
        confidence = batch.get("confidence")
        if confidence is None:
            confidence = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            confidence = confidence.to(device=device, dtype=dtype).view(-1)
        valid_mask = targets.sum(dim=1) > 0
        if self.expert_init_confidence_threshold > 0:
            valid_mask = valid_mask & (confidence >= self.expert_init_confidence_threshold)
        if self.expert_init_skip_low_confidence and self.expert_init_confidence_threshold > 0:
            valid_mask = valid_mask & (confidence >= self.expert_init_confidence_threshold)

        target_dist = targets.clamp_min(0.0)
        target_sums = target_dist.sum(dim=1, keepdim=True)
        nonzero = target_sums.squeeze(1) > 0
        if nonzero.any():
            target_dist[nonzero] = target_dist[nonzero] / target_sums[nonzero].clamp_min(1e-12)
        if self.expert_init_uniform_fallback:
            fallback_mask = ~valid_mask
            if fallback_mask.any():
                target_dist[fallback_mask] = 1.0 / float(self.num_experts)
                valid_mask = valid_mask | fallback_mask
        sources = batch.get("source", [])
        return target_dist, valid_mask, confidence, sources

    def _record_expert_init_diagnostics(self, gates, alpha):
        gates_list = gates if isinstance(gates, list) else [gates]
        batch_size = gates_list[0].size(0)
        device = gates_list[0].device
        dtype = gates_list[0].dtype
        target_dist, valid_mask, confidence, sources = self._prepare_expert_init_targets(batch_size, device, dtype)
        if target_dist is None or valid_mask is None or not bool(valid_mask.any().detach().cpu().item()):
            self.last_expert_init_diagnostics = None
            return
        expert_init_pred = gates_list[0].argmax(dim=1)
        target_max = target_dist.max(dim=1, keepdim=True).values
        target_top = target_dist >= (target_max - 1e-12)
        retained = target_top.gather(1, expert_init_pred.unsqueeze(1)).float().squeeze(1)
        retained = retained[valid_mask]
        self.last_expert_init_diagnostics = {
            "expert_init_alpha": float(alpha),
            "expert_init_retention_top1": retained.mean().detach() if retained.numel() > 0 else torch.zeros((), device=device),
            "expert_init_valid_fraction": valid_mask.float().mean().detach(),
            "expert_init_mean_confidence": confidence[valid_mask].mean().detach() if bool(valid_mask.any().detach().cpu().item()) else torch.zeros((), device=device),
        }

    def _mix_expert_init_gates(self, gates, train):
        alpha = self._current_expert_init_alpha(train)
        if alpha <= 0:
            self._record_expert_init_diagnostics(gates, alpha=0.0)
            return gates, False
        gates_list = gates if isinstance(gates, list) else [gates]
        batch_size = gates_list[0].size(0)
        device = gates_list[0].device
        dtype = gates_list[0].dtype
        target_dist, valid_mask, confidence, sources = self._prepare_expert_init_targets(batch_size, device, dtype)
        if target_dist is None or valid_mask is None or not bool(valid_mask.any().detach().cpu().item()):
            return gates, False

        mixed_gates = []
        for gate_tensor in gates_list:
            mixed = gate_tensor.clone()
            mixed_valid = alpha * target_dist[valid_mask] + (1.0 - alpha) * gate_tensor[valid_mask]
            mixed_valid = mixed_valid / mixed_valid.sum(dim=1, keepdim=True).clamp_min(1e-12)
            mixed[valid_mask] = mixed_valid
            mixed_gates.append(mixed)
        self._record_expert_init_diagnostics(mixed_gates if isinstance(gates, list) else mixed_gates[0], alpha=float(alpha))
        if isinstance(gates, list):
            return mixed_gates, True
        return mixed_gates[0], True

    def _iter_router_parameters(self):
        router_objects = [
            self.w_gate,
            self.w_noise,
            self.multihead_router_logits,
            self.bias,
            self.xmoe_router_proj,
            self.xmoe_noise_proj,
            self.xmoe_expert_embeddings,
            self.prototype_query_proj,
            self.router_prototypes,
            self.instruction_router,
            self.instruction_input_proj,
            self.task_input_proj,
            self.mask_input_proj,
            self.semantic_input_proj,
            self.log_tau,
        ]
        for obj in router_objects:
            if obj is None:
                continue
            if isinstance(obj, nn.Parameter):
                yield obj
            elif isinstance(obj, (list, tuple, nn.ParameterList, nn.ModuleList)):
                for item in obj:
                    if item is None:
                        continue
                    if isinstance(item, nn.Parameter):
                        yield item
                    elif isinstance(item, nn.Module):
                        yield from item.parameters()
            elif isinstance(obj, nn.Module):
                yield from obj.parameters()

    def _set_requires_grad(self, module_or_params, requires_grad):
        if module_or_params is None:
            return
        if isinstance(module_or_params, nn.Parameter):
            module_or_params.requires_grad = requires_grad
            return
        if isinstance(module_or_params, (list, tuple, nn.ModuleList, nn.ParameterList)):
            for item in module_or_params:
                self._set_requires_grad(item, requires_grad)
            return
        for param in module_or_params.parameters():
            param.requires_grad = requires_grad

    def _configure_staged_shared_lora_phase(self, epoch):
        phase = "shared_pretrain" if epoch < self.staged_shared_lora_warmup_epochs else "lora_adapt"
        if phase == self.staged_shared_lora_phase:
            return
        self.staged_shared_lora_phase = phase
        if phase == "shared_pretrain":
            self._set_requires_grad(self.shared_mlps, True)
            self._set_requires_grad(self.experts, False)
            for param in self._iter_router_parameters():
                param.requires_grad = False
        else:
            self._set_requires_grad(self.shared_mlps, False)
            self._set_requires_grad(self.experts, True)
            for param in self._iter_router_parameters():
                param.requires_grad = True

    def _use_dense_warmup(self):
        return (
            self.training
            and self.dense_warmup_epochs > 0
            and self.current_epoch < self.dense_warmup_epochs
        )

    def _current_router_noise_scale(self):
        if self.router_noise_final_scale is None or self.router_noise_decay_epochs <= 0:
            return self.router_noise_scale
        progress = min(max(self.current_epoch, 0), self.router_noise_decay_epochs) / self.router_noise_decay_epochs
        return self.router_noise_scale + progress * (self.router_noise_final_scale - self.router_noise_scale)

    def _build_prototype_query_proj(self, input_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.prototype_router_dim),
            nn.GELU(),
            nn.Linear(self.prototype_router_dim, self.prototype_router_dim),
        )

    def _reset_router_prototypes(self):
        if isinstance(self.router_prototypes, nn.ParameterList):
            for prototypes in self.router_prototypes:
                nn.init.xavier_uniform_(prototypes)
        else:
            nn.init.xavier_uniform_(self.router_prototypes)

    def _iter_router_params(self, param_or_list):
        if isinstance(param_or_list, (list, nn.ParameterList)):
            return param_or_list
        return [param_or_list]

    def _reset_standard_router(self):
        # Original FuseMoE uses exact-zero router weights. Random modes are an
        # explicit DeepSeek-style ablation to remove deterministic top-k ties.
        if self.use_xmoe_router:
            return
        for gate in self._iter_router_params(self.w_gate):
            if self.router_init == "zero":
                nn.init.zeros_(gate)
            elif self.router_init == "normal":
                nn.init.normal_(gate, mean=0.0, std=self.router_init_std)
            elif self.router_init == "xavier":
                nn.init.xavier_uniform_(gate)
            elif self.router_init == "kaiming":
                nn.init.kaiming_uniform_(gate, a=math.sqrt(5))
            else:
                raise ValueError(f"Unknown router_init={self.router_init}")
        for noise in self._iter_router_params(self.w_noise):
            nn.init.zeros_(noise)

    def _reset_xmoe_router(self):
        if isinstance(self.xmoe_router_proj, nn.ModuleList):
            for proj in self.xmoe_router_proj:
                nn.init.xavier_uniform_(proj.weight)
        elif self.xmoe_router_proj is not None:
            nn.init.xavier_uniform_(self.xmoe_router_proj.weight)

        if isinstance(self.xmoe_noise_proj, nn.ModuleList):
            for proj in self.xmoe_noise_proj:
                nn.init.zeros_(proj.weight)
        elif self.xmoe_noise_proj is not None:
            nn.init.zeros_(self.xmoe_noise_proj.weight)

        embeddings = (
            self.xmoe_expert_embeddings
            if isinstance(self.xmoe_expert_embeddings, nn.ParameterList)
            else [self.xmoe_expert_embeddings]
        )
        for expert_embeddings in embeddings:
            nn.init.normal_(expert_embeddings)
            with torch.no_grad():
                expert_embeddings.copy_(
                    F.normalize(expert_embeddings, dim=-1) * self.xmoe_router_init_norm
                )

    def _xmoe_projection_and_logits(self, x, idx=None):
        if not self.use_xmoe_router:
            return None, None
        if isinstance(self.xmoe_router_proj, nn.ModuleList):
            projected = self.xmoe_router_proj[idx](x)
            expert_embeddings = self.xmoe_expert_embeddings[idx]
        else:
            projected = self.xmoe_router_proj(x)
            expert_embeddings = self.xmoe_expert_embeddings
        projected = F.normalize(projected, dim=-1)
        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        if self.gating == "laplace":
            return projected, -torch.cdist(projected, expert_embeddings)
        if self.gating == "gaussian":
            return projected, -torch.cdist(projected, expert_embeddings).pow(2)
        if self.gating == "poly":
            dist = torch.cdist(projected, expert_embeddings)
            return projected, 1.0 / (1.0 + dist.pow(self.poly_power))
        if self.gating == "student_t":
            dist = torch.cdist(projected, expert_embeddings)
            return projected, 1.0 / ((1.0 + (dist.pow(2) / self.student_degree)) ** ((self.student_degree + 1) / 2))
        return projected, projected @ expert_embeddings.t()

    def _xmoe_noise_stddev(self, projected, train, noise_epsilon, idx=None):
        if projected is None or self.xmoe_noise_proj is None:
            return None
        if isinstance(self.xmoe_noise_proj, nn.ModuleList):
            raw_noise_stddev = self.xmoe_noise_proj[idx](projected)
        else:
            raw_noise_stddev = self.xmoe_noise_proj(projected)
        return (self.softplus(raw_noise_stddev) + noise_epsilon) * train * self.xmoe_noise_scale

    def _fuse_multihead_logits(self, logits_by_head, idx):
        if logits_by_head.ndim != 3:
            return logits_by_head
        if self.multihead_router_fusion == "learned":
            weights = torch.softmax(self.multihead_router_logits[idx], dim=0).to(
                device=logits_by_head.device,
                dtype=logits_by_head.dtype,
            )
            return torch.einsum("bhe,h->be", logits_by_head, weights)
        return logits_by_head.mean(dim=1)

    def _router_logits_from_weights(self, x, w_gate, idx=None):
        if w_gate.ndim == 2:
            if self.gating == 'softmax':
                return x @ w_gate
            if self.gating == 'laplace':
                return -torch.cdist(x, torch.t(w_gate))
            if self.gating == "poly":
                dist = torch.cdist(x, torch.t(w_gate))
                return 1.0 / (1.0 + dist.pow(self.poly_power))
            if self.gating == 'gaussian':
                return -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)
            if self.gating == "student_t":
                dist = torch.cdist(x, torch.t(w_gate))
                return 1.0 / ((1.0 + ((dist)**2)/self.student_degree)**((self.student_degree + 1)/2))
            if self.gating == "sigmoid":
                return x @ w_gate
            if self.gating == "sigmoid_dist":
                return -torch.cdist(x, torch.t(w_gate))
            raise ValueError(f"Unknown gating={self.gating}")

        if self.gating in {'softmax', 'sigmoid'}:
            return self._fuse_multihead_logits(torch.einsum("bd,hde->bhe", x, w_gate), idx)

        flat_gate = w_gate.permute(0, 2, 1).reshape(-1, w_gate.size(1))
        dist = torch.cdist(x, flat_gate).view(x.size(0), w_gate.size(0), w_gate.size(2))
        if self.gating == 'laplace':
            per_head = -dist
        elif self.gating == "poly":
            per_head = 1.0 / (1.0 + dist.pow(self.poly_power))
        elif self.gating == 'gaussian':
            per_head = -dist.pow(2)
        elif self.gating == "student_t":
            per_head = 1.0 / ((1.0 + (dist.pow(2) / self.student_degree)) ** ((self.student_degree + 1) / 2))
        elif self.gating == "sigmoid_dist":
            per_head = -dist
        else:
            raise ValueError(f"Unknown gating={self.gating}")
        return self._fuse_multihead_logits(per_head, idx)

    def _router_noise_from_weights(self, x, w_noise, train, noise_epsilon, idx=None):
        if w_noise.ndim == 2:
            raw_noise_stddev = self._sanitize_router_tensor(x @ w_noise, clamp_value=1e2)
        else:
            raw_by_head = self._sanitize_router_tensor(torch.einsum("bd,hde->bhe", x, w_noise), clamp_value=1e2)
            raw_noise_stddev = self._fuse_multihead_logits(raw_by_head, idx)
        noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train * self._current_router_noise_scale())
        return torch.nan_to_num(noise_stddev, nan=noise_epsilon, posinf=1e2, neginf=noise_epsilon).clamp_min(noise_epsilon)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        if self.router_type == 'disjoint':
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.disjoint_k
        else:
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)

        safe_noise_stddev = torch.nan_to_num(
            noise_stddev,
            nan=1e-2,
            posinf=1e2,
            neginf=1e-2,
        ).clamp_min(1e-6)
        standardized_if_in = self._sanitize_router_tensor((clean_values - threshold_if_in) / safe_noise_stddev)
        standardized_if_out = self._sanitize_router_tensor((clean_values - threshold_if_out) / safe_noise_stddev)
        prob_if_in = normal.cdf(standardized_if_in)
        prob_if_out = normal.cdf(standardized_if_out)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _instruction_logits(self, instruction_embedding, batch_size, device, dtype, idx=None):
        if self.instruction_router is None or instruction_embedding is None:
            return None

        instruction_embedding = self._prepare_instruction_embedding(
            instruction_embedding,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        if idx is None:
            logits = self.instruction_router(instruction_embedding)
        else:
            logits = self.instruction_router[idx](instruction_embedding)
        return logits * self.instruction_router_scale

    def _prepare_instruction_embedding(self, instruction_embedding, batch_size, device, dtype):
        instruction_embedding = instruction_embedding.to(device=device, dtype=dtype)
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0)
        if instruction_embedding.size(0) == 1 and batch_size != 1:
            instruction_embedding = instruction_embedding.expand(batch_size, -1)
        elif instruction_embedding.size(0) != batch_size:
            raise ValueError(
                f"instruction_embedding batch size {instruction_embedding.size(0)} "
                f"does not match router batch size {batch_size}"
            )
        return instruction_embedding

    def _fuse_instruction_input(self, x, instruction_embedding, idx=None):
        if self.instruction_input_proj is None or instruction_embedding is None:
            return x

        instruction_embedding = self._prepare_instruction_embedding(
            instruction_embedding,
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype,
        )
        if idx is None:
            instruction_delta = self.instruction_input_proj(instruction_embedding)
        else:
            instruction_delta = self.instruction_input_proj[idx](instruction_embedding)
        return x + self.instruction_router_scale * instruction_delta

    def _prepare_condition_embedding(self, embedding, batch_size, device, dtype, name):
        if embedding is None:
            return None
        embedding = embedding.to(device=device, dtype=dtype)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.size(0) == 1 and batch_size != 1:
            embedding = embedding.expand(batch_size, -1)
        elif embedding.size(0) != batch_size:
            raise ValueError(
                f"{name} batch size {embedding.size(0)} does not match router batch size {batch_size}"
            )
        return embedding

    def _sanitize_router_tensor(self, tensor, clamp_value=1e4):
        if tensor is None:
            return None
        return torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=clamp_value,
            neginf=-clamp_value,
        ).clamp(min=-clamp_value, max=clamp_value)

    def _fuse_task_input(self, x, task_embedding, idx=None):
        if self.task_input_proj is None or task_embedding is None:
            return x
        task_embedding = self._prepare_condition_embedding(
            task_embedding,
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype,
            name="task_embedding",
        )
        if idx is None:
            task_delta = self.task_input_proj(task_embedding)
        else:
            task_delta = self.task_input_proj[idx](task_embedding)
        return x + task_delta

    def _fuse_modality_mask_input(self, x, modality_mask_embedding, idx=None):
        if self.mask_input_proj is None or modality_mask_embedding is None:
            return x
        modality_mask_embedding = self._prepare_condition_embedding(
            modality_mask_embedding,
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype,
            name="modality_mask_embedding",
        )
        if idx is None:
            mask_delta = self.mask_input_proj(modality_mask_embedding)
        else:
            mask_delta = self.mask_input_proj[idx](modality_mask_embedding)
        return x + mask_delta

    def _fuse_interaction_input(self, x, interaction_features, idx=None):
        if self.interaction_input_proj is None or interaction_features is None:
            return x
        interaction_features = self._prepare_condition_embedding(
            interaction_features,
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype,
            name="interaction_features",
        )
        if idx is None:
            interaction_delta = self.interaction_input_proj(interaction_features)
        else:
            interaction_delta = self.interaction_input_proj[idx](interaction_features)
        interaction_delta = self._sanitize_router_tensor(
            interaction_delta * self.interaction_router_scale,
            clamp_value=1e2,
        )
        if self.interaction_router_only:
            return interaction_delta
        return x + interaction_delta

    def _modality_subset_prior_logits(self, modality_mask_embedding, batch_size, device, dtype, idx=None):
        if self.mask_prior_proj is None or modality_mask_embedding is None:
            return None
        modality_mask_embedding = self._prepare_condition_embedding(
            modality_mask_embedding,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name="modality_mask_embedding",
        )
        if idx is None:
            prior_logits = self.mask_prior_proj(modality_mask_embedding)
        else:
            prior_logits = self.mask_prior_proj[idx](modality_mask_embedding)
        scale = self.modality_subset_prior_log_scale.exp().to(device=device, dtype=dtype)
        return prior_logits * scale

    def _mohave_group_prior_logits(self, x, clean_logits, idx=None, interaction_features=None):
        if self.mohave_group_router is None or idx is None:
            return None
        group_input = x
        if self.mohave_use_interaction_group:
            if interaction_features is None:
                extra = x.new_zeros((x.size(0), max(self.interaction_feature_dim, 0)))
            else:
                extra = interaction_features.to(device=x.device, dtype=x.dtype)
            group_input = torch.cat([x, extra], dim=-1)
        group_logits = self.mohave_group_router[idx](group_input)
        group_probs = torch.softmax(group_logits, dim=-1)
        expert_group_ids = self.mohave_expert_group_ids.to(device=x.device)
        expert_prior = torch.gather(
            group_probs,
            1,
            expert_group_ids.unsqueeze(0).expand(x.size(0), -1),
        )
        scale = torch.exp(self.mohave_group_log_scale).to(device=x.device, dtype=clean_logits.dtype)
        if self.gating in {"poly", "student_t", "sigmoid", "sigmoid_dist"}:
            prior = 1.0 + scale * expert_prior.to(dtype=clean_logits.dtype)
        else:
            prior = scale * torch.log(expert_prior.to(dtype=clean_logits.dtype).clamp_min(1e-12))
        with torch.no_grad():
            safe_probs = group_probs.detach().float().clamp_min(1e-12)
            entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
            self._specialization_metric_records.append({
                "mohave_group_entropy": entropy.mean().detach(),
                "mohave_top_group_freq": group_probs.argmax(dim=-1).float().mean().detach(),
                "mohave_group_prior_abs": prior.detach().float().abs().mean(),
            })
        return prior

    def _apply_task_expert_modulation(self, expert_outputs, task_embedding, dispatcher, modality_idx=None):
        if self.task_expert_modulators is None or task_embedding is None or dispatcher is None or not expert_outputs:
            return expert_outputs
        base_task_embedding = task_embedding.to(dtype=expert_outputs[0].dtype, device=expert_outputs[0].device)
        offset = 0
        modulated_outputs = []
        for expert_idx, expert_output in enumerate(expert_outputs):
            part_size = int(dispatcher._part_sizes[expert_idx]) if expert_idx < len(dispatcher._part_sizes) else int(expert_output.size(0))
            if expert_output.numel() == 0:
                modulated_outputs.append(expert_output)
                continue
            batch_indices = dispatcher._batch_index[offset: offset + part_size].to(base_task_embedding.device)
            offset += part_size
            task_cond = base_task_embedding.index_select(0, batch_indices)
            if self.router_type == 'disjoint':
                modulation = self.task_expert_modulators[modality_idx][expert_idx](task_cond)
            else:
                modulation = self.task_expert_modulators[expert_idx](task_cond)
            if self.task_expert_modulation_type == "film":
                gamma, beta = modulation.chunk(2, dim=-1)
                modulated = expert_output * (1.0 + gamma) + beta
            else:
                modulated = expert_output + modulation
            modulated_outputs.append(modulated)
        return modulated_outputs

    def _profile_bank_for_logits(self, num_logits, device, dtype):
        profiles = self.semantic_profile_embeddings.to(device=device, dtype=dtype)
        if self.semantic_profile_proj is not None:
            profiles = self.semantic_profile_proj(profiles)
        if profiles.size(0) >= num_logits:
            return profiles[:num_logits]
        repeats = int(np.ceil(num_logits / profiles.size(0)))
        return profiles.repeat(repeats, 1)[:num_logits]

    def _semantic_logits(self, x, idx=None, num_logits=None):
        if self.semantic_input_proj is None or self.semantic_profile_embeddings is None:
            return None

        if idx is None:
            projected = self.semantic_input_proj(x)
        else:
            projected = self.semantic_input_proj[idx](x)

        profiles = self._profile_bank_for_logits(
            num_logits=num_logits,
            device=x.device,
            dtype=projected.dtype,
        )
        projected = F.normalize(projected, dim=-1)
        profiles = F.normalize(profiles, dim=-1)
        return (projected @ profiles.t()) * self.semantic_profile_scale

    def _semantic_poly_kernel(self, semantic_logits):
        scale = self.semantic_profile_scale
        if scale != 0:
            cosine = semantic_logits / scale
        else:
            cosine = semantic_logits
        cosine = cosine.clamp(-1.0, 1.0)
        distance = (2.0 - 2.0 * cosine).clamp_min(0.0).sqrt()
        return 1.0 / (1.0 + distance.pow(self.poly_power))

    def _apply_semantic_logits(self, logits, x, idx=None):
        semantic_logits = self._semantic_logits(x, idx=idx, num_logits=logits.size(1))
        if semantic_logits is None:
            return logits, None
        if self.gating in ["poly", "student_t", "sigmoid", "sigmoid_dist"]:
            if self.semantic_profile_fusion == "replace" and self.gating == "poly":
                return self._semantic_poly_kernel(semantic_logits), semantic_logits
            semantic_multiplier = torch.exp(semantic_logits)
            if self.semantic_profile_fusion == "replace":
                return semantic_multiplier, semantic_logits
            return logits * semantic_multiplier, semantic_logits
        if self.semantic_profile_fusion == "replace":
            return semantic_logits, semantic_logits
        return logits + semantic_logits, semantic_logits

    def _external_semantic_logits(self, semantic_profile_logits, logits, modality=None):
        if not self.use_semantic_expert_profiles:
            return None
        if semantic_profile_logits is None:
            return None
        if modality is not None and modality not in self.semantic_profile_modalities:
            return None

        external_logits = semantic_profile_logits.to(device=logits.device, dtype=logits.dtype)
        if external_logits.size(1) < logits.size(1):
            repeats = int(np.ceil(logits.size(1) / external_logits.size(1)))
            external_logits = external_logits.repeat(1, repeats)
        external_logits = external_logits[:, :logits.size(1)]
        return external_logits * self.semantic_profile_scale

    def _apply_external_semantic_logits(self, logits, semantic_profile_logits, modality=None):
        external_logits = self._external_semantic_logits(semantic_profile_logits, logits, modality=modality)
        if external_logits is None:
            return logits
        if self.gating in ["poly", "student_t", "sigmoid", "sigmoid_dist"]:
            if self.semantic_profile_fusion == "replace" and self.gating == "poly":
                return self._semantic_poly_kernel(external_logits)
            external_multiplier = torch.exp(external_logits)
            if self.semantic_profile_fusion == "replace":
                return external_multiplier
            return logits * external_multiplier
        if self.semantic_profile_fusion == "replace":
            return external_logits
        return logits + external_logits

    def _prototype_bank(self, idx=None):
        if isinstance(self.router_prototypes, nn.ParameterList):
            return self.router_prototypes[idx]
        return self.router_prototypes

    def _prototype_logits(self, x, idx=None):
        if not self.use_prototype_router:
            return None
        if isinstance(self.prototype_query_proj, nn.ModuleList):
            z = self.prototype_query_proj[idx](x)
        else:
            z = self.prototype_query_proj(x)
        z = F.normalize(z, dim=-1)
        prototypes = F.normalize(self._prototype_bank(idx=idx), dim=-1)
        if self.gating == "poly":
            dist = torch.cdist(z, prototypes)
            return 1.0 / (1.0 + dist.pow(self.poly_power))
        if self.gating == "student_t":
            dist = torch.cdist(z, prototypes)
            return 1.0 / ((1.0 + (dist.pow(2) / self.student_degree)) ** ((self.student_degree + 1) / 2))
        if self.gating in ["laplace", "gaussian", "sigmoid_dist"]:
            dist = torch.cdist(z, prototypes)
            if self.gating == "gaussian":
                return -dist.pow(2)
            if self.gating == "sigmoid_dist":
                return torch.sigmoid(-dist / max(self.prototype_router_temperature, 1e-6))
            return -dist
        return (z @ prototypes.t()) / max(self.prototype_router_temperature, 1e-6)

    def _prototype_orthogonality_loss(self):
        if not self.use_prototype_router or self.prototype_router_orth_coef <= 0:
            return 0
        banks = self.router_prototypes if isinstance(self.router_prototypes, nn.ParameterList) else [self.router_prototypes]
        loss = 0
        for bank in banks:
            prototypes = F.normalize(bank, dim=-1)
            sim = prototypes @ prototypes.t()
            eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
            loss = loss + (sim - eye).pow(2).mean()
        return loss * self.prototype_router_orth_coef

    def _expert_signature(self, expert):
        if isinstance(expert, LoRAExpert):
            return expert.lora_A.weight.flatten()
        if isinstance(expert, MLP):
            return expert.fc1.weight.flatten()
        params = [p.flatten() for p in expert.parameters() if p.requires_grad and p.ndim >= 2]
        if not params:
            return None
        return torch.cat(params)

    def _expert_bank_orthogonality_loss(self, expert_bank):
        signatures = []
        for expert in expert_bank:
            signature = self._expert_signature(expert)
            if signature is not None:
                signatures.append(F.normalize(signature, dim=0))
        if len(signatures) < 2:
            return 0
        matrix = torch.stack(signatures, dim=0)
        sim = matrix @ matrix.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        return (sim - eye).pow(2).mean()

    def _expert_orthogonality_loss(self):
        if self.expert_orth_coef <= 0:
            return 0
        if self.router_type == 'disjoint':
            losses = [self._expert_bank_orthogonality_loss(bank) for bank in self.experts]
            losses = [loss for loss in losses if isinstance(loss, torch.Tensor)]
            if not losses:
                return 0
            return torch.stack(losses).mean() * self.expert_orth_coef
        return self._expert_bank_orthogonality_loss(self.experts) * self.expert_orth_coef

    def _get_logits(
        self,
        x,
        train,
        noise_epsilon,
        idx=None,
        instruction_embedding=None,
        semantic_profile_logits=None,
        modality=None,
        task_embedding=None,
        task_name=None,
        modality_mask_embedding=None,
        interaction_features=None,
    ):
        if self.router_zero_input:
            x = torch.zeros_like(x)
        x = self._fuse_task_input(x, task_embedding, idx=idx)
        x = self._fuse_modality_mask_input(x, modality_mask_embedding, idx=idx)
        x = self._fuse_interaction_input(x, interaction_features, idx=idx)
        x = self._fuse_instruction_input(x, instruction_embedding, idx=idx)
        x = self._sanitize_router_tensor(x, clamp_value=1e2)
        router_input_for_diag = x.detach().float().cpu()
        bias = None
        if idx is not None:
            w_gate = self.w_gate[idx].to(x.device)
            w_noise = self.w_noise[idx].to(x.device)
            if self.use_bias and self.bias is not None:
                if self.router_type in ['permod', 'disjoint']:
                    bias = self.bias[idx]
                else:
                    bias = self.bias
        else:
            w_gate = self.w_gate
            w_noise = self.w_noise
            if self.use_bias and self.bias is not None:
                bias = self.bias

        if self.use_temp and self.log_tau is not None:
            temp = self.log_tau.exp()
        elif self.router_temperature != 1.0:
            temp = torch.tensor(self.router_temperature, device=x.device, dtype=x.dtype)
        else:
            temp = None

        # ---------- Compute clean logits ----------
        prototype_logits = self._prototype_logits(x, idx=idx)
        xmoe_projected = None
        if self.task_specific_router is not None:
            if task_name is None:
                raise ValueError("task_name is required when --use_task_specific_router_heads is enabled")
            if idx is None:
                clean_logits = self.task_specific_router(x, task_name)
            else:
                clean_logits = self.task_specific_router[idx](x, task_name)
        elif prototype_logits is not None:
            clean_logits = prototype_logits
        elif self.use_xmoe_router:
            xmoe_projected, clean_logits = self._xmoe_projection_and_logits(x, idx=idx)
        else:
            clean_logits = self._router_logits_from_weights(x, w_gate, idx=idx)

        if self.gating == "sigmoid":
            raw = clean_logits
            if temp is not None:
                raw = raw / temp
            clean_logits = torch.sigmoid(raw)
        elif self.gating == "sigmoid_dist":
            dist = -clean_logits
            if temp is not None:
                dist = dist / temp
            clean_logits = torch.sigmoid(-dist)

        learned_logits_before_semantic = clean_logits
        clean_logits, semantic_logits = self._apply_semantic_logits(clean_logits, x, idx=idx)
        if semantic_logits is not None:
            learned_mag = learned_logits_before_semantic.detach().float().abs().mean()
            semantic_mag = semantic_logits.detach().float().abs().mean()
            denom = learned_mag + semantic_mag + 1e-12
            self._specialization_metric_records.append({
                "semantic_logit_abs": semantic_mag,
                "learned_logit_abs": learned_mag,
                "semantic_contribution_ratio": semantic_mag / denom,
            })
        clean_logits = self._apply_external_semantic_logits(clean_logits, semantic_profile_logits, modality=modality)

        instruction_logits = self._instruction_logits(
            instruction_embedding,
            batch_size=x.size(0),
            device=x.device,
            dtype=clean_logits.dtype,
            idx=idx,
        )
        if instruction_logits is not None:
            clean_logits = clean_logits + instruction_logits

        prior_logits = self._modality_subset_prior_logits(
            modality_mask_embedding,
            batch_size=x.size(0),
            device=x.device,
            dtype=clean_logits.dtype,
            idx=idx,
        )
        if prior_logits is not None:
            clean_logits = clean_logits + prior_logits
            prior_mag = prior_logits.detach().float().abs().mean()
            learned_mag = learned_logits_before_semantic.detach().float().abs().mean()
            denom = learned_mag + prior_mag + 1e-12
            self._specialization_metric_records.append({
                "modality_subset_prior_abs": prior_mag,
                "patient_logit_abs": learned_mag,
                "modality_subset_prior_ratio": prior_mag / denom,
            })

        mohave_prior = self._mohave_group_prior_logits(
            x,
            clean_logits,
            idx=idx,
            interaction_features=interaction_features,
        )
        if mohave_prior is not None:
            if self.gating in {"poly", "student_t", "sigmoid", "sigmoid_dist"}:
                clean_logits = clean_logits * mohave_prior
            else:
                clean_logits = clean_logits + mohave_prior

        # ---------- Noisy gating ----------
        if self.noisy_gating and prototype_logits is None and self.use_xmoe_router:
            noise_stddev = self._xmoe_noise_stddev(xmoe_projected, train, noise_epsilon, idx=idx)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            if temp is not None:
                clean_logits = clean_logits / temp
                noisy_logits = noisy_logits / temp
            logits = noisy_logits
        elif self.noisy_gating and prototype_logits is None:
            noise_stddev = self._router_noise_from_weights(x, w_noise, train, noise_epsilon, idx=idx)

            if self.gating == 'poly':
                if self.poly_noise_mode == "score" or w_gate.ndim == 3:
                    noise = torch.randn_like(clean_logits) * noise_stddev
                    noisy_logits = clean_logits + noise
                else:
                    dist = torch.cdist(x, torch.t(w_gate))
                    noise = torch.randn_like(dist) * noise_stddev
                    noisy_dist = dist + noise
                    noisy_logits = 1.0 / (1.0 + noisy_dist.pow(self.poly_power))
                    noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                    noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                    if instruction_logits is not None:
                        noisy_logits = noisy_logits + instruction_logits
                    if mohave_prior is not None:
                        noisy_logits = noisy_logits * mohave_prior
                logits = noisy_logits
            elif self.gating == "student_t":
                if w_gate.ndim == 3:
                    noise = torch.randn_like(clean_logits) * noise_stddev
                    noisy_logits = clean_logits + noise
                    logits = noisy_logits
                else:
                    dist = torch.cdist(x, torch.t(w_gate))
                    noise = torch.randn_like(dist) * noise_stddev
                    noisy_dist = dist + noise
                    noisy_logits = 1.0 / ((1.0 + ((noisy_dist)**2)/self.student_degree)**((self.student_degree + 1)/2))
                    noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                    noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                    if instruction_logits is not None:
                        noisy_logits = noisy_logits + instruction_logits
                    if mohave_prior is not None:
                        noisy_logits = noisy_logits * mohave_prior
                    logits = noisy_logits
            elif self.gating == "sigmoid":
                score = self._router_logits_from_weights(x, w_gate, idx=idx)
                if instruction_logits is not None:
                    score = score + instruction_logits
                noise = torch.randn_like(score) * noise_stddev
                noisy_score = score + noise
                if temp is not None:
                    noisy_score = noisy_score / temp
                noisy_logits = torch.sigmoid(noisy_score)
                noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                if mohave_prior is not None:
                    noisy_logits = noisy_logits * mohave_prior
                logits = noisy_logits
            elif self.gating == "sigmoid_dist":
                if w_gate.ndim == 3:
                    score = self._router_logits_from_weights(x, w_gate, idx=idx)
                    noise = torch.randn_like(score) * noise_stddev
                    noisy_score = score + noise
                    if temp is not None:
                        noisy_score = noisy_score / temp
                    noisy_logits = torch.sigmoid(noisy_score)
                    logits = noisy_logits
                else:
                    dist = torch.cdist(x, torch.t(w_gate))
                    noise = torch.randn_like(-dist) * noise_stddev
                    noisy_dist = dist + noise
                    if temp is not None:
                        noisy_dist = noisy_dist / temp
                    noisy_logits = torch.sigmoid(noisy_dist)
                    noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                    noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                    if instruction_logits is not None:
                        noisy_logits = noisy_logits + instruction_logits
                    if mohave_prior is not None:
                        noisy_logits = noisy_logits * mohave_prior
                    logits = noisy_logits
            else:
                noise = torch.randn_like(clean_logits) * noise_stddev
                noisy_logits = clean_logits + noise
                if temp is not None:
                    clean_logits = clean_logits / temp
                    noisy_logits = noisy_logits / temp
                logits = noisy_logits
        else:
            noise_stddev = torch.zeros_like(clean_logits)
            if self.gating == 'softmax' and temp is not None:
                clean_logits = clean_logits / temp
            noisy_logits = clean_logits
            logits = clean_logits

        if bias is not None:
            if self.router_bias_mode == "add":
                logits = logits + bias
                clean_logits = clean_logits + bias
                noisy_logits = noisy_logits + bias
            else:
                logits = logits * torch.exp(bias)
                clean_logits = clean_logits * torch.exp(bias)
                noisy_logits = noisy_logits * torch.exp(bias)
        logits = self._sanitize_router_tensor(logits)
        clean_logits = self._sanitize_router_tensor(clean_logits)
        noisy_logits = self._sanitize_router_tensor(noisy_logits)
        noise_stddev = torch.nan_to_num(
            noise_stddev,
            nan=noise_epsilon,
            posinf=1e2,
            neginf=noise_epsilon,
        ).clamp_min(noise_epsilon)
        return logits, clean_logits, noisy_logits, noise_stddev, router_input_for_diag

    def _dense_gating(self, logits):
        if self.gating == 'softmax':
            if self.normalized:
                gates = self.softmax(logits)
            else:
                gates = torch.exp(logits)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            if self.normalized:
                gates = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
            else:
                gates = torch.exp(logits)
        elif self.gating == 'poly' or self.gating == "student_t" or self.gating == "sigmoid" or self.gating == "sigmoid_dist":
            logits = logits.clamp_min(1e-12)
            if self.normalized:
                gates = logits / logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                gates = logits
        else:
            gates = self.softmax(logits)
        load = self._gates_to_load(gates)
        return gates, load

    def _full_router_probs_for_dynamic_top_k(self, logits):
        if self.gating == 'softmax':
            return self.softmax(logits / self.router_temperature)
        if self.gating == 'laplace' or self.gating == 'gaussian':
            return torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        if self.gating == 'poly' or self.gating == "student_t" or self.gating == "sigmoid" or self.gating == "sigmoid_dist":
            scores = logits.clamp_min(1e-12)
            return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return self.softmax(logits)

    def _dynamic_top_k_values(self, logits, base_k, max_k):
        full_probs = self._full_router_probs_for_dynamic_top_k(logits).detach()
        safe_probs = full_probs.clamp_min(1e-12)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=1)
        max_entropy = math.log(max(int(full_probs.size(1)), 2))
        normalized_entropy = entropy / max_entropy
        top1 = full_probs.max(dim=1).values

        min_k = min(max(self.dynamic_top_k_min, 1), max_k)
        base_k = min(max(base_k, min_k), max_k)
        max_k = min(max(self.dynamic_top_k_max, base_k), max_k)

        k_values = torch.full(
            (logits.size(0),),
            base_k,
            dtype=torch.long,
            device=logits.device,
        )
        high_entropy = normalized_entropy >= self.dynamic_top_k_entropy_threshold
        high_confidence = top1 >= self.dynamic_top_k_confidence_threshold
        k_values = torch.where(high_entropy, torch.full_like(k_values, max_k), k_values)
        k_values = torch.where(high_confidence, torch.full_like(k_values, min_k), k_values)
        self._specialization_metric_records.append({
            "dynamic_top_k_mean": k_values.float().mean().detach(),
            "dynamic_top_k_min_frac": (k_values == min_k).float().mean().detach(),
            "dynamic_top_k_max_frac": (k_values == max_k).float().mean().detach(),
            "dynamic_top_k_top1_confidence": top1.mean().detach(),
            "dynamic_top_k_entropy_norm": normalized_entropy.mean().detach(),
        })
        return k_values

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        # For disjoint routing, logits only cover the local expert group for one
        # modality, so use logits.size(1) instead of the global expert count.
        num_available_experts = logits.size(1)
        effective_k = min(k, num_available_experts)
        selected_k = min(
            max(self.dynamic_top_k_max, effective_k),
            num_available_experts,
        ) if self.use_dynamic_top_k else effective_k
        top_m = effective_k
        if self.use_dynamic_top_k:
            top_m = selected_k
        if self.router_topk_mode == "k_plus_1" and self.noisy_gating:
            top_m = min(top_m + 1, num_available_experts)
        top_logits, top_indices = logits.topk(min(top_m, num_available_experts), dim=1)
        top_k_logits = top_logits[:, :selected_k]
        top_k_indices = top_indices[:, :selected_k]
        if self.use_dynamic_top_k:
            dynamic_k = self._dynamic_top_k_values(logits, effective_k, selected_k)
            position = torch.arange(selected_k, device=logits.device).unsqueeze(0)
            dynamic_mask = position < dynamic_k.unsqueeze(1)
        else:
            dynamic_mask = None
        if self.gating == 'softmax':
            if self.gate_normalization in ["full", "full_renorm"]:
                full_probs = self.softmax(logits / self.router_temperature)
                top_k_gates = torch.gather(full_probs, 1, top_k_indices)
                if dynamic_mask is not None:
                    top_k_gates = top_k_gates * dynamic_mask.to(dtype=top_k_gates.dtype)
                if self.gate_normalization == "full_renorm" and self.normalized:
                    top_k_gates = top_k_gates / top_k_gates.sum(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                if self.normalized:
                    if dynamic_mask is not None:
                        masked_logits = top_k_logits.masked_fill(~dynamic_mask, torch.finfo(top_k_logits.dtype).min)
                        top_k_gates = self.softmax(masked_logits)
                    else:
                        top_k_gates = self.softmax(top_k_logits)
                else:
                    top_k_gates = torch.exp(top_k_logits)
                    if dynamic_mask is not None:
                        top_k_gates = top_k_gates * dynamic_mask.to(dtype=top_k_gates.dtype)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            if self.normalized:
                if dynamic_mask is not None:
                    masked_logits = top_k_logits.masked_fill(~dynamic_mask, torch.finfo(top_k_logits.dtype).min)
                    top_k_gates = torch.exp(masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True))
                else:
                    top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))
            else:
                top_k_gates = torch.exp(top_k_logits)
                if dynamic_mask is not None:
                    top_k_gates = top_k_gates * dynamic_mask.to(dtype=top_k_gates.dtype)
            
        elif self.gating == 'poly' or self.gating == "student_t" or self.gating == "sigmoid" or self.gating == "sigmoid_dist":
            top_k_logits = top_k_logits.clamp_min(1e-12)
            if dynamic_mask is not None:
                top_k_logits = top_k_logits * dynamic_mask.to(dtype=top_k_logits.dtype)
            if self.normalized:
                top_k_gates = top_k_logits / top_k_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                top_k_gates = top_k_logits
            
        # zeros = torch.zeros_like(logits, requires_grad=True)
        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        has_positive_noise = (
            noise_stddev is not None
            and torch.is_tensor(noise_stddev)
            and bool(torch.any(noise_stddev > 0).detach().cpu().item())
        )
        if self.noisy_gating and not self.use_dynamic_top_k and effective_k < num_available_experts and has_positive_noise:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def set_r2t2_adaptor(self, adaptor, layer_id):
        self.r2t2_adaptor = adaptor
        self.r2t2_layer_id = int(layer_id)

    def clear_r2t2_adaptor(self):
        self.r2t2_adaptor = None
        self.r2t2_layer_id = None

    def _apply_r2t2_rerouting(self, gates, train):
        if train or self.r2t2_adaptor is None or self.r2t2_layer_id is None:
            return gates, False
        if self._last_router_inputs_for_diag is None:
            return gates, False
        rerouted = self.r2t2_adaptor.reroute(
            self.r2t2_layer_id,
            self._last_router_inputs_for_diag,
            gates,
        )
        return rerouted, True

    def noisy_top_k_gating(
        self,
        x,
        train,
        noise_epsilon=1e-2,
        modalities=None,
        instruction_embedding=None,
        semantic_profile_logits=None,
        task_embedding=None,
        task_name=None,
        modality_mask_embedding=None,
        interaction_features=None,
    ):
        """Multimodal noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        if self.router_type == 'joint':
            if isinstance(x, list):
                embeddings = torch.concat(x, dim=1)
            else:
                embeddings = x
            all_logits = self._get_logits(
                embeddings,
                train,
                noise_epsilon,
                instruction_embedding=instruction_embedding,
                semantic_profile_logits=semantic_profile_logits,
                task_embedding=task_embedding,
                task_name=task_name,
                modality_mask_embedding=modality_mask_embedding,
                interaction_features=interaction_features,
            )
            logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
            if self.prototype_router_dense or self._use_dense_warmup():
                gates, load = self._dense_gating(logits)
            else:
                gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
            self._last_router_logits_for_loss = logits
            self._last_router_inputs_for_diag = all_logits[4]
            self._last_clean_logits_for_diag = clean_logits.detach().float().cpu()
            return gates, load
        else:
            all_gates, all_loads = [], []
            all_logits_for_loss = []
            all_router_inputs_for_diag = []
            all_clean_logits_for_diag = []
            for i in range(self.num_modalities):
                modality = modalities[i] if modalities is not None else None
                all_logits = self._get_logits(
                    x[i],
                    train,
                    noise_epsilon,
                    idx=i,
                    instruction_embedding=instruction_embedding,
                    semantic_profile_logits=semantic_profile_logits,
                    modality=modality,
                    task_embedding=task_embedding,
                    task_name=task_name,
                    modality_mask_embedding=modality_mask_embedding,
                    interaction_features=interaction_features,
                )
                logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
                if self.router_type == 'permod':
                    if self.prototype_router_dense or self._use_dense_warmup():
                        gates, load = self._dense_gating(logits)
                    else:
                        gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
                else:
                    if self.prototype_router_dense or self._use_dense_warmup():
                        gates, load = self._dense_gating(logits)
                    else:
                        gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.disjoint_k)
                all_gates.append(gates)
                all_loads.append(load)
                all_logits_for_loss.append(logits)
                all_router_inputs_for_diag.append(all_logits[4])
                all_clean_logits_for_diag.append(clean_logits.detach().float().cpu())
            self._last_router_logits_for_loss = all_logits_for_loss
            self._last_router_inputs_for_diag = all_router_inputs_for_diag
            self._last_clean_logits_for_diag = all_clean_logits_for_diag
            return all_gates, all_loads

    def _compute_router_z_loss(self):
        if self.router_z_loss_coef <= 0 or self._last_router_logits_for_loss is None:
            return 0

        logits_list = (
            self._last_router_logits_for_loss
            if isinstance(self._last_router_logits_for_loss, list)
            else [self._last_router_logits_for_loss]
        )
        losses = []
        for logits in logits_list:
            stable_logits = logits.float()
            if self.router_z_loss_type == "squared_logits":
                losses.append(stable_logits.pow(2).mean())
            else:
                losses.append(torch.logsumexp(stable_logits, dim=-1).pow(2).mean())
        return torch.stack(losses).mean() * self.router_z_loss_coef

    def _compute_router_entropy_loss(self, gates):
        if self.router_entropy_coef <= 0:
            return 0

        gates_list = gates if isinstance(gates, list) else [gates]
        entropies = []
        for gate_tensor in gates_list:
            safe_gates = gate_tensor.clamp_min(1e-12)
            entropies.append(-(safe_gates * safe_gates.log()).sum(dim=-1).mean())
        entropy = torch.stack(entropies).mean()
        return -self.router_entropy_coef * entropy

    def _compute_router_variance_loss(self, gates, apply_coef=True):
        if self.router_variance_coef <= 0:
            return 0

        gates_list = gates if isinstance(gates, list) else [gates]
        variances = []
        for gate_tensor in gates_list:
            if gate_tensor.size(0) <= 1:
                continue
            variances.append(gate_tensor.float().var(dim=0, unbiased=False).mean())
        if not variances:
            return 0
        # Negative because the optimizer minimizes the total loss.
        raw_loss = -torch.stack(variances).mean()
        return self.router_variance_coef * raw_loss if apply_coef else raw_loss

    def _selected_output_orthogonality_loss(self, dispatcher, expert_outputs, log_space=True, apply_coef=True):
        if self.output_orth_coef <= 0 or not expert_outputs:
            return 0

        selected = torch.cat(expert_outputs, dim=0)
        if selected.size(0) == 0:
            return 0
        if log_space:
            selected = selected.exp()
        selected = selected.float()

        batch_index = dispatcher._batch_index.to(selected.device)
        losses = []
        cos_abs_values = []
        cos_sq_values = []
        pair_counts = []
        for batch_id in torch.unique(batch_index):
            reps = selected[batch_index == batch_id]
            if reps.size(0) < 2:
                continue
            reps = reps.flatten(start_dim=1)
            pair_losses = []
            for j in range(reps.size(0)):
                for k in range(reps.size(0)):
                    if j == k:
                        continue
                    src_norm = reps[j].pow(2).sum().clamp_min(1e-12).sqrt()
                    dst_norm_sq = reps[k].pow(2).sum().clamp_min(1e-12)
                    dst_norm = dst_norm_sq.sqrt()
                    dot = torch.dot(reps[j], reps[k])
                    cos = dot / (src_norm * dst_norm).clamp_min(1e-12)
                    projection = (dot / dst_norm_sq) * reps[k]
                    pair_losses.append(projection.pow(2).sum())
                    cos_abs_values.append(cos.abs())
                    cos_sq_values.append(cos.pow(2))
            if pair_losses:
                losses.append(torch.stack(pair_losses).mean())
                pair_counts.append(float(len(pair_losses)))
        if not losses:
            return 0
        metric_device = losses[0].device
        self._specialization_metric_records.append({
            "expert_projection_norm": torch.stack(losses).mean().detach(),
            "expert_cos_abs": (torch.stack(cos_abs_values).mean().detach() if cos_abs_values else torch.zeros((), device=metric_device)),
            "expert_cos_sq": (torch.stack(cos_sq_values).mean().detach() if cos_sq_values else torch.zeros((), device=metric_device)),
            "active_expert_pairs": torch.tensor(sum(pair_counts) / max(len(pair_counts), 1), device=metric_device),
        })
        raw_loss = torch.stack(losses).mean()
        return self.output_orth_coef * raw_loss if apply_coef else raw_loss

    def _compute_gate_specialization_metrics(self, gates):
        gates_list = gates if isinstance(gates, list) else [gates]
        metrics = {}
        entropy_vals = []
        margin_vals = []
        batch_var_vals = []
        top1_vals = []
        active_vals = []
        for gate_tensor in gates_list:
            g = gate_tensor.float()
            if g.numel() == 0:
                continue
            safe_g = g.clamp_min(1e-12)
            entropy_vals.append(-(safe_g * safe_g.log()).sum(dim=-1).mean())
            top_vals = torch.topk(g, k=min(2, g.size(-1)), dim=-1).values
            top1_vals.append(top_vals[:, 0].mean())
            if top_vals.size(-1) > 1:
                margin_vals.append((top_vals[:, 0] - top_vals[:, 1]).mean())
            if g.size(0) > 1:
                batch_var_vals.append(g.var(dim=0, unbiased=False).mean())
            active_vals.append((g > 0).float().sum(dim=-1).mean())
        if entropy_vals:
            metrics["gate_entropy"] = torch.stack(entropy_vals).mean().detach()
        if margin_vals:
            metrics["gate_top12_margin"] = torch.stack(margin_vals).mean().detach()
        if batch_var_vals:
            metrics["gate_batch_variance"] = torch.stack(batch_var_vals).mean().detach()
        if top1_vals:
            metrics["gate_top1_weight"] = torch.stack(top1_vals).mean().detach()
        if active_vals:
            metrics["active_experts_per_sample"] = torch.stack(active_vals).mean().detach()
        return metrics

    def _finalize_specialization_diagnostics(self, gates):
        metrics = self._compute_gate_specialization_metrics(gates)
        if self._specialization_metric_records:
            keys = sorted({key for record in self._specialization_metric_records for key in record})
            for key in keys:
                vals = [record[key].float() for record in self._specialization_metric_records if key in record]
                if vals:
                    metrics[key] = torch.stack(vals).mean().detach()
        self.last_specialization_diagnostics = metrics
        return metrics

    def _record_output_orthogonality_loss(self, dispatcher, expert_outputs, log_space=True):
        loss = self._selected_output_orthogonality_loss(
            dispatcher,
            expert_outputs,
            log_space=log_space,
            apply_coef=(self.specialization_loss_mode != "paper"),
        )
        if isinstance(loss, torch.Tensor):
            self._output_orth_losses.append(loss)

    def _consume_output_orthogonality_loss(self, base_loss):
        if not self._output_orth_losses:
            zero = base_loss.new_zeros(())
            return zero
        loss = torch.stack(self._output_orth_losses).mean()
        if self.specialization_loss_mode == "paper":
            target = self._paper_aux_scale_target
            if target is not None:
                scale = target.to(loss.device, loss.dtype) / loss.detach().abs().clamp_min(1e-12)
                loss = loss * scale
            loss = self.output_orth_coef * loss
        return loss

    def _compute_loss(self, gates, loads, loss_coef):
        """Compute load balancing loss across all gates."""
        if self.load_balance_mode == "deepseek_aux":
            loss = self._compute_deepseek_aux_loss(gates)
        else:
            loss = 0
            if isinstance(gates, list):
                for g, l in zip(gates, loads):
                    loss += self.cv_squared(g.sum(0)) + self.cv_squared(l)
            else:
                loss = self.cv_squared(gates.sum(0)) + self.cv_squared(loads)
        balance_component = loss
        proto_loss = self._prototype_orthogonality_loss()
        if isinstance(proto_loss, torch.Tensor):
            loss = loss + proto_loss
        expert_orth_loss = self._expert_orthogonality_loss()
        if isinstance(expert_orth_loss, torch.Tensor):
            loss = loss + expert_orth_loss
        z_loss = self._compute_router_z_loss()
        if isinstance(z_loss, torch.Tensor):
            loss = loss + z_loss
        entropy_loss = self._compute_router_entropy_loss(gates)
        if isinstance(entropy_loss, torch.Tensor):
            loss = loss + entropy_loss
        variance_loss = self._compute_router_variance_loss(
            gates,
            apply_coef=(self.specialization_loss_mode != "paper"),
        )
        if isinstance(variance_loss, torch.Tensor):
            loss = loss + variance_loss
        if self.specialization_loss_mode == "paper":
            target = balance_component.detach().abs().clamp_min(1e-12)
            self._paper_aux_scale_target = target
            scaled_loss = balance_component * self.specialization_aux_coef
            if isinstance(proto_loss, torch.Tensor):
                scaled_loss = scaled_loss + proto_loss
            if isinstance(expert_orth_loss, torch.Tensor):
                scaled_loss = scaled_loss + expert_orth_loss
            if isinstance(z_loss, torch.Tensor):
                scaled_loss = scaled_loss + z_loss
            if isinstance(entropy_loss, torch.Tensor):
                scaled_loss = scaled_loss + entropy_loss
            if isinstance(variance_loss, torch.Tensor):
                scale = target.to(variance_loss.device, variance_loss.dtype) / variance_loss.detach().abs().clamp_min(1e-12)
                scaled_loss = scaled_loss + self.router_variance_coef * variance_loss * scale
        else:
            self._paper_aux_scale_target = None
            scaled_loss = loss * loss_coef
        device = scaled_loss.device
        dtype = scaled_loss.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        self.last_router_aux_components = {
            "balance_unscaled": balance_component.detach(),
            "prototype_orth_unscaled": (proto_loss.detach() if isinstance(proto_loss, torch.Tensor) else zero),
            "expert_orth_unscaled": (expert_orth_loss.detach() if isinstance(expert_orth_loss, torch.Tensor) else zero),
            "z_loss_unscaled": (z_loss.detach() if isinstance(z_loss, torch.Tensor) else zero),
            "entropy_loss_unscaled": (entropy_loss.detach() if isinstance(entropy_loss, torch.Tensor) else zero),
            "router_variance_unscaled": (variance_loss.detach() if isinstance(variance_loss, torch.Tensor) else zero),
            "output_orth_unscaled": zero,
            "total_unscaled": loss.detach(),
            "total_scaled": scaled_loss.detach(),
        }
        return scaled_loss

    def _compute_deepseek_aux_loss(self, gates):
        gates_list = gates if isinstance(gates, list) else [gates]
        losses = []
        for gate_tensor in gates_list:
            gate_float = gate_tensor.float()
            if gate_float.numel() == 0:
                continue
            num_experts = gate_float.size(1)
            selected_frac = (gate_float > 0).float().mean(dim=0)
            score_mean = gate_float.mean(dim=0)
            losses.append((selected_frac * score_mean).sum() * float(num_experts ** 2))
        if not losses:
            if isinstance(gates, list) and gates:
                return gates[0].new_zeros(())
            return gates.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_router_organ_supervision_loss(self, gates, router_organ_targets):
        if (
            not self.use_router_organ_supervision
            or router_organ_targets is None
            or self.router_organ_supervision_coef <= 0
        ):
            return 0

        if isinstance(gates, list):
            gate_tensor = torch.stack(gates, dim=0).mean(dim=0)
        else:
            gate_tensor = gates

        targets = router_organ_targets.to(device=gate_tensor.device, dtype=gate_tensor.dtype)
        if targets.size(1) < gate_tensor.size(1):
            repeats = int(np.ceil(gate_tensor.size(1) / targets.size(1)))
            targets = targets.repeat(1, repeats)
        targets = targets[:, :gate_tensor.size(1)]

        valid = targets.sum(dim=1) > 0
        if valid.sum() == 0:
            return 0

        targets_valid = targets[valid]
        if self.router_organ_supervision_class_balanced:
            class_freq = targets_valid.sum(dim=0).clamp_min(1.0)
            class_weights = class_freq.sum() / class_freq
            class_weights = class_weights / class_weights.mean().clamp_min(1e-12)
            targets_valid = targets_valid * class_weights.unsqueeze(0)

        target_dist = targets_valid / targets_valid.sum(dim=1, keepdim=True).clamp_min(1e-12)
        gates_valid = gate_tensor[valid].clamp_min(1e-12)
        loss = -(target_dist * gates_valid.log()).sum(dim=1).mean()
        return loss * self.router_organ_supervision_coef

    def _compute_semantic_guidance_loss(self, gates):
        if self.semantic_guidance_coef <= 0:
            return 0
        batch_size = gates[0].size(0) if isinstance(gates, list) else gates.size(0)
        reference = gates[0] if isinstance(gates, list) else gates
        target_dist, valid_mask, _, _ = self._prepare_expert_init_targets(
            batch_size=batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        if target_dist is None or valid_mask is None or not bool(valid_mask.any().detach().cpu().item()):
            return 0
        gate_tensor = torch.stack(gates, dim=0).mean(dim=0) if isinstance(gates, list) else gates
        gates_valid = gate_tensor[valid_mask].clamp_min(1e-12)
        target_valid = target_dist[valid_mask]
        loss = -(target_valid * gates_valid.log()).sum(dim=1).mean()
        return loss * self.semantic_guidance_coef


    def _compute_shared_output(self, x):
        """Compute output from shared experts, if any."""
        if self.shared_mlps is None:
            return None

        if self.moe_mixing_space == "linear":
            if self.normalized:
                omega = torch.softmax(self.omega, dim=0)
            else:
                omega = torch.nn.functional.softplus(self.omega)
            out = self.shared_mlps[0](x) * omega[0]
            for i, se in enumerate(self.shared_mlps[1:], 1):
                out = out + se(x) * omega[i]
            return out * self.shared_expert_weight

        if self.normalized:
            log_omega = torch.log_softmax(self.omega, dim=0)
        else:
            log_omega = torch.log(torch.nn.functional.softplus(self.omega) + 1e-12)

        out = self.shared_mlps[0](x) + log_omega[0]
        for i, se in enumerate(self.shared_mlps[1:], 1):
            out = torch.logaddexp(out, se(x) + log_omega[i])
        return out + math.log(max(self.shared_expert_weight, 1e-12))

    def _record_full_expert_output_similarity(self, x, modality_idx=None, experts=None, log_space=True):
        if not self.log_expert_output_diagnostics or experts is None:
            return
        if x is None or x.size(0) == 0:
            return
        records = self.last_expert_output_diagnostics
        if records is None:
            records = []
            self.last_expert_output_diagnostics = records
        with torch.no_grad():
            outputs = []
            norms = []
            for expert in experts:
                out = expert(x)
                if log_space:
                    out = out.exp()
                out = out.float().flatten(start_dim=1)
                norms.append(out.norm(dim=-1).mean())
                outputs.append(out)
            if len(outputs) < 2:
                return
            stacked = torch.stack(outputs, dim=1)
            normalized = F.normalize(stacked, dim=-1)
            sim = torch.einsum("bed,bfd->bef", normalized, normalized).mean(dim=0)
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            offdiag = sim[~eye].mean()
            records.append({
                "modality_idx": modality_idx,
                "sim_matrix": sim.detach().cpu(),
                "offdiag_cos": offdiag.detach().cpu(),
                "expert_norms": torch.stack(norms).detach().cpu(),
            })

    def _apply_shared_semantic_memory(self, expert_inputs):
        if self.shared_semantic_memory is None or not expert_inputs:
            return expert_inputs
        return [
            self._sanitize_router_tensor(
                self.shared_semantic_memory(self._sanitize_router_tensor(expert_in))
            ) if expert_in.numel() > 0 else expert_in
            for expert_in in expert_inputs
        ]

    def _interaction_role_views(self, x_list, modality_idx):
        """Build same-dimensional I2MoE-lite expert views for one modality."""
        if not isinstance(x_list, list) or modality_idx is None:
            return None
        current = x_list[modality_idx]
        if len(x_list) < 2:
            return None
        stacked = torch.stack(x_list, dim=0)
        redundant = stacked.mean(dim=0)
        if len(x_list) == 2:
            others = x_list[1 - modality_idx]
        else:
            other_tensors = [x for idx, x in enumerate(x_list) if idx != modality_idx]
            others = torch.stack(other_tensors, dim=0).mean(dim=0)
        synergy = current * others
        contrast = (current - others).abs()
        return [current, redundant, synergy, contrast]

    def _interaction_expert_inputs(self, dispatcher, x, modality_idx=None, all_modalities=None):
        if not self.use_interaction_experts:
            return dispatcher.dispatch(x)
        role_views = self._interaction_role_views(all_modalities, modality_idx)
        if role_views is None:
            return dispatcher.dispatch(x)
        expert_inputs = []
        offset = 0
        for expert_idx, part_size in enumerate(dispatcher._part_sizes):
            part_size = int(part_size)
            role_view = role_views[expert_idx % len(role_views)]
            if part_size == 0:
                expert_inputs.append(role_view.new_empty((0,) + role_view.shape[1:]))
                continue
            batch_indices = dispatcher._batch_index[offset: offset + part_size].to(role_view.device)
            expert_inputs.append(role_view.index_select(0, batch_indices))
            offset += part_size
        with torch.no_grad():
            if dispatcher._gates.numel() > 0:
                gate_mass = dispatcher._gates.float().mean(dim=0)
                self._specialization_metric_records.append({
                    "interaction_unique_gate": gate_mass[0].detach(),
                    "interaction_redundant_gate": gate_mass[1].detach() if gate_mass.numel() > 1 else gate_mass.new_zeros(()),
                    "interaction_synergy_gate": gate_mass[2].detach() if gate_mass.numel() > 2 else gate_mass.new_zeros(()),
                    "interaction_contrast_gate": gate_mass[3].detach() if gate_mass.numel() > 3 else gate_mass.new_zeros(()),
                })
        return expert_inputs

    def _interaction_role_reweighted_gates(self, gates, modality_idx=None, all_modalities=None):
        if not self.use_interaction_expert_reweighting:
            return gates
        if self.interaction_reweight_heads is None or modality_idx is None:
            return gates
        role_views = self._interaction_role_views(all_modalities, modality_idx)
        if role_views is None:
            return gates
        role_context = torch.cat(role_views, dim=-1)
        logits = self.interaction_reweight_heads[modality_idx](role_context)
        role_weights = torch.softmax(logits, dim=-1).to(dtype=gates.dtype, device=gates.device)
        selected = gates > 0
        effective_gates = gates * role_weights
        selected_mass = effective_gates.sum(dim=-1, keepdim=True)
        fallback_mass = gates.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_gates = torch.where(
            selected_mass > 1e-12,
            effective_gates / selected_mass.clamp_min(1e-12),
            gates / fallback_mass,
        )
        effective_gates = effective_gates.masked_fill(~selected, 0.0)
        with torch.no_grad():
            mean_weights = role_weights.float().mean(dim=0)
            selected_weights = (role_weights * selected.float()).sum(dim=-1)
            self._specialization_metric_records.append({
                "interaction_reweight_unique": mean_weights[0].detach(),
                "interaction_reweight_redundant": mean_weights[1].detach() if mean_weights.numel() > 1 else mean_weights.new_zeros(()),
                "interaction_reweight_synergy": mean_weights[2].detach() if mean_weights.numel() > 2 else mean_weights.new_zeros(()),
                "interaction_reweight_contrast": mean_weights[3].detach() if mean_weights.numel() > 3 else mean_weights.new_zeros(()),
                "interaction_reweight_selected_mass": selected_weights.mean().detach(),
            })
        return effective_gates

    def _compute_routed_output(self, x, gates, modality_idx=None, all_modalities=None):
        """Compute routed expert output for a single modality or joint input."""
        if self.router_type == 'disjoint':
            sub_experts = self.num_experts // self.num_modalities
            dispatcher = SparseDispatcher(sub_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[modality_idx][i](expert_inputs[i]) for i in range(sub_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=modality_idx)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
                self._record_full_expert_output_similarity(x, modality_idx=modality_idx, experts=self.experts[modality_idx], log_space=True)
        elif self.router_type == 'permod':
            gates = self._interaction_role_reweighted_gates(
                gates,
                modality_idx=modality_idx,
                all_modalities=all_modalities,
            )
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = self._interaction_expert_inputs(
                dispatcher,
                x,
                modality_idx=modality_idx,
                all_modalities=all_modalities,
            )
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=modality_idx)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
            self._record_full_expert_output_similarity(x, modality_idx=modality_idx, experts=self.experts, log_space=True)
        else:  # joint
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=None)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
            self._record_full_expert_output_similarity(x, modality_idx=None, experts=self.experts, log_space=True)
        self._record_output_orthogonality_loss(dispatcher, expert_outputs, log_space=True)
        return dispatcher.combine(expert_outputs)

    def _compute_routed_delta(self, x, gates, modality_idx=None):
        """Compute routed low-rank residual deltas for staged shared-LoRA."""
        if self.router_type == 'disjoint':
            sub_experts = self.num_experts // self.num_modalities
            dispatcher = SparseDispatcher(sub_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[modality_idx][i](expert_inputs[i]) for i in range(sub_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=modality_idx)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
            self._record_full_expert_output_similarity(x, modality_idx=modality_idx, experts=self.experts[modality_idx], log_space=False)
        elif self.router_type == 'permod':
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=modality_idx)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
            self._record_full_expert_output_similarity(x, modality_idx=modality_idx, experts=self.experts, log_space=False)
        else:
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs = self._apply_shared_semantic_memory(expert_inputs)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            expert_outputs = self._apply_task_expert_modulation(expert_outputs, self.current_task_embedding, dispatcher, modality_idx=None)
            if not self.training and self.eval_ablate_expert_indices:
                expert_outputs = [
                    out.new_zeros(out.shape) if i in self.eval_ablate_expert_indices else out
                    for i, out in enumerate(expert_outputs)
                ]
            self._record_full_expert_output_similarity(x, modality_idx=None, experts=self.experts, log_space=False)
        self._record_output_orthogonality_loss(dispatcher, expert_outputs, log_space=False)
        return dispatcher.combine_linear(expert_outputs)


    def _combine_shared_and_routed(self, routed_out, shared_out):
        """Combine shared and routed outputs as a mixture within a modality."""
        if not self.training:
            if self.eval_ablate_shared_path:
                shared_out = None
            if self.eval_ablate_routed_path and routed_out is not None:
                routed_out = routed_out.new_zeros(routed_out.shape) if self.moe_mixing_space == "linear" else routed_out.new_full(routed_out.shape, -1e9)
        if shared_out is None:
            self.last_contribution_diagnostics = None
            return routed_out
        with torch.no_grad():
            routed_linear = routed_out.exp() if self.moe_mixing_space != "linear" else routed_out
            shared_linear = shared_out.exp() if self.moe_mixing_space != "linear" else shared_out
            routed_norm_per_sample = routed_linear.float().flatten(start_dim=1).norm(dim=-1)
            shared_norm_per_sample = shared_linear.float().flatten(start_dim=1).norm(dim=-1)
            routed_norm = routed_norm_per_sample.mean()
            shared_norm = shared_norm_per_sample.mean()
            ratio_per_sample = shared_norm_per_sample / (shared_norm_per_sample + routed_norm_per_sample + 1e-12)
            self._specialization_metric_records.append({
                "routed_output_norm": routed_norm.detach(),
                "shared_output_norm": shared_norm.detach(),
                "shared_contribution_ratio": shared_norm.detach() / (shared_norm.detach() + routed_norm.detach() + 1e-12),
            })
            self.last_contribution_diagnostics = {
                "routed_output_norm_per_sample": routed_norm_per_sample.detach().cpu(),
                "shared_output_norm_per_sample": shared_norm_per_sample.detach().cpu(),
                "shared_contribution_ratio_per_sample": ratio_per_sample.detach().cpu(),
            }
        if self.moe_mixing_space == "linear":
            return routed_out + shared_out
        out = torch.logaddexp(routed_out, shared_out)
        if self.normalized:
            out = out - np.log(2)  # normalize by number of components in the mixture
        return out + math.log(max(self.shared_expert_weight, 1e-12))

    def forward(
        self,
        x,
        train=True,
        loss_coef=1e-2,
        modalities=None,
        instruction_embedding=None,
        semantic_profile_logits=None,
        router_organ_targets=None,
        task_embedding=None,
        task_name=None,
        task_id=None,
        modality_mask_embedding=None,
        modality_mask_ids=None,
        modality_mask_strings=None,
        interaction_features=None,
    ):
        joint_x = torch.cat(x, dim=1) if isinstance(x, list) else x
        shared_out = self._compute_shared_output(joint_x)
        self._output_orth_losses = []
        self._paper_aux_scale_target = None
        self._specialization_metric_records = []
        self.last_specialization_diagnostics = None
        self.last_expert_output_diagnostics = []
        self.last_contribution_diagnostics = None

        if self.staged_shared_lora and self.staged_shared_lora_phase == "shared_pretrain":
            if shared_out is None:
                raise RuntimeError("staged shared-LoRA pretrain phase requires shared_out")
            zero_loss = shared_out.new_zeros(())
            self.last_router_diagnostics = None
            self.last_router_aux_components = None
            return shared_out, zero_loss

        gates, load = self.noisy_top_k_gating(
            x,
            train,
            modalities=modalities,
            instruction_embedding=instruction_embedding,
            semantic_profile_logits=semantic_profile_logits,
            task_embedding=task_embedding,
            task_name=task_name,
            modality_mask_embedding=modality_mask_embedding,
            interaction_features=interaction_features,
        )
        self.current_task_name = task_name
        self.current_task_id = task_id
        self.current_task_embedding = task_embedding
        self.current_modality_mask_ids = modality_mask_ids
        self.current_modality_mask_strings = modality_mask_strings
        self.current_interaction_features = interaction_features
        gates, expert_init_applied = self._mix_expert_init_gates(gates, train)
        if expert_init_applied:
            if isinstance(gates, list):
                load = [self._gates_to_load(g) for g in gates]
            else:
                load = self._gates_to_load(gates)
        gates, r2t2_applied = self._apply_r2t2_rerouting(gates, train)
        if r2t2_applied:
            if isinstance(gates, list):
                load = [self._gates_to_load(g) for g in gates]
            else:
                load = self._gates_to_load(gates)
        router_mode = (
            "r2t2"
            if r2t2_applied
            else (
                "expert_init"
                if expert_init_applied
                else ("dense_warmup" if self._use_dense_warmup() else ("prototype" if self.use_prototype_router else "standard"))
            )
        )
        if isinstance(gates, list):
            self.last_router_diagnostics = {
                "router_type": self.router_type,
                "router_mode": router_mode,
                "multihead_permod_router": self.use_multihead_permod_router,
                "multihead_router_heads": self.multihead_router_heads if self.use_multihead_permod_router else 1,
                "multihead_router_fusion": self.multihead_router_fusion if self.use_multihead_permod_router else "",
                "interaction_experts": self.use_interaction_experts,
                "interaction_expert_mode": self.interaction_expert_mode if self.use_interaction_experts else "",
                "interaction_expert_reweighting": self.use_interaction_expert_reweighting,
                "gates": [g.detach().float().cpu() for g in gates],
                "modalities": modalities,
                "task_name": task_name,
                "task_id": task_id,
                "modality_mask_ids": modality_mask_ids.detach().cpu() if isinstance(modality_mask_ids, torch.Tensor) else modality_mask_ids,
                "modality_mask_strings": modality_mask_strings,
                "interaction_features": interaction_features.detach().float().cpu() if isinstance(interaction_features, torch.Tensor) else interaction_features,
                "router_inputs": self._last_router_inputs_for_diag,
                "clean_logits": self._last_clean_logits_for_diag,
            }
        else:
            self.last_router_diagnostics = {
                "router_type": self.router_type,
                "router_mode": router_mode,
                "multihead_permod_router": self.use_multihead_permod_router,
                "multihead_router_heads": self.multihead_router_heads if self.use_multihead_permod_router else 1,
                "multihead_router_fusion": self.multihead_router_fusion if self.use_multihead_permod_router else "",
                "interaction_experts": self.use_interaction_experts,
                "interaction_expert_mode": self.interaction_expert_mode if self.use_interaction_experts else "",
                "interaction_expert_reweighting": self.use_interaction_expert_reweighting,
                "gates": gates.detach().float().cpu(),
                "modalities": modalities,
                "task_name": task_name,
                "task_id": task_id,
                "modality_mask_ids": modality_mask_ids.detach().cpu() if isinstance(modality_mask_ids, torch.Tensor) else modality_mask_ids,
                "modality_mask_strings": modality_mask_strings,
                "interaction_features": interaction_features.detach().float().cpu() if isinstance(interaction_features, torch.Tensor) else interaction_features,
                "router_inputs": self._last_router_inputs_for_diag,
                "clean_logits": self._last_clean_logits_for_diag,
            }
        if self.last_expert_init_diagnostics is not None:
            expert_init_diag = self.last_expert_init_diagnostics
            if "expert_init_targets" in expert_init_diag:
                self.last_router_diagnostics["expert_init_targets"] = expert_init_diag["expert_init_targets"]
            if "expert_init_confidence" in expert_init_diag:
                self.last_router_diagnostics["expert_init_confidence"] = expert_init_diag["expert_init_confidence"]
            if "expert_init_source" in expert_init_diag:
                self.last_router_diagnostics["expert_init_source"] = expert_init_diag["expert_init_source"]
            if "expert_init_alpha" in expert_init_diag:
                self.last_router_diagnostics["expert_init_alpha"] = expert_init_diag["expert_init_alpha"]
        if expert_init_applied:
            loss = joint_x.new_zeros(())
            self.last_router_aux_components = None
        else:
            loss = self._compute_loss(gates, load, loss_coef)
        organ_loss = self._compute_router_organ_supervision_loss(gates, router_organ_targets)
        if isinstance(organ_loss, torch.Tensor):
            loss = loss + organ_loss
        semantic_guidance_loss = self._compute_semantic_guidance_loss(gates)
        if isinstance(semantic_guidance_loss, torch.Tensor):
            loss = loss + semantic_guidance_loss
            if self.last_router_aux_components is not None:
                self.last_router_aux_components["semantic_guidance_scaled"] = semantic_guidance_loss.detach()

        if self.staged_shared_lora:
            if shared_out is None:
                raise RuntimeError("staged shared-LoRA adapt phase requires shared_out")
            if isinstance(gates, list):
                delta = 0
                for j, g in enumerate(gates):
                    delta = delta + self._compute_routed_delta(x[j], g, modality_idx=j)
            else:
                delta = self._compute_routed_delta(joint_x, gates)
            output_orth_loss = self._consume_output_orthogonality_loss(loss)
            loss = loss + output_orth_loss
            if self.last_router_aux_components is not None:
                self.last_router_aux_components["output_orth_unscaled"] = output_orth_loss.detach()
                self.last_router_aux_components["total_scaled"] = loss.detach()
            self._finalize_specialization_diagnostics(gates)
            return shared_out + delta, loss

        if isinstance(gates, list):
            # permod or disjoint
            y = 0
            for j, g in enumerate(gates):
                if self.moe_mixing_space == "linear":
                    routed_out = self._compute_routed_delta(x[j], g, modality_idx=j)
                else:
                    routed_out = self._compute_routed_output(x[j], g, modality_idx=j, all_modalities=x)
                y += routed_out
            y = self._combine_shared_and_routed(y, shared_out)
        else:
            # joint
            if self.moe_mixing_space == "linear":
                routed_out = self._compute_routed_delta(joint_x, gates)
            else:
                routed_out = self._compute_routed_output(joint_x, gates)
            y = self._combine_shared_and_routed(routed_out, shared_out)

        output_orth_loss = self._consume_output_orthogonality_loss(loss)
        loss = loss + output_orth_loss
        if self.last_router_aux_components is not None:
            self.last_router_aux_components["output_orth_unscaled"] = output_orth_loss.detach()
            self.last_router_aux_components["total_scaled"] = loss.detach()
        self._finalize_specialization_diagnostics(gates)
        return y, loss
