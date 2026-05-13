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
from core.activations import ACT2FN
from utils.config import MoEConfig
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
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        # _batch_index: sample index inside a batch that is assigned to particular expert, concatenated
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
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


def build_expert(config: MoEConfig, input_size: int, output_size: int, hidden_size: int):
    if config.expert_type == "mlp":
        return MLP(config, input_size, output_size, hidden_size)
    if config.expert_type == "lora":
        return LoRAExpert(config, input_size, output_size, hidden_size)
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
        self.shared_experts = config.shared_experts
        self.use_temp = config.use_temp
        self.expert_type = config.expert_type
        self.use_instruction_router = config.use_instruction_router
        self.router_instruction_dim = config.router_instruction_dim
        self.instruction_router_scale = config.instruction_router_scale
        self.instruction_router_fusion = config.instruction_router_fusion
        self.use_semantic_expert_profiles = config.use_semantic_expert_profiles
        self.semantic_profile_scale = config.semantic_profile_scale
        self.semantic_profile_fusion = config.semantic_profile_fusion
        self.semantic_profile_source = config.semantic_profile_source
        self.semantic_profile_modalities = config.semantic_profile_modalities or ["txt"]

        # instantiate experts
        if self.router_type == 'disjoint':
            self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
        elif self.router_type == 'permod':
            self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
        else:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

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

        semantic_profiles = config.semantic_profile_embeddings
        if self.use_semantic_expert_profiles and self.semantic_profile_source == "patient":
            if semantic_profiles is None:
                raise ValueError("semantic_profile_embeddings must be set when use_semantic_expert_profiles=True")
            self.register_buffer("semantic_profile_embeddings", semantic_profiles.detach().float())
            profile_dim = semantic_profiles.size(-1)
            if self.router_type in ['disjoint', 'permod']:
                input_dim = self.input_size // self.num_modalities
                self.semantic_input_proj = nn.ModuleList([
                    nn.Linear(input_dim, profile_dim, bias=False)
                    for _ in range(self.num_modalities)
                ])
            else:
                self.semantic_input_proj = nn.Linear(self.input_size, profile_dim, bias=False)
        elif self.use_semantic_expert_profiles:
            if semantic_profiles is None:
                raise ValueError("semantic_profile_embeddings must be set when use_semantic_expert_profiles=True")
            self.register_buffer("semantic_profile_embeddings", semantic_profiles.detach().float())
            self.semantic_input_proj = None
        else:
            self.semantic_profile_embeddings = None
            self.semantic_input_proj = None

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
        else:
            self.shared_mlps = None
            self.omega = None

        if self.use_temp:
            self.log_tau = nn.Parameter(torch.zeros(1))  # log of temperature parameter for softmax gating
        else:
            self.log_tau = None

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

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

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
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

    def _profile_bank_for_logits(self, num_logits, device, dtype):
        profiles = self.semantic_profile_embeddings.to(device=device, dtype=dtype)
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

    def _apply_semantic_logits(self, logits, x, idx=None):
        semantic_logits = self._semantic_logits(x, idx=idx, num_logits=logits.size(1))
        if semantic_logits is None:
            return logits, None
        if self.semantic_profile_fusion == "replace":
            return semantic_logits, semantic_logits
        return logits + semantic_logits, semantic_logits

    def _external_semantic_logits(self, semantic_profile_logits, logits, modality=None):
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
        if self.semantic_profile_fusion == "replace":
            return external_logits
        return logits + external_logits

    def _get_logits(self, x, train, noise_epsilon, idx=None, instruction_embedding=None, semantic_profile_logits=None, modality=None):
        x = self._fuse_instruction_input(x, instruction_embedding, idx=idx)
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
        else:
            temp = None

        # ---------- Compute clean logits ----------
        if self.gating == 'softmax':
            clean_logits = x @ w_gate
        elif self.gating == 'laplace':
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating == "poly":
            dist = torch.cdist(x, torch.t(w_gate))
            clean_logits = 1.0 / (1.0 + dist.pow(self.poly_power))
        elif self.gating == 'gaussian':
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)
        elif self.gating == "student_t":
            dist = torch.cdist(x, torch.t(w_gate))
            clean_logits = 1.0 / ((1.0 + ((dist)**2)/self.student_degree)**((self.student_degree + 1)/2))
        elif self.gating == "sigmoid":
            raw = x @ w_gate
            if temp is not None:
                raw = raw / temp
            clean_logits = torch.sigmoid(raw)
        elif self.gating == "sigmoid_dist":
            dist = torch.cdist(x, torch.t(w_gate))
            if temp is not None:
                dist = dist / temp
            clean_logits = torch.sigmoid(-dist)

        clean_logits, semantic_logits = self._apply_semantic_logits(clean_logits, x, idx=idx)
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

        # ---------- Noisy gating ----------
        if self.noisy_gating:
            raw_noise_stddev = x @ w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)

            if self.gating == 'poly':
                dist = torch.cdist(x, torch.t(w_gate))
                noise = torch.randn_like(dist) * noise_stddev
                noisy_dist = dist + noise
                noisy_logits = 1.0 / (1.0 + noisy_dist.pow(self.poly_power))
                noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                if instruction_logits is not None:
                    noisy_logits = noisy_logits + instruction_logits
                logits = noisy_logits
            elif self.gating == "student_t":
                dist = torch.cdist(x, torch.t(w_gate))
                noise = torch.randn_like(dist) * noise_stddev
                noisy_dist = dist + noise
                noisy_logits = 1.0 / ((1.0 + ((noisy_dist)**2)/self.student_degree)**((self.student_degree + 1)/2))
                noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                if instruction_logits is not None:
                    noisy_logits = noisy_logits + instruction_logits
                logits = noisy_logits
            elif self.gating == "sigmoid":
                score = x @ w_gate
                if instruction_logits is not None:
                    score = score + instruction_logits
                noise = torch.randn_like(score) * noise_stddev
                noisy_score = score + noise
                if temp is not None:
                    noisy_score = noisy_score / temp
                noisy_logits = torch.sigmoid(noisy_score)
                noisy_logits, _ = self._apply_semantic_logits(noisy_logits, x, idx=idx)
                noisy_logits = self._apply_external_semantic_logits(noisy_logits, semantic_profile_logits, modality=modality)
                logits = noisy_logits
            elif self.gating == "sigmoid_dist":
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
                logits = noisy_logits
            else:
                noise = torch.randn_like(clean_logits) * noise_stddev
                noisy_logits = clean_logits + noise
                if temp is not None:
                    noisy_logits = noisy_logits / temp
                logits = noisy_logits
        else:
            noise_stddev = torch.zeros_like(clean_logits)
            noisy_logits = clean_logits
            logits = clean_logits

        if bias is not None:
            # print("Applying bias to logits")
            logits = logits * torch.exp(bias)
            clean_logits = clean_logits * torch.exp(bias)
            noisy_logits = noisy_logits * torch.exp(bias)

        return logits, clean_logits, noisy_logits, noise_stddev

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        # For disjoint routing, logits only cover the local expert group for one
        # modality, so use logits.size(1) instead of the global expert count.
        num_available_experts = logits.size(1)
        effective_k = min(k, num_available_experts)
        top_logits, top_indices = logits.topk(min(effective_k + 1, num_available_experts), dim=1)
        top_k_logits = top_logits[:, :effective_k]
        top_k_indices = top_indices[:, :effective_k]
        if self.gating == 'softmax':
            if self.normalized:
                top_k_gates = self.softmax(top_k_logits)
            else:
                top_k_gates = torch.exp(top_k_logits)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            if self.normalized:
                top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))
            else:
                top_k_gates = torch.exp(top_k_logits)
            
        elif self.gating == 'poly' or self.gating == "student_t" or self.gating == "sigmoid" or self.gating == "sigmoid_dist":
            if self.normalized:
                top_k_gates = top_k_logits / top_k_logits.sum(dim=1, keepdim=True)
            else:
                top_k_gates = top_k_logits
            
        # zeros = torch.zeros_like(logits, requires_grad=True)
        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and effective_k < num_available_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, modalities=None, instruction_embedding=None, semantic_profile_logits=None):
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
            )
            logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
            gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
            return gates, load
        else:
            all_gates, all_loads = [], []
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
                )
                logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
                if self.router_type == 'permod':
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
                else:
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.disjoint_k)
                all_gates.append(gates)
                all_loads.append(load)
            return all_gates, all_loads


    def _compute_loss(self, gates, loads, loss_coef):
        """Compute load balancing loss across all gates."""
        loss = 0
        if isinstance(gates, list):
            for g, l in zip(gates, loads):
                loss += self.cv_squared(g.sum(0)) + self.cv_squared(l)
        else:
            loss = self.cv_squared(gates.sum(0)) + self.cv_squared(loads)
        return loss * loss_coef


    def _compute_shared_output(self, x):
        """Compute output from shared experts, if any."""
        if self.shared_mlps is None:
            return None

        if self.normalized:
            log_omega = torch.log_softmax(self.omega, dim=0)
        else:
            log_omega = torch.log(torch.nn.functional.softplus(self.omega) + 1e-12)

        out = self.shared_mlps[0](x) + log_omega[0]
        for i, se in enumerate(self.shared_mlps[1:], 1):
            out = torch.logaddexp(out, se(x) + log_omega[i])
        return out


    def _compute_routed_output(self, x, gates, modality_idx=None):
        """Compute routed expert output for a single modality or joint input."""
        if self.router_type == 'disjoint':
            sub_experts = self.num_experts // self.num_modalities
            dispatcher = SparseDispatcher(sub_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[modality_idx][i](expert_inputs[i]) for i in range(sub_experts)]
        elif self.router_type == 'permod':
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        else:  # joint
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        return dispatcher.combine(expert_outputs)


    def _combine_shared_and_routed(self, routed_out, shared_out):
        """Combine shared and routed outputs as a mixture within a modality."""
        if shared_out is None:
            return routed_out
        out = torch.logaddexp(routed_out, shared_out)
        if self.normalized:
            out = out - np.log(2)  # normalize by number of components in the mixture
        return out

    def forward(self, x, train=True, loss_coef=1e-2, modalities=None, instruction_embedding=None, semantic_profile_logits=None):
        gates, load = self.noisy_top_k_gating(
            x,
            train,
            modalities=modalities,
            instruction_embedding=instruction_embedding,
            semantic_profile_logits=semantic_profile_logits,
        )
        loss = self._compute_loss(gates, load, loss_coef)

        # global shared output: always computed once on full input
        joint_x = torch.cat(x, dim=1) if isinstance(x, list) else x
        shared_out = self._compute_shared_output(joint_x)

        if isinstance(gates, list):
            # permod or disjoint
            y = 0
            for j, g in enumerate(gates):
                routed_out = self._compute_routed_output(x[j], g, modality_idx=j)
                y += routed_out
            y = self._combine_shared_and_routed(y, shared_out)
        else:
            # joint
            routed_out = self._compute_routed_output(joint_x, gates)
            y = self._combine_shared_and_routed(routed_out, shared_out)

        return y, loss
