# Original FuseMoE vs Official DeepSeek MoE Architecture

This note compares the MoE architecture in the original local FuseMoE code
against official open-source DeepSeek MoE implementations. The goal is to list
architecture differences that could affect expert collapse, expert balance,
specialization, and optimization.

FuseMoE source inspected:

- `/home/pham156/MoE/FuseMoE_ori/core/sparse_moe.py`

Official DeepSeek sources inspected:

- DeepSeek-MoE 16B HF implementation:
  `https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py`
- DeepSeek-V3 inference implementation:
  `https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py`

## 1. High-Level Difference

Original FuseMoE is a multimodal sample-level sparse MoE.

Official DeepSeek MoE is a transformer FFN replacement that routes token hidden
states inside language-model layers.

That is the biggest conceptual difference. Many smaller differences come from
this.

```text
FuseMoE:
  clinical feature vector(s)
  -> router over experts
  -> selected MLP classifiers/regressors
  -> task output

DeepSeek MoE:
  token hidden state
  -> router over FFN experts
  -> selected FFN transformations
  -> transformer hidden state
```

## 2. Routing Unit

### Original FuseMoE

FuseMoE routes one feature vector per sample or per modality.

Supported router structures:

```text
joint:
  concatenate all modalities, then use one router.

permod:
  each modality has its own router over experts.

disjoint:
  each modality routes only inside its assigned expert subset.
```

Relevant source:

```python
if self.router_type == 'disjoint':
    self.w_gate = [Parameter(... num_experts // num_modalities) ...]
elif self.router_type == 'permod':
    self.w_gate = [Parameter(... num_experts) ...]
else:
    self.w_gate = Parameter(input_size, num_experts)
```

In forward:

```python
if self.router_type == 'joint':
    embeddings = torch.concat(x, dim=1)
    gates, load = self._top_k_gating(...)
else:
    for i in range(self.num_modalities):
        gates, load = self._top_k_gating(...)
```

### Official DeepSeek MoE

DeepSeek routes flattened token hidden states:

```python
bsz, seq_len, h = hidden_states.shape
hidden_states = hidden_states.view(-1, h)
logits = F.linear(hidden_states, self.weight, None)
```

DeepSeek-V3 similarly flattens the MoE input:

```python
shape = x.size()
x = x.view(-1, self.dim)
weights, indices = self.gate(x)
```

Implication:

- FuseMoE has far fewer routing decisions per batch.
- DeepSeek has one routing decision per token per MoE layer.
- DeepSeek load statistics are naturally smoother because sequence length
  multiplies the number of router samples.

## 3. Router Parameter Shape

### Original FuseMoE

For `joint`:

```text
w_gate:  [input_size, num_experts]
w_noise: [input_size, num_experts]
```

For `permod`:

```text
w_gate[i]:  [input_size / num_modalities, num_experts]
w_noise[i]: [input_size / num_modalities, num_experts]
```

For `disjoint`:

```text
w_gate[i]:  [input_size / num_modalities, num_experts / num_modalities]
w_noise[i]: [input_size / num_modalities, num_experts / num_modalities]
```

### Official DeepSeek MoE

DeepSeek-MoE 16B:

```text
router weight: [n_routed_experts, hidden_size]
```

DeepSeek-V3:

```text
router weight: [n_routed_experts, dim]
optional bias: [n_routed_experts]
```

Implication:

- FuseMoE router shape depends on multimodal router mode.
- DeepSeek router shape is uniform across tokens and layers.
- FuseMoE `permod` has separate router matrices per modality, so each modality
  can collapse or specialize independently.

## 4. Router Initialization

### Original FuseMoE

FuseMoE initializes both gate and noise matrices to exact zero:

```python
self.w_gate = nn.Parameter(torch.zeros(...), requires_grad=True)
self.w_noise = nn.Parameter(torch.zeros(...), requires_grad=True)
```

At initialization:

```python
clean_logits = x @ w_gate
```

Therefore:

```text
clean_logits are exactly tied for all experts.
```

If noisy gating is disabled, `torch.topk` sees equal values and deterministically
selects the same expert indices.

### Official DeepSeek MoE

DeepSeek-MoE 16B initializes router weights with Kaiming uniform:

```python
self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

DeepSeek-V3 defines a learned router weight:

```python
self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
```

Implication:

- FuseMoE starts with exact router ties.
- Official DeepSeek does not intentionally start from exact tied logits.
- FuseMoE is therefore more vulnerable to deterministic top-k collapse when
  noise is removed or small.

## 5. Router Bias / Gamma / Expert Prior

### Original FuseMoE

Original FuseMoE has no explicit router bias or expert prior term:

```python
clean_logits = x @ w_gate
```

There is no:

```text
x @ w_gate + bias
```

and no separate learned expert prior such as `gamma_i`.

### Official DeepSeek MoE

DeepSeek-MoE 16B uses:

```python
logits = F.linear(hidden_states, self.weight, None)
```

So the HF DeepSeek-MoE 16B router also has no bias in that line.

DeepSeek-V3 has optional expert bias in the inference code:

```python
self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
```

V3 uses the bias for selection:

```python
original_scores = scores
if self.bias is not None:
    scores = scores + self.bias
indices = torch.topk(scores, self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)
```

Implication:

- FuseMoE original and DeepSeek-MoE 16B are both no-bias softmax routers.
- DeepSeek-V3 can use expert bias for selection, but the final route weights
  are gathered from pre-bias scores.
- If you add a `gamma_i` prior to FuseMoE, that becomes a deviation from
  original FuseMoE and from DeepSeek-MoE 16B, but it is conceptually closer to
  DeepSeek-V3's selection-bias idea.

## 6. Noisy Gating

### Original FuseMoE

FuseMoE implements Shazeer-style noisy top-k:

```python
raw_noise_stddev = x @ w_noise
noise_stddev = (softplus(raw_noise_stddev) + noise_epsilon) * train
noise = torch.randn_like(clean_logits) * noise_stddev
logits = clean_logits + noise
```

Because `w_noise` is zero-initialized:

```text
softplus(0) ~= 0.693
```

So early training has:

```text
clean logits: exactly tied
noise std:    about 0.7
```

For distance gates, noise is added to distance before conversion:

```python
noisy_dist = dist + noise
noisy_logits = 1.0 / (1.0 + noisy_dist.pow(poly_power))
```

### Official DeepSeek MoE

Official DeepSeek MoE does not use Shazeer noisy top-k and has no learned
`w_noise` matrix.

DeepSeek-MoE 16B:

```python
scores = logits.softmax(dim=-1)
topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
```

DeepSeek-V3:

```python
scores = linear(x, self.weight)
scores = scores.softmax(...) or scores.sigmoid()
indices = torch.topk(scores, self.topk, dim=-1)[1]
```

Implication:

- FuseMoE can look balanced because random noise spreads expert usage.
- DeepSeek expert balance cannot be explained by this specific noisy-gating
  mechanism.
- If FuseMoE collapses without noise but looks uniform with noise, the uniform
  case may be noise-driven rather than learned specialization.

## 7. Score Functions

### Original FuseMoE

FuseMoE supports multiple score functions:

```python
softmax:    clean_logits = x @ w_gate
laplace:    clean_logits = -cdist(x, w_gate.T)
gaussian:   clean_logits = -cdist(x, w_gate.T)^2
poly:       clean_logits = 1 / (1 + dist^p)
student_t:  clean_logits = 1 / (1 + dist^2 / df)^((df + 1) / 2)
```

Important detail:

- In softmax mode, `w_gate` is a linear classifier.
- In distance modes, `w_gate.T` behaves like learned expert centers.
- But all are initialized to zero in the original code.

### Official DeepSeek MoE

DeepSeek-MoE 16B uses softmax scoring:

```python
scores = logits.softmax(dim=-1)
```

DeepSeek-V3 supports:

```text
softmax
sigmoid
```

It does not use Gaussian, Laplace, poly, or Student-t distance gates.

Implication:

- FuseMoE distance-gating experiments are not official DeepSeek-style routing.
- They are valid ablations, but they answer a different question.

## 8. Top-K Selection and Normalization

### Original FuseMoE

FuseMoE computes `k+1` top values:

```python
top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
top_k_logits = top_logits[:, :k]
top_k_indices = top_indices[:, :k]
```

It uses `k+1` because `_prob_in_top_k` needs a threshold for noisy load
estimation.

Selected weights are normalized depending on score function:

```python
softmax:
    top_k_gates = softmax(top_k_logits)

laplace / gaussian:
    top_k_gates = exp(top_k_logits - logsumexp(top_k_logits))

poly / student_t:
    top_k_gates = top_k_logits / top_k_logits.sum(...)
```

Then a sparse gate matrix is created:

```python
gates = zeros.scatter(1, top_k_indices, top_k_gates)
```

### Official DeepSeek MoE

DeepSeek-MoE 16B:

```python
topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
if self.top_k > 1 and self.norm_topk_prob:
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
```

DeepSeek-V3:

```python
indices = torch.topk(scores, self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)
if self.score_func == "sigmoid":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

Implication:

- FuseMoE normalizes selected values after top-k.
- DeepSeek-MoE 16B top-k values come from full softmax scores and may be
  optionally renormalized.
- DeepSeek-V3 separates selection scores from final gathered weights when bias
  is present.

## 9. Load Balancing Objective

### Original FuseMoE

FuseMoE uses coefficient-of-variation penalties:

```python
loss = cv_squared(gates.sum(0)) + cv_squared(load)
loss *= loss_coef
```

Definitions:

```python
importance = gates.sum(0)
load = number/probability of inputs assigned to each expert
```

When noisy gating is enabled:

```python
load = prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
```

When noisy gating is disabled:

```python
load = (gates > 0).sum(0)
```

### Official DeepSeek MoE

DeepSeek-MoE 16B uses an auxiliary loss based on expert assignment counts and
mean router probabilities:

```python
ce.scatter_add_(..., topk_idx_for_aux_loss, ones)
aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * alpha
```

DeepSeek-V3's released inference code includes routing bias / grouped routing
machinery. V3 is described as using an auxiliary-loss-free balancing strategy,
which is different from FuseMoE's CV loss.

Implication:

- FuseMoE balance loss can be satisfied by random noisy routing.
- DeepSeek-MoE's aux loss is tied to token-level counts and router scores.
- The balancing losses are not equivalent.

## 10. Expert Computation

### Original FuseMoE

FuseMoE experts are classifier/regressor-style MLPs:

```python
self.experts = nn.ModuleList([
    MLP(config, input_size, output_size, hidden_size)
    for _ in range(num_experts)
])
```

Each expert returns log probabilities:

```python
out = self.fc2(out)
out = self.log_soft(out)
return out
```

The dispatcher combines in probability space, then returns log space:

```python
stitched = torch.cat(expert_out, 0).exp()
stitched = stitched.mul(nonzero_gates)
combined = zeros.index_add(...)
return combined.log()
```

So FuseMoE mixture is:

```text
log( sum_i gate_i * exp(expert_log_prob_i) )
```

### Official DeepSeek MoE

DeepSeek experts are FFN transformations inside transformer layers.

DeepSeek-MoE 16B expert MLP uses gated FFN form:

```python
down_proj = down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

Routed outputs are combined in hidden-state space:

```python
y = (expert_outputs * topk_weight.unsqueeze(-1)).sum(dim=1)
```

DeepSeek-V3 expert MLP is also gated:

```python
return w2(silu(w1(x)) * w3(x))
```

Implication:

- FuseMoE mixes output distributions/log-probabilities.
- DeepSeek mixes hidden-state transformations.
- Expert specialization pressure is therefore different.

## 11. Dispatch and Execution

### Original FuseMoE

FuseMoE uses `SparseDispatcher`:

```python
torch.nonzero(gates)
sort by expert
split inputs by expert
run each expert on its assigned minibatch
index_add outputs back to batch
```

This is sparse at the sample level.

### Official DeepSeek MoE

DeepSeek-MoE 16B training path repeats token states for top-k experts:

```python
hidden_states = hidden_states.repeat_interleave(num_experts_per_tok, dim=0)
for i, expert in enumerate(self.experts):
    y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
```

Inference sorts expert indices:

```python
idxs = flat_expert_indices.argsort()
tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
```

DeepSeek-V3 uses local experts and distributed all-reduce:

```python
for i in local_experts:
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx]) * weights[idx, top, None]
if world_size > 1:
    dist.all_reduce(y)
```

Implication:

- FuseMoE sparse dispatch is designed for small batch feature vectors.
- DeepSeek dispatch is designed for many token states and expert parallelism.

## 12. Shared Experts

### Original FuseMoE

Original FuseMoE has no dense shared expert path.

Every expert contribution is selected through sparse gates.

### Official DeepSeek MoE

DeepSeek-MoE 16B has optional shared experts:

```python
if config.n_shared_experts is not None:
    self.shared_experts = DeepseekMLP(...)
...
y = y + self.shared_experts(identity)
```

DeepSeek-V3 has shared experts by construction:

```python
self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
...
z = self.shared_experts(x)
return y + z
```

Implication:

- In DeepSeek, every token receives a dense shared FFN path.
- Routed experts do not have to carry all general knowledge.
- FuseMoE routed experts carry the full MoE output unless another non-MoE path
  exists outside this layer.

## 13. Expert Capacity / Dropping

### Original FuseMoE

The inspected FuseMoE code does not implement a fixed expert capacity or token
dropping. Each expert receives however many samples are assigned:

```python
part_sizes = (gates > 0).sum(0).tolist()
```

### Official DeepSeek MoE

The inspected official DeepSeek-MoE 16B and V3 inference code also routes
selected tokens to experts without an explicit capacity-drop mechanism in these
snippets.

Implication:

- Collapse is not caused by capacity overflow in the inspected FuseMoE code.
- If an expert receives many samples/tokens, the code attempts to process them.

## 14. Layer Placement

### Original FuseMoE

FuseMoE is an explicit cross-modal fusion / prediction module. It is close to
the task head.

### Official DeepSeek MoE

DeepSeek MoE replaces dense FFN sublayers in transformer blocks. MoE routing is
repeated across many layers.

Implication:

- DeepSeek has many MoE routing opportunities per sequence.
- FuseMoE may have only one or a small number of routing modules.
- A bad router in FuseMoE can directly damage the task output; DeepSeek has
  residual paths and repeated transformer layers around the MoE.

## 15. Residual Context Around MoE

### Original FuseMoE

Inside `sparse_moe.py`, the MoE output is directly the weighted expert output:

```python
return y, loss
```

There is no residual connection inside this MoE module.

### Official DeepSeek MoE

DeepSeek MoE sits inside transformer blocks with residual structure around the
attention and FFN/MoE sublayers.

Implication:

- DeepSeek MoE is stabilized by transformer residual pathways.
- FuseMoE's sparse expert output is more directly responsible for task
  prediction.

## 16. Output Space

### Original FuseMoE

The local FuseMoE expert MLP ends with:

```python
self.log_soft = nn.LogSoftmax(1)
```

So expert outputs are task-output log probabilities.

### Official DeepSeek MoE

DeepSeek expert outputs are hidden vectors with the same model dimension as the
transformer stream.

Implication:

- FuseMoE experts specialize over final prediction behavior.
- DeepSeek experts specialize over internal representation transformations.
- Expert usage patterns are not directly comparable.

## 17. Top-K Tie Failure Mode

Original FuseMoE has a specific failure mode:

```text
w_gate = 0
-> clean_logits equal
-> no noise or tiny noise
-> topk selects the same expert indices due to deterministic tie-breaking
-> selected gate weights are uniform among selected experts
```

For `top_k=2`, this can produce:

```text
mass: [0, 0.5, 0.5, 0]
sel:  [0, 1.0, 1.0, 0]
```

This pattern is not evidence of learned expert specialization.

Official DeepSeek is much less exposed to this exact failure mode because its
router weights are not zero-initialized in the same way and because routing is
over many token states.

## 18. Potential Differences Checklist

These are all plausible differences to consider when diagnosing FuseMoE versus
DeepSeek behavior:

| Difference | FuseMoE Original | Official DeepSeek MoE |
|---|---|---|
| Routing object | sample/modality feature | token hidden state |
| Number of routing samples | batch size or batch x modalities | batch x sequence length |
| Router modes | joint/permod/disjoint | no multimodal modes |
| Router init | zero | random learned weight |
| Noise | Shazeer noisy top-k | no Shazeer noisy top-k |
| Noise parameters | learned `w_noise` | none |
| Router bias | none | none in MoE 16B; optional in V3 |
| Score functions | softmax plus distance kernels | softmax; V3 also sigmoid |
| Top-k threshold | computes `k+1` for noisy load | computes `k` |
| Gate normalization | selected top-k normalized | selected top-k optionally normalized/scaled |
| Expert type | task MLP with log-softmax | transformer FFN expert |
| Output mixing | probability/log-probability space | hidden-state space |
| Shared experts | absent in this module | dense shared experts |
| Load balancing | CV squared importance/load | aux count/prob loss or V3 balancing bias |
| Expert parallelism | no distributed expert parallelism in file | V3 supports local experts and all-reduce |
| Residual wrapper | not inside this MoE file | transformer residual block context |
| Capacity/drop | no explicit capacity drop | no explicit drop in inspected inference/HF snippets |
| Specialization target | modality/task prediction | token representation transform |

## 19. Most Relevant Differences For Collapse

For the specific issue "FuseMoE collapses without noise and looks uniform with
noise", the highest-priority differences are:

1. `w_gate` is initialized to zero in FuseMoE.
2. `w_noise` is initialized to zero, which still gives `softplus(0) ~= 0.693`
   noise scale.
3. FuseMoE uses Shazeer noisy top-k; DeepSeek does not.
4. FuseMoE routes far fewer objects per batch.
5. FuseMoE has no built-in shared expert path in this module.
6. FuseMoE mixes expert predictions, while DeepSeek mixes hidden states.

The most direct ablation inside original FuseMoE is:

```text
randomly initialize w_gate
disable or strongly reduce noisy_gating
keep the same data, experts, top-k, and load-balance loss
```

If fixed expert-pair collapse disappears, then the main cause is not "MoE
cannot specialize"; it is the original router initialization/noise/top-k setup.

## 20. Bottom Line

Original FuseMoE and official DeepSeek MoE are not the same MoE architecture.

The closest DeepSeek-like pieces in FuseMoE are:

```text
linear router -> top-k experts -> weighted sum
load balancing auxiliary loss
```

The major non-DeepSeek pieces in original FuseMoE are:

```text
zero router initialization
Shazeer noisy top-k
multimodal joint/permod/disjoint router modes
distance-kernel router options
task-output expert mixing
no dense shared experts inside the MoE layer
```

For diagnosis, do not interpret FuseMoE's noisy uniform usage as equivalent to
DeepSeek balanced routing. In original FuseMoE, uniformity can be produced by
noise plus balance loss even if the clean router has not learned a meaningful
input-dependent partition.

## 21. Week 27 DeepSeek-Inspired FuseMoE Experiments

After the architectural comparison above, we implemented a practical
FuseMoE-side approximation of several DeepSeek-style differences and ran a grid
to test whether they improve expert collapse, routing specialization, and task
performance.

The goal was not to reproduce DeepSeek exactly. The current FuseMoE model is
still a multimodal clinical prediction model that routes sample/modality
feature vectors, not a token-level transformer FFN MoE. The experiments below
therefore test the subset of DeepSeek differences that can be implemented
inside the current FuseMoE code without rewriting the whole model.

### 21.1 Implemented Opt-In Architecture Flags

The following flags were added to the FuseMoE implementation:

```text
--router_init {zero,normal,xavier,kaiming}
--router_init_std FLOAT
--router_bias_mode {mul,add}
--gate_normalization {selected,full,full_renorm}
--router_topk_mode {k_plus_1,k}
--moe_mixing_space {logprob,linear}
--load_balance_mode {cv,deepseek_aux}
```

Interpretation:

- `router_init=zero` preserves original FuseMoE.
- `router_init=xavier` tests DeepSeek-like non-tied random router starts.
- `router_bias_mode=add` is closer to DeepSeek-V3 style additive selection
  bias than the old multiplicative bias behavior.
- `gate_normalization=selected` preserves FuseMoE selected-top-k
  normalization.
- `gate_normalization=full_renorm` first computes full-router softmax scores,
  gathers selected top-k weights, then renormalizes the selected weights.
- `router_topk_mode=k_plus_1` preserves Shazeer/FuseMoE noisy-load threshold.
- `router_topk_mode=k` uses exact top-k selection.
- `moe_mixing_space=logprob` preserves FuseMoE probability/log-probability
  mixing.
- `moe_mixing_space=linear` makes experts emit hidden-feature vectors and
  mixes those vectors directly, closer to transformer FFN expert mixing.
- `load_balance_mode=cv` preserves FuseMoE CV-squared importance/load loss.
- `load_balance_mode=deepseek_aux` uses a count/probability-style auxiliary
  routing loss.

These were implemented as opt-in flags, so original FuseMoE runs remain
available.

### 21.2 Experiment Grid

Grid folder:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/
```

Scripts:

```text
/home/pham156/MoE/FuseMoE_poly/src/scripts/week27_deepseek_arch_grid/job_deepseek_arch_config.sbatch
/home/pham156/MoE/FuseMoE_poly/src/scripts/week27_deepseek_arch_grid/submit_deepseek_arch_grid_96.sh
```

Parsed artifacts:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/deepseek_arch_grid_manifest.tsv
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/deepseek_arch_results.csv
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/deepseek_arch_routing_summary.csv
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/exact_last_router_distributions_selected.csv
```

Grid dimensions:

```text
tasks:     IHM, LOS
seeds:     32, 42
routers:   joint, permod
shared:    0, 1
experts:   4
top-k:     2
epochs:    16
balance:   0.01
modality:  TS + text + CXR
```

Architecture variants:

```text
fusemoe_original
zero_no_noise
random_router
deepseek_gate
deepseek_aux
deepseek_hidden_aux
xmoe_hidden_aux
```

Variant definitions:

```text
fusemoe_original:
  router_init=zero
  noisy_gating=True
  router_topk_mode=k_plus_1
  gate_normalization=selected
  moe_mixing_space=logprob
  load_balance_mode=cv

zero_no_noise:
  router_init=zero
  noisy_gating=False
  router_topk_mode=k
  gate_normalization=selected
  moe_mixing_space=logprob
  load_balance_mode=cv

random_router:
  router_init=xavier
  noisy_gating=False
  router_topk_mode=k
  gate_normalization=selected
  moe_mixing_space=logprob
  load_balance_mode=cv

deepseek_gate:
  router_init=xavier
  noisy_gating=False
  router_topk_mode=k
  router_bias_mode=add
  gate_normalization=full_renorm
  moe_mixing_space=logprob
  load_balance_mode=cv

deepseek_aux:
  router_init=xavier
  noisy_gating=False
  router_topk_mode=k
  router_bias_mode=add
  gate_normalization=full_renorm
  moe_mixing_space=logprob
  load_balance_mode=deepseek_aux

deepseek_hidden_aux:
  router_init=xavier
  noisy_gating=False
  router_topk_mode=k
  router_bias_mode=add
  gate_normalization=full_renorm
  moe_mixing_space=linear
  load_balance_mode=deepseek_aux

xmoe_hidden_aux:
  use_xmoe_router=True
  noisy_gating=False
  router_topk_mode=k
  gate_normalization=full_renorm
  moe_mixing_space=linear
  load_balance_mode=deepseek_aux
```

Total completed jobs:

```text
112 / 112 finished
```

## 22. Performance Results

Mean test performance across all 16 matched runs per architecture variant:

| Variant | n | Mean F1 | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| deepseek_aux | 16 | 0.6027 | 0.8166 | 0.5947 |
| deepseek_gate | 16 | 0.6003 | 0.8141 | 0.5900 |
| fusemoe_original | 16 | 0.5994 | 0.8164 | 0.5957 |
| random_router | 16 | 0.5981 | 0.8170 | 0.5937 |
| zero_no_noise | 16 | 0.5962 | 0.8176 | 0.5951 |
| xmoe_hidden_aux | 16 | 0.5893 | 0.8186 | 0.5924 |
| deepseek_hidden_aux | 16 | 0.5869 | 0.8207 | 0.5998 |

Task-specific mean F1:

| Task | Variant | n | Mean F1 | Mean AUC | Mean AUPRC |
|---|---|---:|---:|---:|---:|
| IHM | deepseek_aux | 8 | 0.4706 | 0.8099 | 0.4506 |
| IHM | deepseek_gate | 8 | 0.4624 | 0.8062 | 0.4467 |
| IHM | random_router | 8 | 0.4608 | 0.8113 | 0.4515 |
| IHM | fusemoe_original | 8 | 0.4593 | 0.8125 | 0.4584 |
| IHM | zero_no_noise | 8 | 0.4572 | 0.8140 | 0.4536 |
| IHM | xmoe_hidden_aux | 8 | 0.4416 | 0.8176 | 0.4540 |
| IHM | deepseek_hidden_aux | 8 | 0.4401 | 0.8205 | 0.4685 |
| LOS | fusemoe_original | 8 | 0.7395 | 0.8202 | 0.7329 |
| LOS | deepseek_gate | 8 | 0.7382 | 0.8220 | 0.7333 |
| LOS | xmoe_hidden_aux | 8 | 0.7371 | 0.8196 | 0.7308 |
| LOS | random_router | 8 | 0.7354 | 0.8227 | 0.7359 |
| LOS | zero_no_noise | 8 | 0.7352 | 0.8212 | 0.7367 |
| LOS | deepseek_aux | 8 | 0.7347 | 0.8233 | 0.7388 |
| LOS | deepseek_hidden_aux | 8 | 0.7337 | 0.8209 | 0.7312 |

Matched deltas versus `fusemoe_original`:

| Variant | Task | n | Mean ΔF1 | Mean ΔAUC | Mean ΔAUPRC | F1 Wins |
|---|---|---:|---:|---:|---:|---:|
| deepseek_aux | IHM | 8 | +0.0113 | -0.0026 | -0.0079 | 5/8 |
| deepseek_gate | IHM | 8 | +0.0030 | -0.0063 | -0.0117 | 5/8 |
| random_router | IHM | 8 | +0.0015 | -0.0012 | -0.0070 | 4/8 |
| zero_no_noise | IHM | 8 | -0.0022 | +0.0014 | -0.0049 | 3/8 |
| xmoe_hidden_aux | IHM | 8 | -0.0177 | +0.0051 | -0.0045 | 3/8 |
| deepseek_hidden_aux | IHM | 8 | -0.0193 | +0.0080 | +0.0100 | 4/8 |
| deepseek_gate | LOS | 8 | -0.0013 | +0.0017 | +0.0004 | 3/8 |
| xmoe_hidden_aux | LOS | 8 | -0.0024 | -0.0006 | -0.0021 | 3/8 |
| random_router | LOS | 8 | -0.0041 | +0.0025 | +0.0029 | 2/8 |
| zero_no_noise | LOS | 8 | -0.0043 | +0.0010 | +0.0038 | 1/8 |
| deepseek_aux | LOS | 8 | -0.0048 | +0.0030 | +0.0059 | 3/8 |
| deepseek_hidden_aux | LOS | 8 | -0.0058 | +0.0007 | -0.0017 | 2/8 |

Best single runs:

| Task | Job | Seed | Router | Shared | Variant | F1 | AUC | AUPRC |
|---|---:|---:|---|---:|---|---:|---:|---:|
| IHM | 10836861 | 32 | permod | 0 | deepseek_hidden_aux | 0.5159 | 0.8238 | 0.4698 |
| LOS | 10836940 | 42 | permod | 0 | fusemoe_original | 0.7497 | 0.8168 | 0.7222 |

Main performance conclusion:

```text
No DeepSeek-inspired variant gives a strong, consistent performance boost.
deepseek_aux is the best mean-F1 variant overall and helps IHM modestly, but it
does not improve LOS F1. hidden-feature mixing improves some AUC/AUPRC values
but hurts F1 and is not yet a reliable replacement for FuseMoE log-probability
mixing.
```

## 23. Routing and Specialization Results

The routing results are more informative than the performance results.

### 23.1 Joint Router

Average test routing diagnostics for `joint`:

| Variant | n | Mean F1 | Entropy | Gate Variance | Top-1 Weight | Top1-Top2 Margin | Mean Mass Range |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek_aux | 8 | 0.5932 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| deepseek_gate | 8 | 0.6010 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| deepseek_hidden_aux | 8 | 0.5863 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_router | 8 | 0.5876 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| fusemoe_original | 8 | 0.5936 | 0.0028 | 0.0126 | 0.9987 | 0.9975 | 0.9514 |
| zero_no_noise | 8 | 0.5983 | 0.0182 | 0.0000 | 0.9933 | 0.9865 | 0.9932 |
| xmoe_hidden_aux | 8 | 0.6013 | 0.5332 | 0.0000 | 0.7598 | 0.5197 | 0.7598 |

Joint-router conclusion:

```text
Joint routing still collapses hard for almost every variant.
DeepSeek-style init, gate normalization, additive bias, and auxiliary balancing
do not fix joint collapse in this sample-level FuseMoE setting. XMoE softens
the collapse but does not produce a reliable performance gain.
```

### 23.2 Permod Router

Average test routing diagnostics for `permod`:

| Variant | n | Mean F1 | Entropy | Gate Variance | Top-1 Weight | Top1-Top2 Margin | Mean Mass Range |
|---|---:|---:|---:|---:|---:|---:|---:|
| fusemoe_original | 8 | 0.6052 | 0.6457 | 0.0645 | 0.6207 | 0.2414 | 0.0214 |
| random_router | 8 | 0.6087 | 0.6201 | 0.0016 | 0.6243 | 0.2487 | 0.4999 |
| deepseek_gate | 8 | 0.5996 | 0.6131 | 0.0014 | 0.6295 | 0.2590 | 0.4969 |
| deepseek_aux | 8 | 0.6121 | 0.5302 | 0.0013 | 0.7154 | 0.4307 | 0.6079 |
| xmoe_hidden_aux | 8 | 0.5774 | 0.5721 | 0.0000 | 0.7066 | 0.4133 | 0.6514 |
| deepseek_hidden_aux | 8 | 0.5875 | 0.1507 | 0.0001 | 0.9457 | 0.8914 | 0.8475 |
| zero_no_noise | 8 | 0.5941 | 0.6931 | 0.0000 | 0.5000 | 0.0000 | 0.5000 |

Permod-router conclusion:

```text
Original FuseMoE permod routing is almost perfectly balanced but likely
noise/balance-driven rather than semantically specialized.

Random-router, deepseek_gate, and deepseek_aux produce more specialized routing
patterns. deepseek_aux has the best mean permod F1, but the improvement is
small.

deepseek_hidden_aux is too sharp and often collapses to dominant experts.
zero_no_noise is not meaningful specialization; it is deterministic top-k tie
behavior.
```

## 24. Noise and Router Initialization

Removing noise has a clear diagnostic effect but is not useful by itself.

Matched `zero_no_noise` versus `fusemoe_original`:

| Task | Mean ΔF1 | Mean ΔAUC | Mean ΔAUPRC | F1 Wins |
|---|---:|---:|---:|---:|
| IHM | -0.0022 | +0.0014 | -0.0049 | 3/8 |
| LOS | -0.0043 | +0.0010 | +0.0038 | 1/8 |

Routing effect:

```text
joint + no noise:
  entropy 0.0182
  top1 weight 0.9933
  mass_range 0.9932
  -> still collapsed

permod + no noise:
  entropy 0.6931
  top1 weight 0.5000
  margin 0.0000
  mass_range 0.5000
  -> deterministic equal top-2 tie, not learned specialization
```

Random/Xavier router initialization helps diagnose the zero-init tie issue, but
it is not a robust performance fix.

Matched `random_router` versus `fusemoe_original`:

| Task | Mean ΔF1 | Mean ΔAUC | Mean ΔAUPRC | F1 Wins |
|---|---:|---:|---:|---:|
| IHM | +0.0015 | -0.0012 | -0.0070 | 4/8 |
| LOS | -0.0041 | +0.0025 | +0.0029 | 2/8 |

For permod specifically:

| Task | Seed | Shared | Original F1 | Random F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|
| IHM | 32 | 0 | 0.4298 | 0.5019 | +0.0721 |
| IHM | 32 | 1 | 0.4576 | 0.4904 | +0.0328 |
| IHM | 42 | 0 | 0.4941 | 0.4679 | -0.0262 |
| IHM | 42 | 1 | 0.4921 | 0.4706 | -0.0215 |
| LOS | 32 | 0 | 0.7373 | 0.7330 | -0.0043 |
| LOS | 32 | 1 | 0.7408 | 0.7330 | -0.0079 |
| LOS | 42 | 0 | 0.7497 | 0.7455 | -0.0042 |
| LOS | 42 | 1 | 0.7405 | 0.7270 | -0.0134 |

Conclusion:

```text
Random init removes the exact zero-logit tie and creates more specialized
routing, but it is seed-sensitive and does not reliably outperform original
noisy FuseMoE.
```

## 25. Exact Across-Seed Routing Distributions

Exact last-epoch router distributions were extracted into:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/exact_last_router_distributions_selected.csv
```

The following examples compare `permod` routing across seeds.

### 25.1 LOS, Shared 0, Original FuseMoE

Original FuseMoE is almost seed-invariant and uniform.

Seed 32:

```text
L0 mass[0.256,0.241,0.252,0.252]
L1 mass[0.248,0.250,0.249,0.254]
L2 mass[0.238,0.261,0.245,0.256]
```

Seed 42:

```text
L0 mass[0.245,0.253,0.255,0.247]
L1 mass[0.253,0.249,0.251,0.247]
L2 mass[0.259,0.248,0.241,0.252]
```

### 25.2 LOS, Shared 0, Random Router

Random router is much more seed-dependent.

Seed 32:

```text
L0 mass[0.004,0.469,0.002,0.526]
L1 mass[0.067,0.132,0.464,0.337]
L2 mass[0.161,0.428,0.246,0.165]
```

Seed 42:

```text
L0 mass[0.070,0.424,0.230,0.276]
L1 mass[0.147,0.009,0.235,0.608]
L2 mass[0.185,0.250,0.313,0.251]
```

### 25.3 IHM, Shared 0, Original FuseMoE

Seed 32:

```text
L0 mass[0.256,0.241,0.252,0.252]
L1 mass[0.248,0.250,0.249,0.254]
L2 mass[0.238,0.261,0.245,0.256]
```

Seed 42:

```text
L0 mass[0.245,0.253,0.255,0.247]
L1 mass[0.253,0.249,0.251,0.247]
L2 mass[0.259,0.248,0.241,0.252]
```

### 25.4 IHM, Shared 0, Random Router

Seed 32:

```text
L0 mass[0.132,0.224,0.004,0.640]
L1 mass[0.415,0.005,0.434,0.145]
L2 mass[0.114,0.570,0.013,0.304]
```

Seed 42:

```text
L0 mass[0.242,0.311,0.219,0.228]
L1 mass[0.428,0.151,0.101,0.319]
L2 mass[0.242,0.173,0.202,0.383]
```

Across-seed conclusion:

```text
original noisy permod:
  stable and almost uniform across seeds
  likely reflects noise/load-balance behavior

random permod:
  non-uniform and more specialized
  substantially changes with seed
  sometimes improves IHM, but does not reliably improve LOS
```

## 26. Current Diagnosis

The strongest diagnosis from the experiments is:

```text
1. Joint routing is the main collapse-prone setting.
2. Permod routing avoids hard collapse, but original FuseMoE permod is likely
   artificially uniform because of Shazeer noise plus balance loss.
3. Removing noise exposes deterministic top-k tie behavior.
4. Random router initialization creates real specialization but is seed-sensitive.
5. DeepSeek-style auxiliary balancing gives the most promising mean result,
   especially for IHM, but the gain is small.
6. Hidden-feature mixing is not currently a reliable improvement in this
   FuseMoE head architecture.
```

Recommended next experiment set:

```text
Keep:
  permod
  fusemoe_original
  deepseek_aux
  deepseek_gate
  random_router as diagnostic

Drop for now:
  joint, unless specifically studying collapse
  deepseek_hidden_aux
  xmoe_hidden_aux
  zero_no_noise except as a tie-collapse diagnostic

Use more seeds:
  32, 42, 52, 62, 72

Report routing:
  mean ± std across seeds
  exact per-layer/per-modality mass distributions for representative seeds
```

## 27. What Was Not Covered

The following DeepSeek differences were not fully reproduced:

- True token-level routing.
- Large number of routing samples from batch times sequence length.
- Actual transformer FFN experts inside the backbone.
- Distributed expert parallelism.
- Full DeepSeek-V3 auxiliary-loss-free bias update strategy.

These require deeper model redesign. The current experiments are best described
as a practical FuseMoE-side ablation of DeepSeek-inspired router initialization,
noise removal, top-k behavior, gate normalization, additive bias, auxiliary
balancing, shared experts, and hidden-feature expert mixing.

## 28. Expert-Specialization Diagnostics Added

After the DeepSeek-style router ablations, we added a more direct diagnostic
for whether the routed experts are actually learning different functions. This
was motivated by the observation that router mass alone is not enough:

```text
balanced routing can still mean redundant experts
collapsed routing can still hide whether unused experts are different
random routing can look specialized but may not correspond to useful cohorts
```

### 28.1 New Diagnostic Flag

Added an opt-in CLI flag:

```bash
--log_expert_output_diagnostics
```

This leaves normal experiments unchanged. When enabled, it prints diagnostics
for train, validation, and test.

Implementation locations:

```text
src/utils/util.py
src/utils/config.py
src/core/module.py
src/core/sparse_moe.py
src/core/train.py
```

The code passed Python compilation:

```bash
python -m py_compile \
  src/core/sparse_moe.py \
  src/core/train.py \
  src/core/module.py \
  src/utils/config.py \
  src/utils/util.py
```

### 28.2 What It Measures

For each MoE layer and modality, the diagnostic records all expert outputs
before routing/gating combination:

```python
outs = torch.stack(expert_outs, dim=1)   # [B, E, D]
outs = F.normalize(outs, dim=-1)
sim = torch.einsum("bed,bfd->bef", outs, outs).mean(dim=0)
```

Printed fields:

```text
offdiag cosine:
  mean pairwise expert-output cosine excluding the diagonal

sim matrix:
  full E x E expert-output cosine matrix

norm:
  mean output norm for each expert
```

Interpretation:

```text
high offdiag cosine  -> expert outputs are redundant
low offdiag cosine   -> experts produce more distinct functions
near-zero norm expert -> expert exists but contributes weakly
```

The diagnostic also records selected-expert cohort statistics:

```text
selected count per expert
label rate per selected expert
CXR missing rate
text missing rate
ECG missing rate
```

This is meant to answer whether routing separates clinically or data-quality
meaningful cohorts, not just whether gate mass is balanced.

### 28.3 Smoke Test

Smoke job:

```text
job id: 10851879
log: out/Week_27/deepseek_arch_grid/10851879_diag_smoke.out
task: IHM
seed: 32
modeltype: TS_CXR_Text
router: permod
experts: 4
top_k: 2
shared_experts: 0
arch_variant: random_router
router_init: xavier
noisy_gating: False
router_noise_scale: 0.0
epochs: 2
```

This smoke test intentionally used only 2 epochs to validate the diagnostic
implementation. It is not directly comparable to the 40-epoch run
`10848526_r40_ihm_s32_p0.out`, though the architecture/router/data settings
are otherwise the same.

Final smoke-test metrics:

```text
best validation f1: 0.3684
test auc:   0.8050
test auprc: 0.4842
test f1:    0.3913
```

### 28.4 Smoke-Test Routing Diagnosis

The random-init no-noise router is not uniform, but it often becomes a hard
deterministic pair selector.

Representative test routing:

```text
L0M2 sel[0.000,1.000,0.000,1.000]
L1M1 sel[1.000,1.000,0.000,0.000]
L1M2 sel[1.000,1.000,0.000,0.000]
L2M1 sel[1.000,1.000,0.000,0.000]
L2M2 sel[1.000,0.000,0.000,1.000]
```

This supports the earlier diagnosis:

```text
random init removes artificial uniformity
but it does not guarantee clinically meaningful specialization
```

### 28.5 Smoke-Test Expert Output Similarity

Some experts are distinct in early layers, but deeper layers can still be
partly redundant.

Representative test off-diagonal expert-output cosine:

```text
L0m0: 0.059
L0m1: 0.131
L0m2: 0.108
L1m0: 0.477
L1m1: 0.136
L1m2: 0.113
L2m0: 0.320
L2m1: 0.077
L2m2: 0.102
```

Interpretation:

```text
Layer 0 mostly has distinct expert outputs.
Layer 1 modality 0 and layer 2 modality 0 show higher redundancy.
Routing specialization and expert-function specialization are not identical.
```

### 28.6 Smoke-Test Cohort Diagnostics

Because this run uses the complete `TS_CXR_Text` setting, missingness is always
zero:

```text
cxr_missing=0.000
text_missing=0.000
ecg_missing=0.000
```

Label-rate differences across routed cohorts exist but are modest:

```text
L0m0e0 label=0.145
L0m0e1 label=0.195
L2m0e0 label=0.191
L2m0e1 label=0.151
```

Interpretation:

```text
The router is not clearly separating strong mortality-risk cohorts in this
2-epoch smoke test. A longer diagnostic run is needed before making a strong
claim.
```

## 29. Expert-Diagnostic Grid Submitted

After the smoke test succeeded, a 16-job diagnostic grid was submitted.

Manifest:

```text
out/Week_27/deepseek_arch_grid/expert_diagnostics_manifest.tsv
```

All jobs use:

```text
modeltype: TS_CXR_Text
router: permod
experts: 4
top_k: 2
shared_experts: 0
epochs: 16
balance_loss_coef: 0.01
router_print_mode: concise
log_expert_output_diagnostics: True
```

Tasks:

```text
IHM
LOS
```

Seeds:

```text
32, 42
```

Compared variants:

```text
fusemoe_original:
  zero router init + Shazeer noisy top-k

zero_no_noise:
  zero router init + no noise

random_router:
  Xavier router init + no noise

random_low_noise_decay:
  Xavier router init + noise scale 0.1 -> 0.0 over 8 epochs
```

Submitted jobs:

```text
10852208 diag_ihm_s32_orig
10852209 diag_ihm_s32_znon
10852210 diag_ihm_s32_rand
10852211 diag_ihm_s32_rndec
10852212 diag_ihm_s42_orig
10852213 diag_ihm_s42_znon
10852214 diag_ihm_s42_rand
10852215 diag_ihm_s42_rndec
10852216 diag_los_s32_orig
10852217 diag_los_s32_znon
10852218 diag_los_s32_rand
10852219 diag_los_s32_rndec
10852220 diag_los_s42_orig
10852221 diag_los_s42_znon
10852222 diag_los_s42_rand
10852223 diag_los_s42_rndec
```

At submission time, all jobs were pending with `(Priority)`.

### 29.1 Scientific Purpose

This grid is not another performance-only search. It is designed to separate
three mechanisms:

```text
1. Original noise-driven exploration:
   Does it create real expert diversity or just uniform routing?

2. Zero init without noise:
   Does deterministic top-k tie behavior explain collapse?

3. Random init without noise:
   Does breaking the initial symmetry create real specialization?

4. Random init with small decaying noise:
   Can we combine symmetry breaking with early exploration without forcing
   uniform routing forever?
```

### 29.2 What To Read From This Grid

For each run, inspect:

```text
performance:
  validation f1 and final test auc/auprc/f1

routing:
  train/validation/test router summary
  exact selected expert distribution per layer and modality

expert diversity:
  expert output offdiag cosine by layer/modality

cohort separation:
  label rate by selected expert
  missingness rate by selected expert
```

Expected useful conclusions:

```text
If original noisy FuseMoE has uniform routing and high expert-output similarity:
  the uniformity is likely not meaningful specialization.

If random init has non-uniform routing but high expert-output similarity:
  the router specializes, but experts remain redundant.

If random init has non-uniform routing and low output similarity:
  true expert specialization is emerging.

If cohort label/missingness rates are nearly identical across experts:
  routing is not clinically/data-quality interpretable yet.
```
