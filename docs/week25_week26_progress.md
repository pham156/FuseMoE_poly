# FuseMoE Week25-Week26 Experiment Log

This document records the main implementation work and experimental evidence from the FuseMoE semantic-routing / prototype-routing / Lingshu / missing-modality exploration.

## 1. Baseline Architecture Clarification

The current FuseMoE pipeline does not consume raw ICU data directly inside the MoE router.

The model first converts each modality into a projected latent sequence:

- time series: projected into `[tt_max, batch, embed_dim]`
- notes: precomputed or BioLongformer/BERT embeddings, then projected into `[tt_max, batch, embed_dim]`
- CXR: image feature embeddings projected into `[tt_max, batch, embed_dim]`
- ECG: ECG feature embeddings projected into `[tt_max, batch, embed_dim]`

For `joint` routing, each modality is flattened and concatenated. With `tt_max=48`, `embed_dim=128`, and 4 modalities, the router input is:

```text
48 * 128 * 4 = 24576
```

The model has 3 stacked MoE layers. With 4 experts, that means:

```text
layer 0: router -> 4 MLP experts
layer 1: router -> 4 MLP experts
layer 2: router -> 4 MLP experts
```

So there are 12 physical expert MLPs in the stacked model, but only 4 expert slots per layer.

Important conclusion:

- This is patient-level / fused-representation-level routing.
- It is not true token-wise routing.
- It is not true sentence-level routing in the mentor note.
- The sentence-level work below was diagnostic only, not integrated into the model forward pass.

## 2. Matched FuseMoE Baseline And Routing Diagnostics

Main files:

- `src/core/sparse_moe.py`
- `src/core/train.py`
- `src/scripts/fusemoe_balance_smoke/`
- `out/Week_26/10718059_fuse_gate_fuse_gate.out`

Implemented diagnostics:

- per-layer router gate mass
- per-layer expert selection frequency
- optional router diagnostic CSV
- balance loss printing
- `--disable_run_folder_save` for experiments where only logs/CSV are needed

Matched baseline run:

```text
task: ihm-48-cxr-notes-ecg
modeltype: TS_CXR_Text
router: joint
experts: 4
top_k: 2
gating: poly
poly_power: 2
balance_loss_coef: 0.01
shared_experts: 0
seed: 30
ratios: 0.2, 0.4, 0.6, 0.8, 1.0
epochs: 8
```

Important caveat:

- This baseline used task `ihm-48-cxr-notes-ecg`, but `modeltype=TS_CXR_Text`.
- That means ECG may be present in the task/data files, but this forward pass does not actually use ECG.
- Later ECG-compatible experiments use `modeltype=TS_CXR_Text_ECG` and `num_modalities=4`.

Observed routing behavior:

- Softmax routing collapsed heavily.
- Polynomial routing improved diversity compared with softmax, but still showed strong layer-wise imbalance.
- Some experts reached near-zero usage in some layers.
- In the full-ratio poly baseline, some layer summaries showed patterns such as:

```text
layer0: e0/e2/e3 active, e1 sometimes near zero
layer1: e1/e3 often dominate
layer2: e1/e2 often dominate, e3 can become near zero
```

Interpretation:

- Top-k does not guarantee balanced expert use.
- With top-k 2, two experts are selected per sample, but the same two experts can be selected almost every time.
- Balance regularization helps only if the router still has useful competing alternatives.

## 3. Balance Loss And Shared Expert Experiments

Main scripts:

- `src/scripts/fusemoe_balance_smoke/job_fusemoe_balance_smoke.sbatch`
- `src/scripts/fusemoe_balance_smoke/job_fusemoe_balance_coef1_week26.sbatch`
- `src/scripts/fusemoe_balance_smoke/job_fusemoe_shared1_balance_coef1_week26.sbatch`
- `src/scripts/fusemoe_balance_smoke/job_fusemoe_balance_gating_week26.sbatch`

Runs/results:

- `out/Week_26/10716257_fuse_bal8grid.out`
- `out/Week_26/10716390_fuse_bal8grid.out`
- `out/Week_26/10716542_fuse_balcoef1.out`
- `out/Week_26/10716543_fuse_shared1.out`

What was tested:

- higher balance regularization
- `balance_loss_coef=1.0`
- shared expert path
- multiple ratios
- 8 epochs instead of short smoke runs

Finding:

- Larger balance loss sometimes improved routing slightly in larger data regimes.
- It did not solve collapse by itself.
- Shared expert did not automatically fix router collapse.

Interpretation:

- The issue is not only regularization strength.
- The router signal itself is weak or unstable for forcing clinically meaningful expert selection.

## 4. Gating Function Comparison

Main scripts:

- `src/scripts/fusemoe_balance_smoke/job_fusemoe_balance_gating_week26.sbatch`
- Original-code comparison under `/home/pham156/MoE/FuseMoE_ori/scripts/original_gating_week26/`

Gating functions tested:

- softmax
- laplace
- polynomial
- unnormalized polynomial

Relevant outputs:

- `out/Week_26/10716750_fuse_gate_fuse_gate.out`
- `out/Week_26/10718057_fuse_gate_fuse_gate.out`
- `out/Week_26/10718058_fuse_gate_fuse_gate.out`
- `out/Week_26/10718059_fuse_gate_fuse_gate.out`
- original-code logs under `/home/pham156/MoE/FuseMoE_ori/out/Week_26/original_gating/`

Finding:

- Softmax showed the clearest collapse.
- Laplace and polynomial-style gates were more stable.
- Polynomial routing remains useful as a routing kernel, but it does not create clinical interpretability by itself.

Current defensible claim:

```text
Polynomial routing helps avoid extreme softmax-style winner-take-all behavior, but expert semantics still require additional constraints or validation.
```

## 5. Semantic Expert Profiles

Main files:

- `src/core/model.py`
- `src/core/sparse_moe.py`
- `src/scripts/semantic_router_diagnostic_week26/`

Implemented flags:

- `--use_semantic_expert_profiles`
- `--semantic_profile_set icu_organ_system`
- `--semantic_profile_scale`
- `--semantic_profile_fusion add|replace`
- `--semantic_profile_source patient|note`
- `--semantic_profile_note_pooling max|mean`
- `--semantic_profile_layers all|first`

Expert profiles used:

```text
cardiovascular
respiratory
renal_metabolic
infection
```

Older severity-style profile was removed because severity behaved like a catch-all expert and competed with every other profile.

Experiments:

- Week26 semantic profile grid:
  - `out/Week_26/10704198_sem_diag26.out`
  - `out/Week_26/10704199_sem_diag26.out`
  - `out/Week_26/10704200_sem_diag26.out`
  - `out/Week_26/10704201_sem_diag26.out`
  - `out/Week_26/10704202_sem_diag26.out`
- Semantic polynomial profile runs:
  - `out/Week_26/10720690_sem_poly26.out`
  - `out/Week_26/10720715_sem_poly26.out`
- Note-source semantic routing:
  - `out/Week_26/10721664_sem_note26.out`
- Replace-mode semantic routing:
  - `out/Week_26/10721792_sem_repl26.out`

Findings:

- Patient-source fixed semantic profiles often collapsed into a small subset of experts.
- Note-source semantic routing did not cleanly recover clinically meaningful routing.
- Replace mode made routing more explicitly controlled by semantic scores, but the semantic scores themselves were weak.

Example from semantic profile run:

```text
layer0 gate mass often concentrated on two experts
layer1/layer2 still developed independent collapse patterns
```

Interpretation:

- Adding fixed semantic priors to the existing patient-level router is not enough.
- The semantic profiles live in a text embedding space, while the router input is a flattened multimodal latent vector.
- This space mismatch is probably a major reason the semantic signal was weak.

## 6. Router Diagnostic CSV For Interpretability

Main files:

- `src/core/train.py`
- `src/core/sparse_moe.py`

Implemented:

- `--log_router_diagnostics`
- `--router_diagnostics_path`
- `--router_diagnostics_layers last|all`
- `--router_diagnostics_max_text_chars`

CSV behavior:

- one row per test sample per logged layer
- includes sample id if available
- includes note text excerpt
- includes layer index
- includes selected expert ids
- includes gate weights
- includes semantic profile labels when semantic routing is active
- deeper layers that are not semantically guided now use plain expert indices rather than fake organ names

Important limitation:

- A single ICU stay can contain many organ systems at once.
- Interpreting one patient row as "this patient is respiratory" is too strong unless there are validated organ labels.
- Cohort-level analysis is more defensible than single-case interpretation.

## 7. Note/Profile Similarity Diagnostics

Main scripts:

- `src/scripts/semantic_router_diagnostic_week26/diagnose_note_profile_similarity.py`
- `src/scripts/semantic_router_diagnostic_week26/job_note_profile_similarity_week26.sbatch`

Outputs:

- `out/Week_26/note_profile_similarity/train_note_profile_similarity.csv`
- `out/Week_26/note_profile_similarity/val_note_profile_similarity.csv`
- `out/Week_26/note_profile_similarity/test_note_profile_similarity.csv`
- `out/Week_26/note_profile_similarity/summary_note_profile_similarity.csv`
- job log: `out/Week_26/10721833_note_sim26.out`

Summary pattern:

```text
test total rows: 777
cardiovascular weak count: 505, top1 count: 727
respiratory weak count: 759, top1 count: 0
renal_metabolic weak count: 174, top1 count: 50
infection weak count: 226, top1 count: 0
```

Interpretation:

- Whole-note similarity is dominated by cardiovascular-like profile scores.
- Respiratory and infection rarely/never become top-1 even though weak keyword labels mark many notes as respiratory/infection.
- This suggests fixed note-profile similarity is not semantically reliable enough in this setup.

## 8. Sentence/Profile Similarity Diagnostics

Main scripts:

- `src/scripts/semantic_router_diagnostic_week26/diagnose_sentence_profile_similarity.py`
- `src/scripts/semantic_router_diagnostic_week26/job_sentence_profile_similarity_week26.sbatch`

Outputs:

- `out/Week_26/sentence_profile_similarity/train_sentence_profile_similarity.csv`
- `out/Week_26/sentence_profile_similarity/val_sentence_profile_similarity.csv`
- `out/Week_26/sentence_profile_similarity/test_sentence_profile_similarity.csv`
- job log: `out/Week_26/10721864_sent_sim26.out`

Sentence top-1 counts:

```text
train: cardiovascular 6525, respiratory 65319, renal_metabolic 505, infection 51609
val:   cardiovascular 1472, respiratory 14257, renal_metabolic 100, infection 11556
test:  cardiovascular 1517, respiratory 13675, renal_metabolic 94, infection 10732
```

Observed score issue:

- Sentence profile scores were extremely close.
- Example values looked like:

```text
0.0744, 0.0723, 0.0739, 0.0700
```

Interpretation:

- Sentence-level semantic scoring is more plausible than whole-note routing, but the raw cosine margins are tiny.
- This diagnostic supports the mentor's sentence-level motivation, but it does not yet provide a strong router signal.
- A true sentence-level router would need sentence/entity aggregation inside the model, not just post-hoc CSV diagnostics.

## 9. Prototype Router

Main files:

- `src/core/sparse_moe.py`
- `src/utils/config.py`
- `src/utils/util.py`
- `src/scripts/prototype_router_week26/`

Implemented flags:

- `--use_prototype_router`
- `--prototype_router_dim`
- `--prototype_router_temperature`
- `--prototype_router_dense`
- `--prototype_router_orth_coef`

Design:

```text
router input x
-> projection z
-> similarity/distance to learnable prototypes
-> dense or top-k gate
-> expert weighted sum
```

Prototype dense run:

- script: `src/scripts/prototype_router_week26/job_prototype_router_dense_week26.sbatch`
- log: `out/Week_26/10722205_proto_dense26.out`
- diagnostics:
  - `out/Week_26/prototype_router/prototype_dense_shared_seed30_ratio_*_router_diagnostics.csv`

Config:

```text
task: ihm-48-cxr-notes-ecg
modeltype: TS_CXR_Text
seed: 30
ratios: 0.2, 0.4, 0.6, 0.8, 1.0
experts: 4
top_k: 4
dense prototype routing: on
shared_experts: 1
prototype_orth_coef: 0.1
```

Important caveat:

- This run used `TS_CXR_Text`, so ECG was not actually used.

Full-ratio routing became much more even:

```text
layer0 roughly [0.24, 0.19, 0.35, 0.21]
layer1 roughly [0.20, 0.17, 0.41, 0.22]
layer2 roughly [0.20, 0.34, 0.20, 0.25]
selection frequency: all experts selected because dense routing/top_k=4
```

Full-ratio metrics:

```text
AUC   about 0.805
AUPRC about 0.443
F1    about 0.444
```

Interpretation:

- Dense prototype routing solves the mechanical collapse problem better than top-k routing.
- However, pure learnable prototypes are not inherently organ-interpretable.
- They should be called clinical latent prototypes unless constrained or validated.

## 10. Weak Organ Router Supervision

Main files:

- `src/core/train.py`
- `src/core/module.py`
- `src/core/sparse_moe.py`
- `src/scripts/prototype_router_week26/analyze_prototype_alignment.py`

Implemented:

- weak note-keyword organ labels
- router organ supervision loss
- class-balanced router supervision
- layer selection for supervision

Flags:

- `--use_router_organ_supervision`
- `--router_organ_supervision_coef`
- `--router_organ_supervision_layers all|first`
- `--router_organ_supervision_class_balanced`

Weak keyword labels:

```text
cardiovascular: blood pressure, hypotension, vasopressor, heart rate, shock, cardiac, etc.
respiratory: oxygen, ventilation, spo2, lung, pneumonia, intubated, pulmonary, cxr, etc.
renal_metabolic: creatinine, bun, urine output, dialysis, electrolytes, sodium, potassium, lactate, etc.
infection: sepsis, fever, antibiotic, wbc, culture, inflammation, etc.
```

All-layer organ-supervised run:

- script: `src/scripts/prototype_router_week26/job_prototype_router_organ_supervised_week26.sbatch`
- log: `out/Week_26/10722760_proto_org26.out`
- modeltype: `TS_CXR_Text_ECG`
- num_modalities: `4`
- ratio: `1.0`
- router supervision: all layers

Observed routing:

```text
layer0 about [0.30, 0.48, 0.10, 0.12]
layer1 about [0.30, 0.48, 0.10, 0.12]
layer2 about [0.30, 0.48, 0.10, 0.12]
```

Interpretation:

- All layers copied the weak-label frequency pattern.
- It did not create useful layer-specific specialization.
- Respiratory/cardio dominated because keyword labels are broad and frequent.

Layer0-only class-balanced run:

- script: `src/scripts/prototype_router_week26/job_prototype_router_organ_l0_grid_week26.sbatch`
- log: `out/Week_26/10722891_proto_l0g26.out`
- modeltype: `TS_CXR_Text_ECG`
- ratio: `1.0`
- coefs: `0.02`, `0.05`
- router supervision: first layer only
- class balanced: yes

Coef `0.02`:

```text
metrics: AUC about 0.791, AUPRC about 0.420, F1 about 0.454
layer0 about [0.288, 0.337, 0.182, 0.193]
layer1 approximately uniform
layer2 approximately uniform
```

Coef `0.05`:

```text
metrics: AUC about 0.772, AUPRC about 0.405, F1 about 0.479
layer0 about [0.298, 0.389, 0.137, 0.176]
layer1 approximately uniform
layer2 approximately uniform
```

Interpretation:

- Layer0-only supervision is structurally cleaner than all-layer supervision.
- It guides only the first MoE layer while leaving deeper layers less constrained.
- Still, clinical alignment is weak because the labels are crude.

## 11. Prototype Alignment Analysis

Main script:

- `src/scripts/prototype_router_week26/analyze_prototype_alignment.py`

Outputs:

- `out/Week_26/prototype_router/alignment_auto/prototype_weak_label_alignment_report.txt`
- `out/Week_26/prototype_router/alignment_auto/prototype_weak_label_alignment_summary.csv`

What it measures:

- For each weak organ label, compare average gate mass for weak-label-positive vs weak-label-negative patients.
- A useful organ expert should show positive lift for the corresponding expert.

Finding:

- Lift values were near zero, often around `0.0000` to `0.001`.
- That means weak organ labels did not meaningfully separate routing behavior.

Interpretation:

- We cannot honestly claim the prototype experts learned organ semantics from the current keyword supervision.
- The current weak labels are mostly a noisy proxy, not reliable clinical labels.

## 12. Lingshu Compatibility And LoRA Work

Main files/scripts:

- `src/core/lingshu_pseudotoken.py`
- `src/scripts/main_mimiciv_lingshu.py`
- `src/scripts/lingshu_lora_experts/`

Scripts created:

- `job_lingshu_pseudotoken_smoke.sbatch`
- `job_lingshu_compat_week26.sbatch`
- `job_lingshu_week22_matched_week26.sbatch`
- `job_lingshu_organ_moe_smoke.sbatch`
- `job_lingshu_organ_moe_los_diag_week26.sbatch`

Logs:

- `out/Week_25/lingshu_smoke_*.out`
- `out/Week_26/lingshu/lingshu_compat26_*.out`
- `out/Week_26/lingshu/lingshu_w22_*.out`
- `out/Week_26/lingshu/lingshu_orgmoe_*.out`

What was learned:

- Lingshu can only consume the current preprocessed embeddings through an adapter/pseudo-token bridge.
- This is not equivalent to feeding raw ICU notes/images/signals into a medical LLM.
- The current bridge mostly makes Lingshu a feature-processing backbone after projection.
- True "each expert is a Lingshu LoRA adapter" is a larger architectural change and does not naturally fit the current precomputed `.pkl` feature pipeline.

Current conclusion:

- Lingshu is possible, but it is not the cheapest or cleanest next step.
- Embedding-level missing-modality reconstruction fits the current data much better.

## 13. Missing-Modality Reconstruction

Main files:

- `src/core/model.py`
- `src/utils/config.py`
- `src/utils/util.py`
- `src/scripts/missing_modality_week26/job_missing_recon_week26.sbatch`

Implemented flags:

- `--use_missing_modality_recon`
- `--missing_modality_recon_coef`
- `--missing_modality_recon_targets`
- `--missing_modality_recon_hidden`

Implementation:

For each projected modality embedding:

```text
proj_x_modality: [tt_max, batch, embed_dim]
pooled_modality = mean over time -> [batch, embed_dim]
```

For a target modality such as CXR:

```text
source_embedding = average pooled embedding of other observed modalities
predicted_cxr = reconstruction_head_cxr(source_embedding)
loss = MSE(predicted_cxr, stop_gradient(true_cxr_embedding))
```

The final training loss becomes:

```text
task_loss
+ balance_loss_coef * balance_loss
+ missing_modality_recon_coef * recon_loss
```

The reconstruction loss is only active when `--use_missing_modality_recon` is passed.

Current submitted run:

- job: `10730151`
- script: `src/scripts/missing_modality_week26/job_missing_recon_week26.sbatch`
- status: finished
- start time: `2026-05-19T00:13:52`
- output:
  - `out/Week_26/10730151_miss_recon26.out`
  - `out/Week_26/10730151_miss_recon26.err`

Config:

```text
task: ihm-48-cxr-notes-ecg
modeltype: TS_CXR_Text_ECG
num_modalities: 4
router: original joint poly
experts: 4
top_k: 2
seed: 30
ratio: 1.0
epochs: 8
missing_recon_coef: 0.05
missing_recon_targets: cxr,ecg,text
```

Result:

```text
Best validation F1: 0.4327

Final test:
AUC:   0.7973
AUPRC: 0.4496
F1:    0.4568
```

Held-out router diagnostics:

```text
diagnostic CSV:
out/Week_26/missing_modality/missing_recon_ecg_seed30_ratio_1p0_coef_0p05_router_diagnostics.csv

test rows: 617

layer0 gate mass: [0.4999, 0.0000, 0.0000, 0.5001]
layer0 selection: [1.0000, 0.0000, 0.0000, 1.0000]

layer1 gate mass: [0.4998, 0.0000, 0.0000, 0.5002]
layer1 selection: [1.0000, 0.0000, 0.0000, 1.0000]

layer2 gate mass: [0.0000, 0.4996, 0.5004, 0.0000]
layer2 selection: [0.0000, 1.0000, 1.0000, 0.0000]
```

Interpretation:

- The reconstruction run did not collapse task performance.
- AUPRC and F1 are reasonable compared with earlier Week26 runs.
- However, the router still collapses into a fixed top-2 expert pair per layer.
- Missing-modality reconstruction alone does not solve routing diversity.
- This run is still useful because it validates that embedding-level reconstruction can be added without breaking the model.

Why this is the current best next step:

- It matches the mentor's missing-domain prediction idea at the embedding level.
- It uses the current preprocessed data directly.
- It avoids pretending fixed text profile cosine scores are clinically reliable.
- It sets up a clean robustness experiment: evaluate performance when CXR/ECG/text are removed or missing.

## 14. What We Can Defend Right Now

Defensible:

- Current FuseMoE uses patient-level routing over precomputed multimodal latent features.
- Softmax-like routing can collapse.
- Polynomial/Laplace-style routing is useful for reducing extreme collapse.
- Fixed organ text-profile cosine similarity is weak in the current feature setting.
- Learnable dense prototype routing improves route balance.
- Pure learnable prototypes are not automatically organ-interpretable.
- Keyword weak supervision is too noisy to prove clinical expert specialization.
- Embedding-level missing-modality reconstruction is the most compatible version of the mentor's missing-domain idea for this codebase.

Not defensible yet:

- Claiming the experts are true organ experts.
- Claiming sentence-level routing is implemented in the actual model.
- Claiming Lingshu-LoRA is already a clean replacement for MLP experts.
- Claiming per-patient routing choices are clinically explanatory without stronger labels.

## 15. Recommended Next Experiments

Immediate:

1. Wait for job `10730151`.
2. Compare its metrics against the matched poly baseline.
3. Compare its router distribution against `10718059_fuse_gate_fuse_gate.out`.

If missing reconstruction does not collapse performance:

1. Add evaluation-time modality removal flags.
2. Run robustness tests:
   - CXR removed
   - ECG removed
   - text removed
   - CXR+ECG removed
3. Compare original FuseMoE vs reconstruction-trained FuseMoE under the same missing-modality setting.

If we still want organ interpretability:

1. Ask for better structured clinical weak-label rules.
2. Prefer structured variables/labs/vitals over note keywords.
3. Validate routing at cohort level:
   - high creatinine/BUN should increase renal/metabolic expert mass
   - abnormal oxygen/ventilation/CXR should increase respiratory expert mass
   - abnormal ECG/HR/BP/vasopressor evidence should increase cardiovascular expert mass
   - fever/WBC/antibiotic/culture/sepsis evidence should increase infection expert mass

## 16. Suggested Paper Story From Current Evidence

A technically honest story:

```text
We first analyze routing collapse in FuseMoE under low-data and full-data ICU prediction.
Polynomial-style routing reduces extreme winner-take-all behavior compared with softmax,
but routing remains difficult to interpret because current ICU inputs are fused,
precomputed, and multi-organ.

Direct fixed semantic anchors from clinical text descriptions are too weak in this
precomputed multimodal latent space. We therefore treat clinical priors as soft
regularization rather than hard labels.

For the current data pipeline, embedding-level missing-domain reconstruction is a
more compatible way to inject clinical structure: the model learns cross-modal
relationships such as TS/Text -> CXR or TS/Text -> ECG while keeping the original
FuseMoE architecture runnable.
```

This keeps polynomial routing in the story while avoiding overclaiming organ-level interpretability.

## 17. Week26 Collapse-Fix Follow-Up: X-MOE Router + Shared Expert

After the semantic-profile and missing-reconstruction experiments, we returned to the simpler question:

```text
Can we fix expert collapse in the original FuseMoE setting without adding clinical semantic constraints?
```

The main issue observed in the original router is that it routes directly from a very large flattened patient vector:

```text
3 modalities: x in R^(48 * 128 * 3) = R^18432
4 modalities: x in R^(48 * 128 * 4) = R^24576
```

This makes the router logits easy to dominate by a few dimensions or one expert's early advantage. Balance loss alone did not fix this reliably.

### 17.1 Implemented X-MOE-Style Router

We added an optional low-dimensional normalized router path behind flags.

Important flags:

```text
--use_xmoe_router
--xmoe_router_dim
--xmoe_noise_scale
--shared_experts
```

The experts still receive the full flattened FuseMoE feature vector. Only the router input is changed:

```text
full flattened patient vector x
        |
        |  router projection only
        v
low-dimensional normalized router embedding z
        |
        |  similarity or distance to expert embeddings
        v
router logits
```

This means X-MOE does not discard the full patient representation used by the MLP experts. It only makes the routing decision in a smaller and better-conditioned space.

Implemented variants:

```text
softmax X-MOE:
    logits = normalize(project(x)) dot normalize(expert_embedding)

laplace X-MOE:
    logits = -distance(project(x), expert_embedding)

poly X-MOE:
    score = 1 / (1 + distance(project(x), expert_embedding)^r)
```

We also added configurable router noise:

```text
effective_noise_std = learned_noise_std * xmoe_noise_scale
```

### 17.2 Shared Expert

We also enabled `--shared_experts 1`.

Motivation:

- Without a shared expert, one routed expert tends to become the general ICU expert.
- With a shared expert, common ICU signal can pass through the shared path.
- Routed experts have less pressure to carry universal information.

For the current joint routing implementation, routed and shared outputs are combined in probability space as an approximate 50/50 mixture:

```text
combined_log_prob = logaddexp(routed_log_prob, shared_log_prob) - log(2)
```

This is not meant to create a clinically interpretable shared expert yet. It is a stabilization mechanism.

### 17.3 Paper-Config Runs

We also tested closer to the FuseMoE paper configuration:

```text
num_experts: 16
top_k: 4
disjoint_top_k: 2
MoE layers: 3
FFN hidden size: 512
epochs: 8 for paper-style checks, 20 for collapse-fix tuning
seed: 30 for most Week26 diagnostics
task: ihm-48-cxr-notes-ecg
router: joint
```

### 17.4 Corrected X-MOE Results Table

These are the verified results from the actual Week26 log files. Earlier notes mixed some job IDs, so this table should be treated as the corrected reference.

| Job ID | Router | Experts / Top-k | Noise | Balance coef | Shared | AUC | AUPRC | F1 | Routing note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `10735378` | softmax X-MOE | 16 / 4 | 1.0 | 0.02 | 1 | 0.8096 | 0.4331 | 0.5087 | Almost perfectly uniform routing; no collapse, possibly too uniform. |
| `10735562` | softmax X-MOE | 16 / 4 | 0.3 | 0.01 | 1 | 0.7992 | 0.4092 | 0.4131 | Poorer performance; layer 2 still had near-dead experts. |
| `10735563` | laplace X-MOE | 16 / 4 | 0.3 | 0.01 | 1 | 0.8111 | 0.4374 | 0.5153 | Best verified F1 so far among this group. |
| `10735663` | softmax X-MOE | 16 / 4 | 0.2 | 0.01 | 1 | 0.8045 | 0.4371 | 0.4820 | Underbalanced; many ignored experts. |
| `10735664` | softmax X-MOE | 16 / 4 | 0.3 | 0.005 | 1 | 0.8090 | 0.4397 | 0.4950 | Better than low-noise softmax, but still some layer-2 dead experts. |

Current interpretation:

- X-MOE + shared expert is the strongest collapse-prevention direction so far.
- Laplace X-MOE with moderate noise currently gives the best verified performance/balance tradeoff.
- Softmax X-MOE is sensitive to noise and balance coefficient.
- Too much regularization/noise makes the router look artificially uniform.
- Too little regularization/noise still leaves ignored experts, especially in deeper MoE layers.

The desired behavior is not perfect uniformity. The goal is:

```text
no dead experts
some specialization
stable task performance
not a fixed top-k pair for every patient
```

### 17.5 Runs Still Being Checked

The following tuning runs were submitted after the corrected table above:

| Job ID | Router | Experts / Top-k | Noise | Balance coef | Shared | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `10735830` | softmax X-MOE | 16 / 4 | 0.5 | 0.005 | 1 | Test whether higher noise with weaker balance improves softmax diversity without overbalancing. |
| `10735831` | softmax X-MOE | 16 / 4 | 0.5 | 0.01 | 1 | Test higher noise with stronger balance. |
| `10735906` | softmax X-MOE | 4 / 2 | 0.5 | 0.01 | 1 | Compare X-MOE collapse behavior in the original 4-expert setting. |
| `10735907` | laplace X-MOE | 4 / 2 | 0.5 | 0.01 | 1 | Compare laplace X-MOE in the original 4-expert setting. |

These should be checked before deciding the next stable setting.

### 17.6 Practical Next Decision

For the next round, prioritize routing stability before adding more semantic constraints.

Recommended order:

1. Finish reading jobs `10735830`, `10735831`, `10735906`, and `10735907`.
2. Pick one stable 16-expert setting and one stable 4-expert setting.
3. Compare against the original softmax baseline using the same seed/task/ratio.
4. Only after this, reintroduce semantic or missing-modality objectives.

Current best candidate to keep:

```text
laplace X-MOE
num_experts = 16
top_k = 4
shared_experts = 1
xmoe_noise_scale around 0.3
balance_coef around 0.01
```

Softmax X-MOE may still be viable, but it needs tuning around:

```text
xmoe_noise_scale: 0.3 to 0.5
balance_coef: 0.005 to 0.01
```

The main paper-relevant conclusion from this section:

```text
Expert collapse in this FuseMoE setup is driven less by the clinical meaning of experts
and more by the conditioning of the router input/logits. A low-dimensional normalized
router space plus an explicit shared expert is currently the most effective engineering
fix for collapse.
```
