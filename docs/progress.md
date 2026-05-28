# FuseMoE Progress Log

This document records the main implementation work and experimental evidence from the FuseMoE semantic-routing / prototype-routing / Lingshu / missing-modality / X-MOE exploration.

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

## 18. Week27 Matched Base vs X-MOE Grid

### 18.1 Goal

After the Week26 routing diagnostics, the next goal was to test whether the X-MOE router path improves performance compared with the base FuseMoE router under matched settings.

The intended Week27 comparison grid is:

```text
tasks: ihm-48-cxr-notes-ecg, los-48-cxr-notes-ecg, pheno-all-cxr-notes-ecg
seeds: 32, 42, 52, 62, 72
gates: softmax, laplace, poly
router variants: base, xmoe
shared experts: 0, 1
routing modes: joint, permod
modalities:
    TS_CXR_Text      -> tstxtcxr
    TS_CXR_Text_ECG  -> tstxtcxrecg
experts: 4
top_k: 2
epochs: 16
balance_loss_coef: 0.01
```

This is designed to answer a narrower question than the semantic-routing experiments:

```text
Holding the rest of the model fixed, does X-MOE improve routing stability and/or task performance?
```

### 18.2 Week27 Scripts

Main files:

- `src/scripts/week27_xmoe_grid/job_week27_xmoe_config.sbatch`
- `src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh`
- `src/scripts/week27_xmoe_grid/README.md`

The job script runs one matched config across the three tasks. The submit script loops over seeds, gates, base/X-MOE, shared/no-shared, joint/permod, and modality sets.

Important correction:

- The first submit script used dependency lanes.
- That was a bad fit for this cluster because it serialized many jobs and caused long pending time.
- The corrected submit behavior is independent jobs by default.
- Dependency lanes are now opt-in only.

Current submit behavior:

```bash
cd /home/pham156/MoE/FuseMoE_poly
bash src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh
```

Optional dependency lanes, only if explicitly needed:

```bash
USE_DEPENDENCIES=1 LANES=8 bash src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh
```

Do not use dependency lanes as the default strategy for this grid. Independent jobs are the safer queue strategy here.

### 18.3 Performance CSV Extraction

Created Week27 result CSVs from completed `.out` logs:

- `out/Week_27/week27_completed_performance.csv`
- `out/Week_27/week27_base_vs_xmoe_matched.csv`
- `out/Week_27/week27_base_vs_xmoe_summary.csv`

Parsing rule:

- The CSV now includes every task block that has already reached `FINAL TEST RESULTS`, including finished task blocks from jobs that are still running later tasks.
- The `job_finished` column records whether the full three-task job has completed.
- Matched comparison requires identical:
  - seed
  - gate
  - shared expert setting
  - routing mode
  - modality set
  - task

Primary metric:

```text
IHM:   f1
LOS:   f1
PHENO: macro_f1
```

Current extraction status:

```text
jobs represented: 233
finished task rows after deduplication: 698
matched base-vs-X-MOE task rows: 347
matched seeds available: seeds 32, 42, 52, 62, 72, but seed 32 and one seed-52 row are incomplete
```

The current Week27 CSV covers all five seeds, but the grid is not perfectly complete: seeds 42, 62, and 72 have all 144 expected task rows; seed 32 has 123 rows; seed 52 has 143 rows. The missing seed-32 rows are poly/permod configs from the cancelled early submission, and seed 52 is missing one poly/X-MOE/shared/permod/tstxtcxr PHENO row.

### 18.4 Current Base vs X-MOE Result

Across the currently matched rows:

```text
X-MOE wins: 174
base wins: 172
ties: 1
```

Mean X-MOE minus base primary metric by task:

| Task | Matched rows | Mean delta | X-MOE win rate | Interpretation |
|---|---:|---:|---:|---|
| IHM | 116 | -0.0042 | 49.0% | X-MOE is slightly worse. |
| LOS | 116 | +0.0008 | 51.0% | Essentially tied. |
| PHENO | 115 | +0.0004 | 50.0% | Essentially tied. |

Mean X-MOE minus base primary metric by gate:

| Gate | Matched rows | Mean delta | X-MOE win rate | Interpretation |
|---|---:|---:|---:|---|
| softmax | 120 | -0.0016 | 47.0% | Slightly negative. |
| laplace | 120 | -0.0001 | 50.0% | Tied. |
| poly | 107 | -0.0014 | 53.0% | Mixed; incomplete because of missing poly/permod rows. |

Mean X-MOE minus base primary metric by routing mode:

| Routing mode | Matched rows | Mean delta | X-MOE win rate | Interpretation |
|---|---:|---:|---:|---|
| joint | 180 | -0.0009 | 51.0% | Essentially tied. |
| permod | 167 | -0.0012 | 50.0% | Essentially tied; incomplete because of missing seed-32 poly/permod rows. |

Largest current positive X-MOE deltas:

| Seed | Task | Gate | Shared | Router | Modalities | Delta | Base | X-MOE |
|---:|---|---|---:|---|---|---:|---:|---:|
| 42 | IHM | laplace | 1 | joint | tstxtcxrecg | +0.0588 | 0.3949 | 0.4537 |
| 42 | IHM | softmax | 1 | joint | tstxtcxrecg | +0.0561 | 0.4151 | 0.4712 |
| 32 | PHENO | softmax | 0 | joint | tstxtcxr | +0.0545 | 0.3285 | 0.3830 |
| 32 | IHM | laplace | 0 | joint | tstxtcxr | +0.0545 | 0.4286 | 0.4831 |
| 42 | IHM | laplace | 1 | joint | tstxtcxr | +0.0521 | 0.4753 | 0.5275 |

Largest current negative X-MOE deltas:

| Seed | Task | Gate | Shared | Router | Modalities | Delta | Base | X-MOE |
|---:|---|---|---:|---|---|---:|---:|---:|
| 42 | IHM | poly | 0 | joint | tstxtcxr | -0.1029 | 0.5176 | 0.4147 |
| 32 | PHENO | softmax | 0 | joint | tstxtcxrecg | -0.0496 | 0.4083 | 0.3587 |
| 32 | IHM | softmax | 0 | permod | tstxtcxrecg | -0.0403 | 0.4815 | 0.4412 |
| 32 | IHM | poly | 0 | joint | tstxtcxr | -0.0394 | 0.4729 | 0.4335 |
| 42 | IHM | softmax | 0 | joint | tstxtcxrecg | -0.0393 | 0.5000 | 0.4607 |

### 18.5 Current Interpretation

The current evidence does not support the strong claim:

```text
X-MOE consistently improves performance.
```

The current evidence supports a weaker and more accurate claim:

```text
X-MOE changes the optimization behavior, but the Week27 matched performance grid is essentially tied overall. It does not currently support a claim that X-MOE improves task performance.
```

This matters because routing stability and task performance are related but not identical:

- X-MOE improves router conditioning.
- X-MOE can reduce extreme collapse.
- But performance can still drop if the low-dimensional routing space selects less useful experts for a task/modality setting.

The most useful next comparison is not a larger blind grid. The next efficient check is:

```text
If performance is the goal, do not expand X-MOE blindly. First isolate why seed 52 hurts X-MOE and whether the missing seed-32 poly/permod configs materially affect the permod/poly slice.
```

### 18.6 Current Practical Recommendation

Do not claim final X-MOE superiority yet.

For the next reporting update, say:

```text
I added a matched Week27 grid to compare base FuseMoE vs X-MOE under the same seed,
gate, router type, shared-expert setting, modality set, and task. After parsing the completed Week27 logs, X-MOE is essentially tied with the base router overall: 174 wins vs 172 base wins, with near-zero mean deltas for LOS/PHENO and a small negative delta for IHM. Current result does not justify claiming a performance boost from X-MOE, though it may still be useful as a routing-stability mechanism.
```

Recommended action before launching more jobs:

1. Finish enough seed coverage for the matched Week27 subset.
2. Aggregate by config across seeds, not by single job.
3. Prioritize the configs where X-MOE showed strong positive/negative movement at seed 32.
4. Avoid submitting the full Cartesian grid again unless queue conditions are favorable.

## 19. Week28 Router Initialization And Mentor Specialization

Week28 shifted the main question from raw task performance to whether router/expert diagnostics show useful specialization:

```text
Does the router assign different clinical cohorts to different experts, or is the apparent expert balance just uniform routing without meaningful specialization?
```

The Week28 logs now include richer diagnostics:

- router mass and selection summaries
- gate entropy, top-1 weight, top-1/top-2 margin
- expert output off-diagonal similarity
- expert cohort statistics, including label rate per routed cohort

The important diagnostic terms are:

- `validation_mass_mean_maxdev`: average absolute deviation of expert mass from uniform `0.25`.
- `val_gate_entropy`: gate entropy; lower means more decisive routing.
- `val_gate_top1_weight`: mean top-1 gate probability; higher means more decisive routing.
- `validation_expert_offdiag_mean`: mean expert output similarity; lower means experts produce more different outputs.
- `val_cohort_label_range`: max expert label rate minus min expert label rate; higher means routed expert cohorts differ more in outcome composition.

CSV outputs were moved into the corresponding Week28 subfolders:

- `out/Week_28/summary/week28_all_results_summary.csv`
- `out/Week_28/summary/week28_aggregate_by_variant.csv`
- `out/Week_28/router_init_diagnostics/week28_router_init_diagnostics_with_specialization.csv`
- `out/Week_28/router_init_diagnostics/week28_router_init_random_noise_comparison.csv`
- `out/Week_28/router_init_diagnostics/week28_permod_router_expert_diagnostics.csv`
- `out/Week_28/router_init_diagnostics/week28_permod_router_expert_diagnostics_aggregate.csv`
- `out/Week_28/mentor_specialization/week28_mentor_specialization_diagnostics.csv`
- `out/Week_28/mentor_specialization/week28_mentor_specialization_diagnostics_aggregate.csv`

Coverage:

```text
manifest rows: 102
completed logs: 101
missing output: 10856471_w28_ihm_s32_orig
```

### 19.1 Week28 Combined Result Summary

The combined Week28 parse covers:

- per-modality router initialization diagnostics
- joint-router random controls
- core mentor specialization
- extended mentor specialization

The current high-level conclusion is:

```text
Uniform routing is not sufficient evidence of useful expert specialization. The strongest diagnostic signal is cohort separation, not perfect load balance.
```

The joint-router controls are unstable and should not be used as the main evidence path. They often show full or near-full router collapse:

```text
validation_mass_mean_maxdev: about 0.333 to 0.375
val_gate_entropy: about 0
val_gate_top1_weight: about 1.0
```

Because of that, Week28 interpretation should concentrate on the per-modality router.

### 19.2 Permod Router Init: Original Zero Init vs Random Init + Noise

The relevant permod CSVs are:

- `out/Week_28/router_init_diagnostics/week28_router_init_diagnostics_with_specialization.csv`
- `out/Week_28/router_init_diagnostics/week28_router_init_random_noise_comparison.csv`
- `out/Week_28/router_init_diagnostics/week28_permod_router_expert_diagnostics.csv`
- `out/Week_28/router_init_diagnostics/week28_permod_router_expert_diagnostics_aggregate.csv`

Ignoring F1/AUC and looking only at router/expert diagnostics:

```text
Permod random init + noise gives better specialization diagnostics than original zero init, especially for IHM and LOS.
```

Original zero init is extremely uniform:

| Task | Variant | Mass dev | Gate entropy | Top-1 weight | Expert offdiag | Label range |
|---|---|---:|---:|---:|---:|---:|
| IHM | original | 0.0072 | 0.6463 | 0.6203 | 0.1964 | 0.0350 |
| LOS | original | 0.0066 | 0.6463 | 0.6200 | 0.3676 | 0.0595 |
| PHENO | original | 0.0060 | 0.6461 | 0.6200 | 0.4496 | 0.0130 |

Random router improves the specialization diagnostics:

| Task | Variant | Mass dev | Gate entropy | Top-1 weight | Expert offdiag | Label range |
|---|---|---:|---:|---:|---:|---:|
| IHM | random_router | 0.1806 | 0.6239 | 0.6401 | 0.0720 | 0.2480 |
| LOS | random_router | 0.1809 | 0.6273 | 0.6308 | 0.1284 | 0.4730 |
| PHENO | random_router | 0.1189 | 0.6424 | 0.6167 | 0.2279 | 0.0185 |

Random low-noise decay also improves the specialization diagnostics:

| Task | Variant | Mass dev | Gate entropy | Top-1 weight | Expert offdiag | Label range |
|---|---|---:|---:|---:|---:|---:|
| IHM | random_low_noise_decay | 0.1696 | 0.6042 | 0.6396 | 0.1133 | 0.1820 |
| LOS | random_low_noise_decay | 0.1913 | 0.5980 | 0.6540 | 0.1556 | 0.2205 |
| PHENO | random_low_noise_decay | 0.1101 | 0.6459 | 0.6111 | 0.3039 | 0.0545 |

Interpretation:

- IHM: random init/noise clearly improves router/expert specialization diagnostics.
- LOS: random init/noise clearly improves expert output diversity and cohort separation.
- PHENO: random init/noise improves expert output diversity; cohort separation improves most with `random_low_noise_decay`, but the signal is weaker than IHM/LOS.

Important caveat:

```text
IHM original seed 32 is missing, so IHM original baseline is less complete than LOS/PHENO.
```

### 19.3 Uniform Routing Interpretation

Uniform routing is useful for preventing dead experts, but it is not sufficient for specialization.

The original zero-init permod router remains very uniform across tasks:

```text
massdev around 0.006
entropy around 0.646
top1 around 0.620
```

However, its label-range diagnostic is tiny:

```text
IHM:   0.035
LOS:   0.0595
PHENO: 0.013
```

That means the experts are used evenly, but they are not clearly receiving different outcome-defined cohorts.

The better target is not maximally uniform or maximally collapsed routing. The better target is:

```text
balanced enough that experts remain alive, but non-uniform enough that routed cohorts and expert outputs differ meaningfully.
```

### 19.4 Joint Router Controls Are Not The Main Evidence

The joint-router results should be de-emphasized.

They show strong collapse:

| Family | Variant | Mass dev | Entropy | Top-1 weight |
|---|---|---:|---:|---:|
| joint_router_init | original | 0.333 to 0.375 | about 0.005 or 0 | about 0.998 to 1.0 |
| joint_router_init | random/noise | 0.375 | 0.0 | 1.0 |

This is deterministic routing, not healthy specialization. Even if some task metrics improve, this is not clean evidence that the architecture learned meaningful expert specialization.

Recommendation:

```text
Ditch joint-router controls as the main research path. Concentrate on per-modality routing.
```

### 19.5 Mentor Specialization Diagnostics

The relevant mentor CSVs are:

- `out/Week_28/mentor_specialization/week28_mentor_specialization_diagnostics.csv`
- `out/Week_28/mentor_specialization/week28_mentor_specialization_diagnostics_aggregate.csv`

The central mentor result is:

```text
Weak semantic mentor pressure can help specialization, but strong semantic forcing or stacked regularization often hurts.
```

For IHM, `sem01` is the cleanest mentor setting:

| Variant | Mass dev | Entropy | Top-1 weight | Expert offdiag | Label range |
|---|---:|---:|---:|---:|---:|
| base | 0.1696 | 0.6042 | 0.6396 | 0.1133 | 0.1820 |
| sem01 | 0.1882 | 0.5630 | 0.6705 | 0.1523 | 0.2665 |
| sem03 | 0.1643 | 0.5996 | 0.6425 | 0.1016 | 0.1495 |
| sem01sh | 0.1561 | 0.5990 | 0.6372 | 0.1216 | 0.2385 |
| sem01reg | 0.1531 | 0.5813 | 0.6570 | 0.1314 | 0.1600 |
| semonly | 0.2103 | 0.6875 | 0.5439 | 0.1786 | 0.0555 |

IHM interpretation:

- `sem01` increases cohort separation from `0.1820` to `0.2665`.
- `sem01` also makes the gate more decisive: entropy drops and top-1 weight rises.
- Stronger semantic scale (`sem03`), semantic-only routing, or stacked regularizers are worse specialization choices.

For LOS, mentor is less convincing because base already has cohort separation:

| Variant | Mass dev | Entropy | Top-1 weight | Expert offdiag | Label range |
|---|---:|---:|---:|---:|---:|
| base | 0.1913 | 0.5980 | 0.6540 | 0.1556 | 0.2205 |
| sem01 | 0.1870 | 0.6234 | 0.6315 | 0.0925 | 0.2195 |
| sem03 | 0.1566 | 0.6302 | 0.6289 | 0.0696 | 0.2130 |
| sem01sh | 0.1173 | 0.6677 | 0.5728 | 0.1521 | 0.1680 |
| sem01reg | 0.1238 | 0.6702 | 0.5689 | 0.1745 | 0.1760 |
| semonly | 0.2144 | 0.6907 | 0.5303 | 0.1858 | 0.2970 |

LOS interpretation:

- `sem01` improves expert output diversity but does not improve cohort label separation.
- shared/regularized variants smooth the router and reduce label separation.
- `semonly` increases label range but has high entropy and low top-1 weight, so it is not the same decisive routing pattern.

For PHENO, only the extended mentor grid applies:

| Variant | Mass dev | Entropy | Top-1 weight | Expert offdiag | Label range |
|---|---:|---:|---:|---:|---:|
| base | 0.1101 | 0.6459 | 0.6111 | 0.3039 | 0.0545 |
| sem01 | 0.1041 | 0.6511 | 0.6066 | 0.2860 | 0.1475 |
| sem01sh | 0.1097 | 0.6437 | 0.6163 | 0.1733 | 0.1135 |
| sem01reg | 0.1017 | 0.6478 | 0.6120 | 0.1440 | 0.0700 |
| sem03 | 0.0972 | 0.6457 | 0.6119 | 0.2992 | 0.0240 |
| semonly | 0.1990 | 0.6920 | 0.5187 | 0.1744 | 0.0590 |

PHENO interpretation:

- `sem01` improves cohort separation the most.
- `sem01sh` and `sem01reg` improve expert output diversity.
- `sem03` worsens cohort separation.
- `semonly` is not a good main setting.

### 19.6 Mentor Specialization Recommendation

Use mentor as a weak nudge, not as the full router.

Current best candidates:

- IHM: `sem01`
- LOS: no strong mentor winner; `sem01` is acceptable but not clearly better than base on cohort separation
- PHENO: `sem01`, `sem01sh`, and `sem01reg` are worth following up

Avoid as a main path:

- `sem03`
- `semonly`
- stacked regularizers unless task-specific evidence supports them

### 19.7 Current Research Interpretation After Week28

The strongest current story is:

```text
FuseMoE's original zero-init permod router is highly balanced but too uniform to show strong expert specialization. Random init/noise and weak semantic mentor pressure can produce more differentiated routed cohorts and lower expert-output similarity, especially for IHM and LOS. Useful specialization should be measured by cohort separation and expert-output diversity, not by uniform expert usage alone.
```

Do not claim:

```text
Uniform routing proves expert specialization.
```

Do claim:

```text
Balanced routing prevents dead experts, but meaningful specialization requires additional evidence: different routed cohorts, different expert outputs, and stable non-collapsed gate behavior.
```
