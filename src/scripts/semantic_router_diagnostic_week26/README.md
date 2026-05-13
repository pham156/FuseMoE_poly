# Week26 Semantic Router Diagnostic

This folder runs a small diagnostic sweep for the semantic organ-profile router.

The goal is to test whether the Week25 result failed because the semantic
profile signal was too weak, too strong, or attached poorly to the current
patient-level MoE input.

## Scope

```text
task: ihm-48-cxr-notes-ecg
router: joint
experts: 4
top_k: 2
gating: softmax
seeds: 1 20 30
ratios: 0.05 0.1 0.4 0.8
semantic_profile_source: patient
```

Semantic settings:

```text
fusion=add, scale=0.1
fusion=add, scale=0.3
fusion=add, scale=1.0
fusion=add, scale=3.0
fusion=replace, scale=1.0
```

This keeps the original baseline runnable because semantic routing is enabled
only through `--use_semantic_expert_profiles`.

## Submit

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/semantic_router_diagnostic_week26/run_semantic_router_diagnostic_week26.sh
```

## Logs

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_26/sem_diag26_<jobid>.out
/home/pham156/MoE/FuseMoE_poly/out/Week_26/sem_diag26_<jobid>.err
```

## Run Folders

```text
/home/pham156/MoE/FuseMoE_poly/run_folder/week26/semantic_router_diagnostic/
```

## Interpretation

Compare against the matched Week22 baseline:

```text
/home/pham156/MoE/FuseMoE_poly/out/csv_exports/Week_22/softmax4_sample_efficiency_summary_completed.csv
```

If low scale improves results, the Week25 semantic bias was probably too
aggressive. If high scale improves results, the profile signal was probably too
weak. If all settings fail, stop expanding this router and move to the
Lingshu-LoRA prototype.
