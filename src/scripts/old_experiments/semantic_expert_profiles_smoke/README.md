# Semantic Expert Profile Smoke Test

This folder runs a minimal smoke test for the opt-in semantic expert profile router.

The smoke uses:

```text
task: ihm-48-cxr-notes-ecg
ratio: 0.025
epochs: 1
router: joint
experts: 4
top_k: 2
instruction_router_fusion: input_add
semantic_profile_set: icu_organ_system
semantic_profile_fusion: add
semantic_profile_scale: 1.0
```

By default, semantic profile routing uses the patient/modality representation
already entering the MoE:

```bash
--semantic_profile_source patient
```

For the mentor-style note-level semantic routing path, use:

```bash
--use_semantic_expert_profiles \
--semantic_profile_source note \
--semantic_profile_note_pooling max \
--semantic_profile_modalities txt
```

This routes from note-level text embeddings to the clinical expert profiles,
then pools note/profile similarities into router logits. For `permod` and
`disjoint`, the note-derived semantic logits are applied only to the listed
modalities, defaulting to `txt`.

Submit:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/semantic_expert_profiles_smoke/run_smoke_semantic_expert_profiles.sh
```

Logs:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_24/sem_profile_<jobid>.out
/home/pham156/MoE/FuseMoE_poly/out/Week_24/sem_profile_<jobid>.err
```

Run folder:

```text
/home/pham156/MoE/FuseMoE_poly/run_folder/week24/semantic_expert_profiles_smoke/
```

## Week 25 Semantic Profile Comparison

This launcher compares semantic expert profile routing without instruction-router
injection.

Scope:

```text
tasks:
  ihm-48-cxr-notes-ecg
  los-48-cxr-notes-ecg
  pheno-all-cxr-notes-ecg
seeds: 30 20 1 50 42
ratios: 0.025 0.05 0.1 0.2 0.4 0.6 0.8 1.0
router configs:
  joint, 4 experts, top-k 2
  permod, 4 experts, top-k 2
  disjoint, 12 total experts, local top-k 2
```

The seed set is a 5-seed subset from the earlier sample-efficiency runs, chosen
to stay comparable with prior experiments while keeping this first
semantic-profile grid smaller than the full 10-seed reproduction.

The disjoint config uses 12 total experts because disjoint routing splits experts
across 3 modalities, so each modality gets 4 local experts matching the 4
semantic profiles.

Submit:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/semantic_expert_profiles_smoke/run_semantic_profiles_week25.sh
```

Logs:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_25/sem_prof25_<jobid>.out
/home/pham156/MoE/FuseMoE_poly/out/Week_25/sem_prof25_<jobid>.err
```

Run folders:

```text
/home/pham156/MoE/FuseMoE_poly/run_folder/week25/semantic_expert_profiles/
```
