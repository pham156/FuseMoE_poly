# Week27 Missing-Modality Mentor Smoke

Goal: test whether the mentor-inspired missing-modality changes help on the actual `missingInd` data without mixing in X-MOE or semantic routing.

This is intentionally a small evidence run.

## Fixed Setup

- tasks: `ihm-48-cxr-notes-ecg-missingInd`, `los-48-cxr-notes-ecg-missingInd`
- modeltype: `TS_CXR_Text_ECG`
- modalities: TS + text + CXR + ECG
- router: `permod`
- gating: `softmax`
- experts: 4
- top-k: 2
- epochs: 8
- seed: 32
- balance loss: 0.01
- router entropy: 0.01
- X-MOE: off
- semantic routing: off

## Variants

1. `base_zero`: original missing modality behavior, zero-filled missing modality.
2. `miss_embed`: learned per-modality missing embeddings.
3. `miss_embed_recon`: learned missing embeddings + embedding reconstruction loss.
4. `miss_embed_recon_orth`: learned missing embeddings + reconstruction + expert orthogonality.

## What To Compare

Primary:

- validation/test F1 for IHM and LOS.
- route distribution stability from concise router lines.

Interpretation:

- If `miss_embed` beats `base_zero`, learned missing embeddings are useful.
- If `miss_embed_recon` beats `miss_embed`, reconstruction is useful.
- If `miss_embed_recon_orth` beats `miss_embed_recon`, expert diversity helps.
- If none improve, do not expand this path yet.

Logs go to:

`/home/pham156/MoE/FuseMoE_poly/out/Week_27/missing_mentor`

