# Reproduce softmax sample-efficiency runs

This folder reproduces the softmax gating baseline over data percentages.

Launcher:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/reproduce_softmax4_sample_efficiency/run_softmax4_sample_efficiency.sh
```

Job structure:

- One submitted job = one task + one router/config + one seed.
- Each job loops over data ratios: `0.025 0.05 0.1 0.2 0.4 1.0`.
- Selected seeds: `30 102 20 1 50 22 123 42 0 32`.
- Seed 112 was not used because it had incomplete full-data rows in the Week 19 summary.

Requested shared-pool configs:

- `joint`: softmax, 4 experts, top-k 2, shared experts 0.
- `joint`: softmax, 3 experts, top-k 2, shared experts 0.
- `permod`: softmax, 4 experts, top-k 2, shared experts 0.
- `permod`: softmax, 3 experts, top-k 2, shared experts 0.

Disjoint configs:

- `disjoint`: softmax, 3 total experts, top-k 1 per modality, shared experts 0.
- `disjoint`: softmax, 6 total experts, top-k 2 per modality, shared experts 0.

Note on disjoint:

The current `disjoint` implementation splits experts by modality using `num_experts // num_modalities`. With 3 modalities, `experts=4, top_k=2` is not valid because each modality only gets `4 // 3 = 1` expert. Similarly, `experts=3, top_k=2` is not valid because each modality gets exactly 1 expert. Therefore this launcher uses clean disjoint baselines: `3/1` and `6/2`.
