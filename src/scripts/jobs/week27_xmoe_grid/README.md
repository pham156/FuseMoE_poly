# Week27 X-MOE Grid

Goal: compare base FuseMoE router vs X-MOE-style router across the matched grid.

Each submitted SLURM job runs one model config over all three tasks:

- `ihm-48-cxr-notes-ecg`, `num_labels=2`, primary metric `f1`
- `los-48-cxr-notes-ecg`, `num_labels=2`, primary metric `f1`
- `pheno-all-cxr-notes-ecg`, `num_labels=25`, primary metric `macro_f1`

Grid:

- seeds: default resubmit set is `42 52 62 72`; set `SEEDS="32 42 52 62 72"` for the full seed set
- gates: `softmax laplace poly`
- poly power: `4`
- X-MOE: off/on
- shared experts: `0 1`
- routers: `joint permod`
- modalities: `TS_CXR_Text`, `TS_CXR_Text_ECG`
- experts: `4`
- top-k: `2`
- epochs per task: `16`
- balance coef: `0.01`

Default submitted config jobs after seed-32 is already done: `4 * 3 * 2 * 2 * 2 * 2 = 192`.
Full submitted config jobs: `5 * 3 * 2 * 2 * 2 * 2 = 240`.

Dependency lanes are disabled by default. Set `USE_DEPENDENCIES=1 LANES=8` only if you want chained throttling.

Dry run:

```bash
cd /home/pham156/MoE/FuseMoE_poly
DRY_RUN=1 bash src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh
```

Submit:

```bash
cd /home/pham156/MoE/FuseMoE_poly
bash src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh
```

Submit full seed set:

```bash
SEEDS="32 42 52 62 72" bash src/scripts/week27_xmoe_grid/submit_week27_xmoe_grid.sh
```

Outputs:

- logs: `/home/pham156/MoE/FuseMoE_poly/out/Week_27`
- manifest: `/home/pham156/MoE/FuseMoE_poly/out/Week_27/week27_xmoe_grid_manifest.tsv`
