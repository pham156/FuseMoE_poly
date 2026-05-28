# Instruction Router Sample Efficiency Pilot

This folder runs a small pilot for the instruction-guided router.

Grid:

- Seeds: `30 20 1`
- Tasks: `ihm-48-cxr-notes-ecg`, `los-48-cxr-notes-ecg`, `pheno-all-cxr-notes-ecg`
- Ratios inside each job: `0.025 0.05 0.1 0.2 0.4 0.6 0.8 1.0`
- Configs:
  - `joint`, 4 experts, top-k 2
  - `joint`, 3 experts, top-k 2
  - `permod`, 4 experts, top-k 2
  - `permod`, 3 experts, top-k 2
  - `disjoint`, 6 experts, disjoint top-k 2

Submit:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/instruction_router_sample_efficiency/run_instruction_router_sample_efficiency.sh
```

Logs:

```text
/home/pham156/MoE/FuseMoE_poly/out/Week_23/instr_router_<jobid>.out
/home/pham156/MoE/FuseMoE_poly/out/Week_23/instr_router_<jobid>.err
```

Run folders:

```text
/home/pham156/MoE/FuseMoE_poly/run_folder/week23/instruction_router_sample_efficiency/
```
