# Lingshu Pseudo-Token LoRA Expert Prototype

This is an isolated prototype for the mentor's frozen medical backbone + LoRA
expert direction.

It does not replace the current FuseMoE MoE path. It uses:

```text
preprocessed ICU features
  TS reg_ts tokens
  note embeddings
  CXR feature embeddings
  optional ECG embeddings
-> modality projectors
-> Lingshu hidden-size pseudo-token embeddings
-> frozen Lingshu backbone with PEFT LoRA
-> pooled hidden state
-> task classifier
```

Missing modalities are represented with learned modality-specific missing
embeddings in the pseudo-token stream. This is a small, local version of the
mentor's missing-domain idea; it is not the full missing-modality prediction
loss yet.

Entry point:

```text
/home/pham156/MoE/FuseMoE_poly/src/scripts/main_mimiciv_lingshu.py
```

Model:

```text
/home/pham156/MoE/FuseMoE_poly/src/core/lingshu_pseudotoken.py
```

Submit smoke:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/lingshu_lora_experts/run_lingshu_pseudotoken_smoke.sh
```

If the model is not in the default scratch cache path, set:

```bash
LINGSHU_MODEL_PATH=/path/to/Lingshu-7B bash run_lingshu_pseudotoken_smoke.sh
```

This requires `peft` when `--lingshu_use_peft_lora True`.

Preflight check:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/lingshu_lora_experts/check_lingshu_environment.sh
```

Week26 diagnostic launcher:

```bash
bash /home/pham156/MoE/FuseMoE_poly/src/scripts/lingshu_lora_experts/run_lingshu_pseudotoken_week26.sh
```

The Week26 launcher runs the same IHM ratios/seeds used in the semantic-router
diagnostic, but through the separate Lingshu pseudo-token path.
