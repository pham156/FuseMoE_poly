# Mixture-of-Experts for Multimodal Fusion

This repository contains implementation from the paper: [FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion](https://arxiv.org/pdf/2402.03226.pdf).

## Set Up Environment

Run the following commands to create a conda environment:
```bash
conda create -n MulEHR python=3.8
source activate MulEHR
pip install -r requirements.txt
```

## Repository Structure

- `src/`: Source code
    - `preprocessing/`: Scripts for MIMIC-III and MIMIC-IV data preprocessing
    - `core/`: Core implementation for the MoE and irregularity/modality encoder module
    - `scripts/`: Scripts to run experiments in different settings
    - `utils/`: Hyper-parameters, I/O, utility functions

## Run Experiments

Under `src/scripts/`:

MIMIC-III experiments
```
sh run.sh
```

MIMIC-IV experiments
```
sh run_mimiciv.sh
```

## Load Results
First change the `filepath` in `load_result.py`, then run
```
python load_result.py
```

## Acknowledgement

Part of our implementations are based on the following papers:
- [Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling](https://arxiv.org/pdf/2210.12156.pdf), ICML'23
- [Integrated multimodal artificial intelligence framework for healthcare applications](https://arxiv.org/pdf/2202.12998.pdf), NPJ Digital Medicine

## Citation

```
@article{han2024fusemoe,
  title={FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion},
  author={Han, Xing and Nguyen, Huy and Harris, Carl and Ho, Nhat and Saria, Suchi},
  journal={arXiv preprint arXiv:2402.03226},
  year={2024}
}
```