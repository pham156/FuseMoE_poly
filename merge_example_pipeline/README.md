# Building Massively Multimodal Foundation Models with Dependency-Aware Experts

## Requirements
You can install them using:
```
pip install -r requirements.txt
```

## Datasets
Here we use the PAMAP2 dataset as an example. You can download the dataset from [here](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).

## Usage

Please run experiments using the following commands to obtain results for PAMAP2 dataset.
```
sh run_pamap_data.sh
```

The implementation is divided into two parts, (1) compute temporal RUS values, and (2) leverage the obtained RUS values to guide the training of multimodal MoE:
- `pamap_rus_multimodal.py`: This file is used to analyze the PAMAP2 dataset using temporal RUS.
- `train_pamap_multimodal.py`: This file is used to train the multimodal MoE model on the PAMAP2 dataset and temporal RUS values.

Computing temporal RUS values can take sometime, especially for high-dimentional dataset. But for each dataset, it only needs to be computed once: the obtained RUS values are stored in the `results` folder and can be reused later. You can comment out the `python pamap_rus_multimodal.py` command in the `run_pamap_data.sh` script if the desired RUS values have already been computed.