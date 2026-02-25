#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J run_mimiciv_poly
#SBATCH -o ../out/Week_3/%x_%j.out
#SBATCH -e ../out/Week_3/%x_%j.err


module load anaconda

# activate path-based env
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe

nvidia-smi

export WANDB_MODE=offline

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
bash run_mimiciv_poly.sh
