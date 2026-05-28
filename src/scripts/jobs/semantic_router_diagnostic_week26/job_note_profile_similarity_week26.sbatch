#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -J note_sim26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_note_sim26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_note_sim26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts/semantic_router_diagnostic_week26
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26/note_profile_similarity

python -u diagnose_note_profile_similarity.py \
  --file_path /home/pham156/MoE/FuseMoE_poly/data/MIMIC-IV \
  --task ihm-48-cxr-notes-ecg \
  --model_name bioLongformer \
  --num_labels 2 \
  --max_length 1024 \
  --num_of_notes 5 \
  --notes_order Last \
  --semantic_profile_set icu_organ_system \
  --note_pooling max \
  --output_dir /home/pham156/MoE/FuseMoE_poly/out/Week_26/note_profile_similarity \
  --max_text_chars 1200
