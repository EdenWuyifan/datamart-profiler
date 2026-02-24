#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 128GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=torch_pr_66_tandon_advanced
#SBATCH --comment="preemption=yes;requeue=true"

set -eo pipefail

module purge
module load anaconda3/2025.06

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(pwd)"
fi
ENV_PREFIX="${SCRIPT_DIR}/penv"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "=== GPU CHECK ==="
nvidia-smi || true

python training/train_cta_classifier_hierarchical.py \
  --synthetic_path training/synthetic_df.csv \
  --output_dir ./model_hierarchical \
  --epochs 20 \
  --weak_positive_weight 0.3 \
  --lambda_contrastive 1.0 --lambda_family 1.0 --lambda_subtype 1.0



