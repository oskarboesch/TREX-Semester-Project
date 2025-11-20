#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fit_gru_%A_%a.out
#SBATCH --error=logs/fit_gru_%A_%a.err

cd ~/TREX-Semester-Project

# Load modules or activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Check if environment exists
if ! conda env list | grep -q "^trex_env"; then
    echo "⚙️  Environment 'trex_env' not found — creating it from environment.yml..."
    conda env create -f environment.yaml -n trex_env
fi

# Activate environment
conda activate trex_env

# === Run all config combinations ===
DATA_DIR=configs/data
FIT_DIR=configs/fit
MODEL_DIR=configs/model

for data_cfg in "$DATA_DIR"/*.yml; do
  for fit_cfg in "$FIT_DIR"/*.yml; do
    for model_cfg in "$MODEL_DIR"/*.yml; do
      echo "Running combination:"
      echo "   data:  $data_cfg"
      echo "   fit:   $fit_cfg"
      echo "   model: $model_cfg"
      echo ""

      python src/fit_gru.py \
        --data_config "$data_cfg" \
        --fit_config "$fit_cfg" \
        --model_config "$model_cfg"
    done
  done
done