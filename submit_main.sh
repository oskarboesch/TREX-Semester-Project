#!/bin/bash
#SBATCH --job-name=fit_cv
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main%j.out
#SBATCH --error=logs/main%j.err

# ============================
#    Environment Setup
# ============================
cd ~/TREX-Semester-Project

source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^trex_env"; then
    echo "⚙️  Environment 'trex_env' not found — creating it..."
    conda env create -f environment.yaml -n trex_env
fi

conda activate trex_env

# ============================
#    Run training
# ============================
python src/main.py