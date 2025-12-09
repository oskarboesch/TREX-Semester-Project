#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

cd ~/TREX-Semester-Project/src

# Load modules or activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Check if environment exists
if ! conda env list | grep -q "^trex_env"; then
    echo "⚙️  Environment 'trex_env' not found — creating it from environment.yml..."
    conda env create -f environment.yaml -n trex_env
fi

# Activate environment
conda activate trex_env

# Run the script
python -m data.preprocess_data
