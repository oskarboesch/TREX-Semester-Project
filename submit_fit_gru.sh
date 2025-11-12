#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --array=0-1                 # Two jobs: 0 and 1
#SBATCH --time=16:00:00
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

# Define arguments for each array job
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    ARGS=""
    echo "🚀 Running fit_gru.py without arguments"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    ARGS="--forward"
    echo "🚀 Running fit_gru.py with --forward"
fi

# Run the script
python src/fit_gru.py $ARGS
