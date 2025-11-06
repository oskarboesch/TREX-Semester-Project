#!/bin/bash
#SBATCH --job-name=fit      # Name of the job
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Adjust CPU allocation if needed
#SBATCH --output=logs/fit.out    # Output log file
#SBATCH --error=logs/fit.err     # Error log file

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

# Run the fitting script
python src/fit.py 