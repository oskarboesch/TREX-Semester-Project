#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fit_%A_%a.out
#SBATCH --error=logs/fit_%A_%a.err

# ============================
#    Argument Parsing
# ============================
DATA_CFG=$1
FIT_CFG=$2
MODEL_CFG=$3

if [ -z "$DATA_CFG" ] || [ -z "$FIT_CFG" ] || [ -z "$MODEL_CFG" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch fit.sh <data_config.yml> <fit_config.yml> <model_config.yml>"
    exit 1
fi

echo "Running with:"
echo "   data:  $DATA_CFG"
echo "   fit:   $FIT_CFG"
echo "   model: $MODEL_CFG"
echo ""

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
python src/fit.py \
    --data_config "$DATA_CFG" \
    --fit_config "$FIT_CFG" \
    --model_config "$MODEL_CFG"
