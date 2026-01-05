# Data-Driven Modeling for Robotic Mechanical Thrombectomy

**TREX Semester Project - Deep Learning for Clot Detection in Endovascular Procedures**

## Project Overview

This project develops a deep learning-based approach for real-time clot detection during robotic mechanical thrombectomy procedures. Using a Gated Recurrent Unit (GRU) neural network, the system analyzes force sensor signals and optional imaging data to automatically detect when a guidewire enters and exits a blood clot during endovascular interventions.

### Key Features
- **Real-time clot boundary detection** using time-series force sensor data
- **Multi-modal learning** incorporating force signals, frequency-domain features, and image masks
- **Robust performance** across different experimental conditions (wire types, clot stiffness, anatomical models)
- **Temporal modeling** using GRU networks for sequential decision-making

## Project Architecture

```
TREX-Semester-Project/
│
├── configs/                      # Configuration files
│   ├── data/                     # Data loading configurations
│   ├── fit/                      # Training hyperparameters
│   └── model/                    # Model architecture configs
│
├── data/                         # Data directory
│   ├── raw/                      # Raw experimental data
│   │   ├── Paper_Experiment/    # Original paper dataset
│   │   └── Extra_Data/          # Additional experiments
│   └── processed/               # Preprocessed tensors
│
├── src/                          # Source code
│   ├── models/
│   │   ├── gru.py               # GRU classifier implementation
│   │   └── helpers.py           # Model utilities
│   ├── data/
│   │   ├── load_data.py         # Data loading and filtering
│   │   ├── preprocess_data.py   # Signal preprocessing
│   │   ├── sensor_dataset.py    # PyTorch dataset class
│   │   └── paths.py             # Path configurations
│   ├── experiments/
│   │   ├── paper_experiment.py  # Original paper data experiments
│   │   ├── full_experiment.py   # Complete dataset experiments
│   │   ├── bent_experiment.py   # Bent wire experiments
│   │   ├── shared.py   # Shared code between all experiments
│   │   └── twist_experiment.py  # Twist technique experiments
│   ├── utils/                    # Utility functions
│   ├── main.py                   # Main entry point
│   ├── fit.py                    # Model training script
│   ├── evaluate.py               # Model evaluation
│   ├── cross_validate.py         # Cross-validation
│   └── inference_time.py         # Performance benchmarking
│
├── models/                       # Saved model checkpoints
│   └── gru_model_*.pt           # Trained GRU models
│
├── results/                      # Experimental results
│   ├── gru_results/             # Training results & plots
│   └── evaluation_results/      # Test set evaluations
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_analysis.ipynb      # Data exploration
│   ├── gru_visualization.ipynb  # Model analysis
│   └── image_analysis.ipynb     # Image feature analysis
│
├── figures/                      # Generated figures
├── logs/                         # SLURM job logs
├── wandb/                        # Weights & Biases tracking
└── submit_*.sh                   # SLURM submission scripts
```

## Model Architecture

<img width="1030" height="363" alt="model drawio" src="https://github.com/user-attachments/assets/100fb0e7-8dd6-4303-94a9-9e793f529167" />

### GRU Classifier

The core model is a multi-layer GRU network with the following architecture:

```python
GRUClassifier(
    input_size=1,        # Force signal + frequency features + optional images
    hidden_size=8,       # Hidden state dimension
    num_layers=1,        # GRU layers
    output_size=1,       # Binary classification (in/out of clot)
    dropout=0.2         # Dropout for regularization
)
```

**Input Features:**
- `signal`: Raw force sensor voltage (1D)
- `bandpower`: Frequency domain features via wavelet transform (optional)
- `images`: Encoded guidewire mask images (optional, via CNN encoder)

**Output:**
- Continuous probability ∈ [0, 1] indicating likelihood of being inside a clot
- Threshold at 0.9 for binary classification

## Data Pipeline

### 1. Data Collection
- Force sensor sampled at **1 kHz**
- Image captured at **10 Hz**
- Multiple experimental conditions:
  - Wire types: Straight vs. Bent
  - Techniques: Twist vs. No-twist
  - Clot stiffness: Soft, Medium, Hard
  - Anatomical models: Conical vs. Realistic anatomy

### 2. Preprocessing
```python
# Signal preprocessing steps:
1. Downsampling from 1000 Hz → 100 Hz (downsampling_freq=10)
2. Z-score normalization: (x - μ) / σ
3. Bandpower computation (optional): Wavelet coefficients
4. Image mask encoding (optional): Exponential Moving Average
5. Label generation from ground truth markers
```

### 3. Training
```bash
# Train on full dataset
python src/main.py

# Or submit to SLURM cluster
sbatch submit_fit.sh
```


## 📊 Example Prediction

Below is an example from the test set showing the model's prediction compared to ground truth:

<img width="1200" height="600" alt="base_eval_plot_10" src="https://github.com/user-attachments/assets/a981e5d9-3177-4cfe-a16d-f61e92f9344e" />


The model accurately detects both the **entry point** (start) and **exit point** (end) of the clot interaction.

## Main Results - Configuration C1 for the Base Experiment

### Configuration C1: Force Signal Only 

**Model Configuration:**
```yaml
features: [signal]
hidden_size: 16
num_layers: 2
dropout: 0.2
downsampling_freq: 10
with_kde_weighting: true
```

### Performance Metrics

| Metric | Test Set Performance |
|--------|---------------------|
| **F1 Score** | 92.44% |
| **Start Detection Accuracy (±1.5 mm)** | 76.19% |
| **End Detection Accuracy (±1.5 mm)** | 61.9% |

### Key Findings

1. **Robustness**: The model generalizes across:
   - Different clot stiffnesses (soft, medium, hard)
   - Wire configurations (straight, bent)
   - Manipulation techniques (twist, no-twist)
   - Anatomical models (conical, realistic)

2. **Real-time capability**: Inference time of **~0.5319 ms per curve** enables real-time deployment on standard hardware.

## Quick Start

### Prerequisites
```bash
# Create conda environment
conda env create -f environment.yaml
conda activate trex-project
```

### Training a New Model
```bash
# Run full experiment pipeline
python src/main.py

# Or customize with config files
python src/fit.py \
    --data-config configs/data/base_both.yml \
    --fit-config configs/fit/base_fit.yml \
    --model-config configs/model/base_model_dropout.yml
```

### Evaluating a Trained Model
```bash
python src/evaluate.py \
    --model-path models/gru_model_2025-12-20_08-37-17.pt \
    --data-config configs/data/base_both.yml
```

### Running on SLURM Cluster
```bash
# Preprocess data
sbatch submit_preprocess.sh

# Train model
sbatch submit_fit.sh

# Cross-validation
sbatch submit_cross_validate.sh
```

## Monitoring Training

The project uses **Weights & Biases** for experiment tracking:
- Training/validation loss curves
- Accuracy, precision, recall metrics
- Sample weight distributions
- Hyperparameter configs

Access runs in the `wandb/` directory or view online dashboard.

## Report
Please find the complete report of the project under :
[Report.pdf](https://github.com/user-attachments/files/24436760/Data_Driven_Modeling_for_Robotic_Mechanical_Thrombectomy.1.pdf)

