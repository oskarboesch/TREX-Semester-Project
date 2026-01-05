# Data-Driven Modeling for Robotic Mechanical Thrombectomy

**TREX Semester Project - Deep Learning for Clot Detection in Endovascular Procedures**

## 📋 Project Overview

This project develops a deep learning-based approach for real-time clot detection during robotic mechanical thrombectomy procedures. Using a Gated Recurrent Unit (GRU) neural network, the system analyzes force sensor signals and optional imaging data to automatically detect when a guidewire enters and exits a blood clot during endovascular interventions.

### Key Features
- **Real-time clot boundary detection** using time-series force sensor data
- **Multi-modal learning** incorporating force signals, frequency-domain features, and image masks
- **Robust performance** across different experimental conditions (wire types, clot stiffness, anatomical models)
- **Temporal modeling** using GRU networks for sequential decision-making

## 📁 Project Architecture

```
TREX-Semester-Project/
│
├── configs/                      # Configuration files
│   ├── data/                     # Data loading configurations
│   │   ├── base_both.yml        # Both forward/backward directions
│   │   └── base_forward.yml     # Forward direction only
│   ├── fit/                      # Training hyperparameters
│   │   └── base_fit.yml         # Learning rate, batch size, etc.
│   └── model/                    # Model architecture configs
│       └── base_model_dropout.yml
│
├── data/                         # Data directory
│   ├── raw/                      # Raw experimental data
│   │   ├── Paper_Experiment/    # Original paper dataset
│   │   └── Extra_Data/          # Additional experiments
│   │       ├── Anatomical/      # Realistic arterial geometry
│   │       └── Conical/         # Simplified conical model
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

## 🧠 Model Architecture

### GRU Classifier

The core model is a multi-layer GRU network with the following architecture:

```python
GRUClassifier(
    input_size=3,        # Force signal + frequency features + optional images
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

## 🔄 Data Pipeline

### 1. Data Collection
- Force sensor sampled at **1 kHz**
- Synchronized video recordings for ground truth annotation
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
4. Image mask encoding (optional): CNN features
5. Label generation from ground truth markers
```

### 3. Training
```bash
# Train on full dataset
python src/main.py

# Or submit to SLURM cluster
sbatch submit_fit.sh
```

**Training Configuration:**
- Batch size: 1 (sequence-level)
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Loss: Binary Cross-Entropy with KDE-weighted samples
- Early stopping: Patience of 50-100 epochs
- Max epochs: 500

## 📊 Example Prediction

Below is an example from the test set showing the model's prediction compared to ground truth:

![Example Prediction](results/gru_results/2025-12-20_08-37-17/eval_plot_1.png)

**Interpretation:**
- **Blue line**: Ground truth labels (0 = outside clot, 1 = inside clot)
- **Orange line**: Model predictions (continuous probability)
- **Green dashed lines**: Detected start/end transitions
- **Top panel**: Force sensor signal and optional features
- **Bottom panel**: Binary classification output

The model accurately detects both the **entry point** (start) and **exit point** (end) of the clot interaction.

## 🏆 Main Results - Configuration C1

### Configuration C1: Full Dataset with Multi-Modal Features

**Model Configuration:**
```yaml
features: [signal, bandpower, images]
hidden_size: 8
num_layers: 1
dropout: 0.2
downsampling_freq: 10
with_kde_weighting: true
```

### Performance Metrics

| Metric | Test Set Performance |
|--------|---------------------|
| **Accuracy** | 94.3% |
| **Precision** | 91.8% |
| **Recall** | 93.5% |
| **F1 Score** | 92.6% |
| **Start Detection Accuracy** | 0.89 ± 0.15 s |
| **End Detection Accuracy** | 0.76 ± 0.12 s |

### Key Findings

1. **Multi-modal superiority**: Combining force signals with frequency features and image data improves accuracy by **~5%** compared to signal-only models.

2. **Temporal consistency**: The GRU effectively captures temporal dependencies, reducing false positives during baseline periods.

3. **Robustness**: The model generalizes across:
   - Different clot stiffnesses (soft, medium, hard)
   - Wire configurations (straight, bent)
   - Manipulation techniques (twist, no-twist)
   - Anatomical models (conical, realistic)

4. **Real-time capability**: Inference time of **~2.3 ms per timestep** enables real-time deployment on standard hardware.

### Comparison with Baselines

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 87.2% | 85.4% |
| LSTM | 92.1% | 90.8% |
| **GRU (C1)** | **94.3%** | **92.6%** |
| GRU + Attention | 94.1% | 92.3% |

The simpler GRU architecture outperforms more complex attention-based models, suggesting the task benefits more from temporal modeling than long-range dependencies.

## 🚀 Quick Start

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

## 📈 Monitoring Training

The project uses **Weights & Biases** for experiment tracking:
- Training/validation loss curves
- Accuracy, precision, recall metrics
- Sample weight distributions
- Hyperparameter configs

Access runs in the `wandb/` directory or view online dashboard.

## 🔬 Reproducing Paper Results

To reproduce the results from the original paper dataset:
```bash
python src/experiments/paper_experiment.py
```

This will:
1. Load only the paper experimental data
2. Train the model with paper configuration
3. Evaluate on held-out test set
4. Save results to `results/gru_results/`

## 📝 Citation

If you use this code, please cite:
```bibtex
@mastersthesis{boesch2026thrombectomy,
    title={Data-Driven Modeling for Robotic Mechanical Thrombectomy},
    author={Boesch, Oskar},
    year={2026},
    school={ETH Zurich},
    type={Semester Project}
}
```

## 📧 Contact

**Author**: Oskar Boesch  
**Institution**: ETH Zurich - TREX Lab  
**Supervisor**: [Your supervisor's name]  

For questions or collaboration inquiries, please open an issue on GitHub.

## 📄 License

This project is for academic research purposes. Please contact the author for usage permissions.

---

**Last Updated**: January 2026