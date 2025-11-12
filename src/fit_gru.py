import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
from load_data import load_data, list_logs, load_labels
from preprocess_data import preprocess_logs
from models.helpers import create_model_params
from models.buckling_model import BucklingModel
from models.double_slope_model import DoubleSlopeModel
import config as config
import argparse
from models.gru import GRUClassifier, train_gru_model, evaluate_gru_model
import torch
from sensor_dataset import SensorDataset
from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser(description="Fit models to experimental data.")
argparser.add_argument("--forward", action="store_true", help="Use Buckling model (default: DoubleSlope model)")
args = argparser.parse_args()
FORWARD = args.forward
DIRECTION = "Forward" if FORWARD else "Backward"
BATCH_SIZE = 32
LOG = True
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DOWNSAMPLING_FREQ = 1  # Hz
NUM_EPOCHS = 500
THRESHOLD = 0.9
WINDOW_SIZE = 50

# Ensure Window size < 30 * downsampling frequency (the recordings are usually at least 30 seconds long)
if WINDOW_SIZE >= 30 * DOWNSAMPLING_FREQ:
    WINDOW_SIZE = 30 * DOWNSAMPLING_FREQ
    print(f"Adjusted window size to {WINDOW_SIZE} to be at least 30 times the downsampling frequency.")

config.ensure_directories()


# Load Data
log_names = list_logs(config.PAPER_EXPERIMENT_DATA_FOLDER)
log_names.drop([7, 158, 174], inplace=True, errors='ignore')
log_names.reset_index(drop=True, inplace=True)
log_names = log_names[log_names["direction"] == DIRECTION].reset_index(drop=True)
print(f"Total logs for {DIRECTION} direction: {len(log_names)}")

test_size = int(0.2 * len(log_names))

train_log_names = log_names.iloc[:-test_size].reset_index(drop=True)
test_log_names  = log_names.iloc[-test_size:].reset_index(drop=True)
print(f"Training logs: {len(train_log_names)}, Testing logs: {len(test_log_names)}")
train_dataset = SensorDataset(train_log_names, window_size=WINDOW_SIZE, mode='train', downsampling_freq=DOWNSAMPLING_FREQ)
test_dataset  = SensorDataset(test_log_names,  window_size=WINDOW_SIZE, mode='eval', downsampling_freq=DOWNSAMPLING_FREQ)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1)

# Define model
gru_model = GRUClassifier(input_size=1, hidden_size=16, num_layers=2, output_size=1)

# Train 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
gru_model.to(device)
print(f"Fitting {DIRECTION} gru model with {NUM_EPOCHS} epochs on {device}...")

# WandB Initialization
if LOG:
    wandb.init(project="gru_model_fitting", name="GRU_Fit_" + RUN_ID, group=DIRECTION)
    wandb.config.update({
        "learning_rate": 0.001,
        "epochs": NUM_EPOCHS,
        "batch_size": 32,
        "downsampling_freq": DOWNSAMPLING_FREQ,
        "threshold": THRESHOLD,
        "window_size": WINDOW_SIZE,
    })
train_gru_model(gru_model, train_loader, criterion, optimizer, NUM_EPOCHS, device, downsampling_freq=DOWNSAMPLING_FREQ, threshold=THRESHOLD, log=LOG, test_loader=test_loader)

# save model
model_save_path = config.MODELS_FOLDER / f"gru_{DIRECTION.lower()}_model_{RUN_ID}.pt"
torch.save(gru_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# evaluate
print("Evaluating model on test set...")
save_folder = config.GRU_RESULTS_FOLDER /  RUN_ID / DIRECTION.lower()
save_folder.mkdir(parents=True, exist_ok=True)
val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(gru_model, test_loader, device, criterion, DOWNSAMPLING_FREQ, save_folder=save_folder, threshold=THRESHOLD)
print(f"Test Set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1 Score: {val_f1:.4f}, Start Accuracy: {np.mean(start_accuracies):.4f}, End Accuracy: {np.mean(end_accuracies):.4f}")