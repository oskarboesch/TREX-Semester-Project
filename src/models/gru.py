from torch import nn
import torch
import wandb
from data.load_data import load_data
import numpy as np
import sys


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # x -> (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, debug=False, h=None):
        if debug:
            print(f"Input shape: {x.shape}, (Bacth size: {x.size(0)}, Sequence length: {x.size(1)}, Input size: {x.size(2)})")
        if h is not None:
            h0 = h
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if debug:
            print(f"hidden state shape: {h0.shape}, (Number of layers: {self.num_layers}, Batch size: {x.size(0)}, Hidden size: {self.hidden_size})")
        out, _ = self.gru(x, h0)
        if debug:
            print(f"GRU output shape: {out.shape}, (Batch size: {out.size(0)}, Sequence length: {out.size(1)}, Hidden size: {out.size(2)})")
        # out: tensor of shape (batch_size, 1, hidden_size)
        out = self.fc(out)[:, -1, :]  # get last time step
        if debug:
            print(f"Fully connected output shape: {out.shape}, (Batch size: {out.size(0)}, Output size: {out.size(1)})")
        # out: tensor of shape (batch_size, output_size)
        return torch.sigmoid(out), h
    

class CNN_GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_channels=16, kernel_size=3, dropout=0.2):
        super(CNN_GRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GRU layer
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected output
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, debug=False):
        # x shape: (batch, seq_len, input_size)
        if debug:
            print(f"Input shape: {x.shape}")
        x = x.permute(0, 2, 1)            # (batch, input_size, seq_len) for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)            # (batch, seq_len, features) for GRU
        x = self.dropout(x)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)                # (batch, seq_len, output_size)
        return torch.sigmoid(out)

    

def train_gru_model(model, train_loader, criterion, optimizer, num_epochs, device, downsampling_freq, window_size, threshold, log=False, test_loader=None):
    model.to(device)
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            model.train()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', file=sys.stderr)
        if log:
            wandb.log({"epoch": epoch+1, "loss": loss.item()})
            if test_loader is not None:
                val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq, window_size, threshold=threshold)
                wandb.log({
                    "epoch": epoch+1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                    "val_start_accuracy": np.mean(start_accuracies),
                    "val_end_accuracy": np.mean(end_accuracies)
                })
    return model

def evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq, window_size, save_folder=None, threshold=0.9, ):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    window_size_samples = window_size * downsampling_freq

    model.eval()
    model.to(device)
    all_targets = []
    all_outputs = []
    loss = 0.0
    start_accuracies = []
    end_accuracies = []
    
    with torch.no_grad():
        for batch_i, (full_curve, full_label) in enumerate(test_loader):
            curve_loss = 0.0
            # we recieve the full curve 
            full_curve, full_label = full_curve.to(device), full_label.to(device)
            seq_len = full_curve.size(1)
            # store output for plotting
            preds_prob = []
            preds_bin = []
            label_windowed = []

            for t in range(seq_len-window_size_samples):
                window = full_curve[:, t:t+window_size_samples, :]
                label =  full_label[:, t+window_size_samples, :]
                y_pred, _ = model(window)
                curve_loss += criterion(y_pred, label)

                y_pred_np = y_pred.squeeze().cpu().numpy()
                y_bin = int(y_pred_np > threshold)

                preds_prob.append(y_pred_np)
                preds_bin.append(y_bin)
                label_windowed.append(label.squeeze().cpu().numpy())

            loss += curve_loss.item() / (seq_len - window_size_samples)

            all_targets.extend(label_windowed)
            all_outputs.extend(preds_bin)

            if save_folder:
                save_path = save_folder / f'eval_plot_{batch_i+1}.png'
                plot_eval(full_curve[:,window_size_samples:].cpu().numpy().flatten(), label_windowed, preds_prob, downsampling_freq, save_path=save_path)
            if np.any(np.diff(label_windowed)==1):
                true_start_time = np.where(np.diff(label_windowed)==1)[0] + 1
                if np.any(np.diff(preds_bin)==1):
                    pred_start_time = np.where(np.diff(preds_bin)==1)[0] + 1
                else :
                    pred_start_time = np.array([])
                start_accuracies.append(start_end_accuracy(pred_start_time, true_start_time, downsampling_freq))


            if np.any(np.diff(label_windowed)==-1):
                true_end_time = np.where(np.diff(label_windowed)==-1)[0] + 1
                if np.any(np.diff(preds_bin)==-1):
                    pred_end_time = np.where(np.diff(preds_bin)==-1)[0] + 1
                else:
                    pred_end_time = np.array([])
                end_accuracies.append(start_end_accuracy(pred_end_time, true_end_time, downsampling_freq))

    accuracy = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}', file=sys.stderr)
    return loss/len(test_loader), accuracy, precision, recall, f1, start_accuracies, end_accuracies
            


def plot_eval(inputs, targets, outputs, downsampling_freq, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(len(inputs)) / downsampling_freq
    plt.plot(timesteps, inputs, label='Sensor Input Normalized', alpha=0.5)
    plt.plot(timesteps, targets, label='True Labels', alpha=0.7)
    plt.plot(timesteps, outputs, label='Predicted Labels', alpha=0.7)
    plt.legend()
    plt.title('GRU Model Evaluation')
    plt.xlabel('Time (s)')
    plt.ylabel('In Clot Probability')
    plt.savefig(save_path)

def start_end_accuracy(pred, true , downsampling_freq, tolerance=1.5):
    if pred.size == 0:
        return False
    pred = pred[0]
    true = true[0]
    return abs(pred - true) <= tolerance * downsampling_freq