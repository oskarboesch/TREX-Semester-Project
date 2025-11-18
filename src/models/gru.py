from torch import nn
import torch
import wandb
from data.load_data import load_data
import numpy as np

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # x -> (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out)
        # out: tensor of shape (batch_size, output_size)
        return torch.sigmoid(out)
    

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
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)            # (batch, input_size, seq_len) for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)            # (batch, seq_len, features) for GRU
        x = self.dropout(x)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)                # (batch, seq_len, output_size)
        return torch.sigmoid(out)

    

def train_gru_model(model, train_loader, criterion, optimizer, num_epochs, device, downsampling_freq, threshold, log=False, test_loader=None):
    model.to(device)
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            model.train()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if log:
            wandb.log({"epoch": epoch+1, "loss": loss.item()})
            if test_loader is not None:
                val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq, threshold=threshold)
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                    "val_start_accuracy": np.mean(start_accuracies),
                    "val_end_accuracy": np.mean(end_accuracies)
                })
    return model

def evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq = None, save_folder=None, threshold=0.5):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    model.eval()
    model.to(device)
    all_targets = []
    all_outputs = []
    loss = 0.0
    start_accuracies = []
    end_accuracies = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # GRU prediction
            y_pred_logits = model(inputs)       # (1, seq_len, 1)
            y_pred_binary = (y_pred_logits > threshold).int()
            y_pred_prob_np = y_pred_logits.squeeze().cpu().numpy()
            y_pred_np = y_pred_binary.squeeze().cpu().numpy()
            y_true_np = targets.squeeze().cpu().numpy()
            all_targets.extend(y_true_np.flatten())
            all_outputs.extend(y_pred_np.flatten())
            loss += criterion(y_pred_logits, targets).item()
            if save_folder:
                save_path = save_folder / f'eval_plot_{i+1}.png'
                plot_eval(inputs.cpu().numpy().flatten(), y_true_np, y_pred_prob_np, downsampling_freq, save_path=save_path)
            
            pred_start_time = np.where(np.diff(y_pred_np)==1)[0] + 1
            pred_end_time = np.where(np.diff(y_pred_np)==-1)[0] + 1
            true_start_time = np.where(np.diff(y_true_np)==1)[0] + 1
            true_end_time = np.where(np.diff(y_true_np)==-1)[0] + 1
            if downsampling_freq is not None:
                start_accuracies.append(start_end_accuracy(pred_start_time, true_start_time, downsampling_freq))
                end_accuracies.append(start_end_accuracy(pred_end_time, true_end_time, downsampling_freq))

    accuracy = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return loss/len(test_loader), accuracy, precision, recall, f1, start_accuracies, end_accuracies
            


def plot_eval(inputs, targets, outputs, downsampling_freq, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(len(inputs)) * downsampling_freq
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