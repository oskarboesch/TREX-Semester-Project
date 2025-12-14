from torch import nn
import torch
import wandb
import numpy as np
import sys
import torch.nn.functional as F
import copy



class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, with_images=False, emb_dim=1):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.with_images = with_images
        if self.with_images:
            self.mask_encoder = MaskEncoder(in_channels=2, emb_dim=emb_dim)
            input_size += emb_dim  # increase input size by embedding dimension
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # x -> (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask=None, h=None):
        if mask is not None and mask.numel() > 0:
            # Encode masks frame-by-frame
            # Expected mask shape: (B, T, C, H, W)
            B, T, C, H, W = mask.shape

            mask = mask.view(B*T, C, H, W)      # merge batch + time
            mask_emb = self.mask_encoder(mask)  # → (B*T, 8)
            mask_emb = mask_emb.view(B, T, -1)  # → (B, T, 8)

            if x.numel() == 0:
                x = mask_emb
            else :  
                x = torch.cat([x, mask_emb], dim=-1)
        else :
            if x.numel() == 0:
                raise ValueError("Either x or mask must be provided")
        if h is not None:
            h0 = h
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out) # (batch_size, seq_length, output_size)
        # out: tensor of shape (batch_size, output_size)
        return torch.sigmoid(out), h
    

class MaskEncoder(nn.Module):
    def __init__(self, in_channels=2, emb_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=(3,7), padding=(1,3))
        self.bn1 = nn.BatchNorm2d(8)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8, emb_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.gap(x).view(x.size(0), -1)
        emb = self.fc(x)  # (B, emb_dim)
        return emb
    

def train_gru_model(model, train_loader, criterion, optimizer, num_epochs, device, downsampling_freq, threshold, log=False, test_loader=None, early_stopping=True, patience=50):
    model.to(device)
    best_val_f1 = 0.0
    counter = 0
    best_model_state = None
    # ensure criterion returns per-sample loss
    for epoch in range(num_epochs):
        for inputs, masks, targets, weights in train_loader:
            model.train()

            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            outputs, _ = model(inputs, masks)
            criterion.reduction = 'none'
            loss = criterion(outputs, targets)
            # apply weights
            loss = (loss * weights.to(device)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq, threshold=threshold)
        # EARLY STOP CHECK
        if early_stopping:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                counter = 0  # reset patience
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                print(f"EarlyStopping: no improvement for {counter}/{patience} epochs.")

            if counter >= patience:
                print("Early stopping triggered.")
                break
        if log:
            wandb.log({"epoch": epoch+1, "loss": loss.item()})
            if test_loader is not None:
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
    # Restore the best model
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_gru_model(model, test_loader, device, criterion, downsampling_freq, save_folder=None, threshold=0.9):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    criterion.reduction = 'mean'
    model.eval()
    model.to(device)
    all_targets = []
    all_outputs = []
    loss = 0.0
    start_accuracies = []
    end_accuracies = []
    
    with torch.no_grad():
        for batch_i, (inputs, masks, targets, _) in enumerate(test_loader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # pass through the model
            outputs, _ = model(inputs, masks)
            batch_loss = criterion(outputs, targets)
            loss += batch_loss.item()

            # change to np for plotting and metrics
            inputs = inputs.cpu().numpy().squeeze()
            masks = masks.cpu().numpy().squeeze()
            targets = targets.cpu().numpy().squeeze()
            outputs = outputs.cpu().numpy().squeeze()

            # collect binary predictions
            preds_bin = (outputs >= threshold).astype(int)
            all_targets.extend(targets.tolist())
            all_outputs.extend(preds_bin.tolist())



            if save_folder:
                save_path = save_folder / f'eval_plot_{batch_i+1}.png'
                plot_eval(inputs, masks, targets, outputs, downsampling_freq, save_path=save_path)

            # Extract start & end indices safely
            true_start_time = np.where(np.diff(targets) == 1)[0]
            pred_start_time = np.where(np.diff(preds_bin) == 1)[0]

            # If empty, set to None
            true_start_time = true_start_time + 1 if len(true_start_time) > 0 else None
            pred_start_time = pred_start_time + 1 if len(pred_start_time) > 0 else None

            start_accuracies.append(
                start_end_accuracy(pred_start_time, true_start_time, downsampling_freq)
            )

            true_end_time = np.where(np.diff(targets) == -1)[0]
            pred_end_time = np.where(np.diff(preds_bin) == -1)[0]

            true_end_time = true_end_time + 1 if len(true_end_time) > 0 else None
            pred_end_time = pred_end_time + 1 if len(pred_end_time) > 0 else None

            end_accuracies.append(
                start_end_accuracy(pred_end_time, true_end_time, downsampling_freq)
            )


    accuracy = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, start_accuracy: {np.mean(start_accuracies):.4f}, end_accuracy: {np.mean(end_accuracies):.4f}')
    return loss/len(test_loader), accuracy, precision, recall, f1, start_accuracies, end_accuracies
            


def plot_eval(inputs, masks, targets, outputs, downsampling_freq, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy safely
    inputs  = np.asarray(inputs)
    targets = np.asarray(targets)
    outputs = np.asarray(outputs)

    T = len(targets)

    # If nothing to plot, skip
    if T == 0:
        return

    timesteps = np.arange(T) / downsampling_freq

    plt.figure(figsize=(12, 6))

    # ---- Plot inputs ONLY if they exist and match length ----
    if inputs.size > 0:
        if inputs.ndim == 1:
            if len(inputs) == T:
                plt.plot(timesteps, inputs, label='Force Sensor', alpha=0.5)
        elif inputs.ndim == 2:
            if inputs.shape[0] == T:
                input_size = inputs.shape[1]
                for i in range(input_size):
                    plt.plot(
                        timesteps,
                        inputs[:, i],
                        alpha=0.3,
                        label=f'Input {i}'
                    )

    # ---- Always plot targets & outputs ----
    plt.plot(timesteps, targets, label='True Labels', alpha=0.7)
    plt.plot(timesteps, outputs, label='Predicted Labels', alpha=0.7)

    plt.legend()
    plt.title('GRU Model Evaluation')
    plt.xlabel('Time (s)')
    plt.ylabel('In Clot Probability')

    plt.savefig(save_path)
    plt.close()


def start_end_accuracy(pred, true , downsampling_freq, tolerance=1.5):
    if pred is None and true is None:
        return True
    if pred is None or true is None:
        return False
    if pred.size == 0 or true.size == 0:
        return False
    pred = pred[0]
    true = true[0]
    return abs(pred - true) <= tolerance * downsampling_freq

def load_gru_model(model_path, input_size, cfg, device):
    model = GRUClassifier(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=1,
        dropout=cfg["dropout"],
        with_images=("images" in cfg["features"]),
        emb_dim=cfg.get("emb_dim", 1)
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model