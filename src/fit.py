import numpy as np
import wandb
from datetime import datetime
from data.load_data import list_logs
import data.paths as paths
import argparse
from models.gru import GRUClassifier, train_gru_model, evaluate_gru_model
import torch
from data.sensor_dataset import SensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.seed import set_seed
from utils.config_loader import load_config
paths.ensure_directories()


def fit(data_cfg, fit_cfg, model_cfg):
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch_generator = set_seed(data_cfg["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    log_names = list_logs(paths.PAPER_EXPERIMENT_DATA_FOLDER)
    if data_cfg["direction"] != "Both":
        log_names = log_names[log_names["direction"] == data_cfg["direction"]].reset_index(drop=True)
    print(f"Total logs for {data_cfg['direction']} direction: {len(log_names)}")

    train_log_names, test_log_names = train_test_split(log_names, test_size=fit_cfg["test_size_ratio"], random_state=data_cfg["seed"])
    train_log_names = train_log_names.reset_index(drop=True)
    test_log_names = test_log_names.reset_index(drop=True)
    print(f"Training logs: {len(train_log_names)}, Testing logs: {len(test_log_names)}")
    train_dataset = SensorDataset(train_log_names, mode='train', downsampling_freq=data_cfg["downsampling_freq"], with_kde_weighting=fit_cfg.get("with_kde_weighting", False), features=data_cfg["features"])
    test_dataset  = SensorDataset(test_log_names, mode='eval', downsampling_freq=data_cfg["downsampling_freq"], mean_force=train_dataset.mean_force, std_force=train_dataset.std_force, features=data_cfg["features"])
    train_loader = DataLoader(train_dataset, batch_size=fit_cfg["batch_size"], shuffle=True, generator=torch_generator)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, generator=torch_generator)

    # Define model
    input_size = train_dataset.input_size
    print(f"Input size: {input_size}")
    gru_model = GRUClassifier(input_size=input_size, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], output_size=1, dropout=model_cfg["dropout"], with_images=("images" in data_cfg["features"])).to(device)

    # Train 
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=fit_cfg["learning_rate"], weight_decay=fit_cfg["weight_decay"])
    print(f"Fitting {data_cfg['direction']} gru model with {fit_cfg['num_epochs']} epochs on {device}...")
    if fit_cfg.get("log", False):
        wandb.init(project="gru_model_fitting", name="GRU_Fit_" + RUN_ID)
        wandb.config.update({**data_cfg, **fit_cfg, **model_cfg})
    train_gru_model(gru_model, train_loader, criterion, optimizer, fit_cfg["num_epochs"], device, downsampling_freq=data_cfg["downsampling_freq"], threshold=fit_cfg["threshold"], log=fit_cfg.get("log", False), test_loader=test_loader)

    # save model
    model_save_path = paths.MODELS_FOLDER / f"gru_{data_cfg['direction'].lower()}_model_{RUN_ID}.pt"
    torch.save(gru_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # evaluate
    print("Evaluating model on test set...")
    save_folder = paths.GRU_RESULTS_FOLDER /  RUN_ID 
    save_folder.mkdir(parents=True, exist_ok=True)
    val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(gru_model, test_loader, device, criterion, data_cfg["downsampling_freq"], save_folder=save_folder, threshold=fit_cfg["threshold"])
    print(f"Test Set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1 Score: {val_f1:.4f}, Start Accuracy: {np.mean(start_accuracies):.4f}, End Accuracy: {np.mean(end_accuracies):.4f}")

    # save the train and test weights in function of lengths as a plot
    if fit_cfg.get("with_kde_weighting", False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.scatter(train_dataset.lengths, train_dataset.sample_weights[:len(train_dataset.lengths)], c='blue', label='Training Samples', alpha=0.6)
        plt.xlabel("Clot Length (samples)")
        plt.ylabel("Sample Weights")
        plt.title("Sample Weights vs Clot Length in Training Set")
        plt.grid()
        plt.savefig(save_folder / "sample_weights_vs_length.png")
        plt.close()
    # save a complete config used for this run
    complete_config = {**data_cfg, **fit_cfg, **model_cfg}
    import yaml
    with open(save_folder / "complete_config.yml", "w") as f:
        yaml.dump(complete_config, f)

def main():

    argparser = argparse.ArgumentParser(description="Fit models to experimental data.")
    argparser.add_argument("--data_config", type=str, required=True, help="Path to data configuration file")
    argparser.add_argument("--fit_config", type=str, required=True, help="Path to fit configuration file")
    argparser.add_argument("--model_config", type=str, required=True, help="Path to model configuration file")
    args = argparser.parse_args()

    data_cfg, fit_cfg, model_cfg = load_config(args.data_config, args.fit_config, args.model_config)
    fit(data_cfg, fit_cfg, model_cfg)

if __name__ == "__main__":
    main()