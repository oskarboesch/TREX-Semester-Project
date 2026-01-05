import numpy as np
import wandb
from datetime import datetime
import data.load_data as ld
import data.paths as paths
import argparse
from models.gru import GRUClassifier, train_gru_model, evaluate_gru_model
import torch
from data.sensor_dataset import SensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from utils.seed import set_seed
from utils.config_loader import load_config
import itertools


paths.ensure_directories()


def cross_validate(data_cfg, fit_cfg, model_cfg, search_space, log_names):


    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch_generator = set_seed(data_cfg["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keys = list(search_space.keys())
    values = list(search_space.values())

    hyperparam_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    best_score = float("inf")
    best_params = None
    all_results = []

    # Load Data
    if data_cfg["direction"] != "Both":
        log_names = log_names[log_names["direction"] == data_cfg["direction"]].reset_index(drop=True)
    print(f"Total logs for {data_cfg['direction']} direction: {len(log_names)}")

    train_log_names, test_log_names = train_test_split(log_names, test_size=fit_cfg["test_size_ratio"], random_state=data_cfg["seed"])
    train_log_names = train_log_names.reset_index(drop=True)
    test_log_names = test_log_names.reset_index(drop=True)

    for hparams in hyperparam_combinations:
        print(f"\n===== Testing hyperparameters: {hparams} =====")

        fold_metrics = run_kfold_cv(
            hparams, train_log_names, K=3,
            data_cfg=data_cfg, fit_cfg=fit_cfg, model_cfg=model_cfg,
            device=device, torch_generator=torch_generator
        )

        all_results.append(fold_metrics)

        print(f"Mean K-Fold Val Loss = {np.mean(fold_metrics['val_loss']):.4f}")

        if np.mean(fold_metrics["val_loss"]) < best_score:
            best_score = np.mean(fold_metrics["val_loss"])
            best_params = hparams

    print("\n==== BEST PARAMS ====")
    print(best_params)
    print(f"Best Mean Validation Loss: {best_score:.4f}")
    print(f"Results for all hyperparameter combinations:")
    for result in all_results:
        print(result)

    return best_params, best_score, all_results



def run_kfold_cv(hparams, train_log_names, K, data_cfg, fit_cfg, model_cfg, device, torch_generator):

    kf = KFold(n_splits=K, shuffle=True, random_state=data_cfg["seed"])

    fold_metrics = {
        "params": hparams,
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
        "start_accuracy": [],
        "end_accuracy": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_log_names)):
        print(f"\n--- Fold {fold+1}/{K} ---")

        fold_train = train_log_names.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_log_names.iloc[val_idx].reset_index(drop=True)

        train_dataset = SensorDataset(
            fold_train, mode="train",
            downsampling_freq=data_cfg["downsampling_freq"],
            with_kde_weighting=fit_cfg.get("with_kde_weighting", False),
            features=data_cfg["features"]
        )

        val_dataset = SensorDataset(
            fold_val, mode="eval",
            downsampling_freq=data_cfg["downsampling_freq"],
            mean_force=train_dataset.mean_force,
            std_force=train_dataset.std_force,
            features=data_cfg["features"]
        )

        train_loader = DataLoader(train_dataset, batch_size=fit_cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Build model with custom hyperparameters
        model = GRUClassifier(
            input_size=train_dataset.input_size,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["num_layers"],
            dropout=model_cfg["dropout"],
            output_size=1,
            with_images=("images" in data_cfg["features"])
        ).to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        if fit_cfg.get("log", False):
            wandb.init(project="gru_model_fitting", name="GRU_Fit_" + f"K{fold+1}")
            wandb.config.update({**data_cfg, **fit_cfg, **model_cfg})

        model = train_gru_model(
            model, train_loader, criterion, optimizer,
            num_epochs=fit_cfg["num_epochs"], device=device,
            downsampling_freq=data_cfg["downsampling_freq"],
            threshold=fit_cfg["threshold"],
            log=False,
            test_loader=val_loader,
            patience=hparams.get("patience", None)
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, start_accuracies, end_accuracies = evaluate_gru_model(
            model, val_loader, device, criterion,
            data_cfg["downsampling_freq"],
            threshold=fit_cfg["threshold"]
        )

        fold_metrics["val_loss"].append(val_loss)
        fold_metrics["val_acc"].append(val_acc)
        fold_metrics["val_prec"].append(val_prec)
        fold_metrics["val_rec"].append(val_rec)
        fold_metrics["val_f1"].append(val_f1)
        fold_metrics["start_accuracy"].append(np.mean(start_accuracies))
        fold_metrics["end_accuracy"].append(np.mean(end_accuracies))

    return fold_metrics

def main():
    argparser = argparse.ArgumentParser(description="Fit models to experimental data.")
    argparser.add_argument("--data_config", type=str, required=True, help="Path to data configuration file")
    argparser.add_argument("--fit_config", type=str, required=True, help="Path to fit configuration file")
    argparser.add_argument("--model_config", type=str, required=True, help="Path to model configuration file")
    args = argparser.parse_args()
    data_cfg, fit_cfg, model_cfg = load_config(args.data_config, args.fit_config, args.model_config)

    search_space = {
    "learning_rate": [1e-4, 1e-3],
    "weight_decay": [0, 1e-5, 1e-4],
    "hidden_size": [4, 8, 16, 32],
    "num_layers": [1, 2],
    }
    if "images" in data_cfg["features"]:
        search_space["emb_dim"] = [1, 4]

    paper_log_names, _ = ld.get_paper_logs()
    anat_log_names, _ = ld.get_anat_logs()
    conical_log_names, _ = ld.get_conical_logs()
    print("="*50)
    print("Cross validation with paper logs")
    cross_validate(data_cfg, fit_cfg, model_cfg, search_space, paper_log_names)
    print("="*50)
    print("Cross validation with anatomical logs")
    cross_validate(data_cfg, fit_cfg, model_cfg, search_space, anat_log_names)
    print("="*50)
    print("Cross validation with conical logs")
    cross_validate(data_cfg, fit_cfg, model_cfg, search_space, conical_log_names)


if __name__ == "__main__":
    main()