from models.gru import evaluate_gru_model, GRUClassifier, load_gru_model
from data.sensor_dataset import SensorDataset
from torch.utils.data import DataLoader
import torch
import data.paths as paths
import yaml
import data.load_data as ld
import numpy as np

def evaluate(model_path, cfg, log_names, folder_name, mean_force, std_force):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get run_id from folder which model is in
    run_id = str(model_path).split("/")[-2]

    if cfg["direction"] != "Both":
        log_names = log_names[log_names["direction"] == cfg["direction"]].reset_index(drop=True)
    print(f"Total logs for {cfg['direction']} direction: {len(log_names)}")

    dataset = SensorDataset(log_names, mode='eval', downsampling_freq=cfg["downsampling_freq"], with_kde_weighting=cfg.get("with_kde_weighting", False), features=cfg["features"], mean_force=mean_force, std_force=std_force)
    input_size = dataset.input_size
    model = load_gru_model(model_path, input_size, cfg, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = torch.nn.BCELoss()
    save_folder = paths.EVALUATION_RESULTS / run_id / folder_name
    save_folder.mkdir(parents=True, exist_ok=True)

    loss, accuracy, precision, recall, f1, start_accuracies, end_accuracies = evaluate_gru_model(model, dataloader, device, criterion=criterion, downsampling_freq=cfg.get("downsampling_freq", 1), save_folder=save_folder)

    # save results as txt
    with open(save_folder / "results.txt", "w") as f:
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Start Mean Accuracies: {np.mean(start_accuracies)}\n")
        f.write(f"End Mean Accuracies: {np.mean(end_accuracies)}\n")


    
