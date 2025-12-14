from utils.config_loader import load_config
import data.load_data as ld
from cross_validate import cross_validate
from fit import fit
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import yaml
from data.sensor_dataset import SensorDataset
import data.paths as paths

def paper_data_experiment(data_cfg_path):
    ## First Experiment -> use the data from published paper:
    # Let's get simplest config first with only signal data

    print("Starting paper data experiment with data config:", data_cfg_path)
    fit_cfg_path = "configs/fit/base_fit.yml"
    model_cfg_path = "configs/model/base_model_dropout.yml"

    data_cfg, fit_cfg, model_cfg = load_config(data_cfg_path, fit_cfg_path, model_cfg_path)

    # Get the paper dataset
    log_names, folder_name = ld.get_paper_logs()

    train_log_names, test_log_names = train_test_split(log_names, test_size=0.2, random_state=42)

    # we need the train dataset to get mean and std of force for normalization during evaluation
    train_dataset = SensorDataset(log_names=train_log_names, mode='train', downsampling_freq=data_cfg["downsampling_freq"], with_kde_weighting=data_cfg.get("with_kde_weighting", False), features=data_cfg["features"], mean_force=None, std_force=None)

    search_space = {
    "learning_rate": [1e-4, 1e-3],
    "weight_decay": [0, 1e-5, 1e-4],
    "hidden_size": [4, 8, 16, 32],
    "num_layers": [1, 2],
    }
    if "images" in data_cfg["features"]:
        search_space["emb_dim"] = [1, 2]

    # Now find the best parameters
    best_params, best_score, all_results = cross_validate(data_cfg, fit_cfg, model_cfg, search_space, train_log_names)

    # Use them to train the model again on complete train dataset
    fit_cfg['learning_rate'] = best_params["learning_rate"]
    fit_cfg['weight_decay'] = best_params["weight_decay"]
    model_cfg['hidden_size'] = best_params["hidden_size"]
    model_cfg['num_layers'] = best_params["num_layers"]

    # Now we log on wandb
    fit_cfg['log'] = True

    model_path = fit(data_cfg, fit_cfg, model_cfg, log_names)

    # complete config is under results/gru_results/<run_id>/complete_config.yml and model_path = gru_model_YYYY-MM-DD_HH-MM-SS.pt
    complete_cfg_path = paths.GRU_RESULTS_FOLDER / str(model_path).split("/")[-1].replace("gru_model_", "").replace(".pt", "") / "complete_config.yml"
    with open(complete_cfg_path, "r") as f:
        complete_cfg = yaml.safe_load(f)

    # Let's evaluate on paper dataset
    evaluate(model_path, complete_cfg, test_log_names, folder_name, train_dataset.mean_force, train_dataset.std_force)
    
    # We can also evaluate on Bent data
    bent_conical_log_names, bent_conical_folder_name = ld.get_conical_logs(bent=True)
    evaluate(model_path, complete_cfg, bent_conical_log_names, bent_conical_folder_name, train_dataset.mean_force, train_dataset.std_force)
    # On Anatomical Data
    anat_log_names, anat_folder_name = ld.get_anat_logs()
    evaluate(model_path, complete_cfg, anat_log_names, anat_folder_name, train_dataset.mean_force, train_dataset.std_force)

    # On Conical data but without clot ? 
    no_clot_log_names, no_clot_folder = ld.get_conical_logs(with_clot=False)
    evaluate(model_path, complete_cfg, no_clot_log_names, no_clot_folder, train_dataset.mean_force, train_dataset.std_force)

def paper_data_experiments_all():
    data_cfg_paths = [ "configs/data/base_forward.yml", # simplest model with only forward data
                      "configs/data/forward_freq.yml",  # add frequency components
                      "configs/data/forward_image.yml", # use only images as input
                      "configs/data/complete_forward.yml", # use complete forward data (signal + freq + image)
                      "configs/data/complete_backward.yml", # use complete backward data
                       "configs/data/complete_both.yml" # use complete data of both directions
    ]
    for data_cfg_path in data_cfg_paths:
        paper_data_experiment(data_cfg_path)

    print("="*10 + "All paper data experiments completed."+ "="*10)