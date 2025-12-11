from utils.config_loader import load_config
import data.load_data as ld
from cross_validate import cross_validate
from fit import fit
from sklearn.model_selection import train_test_split
from evaluate import evaluate

def twist_data_experiment(data_cfg_path):
    ## First Experiment -> use the data from published paper:
    # Let's get simplest config first with only signal data
    fit_cfg_path = "configs/fit/base_fit.yml"
    model_cfg_path = "configs/model/base_model_dropout.yml"

    data_cfg, fit_cfg, model_cfg = load_config(data_cfg_path, fit_cfg_path, model_cfg_path)

    # Get the paper dataset
    log_names, folder_name = ld.get_conical_logs(twist=True)

    train_log_names, test_log_names = train_test_split(log_names, test_size=0.2, random_state=42)

    search_space = {
    "learning_rate": [1e-4, 1e-3],
    "weight_decay": [0, 1e-5, 1e-4],
    "hidden_size": [4, 8, 16, 32],
    "num_layers": [1, 2],
    }

    if "images" in data_cfg["features"]:
        search_space["emb_dim"] = [1, 4]

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

    # Let's evaluate on the twist dataset
    evaluate(model_path, data_cfg, test_log_names, folder_name)
    
    # We can also evaluate on Paper data
    paper_log_names, paper_folder_name = ld.get_paper_logs()
    evaluate(model_path, data_cfg, paper_log_names, paper_folder_name)

    # On Bent data
    bent_conical_log_names, bent_conical_folder_name = ld.get_conical_logs(bent=True)
    evaluate(model_path, data_cfg, bent_conical_log_names, bent_conical_folder_name)

    # On Anatomical Twist Data
    anat_log_names, anat_folder_name = ld.get_anat_logs(twist=True)
    evaluate(model_path, data_cfg, anat_log_names, anat_folder_name)

    # On Conical data but without clot ? 
    no_clot_log_names, no_clot_folder = ld.get_conical_logs(with_clot=False)
    evaluate(model_path, data_cfg, no_clot_log_names, no_clot_folder)


def twist_data_experiments_all():
    data_cfg_paths = [ "configs/data/base_forward.yml", # simplest model with only forward data
                      "configs/data/forward_freq.yml",  # add frequency components
                      "configs/data/forward_image.yml", # use only images as input
                      "configs/data/complete_forward.yml", # use complete forward data (signal + freq + image)
                      "configs/data/complete_backward.yml", # use complete backward data
                       "configs/data/complete_both.yml" # use complete data of both directions
    ]
    for data_cfg_path in data_cfg_paths:
        twist_data_experiment(data_cfg_path)

