from typing import Callable, List, Tuple, Dict
import yaml
from sklearn.model_selection import train_test_split

from utils.config_loader import load_config
from cross_validate import cross_validate
from fit import fit
from evaluate import evaluate
from data.sensor_dataset import SensorDataset
import data.paths as paths

# Type aliases to clarify the expected signatures for dataset providers
LogFetcher = Callable[[], Tuple[List[str], str]]
EvalSet = Tuple[str, LogFetcher]
EvalSetBuilder = Callable[[List[str], str], List[EvalSet]]

FIT_CFG_PATH = "configs/fit/base_fit.yml"
MODEL_CFG_PATH = "configs/model/base_model_dropout.yml"

def _build_search_space(features: List[str]) -> Dict[str, List]:
    base_search = {
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "hidden_size": [8, 16, 32],
        "num_layers": [1, 2],
        "patience": [50],
    }

    if "images" in features:
        base_search.update(
            {
                "emb_dim": [1, 2],
                "hidden_size": [8, 16],
                "num_layers": [1],
                "learning_rate": [1e-3],
                "weight_decay": [1e-4],
                "patience": [100],
            }
        )

    return base_search

def _apply_best_params(best_params: Dict, fit_cfg: Dict, model_cfg: Dict) -> None:
    for key in ["learning_rate", "weight_decay", "patience"]:
        fit_cfg[key] = best_params[key]

    for key in ["hidden_size", "num_layers"]:
        model_cfg[key] = best_params[key]

    if "emb_dim" in best_params:
        model_cfg["emb_dim"] = best_params["emb_dim"]

def run_experiment(
    experiment_name: str,
    data_cfg_path: str,
    train_logs_fn: LogFetcher,
    eval_sets_builder: EvalSetBuilder,
) -> None:
    """Shared train/eval loop for experiment variants that only differ by datasets."""
    print(f"Starting {experiment_name} experiment with data config: {data_cfg_path}")

    data_cfg, fit_cfg, model_cfg = load_config(data_cfg_path, FIT_CFG_PATH, MODEL_CFG_PATH)

    log_names, folder_name = train_logs_fn()
    train_log_names, test_log_names = train_test_split(log_names, test_size=0.2, random_state=42)

    train_dataset = SensorDataset(
        log_names=train_log_names,
        mode="train",
        downsampling_freq=data_cfg["downsampling_freq"],
        with_kde_weighting=data_cfg.get("with_kde_weighting", False),
        features=data_cfg["features"],
        mean_force=None,
        std_force=None,
    )

    search_space = _build_search_space(data_cfg["features"])
    best_params, _, _ = cross_validate(
        data_cfg,
        fit_cfg,
        model_cfg,
        search_space,
        train_log_names,
    )

    _apply_best_params(best_params, fit_cfg, model_cfg)

    fit_cfg["log"] = True
    model_path = fit(data_cfg, fit_cfg, model_cfg, train_log_names)

    complete_cfg_path = (
        paths.GRU_RESULTS_FOLDER
        / str(model_path).split("/")[-1].replace("gru_model_", "").replace(".pt", "")
        / "complete_config.yml"
    )
    with open(complete_cfg_path, "r") as f:
        complete_cfg = yaml.safe_load(f)

    mean_force = train_dataset.mean_force
    std_force = train_dataset.std_force

    eval_sets = eval_sets_builder(test_log_names, folder_name)
    for description, fetch_logs in eval_sets:
        eval_log_names, eval_folder_name = fetch_logs()
        print(f"[{experiment_name}] Evaluating on {description} ({len(eval_log_names)} logs)")
        evaluate(
            model_path,
            complete_cfg,
            eval_log_names,
            eval_folder_name,
            mean_force,
            std_force,
        )
