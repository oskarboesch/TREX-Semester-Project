import yaml

def load_config(data_config_path: str, fit_config_path: str, model_config_path: str):

    with open(data_config_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    with open(fit_config_path, "r") as f:
        fit_cfg = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # check that direction belongs to allowed values
    allowed_directions = ["Forward", "Backward", "Both"]
    if data_cfg["direction"] not in allowed_directions:
        raise ValueError(f"Invalid direction '{data_cfg['direction']}' in data config. Allowed values are {allowed_directions}.")

    return data_cfg, fit_cfg, model_cfg
