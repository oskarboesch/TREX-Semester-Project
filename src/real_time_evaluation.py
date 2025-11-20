import sys
from pathlib import Path
import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_src_in_path():
    # make imports from project `src` work when running this script
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


ensure_src_in_path()

from data.load_data import load_data
from data.preprocess_data import preprocess_log
from models.gru import GRUClassifier, CNN_GRUClassifier


def find_run_config(run_id: str):
    base = Path.cwd() / "results"
    # search recursively for a folder that matches run_id
    matches = list((base).rglob(run_id))
    for m in matches:
        if m.is_dir():
            cfg = m / "complete_config.yml"
            if cfg.exists():
                return cfg
    return None


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_checkpoint_for_run(run_id: str):
    models_dir = Path.cwd() / "models"
    if not models_dir.exists():
        return None
    matches = list(models_dir.rglob(f"*{run_id}*.pt"))
    if matches:
        return matches[0]
    # fallback: any .pt that contains run_id in name
    for p in models_dir.rglob("*.pt"):
        if run_id in p.name:
            return p
    return None


def infer_model_from_state(state_dict: dict, config: dict):
    # detect CNN vs GRU by looking for conv1 weight
    keys = list(state_dict.keys())
    # sometimes state_dict may be nested under keys like 'model_state_dict'
    if any(k.endswith("conv1.weight") or ".conv1.weight" in k for k in keys):
        # CNN_GRU
        # infer shapes
        conv_key = next(k for k in keys if k.endswith("conv1.weight") or ".conv1.weight" in k)
        conv_w = state_dict[conv_key]
        # conv_w shape: (out_channels, in_channels, kernel_size)
        cnn_channels = conv_w.shape[0]
        input_size = conv_w.shape[1]
        # fc weight to infer hidden and output
        fc_key = next((k for k in keys if k.endswith("fc.weight") or ".fc.weight" in k), None)
        if fc_key is None:
            raise RuntimeError("Cannot find fc.weight in state_dict to infer sizes")
        fc_w = state_dict[fc_key]
        output_size = int(fc_w.shape[0])
        hidden_size = int(fc_w.shape[1])
        num_layers = int(config.get("num_layers", 1))
        model = CNN_GRUClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, cnn_channels=cnn_channels, dropout=config.get("dropout", 0.2))
        return model
    else:
        # GRU only
        # find fc weight
        fc_key = next((k for k in keys if k.endswith("fc.weight") or ".fc.weight" in k), None)
        if fc_key is None:
            raise RuntimeError("Cannot find fc.weight in state_dict to infer sizes")
        fc_w = state_dict[fc_key]
        output_size = int(fc_w.shape[0])
        hidden_size = int(fc_w.shape[1])
        # try to infer input_size from gru weight_ih_l0
        gru_key = next((k for k in keys if k.endswith("gru.weight_ih_l0") or ".gru.weight_ih_l0" in k), None)
        if gru_key is not None:
            gru_w = state_dict[gru_key]
            input_size = int(gru_w.shape[1])
        else:
            # default to 1 (e.g., single force sensor)
            input_size = int(config.get("input_size", 1))
        num_layers = int(config.get("num_layers", 1))
        model = GRUClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=config.get("dropout", 0.2))
        return model


def load_model_for_run(run_id: str, device="cpu"):
    cfg_path = find_run_config(run_id)
    if cfg_path is None:
        raise FileNotFoundError(f"Config for run {run_id} not found in results/")
    config = load_yaml(cfg_path)

    ckpt = find_checkpoint_for_run(run_id)
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint for run {run_id} not found in models/")

    checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))

    # determine where the state_dict sits
    if isinstance(checkpoint, dict):
        # try common keys
        for key in ["state_dict", "model_state_dict", "model_state", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        else:
            # maybe it's already a state_dict
            state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    else:
        raise RuntimeError("Unsupported checkpoint format")

    # infer and build model
    model = infer_model_from_state(state_dict, config)
    # load state dict (try matching prefixes)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # try removing "module." prefix
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state)

    model.to(device)
    model.eval()
    return model, config


def get_processed_log_df(log_path: str, downsampling_freq: int, direction: str = None):
    # load single log CSV into DataFrame and preprocess using existing helper
    logs_df = pd.DataFrame([{"path": str(log_path), "path_markers": str(Path(log_path).parent / "markers.mat"), "direction": direction if direction is not None else "Forward"}])
    df_list = load_data(logs_df, heads_keep=["timestamp [s]", "Force sensor voltage [V]"], heads_rename=["timestamps", "force_sensor_v"], fss=None)
    # preprocess_log expects DataFrame and head name
    df_down = preprocess_log(df_list[0], head="force_sensor_mN" if "force_sensor_mN" in df_list[0].columns else "force_sensor_v", direction=logs_df.loc[0, "direction"], downsampling_freq=downsampling_freq)
    return df_down


def get_window(df_down: pd.DataFrame, time_s: float, window_seconds: float, downsampling_freq: int, mean_force=None, std_force=None):
    # df_down expected to have 'timestamps' and 'force_sensor_mN' or 'force_sensor_v'
    head = "force_sensor_mN" if "force_sensor_mN" in df_down.columns else "force_sensor_v"
    timestamps = df_down["timestamps"].values
    values = df_down[head].values

    # desired window: (time_s - window_seconds, time_s]
    start_t = time_s - window_seconds + (1.0 / downsampling_freq)
    if start_t < timestamps[0]:
        # pad at left with first value
        pad_count = int(round((timestamps[0] - start_t) * downsampling_freq))
    else:
        pad_count = 0

    keep_mask = (timestamps >= start_t) & (timestamps <= time_s)
    window_vals = values[keep_mask]

    if pad_count > 0:
        pad_vals = np.full((pad_count,), values[0])
        window_vals = np.concatenate([pad_vals, window_vals])

    # if longer or shorter than expected, trim or pad
    expected_len = int(round(window_seconds * downsampling_freq))
    if len(window_vals) < expected_len:
        pad_vals = np.full((expected_len - len(window_vals),), window_vals[0] if len(window_vals) > 0 else 0.0)
        window_vals = np.concatenate([pad_vals, window_vals])
    elif len(window_vals) > expected_len:
        window_vals = window_vals[-expected_len:]

    # normalize
    if mean_force is None or std_force is None:
        mean_force = window_vals.mean()
        std_force = window_vals.std() if window_vals.std() > 0 else 1.0

    window_norm = (window_vals - mean_force) / std_force
    # shape -> (seq_len, 1)
    return window_norm.reshape(-1, 1), timestamps[keep_mask][-expected_len:] if np.any(keep_mask) else None


def stream_predict_plot(model, df_down: pd.DataFrame, config: dict, device="cpu", sim_rate=10.0):
    # config expects downsampling_freq and window_size
    downsampling_freq = int(config.get("downsampling_freq", 5))
    window_size = float(config.get("window_size", 10.0))

    timestamps = df_down["timestamps"].values
    head = "force_sensor_mN" if "force_sensor_mN" in df_down.columns else "force_sensor_v"
    values = df_down[head].values

    seq_len = int(round(window_size * downsampling_freq))

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line_force, = ax.plot([], [], label="force")
    line_pred, = ax.plot([], [], label="pred_prob")
    vline = ax.axvline(0, color='k', linestyle='--')
    ax.set_ylim(values.min() - abs(values.min())*0.1, values.max() + abs(values.max())*0.1)
    ax.legend()

    start_time = timestamps[0]
    end_time = timestamps[-1]

    # simulate time steps at sim_rate Hz (sim_rate steps per second of real time)
    step_s = 1.0 / downsampling_freq
    sim_step = step_s  # advance by one sample per iteration by default

    t = start_time + window_size
    while t <= end_time:
        window, win_timestamps = get_window(df_down, t, window_size, downsampling_freq)
        if window is None:
            t += sim_step
            continue

        x = torch.tensor(window.astype(np.float32)).unsqueeze(0).to(device)  # (1, seq_len, input_size)
        with torch.no_grad():
            out = model(x)  # (1, seq_len, output_size)
        out_np = out.squeeze().cpu().numpy()
        if out_np.ndim > 1:
            out_np = out_np[:, 0]

        # update plot to show last window
        times_plot = np.linspace(t - window_size + (1.0 / downsampling_freq), t, seq_len)
        line_force.set_data(times_plot, window.flatten())
        line_pred.set_data(times_plot, out_np)
        ax.set_xlim(times_plot[0], times_plot[-1])
        vline.set_xdata([t])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(1.0 / sim_rate)
        t += sim_step

    plt.ioff()


def predict_at_time(model, df_down: pd.DataFrame, config: dict, time_s: float, device="cpu"):
    downsampling_freq = int(config.get("downsampling_freq", 5))
    window_size = float(config.get("window_size", 10.0))
    window, win_timestamps = get_window(df_down, time_s, window_size, downsampling_freq)
    x = torch.tensor(window.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    out_np = out.squeeze().cpu().numpy()
    return window.flatten(), out_np, win_timestamps


def main():
    parser = argparse.ArgumentParser(description="Real-time evaluation of GRU/CNN-GRU models on curve data")
    parser.add_argument("run_id", help="Run id (folder name under results) or timestamp used in model filenames")
    parser.add_argument("log_path", help="Path to a CSV log to simulate (single file)")
    parser.add_argument("--mode", choices=["stream", "at"], default="stream")
    parser.add_argument("--time", type=float, default=0.0, help="Time in seconds to evaluate when mode=='at'")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim_rate", type=float, default=10.0, help="Visualization refresh rate (Hz) for stream mode")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_model_for_run(args.run_id, device=device)

    downsampling_freq = int(config.get("downsampling_freq", 5))
    df_down = get_processed_log_df(args.log_path, downsampling_freq, direction=None)

    if args.mode == "stream":
        stream_predict_plot(model, df_down, config, device=device, sim_rate=args.sim_rate)
    else:
        window_vals, out_np, win_timestamps = predict_at_time(model, df_down, config, args.time, device=device)
        # simple plot
        times_plot = np.linspace(args.time - float(config.get("window_size", 10.0)) + (1.0 / downsampling_freq), args.time, int(round(float(config.get("window_size", 10.0)) * downsampling_freq)))
        plt.figure(figsize=(10, 4))
        plt.plot(times_plot, window_vals, label="force (normalized)")
        if out_np.ndim > 1:
            out_np = out_np[:, 0]
        plt.plot(times_plot, out_np, label="pred_prob")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
