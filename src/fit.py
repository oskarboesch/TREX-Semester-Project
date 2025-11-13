import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
from data.load_data import load_data, list_logs, load_labels
from data.preprocess_data import preprocess_logs
from models.helpers import create_model_params
from models.buckling_model import BucklingModel
from models.double_slope_model import DoubleSlopeModel
import data.paths as paths
import argparse

# ============================
# CONFIGURATION
# ============================
argparser = argparse.ArgumentParser(description="Fit models to experimental data.")
argparser.add_argument("--forward", action="store_true", help="Use Buckling model (default: DoubleSlope model)")
args = argparser.parse_args()
FORWARD = args.forward
LOG = True
N_REPETITIONS = 50
TOLERANCE = 1.5
SEED = 0
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_NAME = "Buckling" if FORWARD else "DoubleSlope"
DIRECTION = "Forward" if FORWARD else "Backward"

paths.ensure_directories()
np.random.seed(SEED)

# ============================
# LOAD & PREPROCESS DATA
# ============================

heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
heads_rename = ["timestamps", "force_sensor_v"]
f_s = 1000
fss = 568.5

log_names = list_logs(paths.PAPER_EXPERIMENT_DATA_FOLDER)
log_names.drop([7, 158, 174], inplace=True, errors='ignore')
log_names.reset_index(drop=True, inplace=True)

logs = load_data(log_names, heads_keep, heads_rename, fss)
logs_fit = log_names.copy()
data_fit = logs.copy()
data_fit_plot = logs.copy()
labels = load_labels(logs_fit)
preprocess_logs(logs_fit, data_fit, data_fit_plot)

xdata = [data_fit[i]["timestamps"].values for i in range(len(data_fit)) if logs_fit["direction"][i] == DIRECTION]
ydata = [data_fit[i]["force_sensor_mN"].values for i in range(len(data_fit)) if logs_fit["direction"][i] == DIRECTION]
labels = [labels[i] for i in range(len(data_fit)) if logs_fit["direction"][i] == DIRECTION]

model_params = create_model_params()

# ============================
# SETUP RESULTS DIRECTORY
# ============================

results_root = Path(paths.RESULTS_FOLDER) / MODEL_NAME / RUN_ID
results_root.mkdir(parents=True, exist_ok=True)

# ============================
# MAIN FIT LOOP
# ============================

rep_results_fit = []
rep_rng_seeds = np.random.randint(0, 2**32, size=N_REPETITIONS)

for i_repeat, seed_i in enumerate(rep_rng_seeds, start=1):
    np.random.seed(seed_i)
    subfolder = results_root / f"repetition-{i_repeat}"
    subfolder.mkdir(parents=True, exist_ok=True)

    # W&B INIT (grouping by model type)
    if LOG:
        wandb.init(
            project="SLSQP_Model_Fitting",
            group=MODEL_NAME,                    # Grouped by model type
            name=f"{MODEL_NAME.lower()}_rep{i_repeat}_run{RUN_ID}",  # Unique name per repetition
            config={
                "seed": int(seed_i),
                "forward": FORWARD,
                "model_params": model_params,
            },
            dir="../wandb",
            reinit=True
        )

    # Choose the correct model
    if FORWARD:
        model = BucklingModel(model_params, xdata, ydata, labels, log=LOG)
    else:
        model = DoubleSlopeModel(model_params, xdata, ydata, labels, log=LOG)

    # Fit model
    options = {"maxiter": int(1e4), "ftol": 1e-8, "disp": False}
    res = model.fit(method="SLSQP", options=options)
    rep_results_fit.append(res)

    # Save results
    model_save_path = subfolder / f"{MODEL_NAME.lower()}_model_rep{i_repeat}.npz"
    np.savez(model_save_path, res=res, model_params=model_params)

    if LOG:
        wandb.log({
            "repeat": i_repeat,
            "final_loss": res.fun,
            "seed": int(seed_i)
        })
        wandb.finish()

    print(f"[{i_repeat}/{N_REPETITIONS}] Finished repetition {i_repeat} with loss {res.fun:.4e}")

# ============================
# SAVE SUMMARY
# ============================

summary_path = results_root / "summary.npz"
np.savez(summary_path, rep_results_fit=rep_results_fit, rep_rng_seeds=rep_rng_seeds)

print(f"All repetitions finished for {MODEL_NAME}")
