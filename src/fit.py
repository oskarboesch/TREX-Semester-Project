import numpy as np
import wandb
import os
from datetime import datetime
from pathlib import Path
import shutil
from load_data import load_data, list_logs
from preprocess_data import preprocess_logs
from models.helpers import create_model_params
from models.buckling_model import BucklingModel
import config as config

# ============================
# CONFIGURATION
# ============================

LOG = True
N_REPETITIONS = 50
CV_REPEATS = 40
TOLERANCE = 1.5
ALPHA = 0.1
SEED = 0
SAVE_SVG = True
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if LOG:
    wandb.init(project="buckling-model", dir="../wandb")

config.ensure_directories()
np.random.seed(SEED)

# ============================
# LOAD & PREPROCESS DATA
# ============================

heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
heads_rename = ["timestamps", "force_sensor_v"]
f_s = 1000
fss = 568.5

log_names = list_logs(config.PAPER_EXPERIMENT_DATA_FOLDER)
log_names.drop([7, 158, 174], inplace=True, errors='ignore')
log_names.reset_index(drop=True, inplace=True)

logs = load_data(log_names, heads_keep, heads_rename, fss)
logs_fit = log_names.copy()
data_fit = logs.copy()
data_fit_plot = logs.copy()
preprocess_logs(logs_fit, data_fit, data_fit_plot)

xdata = [df["timestamps"].values for df in data_fit]
ydata = [df["force_sensor_mN"].values for df in data_fit]
model_params = create_model_params()

# ============================
# SETUP RESULTS DIRECTORY
# ============================

results_root = Path(config.RESULTS_FOLDER) / RUN_ID
results_root.mkdir(parents=True, exist_ok=True)

# ============================
# MAIN FIT LOOP
# ============================

rep_results_fit = []
rep_results_eval = []
rep_rng_seeds = np.random.randint(0, 2**32, size=N_REPETITIONS)

for i_repeat, seed_i in enumerate(rep_rng_seeds, start=1):
    np.random.seed(seed_i)
    subfolder = results_root / f"repetition-{i_repeat}"
    subfolder.mkdir(parents=True, exist_ok=True)

    log_path = subfolder / "results.txt"
    with open(log_path, "w") as log_file:
        print(f"=== Repetition {i_repeat}/{N_REPETITIONS} ===", file=log_file)
        print(f"Seed: {seed_i}", file=log_file)

        # Initialize model
        model_bk = BucklingModel(model_params, xdata, ydata, log=LOG)

        # Define optimizer options (like fmincon SQP)
        options = {
            "maxiter": int(1e4),
            "ftol": 1e-8,
            "disp": False
        }

        # Fit model
        res_bk = model_bk.fit(method="SLSQP", options=options)
        rep_results_fit.append(res_bk)

        # Save intermediate results
        model_save_path = subfolder / f"buckling_model_rep{i_repeat}.npz"
        np.savez(model_save_path, res_bk=res_bk, model_params=model_params)

        if LOG:
            wandb.log({
                "repeat": i_repeat,
                "final_loss": res_bk.fun,
                "seed": int(seed_i)
            })

        print(f"Repetition {i_repeat} completed with loss {res_bk.fun:.4e}", file=log_file)

    # === Optional: Evaluation step ===
    # (You can plug in your `evaluate_fit_light` equivalent here if available)
    # results_eval_i = evaluate_fit_light(...)
    # rep_results_eval.append(results_eval_i)

    print(f"[{i_repeat}/{N_REPETITIONS}] Finished repetition {i_repeat}")

# ============================
# SAVE SUMMARY
# ============================

summary_path = results_root / "summary.npz"
np.savez(summary_path, rep_results_fit=rep_results_fit, rep_rng_seeds=rep_rng_seeds)

if LOG:
    wandb.finish()
