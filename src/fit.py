from load_data import load_data, list_logs
from preprocess_data import preprocess_logs
from models.helpers import create_model_params
from models.buckling_model import BucklingModel
import config as config
import numpy as np
import wandb
from datetime import datetime

LOG = True

if LOG:
    wandb.init(project="buckling-model", dir="../wandb" )

config.ensure_directories()
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
heads_rename = ["timestamps", "force_sensor_v"]
f_s = 1000
fss = 568.5
log_names = list_logs(config.PAPER_EXPERIMENT_DATA_FOLDER)
# discard logs 7, 158, 174, errors ? 
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


model_bk = BucklingModel(model_params, xdata, ydata, log=LOG)
res_bk = model_bk.fit(method="SLSQP", options={
    "maxiter": 10e4,
    "ftol": 1e-8,
    "disp": True
})
# save model
model_save_path = config.MODELS_FOLDER / f"buckling_model_{run_id}.npz"
np.savez(model_save_path, res_bk=res_bk, model_params=model_params)

if LOG:
    # === 6. Log final result ===
    wandb.log({"final_loss": res_bk.fun})
    wandb.finish()
