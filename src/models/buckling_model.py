import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from models.helpers import step, bounded
import wandb


class BucklingModel:
    def __init__(self, model_params, xdata, ydata, log=False):
        self.name = "Buckling"
        self.x_names = [
            "x_start", "x_end", "y_start", "y_end",
            "f_pun_rise", "f_pun_fall",
            "dx_rise", "df_rise", "k_rise",
            "dx_pun", "dx_fall"
        ]
        self.log = log

        # Ensure xdata and ydata are lists (multiple curves)
        if not isinstance(xdata, list):
            xdata = [xdata]
        if not isinstance(ydata, list):
            ydata = [ydata]

        self.n_curves = len(xdata)
        self.xdata = xdata
        self.ydata = ydata

        # Compute bounds across all curves
        x_min = min([np.min(x) for x in xdata])
        x_max = max([np.max(x) for x in xdata])
        y_min = min([np.min(y) for y in ydata])
        y_max = max([np.max(y) for y in ydata])
        y_range = y_max - y_min
        x_range = x_max - x_min

        # Lower/upper bounds per curve
        self.lb_curve = [
            x_min + model_params['margin'][0],
            x_min + model_params['margin'][0],
            y_min, y_min,
            0, 0,
            model_params['dx_min'], 0, 1,
            model_params['dx_min'], model_params['dx_min']
        ]
        self.ub_curve = [
            x_max - model_params['margin'][1],
            x_max - model_params['margin'][1],
            y_max, y_max,
            y_range, y_range,
            model_params['dx_max'], y_range, 2,
            x_range, x_range
        ]

        # Repeat bounds for multiple curves
        self.lb = np.tile(self.lb_curve, self.n_curves)
        self.ub = np.tile(self.ub_curve, self.n_curves)
        for i, (l, u) in enumerate(zip(self.lb, self.ub)):
            if l > u:
                print(f"Inconsistent bounds at index {i}: lb={l}, ub={u}")
        self.x0 = (self.lb + self.ub) / 2

        for i, (x0i, l, u) in enumerate(zip(self.x0, self.lb, self.ub)):
            if not (l <= x0i <= u):
                print(f"x0[{i}]={x0i} outside bounds [{l}, {u}]")
    
    def model_fun(self, x, params):
        (
            x_start, x_end, y_start, y_end,
            f_pun_rise, f_pun_fall,
            dx_rise, df_rise, k_rise,
            dx_pun, dx_fall
        ) = params
        
        term1 = y_start
        term2 = bounded(x - x_start, 0, dx_rise, 1)**k_rise * (f_pun_rise + df_rise) * step(x_start + dx_rise, x)
        term3 = bounded(x_start + dx_rise + dx_pun - x, 0, dx_pun, 1)**2 * df_rise * step(x, x_start + dx_rise)
        term4 = f_pun_rise * step(x, x_start + dx_rise)
        term5 = bounded(x, x_start + dx_rise, x_end, y_end + f_pun_fall - f_pun_rise - y_start)
        term6 = bounded(x_end + dx_fall - x, 0, dx_fall, 1)**2 * f_pun_fall * step(x, x_end)
        term7 = -f_pun_fall * step(x, x_end)
        return term1 + term2 + term3 + term4 + term5 + term6 + term7

    def objective(self, all_params):
        """Vectorized sum-of-squares across all curves"""
        sse = 0
        n_vars = len(self.lb_curve)
        for i in range(self.n_curves):
            idx = slice(i * n_vars, (i + 1) * n_vars)
            sse += np.sum((self.model_fun(self.xdata[i], all_params[idx]) - self.ydata[i])**2)
        
        if self.log:
            param_dict = {}
            for i in range(self.n_curves):
                idx = slice(i * n_vars, (i + 1) * n_vars)
                for name, val in zip(self.x_names, all_params[idx]):
                    param_dict[f"{name}_{i}"] = val
                # only log the first curve parameters to avoid redundancy
                if i == 0:
                    wandb.log({**param_dict})
            wandb.log({"train_sse": sse})

        return sse

    def fit(self, method="SLSQP", options=None):
        if options is None:
            options = {"maxiter": 10000, "ftol": 1e-8, "disp": True}
        res = minimize(
            self.objective, self.x0,
            bounds=list(zip(self.lb, self.ub)),
            method=method,
            options=options
        )
        return res
    
    def evaluate_accuracy(self, all_params):
        """ Evaluate accuracy from Start and End points """
        # TODO: Implement evaluation logic if needed
        pass