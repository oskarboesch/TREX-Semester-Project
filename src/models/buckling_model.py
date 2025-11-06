from models.helpers import step, bounded
from models.base_model import BaseModel
import numpy as np

class BucklingModel(BaseModel):
    def __init__(self, model_params, xdata, ydata, labels, log=False):
        super().__init__("Buckling", model_params, xdata, ydata, labels, log)
        self.x_names = [
            "x_start", "x_end", "y_start", "y_end",
            "f_pun_rise", "f_pun_fall",
            "dx_rise", "df_rise", "k_rise",
            "dx_pun", "dx_fall"
        ]

        # Lower/upper bounds per curve
        self.lb_curve = [
            self.x_min + model_params['margin'][0],
            self.x_min + model_params['margin'][0],
            self.y_min, self.y_min,
            0, 0,
            model_params['dx_min'], 0, 1,
            model_params['dx_min'], model_params['dx_min']
        ]
        self.ub_curve = [
            self.x_max - model_params['margin'][1],
            self.x_max - model_params['margin'][1],
            self.y_max, self.y_max,
            self.y_range, self.y_range,
            model_params['dx_max'], self.y_range, 2,
            self.x_range, self.x_range
        ]
                # Repeat bounds for multiple curves
        self.lb = np.tile(self.lb_curve, self.n_curves)
        self.ub = np.tile(self.ub_curve, self.n_curves)
        for i, (l, u) in enumerate(zip(self.lb, self.ub)):
            if l > u:
                print(f"Inconsistent bounds at index {i}: lb={l}, ub={u}")
        self.x0 = np.random.uniform(self.lb, self.ub)

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