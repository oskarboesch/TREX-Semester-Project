import numpy as np
from models.helpers import bounded
from models.base_model import BaseModel

class DoubleSlopeModel(BaseModel):
    def __init__(self, model_params, xdata, ydata, labels, log=False):
        super().__init__("DoubleSlope", model_params, xdata, ydata, labels, log)
        self.x_names = ["x_start", "x_end", "f_0", "y_start", "y_end"]


        # Define parameter bounds
        self.lb_curve = np.array([
            self.x_min + model_params["margin"][0],  # x_start
            self.x_min + model_params["margin"][0],  # x_end
            self.y_min,                              # f_0
            self.y_min,                              # y_start
            self.y_min                               # y_end
        ])

        self.ub_curve = np.array([
            self.x_max - model_params["margin"][1],  # x_start
            self.x_max - model_params["margin"][1],  # x_end
            self.y_max,                              # f_0
            self.y_max,                              # y_start
            self.y_max                               # y_end
        ])
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

    # ----------------- MODEL FUNCTION -----------------
    def model_fun(self, x, params):
        """
        Compute model output y for input x and parameters P = [x_start, x_end, f_0, y_start, y_end]
        """
        x_start, x_end, f_0, y_start, y_end = params
        term1 = bounded(x, x[0], x_start, y_start - f_0)
        term2 = bounded(x, x_start, x_end, y_end - y_start)
        return f_0 + term1 + term2