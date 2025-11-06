import numpy as np
from scipy.optimize import minimize
from models.helpers import bounded


class DoubleSlopeModel:
    def __init__(self, model_params, xdata, ydata, log=False):
        """
        Double Slope model (backward fit version)
        - Start with a slope outside clot
        - Steeper slope in clot
        - Plateau after clot

        Args:
            model_params (dict): parameters containing margin, dx_min, dx_max, slope_min
            xdata, ydata (arrays): input data
        """
        self.name = "DoubleSlope"
        self.x_names = ["x_start", "x_end", "f_0", "y_start", "y_end"]
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

        # Define parameter bounds
        self.lb = np.array([
            x_min + model_params["margin"][0],  # x_start
            x_min + model_params["margin"][0],  # x_end
            y_min,                              # f_0
            y_min,                              # y_start
            y_min                               # y_end
        ])

        self.ub = np.array([
            x_max - model_params["margin"][1],  # x_start
            x_max - model_params["margin"][1],  # x_end
            y_max,                              # f_0
            y_max,                              # y_start
            y_max                               # y_end
        ])

        # Midpoint initialization
        self.x0 = (self.lb + self.ub) / 2

        # Linear inequality constraints (A @ P <= b)
        # x_start <= x_end, f_0 <= y_start, y_start <= y_end
        self.A = np.array([
            [1, -1, 0, 0, 0],
            [0, 0, 1, -1, 0],
            [0, 0, 0, 1, -1]
        ])
        self.b = np.zeros(3)

        # Define auxiliary selectors
        self.A_start = np.array([1, 0, 0, 0, 0])
        self.A_end = np.array([0, 1, 0, 0, 0])

    # ----------------- MODEL FUNCTION -----------------
    def model_fun(self, x, params):
        """
        Compute model output y for input x and parameters P = [x_start, x_end, f_0, y_start, y_end]
        """
        x_start, x_end, f_0, y_start, y_end = params
        term1 = bounded(x, x[0], x_start, y_start - f_0)
        term2 = bounded(x, x_start, x_end, y_end - y_start)
        return f_0 + term1 + term2

    # ----------------- OBJECTIVE FUNCTION -----------------
    def objective(self, params):
        """Sum of squared errors between model and data"""
        y_pred = self.model_fun(self.xdata, params)
        return np.sum((y_pred - self.ydata) ** 2)

    # ----------------- FIT FUNCTION -----------------
    def fit(self, method="SLSQP", options=None):
        """Fit model parameters using constrained optimization"""
        if options is None:
            options = {"maxiter": 10000, "ftol": 1e-8, "disp": True}

        # Linear constraints: A @ x <= b
        lin_con = {"type": "ineq", "fun": lambda P: self.b - self.A @ P}

        res = minimize(
            self.objective,
            self.x0,
            bounds=list(zip(self.lb, self.ub)),
            constraints=[lin_con],
            method=method,
            options=options,
        )
        return res
