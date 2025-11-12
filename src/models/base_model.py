from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.linalg import block_diag
from models.helpers import step, bounded
import wandb


class BaseModel(ABC):
    def __init__(self, name, model_params, xdata, ydata, labels, log=False):
        self.name = name
        self.model_params = model_params
        self.log = log

        # Ensure xdata and ydata are lists (multiple curves)
        if not isinstance(xdata, list):
            xdata = [xdata]
        if not isinstance(ydata, list):
            ydata = [ydata]
        if not isinstance(labels, list):
            labels = [labels]

        self.n_curves = len(xdata)
        self.xdata = xdata
        self.ydata = ydata
        self.labels = labels

        # Compute bounds across all curves
        self.x_min = min([np.min(x) for x in xdata])
        self.x_max = max([np.max(x) for x in xdata])
        self.y_min = min([np.min(y) for y in ydata])
        self.y_max = max([np.max(y) for y in ydata])
        self.y_range = self.y_max - self.y_min
        self.x_range = self.x_max - self.x_min

        # force herited classes to define these
        self.lb_curve = None
        self.ub_curve = None
        self.x0 = None

    @abstractmethod
    def model_fun(self, x, params):
        """ Model function to be implemented in child classes """
        pass

    def objective(self, all_params):
        """Vectorized sum-of-squares across all curves"""
        sse = 0
        start_accuracies = []
        end_accuracies = []
        n_vars = len(self.lb_curve)
        for i in range(self.n_curves):
            idx = slice(i * n_vars, (i + 1) * n_vars)
            sse += np.sum((self.model_fun(self.xdata[i], all_params[idx]) - self.ydata[i])**2)
            start_accuracies.append(self.evaluate_accuracy(all_params[idx], self.labels[i], start=True))
            end_accuracies.append(self.evaluate_accuracy(all_params[idx], self.labels[i], start=False))
            if self.log:
                if i == 0:
                    param_dict = {}
                    for name, val in zip(self.x_names, all_params[idx]):
                        param_dict[f"{name}_{i}"] = val
                        wandb.log({**param_dict})
        if self.log:
            wandb.log({"train_sse": sse, "train_start_accuracy": np.mean(start_accuracies), "train_end_accuracy": np.mean(end_accuracies)})

        return sse

    def fit(self, method="SLSQP", options=None, global_search=False, niter=100, T=1.0, seed=None):
        """
        Fit model parameters using local or global optimization.

        Parameters
        ----------
        method : str
            Local optimization method (default: "SLSQP").
        options : dict
            Options for the local optimizer.
        global_search : bool
            If True, uses basinhopping global optimization.
        niter : int
            Number of basinhopping iterations.
        T : float
            Temperature parameter for basinhopping acceptance.
        seed : int or None
            Random seed for reproducibility.
        """
        if options is None:
            options = {"maxiter": 10000, "ftol": 1e-8, "disp": True}

        bounds = list(zip(self.lb, self.ub))
        x0 = self.x0

        if global_search:
            # Define local minimizer arguments
            minimizer_kwargs = {
                "method": method,
                "bounds": bounds,
                "options": options,
            }

            # Run basinhopping
            res = basinhopping(
                self.objective,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=niter,
                T=T,
                seed=seed,
                disp=True
            )
        else:
            # Standard local optimization
            res = minimize(
                self.objective,
                x0,
                method=method,
                bounds=bounds,
                options=options
            )

        return res

    def evaluate_accuracy(self, parameters, labels, start=True):
        """ Evaluate accuracy from Start and End points for all logs in list of logs. Accuracy is defined as +- 1.5 of labels for x_start and x_end """
        # check if parameters is for multiple curves
        if start:
            label = labels['Start'].values[0]
            x_param = parameters[0]
        else:
            label = labels['End'].values[0]
            x_param = parameters[1]
        accuracy = abs(x_param - label) <= 1.5
        return accuracy