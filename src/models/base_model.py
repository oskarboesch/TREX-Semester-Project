from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
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
            start_accuracy = self.evaluate_accuracy(all_params[:n_vars], start=True)
            end_accuracy = self.evaluate_accuracy(all_params[:n_vars], start=False)
            wandb.log({"train_sse": sse, "train_start_accuracy": start_accuracy, "train_end_accuracy": end_accuracy })

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

    def evaluate_accuracy(self, parameters, start=True):
        """ Evaluate accuracy from Start and End points for all logs in list of logs. Accuracy is defined as +- 1.5 of labels for x_start and x_end """
        accuracies = []
        for labels in self.labels:
            if start:
                label = labels['Start'].values[0]
                x_param = parameters[0]
            else:
                label = labels['End'].values[0]
                x_param = parameters[1]
            accuracy = abs(x_param - label) <= 1.5
            accuracies.append(accuracy)
        return np.mean(accuracies)