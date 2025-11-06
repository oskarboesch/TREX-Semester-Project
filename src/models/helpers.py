import numpy as np
def step(x, a):
    """
    MATLAB equivalent:
        y = double(x >= a)
    Returns 1 where (x - a) >= 0, else 0.
    """
    return (x >= a).astype(float)

def positive(x, a):
    """
    MATLAB equivalent:
        y = step(x, a) * (x - a)
    Returns (x - a) when (x - a) >= 0, else 0.
    """
    return step(x, a) * (x - a)

def bounded(x, a, b, scale=1.0):
    """
    MATLAB equivalent:
        if b > a:
            y = positive(x, a) - positive(x, b)
            if nargin > 3:
                y = y * scale / (b - a)
        else:
            y = step(x, a)
            if nargin > 3:
                y = y * scale
    """
    if b > a:
        y = positive(x, a) - positive(x, b)
        y = y * scale / (b - a)
    else:
        y = step(x, a) * scale
    return y

def create_model_params(
    margin=(0, 0),
    dx_min=1e-6,
    dx_max=10,
    dx_max_start=5,
    dx_max_end=10,
    dx_rotate_max=5,
    dx_bend_max=10
):
    """
    Initialize the parameters for the models.

    Parameters
    ----------
    margin : tuple of 2 floats
        Length during which the signal is considered stationary, before and after the clot.
    dx_min : float
        Minimum duration for a smooth signal increase.
    dx_max : float
        Maximum duration for smooth transitions.
    dx_max_start : float
        Maximum duration for transition when entering the clot.
    dx_max_end : float
        Maximum duration for transition when exiting the clot.
    dx_rotate_max : float
        Maximum duration for which the guide is rotating.
    dx_bend_max : float
        Maximum duration for which the guide is bending at the tip.

    Returns
    -------
    model_params : dict
        Dictionary containing all model parameters.
    """

    # Validate inputs
    if not (isinstance(margin, (list, tuple)) and len(margin) == 2 and all(isinstance(x, (int, float)) for x in margin)):
        raise ValueError("margin must be a list or tuple of two numbers")
    
    for param_name, param_value in [
        ("dx_min", dx_min),
        ("dx_max", dx_max),
        ("dx_max_start", dx_max_start),
        ("dx_max_end", dx_max_end),
        ("dx_rotate_max", dx_rotate_max),
        ("dx_bend_max", dx_bend_max)
    ]:
        if not (isinstance(param_value, (int, float)) and param_value > 0):
            raise ValueError(f"{param_name} must be a positive number")
    
    model_params = {
        "margin": tuple(margin),
        "dx_min": dx_min,
        "dx_max": dx_max,
        "dx_max_start": dx_max_start,
        "dx_max_end": dx_max_end,
        "dx_rotate_max": dx_rotate_max,
        "dx_bend_max": dx_bend_max
    }

    return model_params
