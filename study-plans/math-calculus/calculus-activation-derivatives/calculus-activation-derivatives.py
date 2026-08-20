import numpy as np

def activation_derivative(name, x):
    """
    Returns: list of floats (the derivative evaluated at each x)
    """
    x = np.array(x, dtype=np.float64)

    if name == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return (s * (1 - s)).tolist()

    if name == "tanh":
        t = np.tanh(x)
        return (1 - t ** 2).tolist()

    if name == "relu":
        return np.where(x > 0, 1, 0).tolist()

    if name == "swish":
        s = 1 / (1 + np.exp(-x))
        return (s + x * s * (1 - s)).tolist()