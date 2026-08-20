import numpy as np

def activation_continuity_analysis(x):
    """
    Returns: dict mapping 'relu', 'leaky_relu', 'gelu' to lists of non-differentiable x values
    """
    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x):
        return np.where(x >= 0, x, 0.01 * x)

    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    x = np.array(x, dtype=np.float64)
    h = 1e-7
    tol = 1e-5
    result = {}

    for name, fn in [
        ("relu", relu),
        ("leaky_relu", leaky_relu),
        ("gelu", gelu)
    ]:
        left_d = (fn(x) - fn(x - h)) / h
        right_d = (fn(x + h) - fn(x)) / h
        diff = np.abs(left_d - right_d) >= tol
        result[name] = x[diff].tolist()
    
    return result