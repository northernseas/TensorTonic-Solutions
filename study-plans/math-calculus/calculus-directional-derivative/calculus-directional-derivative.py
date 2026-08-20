import numpy as np

def directional_derivative_analysis(w, v):
    """
    Returns: dict with 'gradient' (list), 'directional_derivative' (float), 'steepest_descent_direction' (list), 'max_descent_rate' (float)
    """
    w = np.array(w, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    grad = np.array([2 * w[0], 6 * w[1]])

    directional_derivative = grad @ v

    grad_norm = np.linalg.norm(grad)
    if grad_norm > 0:
        steepest_descent_direction = (-grad / grad_norm).tolist()
        max_descent_rate = -grad_norm
    else:
        steepest_descent_direction = [0, 0]
        max_descent_rate = 0
    
    return {
        "gradient": grad,
        "directional_derivative": directional_derivative,
        "steepest_descent_direction": steepest_descent_direction,
        "max_descent_rate": max_descent_rate
    }