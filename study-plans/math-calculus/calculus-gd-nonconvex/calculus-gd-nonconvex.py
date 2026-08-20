import numpy as np

def gradient_descent_nonconvex(w0, lr, n_iters):
    """
    Returns: dict with 'critical_points' (sorted list), 'classifications' (list of strings), 'trajectory' (list of n_iters+1 floats)
    """
    critical_points = [-np.sqrt(1.5), 0, np.sqrt(1.5)]

    classifications = []
    for cp in critical_points:
        lpp = 12 * cp ** 2 - 6
        if lpp > 0: classifications.append("local_min")
        else: classifications.append("local_max")
    
    w = w0
    trajectory = [w]
    for _ in range(n_iters):
        # dL/dw = 4 * w ** 3 - 6 * w
        # d2L/dw2 = 12 * w ** 2 - 6
        g = 4 * w ** 3 - 6 * w
        w = w - lr * g
        trajectory.append(w)

    return {
        "critical_points": critical_points,
        "classifications": classifications,
        "trajectory": trajectory,
    }