import numpy as np

def taylor_approximation(w0, n_points=100):
    """
    Returns: dict with 'w_values', 'true_loss', 'linear_approx', 'quadratic_approx' (lists) and 'L_at_w0', 'dL_at_w0', 'd2L_at_w0' (floats)
    """
    l = np.sin(w0) + 0.1 * w0 ** 2
    lp = np.cos(w0) + 0.2 * w0
    lpp = -np.sin(w0) + 0.2

    w = np.linspace(w0 - 2, w0 + 2, n_points)

    t = np.sin(w) + 0.1 * w ** 2
    t1 = l + lp * (w - w0)
    t2 = l + lp * (w - w0) + 0.5 * lpp * (w - w0) ** 2

    return {
        "w_values": w,
        "true_loss": t,
        "linear_approx": t1,
        "quadratic_approx": t2,
        "L_at_w0": l,
        "dL_at_w0": lp,
        "d2L_at_w0": lpp,
    }