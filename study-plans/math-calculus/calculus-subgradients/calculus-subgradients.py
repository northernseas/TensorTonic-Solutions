import numpy as np

def subgradient_analysis(x_points, w_init, lr, n_iters):
    """
    Returns: dict with 'abs_subgrad', 'relu_subgrad', 'w_trajectory' (lists) and 'w_final' (float)
    """
    x = np.array(x_points, dtype=np.float64)

    abs_subgrad = np.sign(x).tolist()
    relu_subgrad = np.where(x > 0, 1, 0).tolist()

    w = w_init
    w_trajectory = [w]
    for _ in range(n_iters):
        grad = np.sign(w - 3) + w
        w = w - lr * grad
        w_trajectory.append(w)

    return {
        "abs_subgrad": abs_subgrad,
        "relu_subgrad": relu_subgrad,
        "w_trajectory": w_trajectory,
        "w_final": w
    }