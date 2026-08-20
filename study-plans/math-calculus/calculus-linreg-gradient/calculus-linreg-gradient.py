import numpy as np

def linear_regression_gradient(X, y, w):
    """
    Returns: dict with 'loss' (float), 'analytical_gradient', 'numerical_gradient' (lists of floats)
    """
    X = np.array(X, dtype=np.float64) # (N, D)
    y = np.array(y, dtype=np.float64) # (N,)
    w = np.array(w, dtype=np.float64) # (D,)

    def l(X, y, w):
        return np.sum((X @ w - y) ** 2)

    def l_batch(X, y, W):
        return np.sum((X @ W.T - y[:, None]) ** 2, axis=0)

    analytical_gradient = 2 * X.T @ (X @ w - y)

    h = 1e-5
    W_plus = w + h * np.eye(len(w))
    W_minus = w - h * np.eye(len(w))
    numerical_gradient = (l_batch(X, y, W_plus) - l_batch(X, y, W_minus)) / (2 * h)

    return {
        "loss": l(X, y, w),
        "analytical_gradient": analytical_gradient,
        "numerical_gradient": numerical_gradient,
    }