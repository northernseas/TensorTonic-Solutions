import numpy as np

def loss_curvature_analysis(y_hat, y, delta):
    """
    Returns: dict with 'mse', 'ce', 'huber' keys, each containing 'dL' and 'd2L' lists
    """
    y_hat = np.asarray(y_hat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # MSE
    mse_dL = 2 * (y_hat - y)
    mse_d2L = np.full_like(y_hat, 2)

    # BCE
    ce_dL = -y / y_hat + (1 - y) / (1 - y_hat)
    ce_d2L = y / (y_hat ** 2) + (1 - y) / (1 - y_hat) ** 2

    # Huber
    tmp = np.abs(y_hat - y) <= delta
    huber_dL = np.where(tmp, y_hat - y, delta * np.sign(y_hat - y))
    huber_d2L = np.where(tmp, 1, 0)
    
    return {
        "mse": {"dL": mse_dL.tolist(), "d2L": mse_d2L.tolist()},
        "ce": {"dL": ce_dL.tolist(), "d2L": ce_d2L.tolist()},
        "huber": {"dL": huber_dL.tolist(), "d2L": huber_d2L.tolist()},
    }