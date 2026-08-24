import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Returns: dict with keys "mse", "mae", "r2" rounded to 4 decimal places
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    r = y_true - y_pred

    mse = np.mean(r ** 2)

    mae = np.mean(np.abs(r))

    r2_a = np.sum(r ** 2)
    r2_b = np.sum((y_true - y_true.mean()) ** 2)
    r2 = (1 - r2_a / r2_b) if r2_b != 0 else 0.0

    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2":  round(r2, 4),
    }