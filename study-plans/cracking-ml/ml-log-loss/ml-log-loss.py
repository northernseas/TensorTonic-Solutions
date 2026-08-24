import numpy as np

def log_loss(y_true, y_pred):
    """
    Returns: float
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss