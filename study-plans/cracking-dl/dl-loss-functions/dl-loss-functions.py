import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if loss_type == "mse":
        loss = np.mean((y_true - y_pred) ** 2)

    elif loss_type == "bce":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    elif loss_type == "cce":
        h = y_pred - y_pred.max(axis=-1, keepdims=True)
        log_y_pred = h - np.log(np.sum(np.exp(h), axis=-1, keepdims=True)) # (N, C)
        loss = -np.mean(log_y_pred[range(len(y_pred)), y_true])

    elif loss_type == "hinge":
        loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    return np.round(loss, 4)