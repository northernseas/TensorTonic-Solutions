import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape
    
    w = np.zeros(D)
    b = 0

    for _ in range(epochs):
        y_hat = X @ w + b

        loss = np.mean((y_hat - y) ** 2)

        dl_dw = 2 / N * X.T @ (y_hat - y)
        dl_db = 2 / N * np.sum(y_hat - y)

        w = w - lr * dl_dw
        b = b - lr * dl_db
    
    return w, b