import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    N, D = X.shape
    
    w = np.zeros(D)
    b = 0

    for _ in range(n_iters):
        z = X @ w + b
        y_hat = sigmoid(z)

        loss = -1 / N * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # dl_dyhat = -1 / N * (y / y_hat - (1 - y) / (1 - y_hat))
        # dyhat_dz = y_hat * (1 - y_hat)
        # dl_dz = dl_dyhat * dyhat_dz = (y_hat - y) / N
        dl_dz = (y_hat - y) / N   # (N)

        dl_dw = X.T @ dl_dz    # (D, N) @ (N)
        dl_db = np.sum(dl_dz)

        w -= lr * dl_dw
        b -= lr * dl_db

    return w, b