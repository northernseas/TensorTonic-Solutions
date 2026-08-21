import numpy as np

def learning_rate_sweep(X, y, learning_rates, n_epochs):
    """Train a linear model with each learning rate and return loss curves.

    Returns: see problem description for expected output format
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape
    
    all_losses = []

    for lr in learning_rates:
        w = np.zeros(D)
        losses = []
        for _ in range(n_epochs):
            loss = np.mean((X @ w - y) ** 2)
            losses.append(loss)
            grad = 2 / N * X.T @ (X @ w - y)
            w = w - lr * grad
        all_losses.append(losses)

    return all_losses