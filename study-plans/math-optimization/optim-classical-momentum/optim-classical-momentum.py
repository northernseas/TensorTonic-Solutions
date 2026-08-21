import numpy as np

def momentum_gd(X, y, lr, beta, n_epochs):
    """
    Returns: tuple of (vanilla_losses, momentum_losses), each a list of MSE values
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w_1 = np.zeros(D)
    w_2 = np.zeros(D)
    v = np.zeros(D)

    vanilla_losses = []
    momentum_losses = []
    
    for _ in range(n_epochs):
        # Compute losses
        loss_1 = np.mean((X @ w_1 - y) ** 2)
        loss_2 = np.mean((X @ w_2 - y) ** 2)
        vanilla_losses.append(loss_1)
        momentum_losses.append(loss_2)
        
        # Vanilla GD
        grad_1 = 2 / N * X.T @ (X @ w_1 - y)
        w_1 = w_1 - lr * grad_1

        # Momentum GD
        grad_2 = 2 / N * X.T @ (X @ w_2 - y)
        v = beta * v + grad_2
        w_2 = w_2 - lr * v
    
    return vanilla_losses, momentum_losses