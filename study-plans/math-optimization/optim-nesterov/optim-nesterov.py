import numpy as np

def nesterov_momentum(X, y, lr, beta, n_epochs):
    """
    Returns: tuple of (classical_losses, nesterov_losses), each a list of MSE values
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w_1 = np.zeros(D)
    w_2 = np.zeros(D)
    v_1 = np.zeros(D)
    v_2 = np.zeros(D)

    classical_losses = []
    nesterov_losses = []
    
    for _ in range(n_epochs):
        # Compute losses
        loss_1 = np.mean((X @ w_1 - y) ** 2)
        loss_2 = np.mean((X @ w_2 - y) ** 2)
        classical_losses.append(loss_1)
        nesterov_losses.append(loss_2)

        # Compute grad
        grad = lambda w: 2 / N * X.T @ (X @ w - y)
        
        # Classical Momentum
        v_1 = beta * v_1 + grad(w_1)
        w_1 = w_1 - lr * v_1

        # Nesterov Momentum
        v_2 = beta * v_2 + grad(w_2 - lr * beta * v_2)
        w_2 = w_2 - lr * v_2
    
    return classical_losses, nesterov_losses
