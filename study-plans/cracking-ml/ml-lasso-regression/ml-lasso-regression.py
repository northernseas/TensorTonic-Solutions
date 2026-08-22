def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D)
    b = 0

    for _ in range(epochs):
        y_hat = X @ w + b

        loss = np.mean((y_hat - y) ** 2) + alpha * np.sum(np.abs(w))

        dw = 2 / N * X.T @ (y_hat - y)
        db = 2 / N * np.sum(y_hat - y)

        dw += alpha * np.sign(w)
        
        w -= lr * dw
        b -= lr * db
    
    return w, b