import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    N, D = X.shape

    w = np.zeros(D)
    b = 0
    
    for _ in range(epochs):
        for i in range(N):
            z = w @ X[i] + b
            y_hat = 1 if z >= 0 else 0
            w = w + lr * (y[i] - y_hat) * X[i]
            b = b + lr * (y[i] - y_hat)

    return w, b