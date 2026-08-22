import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    """
    Returns: A list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape
    
    w = np.zeros(D)
    b = 0

    w_accum = np.zeros(D)
    b_accum = 0
    
    for _ in range(n_epochs):
        for i in range(N):
            x, y = X_train[i], y_train[i]
            if y * (x @ w + b) <= 0:
                w += y * x
                b += y
            w_accum += w
            b_accum += b

    w_accum = w_accum / (N * n_epochs)
    b_accum = b_accum / (N * n_epochs)

    preds = []

    for x_test in X_test:
        score = x_test @ w_accum + b_accum
        preds.append(1 if score > 0 else -1)
    
    return preds