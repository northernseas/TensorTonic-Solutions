import numpy as np

def svm_sgd(X_train, y_train, X_test, lr=0.01, lam=0.01, n_epochs=100):
    """
    Train an SVM using SGD on hinge loss with L2 regularization.

    Parameters:
    - X_train: Training feature matrix (n samples, d features)
    - y_train: Training labels (-1 or +1)
    - X_test: Test feature matrix
    - lr: Learning rate
    - lam: L2 regularization strength
    - n_epochs: Number of training epochs

    Returns: list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape

    w = np.zeros(D)
    b = 0

    for _ in range(n_epochs):
        for i in range(N):
            x, y = X_train[i], y_train[i]
            margin = y * (w @ x + b)
            if margin < 1:
                w = w - lr * (lam * w - y * x)
                b = b + lr * y
            else:
                w = w - lr * lam * w

    preds = []
    for x_test in X_test:
        score = w @ x_test + b
        preds.append(1 if score > 0 else -1)

    return preds