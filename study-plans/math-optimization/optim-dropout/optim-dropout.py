import numpy as np

def dropout_mlp(X_train, y_train, X_test, y_test, hidden_size, lr, dropout_rate, n_epochs, seed):
    """
    Returns: list of test accuracies, one per epoch.
    """
    np.random.seed(seed)
    
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64).reshape(-1, 1)
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64).reshape(-1, 1)
    
    N, input_size = X_train.shape

    # He/Kaiming initialization for weights
    w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
    b2 = np.zeros(1)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    relu = lambda x: np.maximum(x, 0)

    accuracies = []
    
    for _ in range(n_epochs):
        # Forward
        z1 = X_train @ w1 + b1
        a1 = relu(z1)
        mask = np.random.rand(*a1.shape) > dropout_rate
        a1_dropout = a1 * mask / (1 - dropout_rate)
        z2 = a1_dropout @ w2 + b2
        y = sigmoid(z2)

        # Loss
        loss = -np.mean(y_train * np.log(y) + (1 - y_train) * np.log(1 - y))

        # Backward
        dL_dy = -1 / N * ((y_train / y) - (1 - y_train) / (1 - y)) # (N, 1)
        dy_dz2 = y * (1 - y) # (N, 1)
        
        dz2 = dL_dy * dy_dz2 # (N, 1)
        dw2 = a1_dropout.T @ dz2 # (H, 1)
        db2 = np.sum(dz2, axis=0) # (1)

        da1d = dz2 @ w2.T # (N, H)
        da1 = da1d * mask / (1 - dropout_rate) # (N, H)

        dz1 = da1 * np.where(z1 > 0, 1, 0) # (N, H)
        dw1 = X_train.T @ dz1 # (D, H)
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1
        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2

        # Test accuracy
        y = sigmoid(relu(X_test @ w1 + b1) @ w2 + b2)
        preds = y >= 0.5
        accuracy = np.mean(preds == y_test)
        accuracies.append(accuracy)

    return accuracies