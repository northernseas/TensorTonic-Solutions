import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape
    M, _ = X_test.shape

    # Distances: (M, 1, D) - (1, N, D) = (M, N, D) -> (M, N)
    d = np.sqrt(
        np.sum(
            (X_test[:, None, :] - X_train[None, :, :]) ** 2,
            axis=-1
        )
    )

    # k nearest neighbors
    topk = np.argsort(d, axis=-1)[:, :k] # (M, 3)

    # get labels
    topk_y = y_train[topk] # (M, 3)

    # majority vote
    classes = np.unique(y_train) # (C)

    # topk_y[:, :, None] # (M, 3, 1)
    # classes[None, None, :] # (1, 1, C)
    counts = topk_y[:, :, None] == classes[None, None, :] # (M, 3, C)
    counts = np.sum(counts, axis=1) # (M, C)
    
    y_pred = classes[np.argmax(counts, axis=-1)]
    
    return y_pred