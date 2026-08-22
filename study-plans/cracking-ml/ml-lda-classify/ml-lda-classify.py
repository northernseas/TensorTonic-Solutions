import numpy as np

def lda_classify(X_train, y_train, X_test):
    """
    Returns: list of predicted class labels for each test point
    """

    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape
    
    classes = []

    cov = np.zeros((D, D))
    
    for c in np.unique(y_train):
        X_c = X_train[y_train == c]
        prior = len(X_c) / N
        mean = X_c.mean(axis=0)
        cov += (X_c - mean).T @ (X_c - mean)
        classes.append((c, prior, mean))

    cov = cov / (N - len(classes))
    cov += np.eye(D) * 1e-6

    cov_inv = np.linalg.inv(cov)


    # W: (D, K)   b: (K)    scores: (M, K)

    w = np.stack([
        cov_inv @ mean
        for _, prior, mean in classes
    ])

    b = np.array([
        -0.5 * mean @ cov_inv @ mean + np.log(prior)
        for _, prior, mean in classes
    ])
    
    scores = X_test @ w.T + b
    preds = np.argmax(scores, axis=-1)
    labels = np.array([c for c, _, _ in classes])
    return labels[preds]
    
    # preds = []
    # for x_test in X_test:
    #     max_f = float("-inf")
    #     pred = None
    #     for c, prior, mean in classes:
    #         f = (
    #             x_test @ cov_inv @ mean
    #             - 0.5 * mean @ cov_inv @ mean
    #             + np.log(prior)
    #         )

    #         if f > max_f:
    #             max_f = f
    #             pred = c
    #     preds.append(pred)

    return preds