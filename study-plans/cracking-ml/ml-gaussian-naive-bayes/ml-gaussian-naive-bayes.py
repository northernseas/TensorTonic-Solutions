import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape

    classes = []

    for c in np.unique(y_train):
        X_c = X_train[y_train == c]
        prior = len(X_c) / N
        mean = X_c.mean(axis=0) # (D)
        var = X_c.var(axis=0) + 1e-9 # (D)
        classes.append((c, prior, mean, var))

    y_test = []
    for x_test in X_test:
        max_log_p = float("-inf")
        pred = None
        for c, prior, mean, var in classes:
            # p(c|x) = prod p(x_i|c) * p(c)
            # log p(c|x) = log(p(c)) + sum(log(p(x_i|c)))

            p = - 1 / 2 * np.log(2 * np.pi * var) - ((x_test - mean) ** 2) / (2 * var)
            log_p = np.log(prior) + np.sum(p)

            if log_p > max_log_p:
                max_log_p = log_p
                pred = c
        
        y_test.append(pred)

    return y_test