import numpy as np

def kfold_cv(X, y, model_fn, k=5, seed=42):
    """
    Returns: tuple of (list of per-fold accuracies, mean accuracy)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    indices = np.arange(len(X))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    folds = np.array_split(indices, k)

    acc = []
    
    for fold in folds:
        train_fold_idx = [i for i in indices if i not in fold]
        X_train = X[train_fold_idx]
        y_train = y[train_fold_idx]
        X_val = X[fold]
        y_val = y[fold]

        y_val_pred = model_fn(X_train, y_train)(X_val)

        acc_ = np.mean(y_val == y_val_pred)

        acc.append(round(acc_, 4))

    acc = np.array(acc)

    return acc, round(acc.mean(), 4)