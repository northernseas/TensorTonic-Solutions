import numpy as np

def categorical_encode(data, method="label"):
    """
    Returns: encoded result based on method
    """
    if method == "label":
        classes, labels = np.unique(data, return_inverse=True)
        return {
            "encoded": labels.tolist(),
            "classes": classes.tolist()
        }
    elif method == "onehot":
        _, labels = np.unique(data, return_inverse=True)
        nb_samples = len(data)
        nb_classes = np.max(labels) + 1
        res = np.zeros((nb_samples, nb_classes), dtype=int)
        np.add.at(res, (range(nb_samples), labels), 1)
        return res.tolist()
    else:
        raise ValueError()