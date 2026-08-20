import numpy as np

def sigmoid_squeeze_analysis(x):
    """
    Returns: dict with 'bounds' (list of [lower, sigmoid, upper] triples) and 'is_saturated' (list of bools)
    """
    x = np.array(x, dtype=np.float64)

    sigma = 1 / (1 + np.exp(-x))

    lower = np.maximum(0, 1 - np.exp(-x))
    upper = np.minimum(1, np.exp(x))

    bounds = np.stack((lower, sigma, upper), axis=-1).tolist()

    is_saturated = (np.minimum(sigma, 1 - sigma) < 1e-4).tolist()
    
    return {
        "bounds": bounds,
        "is_saturated": is_saturated,
    }
