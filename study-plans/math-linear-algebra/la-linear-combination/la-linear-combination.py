import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors = np.array(vectors, dtype=np.float64)
    coefficients = np.array(coefficients, dtype=np.float64)
    
    return (vectors * coefficients[:, None]).sum(axis=0)