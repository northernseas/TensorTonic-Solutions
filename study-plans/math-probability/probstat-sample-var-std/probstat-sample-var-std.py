import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x, dtype=np.float64)

    var = ((x - x.mean()) ** 2).sum() / (len(x) - 1)
    std = np.sqrt(var)

    return { "variance": var, "std_dev": std }