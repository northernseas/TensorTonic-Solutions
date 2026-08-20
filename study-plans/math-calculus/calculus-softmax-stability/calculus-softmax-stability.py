import numpy as np

def softmax_stability_analysis(z):
    """
    Returns: dict with 'naive', 'stable' (lists of floats) and 'naive_has_issues' (bool)
    """
    z = np.array(z, dtype=np.float64)

    naive = np.exp(z) / np.sum(np.exp(z))
    
    naive_has_issues = np.any(np.isnan(naive)) or np.any(np.isinf(naive))

    stable = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))

    return {
        "naive": naive.tolist(),
        "stable": stable.tolist(),
        "naive_has_issues": bool(naive_has_issues)
    }