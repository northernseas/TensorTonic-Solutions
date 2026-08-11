import numpy as np

def standard_errors(samples):
    """
    Returns: dict with 'standard_errors' (list of floats) and 'comparison'.
    """
    standard_errors = [
        round(np.std(sample, ddof=1).item() / len(sample) ** 0.5, 4)
        for sample in samples
    ]
    
    mean_se = round(np.mean(standard_errors).item(), 4)

    return {
        "standard_errors": standard_errors,
        "mean_se": mean_se
    }