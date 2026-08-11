import numpy as np
from scipy import stats

def clt_confidence_interval(data, confidence):
    """
    Returns: [mean, std_error, ci_lower, ci_upper] as a list.
    """
    mean = round(np.mean(data).item(), 4)
    std = round(np.std(data, ddof=1).item() / (len(data) ** 0.5), 4)

    z = round(float(stats.norm.ppf((1 + confidence) / 2)), 4)
    
    ci_lower = round(mean - z * std, 4)
    ci_upper = round(mean + z * std, 4)

    return [mean, std, ci_lower, ci_upper]