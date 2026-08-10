import numpy as np

def skewness_kurtosis(data):
    """
    Returns: dict with 'skewness', 'kurtosis', and interpretation strings.
    """
    x = np.array(data, dtype=np.float64)
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)

    skew = n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3)

    kurt_prefix = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    kurt_suffix = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    kurt = kurt_prefix * np.sum(((x - m) / s) ** 4) - kurt_suffix

    skew = round(skew, 4)
    kurt = round(kurt, 4)
    
    if skew > 0.5:
        skew_interp = "right-skewed"
    elif skew < -0.5:
        skew_interp = "left-skewed"
    else:
        skew_interp = "approximately symmetric"

    if kurt > 1:
        kurt_interp = "leptokurtic"
    elif kurt < -1:
        kurt_interp = "platykurtic"
    else:
        kurt_interp = "mesokurtic"

    return {
        "skewness": skew,
        "kurtosis": kurt,
        "skew_interpretation": skew_interp,
        "kurtosis_interpretation": kurt_interp,
    }