from math import exp

def exponential_distribution(lam, t):
    """
    Returns: dict with 'pdf', 'cdf', 'survival', 'mean', 'variance' as floats.
    """
    pdf = lam * exp(-lam * t)
    cdf = 1 - exp(-lam * t)
    s = exp(-lam * t)
    mu = 1 / lam
    var = 1 / (lam ** 2)
    return {
        "pdf": round(pdf, 4),
        "cdf": round(cdf, 4),
        "survival": round(s, 4),
        "mean": round(mu, 4),
        "variance": round(var, 4),
    }