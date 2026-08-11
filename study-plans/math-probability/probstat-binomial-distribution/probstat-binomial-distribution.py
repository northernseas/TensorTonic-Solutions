from math import comb

def binomial_distribution(n, p, threshold):
    """
    Returns: dict with 'pmf' (list), 'mean', 'variance', 'tail_prob' as floats.
    """
    pmf = [
        round(comb(n, k) * p ** k * (1 - p) ** (n - k), 4)
        for k in range(n + 1)
    ]

    mean = round(n * p, 4)
    var = round(n * p * (1 - p), 4)
    prob_at_least = round(sum(pmf[threshold:]), 4)
    
    return {
        "pmf": pmf,
        "mean": mean,
        "variance": var,
        "prob_at_least": prob_at_least
    }