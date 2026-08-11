from math import factorial, exp

def poisson_distribution(lam, max_k):
    """
    Returns: [pmf_list, cdf_at_max_k, p_zero] as a list.
    """
    pmf = [
        round((lam ** k * exp(-lam)) / factorial(k), 4)
        for k in range(max_k + 1)
    ]
    cdf_at_max_k = round(sum(pmf), 4)
    p_zero = pmf[0]
    return [pmf, cdf_at_max_k, p_zero]