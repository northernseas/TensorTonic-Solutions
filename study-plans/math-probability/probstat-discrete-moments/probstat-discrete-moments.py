def discrete_moments(values, probabilities):
    """
    Returns: [E_X, E_X2, variance, std_dev] as a list.
    """
    e = round(sum((v * p for v, p in zip(values, probabilities))), 4)
    e2 = round(sum(((v ** 2) * p for v, p in zip(values, probabilities))), 4)
    var = round(e2 - e ** 2, 4)
    std = round(var ** 0.5, 4)
    return [e, e2, var, std]