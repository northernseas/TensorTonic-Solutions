def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    """
    Returns: float, the posterior probability P(A|B).
    """
    # P(A|B) = P(B|A)P(A) / P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    return p_b_given_a * p_a / p_b