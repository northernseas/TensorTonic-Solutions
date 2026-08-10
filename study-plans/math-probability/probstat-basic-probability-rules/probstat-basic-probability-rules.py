def basic_probability(p_a, p_b, p_a_and_b):
    """
    Returns: [p_union, p_a_complement, p_b_complement, p_a_and_not_b] as a list.
    """
    return [
        round(p_a + p_b - p_a_and_b, 4),
        round(1 - p_a, 4),
        round(1 - p_b, 4),
        round(p_a - p_a_and_b, 4)
    ]