import numpy as np

def vector_valued_hessian(x, y, z):
    """
    Returns: 3x3x3 list of lists of lists, each entry rounded to 6 decimals
    """
    H1 = [
        [2 * y, 2 * x, 0],
        [2 * x, 0, 0],
        [0, 0, 0],
    ]
    H2 = [
        [0, 0, 0],
        [0, 2 * z, 2 * y],
        [0, 2 * y, 0],
    ]
    H3 = [
        [0, 0, 2 * z],
        [0, 0, 0],
        [2 * z, 0, 2 * x],
    ]
    return [H1, H2, H3]