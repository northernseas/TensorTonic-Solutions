import numpy as np

def jacobian_3d(x, y, z):
    """
    Returns: 3x3 list of lists, each entry rounded to 6 decimals
    """
    return [[2 * x * y, x ** 2, 0],
            [0, 2 * y, 1],
            [z ** 2, 0, 2 * z * x]]