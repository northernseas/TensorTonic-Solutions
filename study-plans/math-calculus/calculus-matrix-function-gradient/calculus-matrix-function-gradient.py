import numpy as np

def matrix_polynomial_gradient(x):
    """
    Returns: 3x3 list of lists, each entry rounded to 6 decimals
    """
    return [[1, 2 * x, 3 * x ** 2],
            [2 * x, 3 * x ** 2, 4 * x ** 3],
            [3 * x ** 2, 4 * x ** 3, 5 * x ** 4]]