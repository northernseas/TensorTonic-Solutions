import numpy as np

def second_order_partials(x, y):
    """
    Returns: dict with 'd2f_dx2' and 'd2f_dy2' (floats), each rounded to 6 decimals
    """
    d2f_dx2 = -y ** 2 * np.sin(x * y)
    d2f_dy2 = -x ** 2 * np.sin(x * y)

    return {
        "d2f_dx2": round(d2f_dx2.item(), 6),
        "d2f_dy2": round(d2f_dy2.item(), 6),
    }