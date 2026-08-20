import numpy as np

def backprop_graph(w1, w2, x):
    """
    Returns: dict with 'forward' (dict of intermediate values) and 'gradients' (dict with d_w1, d_w2, d_x)
    """
    # Forward
    x_sq = x ** 2
    prod1 = w1 * x
    prod2 = w2 * x_sq
    sum_val = prod1 + prod2
    output = np.maximum(0, sum_val)

    # Backward
    d_output = 1
    d_sum = d_output * np.where(sum_val > 0, 1, 0)
    d_prod1 = d_sum
    d_prod2 = d_sum
    d_w1 = d_prod1 * x
    d_x = d_prod1 * w1
    d_w2 = d_prod2 * x_sq
    d_x += d_prod2 * 2 * x * w2

    return {
        "forward": {
            "x_sq": x_sq,
            "prod1": prod1,
            "prod2": prod2,
            "sum_val": sum_val,
            "output": output,
        },
        "gradients": {
            "d_w1": d_w1,
            "d_w2": d_w2,
            "d_x": d_x,
        }
    }