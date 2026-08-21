import numpy as np

def vanilla_gradient_descent(x0, y0, lr, n_iters):
    """
    Returns: dict with 'trajectory' (list of [x,y] pairs), 'final_point' ([x,y]), 'final_value' (float)
    """
    x = x0
    y = y0
    trajectory = [(x, y)]

    for _ in range(n_iters):
        df_dx = 2 * x
        df_dy = 6 * y
        x = x - lr * df_dx
        y = y - lr * df_dy
        trajectory.append((x, y))

    final_point = (x, y)
    final_value = x ** 2 + 3 * y ** 2
    
    return {
        "trajectory": trajectory,
        "final_point": final_point,
        "final_value": final_value,
    }