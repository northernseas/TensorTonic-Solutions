import numpy as np

FUNCTIONS = {
    "sin": np.sin,
    "exp": np.exp,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "x_cubed": lambda z: z ** 3,
}

def gradient_check(f_name, x, analytical_deriv):
    """
    Returns: dict with 'h_values', 'forward_errors', 'backward_errors', 'central_errors' (lists) and 'optimal_h_forward', 'optimal_h_backward', 'optimal_h_central' (floats)
    """
    fn = FUNCTIONS[f_name]
    x = np.array(x, dtype=np.float64)
    h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    forward_errors, backward_errors, central_errors = [], [], []
    
    for h in h_values:
        fwd = (fn(x + h) - fn(x)) / h
        bwd = (fn(x) - fn(x - h)) / h
        ctr = (fn(x + h) - fn(x - h)) / (2 * h)
        forward_errors.append(float(abs(fwd - analytical_deriv)))
        backward_errors.append(float(abs(bwd - analytical_deriv)))
        central_errors.append(float(abs(ctr - analytical_deriv)))

    opt_fwd = h_values[int(np.argmin(forward_errors))]
    opt_bwd = h_values[int(np.argmin(backward_errors))]
    opt_ctr = h_values[int(np.argmin(central_errors))]
    
    return {
        "h_values": h_values,
        "forward_errors": forward_errors,
        "backward_errors": backward_errors,
        "central_errors": central_errors,
        "optimal_h_forward": opt_fwd,
        "optimal_h_backward": opt_bwd,
        "optimal_h_central": opt_ctr,
    }