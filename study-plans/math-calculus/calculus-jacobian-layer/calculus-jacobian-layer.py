import numpy as np

def jacobian_network_layer(W, b, x):
    """
    Returns: dict with 'z', 'f' (lists), 'analytical_jacobian', 'numerical_jacobian' (2x3 lists of lists)
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.array(x, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    z = W @ x + b
    f = sigmoid(z) # (2,)

    # df/dx = df/dz * dz/dx
    sigma_prime = f * (1 - f)
    analytical = sigma_prime[:, None] * W

    h = 1e-5
    m, d = W.shape
    numerical = np.zeros((m, d))
    for j in range(d):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        f_plus = 1.0 / (1.0 + np.exp(-(W @ x_plus + b)))
        f_minus = 1.0 / (1.0 + np.exp(-(W @ x_minus + b)))
        numerical[:, j] = (f_plus - f_minus) / (2.0 * h)

    return {
        "z": z.tolist(),
        "f": f.tolist(),
        "analytical_jacobian": analytical.tolist(),
        "numerical_jacobian": numerical.tolist()
    }