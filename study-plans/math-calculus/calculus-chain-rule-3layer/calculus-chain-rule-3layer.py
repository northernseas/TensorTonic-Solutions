import numpy as np

def chain_rule_3layer(w1, w2, w3, x):
    """
    Returns: dict with 'factors' (list of 6 floats), 'analytical_gradient' (float), 'numerical_gradient' (float)
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    h1 = w1 * x
    z1 = sigmoid(h1)
    h2 = w2 * z1
    z2 = sigmoid(h2)
    h3 = w3 * z2
    y  = sigmoid(h3)

    # dy/dw1 = dy/dh3 * dh3/dz2 * dz2/dh2 * dh2/dz1 * dz1/dh1 * dh1/w1

    dy_dh3 = y * (1 - y)
    dh3_dz2 = w3
    dz2_dh2 = z2 * (1 - z2)
    dh2_dz1 = w2
    dz1_dh1 = z1 * (1 - z1)
    dh1_dw1 = x

    factors = [
        dy_dh3,
        dh3_dz2,
        dz2_dh2,
        dh2_dz1,
        dz1_dh1,
        dh1_dw1
    ]

    dy_dw1 = dy_dh3 * dh3_dz2 * dz2_dh2 * dh2_dz1 * dz1_dh1 * dh1_dw1

    h = 1e-5
    def forward(w1_):
        z1_ = sigmoid(w1_ * x)
        z2_ = sigmoid(w2 * z1_)
        y_  = sigmoid(w3 * z2_)
        return y_
    numerical = (forward(w1 + h) - forward(w1 - h)) / (2 * h)
    
    return {
        "factors": factors,
        "analytical_gradient": dy_dw1,
        "numerical_gradient": numerical,
    }