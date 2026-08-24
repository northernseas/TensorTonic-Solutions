import numpy as np

def weight_init_params(layer_dims, method):
    """
    Returns: list of dicts with keys 'fan_in', 'fan_out', 'shape', 'scale'
    """
    res = []

    for l in range(len(layer_dims) - 1):
        fan_in = layer_dims[l]
        fan_out = layer_dims[l + 1]

        if method == "random_normal":
            scale = 1.0
        elif method == "kaiming_normal":
            scale = np.sqrt(2 / fan_in)
        elif method == "kaiming_uniform":
            scale = np.sqrt(6 / fan_in)
        elif method == "xavier_normal":
            scale = np.sqrt(2 / (fan_in + fan_out))
        elif method == "xavier_uniform":
            scale = np.sqrt(6 / (fan_in + fan_out))

        res.append({
            "fan_in": fan_in,
            "fan_out": fan_out,
            "shape": (fan_out, fan_in),
            "scale": round(scale, 4)
        })

    return res