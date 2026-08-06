import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    data = np.array(data, dtype=np.float64)
    pad = lambda x: np.pad(x, pad_width, mode="constant", constant_values=0.0)
    return np.stack((
        pad(np.round(data, decimals)),
        pad(np.floor(data)),
        pad(np.ceil(data)),
    ))