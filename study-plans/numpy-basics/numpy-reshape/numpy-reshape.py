import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.array(data, dtype=np.float64)
    D1, D2 = arr.shape
    if operation == "flatten":
        arr = arr.flatten()
    elif operation == "transpose":
        arr = arr.transpose()
    elif operation == "add_batch":
        arr = arr.reshape(1, D1, D2)
    return arr