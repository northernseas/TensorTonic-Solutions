import numpy as np

def tile_diff(data, reps):
    """Returns: np.ndarray of shape (2, m*reps, n), stacked tiled array and padded differences"""
    data = np.array(data, dtype=np.float64)
    
    tiled = np.tile(data, (reps, 1))

    diff = np.pad(np.diff(tiled, axis=0), ((0, 1), (0, 0)), mode="constant")

    return np.stack((
        tiled,
        diff
    ))