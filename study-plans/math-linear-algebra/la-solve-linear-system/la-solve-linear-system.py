import numpy as np

def solve_linear_system(A, b):
    """
    Returns: float64 array, the solution x to A @ x = b.
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    M, N = A.shape

    if M == N: # square
        x = np.linalg.solve(A, b) # uses LU decomposition
    elif M > N: # overdetermined
        x = np.linalg.solve(A.T @ A, A.T @ b)
    elif M < N: # underdetermined
        y = np.linalg.solve(A @ A.T, b)
        x = A.T @ y
    return x