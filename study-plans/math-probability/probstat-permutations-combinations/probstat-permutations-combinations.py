from math import factorial, perm, comb

def perms_and_combs(n, r):
    """
    Returns: [permutations, combinations, factorial] as a list.
    """
    return [
        perm(n, r),
        comb(n, r),
        factorial(n)
    ]