import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gpusync import gpusync_kernel

@gpusync_kernel
def matrix_mult_kernel(ctx: int, A: list, B: list, N: int) -> tuple:
    """
    Multiplication de matrice 2D carrée (N x N) point par point.
    ctx = i * N + j
    """
    row: int = ctx / N
    col: int = ctx % N
    
    acc = 0.0
    for k in range(N):
        acc = acc + A[row * N + k] * B[k * N + col]
        
    return acc
