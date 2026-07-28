import numba
import numpy as np

def my_kernel(ctx, arr, out):
    out[ctx] = arr[ctx] * 2.0

my_kernel_jit = numba.njit(my_kernel)

@numba.njit(parallel=True)
def run_parallel(arr, out):
    for i in numba.prange(len(arr)):
        my_kernel_jit(i, arr, out)

arr = np.ones(10, dtype=np.float32)
out = np.zeros(10, dtype=np.float32)
run_parallel(arr, out)
print("SUCCESS:", out)
