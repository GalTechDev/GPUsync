import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gpusync.compiler import gpusync_kernel


def test_compiler_transpilation():
    @gpusync_kernel
    def simple_add_kernel(ctx: int, a: list, b: list) -> tuple:
        return a[ctx] + b[ctx]

    # Test OpenCL output
    opencl_src = simple_add_kernel.opencl_code
    assert "__kernel void simple_add_kernel" in opencl_src
    assert "get_global_id(0)" in opencl_src
    assert "a[ctx] + b[ctx]" in opencl_src

    # Test CUDA output
    cuda_src = simple_add_kernel.cuda_code
    assert "__global__ void simple_add_kernel" in cuda_src
    assert "blockIdx.x * blockDim.x + threadIdx.x" in cuda_src
    assert "a[ctx] + b[ctx]" in cuda_src


def test_compiler_dimension_mapping():
    # Test our new dynamic dimension mapping in CUDA
    @gpusync_kernel
    def custom_dim_kernel(a: list):
        get_global_id(0)
        get_global_id(1)
        get_global_id(2)
        get_local_id(0)
        get_local_id(1)
        get_group_id(0)
        get_group_id(1)

    cuda_src = custom_dim_kernel.cuda_code

    # get_global_id(0) -> x, (1) -> y, (2) -> z
    assert "blockIdx.x * blockDim.x + threadIdx.x" in cuda_src
    assert "blockIdx.y * blockDim.y + threadIdx.y" in cuda_src
    assert "blockIdx.z * blockDim.z + threadIdx.z" in cuda_src

    # get_local_id(0) -> x, (1) -> y
    assert "threadIdx.x" in cuda_src
    assert "threadIdx.y" in cuda_src

    # get_group_id(0) -> x, (1) -> y
    assert "blockIdx.x" in cuda_src
    assert "blockIdx.y" in cuda_src


def test_compiler_math_functions():
    @gpusync_kernel
    def math_kernel(a: list, b: list) -> tuple:
        val_min = min(a[0], b[0])
        val_max = max(a[0], b[0])
        val_abs = abs(a[0])
        return val_min, val_max, val_abs

    # OpenCL translations
    opencl_src = math_kernel.opencl_code
    assert "fmin(" in opencl_src
    assert "fmax(" in opencl_src
    assert "fabs(" in opencl_src

    # CUDA translations
    cuda_src = math_kernel.cuda_code
    assert "fminf(" in cuda_src
    assert "fmaxf(" in cuda_src
    assert "fabsf(" in cuda_src
