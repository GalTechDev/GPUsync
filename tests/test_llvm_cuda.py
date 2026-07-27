import ast

from gpusync.compiler import gpusync_kernel
from gpusync.compiler.llvm_cuda import compile_to_llvm_cuda


def test_llvm_cuda_transpilation():
    source = """
def multiply_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] * b[ctx]
"""
    tree = ast.parse(source)
    func_def = tree.body[0]
    ptx_code = compile_to_llvm_cuda(func_def, "multiply_kernel", num_returns=1)

    assert ptx_code is not None
    assert "GPUsync LLVM NVPTX" in ptx_code
    assert "multiply_kernel" in ptx_code


def test_gpusync_kernel_ptx_attribute():
    @gpusync_kernel
    def square_kernel(ctx: int, a: list) -> tuple:
        return a[ctx] * a[ctx]

    assert hasattr(square_kernel, "ptx_code")
    assert square_kernel.ptx_code is not None
    assert "square_kernel" in square_kernel.ptx_code
