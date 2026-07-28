import ast

from gpusync.compiler import gpusync_kernel
from gpusync.compiler.llvm import compile_to_llvm
from gpusync.devices.llvm import LLVMDeviceManager


def test_llvm_transpilation():
    source = """
def add_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx]
"""
    tree = ast.parse(source)
    func_def = tree.body[0]
    code = compile_to_llvm(func_def, "add_kernel", num_returns=1)

    assert 'define void @"add_kernel"(' in code
    assert 'float* %"a"' in code
    assert 'float* %"b"' in code
    assert 'float* %"__out_0"' in code


def test_llvm_device_manager_discovery():
    devices = LLVMDeviceManager.discover_devices()
    assert len(devices) > 0
    assert "CPU" in devices[0].name

    manager = LLVMDeviceManager(devices[0])
    assert manager.backend == "llvm"


def test_llvm_wrapper_attribute():
    @gpusync_kernel
    def my_kernel(ctx: int, a: list) -> tuple:
        return a[ctx] * 2.0

    assert hasattr(my_kernel, "llvm_code")
    assert "my_kernel" in my_kernel.llvm_code
