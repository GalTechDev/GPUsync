from .opencl import OpenCLDeviceManager
from .cuda import CUDADeviceManager
from .llvm import LLVMDeviceManager

ALL_DEVICE_MANAGERS = [OpenCLDeviceManager, CUDADeviceManager, LLVMDeviceManager]

__all__ = ["OpenCLDeviceManager", "CUDADeviceManager", "LLVMDeviceManager", "ALL_DEVICE_MANAGERS"]
