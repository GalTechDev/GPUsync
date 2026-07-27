from .cuda import CUDADeviceManager
from .llvm import LLVMDeviceManager
from .opencl import OpenCLDeviceManager

ALL_DEVICE_MANAGERS = [OpenCLDeviceManager, CUDADeviceManager, LLVMDeviceManager]

__all__ = ["OpenCLDeviceManager", "CUDADeviceManager", "LLVMDeviceManager", "ALL_DEVICE_MANAGERS"]
