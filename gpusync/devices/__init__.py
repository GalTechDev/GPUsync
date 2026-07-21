from .opencl import OpenCLDeviceManager
from .cuda import CUDADeviceManager

ALL_DEVICE_MANAGERS = [OpenCLDeviceManager, CUDADeviceManager]

__all__ = ["OpenCLDeviceManager", "CUDADeviceManager", "ALL_DEVICE_MANAGERS"]
