"""GPUsync v2 — écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU.

    import numpy as np
    import gpusync as gs

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std

    x = np.random.rand(1_000_000).astype(np.float32)
    y = normalize(x, x.mean(), x.std())     # gs.Array : calcul lancé sur le GPU
    print(y[:5])                            # attend la fin et rapatrie

État : jalon M1 (kernels élément par élément). Voir ARCHITECTURE.md.
"""

from . import _core
from ._array import Array
from ._core import (
    BackendStatus,
    Device,
    DeviceError,
    GpusyncError,
    KernelSyntaxError,
    ShapeMismatch,
    TypeMismatch,
    UnsupportedFeature,
    __version__,
    backends,
    devices,
)
from ._device import CompatibilityWarning, default_device, device, set_default_device
from ._kernel import Kernel, kernel
from ._math import *  # noqa: F403  (sqrt, exp, … utilisables dans les kernels)
from ._math import __all__ as _math_all
from ._types import f32, f64, i32, u32

__all__ = [
    "kernel", "Kernel", "Array",
    "devices", "backends", "Device", "BackendStatus",
    "device", "default_device", "set_default_device",
    "f32", "f64", "i32", "u32",
    "CompatibilityWarning",
    "GpusyncError", "KernelSyntaxError", "TypeMismatch", "ShapeMismatch", "UnsupportedFeature", "DeviceError",
    "__version__",
    *_math_all,
]
