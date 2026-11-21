"""GPUsync v2 — écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU.

    import numpy as np
    import gpusync as gs

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std

    x = np.random.rand(1_000_000).astype(np.float32)
    y = normalize(x, x.mean(), x.std())     # gs.Array : calcul lancé sur le GPU
    print(y[:5])                            # attend la fin et rapatrie

État : jalon M6 (élément par élément, avec index, mode explicite à atomiques ; sans limite de taille ; await, DLPack ; CUDA). Voir ARCHITECTURE.md.
"""

def _nvrtc_dirs() -> None:
    """Dossiers des roues pip NVIDIA (`nvidia-cuda-nvrtc-cu12`…) où le backend CUDA cherche NVRTC,
    en plus du toolkit CUDA et du PATH. Lu au premier kernel compilé en CUDA."""
    import glob
    import os
    import sys

    if "GPUSYNC_NVRTC_DIRS" in os.environ:  # choisi par l'utilisateur (vide : aucune roue)
        return
    found = []
    for base in sys.path:
        for pattern in ("nvidia/*/bin", "nvidia/*/lib", "nvidia/*/bin/*", "nvidia/*/lib/*"):
            for d in glob.glob(os.path.join(base or ".", pattern)):
                if os.path.isdir(d) and glob.glob(os.path.join(d, "*nvrtc*")):
                    found.append(d)
    if found:
        os.environ["GPUSYNC_NVRTC_DIRS"] = os.pathsep.join(found)


_nvrtc_dirs()

from . import _core  # noqa: E402
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
from ._device import CompatibilityWarning, default_device, device, set_default_device, set_memory_budget
from ._kernel import Kernel, OutOfBounds, OutOfMemory, kernel
from ._math import *  # noqa: F403  (sqrt, exp, … utilisables dans les kernels)
from ._math import __all__ as _math_all
from ._types import Out, f32, f64, i32, i64, u32, u64

__all__ = [
    "kernel", "Kernel", "Array",
    "devices", "backends", "Device", "BackendStatus",
    "device", "default_device", "set_default_device", "set_memory_budget",
    "f32", "f64", "i32", "u32", "i64", "u64", "Out",
    "CompatibilityWarning", "OutOfMemory", "OutOfBounds",
    "GpusyncError", "KernelSyntaxError", "TypeMismatch", "ShapeMismatch", "UnsupportedFeature", "DeviceError",
    "__version__",
    *_math_all,
]
