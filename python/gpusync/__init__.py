"""GPUsync v2 — écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU.

En construction (jalon M0) : seule la découverte des périphériques est disponible.

    >>> import gpusync as gs
    >>> gs.devices()
    [Device('Apple M5 Pro', backend='metal', f64=False, unified_memory=True), ...]
"""

from ._core import BackendStatus, Device, __version__, backends, devices

__all__ = ["Device", "BackendStatus", "devices", "backends", "__version__"]
