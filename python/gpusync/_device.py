"""Choix du périphérique : par défaut le premier de `gs.devices()`, sinon `device=...`,
`with gs.device(...)` ou `gs.set_default_device(...)`.

Un périphérique se désigne par un objet `Device`, un identifiant (`"opencl:1"`), un nom de
backend (`"opencl"`, `"metal"`, `"vulkan"`, `"dx12"`, `"cpu"`) ou `"gpu"` (premier GPU).
"""

from __future__ import annotations

import contextlib
import threading

from . import _core

_local = threading.local()
_default: str | None = None


def resolve(device=None) -> str:
    """Renvoie l'identifiant (`"metal:0"`…) du périphérique demandé."""
    if device is None:
        stack = getattr(_local, "stack", None)
        device = stack[-1] if stack else _default
    devs = _core.devices()
    if device is None:
        if not devs:
            raise _core.DeviceError("aucun périphérique disponible")
        return devs[0].id
    if isinstance(device, _core.Device):
        return device.id
    if not isinstance(device, str):
        raise TypeError(f"device : Device ou chaîne attendu, reçu {type(device).__name__}")
    want = device.strip().lower()
    ids = [d.id for d in devs]
    if ":" in want:
        if want in ids:
            return want
    elif want == "gpu":
        gpus = [d.id for d in devs if d.kind.endswith("gpu")]
        if gpus:
            return gpus[0]
    else:
        for d in devs:
            if d.backend == want:
                return d.id
    raise _core.DeviceError(f"périphérique « {device} » introuvable. Disponibles : {', '.join(ids)}")


def set_default_device(device) -> None:
    """Change le périphérique par défaut (None : revenir au premier de gs.devices())."""
    global _default
    _default = None if device is None else resolve(device)


@contextlib.contextmanager
def device(device):
    """`with gs.device("opencl"): ...` : périphérique utilisé dans le bloc (par thread)."""
    dev_id = resolve(device)
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stack.append(dev_id)
    try:
        yield next(d for d in _core.devices() if d.id == dev_id)
    finally:
        stack.pop()


def default_device():
    """Le périphérique qui serait utilisé maintenant."""
    dev_id = resolve(None)
    return next(d for d in _core.devices() if d.id == dev_id)
