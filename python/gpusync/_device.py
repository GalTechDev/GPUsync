"""Choix du périphérique (§4.8 de l'architecture).

Sans `device=` ni `with gs.device(...)`, le périphérique prévu est le premier du **classement** :
GPU dédiés (Vulkan, puis OpenCL), puis GPU intégrés (Metal, puis OpenCL : zéro-copie), puis CPU.
Si ce périphérique ne sait pas exécuter le kernel (ex. `gs.exp` en f64 sur Vulkan), l'option `compat`
du kernel décide : `"auto"` → avertissement puis repli sur OpenCL (même GPU de préférence), sinon `cpu:0` ;
`"strict"` → `gs.UnsupportedFeature`.

Un périphérique se désigne par un objet `Device`, un identifiant (`"opencl:1"`), un nom de backend
(`"opencl"`, `"metal"`, `"vulkan"`, `"dx12"`, `"cpu"`) ou `"gpu"` (premier GPU du classement).
"""

from __future__ import annotations

import contextlib
import threading

from . import _core

_local = threading.local()
_default: str | None = None

_DISCRETE = {"cuda": 0, "vulkan": 1, "opencl": 2, "dx12": 3, "metal": 4}
_INTEGRATED = {"metal": 0, "opencl": 1, "vulkan": 2, "dx12": 3}


class CompatibilityWarning(UserWarning):
    """Le kernel s'exécute sur un autre périphérique que prévu, faute d'une opération disponible."""


def _rank(d) -> tuple:
    if d.backend == "cpu":
        return (3, 0, d.index)
    if d.kind == "discrete_gpu":
        return (0, _DISCRETE.get(d.backend, 9), d.index)
    if d.kind == "cpu":  # OpenCL sur CPU (POCL…)
        return (2, 0, d.index)
    return (1, _INTEGRATED.get(d.backend, 9), d.index)


def ranked() -> list:
    """Périphériques dans l'ordre de préférence."""
    return sorted(_core.devices(), key=_rank)


def capable(dev, req: dict) -> bool:
    """Le périphérique sait-il exécuter un kernel ayant ces besoins ?"""
    return (dev.f64 or not req["f64"]) and (dev.f64_math or not req["f64_math"])


def _by_id(dev_id: str):
    return next(d for d in _core.devices() if d.id == dev_id)


def resolve(device=None) -> str:
    """Identifiant (`"metal:0"`…) du périphérique prévu : demandé, ou premier du classement."""
    if device is None:
        stack = getattr(_local, "stack", None)
        device = stack[-1] if stack else _default
    devs = ranked()
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


def fallback(planned: str, req: dict) -> str | None:
    """Repli pour un kernel que `planned` ne sait pas exécuter : OpenCL sur le même GPU physique,
    puis un autre OpenCL compatible, puis cpu:0."""
    plan = _by_id(planned)
    opencl = [d for d in ranked() if d.backend == "opencl" and capable(d, req)]
    same_gpu = [d for d in opencl if d.name == plan.name]
    for d in same_gpu + opencl:
        return d.id
    cpu = [d for d in ranked() if d.backend == "cpu" and capable(d, req)]
    return cpu[0].id if cpu else None


def set_default_device(device) -> None:
    """Change le périphérique par défaut (None : revenir au premier du classement)."""
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
        yield _by_id(dev_id)
    finally:
        stack.pop()


def default_device(kernel=None):
    """Le périphérique qui serait utilisé maintenant (pour `kernel`, compte tenu de ses besoins)."""
    dev_id = resolve(None)
    if kernel is not None:
        dev_id = kernel._choose(None)[0]
    return _by_id(dev_id)


def set_memory_budget(nbytes=None) -> None:
    """Budget mémoire par périphérique pour le mode Résident (None : 50 % de la mémoire du périphérique).
    Au-delà, les calculs passent en mode Flux (par blocs), sans limite de taille."""
    limits = _core.get_transfer_limits()
    _core.set_transfer_limits(limits["chunk_bytes"], None if nbytes is None else int(nbytes), limits["direct_chunk_bytes"])


def _set_transfer_limits(chunk_bytes=64 << 20, budget_bytes=None, direct_chunk_bytes=None) -> None:
    """Réglages internes du moteur de transferts (tests : forcer le découpage sur de petits tableaux)."""
    _core.set_transfer_limits(int(chunk_bytes), budget_bytes, direct_chunk_bytes)
