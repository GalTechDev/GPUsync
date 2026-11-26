"""Device choice (§4.8 of the architecture).

Without `device=` or `with gs.device(...)`, the planned device is the first of the **ranking**:
discrete GPUs (CUDA, then Vulkan, then OpenCL), then integrated GPUs (Metal, then OpenCL: zero-copy), then CPU.
If that device cannot run the kernel (e.g. `gs.exp` in f64 on Vulkan), the kernel's `compat` option
decides: `"auto"` → warning then fallback to OpenCL (same GPU if possible), otherwise `cpu:0`;
`"strict"` → `gs.UnsupportedFeature`.

A device is named by a `Device` object, an identifier (`"opencl:1"`), a backend name
(`"cuda"`, `"opencl"`, `"metal"`, `"vulkan"`, `"dx12"`, `"cpu"`) or `"gpu"` (first GPU of the ranking).
"""

from __future__ import annotations

import contextlib
import os
import threading

from . import _core

_local = threading.local()
_default: str | None = None

_DISCRETE = {"cuda": 0, "vulkan": 1, "opencl": 2, "dx12": 3, "metal": 4}
_INTEGRATED = {"metal": 0, "opencl": 1, "vulkan": 2, "dx12": 3}


class CompatibilityWarning(UserWarning):
    """The kernel runs on another device than planned, because an operation is missing."""


def _rank(d) -> tuple:
    if d.backend == "cpu":
        return (3, 0, d.index)
    if d.kind == "discrete_gpu":
        return (0, _DISCRETE.get(d.backend, 9), d.index)
    if d.kind == "cpu":  # OpenCL on a CPU (POCL…)
        return (2, 0, d.index)
    return (1, _INTEGRATED.get(d.backend, 9), d.index)


class _Known:
    """Discovered devices, sorted, and answers already given: read again at every kernel launch."""

    def __init__(self) -> None:
        self.ranked = tuple(sorted(_core.devices(), key=_rank))
        self.by_id = {d.id: d for d in self.ranked}
        self.resolved: dict[str, str] = {}


_known_devices: _Known | None = None


def _known() -> _Known:
    global _known_devices
    k = _known_devices
    if k is None:
        k = _known_devices = _Known()
    return k


def devices(refresh: bool = False) -> list:
    """Available devices (discovery done once; `refresh=True` runs it again)."""
    global _known_devices
    if refresh:
        _known_devices = None
    return _core.devices(refresh=refresh)


def backends(refresh: bool = False) -> list:
    """Status of each backend (useful to understand why a GPU does not show up)."""
    global _known_devices
    if refresh:
        _known_devices = None
    return _core.backends(refresh=refresh)


def ranked() -> list:
    """Devices in order of preference."""
    return list(_known().ranked)


def capable(dev, req: dict) -> bool:
    """Can the device run a kernel with these requirements?"""
    return (
        (dev.f64 or not req["f64"])
        and (dev.f64_math or not req["f64_math"])
        and (getattr(dev, "int64", True) or not req.get("int64"))
    )


def _by_id(dev_id: str):
    return _known().by_id[dev_id]


def resolve(device=None) -> str:
    """Identifier (`"metal:0"`…) of the planned device: requested, or first of the ranking."""
    if device is None:
        stack = getattr(_local, "stack", None)
        # Ordre : `with gs.device(...)`, `gs.set_default_device(...)`, variable GPUSYNC_DEVICE.
        device = stack[-1] if stack else (_default or os.environ.get("GPUSYNC_DEVICE") or None)
    known = _known()
    devs = known.ranked
    if device is None:
        if not devs:
            raise _core.DeviceError("no device available")
        return devs[0].id
    if isinstance(device, _core.Device):
        return device.id
    if not isinstance(device, str):
        raise TypeError(f"device: Device or string expected, got {type(device).__name__}")
    found = known.resolved.get(device)
    if found is None:
        found = known.resolved[device] = _find(device, devs)
    return found


def _find(device: str, devs) -> str:
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
    raise _core.DeviceError(f"device \"{device}\" not found. Available: {', '.join(ids)}")


def fallback(planned: str, req: dict) -> str | None:
    """Fallback for a kernel that `planned` cannot run: CUDA or OpenCL on the same physical
    GPU, then another compatible CUDA / OpenCL device, then cpu:0."""
    plan = _by_id(planned)
    full = [d for d in ranked() if d.backend in ("cuda", "opencl") and capable(d, req)]
    same_gpu = [d for d in full if d.name == plan.name]
    for d in same_gpu + full:
        return d.id
    cpu = [d for d in ranked() if d.backend == "cpu" and capable(d, req)]
    return cpu[0].id if cpu else None


def set_default_device(device) -> None:
    """Changes the default device (None: back to the first of the ranking)."""
    global _default
    _default = None if device is None else resolve(device)


@contextlib.contextmanager
def device(device):
    """`with gs.device("opencl"): ...`: device used in the block (per thread)."""
    dev_id = resolve(device)
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stack.append(dev_id)
    try:
        yield _by_id(dev_id)
    finally:
        stack.pop()


@contextlib.contextmanager
def batch(device=None):
    """`with gs.batch(): ...`: groups several kernels into a single GPU submission (wgpu / Metal / Vulkan / DX12).

    Every pass recorded in the block shares a single `CommandEncoder`, and they are submitted
    together when the block exits (or automatically before a CPU read or `gs.Window.present`).
    On the other backends (OpenCL, CUDA, CPU), the block has no effect (compatible everywhere).
    """
    dev_id = resolve(device)
    _core.begin_batch(dev_id)
    try:
        yield
    finally:
        _core.end_batch(dev_id)


def sync(device=None) -> None:
    """Waits for all work submitted on `device` (without copying the results to RAM)."""
    _core.sync(resolve(device))


def default_device(kernel=None):
    """The device that would be used now (for `kernel`, given its requirements)."""
    dev_id = resolve(None)
    if kernel is not None:
        dev_id = kernel._choose(None)[0]
    return _by_id(dev_id)


def set_memory_budget(nbytes=None) -> None:
    """Memory budget per device for Resident mode (None: 50 % of the device's memory).
    Beyond it, computations switch to Flux mode (chunk by chunk), with no size limit."""
    limits = _core.get_transfer_limits()
    _core.set_transfer_limits(limits["chunk_bytes"], None if nbytes is None else int(nbytes), limits["direct_chunk_bytes"])


def _set_transfer_limits(chunk_bytes=64 << 20, budget_bytes=None, direct_chunk_bytes=None) -> None:
    """Internal settings of the transfer engine (tests: force chunking on small arrays)."""
    _core.set_transfer_limits(int(chunk_bytes), budget_bytes, direct_chunk_bytes)
