"""`gs.Array`: result of a kernel, kept on the device until it is read (§4.6).

- passing a `gs.Array` to another kernel: no copy to the host;
- `np.asarray(y)`, `y.numpy()`, `y[i]`, `print(y)`: wait for the computation and copy back;
- on unified memory (OpenCL), the returned NumPy array *is* the result's memory (zero-copy);
- `await y`: waits for the computation without blocking the asyncio loop (wait on a thread, GIL released);
- `np.from_dlpack(y)`, `torch.from_dlpack(y)`: DLPack export (host memory, no copy);
- `y.save("r.npy")`: writes the result to a .npy file, chunk by chunk from the GPU;
- `y + 1`, `y == 2`, `-y`, `np.add(y, 1)`: NumPy operators and ufuncs, computed on the host after reading,
  return a NumPy array (to stay on the GPU, write a kernel).
"""

from __future__ import annotations

import asyncio
import math
import threading

import numpy as np


class Array(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = ("_handle", "_host", "_shape", "_dtype", "_size", "_ready", "_lock", "_check", "device")
    __array_priority__ = 1000

    def __init__(self, handle, host: np.ndarray | None, device: str, *, shape=None, dtype=None) -> None:
        self._handle = handle  # _core.DeviceArray, or None if already in host memory
        self._host = host  # final destination (None until a host read is requested)
        if host is not None:
            self._shape = host.shape
            self._dtype = host.dtype
            self._size = host.size
        else:
            self._shape = tuple(shape)
            self._dtype = np.dtype(dtype)
            self._size = math.prod(self._shape)
        self._ready = handle is None
        self._lock = threading.Lock()  # read possible from a thread (await) and from the main thread
        self._check = None  # explicit mode: checks that no write was rejected
        self.device = device

    # ── shape and type: known without waiting for the computation ──
    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size * self._dtype.itemsize

    @property
    def mode(self) -> str:
        """Execution mode (§9.1): "direct" (zero-copy), "resident" (kept on the GPU), "flux" (chunk by chunk,
        already in its destination) or "cpu"."""
        return self._handle.mode if self._handle is not None else "host"

    @property
    def zero_copy(self) -> bool:
        """Is the result computed directly in the NumPy array's memory (Direct mode)?"""
        return self.mode == "direct"

    # ── reading ──
    def numpy(self) -> np.ndarray:
        """Waits for the computation and returns the result as a NumPy array."""
        if not self._ready:
            with self._lock:
                if not self._ready:
                    if self._host is None:
                        self._host = np.empty(self._shape, dtype=self._dtype)
                    self._handle.read_into(self._host.ctypes.data)
                    self._ready = True
        if self._check is not None:
            self._check()
        return self._host

    @property
    def ready(self) -> bool:
        """Has the result already been read (no wait to expect)?"""
        return self._ready

    def __await__(self):
        """`await y`: waits for the computation and the read on a thread, then returns `y` (ready)."""
        return self._wait().__await__()

    async def _wait(self):
        if not self._ready:
            await asyncio.get_running_loop().run_in_executor(None, self.numpy)
        return self

    # ── operators: NumPy's, on the read result (§4.6) ──
    __hash__ = object.__hash__  # `==` compares values, but a gs.Array stays usable as a key (identity)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """`y + 1`, `y == 2`, `np.maximum(y, 0)`…: waits for the computation, reads the result and lets NumPy do it."""
        def host(v):
            return v.numpy() if isinstance(v, Array) else v

        inputs = tuple(host(v) for v in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(host(v) for v in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

    # ── interoperability ──
    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """DLPack export of the result in host memory (waits for the computation, no copy)."""
        host = self.numpy()
        kw = {k: v for k, v in (("max_version", max_version), ("dl_device", dl_device), ("copy", copy)) if v is not None}
        try:
            return host.__dlpack__(stream=stream, **kw)
        except TypeError:  # NumPy < 2: only `stream` is accepted
            return host.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._host.__dlpack_device__() if self._host is not None else (1, 0)

    def save(self, path) -> None:
        """Writes the result to a .npy file (np.load reads it back). A result kept on the GPU is
        copied back chunk by chunk straight into the file, without a full copy in RAM."""
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=self.dtype, shape=self.shape)
        if self._ready or mm.size == 0:
            mm[...] = self._host
        else:
            with self._lock:
                if self._ready:
                    mm[...] = self._host
                else:
                    self._handle.read_into(mm.ctypes.data)
        mm.flush()
        del mm
        if self._check is not None:
            self._check()

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        if dtype is not None and np.dtype(dtype) != a.dtype:
            return a.astype(dtype)
        return a.copy() if copy else a

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __getitem__(self, key):
        return self.numpy()[key]

    def __iter__(self):
        return iter(self.numpy())

    # ── reductions: computed on the host by NumPy, after reading (§4.6) ──
    def sum(self, axis=None, **kw):
        return self.numpy().sum(axis=axis, **kw)

    def mean(self, axis=None, **kw):
        return self.numpy().mean(axis=axis, **kw)

    def min(self, axis=None, **kw):
        return self.numpy().min(axis=axis, **kw)

    def max(self, axis=None, **kw):
        return self.numpy().max(axis=axis, **kw)

    def std(self, axis=None, **kw):
        return self.numpy().std(axis=axis, **kw)

    def __repr__(self) -> str:
        body = np.array2string(self.numpy(), separator=", ", prefix="gs.Array(")
        return f"gs.Array({body}, dtype={self.dtype}, device='{self.device}')"

    def __str__(self) -> str:
        return str(self.numpy())


class Results(tuple):
    """Outputs of a kernel with several outputs: a tuple of gs.Array, awaitable in one go
    (`mean, std = await stats(x)`)."""

    __slots__ = ()

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self):
        await asyncio.gather(*self)
        return self
