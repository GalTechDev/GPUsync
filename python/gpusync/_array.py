"""`gs.Array` : résultat d'un kernel, resté sur le périphérique jusqu'à sa lecture (§4.6).

- passer un `gs.Array` à un autre kernel : aucune copie vers l'hôte ;
- `np.asarray(y)`, `y.numpy()`, `y[i]`, `print(y)` : attendent la fin du calcul et rapatrient ;
- sur mémoire unifiée (OpenCL), le tableau NumPy renvoyé *est* la mémoire du résultat (zéro-copie) ;
- `await y` : attend la fin du calcul sans bloquer la boucle asyncio (attente sur un thread, GIL libéré) ;
- `np.from_dlpack(y)`, `torch.from_dlpack(y)` : export DLPack (mémoire hôte, sans copie) ;
- `y.save("r.npy")` : écrit le résultat dans un fichier .npy, par blocs depuis le GPU.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np


class Array:
    __slots__ = ("_handle", "_host", "_ready", "_lock", "device")
    __array_priority__ = 1000

    def __init__(self, handle, host: np.ndarray, device: str) -> None:
        self._handle = handle  # _core.DeviceArray, ou None si déjà en mémoire hôte
        self._host = host  # destination finale du résultat (forme et type définitifs)
        self._ready = handle is None
        self._lock = threading.Lock()  # lecture possible depuis un thread (await) et le thread principal
        self.device = device

    # ── forme et type : connus sans attendre le calcul ──
    @property
    def shape(self) -> tuple:
        return self._host.shape

    @property
    def dtype(self) -> np.dtype:
        return self._host.dtype

    @property
    def ndim(self) -> int:
        return self._host.ndim

    @property
    def size(self) -> int:
        return self._host.size

    @property
    def nbytes(self) -> int:
        return self._host.nbytes

    @property
    def mode(self) -> str:
        """Mode d'exécution (§9.1) : "direct" (zéro-copie), "resident" (resté sur le GPU), "flux" (par blocs,
        déjà dans sa destination) ou "cpu"."""
        return self._handle.mode if self._handle is not None else "host"

    @property
    def zero_copy(self) -> bool:
        """Le résultat est-il calculé directement dans la mémoire du tableau NumPy (mode Direct) ?"""
        return self.mode == "direct"

    # ── lecture ──
    def numpy(self) -> np.ndarray:
        """Attend la fin du calcul et renvoie le résultat sous forme de tableau NumPy."""
        if not self._ready:
            with self._lock:
                if not self._ready:
                    self._handle.read_into(self._host.ctypes.data)
                    self._ready = True
        return self._host

    @property
    def ready(self) -> bool:
        """Le résultat a-t-il déjà été lu (aucune attente à prévoir) ?"""
        return self._ready

    def __await__(self):
        """`await y` : attend la fin du calcul et la lecture sur un thread, puis rend `y` (prêt)."""
        return self._wait().__await__()

    async def _wait(self):
        if not self._ready:
            await asyncio.get_running_loop().run_in_executor(None, self.numpy)
        return self

    # ── interopérabilité ──
    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Export DLPack du résultat en mémoire hôte (attend la fin du calcul, sans copie)."""
        host = self.numpy()
        kw = {k: v for k, v in (("max_version", max_version), ("dl_device", dl_device), ("copy", copy)) if v is not None}
        try:
            return host.__dlpack__(stream=stream, **kw)
        except TypeError:  # NumPy < 2 : seul `stream` est accepté
            return host.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._host.__dlpack_device__()

    def save(self, path) -> None:
        """Écrit le résultat dans un fichier .npy (np.load le relit). Un résultat resté sur le GPU est
        rapatrié par blocs directement dans le fichier, sans copie complète en RAM."""
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

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        if dtype is not None and np.dtype(dtype) != a.dtype:
            return a.astype(dtype)
        return a.copy() if copy else a

    def __len__(self) -> int:
        return len(self._host)

    def __getitem__(self, key):
        return self.numpy()[key]

    def __iter__(self):
        return iter(self.numpy())

    # ── réductions : calculées sur l'hôte par NumPy, après lecture (§4.6) ──
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
    """Sorties d'un kernel à plusieurs sorties : un tuple de gs.Array, attendable en une fois
    (`mean, std = await stats(x)`)."""

    __slots__ = ()

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self):
        await asyncio.gather(*self)
        return self
