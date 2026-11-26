"""`gs.Array` : résultat d'un kernel, resté sur le périphérique jusqu'à sa lecture (§4.6).

- passer un `gs.Array` à un autre kernel : aucune copie vers l'hôte ;
- `np.asarray(y)`, `y.numpy()`, `y[i]`, `print(y)` : attendent la fin du calcul et rapatrient ;
- sur mémoire unifiée (OpenCL), le tableau NumPy renvoyé *est* la mémoire du résultat (zéro-copie) ;
- `await y` : attend la fin du calcul sans bloquer la boucle asyncio (attente sur un thread, GIL libéré) ;
- `np.from_dlpack(y)`, `torch.from_dlpack(y)` : export DLPack (mémoire hôte, sans copie) ;
- `y.save("r.npy")` : écrit le résultat dans un fichier .npy, par blocs depuis le GPU ;
- `y + 1`, `y == 2`, `-y`, `np.add(y, 1)` : opérateurs et ufuncs de NumPy, calculés sur l'hôte après lecture,
  rendent un tableau NumPy (pour rester sur le GPU, écrire un kernel).
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
        self._handle = handle  # _core.DeviceArray, ou None si déjà en mémoire hôte
        self._host = host  # destination finale (None tant qu'aucune lecture hôte n'est demandée)
        if host is not None:
            self._shape = host.shape
            self._dtype = host.dtype
            self._size = host.size
        else:
            self._shape = tuple(shape)
            self._dtype = np.dtype(dtype)
            self._size = math.prod(self._shape)
        self._ready = handle is None
        self._lock = threading.Lock()  # lecture possible depuis un thread (await) et le thread principal
        self._check = None  # mode explicite : vérifie qu'aucune écriture n'a été refusée
        self.device = device

    # ── forme et type : connus sans attendre le calcul ──
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
                    if self._host is None:
                        self._host = np.empty(self._shape, dtype=self._dtype)
                    self._handle.read_into(self._host.ctypes.data)
                    self._ready = True
        if self._check is not None:
            self._check()
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

    # ── opérateurs : ceux de NumPy, sur le résultat lu (§4.6) ──
    __hash__ = object.__hash__  # `==` compare les valeurs, mais un gs.Array reste utilisable comme clé (identité)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """`y + 1`, `y == 2`, `np.maximum(y, 0)`… : attend le calcul, lit le résultat et laisse NumPy faire."""
        def host(v):
            return v.numpy() if isinstance(v, Array) else v

        inputs = tuple(host(v) for v in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(host(v) for v in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

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
        return self._host.__dlpack_device__() if self._host is not None else (1, 0)

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
