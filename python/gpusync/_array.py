"""`gs.Array` : résultat d'un kernel, resté sur le périphérique jusqu'à sa lecture (§4.6).

- passer un `gs.Array` à un autre kernel : aucune copie vers l'hôte ;
- `np.asarray(y)`, `y.numpy()`, `y[i]`, `print(y)` : attendent la fin du calcul et rapatrient ;
- sur mémoire unifiée (OpenCL), le tableau NumPy renvoyé *est* la mémoire du résultat (zéro-copie).
"""

from __future__ import annotations

import numpy as np


class Array:
    __slots__ = ("_handle", "_host", "_ready", "device")
    __array_priority__ = 1000

    def __init__(self, handle, host: np.ndarray, device: str) -> None:
        self._handle = handle  # _core.DeviceArray, ou None si déjà en mémoire hôte
        self._host = host  # destination finale du résultat (forme et type définitifs)
        self._ready = handle is None
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
    def zero_copy(self) -> bool:
        """Le résultat est-il calculé directement dans la mémoire du tableau NumPy ?"""
        return self._handle is not None and self._handle.host_ptr == self._host.ctypes.data

    # ── lecture ──
    def numpy(self) -> np.ndarray:
        """Attend la fin du calcul et renvoie le résultat sous forme de tableau NumPy."""
        if not self._ready:
            self._handle.read_into(self._host.ctypes.data)
            self._ready = True
        return self._host

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

    def __repr__(self) -> str:
        body = np.array2string(self.numpy(), separator=", ", prefix="gs.Array(")
        return f"gs.Array({body}, dtype={self.dtype}, device='{self.device}')"

    def __str__(self) -> str:
        return str(self.numpy())
