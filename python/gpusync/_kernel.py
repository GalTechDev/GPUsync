"""`@gs.kernel` : fonction Python → kernel GPU élément par élément (§4.2).

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std

    y = normalize(x, 0.5, 2.0)          # gs.Array, calcul lancé
    np.asarray(y)                       # attend et rapatrie
"""

from __future__ import annotations

import functools

import numpy as np

from . import _core
from ._array import Array
from ._ast import KernelDeclarationError, kernel_json
from ._device import resolve

_ALIGN = 4096  # alignement des sorties : zéro-copie garanti sur les GPU intégrés (Intel, Apple)


def _aligned_empty(shape, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw = np.empty(nbytes + _ALIGN, dtype=np.uint8)
    offset = (-raw.ctypes.data) % _ALIGN
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


class Kernel:
    """Kernel compilé à la demande pour chaque périphérique."""

    def __init__(self, func, *, cast: str | None = None) -> None:
        if cast not in (None, "auto"):
            raise ValueError("cast : None ou 'auto'")
        functools.update_wrapper(self, func)
        self.py_func = func
        self._cast = cast
        try:
            spec, self._params, self._out = kernel_json(func)
        except KernelDeclarationError as e:
            raise _core.KernelSyntaxError(f"kernel '{func.__name__}' : {e}") from None
        try:
            self._ir = _core.compile_kernel(spec)
        except _core.GpusyncError as e:
            where = getattr(func, "__code__", None)
            file = where.co_filename if where else "?"
            raise type(e)(f"kernel '{func.__name__}' ({file}), {e}") from None
        self._arrays = [(n, d) for n, k, d in self._params if k == "array"]
        self._scalars = [(n, d) for n, k, d in self._params if k == "scalar"]

    def __repr__(self) -> str:
        sig = ", ".join(f"{n}: gs.{d.name}{'[:]' if k == 'array' else ''}" for n, k, d in self._params)
        return f"<gs.kernel {self.__name__}({sig}) -> gs.{self._out.name}[:]>"

    def source(self, target: str = "opencl") -> str:
        """Code généré : "opencl", "wgsl" ou "ir"."""
        return self._ir.source(target)

    def __call__(self, *args, device=None) -> Array:
        if len(args) != len(self._params):
            raise TypeError(f"{self.__name__}() attend {len(self._params)} argument(s), {len(args)} reçu(s)")
        dev_id = resolve(device)

        arrays, keepalive, shape, first = [], [], None, None
        scalars = []
        for value, (name, kind, dtype) in zip(args, self._params):
            if kind == "scalar":
                if isinstance(value, (np.ndarray, Array)) and np.ndim(value) != 0:
                    raise _core.TypeMismatch(f"paramètre '{name}' : scalaire attendu (gs.{dtype.name}), reçu un tableau")
                scalars.append(value.item() if isinstance(value, np.generic) else value)
                continue
            if isinstance(value, Array):
                if value.dtype != dtype.np:
                    raise _core.TypeMismatch(
                        f"argument '{name}' : gs.{dtype.name}[:] attendu, reçu un gs.Array de type {value.dtype}"
                    )
                arr_shape = value.shape
                if value._ready:
                    host = value._host
                    arrays.append((host.ctypes.data, host.nbytes))
                    keepalive.append(host)
                else:
                    arrays.append(value._handle)
                    keepalive.append(value)  # sa mémoire hôte peut être lue (zéro-copie)
            else:
                a = np.asarray(value)
                if a.dtype != dtype.np:
                    if self._cast == "auto" and a.dtype.kind in "biuf":
                        a = a.astype(dtype.np)
                    else:
                        raise _core.TypeMismatch(
                            f"argument '{name}' : gs.{dtype.name}[:] attendu, reçu un tableau {a.dtype}. "
                            f"Utilisez {name}.astype(np.{dtype.np.name}), ou @gs.kernel(cast='auto') pour convertir"
                        )
                a = np.ascontiguousarray(a)
                arr_shape = a.shape
                arrays.append((a.ctypes.data, a.nbytes))
                keepalive.append(a)
            if shape is None:
                shape, first = arr_shape, name
            elif arr_shape != shape:
                raise _core.ShapeMismatch(
                    f"{self.__name__}() : les tableaux doivent avoir la même forme ('{first}' {shape}, '{name}' {arr_shape})"
                )

        out = _aligned_empty(shape, self._out.np)
        if out.size == 0:
            return Array(None, out, dev_id)
        keepalive.append(out)
        handle = self._ir.launch(dev_id, arrays, scalars, out.size, out.ctypes.data, keepalive)
        return Array(handle, out, dev_id)


def kernel(func=None, *, cast: str | None = None):
    """Décorateur : `@gs.kernel` ou `@gs.kernel(cast="auto")`."""
    if func is None:
        return lambda f: Kernel(f, cast=cast)
    return Kernel(func, cast=cast)
