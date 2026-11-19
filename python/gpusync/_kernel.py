"""`@gs.kernel` : fonction Python → kernel GPU élément par élément (§4.2).

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std

    y = normalize(x, 0.5, 2.0)          # gs.Array, calcul lancé
    np.asarray(y)                       # attend et rapatrie
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from . import _core
from ._array import Array
from ._ast import KernelDeclarationError, kernel_json
from ._device import CompatibilityWarning, _by_id, capable, fallback, ranked, resolve

_ALIGN = 16384  # alignement des sorties : une page sur Apple Silicon (16 Kio), 4 sur ailleurs


class OutOfMemory(_core.GpusyncError):
    """La sortie ne peut pas être allouée en mémoire : passez une destination avec out=."""


def _aligned_empty(shape, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    try:
        raw = np.empty(nbytes + _ALIGN, dtype=np.uint8)
    except MemoryError:
        raise OutOfMemory(
            f"impossible d'allouer la sortie ({nbytes / 2**30:.1f} Gio) en mémoire. Passez une destination, "
            f"ex. out=np.memmap('sortie.bin', dtype=np.{dtype.name}, mode='w+', shape={tuple(shape)})"
        ) from None
    offset = (-raw.ctypes.data) % _ALIGN
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


class Kernel:
    """Kernel compilé à la demande pour chaque périphérique."""

    def __init__(self, func, *, cast: str | None = None, compat: str = "auto") -> None:
        if cast not in (None, "auto"):
            raise ValueError("cast : None ou 'auto'")
        if compat not in ("auto", "strict"):
            raise ValueError("compat : 'auto' (repli avec avertissement) ou 'strict' (erreur)")
        self._compat = compat
        self._warned: set = set()
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
        self._req = self._ir.requirements
        self._arrays = [(n, d) for n, k, d in self._params if k == "array"]
        self._scalars = [(n, d) for n, k, d in self._params if k == "scalar"]

    def __repr__(self) -> str:
        sig = ", ".join(f"{n}: gs.{d.name}{'[:]' if k == 'array' else ''}" for n, k, d in self._params)
        return f"<gs.kernel {self.__name__}({sig}) -> gs.{self._out.name}[:]>"

    def _choose(self, device) -> tuple[str, str | None]:
        """(périphérique d'exécution, message d'avertissement éventuel) selon `compat` (§4.8)."""
        planned = resolve(device)
        if capable(_by_id(planned), self._req):
            return planned, None
        plan = _by_id(planned)
        if self._req["f64_math"] and plan.f64:
            what = f"{self._req['f64_math_op']} en f64"
        else:
            what = "f64"
        compatible = [d.id for d in ranked() if capable(d, self._req)]
        if self._compat == "strict":
            raise _core.UnsupportedFeature(
                f"kernel '{self.__name__}' : {what} n'est pas disponible sur {planned} ({plan.name}). "
                f"Périphériques compatibles : {', '.join(compatible) or 'aucun'} "
                "(ou @gs.kernel(compat='auto') pour s'y replier automatiquement)"
            )
        target = fallback(planned, self._req)
        if target is None:
            raise _core.UnsupportedFeature(f"kernel '{self.__name__}' : {what} n'est disponible sur aucun périphérique")
        msg = f"kernel '{self.__name__}' : {what} n'est pas disponible sur {planned} ({plan.name}) → exécuté sur {target}"
        if target == "cpu:0":
            msg += " (interpréteur : résultat correct, mais lent)"
        return target, msg

    def source(self, target: str = "opencl") -> str:
        """Code généré : "opencl", "wgsl" ou "ir"."""
        return self._ir.source(target)

    def stream(self, source, *rest, device=None):
        """Traite un flux de blocs (générateur, itérable) et renvoie un itérateur de résultats NumPy.

        Chaque élément de `source` est le premier argument du kernel (un tableau), ou un tuple des
        premiers arguments ; `rest` complète l'appel. Le bloc suivant est lancé avant de rendre le
        résultat du bloc courant : calcul et consommation se recouvrent.
        """
        pending = None
        for item in source:
            args = (item if isinstance(item, tuple) else (item,)) + rest
            current = self(*args, device=device)
            if pending is not None:
                yield pending.numpy()
            pending = current
        if pending is not None:
            yield pending.numpy()

    def __call__(self, *args, device=None, out=None) -> Array:
        if len(args) != len(self._params):
            raise TypeError(f"{self.__name__}() attend {len(self._params)} argument(s), {len(args)} reçu(s)")
        dev_id, warning = self._choose(device)
        if warning and warning not in self._warned:
            self._warned.add(warning)
            warnings.warn(warning, CompatibilityWarning, stacklevel=2)

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

        if out is None:
            out = _aligned_empty(shape, self._out.np)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out : tableau NumPy (ou np.memmap) attendu")
            if out.dtype != self._out.np or out.shape != shape:
                raise _core.ShapeMismatch(
                    f"out : forme {shape} et type {self._out.np} attendus, reçu {out.shape} {out.dtype}"
                )
            if not (out.flags.c_contiguous and out.flags.writeable):
                raise _core.TypeMismatch("out : le tableau doit être contigu et modifiable")
        if out.size == 0:
            return Array(None, out, dev_id)
        keepalive.append(out)
        handle = self._ir.launch(dev_id, arrays, scalars, out.size, out.ctypes.data, keepalive)
        return Array(handle, out, dev_id)


def kernel(func=None, *, cast: str | None = None, compat: str = "auto"):
    """Décorateur : `@gs.kernel`, `@gs.kernel(cast="auto")`, `@gs.kernel(compat="strict")`."""
    if func is None:
        return lambda f: Kernel(f, cast=cast, compat=compat)
    return Kernel(func, cast=cast, compat=compat)
