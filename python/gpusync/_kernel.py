"""`@gs.kernel` : fonction Python → kernel GPU (§4.2, §4.3).

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std             # élément par élément

    @gs.kernel
    def dist(x: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
        i, j = gs.index()                   # une case de la sortie
        acc = 0.0
        for k in range(x.shape[1]):
            acc += (x[i, k] - x[j, k]) ** 2
        return gs.sqrt(acc)

    y = normalize(x, 0.5, 2.0)          # gs.Array, calcul lancé
    np.asarray(y)                       # attend et rapatrie
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from . import _core
from ._array import Array, Results
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


def _host_array(value, name) -> np.ndarray:
    """Tableau NumPy, liste, ou objet DLPack (PyTorch, JAX, CuPy…) en mémoire hôte, sans copie."""
    if isinstance(value, np.ndarray) or not hasattr(value, "__dlpack__"):
        return np.asarray(value)
    try:
        return np.from_dlpack(value)
    except (BufferError, RuntimeError, TypeError) as e:
        raise _core.TypeMismatch(
            f"argument '{name}' : objet DLPack illisible depuis l'hôte ({e}). S'il est sur un GPU, "
            f"ramenez-le en mémoire hôte d'abord (ex. tenseur.cpu())"
        ) from None


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
            spec, self._params, self._outs, self._tuple = kernel_json(func)
        except KernelDeclarationError as e:
            raise _core.KernelSyntaxError(f"kernel '{func.__name__}' : {e}") from None
        try:
            self._ir = _core.compile_kernel(spec)
        except _core.GpusyncError as e:
            where = getattr(func, "__code__", None)
            file = where.co_filename if where else "?"
            raise type(e)(f"kernel '{func.__name__}' ({file}), {e}") from None
        self._req = self._ir.requirements
        self._indexed = self._ir.grid == "indexed"
        self._dims = self._ir.dims  # dimensions cachées, dans l'ordre des scalaires ajoutés

    def __repr__(self) -> str:
        def ty(k, d, dims):
            if k == "scalar":
                return f"gs.{d.name}"
            return f"gs.{d.name}[{', '.join(':' if x is None else repr(x) for x in dims)}]"

        sig = ", ".join(f"{n}: {ty(k, d, dims)}" for n, k, d, dims in self._params)
        outs = [repr(o) for o in self._outs]
        ret = f"({', '.join(outs)})" if self._tuple else outs[0]
        return f"<gs.kernel {self.__name__}({sig}) -> {ret}>"

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
        def read(r):
            return tuple(a.numpy() for a in r) if isinstance(r, tuple) else r.numpy()

        pending = None
        for item in source:
            args = (item if isinstance(item, tuple) else (item,)) + rest
            current = self(*args, device=device)
            if pending is not None:
                yield read(pending)
            pending = current
        if pending is not None:
            yield read(pending)

    def __call__(self, *args, device=None, out=None):
        """Lance le kernel ; renvoie un gs.Array, ou un tuple de gs.Array si plusieurs sorties."""
        if len(args) != len(self._params):
            raise TypeError(f"{self.__name__}() attend {len(self._params)} argument(s), {len(args)} reçu(s)")
        dev_id, warning = self._choose(device)
        if warning and warning not in self._warned:
            self._warned.add(warning)
            warnings.warn(warning, CompatibilityWarning, stacklevel=2)

        arrays, keepalive, shapes = [], [], []
        scalars = []
        for value, (name, kind, dtype, _) in zip(args, self._params):
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
                a = _host_array(value, name)
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
            shapes.append((name, tuple(arr_shape)))

        if self._indexed:
            shape, hidden = self._bind_dims(shapes)
            scalars.extend(hidden)
        else:
            (first, shape), *others = shapes
            for name, s in others:
                if s != shape:
                    raise _core.ShapeMismatch(
                        f"{self.__name__}() : les tableaux doivent avoir la même forme ('{first}' {shape}, '{name}' {s})"
                    )

        outs = self._outputs(out, shape)
        if outs[0].size == 0:
            results = Results(Array(None, o, dev_id) for o in outs)
        else:
            keepalive.extend(outs)
            handles = self._ir.launch(dev_id, arrays, scalars, outs[0].size, [o.ctypes.data for o in outs], keepalive)
            results = Results(Array(h, o, dev_id) for h, o in zip(handles, outs))
        return results if self._tuple else results[0]

    def _bind_dims(self, shapes) -> tuple[tuple, list]:
        """Kernel avec index : taille de chaque dimension nommée, d'après la forme des entrées.

        Renvoie (forme des sorties, valeurs des dimensions cachées dans l'ordre de l'IR).
        """
        sizes: dict[str, tuple[int, str]] = {}
        params = [p for p in self._params if p[1] == "array"]
        for (name, shape), (_, _, _, dims) in zip(shapes, params):
            if len(shape) != len(dims):
                want = ", ".join(":" if d is None else d for d in dims)
                raise _core.ShapeMismatch(
                    f"{self.__name__}() : '{name}' doit avoir {len(dims)} dimension(s) [{want}], reçu la forme {shape}"
                )
            for axis, (d, size) in enumerate(zip(dims, shape)):
                key = d if d is not None else f"{name}.{axis}"
                if key in sizes and sizes[key][0] != size:
                    prev, who = sizes[key]
                    raise _core.ShapeMismatch(
                        f"{self.__name__}() : dimension '{key}' incohérente : '{who}' → {prev}, '{name}' → {size}"
                    )
                sizes.setdefault(key, (size, name))
        for key, (size, who) in sizes.items():
            if size >= 2**31:
                raise _core.ShapeMismatch(f"{self.__name__}() : dimension '{key}' ({size}) au-delà de 2³¹ − 1")
        shape = tuple(sizes[d][0] for d in self._outs[0].dims)
        return shape, [sizes[d][0] for d in self._dims]

    def _outputs(self, out, shape) -> list:
        """Destinations des sorties : allouées (alignées) ou fournies par `out=`."""
        if out is None:
            return [_aligned_empty(shape, o.dtype.np) for o in self._outs]
        given = list(out) if isinstance(out, (tuple, list)) else [out]
        if len(given) != len(self._outs):
            raise TypeError(f"out : {len(self._outs)} tableau(x) attendu(s) (un par sortie), reçu {len(given)}")
        for o, t in zip(given, self._outs):
            if not isinstance(o, np.ndarray):
                raise TypeError("out : tableau NumPy (ou np.memmap) attendu")
            if o.dtype != t.dtype.np or o.shape != shape:
                raise _core.ShapeMismatch(
                    f"out : forme {shape} et type {t.dtype.np} attendus, reçu {o.shape} {o.dtype}"
                )
            if not (o.flags.c_contiguous and o.flags.writeable):
                raise _core.TypeMismatch("out : le tableau doit être contigu et modifiable")
        return given


def kernel(func=None, *, cast: str | None = None, compat: str = "auto"):
    """Décorateur : `@gs.kernel`, `@gs.kernel(cast="auto")`, `@gs.kernel(compat="strict")`."""
    if func is None:
        return lambda f: Kernel(f, cast=cast, compat=compat)
    return Kernel(func, cast=cast, compat=compat)
