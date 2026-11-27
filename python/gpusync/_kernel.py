"""`@gs.kernel`: Python function → GPU kernel (§4.2, §4.3).

    @gs.kernel
    def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
        return (x - mean) / std             # element-wise

    @gs.kernel
    def dist(x: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
        i, j = gs.index()                   # one cell of the output
        acc = 0.0
        for k in range(x.shape[1]):
            acc += (x[i, k] - x[j, k]) ** 2
        return gs.sqrt(acc)

    y = normalize(x, 0.5, 2.0)          # gs.Array, computation launched
    np.asarray(y)                       # waits and copies back
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np

from . import _core
from ._array import Array, Results
from ._ast import KernelDeclarationError, kernel_json
from ._device import CompatibilityWarning, _by_id, _known, capable, fallback, ranked, resolve

_ALIGN = 16384  # output alignment: a page on Apple Silicon (16 KiB), 4 elsewhere


class OutOfBounds(_core.GpusyncError, IndexError):
    """Out-of-bounds atomic write into a gs.Out output: it was not done (the others were)."""


class OutOfMemory(_core.GpusyncError):
    """The output cannot be allocated in memory: pass a destination with out=."""


def _aligned_empty(shape, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    try:
        raw = np.empty(nbytes + _ALIGN, dtype=np.uint8)
    except MemoryError:
        raise OutOfMemory(
            f"cannot allocate the output ({nbytes / 2**30:.1f} GiB) in memory. Pass a destination, "
            f"e.g. out=np.memmap('output.bin', dtype=np.{dtype.name}, mode='w+', shape={tuple(shape)})"
        ) from None
    offset = (-raw.ctypes.data) % _ALIGN
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


def _host_array(value, name) -> np.ndarray:
    """NumPy array, list, or DLPack object (PyTorch, JAX, CuPy…) in host memory, without copy."""
    if isinstance(value, np.ndarray) or not hasattr(value, "__dlpack__"):
        return np.asarray(value)
    try:
        return np.from_dlpack(value)
    except (BufferError, RuntimeError, TypeError) as e:
        raise _core.TypeMismatch(
            f"argument '{name}': DLPack object unreadable from the host ({e}). If it is on a GPU, "
            f"move it to host memory first (e.g. tensor.cpu())"
        ) from None


class Kernel:
    """Kernel compiled on demand for each device."""

    def __init__(self, func, *, cast: str | None = None, compat: str = "auto", grid=None, group=None,
                 cells=None) -> None:
        if cast not in (None, "auto"):
            raise ValueError("cast: None or 'auto'")
        if compat not in ("auto", "strict"):
            raise ValueError("compat: 'auto' (fallback with a warning) or 'strict' (error)")
        self._compat = compat
        self._warned: set = set()
        functools.update_wrapper(self, func)
        self.py_func = func
        self._cast = cast
        if grid is not None and not (isinstance(grid, str) or (isinstance(grid, (tuple, list)) and grid)):
            raise ValueError('grid: name of an input (grid="x") or tuple of dimensions (grid=("n", "n"))')
        _check_group(group)
        self._grid = grid
        self._group = group
        try:
            spec, self._params, self._outs, self._tuple, self._fills = kernel_json(func, grid, group, cells)
        except KernelDeclarationError as e:
            raise _core.KernelSyntaxError(f"kernel '{func.__name__}': {e}") from None
        try:
            self._ir = _core.compile_kernel(spec)
        except _core.GpusyncError as e:
            where = getattr(func, "__code__", None)
            file = where.co_filename if where else "?"
            raise type(e)(f"kernel '{func.__name__}' ({file}), {e}") from None
        self._req = self._ir.requirements
        self._indexed = self._ir.grid == "indexed"
        self._capable_on: tuple = (None, set())  # (known devices, capable identifiers)
        self._bound: dict = {}  # input shapes → dimensions already checked
        self._dims = self._ir.dims  # hidden dimensions, in the order of the appended scalars

    def __repr__(self) -> str:
        def ty(k, d, dims):
            if k == "scalar":
                return f"gs.{d.name}"
            t = f"gs.{d.name}[{', '.join(':' if x is None else repr(x) for x in dims)}]"
            return f"gs.Out[{t}]" if k == "out" else t

        sig = ", ".join(f"{n}: {ty(k, d, dims)}" for n, k, d, dims in self._params)
        if self._grid is not None:
            return f"<gs.kernel {self.__name__}({sig}) grid={self._grid!r}>"
        outs = [repr(o) for o in self._outs]
        ret = f"({', '.join(outs)})" if self._tuple else outs[0]
        return f"<gs.kernel {self.__name__}({sig}) -> {ret}>"

    def _choose(self, device) -> tuple[str, str | None]:
        """(execution device, possible warning message) according to `compat` (§4.8)."""
        planned = resolve(device)
        known = _known()
        if self._capable_on[0] is not known:
            self._capable_on = (known, set())
        if planned in self._capable_on[1]:
            return planned, None
        if capable(_by_id(planned), self._req):
            self._capable_on[1].add(planned)
            return planned, None
        plan = _by_id(planned)
        if self._req.get("int64") and not getattr(plan, "int64", True):
            what = "computing with 64-bit integers (gs.i64, gs.u64)"
        elif self._req["f64_math"] and plan.f64:
            what = f"{self._req['f64_math_op']} in f64"
        else:
            what = "f64"
        compatible = [d.id for d in ranked() if capable(d, self._req)]
        if self._compat == "strict":
            raise _core.UnsupportedFeature(
                f"kernel '{self.__name__}': {what} is not available on {planned} ({plan.name}). "
                f"Compatible devices: {', '.join(compatible) or 'none'} "
                "(or @gs.kernel(compat='auto') to fall back to them automatically)"
            )
        target = fallback(planned, self._req)
        if target is None:
            raise _core.UnsupportedFeature(f"kernel '{self.__name__}': {what} is not available on any device")
        msg = f"kernel '{self.__name__}': {what} is not available on {planned} ({plan.name}) → running on {target}"
        if target == "cpu:0":
            msg += " (interpreter: correct result, but slow)"
        return target, msg

    def source(self, target: str = "opencl") -> str:
        """Generated code: "cuda", "opencl", "wgsl" or "ir"."""
        return self._ir.source(target)

    def stream(self, source, *rest, device=None):
        """Processes a stream of chunks (generator, iterable) and returns an iterator of NumPy results.

        Each item of `source` is the first argument of the kernel (an array), or a tuple of the
        first arguments; `rest` completes the call. The next chunk is launched before the current
        chunk's result is returned: computation and consumption overlap.
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
        """Runs the kernel; returns a gs.Array, or a tuple of gs.Array if there are several outputs.

        With grid=: the arguments are the inputs (not the gs.Out), and the results are the gs.Out,
        zero-filled, or starting from the contents of `out=` (accumulation over several calls).
        """
        n_args = sum(k != "out" for _, k, _, _ in self._params)
        if len(args) != n_args:
            raise TypeError(f"{self.__name__}() expects {n_args} argument(s), {len(args)} given")
        dev_id, warning = self._choose(device)
        if warning and warning not in self._warned:
            self._warned.add(warning)
            warnings.warn(warning, CompatibilityWarning, stacklevel=2)

        arrays, keepalive, shapes = [], [], []
        scalars = []
        has_dev_input = False
        it = iter(args)
        for name, kind, dtype, _ in self._params:
            if kind == "out":
                arrays.append(None)  # allocated below, once the dimensions are known
                continue
            value = next(it)
            if kind == "scalar":
                if isinstance(value, (np.ndarray, Array)) and np.ndim(value) != 0:
                    raise _core.TypeMismatch(f"parameter '{name}': scalar expected (gs.{dtype.name}), got an array")
                scalars.append(value.item() if isinstance(value, np.generic) else value)
                continue
            if isinstance(value, Array):
                if value.dtype != dtype.np:
                    raise _core.TypeMismatch(
                        f"argument '{name}': gs.{dtype.name}[:] expected, got a gs.Array of type {value.dtype}"
                    )
                arr_shape = value.shape
                if value._ready:
                    host = value._host
                    arrays.append((host.ctypes.data, host.nbytes))
                    keepalive.append(host)
                else:
                    arrays.append(value._handle)
                    if value.mode in ("resident", "direct") and value.device == dev_id:
                        has_dev_input = True
                    if value.mode != "resident":
                        keepalive.append(value)  # its host memory may be read (zero-copy)
            else:
                a = _host_array(value, name)
                if a.dtype != dtype.np:
                    if self._cast == "auto" and a.dtype.kind in "biuf":
                        a = a.astype(dtype.np)
                    else:
                        raise _core.TypeMismatch(
                            f"argument '{name}': gs.{dtype.name}[:] expected, got a {a.dtype} array. "
                            f"Use {name}.astype(np.{dtype.np.name}), or @gs.kernel(cast='auto') to convert"
                        )
                a = np.ascontiguousarray(a)
                arr_shape = a.shape
                arrays.append((a.ctypes.data, a.nbytes))
                keepalive.append(a)
            shapes.append((name, tuple(arr_shape)))

        if self._grid is not None:
            return self._call_explicit(dev_id, arrays, keepalive, shapes, scalars, out)
        if self._indexed:
            size = self._bind_dims(shapes)
            shape = tuple(size(d) for d in self._outs[0].dims)
            scalars.extend(size(d) for d in self._dims)
        else:
            (first, shape), *others = shapes
            for name, s in others:
                if s != shape:
                    raise _core.ShapeMismatch(
                        f"{self.__name__}(): the arrays must have the same shape ('{first}' {shape}, '{name}' {s})"
                    )

        total_elems = math.prod(shape)
        # Resident path chained on the GPU: avoid allocating megabytes of unused host RAM
        # for intermediate outputs (they are allocated lazily if .numpy() is called).
        if out is None and dev_id != "cpu:0" and has_dev_input and total_elems > 0:
            handles = self._ir.launch(dev_id, arrays, scalars, total_elems, [0] * len(self._outs), keepalive)
            results = Results(Array(h, None, dev_id, shape=shape, dtype=o.dtype.np) for h, o in zip(handles, self._outs))
            return results if self._tuple else results[0]

        outs = self._outputs(out, shape)
        if outs[0].size == 0:
            results = Results(Array(None, o, dev_id) for o in outs)
        else:
            keepalive.extend(outs)
            handles = self._ir.launch(dev_id, arrays, scalars, outs[0].size, [o.ctypes.data for o in outs], keepalive)
            results = Results(Array(h, o, dev_id) for h, o in zip(handles, outs))
        return results if self._tuple else results[0]

    def _call_explicit(self, dev_id, arrays, keepalive, shapes, scalars, out):
        """Explicit mode: grid given by grid=, whole gs.Out outputs written by atomics."""
        size = self._bind_dims(shapes)
        if isinstance(self._grid, str):
            dims = next(d for n, k, _, d in self._params if n == self._grid)
            grid = tuple(size(d if d is not None else f"{self._grid}.{a}") for a, d in enumerate(dims))
        else:
            grid = tuple(size(d) for d in self._grid)
        outs_decl = [(n, d, dims) for n, k, d, dims in self._params if k == "out"]
        # An output dimension missing from the inputs may come from `out=` (e.g. the width of a block).
        given_out = [] if out is None else (list(out) if isinstance(out, (tuple, list)) else [out])
        known = {}
        for (n, _, dims), o in zip(outs_decl, given_out):
            for d, extent in zip(dims, getattr(o, "shape", ())):
                if isinstance(d, str):
                    try:
                        size(d)
                    except KeyError:
                        known.setdefault(d, extent)
        if known:
            inner = size

            def size(d, inner=inner):  # noqa: F811
                return known[d] if d in known else inner(d)

        shapes_out = []
        for n, _, dims in outs_decl:
            try:
                shapes_out.append(tuple(size(d) for d in dims))
            except KeyError as e:
                raise _core.ShapeMismatch(
                    f"{self.__name__}(): dimension {e} of output '{n}' unknown (missing from the inputs)"
                ) from None
        if out is None:
            results = []
            for (n, d, _), shp in zip(outs_decl, shapes_out):
                a = _aligned_empty(shp, d.np)
                a.fill(self._fills.get(n, 0))
                results.append(a)
        else:
            given = list(out) if isinstance(out, (tuple, list)) else [out]
            if len(given) != len(outs_decl):
                raise TypeError(f"out: {len(outs_decl)} array(s) expected (one per gs.Out), got {len(given)}")
            results = []
            for o, (n, d, _), shp in zip(given, outs_decl, shapes_out):
                if isinstance(o, Array):
                    if o.dtype != d.np or o.shape != shp:
                        raise _core.ShapeMismatch(
                            f"out: for '{n}', array of shape {shp} and type {d.np} expected"
                        )
                    if not o._ready and o.mode in ("resident", "direct") and o.device == dev_id:
                        results.append(o)
                        if o.mode != "resident":
                            keepalive.append(o)
                    else:
                        # Host copy so as not to silently mutate an already materialized gs.Array
                        results.append(np.ascontiguousarray(o.numpy()).copy())
                else:
                    if not isinstance(o, np.ndarray) or o.dtype != d.np or o.shape != shp:
                        raise _core.ShapeMismatch(
                            f"out: for '{n}', NumPy array of shape {shp} and type {d.np} expected"
                        )
                    if not (o.flags.c_contiguous and o.flags.writeable):
                        raise _core.TypeMismatch("out: the array must be contiguous and writable")
                    results.append(o)
        err = np.zeros(2, dtype=np.uint32)  # gs_err: rejected write (output, index)
        slots = iter(results)
        for k, a in enumerate(arrays):
            if a is None:
                r = next(slots)
                if isinstance(r, Array):
                    arrays[k] = r._handle
                else:
                    arrays[k] = (r.ctypes.data, r.nbytes)
        arrays.append((err.ctypes.data, err.nbytes))
        scalars.extend(size(d) for d in self._dims)
        n = math.prod(grid)
        if n == 0:
            arrs = Results(r if isinstance(r, Array) else Array(None, r, dev_id) for r in results)
            return arrs if self._tuple else arrs[0]
        for r in results:
            if not isinstance(r, Array):
                keepalive.append(r)
        keepalive.append(err)
        handles = self._ir.launch(dev_id, arrays, scalars, n, [], keepalive)
        err_arr = Array(handles[-1], err, dev_id)
        names = [n for n, _, _ in outs_decl]
        kname = self.__name__

        def check():
            e = err_arr.numpy()
            if e[0]:
                who = names[int(e[0]) - 1]
                raise OutOfBounds(
                    f"kernel '{kname}': gs.atomic_*({who}, …) with an out-of-bounds index (e.g. "
                    f"{'flattened ' if len(shapes_out[int(e[0]) - 1]) > 1 else ''}index {int(e[1:2].view(np.int32)[0])} for "
                    f"{who} of shape {shapes_out[int(e[0]) - 1]}); these writes "
                    "were not done, the others were. Clamp the index, e.g. gs.min(k, n - 1)"
                )

        arrs = []
        for h, r, (_, d, _), shp in zip(handles[:-1], results, outs_decl, shapes_out):
            a = Array(h, None, dev_id, shape=shp, dtype=d.np) if isinstance(r, Array) else Array(h, r, dev_id)
            a._check = check
            arrs.append(a)
        arrs = Results(arrs)
        return arrs if self._tuple else arrs[0]

    def _bind_dims(self, shapes):
        """Indexed kernel: size of each named dimension, from the shapes of the inputs.

        Returns `size(name)`: the size of a dimension (a name, or an integer size written in digits).
        """
        key_shapes = tuple(shapes)
        bound = self._bound.get(key_shapes)
        if bound is not None:
            return bound
        sizes: dict[str, tuple[int, str]] = {}
        params = [p for p in self._params if p[1] == "array"]
        for (name, shape), (_, _, _, dims) in zip(shapes, params):
            if len(shape) != len(dims):
                want = ", ".join(":" if d is None else d for d in dims)
                raise _core.ShapeMismatch(
                    f"{self.__name__}(): '{name}' must have {len(dims)} dimension(s) [{want}], got shape {shape}"
                )
            for axis, (d, size) in enumerate(zip(dims, shape)):
                key = str(d) if d is not None else f"{name}.{axis}"
                if isinstance(d, int) and size != d:
                    raise _core.ShapeMismatch(
                        f"{self.__name__}(): '{name}': axis {axis} must have {d} elements, got {size}"
                    )
                if key in sizes and sizes[key][0] != size:
                    prev, who = sizes[key]
                    raise _core.ShapeMismatch(
                        f"{self.__name__}(): inconsistent dimension '{key}': '{who}' → {prev}, '{name}' → {size}"
                    )
                sizes.setdefault(key, (size, name))
        for key, (size, who) in sizes.items():
            if size >= 2**31:
                raise _core.ShapeMismatch(f"{self.__name__}(): dimension '{key}' ({size}) beyond 2³¹ − 1")

        def size(d):
            d = str(d)
            if d in sizes:
                return sizes[d][0]
            if d.isdigit():
                return int(d)
            raise KeyError(f"'{d}'")

        if len(self._bound) >= 64:
            self._bound.clear()
        self._bound[key_shapes] = size
        return size

    def _outputs(self, out, shape) -> list:
        """Destinations of the outputs: allocated (aligned) or given with `out=`."""
        if out is None:
            return [_aligned_empty(shape, o.dtype.np) for o in self._outs]
        given = list(out) if isinstance(out, (tuple, list)) else [out]
        if len(given) != len(self._outs):
            raise TypeError(f"out: {len(self._outs)} array(s) expected (one per output), got {len(given)}")
        for o, t in zip(given, self._outs):
            if not isinstance(o, np.ndarray):
                raise TypeError("out: NumPy array (or np.memmap) expected")
            if o.dtype != t.dtype.np or o.shape != shape:
                raise _core.ShapeMismatch(
                    f"out: shape {shape} and type {t.dtype.np} expected, got {o.shape} {o.dtype}"
                )
            if not (o.flags.c_contiguous and o.flags.writeable):
                raise _core.TypeMismatch("out: the array must be contiguous and writable")
        return given


def _check_group(group):
    if group is not None and (isinstance(group, bool) or not isinstance(group, int) or not 1 <= group <= 1024):
        raise ValueError("group: number of threads per group, integer between 1 and 1024 (e.g. group=256)")


def kernel(func=None, *, cast: str | None = None, compat: str = "auto", grid=None, group=None, cells=None):
    """Decorator: `@gs.kernel`, `@gs.kernel(cast="auto")`, `@gs.kernel(compat="strict")`,
    `@gs.kernel(grid="data")` (explicit mode: gs.Out outputs written by gs.atomic_*),
    `@gs.kernel(group=256)` (thread groups: gs.shared, gs.lane, gs.barrier),
    `@gs.kernel(group=1024, cells=8)` (8 output cells per thread, written by gs.store)."""
    _check_group(group)
    if func is None:
        return lambda f: Kernel(f, cast=cast, compat=compat, grid=grid, group=group, cells=cells)
    return Kernel(func, cast=cast, compat=compat, grid=grid, group=group, cells=cells)
