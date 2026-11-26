"""Kernel types: `gs.f32`, `gs.f32[:]`, etc.

A scalar type serves at once as an annotation (`x: gs.f32[:]`, `s: gs.f32`), as a conversion inside
a kernel (`gs.i32(x)`) and as a conversion in plain Python (`gs.f32(3)` → `numpy.float32(3.0)`).
"""

from __future__ import annotations

import numpy as np


class DType:
    """Type scalaire d'un kernel."""

    __slots__ = ("name", "np")

    def __init__(self, name: str, np_dtype) -> None:
        self.name = name
        self.np = np.dtype(np_dtype)

    def __getitem__(self, dims) -> "ArrayType":
        """Array type.

        - `gs.f32[:]`: array of f32 (element-wise kernel: any shape);
        - `gs.f32["n", "t"]`: array with 2 named dimensions; the same letter in several
          parameters means the same size, and the output reuses these names (`-> gs.f32["n", "n"]`);
        - `gs.f32[:, :]`: 2 unnamed dimensions (any size, readable through `x.shape[k]`);
        - `gs.i32[10]`: fixed size (mostly for `gs.Out[gs.i32[10]]` outputs).
        """
        items = dims if isinstance(dims, tuple) else (dims,)
        out = []
        for d in items:
            if isinstance(d, slice) and d == slice(None):
                out.append(None)
            elif isinstance(d, str) and d.isidentifier():
                out.append(d)
            elif isinstance(d, int) and not isinstance(d, bool) and d >= 1:
                out.append(d)
            else:
                raise TypeError(
                    f"gs.{self.name}[{d!r}]: expected dimension ':', a name or a size ≥ 1 (e.g. gs.{self.name}[\"n\", \"t\"], gs.{self.name}[10])"
                )
        return ArrayType(self, tuple(out))

    def __call__(self, value):
        """Conversion in plain Python (outside a kernel)."""
        return self.np.type(value)

    def __repr__(self) -> str:
        return f"gs.{self.name}"

    def __eq__(self, other) -> bool:
        return isinstance(other, DType) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class ArrayType:
    """Array of elements of a scalar type (`gs.f32[:]`, `gs.f32["n", "t"]`)."""

    __slots__ = ("dtype", "dims")

    def __init__(self, dtype: DType, dims: tuple = (None,)) -> None:
        self.dtype = dtype
        self.dims = dims  # name of each dimension, None if unnamed

    def __repr__(self) -> str:
        inner = ", ".join(":" if d is None else repr(d) for d in self.dims)
        return f"gs.{self.dtype.name}[{inner}]"


class OutType:
    """Explicit-mode output (`counts: gs.Out[gs.i32[10]]`): allocated and zero-filled by the
    library, written by `gs.atomic_add` / `atomic_min` / `atomic_max`, returned as a result."""

    __slots__ = ("array", "fill")

    def __init__(self, array: ArrayType, fill=0) -> None:
        self.array = array
        self.fill = fill  # start value of each cell (0, or e.g. gs.inf for atomic_min)

    def __repr__(self) -> str:
        return f"gs.Out[{self.array!r}]" if self.fill == 0 else f"gs.Out[{self.array!r}, {self.fill!r}]"


class _Out:
    def __getitem__(self, t) -> OutType:
        """`gs.Out[gs.i32[10]]` (cells at zero) or `gs.Out[gs.f32["m"], gs.inf]` (start value)."""
        fill = 0
        if isinstance(t, tuple):
            if len(t) != 2 or not isinstance(t[1], (int, float)) or isinstance(t[1], bool):
                raise TypeError("gs.Out[type, start value]: e.g. gs.Out[gs.f32[\"m\"], gs.inf]")
            t, fill = t
        if not isinstance(t, ArrayType):
            raise TypeError("gs.Out[...] expects an array type, e.g. gs.Out[gs.i32[10]] or gs.Out[gs.f32[\"n\"]]")
        if any(d is None for d in t.dims):
            raise TypeError(f"gs.Out[{t!r}]: give the size of each dimension (name of an input dimension or integer)")
        return OutType(t, fill)

    def __repr__(self) -> str:
        return "gs.Out"


Out = _Out()


f32 = DType("f32", np.float32)
f64 = DType("f64", np.float64)
i32 = DType("i32", np.int32)
u32 = DType("u32", np.uint32)
i64 = DType("i64", np.int64)
u64 = DType("u64", np.uint64)

DTYPES = {d.name: d for d in (f32, f64, i32, u32, i64, u64)}
BY_NUMPY = {d.np: d for d in DTYPES.values()}
