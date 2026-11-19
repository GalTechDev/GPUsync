"""Types des kernels : `gs.f32`, `gs.f32[:]`, etc.

Un type scalaire sert à la fois d'annotation (`x: gs.f32[:]`, `s: gs.f32`), de conversion dans
un kernel (`gs.i32(x)`) et de conversion en Python ordinaire (`gs.f32(3)` → `numpy.float32(3.0)`).
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
        """`gs.f32[:]` : tableau de f32 (kernel élément par élément, toute forme)."""
        if dims != slice(None):
            raise TypeError(
                f"gs.{self.name}[...] : seul gs.{self.name}[:] est supporté pour l'instant "
                "(les dimensions nommées arrivent au jalon M2)"
            )
        return ArrayType(self)

    def __call__(self, value):
        """Conversion en Python ordinaire (hors kernel)."""
        return self.np.type(value)

    def __repr__(self) -> str:
        return f"gs.{self.name}"

    def __eq__(self, other) -> bool:
        return isinstance(other, DType) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class ArrayType:
    """Tableau d'éléments d'un type scalaire (`gs.f32[:]`)."""

    __slots__ = ("dtype",)

    def __init__(self, dtype: DType) -> None:
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"gs.{self.dtype.name}[:]"


f32 = DType("f32", np.float32)
f64 = DType("f64", np.float64)
i32 = DType("i32", np.int32)
u32 = DType("u32", np.uint32)

DTYPES = {d.name: d for d in (f32, f64, i32, u32)}
BY_NUMPY = {d.np: d for d in DTYPES.values()}
