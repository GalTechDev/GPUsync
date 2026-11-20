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
        """Type tableau.

        - `gs.f32[:]` : tableau de f32 (kernel élément par élément : toute forme) ;
        - `gs.f32["n", "t"]` : tableau à 2 dimensions nommées ; une même lettre dans plusieurs
          paramètres impose la même taille, et la sortie reprend ces noms (`-> gs.f32["n", "n"]`) ;
        - `gs.f32[:, :]` : 2 dimensions sans nom (taille libre, lisible via `x.shape[k]`).
        """
        items = dims if isinstance(dims, tuple) else (dims,)
        out = []
        for d in items:
            if isinstance(d, slice) and d == slice(None):
                out.append(None)
            elif isinstance(d, str) and d.isidentifier():
                out.append(d)
            else:
                raise TypeError(
                    f"gs.{self.name}[{d!r}] : dimension attendue ':' ou un nom (ex. gs.{self.name}[\"n\", \"t\"])"
                )
        return ArrayType(self, tuple(out))

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
    """Tableau d'éléments d'un type scalaire (`gs.f32[:]`, `gs.f32["n", "t"]`)."""

    __slots__ = ("dtype", "dims")

    def __init__(self, dtype: DType, dims: tuple = (None,)) -> None:
        self.dtype = dtype
        self.dims = dims  # nom de chaque dimension, None si sans nom

    def __repr__(self) -> str:
        inner = ", ".join(":" if d is None else repr(d) for d in self.dims)
        return f"gs.{self.dtype.name}[{inner}]"


f32 = DType("f32", np.float32)
f64 = DType("f64", np.float64)
i32 = DType("i32", np.int32)
u32 = DType("u32", np.uint32)

DTYPES = {d.name: d for d in (f32, f64, i32, u32)}
BY_NUMPY = {d.np: d for d in DTYPES.values()}
