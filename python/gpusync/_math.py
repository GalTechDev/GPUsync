"""Fonctions mathématiques utilisables dans un kernel (`gs.sqrt(x)`…).

Dans un kernel, elles sont traduites en fonctions natives du GPU. Hors kernel, ce sont les
fonctions NumPy correspondantes : la fonction Python d'origine reste exécutable telle quelle.
"""

import functools as _functools
import math as _m

import numpy as _np

sqrt = _np.sqrt
exp = _np.exp
exp2 = _np.exp2
log = _np.log
log2 = _np.log2
sin = _np.sin
cos = _np.cos
tan = _np.tan
tanh = _np.tanh
atan2 = _np.arctan2
floor = _np.floor
ceil = _np.ceil
pow = _np.power
abs = _np.abs


def min(*values):
    """Minimum élément par élément de 2 valeurs ou plus (np.minimum n'en prend que 2)."""
    return _functools.reduce(_np.minimum, values)


def max(*values):
    """Maximum élément par élément de 2 valeurs ou plus (np.maximum n'en prend que 2)."""
    return _functools.reduce(_np.maximum, values)



def mulhi(a, b):
    """Partie haute du produit complet de deux entiers de même type (32 ou 64 bits) : pour des u32,
    (a * b) >> 32 calculé sans débordement. Dans un kernel : instruction native du GPU si elle existe."""
    a, b = _np.asarray(a), _np.asarray(b)
    bits = a.dtype.itemsize * 8
    full = a.astype(object) * b.astype(object)
    return _np.asarray(full >> bits).astype(a.dtype)


def index():
    """Dans un kernel avec index : coordonnées de la case de sortie calculée (`i, j = gs.index()`).

    Hors kernel, il n'y a pas de case courante : appelez le kernel, pas la fonction Python.
    """
    raise RuntimeError("gs.index() n'a de sens que dans un kernel (@gs.kernel) avec des dimensions nommées")


def local(dtype, n):
    """Tableau local à chaque case (`buf = gs.local(gs.f32, 64)`), mis à zéro, de taille constante."""
    return _np.zeros(n, dtype=dtype.np)


def atomic_add(out, index, value):
    """Dans un kernel avec grid= : out[index] += value, sans conflit entre cases de la grille."""
    _np.add.at(out, index, value)


def atomic_min(out, index, value):
    """Dans un kernel avec grid= : out[index] = min(out[index], value), sans conflit."""
    _np.minimum.at(out, index, value)


def atomic_max(out, index, value):
    """Dans un kernel avec grid= : out[index] = max(out[index], value), sans conflit."""
    _np.maximum.at(out, index, value)


pi = _m.pi
e = _m.e
tau = _m.tau
inf = _m.inf
nan = _m.nan

__all__ = [
    "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "tanh", "atan2",
    "floor", "ceil", "pow", "abs", "min", "max", "mulhi", "index", "local", "atomic_add", "atomic_min", "atomic_max", "pi", "e", "tau", "inf", "nan",
]
