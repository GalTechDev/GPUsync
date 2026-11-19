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

pi = _m.pi
e = _m.e
tau = _m.tau
inf = _m.inf
nan = _m.nan

__all__ = [
    "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "tanh", "atan2",
    "floor", "ceil", "pow", "abs", "min", "max", "pi", "e", "tau", "inf", "nan",
]
