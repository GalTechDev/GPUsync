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


def shared(dtype, n):
    """Dans un kernel avec group= : tableau commun aux threads du groupe (`t = gs.shared(gs.f32, 256)`),
    mis à zéro, de taille constante. Écrire puis lire une case d'un autre thread demande gs.barrier()."""
    return _np.zeros(n, dtype=dtype.np)


def lane():
    """Dans un kernel avec group= : position du thread dans son groupe, de 0 à group − 1."""
    raise RuntimeError("gs.lane() n'a de sens que dans un kernel avec group= (@gs.kernel(group=256))")


def barrier():
    """Dans un kernel avec group= : attend tous les threads du groupe ; leurs écritures dans gs.shared
    sont ensuite visibles. Tous les threads du groupe doivent atteindre la même barrière."""
    raise RuntimeError("gs.barrier() n'a de sens que dans un kernel avec group= (@gs.kernel(group=256))")


def matmul_u8(c, a, b, m, n, k):
    """Dans un kernel avec group= : produit de matrices d'octets, collectif (tout le groupe, barrière
    avant et après). a : m lignes de k octets, b : n colonnes de k octets (4 octets par mot u32, octet de
    poids faible en premier) ; c[i·n + j] += Σ_t a[i, t]·b[j, t], sommes modulo 2³². Tensor Cores en CUDA
    (Ampere et suivants), boucle portable ailleurs."""
    ab = _np.asarray(a, dtype=_np.uint32)[: m * k // 4].view(_np.uint8).reshape(m, k).astype(_np.uint64)
    bb = _np.asarray(b, dtype=_np.uint32)[: n * k // 4].view(_np.uint8).reshape(n, k).astype(_np.uint64)
    prod = (ab @ bb.T) & 0xFFFFFFFF
    cc = _np.asarray(c)
    cc[: m * n] = ((cc[: m * n].astype(_np.uint64) + prod.reshape(-1)) & 0xFFFFFFFF).astype(cc.dtype)


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
    "floor", "ceil", "pow", "abs", "min", "max", "mulhi", "index", "local", "shared", "lane", "barrier", "matmul_u8", "atomic_add", "atomic_min", "atomic_max", "pi", "e", "tau", "inf", "nan",
]
