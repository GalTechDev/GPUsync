"""Math functions usable in a kernel (`gs.sqrt(x)`…).

Inside a kernel they are translated to native GPU functions. Outside a kernel they are the matching
NumPy functions: the original Python function stays runnable as is.
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
    """Element-wise minimum of 2 values or more (np.minimum takes only 2)."""
    return _functools.reduce(_np.minimum, values)


def max(*values):
    """Element-wise maximum of 2 values or more (np.maximum takes only 2)."""
    return _functools.reduce(_np.maximum, values)



def mulhi(a, b):
    """High half of the full product of two integers of the same type (32 or 64 bits): for u32,
    (a * b) >> 32 computed without overflow. In a kernel: native GPU instruction when there is one."""
    a, b = _np.asarray(a), _np.asarray(b)
    bits = a.dtype.itemsize * 8
    full = a.astype(object) * b.astype(object)
    return _np.asarray(full >> bits).astype(a.dtype)


def index():
    """In an indexed kernel: coordinates of the output cell being computed (`i, j = gs.index()`).

    Outside a kernel there is no current cell: call the kernel, not the Python function.
    """
    raise RuntimeError("gs.index() only makes sense inside a kernel (@gs.kernel) with named dimensions")


def local(dtype, n):
    """Array local to each cell (`buf = gs.local(gs.f32, 64)`), zero-filled, of constant size."""
    return _np.zeros(n, dtype=dtype.np)


def shared(dtype, n):
    """In a kernel with group=: array shared by the group's threads (`t = gs.shared(gs.f32, 256)`),
    zero-filled, of constant size. Writing then reading a cell of another thread needs gs.barrier()."""
    return _np.zeros(n, dtype=dtype.np)


def lane():
    """In a kernel with group=: position of the thread in its group, from 0 to group − 1."""
    raise RuntimeError("gs.lane() only makes sense inside a kernel with group= (@gs.kernel(group=256))")


def barrier():
    """In a kernel with group=: waits for every thread of the group; their writes to gs.shared
    are visible afterwards. Every thread of the group must reach the same barrier."""
    raise RuntimeError("gs.barrier() only makes sense inside a kernel with group= (@gs.kernel(group=256))")


def store(k, value):
    """In a kernel with cells=: writes the thread's k-th output cell (cells start + r + k·group)."""
    raise RuntimeError("gs.store() only makes sense inside a kernel with cells= (@gs.kernel(group=1024, cells=8))")


def matmul_u8(c, a, b, m, n, k):
    """In a kernel with group=: byte matrix product, collective (the whole group, barrier
    before and after). a: m rows of k bytes, b: n columns of k bytes (4 bytes per u32 word, least
    significant byte first); c[i·n + j] += Σ_t a[i, t]·b[j, t], sums modulo 2³². Tensor Cores with CUDA
    (Ampere and later), a portable loop elsewhere."""
    ab = _np.asarray(a, dtype=_np.uint32)[: m * k // 4].view(_np.uint8).reshape(m, k).astype(_np.uint64)
    bb = _np.asarray(b, dtype=_np.uint32)[: n * k // 4].view(_np.uint8).reshape(n, k).astype(_np.uint64)
    prod = (ab @ bb.T) & 0xFFFFFFFF
    cc = _np.asarray(c)
    cc[: m * n] = ((cc[: m * n].astype(_np.uint64) + prod.reshape(-1)) & 0xFFFFFFFF).astype(cc.dtype)


def atomic_add(out, index, value):
    """In a kernel with grid=: out[index] += value, without conflicts between grid cells."""
    _np.add.at(out, index, value)


def atomic_min(out, index, value):
    """In a kernel with grid=: out[index] = min(out[index], value), without conflicts."""
    _np.minimum.at(out, index, value)


def atomic_max(out, index, value):
    """In a kernel with grid=: out[index] = max(out[index], value), without conflicts."""
    _np.maximum.at(out, index, value)


pi = _m.pi
e = _m.e
tau = _m.tau
inf = _m.inf
nan = _m.nan

__all__ = [
    "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "tanh", "atan2",
    "floor", "ceil", "pow", "abs", "min", "max", "mulhi", "index", "local", "shared", "lane", "barrier", "matmul_u8", "store", "atomic_add", "atomic_min", "atomic_max", "pi", "e", "tau", "inf", "nan",
]
