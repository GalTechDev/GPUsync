"""64-bit integers (gs.i64, gs.u64): arithmetic, Python semantics, conversions, scalars,
implicit widening from 32 bits; compared with NumPy on every device."""

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(64)
BIG = 1 << 40


@gs.kernel
def arith(x: gs.i64[:], y: gs.i64[:], c: gs.i64) -> (gs.i64[:], gs.i64[:], gs.i64[:], gs.i64[:]):
    return x * 3 + y - c, x // y, x % y, gs.max(gs.abs(x), y)


@gs.kernel
def arith_u(x: gs.u64[:], y: gs.u64[:]) -> (gs.u64[:], gs.u64[:], gs.u64[:]):
    return x + y * 2, x // y, x % y


@gs.kernel
def widen(a: gs.i32[:], b: gs.i64[:], u: gs.u32[:]) -> (gs.i64[:], gs.u64[:], gs.f32[:], gs.i32[:]):
    # i32 + i64 → i64; u32 → u64 without a written conversion; i64 → f32; explicit i64 → i32.
    w = gs.u64(0)
    w = w + u
    return a + b, w * 3, gs.f32(b), gs.i32(b % 1000)


@gs.kernel
def cmp_select(x: gs.i64[:], y: gs.i64[:]) -> gs.i64[:]:
    return x if x > y else y - BIG


@gs.kernel
def residues(digits: gs.i32["n", "k"], d: gs.i64["n"]) -> gs.i64["n"]:
    """Remainder of a big number (base-256 digits, most significant first) modulo a divisor < 2⁴⁰."""
    i = gs.index()
    r = 0
    m = d[i]
    for j in range(digits.shape[1]):
        r = (r * 256 + digits[i, j]) % m
    return r


def ints(n, lo, hi, dtype=np.int64):
    return RNG.integers(lo, hi, n, dtype=dtype)


@pytest.mark.parametrize("dev", DEVICES)
def test_arithmetic_i64(dev):
    x = ints(5000, -BIG, BIG)
    y = ints(5000, 1, 1 << 30) * RNG.choice([-1, 1], 5000)
    c = (1 << 45) + 7
    s, q, r, m = arith(x, y, c, device=dev)
    check(s, x * 3 + y - c)
    check(q, x // y)  # Python semantics: rounded toward −∞
    check(r, x % y)  # remainder has the sign of the divisor
    check(m, np.maximum(np.abs(x), y))


@pytest.mark.parametrize("dev", DEVICES)
def test_arithmetic_u64(dev):
    x = ints(4000, 0, 1 << 62, np.uint64)
    y = ints(4000, 1, 1 << 40, np.uint64)
    s, q, r = arith_u(x, y, device=dev)
    check(s, x + y * np.uint64(2))
    check(q, x // y)
    check(r, x % y)


@pytest.mark.parametrize("dev", DEVICES)
def test_widening_and_conversions(dev):
    a = ints(3000, -(1 << 31), 1 << 31, np.int32)
    b = ints(3000, -BIG, BIG)
    u = ints(3000, 0, 1 << 32, np.uint32)
    s, w, f, t = widen(a, b, u, device=dev)
    check(s, a.astype(np.int64) + b)
    check(w, u.astype(np.uint64) * np.uint64(3))
    check(f, b.astype(np.float32))
    check(t, (b % 1000).astype(np.int32))


@pytest.mark.parametrize("dev", DEVICES)
def test_comparison_and_constant(dev):
    x = ints(2000, -BIG, BIG)
    y = ints(2000, -BIG, BIG)
    check(cmp_select(x, y, device=dev), np.where(x > y, x, y - BIG))


@pytest.mark.parametrize("dev", DEVICES)
def test_residues_of_big_number(dev):
    """The core of the deep sieve: big number (1,000 base-256 digits) modulo d < 2⁴⁰."""
    n = 300
    number = int.from_bytes(RNG.bytes(1000), "big")
    digits = np.frombuffer(number.to_bytes(1000, "big"), np.uint8).astype(np.int32)
    d = ints(n, 3, BIG)
    got = residues(np.tile(digits, (n, 1)), d, device=dev)
    check(got, np.array([number % int(v) for v in d], np.int64))


def test_messages(tmp_path):
    path = tmp_path / "k.py"
    path.write_text(
        "import gpusync as gs\n\n@gs.kernel(grid='x')\n"
        "def k(x: gs.i64['n'], c: gs.Out[gs.i64[4]]):\n    i = gs.index()\n    gs.atomic_add(c, 0, x[i])\n"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("k", path)
    with pytest.raises(gs.TypeMismatch, match="gs.Out output in i64 not supported"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    path.write_text(
        "import gpusync as gs\n\n@gs.kernel\n"
        "def k(x: gs.i32[:], y: gs.u64[:]) -> gs.u64[:]:\n    return x + y\n"
    )
    spec = importlib.util.spec_from_file_location("k2", path)
    with pytest.raises(gs.TypeMismatch, match="between i32 and u64: explicit conversion"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_scalar_out_of_range():
    with pytest.raises(gs.TypeMismatch, match="range of i64"):
        arith(np.zeros(4, np.int64), np.ones(4, np.int64), 1 << 70)
