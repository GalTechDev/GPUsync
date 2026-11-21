"""Opérations bit à bit (&, |, ^, ~, <<, >>) et gs.mulhi, sur les quatre types entiers, comparées à
NumPy (et à Python pour les décalages au-delà de la largeur) sur chaque périphérique."""

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(7)


def has_int64(dev):
    return next(d for d in gs.devices() if d.id == dev).int64


@gs.kernel
def bits32(x: gs.i32[:], y: gs.i32[:], n: gs.i32[:]) -> (gs.i32[:], gs.i32[:], gs.i32[:], gs.i32[:], gs.i32[:], gs.i32[:], gs.i32[:]):
    return x & y, x | y, x ^ y, ~x, x << n, x >> n, (x << 3) ^ (y >> 5)


@gs.kernel
def bitsu32(x: gs.u32[:], y: gs.u32[:], n: gs.u32[:]) -> (gs.u32[:], gs.u32[:], gs.u32[:], gs.u32[:], gs.u32[:], gs.u32[:]):
    return x & 0xFFFF, x | y, x ^ y, ~x, x << n, x >> n


@gs.kernel
def bits64(x: gs.i64[:], y: gs.i64[:], n: gs.i64[:]) -> (gs.i64[:], gs.i64[:], gs.i64[:], gs.i64[:], gs.i64[:], gs.i64[:]):
    return x & y, x | y, x ^ ~y, x << n, x >> n, x << 40


@gs.kernel
def bitsu64(x: gs.u64[:], y: gs.u64[:], n: gs.u64[:]) -> (gs.u64[:], gs.u64[:], gs.u64[:], gs.u64[:]):
    return x & y, x ^ y, x << n, (x >> n) | (y >> 63)


@gs.kernel
def hi32(a: gs.u32[:], b: gs.u32[:], c: gs.i32[:], d: gs.i32[:]) -> (gs.u32[:], gs.i32[:]):
    return gs.mulhi(a, b), gs.mulhi(c, d)


@gs.kernel
def hi64(a: gs.u64[:], b: gs.u64[:], c: gs.i64[:], d: gs.i64[:]) -> (gs.u64[:], gs.i64[:]):
    return gs.mulhi(a, b), gs.mulhi(c, d)


@gs.kernel
def montgomery(a: gs.u32[:], b: gs.u32[:], p: gs.u32, pinv: gs.u32) -> gs.u32[:]:
    """Produit de Montgomery a·b·2⁻³² mod p (p < 2³¹) en arithmétique 32 bits seulement."""
    lo = a * b
    hi = gs.mulhi(a, b)
    m = lo * pinv
    t = hi - gs.mulhi(m, p)
    return t + p if t >= 0x80000000 else t


@gs.kernel
def flags(x: gs.i32[:], y: gs.i32[:]) -> gs.i32[:]:
    a = x > 0
    b = y > 0
    r = 0
    r |= 1 if a & b else 0
    r |= 2 if a | b else 0
    r |= 4 if a ^ b else 0
    r <<= 1
    return r


def shift_ref(x, n, width, left):
    """Décalage avec la sémantique de GPUsync : compte lu comme non signé, ≥ largeur → 0 / −1."""
    out = []
    for v, k in zip(x.tolist(), n.tolist()):
        k = k % (1 << width)  # non signé
        if left:
            r = (v << k) if k < width else 0
            r &= (1 << width) - 1
        else:
            r = v >> k if k < width else (-1 if v < 0 else 0)
        out.append(r)
    out = np.array(out, dtype=object)
    if left:
        out = out.astype(np.uint64 if width == 64 else np.uint32)
    return out.astype(x.dtype)


@pytest.mark.parametrize("dev", DEVICES)
def test_bits_i32_u32(dev):
    x = RNG.integers(-(1 << 31), 1 << 31, 4000, dtype=np.int32)
    y = RNG.integers(-(1 << 31), 1 << 31, 4000, dtype=np.int32)
    n = RNG.integers(-3, 40, 4000, dtype=np.int32)
    a, o, e, inv, sl, sr, mix = bits32(x, y, n, device=dev)
    check(a, x & y)
    check(o, x | y)
    check(e, x ^ y)
    check(inv, ~x)
    check(sl, shift_ref(x, n, 32, True))
    check(sr, shift_ref(x, n, 32, False))
    check(mix, (x << 3) ^ (y >> 5))

    u = x.view(np.uint32)
    v = y.view(np.uint32)
    m = n.view(np.uint32)
    a, o, e, inv, sl, sr = bitsu32(u, v, m, device=dev)
    check(a, u & 0xFFFF)
    check(o, u | v)
    check(e, u ^ v)
    check(inv, ~u)
    check(sl, shift_ref(u, m, 32, True))
    check(sr, shift_ref(u, m, 32, False))


@pytest.mark.parametrize("dev", DEVICES)
def test_bits_i64_u64(dev):
    if not has_int64(dev):
        pytest.skip("pas d'entiers 64 bits")
    x = RNG.integers(-(1 << 63), (1 << 63) - 1, 3000, dtype=np.int64)
    y = RNG.integers(-(1 << 63), (1 << 63) - 1, 3000, dtype=np.int64)
    n = RNG.integers(-2, 70, 3000, dtype=np.int64)
    a, o, e, sl, sr, c = bits64(x, y, n, device=dev)
    check(a, x & y)
    check(o, x | y)
    check(e, x ^ ~y)
    check(sl, shift_ref(x, n, 64, True))
    check(sr, shift_ref(x, n, 64, False))
    check(c, shift_ref(x, np.full_like(x, 40), 64, True))

    u, v, m = x.view(np.uint64), y.view(np.uint64), n.view(np.uint64)
    a, e, sl, sr = bitsu64(u, v, m, device=dev)
    check(a, u & v)
    check(e, u ^ v)
    check(sl, shift_ref(u, m, 64, True))
    check(sr, shift_ref(u, m, 64, False) | (v >> np.uint64(63)))


@pytest.mark.parametrize("dev", DEVICES)
def test_mulhi(dev):
    a = RNG.integers(0, 1 << 32, 4000, dtype=np.uint32)
    b = RNG.integers(0, 1 << 32, 4000, dtype=np.uint32)
    hu, hs = hi32(a, b, a.view(np.int32), b.view(np.int32), device=dev)
    check(hu, gs.mulhi(a, b))
    check(hs, gs.mulhi(a.view(np.int32), b.view(np.int32)))
    if has_int64(dev):
        c = RNG.integers(0, 1 << 64, 4000, dtype=np.uint64)
        d = RNG.integers(0, 1 << 64, 4000, dtype=np.uint64)
        hu, hs = hi64(c, d, c.view(np.int64), d.view(np.int64), device=dev)
        check(hu, gs.mulhi(c, d))
        check(hs, gs.mulhi(c.view(np.int64), d.view(np.int64)))


@pytest.mark.parametrize("dev", DEVICES)
def test_montgomery(dev):
    p = 998_244_353
    pinv = pow(p, -1, 1 << 32)  # p·pinv ≡ 1 mod 2³²
    a = RNG.integers(0, p, 5000, dtype=np.uint32)
    b = RNG.integers(0, p, 5000, dtype=np.uint32)
    r = montgomery(a, b, p, pinv, device=dev)
    rinv = pow(1 << 32, -1, p)
    ref = np.array([(int(x) * int(y) * rinv) % p for x, y in zip(a, b)], np.uint32)
    check(np.asarray(r) % p, ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_booleans(dev):
    x = RNG.integers(-5, 5, 1000, dtype=np.int32)
    y = RNG.integers(-5, 5, 1000, dtype=np.int32)
    a, b = x > 0, y > 0
    ref = ((a & b) * 1 | (a | b) * 2 | (a ^ b) * 4) << 1
    check(flags(x, y, device=dev), ref.astype(np.int32))


def test_messages(tmp_path):
    import importlib.util

    def load(src, name):
        path = tmp_path / f"{name}.py"
        path.write_text("import gpusync as gs\n\n" + src)
        spec = importlib.util.spec_from_file_location(name, path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    with pytest.raises(gs.TypeMismatch, match="attend des entiers"):
        load("@gs.kernel\ndef k(x: gs.f32[:]) -> gs.f32[:]:\n    return x & 1\n", "k1")
    with pytest.raises(gs.TypeMismatch, match="~ attend un entier"):
        load("@gs.kernel\ndef k(x: gs.f32[:]) -> gs.f32[:]:\n    return ~x\n", "k2")
    with pytest.raises(gs.TypeMismatch, match="mulhi"):
        load("@gs.kernel\ndef k(x: gs.f32[:]) -> gs.f32[:]:\n    return gs.mulhi(x, x)\n", "k3")


@gs.kernel
def literals(x: gs.i64[:]) -> gs.i64[:]:
    return x + ((1 << 40) | 0xFF) - (~5 ^ 3) + (-17 >> 2)


def test_literal_folding():
    x = np.arange(10, dtype=np.int64)
    check(literals(x, device="cpu"), x + ((1 << 40) | 0xFF) - (~5 ^ 3) + (-17 >> 2))
