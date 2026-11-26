"""Jalon M2 : kernels avec index, dimensions nommées, plusieurs sorties, tableaux locaux.

Chaque kernel est comparé, sur chaque périphérique présent, à une référence NumPy ; le DTW est aussi
comparé au résultat du prototype GPUsync (tests/data, calculé par l'ancienne bibliothèque).
"""

from pathlib import Path

import numpy as np
import pytest

import gpusync as gs

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(2026)
DATA = Path(__file__).parent / "data"


def check(y, ref, rtol=2e-6, atol=1e-5):
    got = np.asarray(y)
    ref = np.asarray(ref)
    assert got.shape == ref.shape and got.dtype == ref.dtype, (got.shape, got.dtype, ref.shape, ref.dtype)
    if got.dtype.kind in "iu":
        bad = np.flatnonzero(got != ref)
    else:
        bad = np.flatnonzero(~np.isclose(got, ref, rtol=rtol, atol=atol))
    assert bad.size == 0, f"{bad.size} écart(s), ex. indice {bad[0]} : {got.flat[bad[0]]} au lieu de {ref.flat[bad[0]]}"


# ───────────────────────────── kernels ─────────────────────────────


@gs.kernel
def pairwise(x: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
    i, j = gs.index()
    acc = 0.0
    for k in range(x.shape[1]):
        d = x[i, k] - x[j, k]
        acc += d * d
    return gs.sqrt(acc)


@gs.kernel
def matvec(a: gs.f32["m", "k"], v: gs.f32["k"]) -> gs.f32["m"]:
    i = gs.index()
    acc = 0.0
    for c in range(len(v)):
        acc += a[i, c] * v[c]
    return acc


@gs.kernel
def transpose(a: gs.i32[:, :], b: gs.i32["c", "r"]) -> gs.i32["c", "r"]:
    # `a` sans noms : sa forme est lue avec a.shape ; `b` ne sert qu'à nommer la sortie.
    i, j = gs.index()
    return a[j, i] + b[i, j] * 0 + a.shape[-1] - a.shape[1]


@gs.kernel
def moments(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"], gs.i32["n"]):
    """Plusieurs sorties : moyenne, écart type et indice du maximum de chaque ligne."""
    i = gs.index()
    t = x.shape[1]
    s = 0.0
    s2 = 0.0
    best = 0
    for k in range(t):
        v = x[i, k]
        s += v
        s2 += v * v
        if v > x[i, best]:
            best = k
    mean = s / t
    return mean, gs.sqrt(gs.max(s2 / t - mean * mean, 0.0)), best


@gs.kernel
def split(x: gs.f32[:], y: gs.f32[:]) -> (gs.f32[:], gs.f32[:]):
    """Plusieurs sorties, élément par élément."""
    return x + y, x * y


@gs.kernel
def prefix(x: gs.f32["n", "t"]) -> gs.f32["n", "t"]:
    """Tableau local : somme cumulée de chaque ligne, puis relue à l'envers."""
    i, j = gs.index()
    buf = gs.local(gs.f32, 64)
    acc = 0.0
    for k in range(x.shape[1]):
        acc += x[i, k]
        buf[k] = acc
    return buf[j] - buf[x.shape[1] - 1 - j]


@gs.kernel
def cube(x: gs.f32["a", "b", "c"]) -> gs.f32["c", "a", "b"]:
    i, j, k = gs.index()
    return x[j, k, i] * 2.0 + gs.f32(i)


INF = 3.0e38
MAXT = 256


@gs.kernel
def dtw(x: gs.f32["n", "t"], w: gs.i32) -> gs.f32["n", "n"]:
    """DTW avec fenêtre de Sakoe-Chiba entre les séries i et j (deux lignes de la matrice dp)."""
    i, j = gs.index()
    t = x.shape[1]
    prev = gs.local(gs.f32, MAXT + 1)
    cur = gs.local(gs.f32, MAXT + 1)
    for c in range(t + 1):
        prev[c] = INF
    prev[0] = 0.0
    for a in range(1, t + 1):
        for c in range(t + 1):
            cur[c] = INF
        for b in range(gs.max(1, a - w), gs.min(t, a + w) + 1):
            d = x[i, a - 1] - x[j, b - 1]
            cur[b] = d * d + gs.min(prev[b], cur[b - 1], prev[b - 1])
        for c in range(t + 1):
            prev[c] = cur[c]
    return gs.sqrt(prev[t])


def dtw_ref(x, w):
    """Référence NumPy (même algorithme que le prototype, en float64)."""
    n, t = x.shape
    x = x.astype(np.float64)
    out = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        dp = np.full((n, t + 1, t + 1), np.inf)
        dp[:, 0, 0] = 0.0
        for a in range(1, t + 1):
            for b in range(max(1, a - w), min(t, a + w) + 1):
                cost = (x[i, a - 1] - x[:, b - 1]) ** 2
                dp[:, a, b] = cost + np.minimum(np.minimum(dp[:, a - 1, b], dp[:, a, b - 1]), dp[:, a - 1, b - 1])
        out[i] = np.sqrt(dp[:, t, t])
    return out


# ───────────────────────────── tests ─────────────────────────────


@pytest.mark.parametrize("dev", DEVICES)
def test_distances(dev):
    x = RNG.random((53, 7), dtype=np.float32)
    ref = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    check(pairwise(x, device=dev), ref.astype(np.float32))


@pytest.mark.parametrize("dev", DEVICES)
def test_matvec_shared_dimension(dev):
    a = RNG.random((300, 17), dtype=np.float32)
    v = RNG.random(17, dtype=np.float32)
    check(matvec(a, v, device=dev), (a @ v).astype(np.float32), rtol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_unnamed_dimensions_and_shape(dev):
    a = RNG.integers(-100, 100, (5, 9), dtype=np.int32)
    b = np.zeros((9, 5), dtype=np.int32)
    check(transpose(a, b, device=dev), a.T.copy())


@pytest.mark.parametrize("dev", DEVICES)
def test_multiple_outputs_with_index(dev):
    x = RNG.standard_normal((211, 13)).astype(np.float32)
    mean, std, arg = moments(x, device=dev)
    check(mean, x.mean(1), rtol=1e-5)
    check(std, x.std(1), rtol=1e-4, atol=1e-4)
    check(arg, x.argmax(1).astype(np.int32))


@pytest.mark.parametrize("dev", DEVICES)
def test_multiple_outputs_elementwise(dev):
    x = RNG.random(1001, dtype=np.float32)
    y = RNG.random(1001, dtype=np.float32)
    s, p = split(x, y, device=dev)
    check(s, x + y)
    check(p, x * y)


@pytest.mark.parametrize("dev", DEVICES)
def test_local_array(dev):
    x = RNG.random((40, 30), dtype=np.float32)
    c = np.cumsum(x, axis=1, dtype=np.float32)
    check(prefix(x, device=dev), c - c[:, ::-1], rtol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_three_dimensions(dev):
    x = RNG.random((4, 5, 6), dtype=np.float32)
    ref = np.transpose(x, (2, 0, 1)) * 2.0 + np.arange(6, dtype=np.float32)[:, None, None]
    check(cube(x, device=dev), ref.astype(np.float32))


@pytest.mark.parametrize("dev", DEVICES)
def test_dtw_like_prototype(dev):
    x = np.load(DATA / "oxygene_series.npy")
    proto = np.load(DATA / "oxygene_dtw_w5_prototype.npy")
    y = np.asarray(dtw(x, 5, device=dev))
    check(y, proto, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y[:40, :40], dtw_ref(x[:40], 5), rtol=1e-5, atol=1e-5)
    assert np.allclose(np.diag(y), 0.0) and np.allclose(y, y.T, rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_chaining_on_device(dev):
    """La sortie d'un kernel avec index sert d'entrée au suivant, sans repasser par l'hôte."""
    x = RNG.random((60, 4), dtype=np.float32)
    d = pairwise(x, device=dev)
    v = np.ones(60, dtype=np.float32)
    check(matvec(d, v, device=dev), np.asarray(d).sum(1), rtol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_out_tuple(dev):
    x = RNG.standard_normal((50, 8)).astype(np.float32)
    outs = (np.empty(50, np.float32), np.empty(50, np.float32), np.empty(50, np.int32))
    r = moments(x, device=dev, out=outs)
    for a, o in zip(r, outs):
        a.numpy()
    check(outs[0], x.mean(1), rtol=1e-5)
    check(outs[2], x.argmax(1).astype(np.int32))


def test_reductions_on_host():
    x = RNG.random((30, 3), dtype=np.float32)
    d = pairwise(x)
    ref = np.asarray(d)
    assert d.sum() == ref.sum() and d.max() == ref.max() and d.min() == 0.0
    np.testing.assert_allclose(d.mean(axis=0), ref.mean(axis=0))
    np.testing.assert_allclose(d.std(), ref.std())


def test_operators_compare_values_like_numpy():
    x = RNG.random((8, 3), dtype=np.float32)
    d, ref = pairwise(x), pairwise(x).numpy()
    # Avant : `d == 0` rendait False (comparaison d'objets) et `d + 1` levait TypeError
    eq = d == 0
    assert isinstance(eq, np.ndarray) and eq.dtype == bool and eq.sum() == 8
    assert np.array_equal(d + 1, ref + 1) and np.array_equal(1 - d, 1 - ref) and np.array_equal(-d, -ref)
    assert np.array_equal(d * d, ref * ref) and np.array_equal(d >= 0.5, ref >= 0.5)
    assert np.array_equal(np.maximum(d, 0.5), np.maximum(ref, 0.5)) and np.array_equal(ref + d, 2 * ref)
    # Lu sans être consommé : passer d à un autre kernel reste possible, et il garde son identité (clé de dict)
    assert np.allclose(matvec(d, np.ones(8, np.float32)), ref.sum(axis=1))
    assert {d: 1}[d] == 1 and d in [d]


def test_shape_errors():
    with pytest.raises(gs.ShapeMismatch, match="dimension 'k' incohérente : 'a' → 4, 'v' → 5"):
        matvec(np.zeros((3, 4), np.float32), np.zeros(5, np.float32))
    with pytest.raises(gs.ShapeMismatch, match=r"'x' doit avoir 2 dimension\(s\) \[n, t\]"):
        pairwise(np.zeros(10, np.float32))
    with pytest.raises(TypeError, match="3 tableau"):
        moments(np.zeros((3, 4), np.float32), out=np.zeros(3, np.float32))
    with pytest.raises(gs.ShapeMismatch, match="out : forme"):
        split(np.zeros(4, np.float32), np.zeros(4, np.float32), out=(np.zeros(4, np.float32), np.zeros(3, np.float32)))


def test_readable_signature():
    assert repr(moments) == "<gs.kernel moments(x: gs.f32['n', 't']) -> (gs.f32['n'], gs.f32['n'], gs.i32['n'])>"
    assert "gs_out2" in moments.source("opencl")
