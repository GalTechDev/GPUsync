"""Milestone M5: explicit mode (grid=, gs.Out, atomic operations), on every device.

Real cases (decision D7): histogram, 2-D histogram, weighted sum per cell (scatter-add in
f32, atomics emulated by compare-and-swap), minimum and maximum per cell, pair counting.
"""

import math

import numpy as np
import pytest

import gpusync as gs
from gpusync._device import _set_transfer_limits

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(55)
BINS = 16

CONFIGS = {
    "default": None,
    "chunks": dict(chunk_bytes=4000, budget_bytes=200_000, direct_chunk_bytes=12_000),
}


@pytest.fixture(params=list(CONFIGS))
def limits(request):
    if CONFIGS[request.param]:
        _set_transfer_limits(**CONFIGS[request.param])
    yield request.param
    _set_transfer_limits()


@gs.kernel(grid="data")
def histogram(data: gs.f32["n"], counts: gs.Out[gs.i32[BINS]]):
    i = gs.index()
    k = int(data[i] * BINS)
    gs.atomic_add(counts, gs.min(gs.max(k, 0), BINS - 1), 1)


@gs.kernel(grid=("n",))
def histogram2d(xy: gs.f32["n", 2], h: gs.Out[gs.u32[8, 8]]):
    i = gs.index()  # grid: dimension "n" of `xy` (one thread per point)
    a = gs.min(int(xy[i, 0] * 8.0), 7)
    b = gs.min(int(xy[i, 1] * 8.0), 7)
    gs.atomic_add(h, (a, b), 1)


@gs.kernel(grid=("n",))
def per_bin(idx: gs.i32["n"], w: gs.f32["n"], total: gs.Out[gs.f32["m"]],
            lo: gs.Out[gs.f32["m"], math.inf], hi: gs.Out[gs.f32["m"], -math.inf],
            top: gs.Out[gs.i32["m"], -1], sizes: gs.i32["m"]):
    i = gs.index()
    b = idx[i]
    gs.atomic_add(total, b, w[i])
    gs.atomic_min(lo, b, w[i])
    gs.atomic_max(hi, b, w[i])
    gs.atomic_max(top, b, i)


@gs.kernel(grid=("n", "n"))
def close_pairs(p: gs.f32["n", "d"], r: gs.f32, count: gs.Out[gs.i32[1]]):
    i, j = gs.index()
    if i >= j:
        return
    acc = 0.0
    for k in range(p.shape[1]):
        t = p[i, k] - p[j, k]
        acc += t * t
    if acc < r * r:
        gs.atomic_add(count, 0, 1)


@gs.kernel(grid="data")
def unbounded(data: gs.f32["n"], counts: gs.Out[gs.i32[BINS]]):
    i = gs.index()
    gs.atomic_add(counts, int(data[i] * BINS), 1)


@pytest.mark.parametrize("dev", DEVICES)
def test_histogram(dev, limits):
    data = RNG.random(200_003, dtype=np.float32)
    want = np.bincount(np.minimum((data * BINS).astype(np.int64), BINS - 1), minlength=BINS).astype(np.int32)
    c = histogram(data, device=dev)
    assert c.shape == (BINS,) and c.dtype == np.int32
    check(c, want)


@pytest.mark.parametrize("dev", DEVICES)
def test_histogram_2d(dev, limits):
    xy = RNG.random((50_000, 2), dtype=np.float32)
    want, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=8, range=[[0, 1], [0, 1]])
    check(histogram2d(xy, device=dev), want.astype(np.uint32))


@pytest.mark.parametrize("dev", DEVICES)
def test_per_cell_floats(dev, limits):
    n, m = 30_000, 97
    idx = RNG.integers(0, m, n).astype(np.int32)
    w = RNG.standard_normal(n).astype(np.float32)
    total, lo, hi, top = per_bin(idx, w, np.zeros(m, np.int32), device=dev)
    check(total, np.bincount(idx, w, minlength=m).astype(np.float32), rtol=1e-3, atol=1e-3)
    ref_lo = np.full(m, np.inf, np.float32)
    np.minimum.at(ref_lo, idx, w)
    ref_hi = np.full(m, -np.inf, np.float32)
    np.maximum.at(ref_hi, idx, w)
    ref_top = np.full(m, -1, np.int32)
    np.maximum.at(ref_top, idx, np.arange(n, dtype=np.int32))
    check(lo, ref_lo)  # min and max are exact (no ordering rounding)
    check(hi, ref_hi)
    check(top, ref_top)


@pytest.mark.parametrize("dev", DEVICES)
def test_named_grid_and_return(dev):
    p = RNG.random((400, 3), dtype=np.float32)
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    want = int((np.triu(d2 < 0.2**2, k=1)).sum())
    assert int(np.asarray(close_pairs(p, 0.2, device=dev))[0]) == want


@pytest.mark.parametrize("dev", DEVICES)
def test_accumulate_with_out(dev):
    a = RNG.random(10_000, dtype=np.float32)
    b = RNG.random(7_000, dtype=np.float32)
    acc = np.zeros(BINS, np.int32)
    histogram(a, device=dev, out=acc).numpy()
    histogram(b, device=dev, out=acc).numpy()
    both = np.concatenate([a, b])
    want = np.bincount(np.minimum((both * BINS).astype(np.int64), BINS - 1), minlength=BINS)
    check(acc, want.astype(np.int32))


@pytest.mark.parametrize("dev", DEVICES)
def test_index_out_of_bounds(dev):
    data = RNG.random(5000, dtype=np.float32)
    data[[10, 20, 30]] = [1.0, 1.5, -0.5]  # cells 16, 24 and -8: outside [0, 16)
    c = unbounded(data, device=dev)
    with pytest.raises(gs.OutOfBounds, match=r"gs.atomic_\*\(counts.*out-of-bounds.*shape \(16,\)"):
        np.asarray(c)
    assert int(c._host.sum()) == 5000 - 3  # the valid writes were done


@pytest.mark.parametrize("dev", DEVICES)
def test_await(dev):
    import asyncio

    data = RNG.random(20_000, dtype=np.float32)

    async def main():
        return await histogram(data, device=dev)

    c = asyncio.run(main())
    assert c.ready and int(np.asarray(c).sum()) == 20_000


def test_signature_and_code():
    assert repr(histogram) == "<gs.kernel histogram(data: gs.f32['n'], counts: gs.Out[gs.i32[16]]) grid='data'>"
    cl, wg = per_bin.source("opencl"), per_bin.source("wgsl")
    assert "gs_atomic_add_f32" in cl and "atomic_cmpxchg" in cl and "atomic_max(&" in cl
    assert "atomicCompareExchangeWeak" in wg and "array<atomic<i32>>" in wg


def compile_error(src, tmp_path):
    path = tmp_path / "k.py"
    path.write_text("import gpusync as gs\n\n" + src)
    import importlib.util

    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    with pytest.raises(gs.GpusyncError) as info:
        spec.loader.exec_module(mod)
    return str(info.value)


ERRORS = [
    ("@gs.kernel\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    pass\n", ["gs.Out", "grid="]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n']) -> gs.f32['n']:\n    i = gs.index()\n    return x[i]\n", ["remove the return type"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n']):\n    pass\n", ["at least one output"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    i = gs.index()\n    c[0] = 1\n", ["gs.Out output", "gs.atomic_add(c"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    i = gs.index()\n    gs.atomic_add(c, 0, c[1])\n", ["write-only"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.f64[4]]):\n    i = gs.index()\n    gs.atomic_add(c, 0, 1.0)\n", ["f64"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4, 4]]):\n    i = gs.index()\n    gs.atomic_add(c, i, 1)\n", ["2 dimension(s), 1 index(es)"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    i = gs.index()\n    gs.atomic_add(x, 0, 1.0)\n", ["gs.Out output of the kernel"]),
    ("@gs.kernel(grid='y')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    i = gs.index()\n    gs.atomic_add(c, 0, 1)\n", ["grid=\"y\"", "not an input array"]),
    ("@gs.kernel(grid='x')\ndef k(x: gs.f32['n'], c: gs.Out[gs.i32[4]]):\n    i = gs.index()\n    return 1\n", ["return without a value"]),
    ("@gs.kernel\ndef k(x: gs.f32['n']) -> gs.f32['n']:\n    i = gs.index()\n    gs.atomic_add(x, 0, 1.0)\n    return x[i]\n", ["grid="]),
]


@pytest.mark.parametrize("src,fragments", ERRORS)
def test_errors(src, fragments, tmp_path):
    msg = compile_error(src, tmp_path)
    for f in fragments:
        assert f in msg, f"« {f} » absent de : {msg}"


def test_unknown_output_dimension():
    @gs.kernel(grid="x")
    def k(x: gs.f32["n"], c: gs.Out[gs.i32["m"]]):
        i = gs.index()
        gs.atomic_add(c, 0, 1)

    with pytest.raises(gs.ShapeMismatch, match="'m'.*'c' unknown"):
        k(np.zeros(4, np.float32))


@gs.kernel(grid="x")
def flag_multiples(x: gs.i32["n"], flags: gs.Out[gs.i32["w"], 1]):
    """The width `w` of the output appears in no input: it comes from `out=`."""
    i = gs.index()
    k = x[i]
    while k < flags.shape[0]:
        gs.atomic_min(flags, k, 0)
        k += x[i]


@pytest.mark.parametrize("dev", DEVICES)
def test_output_dimension_given_by_out(dev):
    steps = np.array([3, 5, 7], np.int32)
    flags = np.ones(100, np.int32)
    flag_multiples(steps, device=dev, out=flags).numpy()
    want = np.array([int(all(i % s for s in (3, 5, 7)) or i == 0) for i in range(100)], np.int32)
    want[0] = 1  # k starts at x[i]: index 0 is never cleared
    check(flags, want)
