"""Lifetime of host memory: sources freed or modified right after the call, intermediate results
freed without waiting, bursts of small uploads. Values must stay correct."""

import gc
import threading
import time

import numpy as np
import pytest

import gpusync as gs

GPUS = [d.id for d in gs.devices() if d.backend != "cpu"]


@gs.kernel
def inc(x: gs.f32[:]) -> gs.f32[:]:
    return x + 1.0


@gs.kernel
def slow(x: gs.f32["n"], steps: gs.i32) -> gs.f32["n"]:
    """Long enough to still be running when the host frees its arrays."""
    i = gs.index()
    v = x[i]
    for k in range(steps):
        v = v * 0.5 + x[(i + k) % x.shape[0]] * 0.5
    return v


@gs.kernel
def add(x: gs.f32["n"], y: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    return x[i] + y[i]


def aligned(n, value):
    raw = np.empty(n * 4 + 16384, dtype=np.uint8)
    off = (-raw.ctypes.data) % 16384
    a = raw[off : off + n * 4].view(np.float32)
    a[:] = value
    return a


@pytest.mark.parametrize("dev", GPUS)
def test_source_of_a_chain_released_while_running(dev):
    """Zero-copy result read by a long kernel, then freed at once; its memory is recycled and overwritten."""
    n = 1 << 18
    for k in range(12):
        src = inc(aligned(n, float(k)), device=dev)  # Direct on unified memory
        out = slow(src, 200, device=dev)
        del src
        gc.collect()
        trash = [np.full(n + 8192, -1e9, np.float32) for _ in range(4)]
        np.testing.assert_allclose(out.numpy(), k + 1.0, rtol=1e-4)
        del trash


@pytest.mark.parametrize("dev", GPUS)
def test_intermediate_results_released_while_running(dev):
    n = 1 << 18
    src = inc(aligned(n, 1.0), device=dev)
    outs = []
    for _ in range(10):
        a = slow(src, 100, device=dev)
        outs.append(inc(a, device=dev))
        del a
    del src
    gc.collect()
    trash = [np.full(n + 8192, -1e9, np.float32) for _ in range(4)]
    for o in outs:
        np.testing.assert_allclose(o.numpy(), 3.0, rtol=1e-4)
    del trash


@pytest.mark.parametrize("dev", GPUS)
def test_source_used_after_being_read(dev):
    src = inc(aligned(1 << 18, 1.0), device=dev)
    assert src.numpy()[0] == 2.0
    out = slow(src, 200, device=dev)
    del src
    gc.collect()
    trash = [np.full((1 << 18) + 8192, -1e9, np.float32) for _ in range(4)]
    np.testing.assert_allclose(out.numpy(), 2.0, rtol=1e-4)
    del trash


@pytest.mark.parametrize("dev", GPUS)
@pytest.mark.parametrize("size", [1, 17, 1_000, 70_000, 300_000])
def test_burst_of_small_uploads(dev, size):
    """More uploads in flight than the pool: each call keeps its own values, even if the caller
    frees its inputs at once."""
    base = inc(inc(np.zeros(size, np.float32), device=dev), device=dev)  # equals 2, on the device
    outs = [add(base, np.full(size, float(k), np.float32), device=dev) for k in range(300)]
    for k, o in enumerate(outs):
        assert np.all(o.numpy() == k + 2), f"call {k}"


@pytest.mark.parametrize("dev", GPUS)
def test_uploads_from_several_threads(dev):
    errors = []

    def worker(seed):
        try:
            rng = np.random.default_rng(seed)
            base = inc(inc(np.zeros(3_000, np.float32), device=dev), device=dev)
            for _ in range(60):
                xs = [rng.random(3_000).astype(np.float32) for _ in range(4)]
                outs = [add(base, x.copy(), device=dev) for x in xs]
                for x, o in zip(xs, outs):
                    if not np.allclose(o.numpy(), x + 2, atol=1e-5):
                        errors.append(f"thread {seed}: wrong result")
        except Exception as e:  # read back by the test
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_refresh_keeps_kernels_working():
    before = [d.id for d in gs.devices()]
    assert [d.id for d in gs.devices(refresh=True)] == before
    assert len(gs.backends(refresh=True)) >= 1
    assert inc(np.zeros(4, np.float32)).numpy().tolist() == [1.0] * 4


@pytest.mark.parametrize("dev", GPUS)
@pytest.mark.parametrize("n", [1 << 14, 1 << 16, 1 << 18])
def test_source_mixing_devices_released_while_running(dev, n):
    """Zero-copy source with an input coming from another device: already synchronized when created,
    it must not be freed while a kernel reads it in place."""
    other = next((d.id for d in gs.devices() if d.id != dev and d.backend != "cpu"), "cpu:0")
    ones = inc(inc(aligned(n, -1.0), device=other), device=other)
    for k in range(20):
        src = add(aligned(n, float(k)), ones, device=dev)
        a = slow(src, 3_000, device=dev)
        out = inc(a, device=dev)
        del src, a
        gc.collect()
        trash = [np.full(n + 4096, -1e9, np.float32) for _ in range(8)]
        np.testing.assert_allclose(out.numpy(), k + 2.0, rtol=1e-3)
        del trash


@gs.kernel
def _copy_small(x: gs.f32[:]) -> gs.f32[:]:
    return x


@gs.kernel
def _long_work(x: gs.f32["n"], k: gs.i32) -> gs.f32["n"]:
    i = gs.index()
    a = x[i]
    for _ in range(k):
        a = gs.sin(a) + 0.1
    return a


@pytest.mark.parametrize("device", [d.id for d in gs.devices() if d.backend == "metal"])
def test_small_page_aligned_array_is_copied(device):
    """A small array placed at the start of a page is copied: freeing it does not wait for running work."""
    raw = np.zeros(16384 * 3, dtype=np.uint8)
    off = -raw.ctypes.data % 16384
    small = raw[off : off + 768].view(np.float32)
    assert small.ctypes.data % 16384 == 0
    big = np.ones(1_000_000, dtype=np.float32)
    _copy_small(small, device=device)
    source = _copy_small(big, device=device)
    warm = _long_work(source, 5, device=device)
    gs.sync(device)
    with gs.batch(device):
        heavy = _long_work(source, 4000, device=device)
    t0 = time.perf_counter()
    r = _copy_small(small, device=device)
    assert r.mode == "resident"
    del r
    released = time.perf_counter() - t0
    gs.sync(device)
    total = time.perf_counter() - t0
    assert total > 0.005  # the heavy batch really takes time
    assert released < 0.5 * total
    assert np.isfinite(heavy.numpy()).all() and warm is not None
