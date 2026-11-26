"""gs.batch(): same results as outside a batch, including with several threads, reads in the middle,
buffers reused by the current batch and chunked inputs."""

import threading

import numpy as np
import pytest

import gpusync as gs

GPUS = [d.id for d in gs.devices() if d.backend != "cpu"]


@gs.kernel
def inc(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    return x[i] + 1.0


@gs.kernel
def add1(x: gs.f32[:]) -> gs.f32[:]:
    return x + 1.0


@gs.kernel(grid="idx")
def hist(idx: gs.i32["m"], bins: gs.Out[gs.i32["w"], 0]):
    i = gs.index()
    gs.atomic_add(bins, idx[i], 1)


@pytest.mark.parametrize("dev", GPUS)
def test_read_in_the_middle(dev):
    x = np.arange(100_000, dtype=np.float32)
    with gs.batch(dev):
        a = inc(x, device=dev)
        for _ in range(20):
            a = inc(a, device=dev)
        assert np.array_equal(a.numpy(), x + 21)
        for _ in range(10):
            a = inc(a, device=dev)
    assert np.array_equal(a.numpy(), x + 31)


@pytest.mark.parametrize("dev", GPUS)
def test_released_inputs_keep_their_values(dev):
    outs = []
    with gs.batch(dev):
        for k in range(60):
            a = inc(np.full(5_000, k, np.float32), device=dev)
            outs.append(add1(a, device=dev))
            del a
    assert all(np.all(o.numpy() == k + 2) for k, o in enumerate(outs))


@pytest.mark.parametrize("dev", GPUS)
def test_exception_closes_the_batch(dev):
    with pytest.raises(RuntimeError, match="stop"):
        with gs.batch(dev):
            a = inc(np.ones(10, np.float32), device=dev)
            raise RuntimeError("stop")
    assert np.all(inc(a, device=dev).numpy() == 3)


@pytest.mark.parametrize("dev", GPUS)
def test_nested(dev):
    x = np.zeros(1_000, np.float32)
    with gs.batch(dev):
        a = inc(x, device=dev)
        with gs.batch(dev):
            a = inc(a, device=dev)
        a = inc(a, device=dev)
    assert np.all(a.numpy() == 3)


@pytest.mark.parametrize("dev", GPUS)
def test_several_threads(dev):
    errors = []

    def worker(seed):
        try:
            x = np.random.default_rng(seed).random(20_000).astype(np.float32)
            for _ in range(150):
                with gs.batch(dev):
                    a = inc(x, device=dev)
                    for _ in range(4):
                        a = add1(a, device=dev)
                if not np.allclose(a.numpy(), x + 5, atol=1e-5):
                    errors.append(f"thread {seed}: wrong result")
        except Exception as e:  # read back by the test
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


@pytest.mark.parametrize("dev", GPUS)
def test_threads_with_and_without_batch(dev):
    errors = []

    def worker(seed, batched):
        try:
            x = np.random.default_rng(seed).random(20_000).astype(np.float32)
            for _ in range(100):
                if batched:
                    with gs.batch(dev):
                        a = add1(inc(x, device=dev), device=dev)
                else:
                    a = add1(inc(x, device=dev), device=dev)
                if not np.allclose(a.numpy(), x + 2, atol=1e-5):
                    errors.append(f"thread {seed}: wrong result")
        except Exception as e:  # read back by the test
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(s, s % 2 == 0)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


@pytest.mark.parametrize("dev", GPUS)
def test_chunked_explicit_kernel(dev):
    """Input uploaded in small chunks into the same buffer: each chunk must wait for the previous computation."""
    idx = np.zeros(4_000_001, dtype=np.int32)
    idx[1:] = np.arange(4_000_000) // 500_000
    idx = idx[1:]  # unaligned address: no zero-copy
    gs.set_memory_budget(6 << 20)
    try:
        with gs.batch(dev):
            r = hist(idx, device=dev, out=np.zeros(8, np.int32)).numpy()
    finally:
        gs.set_memory_budget(None)
    assert r.tolist() == [500_000] * 8
