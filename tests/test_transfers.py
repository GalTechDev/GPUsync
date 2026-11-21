"""Moteur de transferts (jalon M1.5, ARCHITECTURE.md §9).

Les réglages internes (taille de bloc, budget) sont réduits pour forcer, sur de petits tableaux,
chaque mode et un découpage en de nombreux blocs dont le dernier est incomplet.
"""

import warnings

import numpy as np
import pytest

import gpusync as gs
from gpusync._device import _set_transfer_limits

DEVICES = [d.id for d in gs.devices()]
N = 10_007
RNG = np.random.default_rng(99)


@gs.kernel
def affine(x: gs.f32[:], y: gs.f32[:], a: gs.f32) -> gs.f32[:]:
    return x * a + y


@gs.kernel
def int_mix(a: gs.i32[:], b: gs.i32[:]) -> gs.i32[:]:
    return a // 7 + b % 5 - a * b


@gs.kernel
def f64_axpy(x: gs.f64[:], s: gs.f64) -> gs.f64[:]:
    return x * s + 0.5


# Réglages forcés : (taille de bloc, budget, taille de bloc Direct)
CONFIGS = {
    "flux": dict(chunk_bytes=4000, budget_bytes=1, direct_chunk_bytes=12_000),
    "resident_by_blocks": dict(chunk_bytes=4000, budget_bytes=None, direct_chunk_bytes=12_000),
}


@pytest.fixture(params=list(CONFIGS))
def config(request):
    _set_transfer_limits(**CONFIGS[request.param])
    yield request.param
    _set_transfer_limits()  # valeurs par défaut


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def info(dev_id):
    return next(d for d in gs.devices() if d.id == dev_id)


def expected_mode(dev_id, config):
    d = info(dev_id)
    if d.backend == "cpu":
        return "cpu"
    if (d.backend == "opencl" and d.unified_memory) or d.backend == "metal":
        return "direct"
    return "flux" if config == "flux" else "resident"


def test_modes_and_results(device, config):
    x, y = RNG.random(N).astype(np.float32), RNG.random(N).astype(np.float32)
    r = affine(x, y, 2.5, device=device)
    assert r.mode == expected_mode(device, config)
    np.testing.assert_allclose(np.asarray(r), x * np.float32(2.5) + y, rtol=1e-6)

    a = RNG.integers(-1000, 1000, N).astype(np.int32)
    b = RNG.integers(-1000, 1000, N).astype(np.int32)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(np.asarray(int_mix(a, b, device=device)), a // 7 + b % 5 - a * b)


def test_f64_by_blocks(device, config):
    if not info(device).f64:
        pytest.skip("pas de f64 sur ce périphérique")
    x = RNG.random(N)
    np.testing.assert_allclose(np.asarray(f64_axpy(x, 1.5, device=device)), x * 1.5 + 0.5, rtol=1e-15)


def test_chaining_from_one_mode_to_another(device, config):
    x, y = RNG.random(N).astype(np.float32), RNG.random(N).astype(np.float32)
    _set_transfer_limits()  # premier appel en mode par défaut (Résident ou Direct)
    r1 = affine(x, y, 2.0, device=device)
    _set_transfer_limits(**CONFIGS[config])  # second appel en mode forcé, avec r1 en entrée
    r2 = affine(r1, y, 0.5, device=device)
    np.testing.assert_allclose(np.asarray(r2), (x * 2 + y) * np.float32(0.5) + y, rtol=1e-6)


def test_out_and_memmap(device, config, tmp_path):
    x, y = RNG.random(N).astype(np.float32), RNG.random(N).astype(np.float32)
    src = np.memmap(tmp_path / "x.f32", dtype=np.float32, mode="w+", shape=(N,))
    src[:] = x
    dst = np.memmap(tmp_path / "y.f32", dtype=np.float32, mode="w+", shape=(N,))
    r = affine(src, y, 3.0, device=device, out=dst)
    assert r.numpy() is dst  # le résultat est écrit dans la destination fournie
    dst.flush()
    np.testing.assert_allclose(np.fromfile(tmp_path / "y.f32", dtype=np.float32), x * 3 + y, rtol=1e-6)


def test_stream(device, config):
    blocks = [RNG.random(1234).astype(np.float32) for _ in range(5)]
    y = np.ones(1234, np.float32)
    got = list(affine.stream(iter(blocks), y, 2.0, device=device))
    assert len(got) == 5
    for b, g in zip(blocks, got):
        np.testing.assert_allclose(g, b * 2 + 1, rtol=1e-6)


def test_invalid_out():
    x = np.ones(10, np.float32)
    with pytest.raises(gs.ShapeMismatch, match="out"):
        affine(x, x, 1.0, device="cpu", out=np.empty(11, np.float32))
    with pytest.raises(gs.ShapeMismatch, match="out"):
        affine(x, x, 1.0, device="cpu", out=np.empty(10, np.float64))
    with pytest.raises(gs.TypeMismatch, match="contigu"):
        affine(x, x, 1.0, device="cpu", out=np.empty(20, np.float32)[::2])


def test_out_of_memory_suggests_out(monkeypatch):
    import gpusync._kernel as kmod

    def fail(*a, **k):
        raise MemoryError

    monkeypatch.setattr(kmod.np, "empty", fail)
    with pytest.raises(gs.OutOfMemory, match=r"out=np\.memmap"):
        affine(np.ones(10, np.float32), np.ones(10, np.float32), 1.0, device="cpu")


def test_public_budget():
    gs.set_memory_budget(1)
    try:
        d = next((d for d in gs.devices() if not (d.backend == "opencl" and d.unified_memory) and d.backend not in ("cpu", "metal")), None)
        if d is None:
            pytest.skip("aucun périphérique concerné par le budget")
        x = RNG.random(N).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = affine(x, x, 1.0, device=d)
        assert r.mode == "flux"
    finally:
        gs.set_memory_budget(None)
