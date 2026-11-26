"""Several output cells per thread (`@gs.kernel(group=g, cells=c)`, `gs.store`): each group
owns a tile of g·c cells; compared with NumPy on every device."""

import importlib.util

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(12)


@gs.kernel(group=64, cells=4)
def reverse_tile(x: gs.f32["b", "n"]) -> gs.f32["b", "n"]:
    """Each tile of 256 cells is reversed, in shared memory (4 cells per thread)."""
    i, j = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, 256)
    for k in range(4):
        t[r + 64 * k] = x[i, j + 64 * k]
    gs.barrier()
    for k in range(4):
        gs.store(k, t[255 - (r + 64 * k)] * 2.0)


@gs.kernel(group=32, cells=8)
def positions(x: gs.i32["n"]) -> gs.i32["n"]:
    """Each cell receives its own index, computed from gs.index() and k."""
    j = gs.index()
    for k in range(8):
        gs.store(k, j + 32 * k)
    return


@pytest.mark.parametrize("dev", DEVICES)
def test_reverse_tile(dev):
    x = RNG.random((5, 4 * 256), dtype=np.float32)
    ref = (x.reshape(-1, 256)[:, ::-1] * 2).reshape(x.shape)
    check(reverse_tile(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_each_cell_written_once(dev):
    n = 256 * 37
    check(positions(np.zeros(n, np.int32), device=dev), np.arange(n, dtype=np.int32))


@pytest.mark.parametrize("dev", DEVICES)
def test_many_tiles_chained(dev):
    x = RNG.random((40, 16 * 256), dtype=np.float32)
    y = reverse_tile(reverse_tile(x, device=dev), device=dev)
    check(y, x * 4)


def test_tile_size_check():
    with pytest.raises(gs.ShapeMismatch, match="whole tiles of 256"):
        reverse_tile(np.zeros((2, 100), np.float32), device="cpu")


def test_errors(tmp_path):
    def load(src, name):
        p = tmp_path / f"{name}.py"
        p.write_text("import gpusync as gs\n\n" + src)
        spec = importlib.util.spec_from_file_location(name, p)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    with pytest.raises(gs.KernelSyntaxError, match="group="):
        load('@gs.kernel(cells=4)\ndef k(x: gs.f32["n"]) -> gs.f32["n"]:\n    i = gs.index()\n    gs.store(0, x[i])\n', "c1")
    with pytest.raises(gs.KernelSyntaxError, match="return without a value"):
        load('@gs.kernel(group=32, cells=4)\ndef k(x: gs.f32["n"]) -> gs.f32["n"]:\n    i = gs.index()\n    return x[i]\n', "c2")
    with pytest.raises(gs.KernelSyntaxError, match="cells="):
        load('@gs.kernel(group=32)\ndef k(x: gs.f32["n"]) -> gs.f32["n"]:\n    i = gs.index()\n    gs.store(0, x[i])\n    return x[i]\n', "c3")
