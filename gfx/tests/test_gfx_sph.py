"""SPH solver on a sorted grid: GPU sort, densities, incompressibility, robustness."""

import numpy as np
import pytest

import gpusync as gs
import gpusync.gfx as gfx
from gpusync.gfx.kernels import ksph
from gpusync.gfx.kernels.k3d import upload_u32

GPUS = [d.id for d in gs.devices() if not d.id.startswith("cpu")]


@pytest.mark.parametrize("dev", GPUS)
@pytest.mark.parametrize("size", [1, 2, 16, 128, 256, 512, 4096, 32768])
def test_sort_keys(dev, size):
    keys = np.random.default_rng(size).integers(0, 2**32, size, dtype=np.uint32)
    got = ksph.sort_keys(upload_u32(keys, device=dev), size, device=dev).numpy()
    np.testing.assert_array_equal(got, np.sort(keys))


@pytest.mark.parametrize("dev", GPUS)
def test_sort_keys_with_duplicates_and_sorted_input(dev):
    for keys in (np.zeros(1024, np.uint32), np.arange(1024, dtype=np.uint32), np.arange(1024, dtype=np.uint32)[::-1].copy()):
        got = ksph.sort_keys(upload_u32(keys, device=dev), 1024, device=dev).numpy()
        np.testing.assert_array_equal(got, np.sort(keys))


def test_density_matches_direct_sum():
    """Sorted grid against a sum over all pairs, far from and near the walls."""
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water_sph(3000), layout="dam_break")
    for _ in range(40):
        fl.step(1.0 / 60.0)
    dims, cell, h, m = fl._sph_dims, float(fl._sph_cell), fl._sph_h, fl._sph_spacing**3
    bx, by, bz = (float(b) for b in fl.config.bounds)
    keys = ksph.sph_cell_keys(fl.pos_gpu, fl._sph_key_tpl, *dims, cell, bx, bz, 0.0, device=fl.device)
    keys = ksph.sort_keys(keys, fl._sph_keys, device=fl.device)
    start, count = ksph.sph_cell_ranges(keys, fl._sph_cell_tpl, device=fl.device)
    spos = ksph.sph_gather(fl.pos_gpu, keys, device=fl.device)
    rho = ksph.sph_density(spos, start, count, *dims, cell, h, m, bx, by, bz, 0.0, device=fl.device).numpy()
    assert int(count.numpy().sum()) == 3000

    p = spos.numpy()[:, :3].astype(np.float64)
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    q = np.sqrt(d2) / h
    spline = np.where(q < 0.5, 1.0 - 6.0 * q**2 * (1.0 - q), np.where(q < 1.0, 2.0 * (1.0 - q) ** 3, 0.0))
    w = spline.sum(1) * 8.0 / (np.pi * h**3) * m
    # Full half-space at distance `dist`: mass of the spline on the plane at distance t, integrated
    t = (np.arange(400) + 0.5) / 400
    u = 1.0 - t
    plane = np.where(t < 0.5, 16.0 * (0.0875 - t**2 * (0.5 + t**2 * (1.2 * t - 1.5))), 32.0 * u**4 * (0.25 - 0.2 * u))
    tail = np.append(np.cumsum(plane[::-1])[::-1] / 400, 0.0)
    for dist in (p[:, 0] + bx, bx - p[:, 0], p[:, 1], p[:, 2] + bz, bz - p[:, 2]):
        w += np.interp(dist / h, np.arange(401) / 400, tail)
    np.testing.assert_allclose(rho, w, rtol=2e-3)


def test_sph_keeps_volume_and_settles():
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water_sph(6000), layout="dam_break")
    for _ in range(2400):
        fl.step(1.0 / 60.0)
    p, v = fl.pos_gpu.numpy(), fl.vel_gpu.numpy()
    bx, by, bz = fl.config.bounds
    assert np.isfinite(p).all() and np.isfinite(v).all()
    assert np.all(np.abs(p[:, 0]) <= bx) and np.all(np.abs(p[:, 2]) <= bz) and p[:, 1].min() >= 0.0
    # 28 % of the basin: mean height of half of 0.28 · by
    assert abs(p[:, 1].mean() - 0.14 * by) < 0.12
    assert np.linalg.norm(v[:, :3], axis=1).mean() < 0.03
    # Lateral symmetry: nothing pushes the water along z
    assert abs(p[:, 2].mean()) < 0.06


@pytest.mark.parametrize("layout", ["dam_break", "column", "dual_wave", "rain"])
def test_sph_survives_strong_pushes(layout):
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water_sph(2000), layout=layout)
    for k in range(120):
        if k % 30 == 10:
            fl.push_wave(force_x=200.0, force_z=-120.0)
        fl.step(1.0 / 60.0)
    assert np.isfinite(fl.pos_gpu.numpy()).all() and np.isfinite(fl.vel_gpu.numpy()).all()


def test_sph_particle_limits():
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water_sph(1), layout="dam_break")
    fl.step(1.0 / 60.0)
    assert np.isfinite(fl.pos_gpu.numpy()).all()
    with pytest.raises(ValueError, match="262,144"):
        gfx.GpuFluid3D(gfx.Fluid3DConfig.water_sph((1 << 18) + 1))
