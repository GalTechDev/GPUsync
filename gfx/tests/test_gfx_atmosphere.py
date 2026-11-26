"""Moist atmosphere: compared with exact solutions (rest, waves, saturation, conservation)."""

import math

import numpy as np
import pytest

import gpusync.gfx as gfx
from gpusync.gfx import atmosphere as atm
from gpusync.gfx import sky


def _dry(**kw):
    base = dict(nx=24, ny=20, nz=32, dx=100.0, dy=100.0, dz=80.0, smagorinsky=0.18)
    return gfx.AtmosphereConfig(**{**base, **kw})


def _totals(a):
    """Total water (rain on the ground included, kg/m²) and integrated static energy."""
    col = a.rho0[:, None, None] * a.config.dz / (a.config.nx * a.config.ny)
    rain = float(a.surface_rain.numpy().astype(np.float64).mean())

    def integral(f):
        return float((f.numpy().astype(np.float64) * col).sum())

    return integral(a.qt) + integral(a.qr) + rain, integral(a.s) - atm.LV / atm.CP * rain


def test_stratified_air_stays_at_rest():
    a = gfx.Atmosphere(_dry())
    for _ in range(40):
        a.step(2.0)
    assert max(np.abs(f.numpy()).max() for f in (a.u, a.v, a.w)) < 1e-6
    assert np.abs(a.t.numpy() - a.t0[:, None, None]).max() < 1e-3


def test_uniform_wind_is_preserved():
    a = gfx.Atmosphere(_dry(smagorinsky=0.0, u=lambda z: 0 * z + 10.0, v=lambda z: 0 * z - 5.0))
    for _ in range(50):
        a.step(2.0)
    assert np.abs(a.u.numpy() - 10.0).max() < 1e-4 and np.abs(a.v.numpy() + 5.0).max() < 1e-4
    assert np.abs(a.w.numpy()).max() < 1e-5


@pytest.mark.parametrize("shape", [(16, 12, 20), (24, 1, 33), (9, 7, 5)])
def test_projection_removes_divergence(shape):
    """Direct solver: divergence-free mass flux, for arbitrary sizes."""
    nx, ny, nz = shape
    a = gfx.Atmosphere(_dry(nx=nx, ny=ny, nz=nz))
    rng = np.random.default_rng(0)
    a.u, a.v, a.w = (a._up(rng.normal(size=(nz, ny, nx))) for _ in range(3))
    before = np.abs(a.divergence()).max()
    a.u, a.v, a.w = a._project(a.u, a.v, a.w)
    assert np.abs(a.divergence()).max() < 1e-5 * before


def test_gravity_wave_frequency():
    """Standing wave: frequency from the anelastic dispersion relation."""
    n2, length, height = 1e-4, 1e4, 1e4
    cfg = gfx.AtmosphereConfig(
        nx=32, ny=1, nz=32, dx=length / 32, dy=length / 32, dz=height / 32, smagorinsky=0.0, rain=False,
        theta_l=lambda z: 300.0 * np.exp(n2 * z / atm.GRAVITY),
    )  # fmt: skip
    x, z = (np.arange(32) + 0.5) * cfg.dx, (np.arange(32) + 0.5) * cfg.dz
    mode = np.cos(2 * np.pi * x / length)[None, None, :] * np.sin(np.pi * z / height)[:, None, None]
    cfg.initial = {"theta_l": 0.05 * mode}
    a = gfx.Atmosphere(cfg)
    amp = []
    for _ in range(300):
        amp.append(float((a.b.numpy() * mode).sum()))
        a.step(5.0)
    amp = np.array(amp)
    s = np.where(np.diff(np.sign(amp)) != 0)[0]
    zero = 5.0 * (s - amp[s] / (amp[s + 1] - amp[s]))
    period = 2.0 * np.mean(np.diff(zero))
    k, m = 2 * np.pi / length, np.pi / height
    scale_height = -1.0 / np.polyfit(a.z, np.log(a.rho0), 1)[0]
    expected = 2 * np.pi / math.sqrt(n2 * k**2 / (k**2 + m**2 + 0.25 / scale_height**2))
    assert abs(period / expected - 1.0) < 0.01


def test_saturation_adjustment_matches_the_exact_solution():
    """f32 kernel against an f64 solution, from dry air to very cloudy air."""
    qt = np.linspace(0.0, 0.020, 24)
    cfg = _dry(nx=24, ny=1, nz=24, dz=300.0, theta_l=lambda z: 298.0 + 0.004 * z, qt=lambda z: 0.0 * z + 0.004)
    a = gfx.Atmosphere(cfg)
    a.qt = a._up(np.broadcast_to(qt[None, None, :], (24, 1, 24)))
    t_l = (298.0 + 0.004 * a.z) * a.exner0
    a.s = a._up((t_l + atm.GRAVITY / atm.CP * a.z - atm.S_REF)[:, None, None] * np.ones((1, 1, 24)))
    a.diagnose()
    t, qc = atm.saturation_adjust((t_l / a.exner0)[:, None] * np.ones((1, 24)), qt[None, :] * np.ones((24, 1)), a.p0[:, None])
    np.testing.assert_allclose(a.t.numpy()[:, 0, :], t, atol=3e-3)
    np.testing.assert_allclose(a.qc.numpy()[:, 0, :], qc, atol=3e-6)
    assert qc.max() > 5e-3 and (qc == 0).any()


def _sounding(z):
    """800 m mixed layer, then stable air, drier and drier."""
    th = np.where(z < 800.0, 300.0, 300.0 + 0.0035 * (z - 800.0))
    return th, np.where(z < 800.0, 0.0135, np.maximum(0.0120 * np.exp(-(z - 800.0) / 1700.0), 5e-4))


def _thermal(**kw):
    n, dx = 48, 100.0
    cfg = gfx.AtmosphereConfig(
        nx=n, ny=n, nz=50, dx=dx, dy=dx, dz=dx, surface_pressure=1e5,
        theta_l=lambda z: _sounding(z)[0], qt=lambda z: _sounding(z)[1], **kw,
    )  # fmt: skip
    x, z = (np.arange(n) + 0.5) * dx - n * dx / 2, (np.arange(50) + 0.5) * dx
    r = np.sqrt(x[None, None, :] ** 2 + x[None, :, None] ** 2 + (z[:, None, None] - 450.0) ** 2) / 600.0
    cfg.initial = {"theta_l": 1.5 * np.where(r < 1, np.cos(0.5 * np.pi * np.minimum(r, 1)) ** 2, 0.0)}
    return gfx.Atmosphere(cfg)


def test_thermal_condenses_at_the_lcl_and_conserves_water_and_energy():
    a = _thermal()
    assert a.qc.numpy().max() == 0.0
    water, energy = _totals(a)
    lcl = np.interp(-atm.lifting_condensation_pressure(float(a.t0[0]), 0.0135, float(a.p0[0])), -a.p0, a.z)
    lowest = []
    for _ in range(20):
        a.run(60.0)
        qc = a.qc.numpy()
        if qc.max() > 1e-5:
            lowest.append(a.z[np.where((qc > 1e-5).any((1, 2)))[0].min()])
            k = np.unravel_index(qc.argmax(), qc.shape)[0]
            # Never more water than the adiabatic ascent of the coldest air of the mixed layer
            assert qc.max() <= 1.03 * float(atm.saturation_adjust(300.0, 0.0135, a.p0[k])[1]) + 5e-5
    assert lowest, "no cloud"
    assert lcl - 50.0 <= min(lowest) <= lcl + 200.0
    w2, e2 = _totals(a)
    assert abs(w2 / water - 1.0) < 1e-5
    assert abs(e2 - energy) < 1e-4 * abs(energy)
    assert np.abs(a.divergence()).max() < 1e-5


def test_rain_falls_and_the_budget_closes():
    a = _thermal()
    rain = np.zeros((50, 48, 48))
    rain[10:15, 20:28, 20:28] = 3e-3
    a.qr = a._up(rain)
    a.s = a._up(a.s.numpy() - atm.LV / atm.CP * rain)
    a.diagnose()
    water, energy = _totals(a)
    a.run(900.0)
    assert a.surface_rain.numpy().max() > 0.1
    w2, e2 = _totals(a)
    assert abs(w2 / water - 1.0) < 1e-4
    assert abs(e2 - energy) < 1e-3 * abs(energy)


def test_surface_flux_warms_the_domain():
    flux = 0.05
    a = gfx.Atmosphere(_dry(surface_theta_flux=flux, surface_qt_flux=1e-4, friction_velocity=0.3))
    water, energy = _totals(a)
    a.run(300.0)
    w2, e2 = _totals(a)
    rho_s = a.rho_f[0]
    assert abs((e2 - energy) / (flux * a.exner0[0] * rho_s * 300.0) - 1.0) < 0.02
    assert abs((w2 - water) / (1e-4 * rho_s * 300.0) - 1.0) < 0.02


# ───────────────────────── Rendering ─────────────────────────


def test_path_traced_slab_matches_direct_transmission_and_energy():
    """White ground, black sky: without cloud, the ground radiance equals E·albedo·sin(h)/π; a cloud reduces it."""
    a = _thermal()
    r = gfx.SkyRenderer(a, sun_elevation=60.0, ground_albedo=(0.5, 0.5, 0.5), aerosol=0.0)
    r._sky = r.atmosphere._up(np.zeros(r._sky.shape))
    view = dict(position=(2400.0, 2400.0, 300.0), azimuth=0.0, elevation=-89.0, fov=20.0, samples=8)
    r.render(64, 48, **view)
    expected = r._sun_gpu.numpy()[4:7] * 0.5 / math.pi * math.sin(math.radians(60.0))
    np.testing.assert_allclose(r.radiance.mean((0, 1)), expected, rtol=0.03)

    qc = np.zeros((50, 48, 48))
    qc[15:22] = 1e-3
    a.qt = a._up(qc + 0.001)
    a.qc = a._up(qc)
    r.update()
    r.render(64, 48, **{**view, "samples": 64})
    # Thick homogeneous layer (reduced tau ~ 35): darkened but not black (multiple scattering)
    ratio = r.radiance.mean() / expected.mean()
    assert 0.05 < ratio < 0.6
    assert np.isfinite(r.radiance).all()


def test_sky_is_blue_and_sun_reddens_at_the_horizon():
    table = sky.sky_table(np.array([0.0, math.cos(0.8), math.sin(0.8)]), width=32, height=24)
    zenith = table[-1].mean(0)
    assert zenith[2] > 1.5 * zenith[0] and (table > 0).all()
    high = sky.sun_transmittance([0.0, math.cos(1.0), math.sin(1.0)], 0.0)
    low = sky.sun_transmittance([0.0, math.cos(0.05), math.sin(0.05)], 0.0)
    assert high.min() > 0.6 and low[0] / low[2] > 3.0 > 1.5 > high[0] / high[2]


@pytest.mark.parametrize("shape", [(160, 96, 100), (15, 8, 70), (7, 1, 12)])
def test_pressure_solver_inverts_the_discrete_operator(shape):
    """Two-pass and column transform: div(rho grad p) gives back the right-hand side, arbitrary sizes."""
    nx, ny, nz = shape
    a = gfx.Atmosphere(_dry(nx=nx, ny=ny, nz=nz, dz=30.0))
    rng = np.random.default_rng(2)
    u, v, w = (a._up(rng.normal(size=(nz, ny, nx))) for _ in range(3))
    before = np.abs(atm.atm_divergence(u, v, w, a.ref, 0.01, 0.01, 1.0 / 30.0, device=a.device).numpy()).max()
    p = a._project(u, v, w)
    after = np.abs(atm.atm_divergence(*p, a.ref, 0.01, 0.01, 1.0 / 30.0, device=a.device).numpy()).max()
    assert after < 1e-5 * before
    # Idempotent projection: the second one changes nothing more
    for f, g in zip(p, a._project(*p)):
        assert np.abs(f.numpy() - g.numpy()).max() < 1e-3


def test_scalar_advection_is_sharp_bounded_and_conservative():
    """Square pulse carried around a full turn: minimal overshoot, no loss, and fronts of a few cells."""
    n = 64
    cfg = _dry(nx=n, ny=1, nz=4, dx=100.0, dy=100.0, dz=100.0, smagorinsky=0.0, rain=False, u=lambda z: 0 * z + 10.0)
    cfg.theta_l = lambda z: 0 * z + 300.0
    a = gfx.Atmosphere(cfg)
    x = np.arange(n)
    step = np.where((x >= 20) & (x < 36), 2e-3, 1e-3)
    a.qt = a._up(step[None, None, :] * np.ones((4, 1, 1)))
    a.b = a._zero
    for _ in range(128):
        a.step(5.0)
        a.u, a.v, a.w = a._up(np.full((4, 1, n), 10.0)), a._zero, a._zero
        a.b = a._zero
    q = a.qt.numpy()[2, 0]
    assert q.min() > 0.98e-3 and q.max() < 2.02e-3  # 2 % of the step: MP is strict only at small time steps
    total = (a.qt.numpy().astype(np.float64).mean((1, 2)) * a.rho0).sum() / a.rho0.sum()
    assert abs(total / step.mean() - 1.0) < 1e-5
    assert np.abs(q - step).mean() < 0.06e-3  # mean error below 6 % of the step


# ───────────────────────── Nested grids ─────────────────────────


def _pair(wind=0.0, **kw):
    cfg = gfx.AtmosphereConfig(
        nx=24, ny=24, nz=20, dx=100.0, dy=100.0, dz=100.0, u=lambda z: 0 * z + wind, v=lambda z: 0 * z - 0.4 * wind, **kw
    )
    parent = gfx.Atmosphere(cfg)
    return parent, gfx.NestedAtmosphere(parent, size=1200.0, dx=25.0, dz=50.0)


@pytest.mark.parametrize("n", [16, 15, 7, 40])
def test_cosine_transform_matches_definition_and_inverts(n):
    from gpusync.gfx.kernels import knest

    x = np.random.default_rng(n).normal(size=(3, 5, n)).astype(np.float32)
    a, t = 2 * np.pi * np.arange(n) / n, 0.5 * np.pi * np.arange(n) / n
    trig = np.stack([np.cos(a), np.sin(a), np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    sp = atm._split(n)
    got = knest.dct_x2(*knest.dct_x1(x, trig, *sp), trig, *sp)
    k, m = np.arange(n)[:, None], np.arange(n)[None]
    np.testing.assert_allclose(got.numpy(), np.einsum("zym,km->zyk", x, np.cos(np.pi * (2 * m + 1) * k / (2 * n))), atol=2e-5)
    np.testing.assert_allclose(knest.idct_x2(*knest.idct_x1(got, trig, *sp), trig, *sp).numpy(), x, atol=5e-6)


@pytest.mark.parametrize("wind", [0.0, 9.0])
def test_nest_keeps_rest_and_uniform_wind(wind):
    """Air crosses the edges of the fine grid without creating anything."""
    _, n = _pair(wind)
    n.run(120.0)
    assert abs(n.time - 120.0) < 1e-6
    assert np.abs(n.u.numpy() - wind).max() < 1e-3 and np.abs(n.v.numpy() + 0.4 * wind).max() < 1e-3
    assert np.abs(n.w.numpy()).max() < 1e-3
    assert np.abs(n.divergence()).max() < 1e-6


def test_nest_projection_removes_divergence_with_fixed_boundary_flow():
    _, n = _pair()
    rng = np.random.default_rng(1)
    u, v, w = (rng.normal(size=n.u.shape) for _ in range(3))
    before = np.abs(atm.atm_divergence(*map(n._up, (u, v, w)), n.ref, 0.04, 0.04, 0.02).numpy()).max()
    n.u, n.v, n.w = n._project(*map(n._up, (u, v, w)))
    div = n.divergence()
    # The net flux through the edges, imposed here at random, can only spread out uniformly
    residual = div / n.rho0[:, None, None]
    assert np.abs(residual - residual.mean()).max() < 1e-5 * before
    h = gfx.nest.HALO
    np.testing.assert_allclose(n.u.numpy()[:, :, : h + 1], u[:, :, : h + 1].astype(np.float32))  # imposed edge intact


def test_nest_follows_parent_and_moves():
    parent = _thermal()
    n = gfx.NestedAtmosphere(parent, size=2400.0, dx=50.0, dz=50.0)
    n.run(600.0)
    h = gfx.nest.HALO
    fine = n.qt.numpy()[:, h:-h, h:-h].mean((1, 2))
    i0 = round((n.origin[0] + h * 50.0) / 100.0)
    coarse = parent.qt.numpy()[:, i0 : i0 + 24, i0 : i0 + 24].mean((1, 2))
    np.testing.assert_allclose(fine, np.interp(n.z, parent.z, coarse), atol=4e-4)
    assert n.w.numpy().max() > 1.0  # the thermal is indeed there
    before, origin = n.s.numpy().copy(), n.origin
    cx = origin[0] + 0.5 * n.config.nx * 50.0
    assert n.move_to(cx + 200.0, cx - 100.0)
    assert n.origin == (origin[0] + 200.0, origin[1] - 100.0)
    np.testing.assert_array_equal(n.s.numpy()[:, h + 2 : -h, h : -h - 4], before[:, h : -h - 2, h + 4 : -h])
    assert np.abs(n.divergence()).max() < 1e-5
    assert not n.move_to(cx + 200.0, cx - 100.0)
    n.run(60.0)
    assert np.isfinite(n.w.numpy()).all()


def test_nested_rendering_is_consistent():
    """Same cloud and same mesh in both grids: the nested rendering must give the same image."""
    a = _thermal()
    n = gfx.NestedAtmosphere(a, size=2400.0, dx=100.0, dz=100.0)
    view = dict(position=(2400.0, 600.0, 20.0), azimuth=0.0, elevation=25.0, fov=50.0, samples=200)
    means = []
    for cloud in (False, True):
        for grid in (a, n):
            k, j, i = np.indices((grid.config.nz, grid.config.ny, grid.config.nx))
            ox, oy = getattr(grid, "origin", (0.0, 0.0))
            x, y, z = (ox + (i + 0.5) * grid.config.dx, oy + (j + 0.5) * grid.config.dx, (k + 0.5) * grid.config.dz)
            r = np.sqrt((x - 2400.0) ** 2 + (y - 2400.0) ** 2 + (z - 1500.0) ** 2)
            qc = np.where(cloud, np.maximum(0.6e-3 * (1.0 - r / 600.0), 0.0), 0.0)
            grid.qc, grid.qt = grid._up(qc), grid._up(qc + 0.002)
        for nested in (None, n):
            rend = gfx.SkyRenderer(a, nested=nested, sun_elevation=50.0)
            rend.render(96, 64, **view)
            assert np.isfinite(rend.radiance).all()
            means.append(rend.radiance.mean((0, 1)))
    np.testing.assert_allclose(means[1], means[0], rtol=2e-3)
    np.testing.assert_allclose(means[3], means[2], rtol=0.015)
    assert means[2].mean() > 1.05 * means[0].mean()


# ───────────────────────── Real-time rendering ─────────────────────────


def _ball(grid, amount=0.6e-3, radius=600.0, center=(2400.0, 2400.0, 1500.0)):
    k, j, i = np.indices((grid.config.nz, grid.config.ny, grid.config.nx))
    ox, oy = getattr(grid, "origin", (0.0, 0.0))
    x, y, z = (ox + (i + 0.5) * grid.config.dx, oy + (j + 0.5) * grid.config.dx, (k + 0.5) * grid.config.dz)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    qc = np.maximum(amount * (1.0 - r / radius), 0.0)
    # Deficit outside the cloud, so that the edge interpolates as in a real simulation
    grid.qc, grid.qt = grid._up(qc), grid._up(qc + 0.002)


@pytest.mark.parametrize("sun_azimuth", [180.0, 90.0, 10.0])
def test_live_renderer_matches_path_tracing(sun_azimuth):
    """Back to the sun, side-lit, backlit: on the cloud pixels only, the reference radiance within 10 %
    (8 % measured on this very dense ball); the clear sky must be identical."""
    a = _thermal()
    _ball(a)
    view = ((2400.0, 600.0, 20.0), 0.0, 30.0, 50.0)
    ref = gfx.SkyRenderer(a, sun_elevation=40.0, sun_azimuth=sun_azimuth)
    ref.render(96, 64, position=view[0], azimuth=view[1], elevation=view[2], fov=view[3], samples=400)
    live = gfx.CloudRenderer(a, sun_elevation=40.0, sun_azimuth=sun_azimuth)
    live.refresh_light(150)
    cam = live.camera(*view)
    img = np.mean([np.stack([c.numpy() for c in live.march(cam, 96, 64, frame=f)], axis=-1) for f in range(8)], axis=0)
    _ball(a, amount=0.0)
    live.update()
    clear = np.stack([c.numpy() for c in live.march(cam, 96, 64)], axis=-1)
    cloud = np.abs(img - clear).sum(-1) > 0.05 * clear.sum(-1)
    assert cloud.mean() > 0.1
    np.testing.assert_allclose(img[cloud].mean(0), ref.radiance[cloud].mean(0), rtol=0.1)
    sky_only = gfx.SkyRenderer(a, sun_elevation=40.0, sun_azimuth=sun_azimuth)
    sky_only.render(96, 64, position=view[0], azimuth=view[1], elevation=view[2], fov=view[3], samples=16)
    np.testing.assert_allclose(clear.mean((0, 1)), sky_only.radiance.mean((0, 1)), rtol=2e-3)


def test_sun_depth_skips_empty_blocks_without_losing_water():
    """Homogeneous layer: exact oblique optical depth, and zero above."""
    a = _thermal()
    qc = np.zeros((50, 48, 48))
    qc[20:30] = 4e-4
    a.qc, a.qt = a._up(qc), a._up(qc + 0.002)
    r = gfx.SkyRenderer(a, sun_elevation=35.0)
    tau = r._sun_depth.numpy()
    c = a.config
    k = (np.arange(50) + 0.5) * c.dz
    column = r._scale * np.sum((a.rho0[20:30] * 4e-4) ** (2 / 3)) * c.dz / math.sin(math.radians(35.0))
    np.testing.assert_allclose(tau[:15], column, rtol=0.1)
    assert tau[:15].std() < 5e-3 * column
    assert np.all(tau[k > 31.5 * c.dz] == 0.0)


def test_light_cache_follows_the_clouds():
    """Tiles only in the cloud blocks; a step that changes nothing keeps the paths, a new cloud starts from zero."""
    a = _thermal()
    _ball(a)
    r = gfx.CloudRenderer(a)
    full = 8 * a.config.nx * a.config.ny * a.config.nz
    assert 0 < r.cache_cells < 0.4 * full
    for _ in range(8):
        r.refresh_light(1, parts=4)
    count = r._cache.numpy()[:, 6]
    used = r._tiles.numpy()[:, 3] >= 0
    assert np.all(count[used] == 2.0) and np.all(count[~used] == 0.0)
    r.update()
    r.refresh_light(1, parts=1 << 20)
    after = r._cache.numpy()[:, 6]
    assert abs(after.sum() - count.sum()) <= 64
    _ball(a, center=(3600.0, 2400.0, 1500.0))
    r.update()
    r.refresh_light(1, parts=1 << 20)
    tiles, count = r._tiles.numpy(), r._cache.numpy()[:, 6].max(axis=1)
    used = tiles[:, 3] >= 0
    far = used & (tiles[:, 0] > 2 * 34)
    assert far.any() and count[far].max() == 0.0
    assert np.isfinite(np.stack([c.numpy() for c in r.march(r.camera((2400.0, 600.0, 20.0), 20.0, 25.0), 64, 48)])).all()


def test_live_renderer_survives_a_clear_sky():
    a = _thermal()
    _ball(a, amount=0.0)
    r = gfx.CloudRenderer(a)
    r.refresh_light(2)
    img = np.stack([c.numpy() for c in r.march(r.camera((2400.0, 600.0, 20.0), 0.0, 10.0), 64, 48)])
    assert np.isfinite(img).all() and img.min() >= 0.0 and r.cache_cells == 0
