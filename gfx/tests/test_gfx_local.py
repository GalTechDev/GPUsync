"""Coupling between the weather map and the 3D clouds: tracking, moving, split steps, ice, sun."""

import numpy as np
import pytest

import gpusync.gfx as gfx
from gpusync.gfx import atmosphere as atm
from gpusync.gfx import weather as wx


def _weather(**kw):
    rh = lambda y, p: np.where(p > 2e4, np.clip(0.6 - 0.4 * (1e5 - p) / 8e4, 0.15, 0.6), 0.02)
    base = dict(nx=32, ny=32, nz=30, dx=40e3, physics=True, relative_humidity=rh, start_hour=10.0, land=np.ones((32, 32)))
    w = gfx.Weather(gfx.WeatherConfig(**{**base, **kw}))
    w.run(3600.0)
    return w


_SMALL = dict(size=8e3, dx=500.0, dz=400.0, top=16e3)


def test_split_step_is_the_same_step():
    cfg = gfx.AtmosphereConfig.trade_cumulus(size=1600.0)
    a, b = gfx.Atmosphere(cfg), gfx.Atmosphere(gfx.AtmosphereConfig.trade_cumulus(size=1600.0))
    for _ in range(5):
        a.step(4.0)
        n = sum(1 for _ in b.step_parts(4.0))
    assert n > 10 and a.time == b.time
    for name in ("u", "w", "s", "qt"):
        assert np.array_equal(getattr(a, name).numpy(), getattr(b, name).numpy())


def test_visible_state_changes_only_at_the_end_of_a_split_step():
    a = gfx.Atmosphere(gfx.AtmosphereConfig.trade_cumulus(size=1600.0))
    a.step(4.0)
    before = a.s
    parts = a.step_parts(4.0)
    for _ in range(8):
        next(parts)
    assert a.s is before
    for _ in parts:
        pass
    assert a.s is not before


def test_frame_change_keeps_the_wind():
    a = gfx.Atmosphere(gfx.AtmosphereConfig(nx=16, ny=16, nz=16, u=lambda z: 0 * z + 12.0, frame_u=5.0))
    a.set_frame(10.0, -2.0)
    m = a.level_means()
    assert np.allclose(m["u"], 12.0, atol=1e-5) and np.allclose(m["v"], 0.0, atol=1e-5)
    assert np.abs(a.u.numpy() - 2.0).max() < 1e-5


def test_level_means_and_time_step():
    a = gfx.Atmosphere(gfx.AtmosphereConfig.trade_cumulus(size=3200.0))
    a.run(300.0)
    m = a.level_means()
    assert np.allclose(m["s"], a.s.numpy().astype(np.float64).mean((1, 2)), atol=1e-4)
    assert np.allclose(m["t"], a.t.numpy().astype(np.float64).mean((1, 2)), atol=1e-2)
    rate = sum(np.abs(f.numpy()).max() / d for f, d in ((a.u, 100.0), (a.v, 100.0), (a.w, 40.0)))
    assert a.stable_dt() == pytest.approx(min(a.config.max_dt, a.config.cfl / rate), rel=1e-5)


def test_rebase_keeps_the_air_and_moves_the_pressure():
    cfg = gfx.AtmosphereConfig(nx=16, ny=16, nz=32, dz=200.0, qt=lambda z: 5e-3 * np.exp(-z / 2000))
    a = gfx.Atmosphere(cfg)
    t = a.t.numpy().mean((1, 2))
    p = a.p0.copy()
    a.rebase(a.t0, a.qv0, a.qc0, cfg.surface_pressure - 2000.0)
    assert np.abs(a.t.numpy().mean((1, 2)) - t).max() < 2e-3
    assert np.allclose(a.p0, p * (1 - 2000.0 / 1.015e5), rtol=2e-3)
    for _ in range(20):
        a.step(5.0)
    assert np.abs(a.w.numpy()).max() < 1e-4


def test_ice_saturation_matches_the_exact_adjustment():
    """Mixed saturation: temperature and cloud water compared with the double-precision numerical solution."""
    theta, q = (lambda z: 300.0 + 0.004 * z), (lambda z: 12e-3 * np.exp(-z / 3500.0))
    a = gfx.Atmosphere(gfx.AtmosphereConfig(nx=16, ny=16, nz=60, dz=200.0, theta_l=theta, qt=q, ice=True))
    t, qc = atm.saturation_adjust(theta(a.z), q(a.z), a.p0, True)
    assert np.abs(a.t.numpy().mean((1, 2)) - t).max() < 0.02
    assert np.abs(a.qc.numpy().mean((1, 2)) - qc).max() < 5e-6
    cold = t < 240.0
    assert cold.any() and (atm.saturation_mixing(t, a.p0, True)[cold] < 0.8 * atm.saturation_mixing(t, a.p0)[cold]).all()


def test_falling_ice_conserves_water_and_energy():
    theta, q = (lambda z: 300.0 + 0.004 * z), (lambda z: 8e-3 * np.exp(-z / 4500.0))
    a = gfx.Atmosphere(gfx.AtmosphereConfig(nx=16, ny=16, nz=60, dz=200.0, theta_l=theta, qt=q, ice=True, smagorinsky=0.0, rain=False))
    mass = a.rho0[:, None, None] * a.config.dz

    def totals():
        return (a.qt.numpy().astype(np.float64) * mass).sum(), (a.s.numpy().astype(np.float64) * mass).sum()

    (w0, e0), high = totals(), a.qc.numpy()[45:].sum()
    for _ in range(60):
        a.step(5.0)
    w1, e1 = totals()
    assert a.qc.numpy()[45:].sum() < 0.97 * high
    assert abs(w1 / w0 - 1.0) < 2e-5 and abs(e1 - e0) < 2e-3 * mass.sum()


def test_sun_position_at_the_solstice():
    w = gfx.Weather(gfx.WeatherConfig(nx=16, ny=16, dx=20e3, latitude=45.0, day=172, start_hour=12.0))
    el, az = w.sun_position(0.0, 0.0)
    assert el == pytest.approx(45.0 + 23.44, abs=0.1) and az == pytest.approx(180.0, abs=1.0)
    w.time = 4 * 3600.0
    el, az = w.sun_position(0.0, 0.0)
    assert 30.0 < el < 45.0 and 250.0 < az < 290.0  # in the west in the afternoon


def test_clouds_follow_the_column_of_the_map():
    w = _weather()
    loc = gfx.LocalClouds(w, 6e5, 1e5, **_SMALL)
    loc.run(3600.0)
    assert 0.0 <= w.time - loc.time <= w.dt + 1e-3
    c, m, z = loc._levels(w.sample(loc.x, loc.y)), loc.atmosphere.level_means(), loc.atmosphere.z
    low = z < 10e3
    assert np.sqrt(((m["t"] - c["t"])[low] ** 2).mean()) < 0.7
    assert np.sqrt(((m["qt"] - c["qv"] - c["qc"])[low] ** 2).mean()) < 6e-4
    assert np.hypot(m["u"] - c["u"], m["v"] - c["v"])[low].max() < 2.0
    assert loc.ground["surface_pressure"] == pytest.approx(float(loc.atmosphere.config.surface_pressure), abs=50.0)


def test_moving_takes_the_air_of_the_new_place():
    """A 600 km move to the north immediately cools the tile by the difference between the two columns."""
    w = _weather(temperature=lambda y, p: wx.jet_temperature(y, p, width=5e5))
    loc = gfx.LocalClouds(w, 6e5, -3e5, **_SMALL)
    loc.run(120.0)
    before = loc.atmosphere.level_means()["t"]
    a, b = (loc._levels(w.sample(6e5, y))["t"] for y in (-3e5, 3e5))
    loc.move_to(6e5, 3e5)
    loc.run(70.0)
    change = loc.atmosphere.level_means()["t"] - before
    k = (loc.atmosphere.z > 1500) & (loc.atmosphere.z < 8000)
    assert (b - a)[k].mean() < -5.0
    assert np.abs(change - (b - a))[k].max() < 1.5
    assert np.isfinite(loc.atmosphere.w.numpy()).all() and loc.atmosphere.qt.numpy().min() > -1e-6


def test_camera_stays_above_the_player():
    """The tile slides with the wind: the camera position in the tile moves back by as much, modulo its width."""
    w = _weather()
    loc = gfx.LocalClouds(w, 6e5, 1e5, **_SMALL)
    p0 = loc.camera_position(50.0)
    loc.run(300.0)
    p1 = loc.camera_position(50.0)
    side = loc.atmosphere.config.nx * loc.atmosphere.config.dx
    moved = np.array([loc.atmosphere.config.frame_u, loc.atmosphere.config.frame_v]) * 300.0
    gap = (p0[:2] - p1[:2] - moved + 0.5 * side) % side - 0.5 * side
    assert np.abs(gap).max() < 1.0 and p1[2] == 50.0
    assert (p1[:2] >= 0).all() and (p1[:2] < side).all()


def test_small_sun_move_keeps_the_light_cache():
    a = gfx.Atmosphere(gfx.AtmosphereConfig.trade_cumulus(size=1600.0))
    a.run(60.0)
    r = gfx.CloudRenderer(a, sun_elevation=40.0, sun_azimuth=120.0)
    data = r.sun_data(42.0, 121.0)
    r.set_sun(data=data, keep=True)
    assert r.sun[2] == pytest.approx(np.sin(np.radians(42.0)))
    img = r.march(r.camera(np.array([800.0, 0.0, 20.0]), 0.0, 10.0), 32, 24)
    assert np.isfinite(img[0].numpy()).all()


def test_refined_grid_continues_the_same_air():
    cfg = gfx.AtmosphereConfig.trade_cumulus(size=3200.0, dx=200.0)
    z = (np.arange(cfg.nz) + 0.5) * cfg.dz
    cfg.initial = {"theta_l": atm.thermals(cfg.nx, cfg.ny, z, cfg.dx, depth=300.0, amplitude=0.05)}
    a = gfx.Atmosphere(cfg)
    a.run(900.0)
    b = a.refined(2, noise=0.0)
    assert (b.config.nx, b.config.dx, b.time) == (2 * cfg.nx, 100.0, a.time)
    ma, mb = a.level_means(), b.level_means()
    assert np.abs(ma["s"] - mb["s"]).max() < 1e-3 and np.abs(ma["qt"] - mb["qt"]).max() < 1e-7
    assert np.abs(ma["u"] - mb["u"]).max() < 1e-3
    assert np.abs(b.divergence()).max() < 1e-5
    b.run(300.0)
    assert np.isfinite(b.w.numpy()).all() and np.abs(b.w.numpy()).max() < 10.0


def test_coarse_start_then_final_mesh():
    w = _weather()
    loc = gfx.LocalClouds(w, 6e5, 1e5, coarse=2, **{**_SMALL, "size": 16e3})
    assert loc.refinable and loc.atmosphere.config.dx == 1000.0
    loc.run(300.0)
    loc.refine()
    assert not loc.refinable and loc.atmosphere.config.dx == 500.0
    loc.run(180.0)
    assert np.isfinite(loc.atmosphere.w.numpy()).all()


def test_thermals_have_the_requested_size_and_strength():
    z = np.arange(20) * 100.0 + 50.0
    f = atm.thermals(64, 64, z, 100.0, depth=800.0, amplitude=0.3)
    assert f[0].std() == pytest.approx(0.3 * (1 - 50.0 / 960.0), rel=1e-6) and np.abs(f[10:]).max() == 0.0
    k = np.fft.fftfreq(64, 100.0)
    power = np.abs(np.fft.fft2(f[0])) ** 2
    kr = np.hypot(k[None], k[:, None])
    assert 1.0 / ((kr * power).sum() / power.sum()) == pytest.approx(1600.0, rel=0.3)


def test_split_nested_tick_is_the_same_tick():
    out = []
    for split in (False, True):
        a = gfx.Atmosphere(gfx.AtmosphereConfig.trade_cumulus(size=3200.0))
        n = gfx.NestedAtmosphere(a, size=1200.0, dx=50.0, dz=40.0)
        for _ in range(3):
            if split:
                assert sum(1 for _ in n.tick_parts()) > 10
            else:
                n.tick()
        out.append((n.time, n.s.numpy(), n.w.numpy()))
    assert out[0][0] == out[1][0] and all(np.array_equal(x, y) for x, y in zip(out[0][1:], out[1][1:]))


def test_weather_resumes_from_a_file(tmp_path):
    w = _weather()
    w.save(tmp_path / "etat.npz")
    r = gfx.Weather(w.config)
    r.load(tmp_path / "etat.npz")
    assert r.time == w.time and np.array_equal(r.u.numpy(), w.u.numpy()) and np.array_equal(r.pip.numpy(), w.pip.numpy())
    assert r.map(32, 24).numpy().shape == (24, 32)
    r.run(600.0)
    assert np.isfinite(r.th.numpy()).all()
    other = gfx.Weather(gfx.WeatherConfig(nx=16, ny=16, nz=30))
    with pytest.raises(ValueError, match="configuration"):
        other.load(tmp_path / "etat.npz")


def test_nest_follows_a_frame_change_and_an_air_shift():
    cfg = gfx.AtmosphereConfig(nx=32, ny=32, nz=16, dx=200.0, dz=100.0, u=lambda z: 0 * z + 12.0, v=lambda z: 0 * z + 3.0, frame_u=5.0)
    parent = gfx.Atmosphere(cfg)
    nest = gfx.NestedAtmosphere(parent, size=1600.0, dx=50.0, dz=50.0)
    nest.tick()
    before = nest.s.numpy()
    parent.set_frame(10.0, -2.0)
    parent.offset(s=np.full(cfg.nz, 0.5))
    nest.follow()
    assert (nest.config.frame_u, nest.config.frame_v) == (10.0, -2.0)
    inner = (slice(None), slice(8, -8), slice(8, -8))
    # True wind unchanged, air shifted by the same amount, no divergence added
    assert np.allclose(nest.u.numpy()[inner], 2.0, atol=1e-3) and np.allclose(nest.v.numpy()[inner], 5.0, atol=1e-3)
    assert np.allclose(nest.s.numpy() - before, 0.5, atol=1e-4)
    for _ in range(3):
        nest.tick()
    assert np.abs(nest.u.numpy()[inner] - 2.0).max() < 0.05


def test_nest_moves_by_the_shortest_way_on_a_periodic_parent():
    parent = gfx.Atmosphere(gfx.AtmosphereConfig(nx=32, ny=32, nz=8, dx=200.0, dz=200.0))
    nest = gfx.NestedAtmosphere(parent, size=1600.0, dx=50.0, dz=100.0, center=(400.0, 3200.0))
    start = nest.origin
    nest.move_to(6400.0 - 200.0, 3200.0)  # 600 m beyond the edge, not 5.8 km
    assert np.isclose(nest.origin[0] - start[0], -600.0) and nest.origin[1] == start[1]
    x, y = gfx.nest.nearest(nest, 6400.0 - 150.0, 3200.0)
    assert np.isclose(x, -150.0) and y == 3200.0


def test_local_clouds_with_a_nest_around_the_player():
    w = _weather()
    loc = gfx.LocalClouds(w, 600e3, 640e3, **_SMALL)
    loc.run(600.0)
    nest = loc.add_nest(size=1600.0, dx=100.0, dz=200.0)
    assert loc.step_length < loc.dt
    for _ in range(30):
        loc.step()
        assert loc.clock <= loc.time + 1e-6 and loc.time - loc.clock < loc.dt + 1e-6
        assert abs(w.time - loc.time) < 2.0 * w.dt  # the map advances along with the tile
    assert np.isfinite(nest.t.numpy()).all() and np.abs(nest.divergence()).max() < 1e-4
    # The player moves 3 km away: the fine grid catches up, and the camera stays inside it
    loc.move_to(loc.x + 3000.0, loc.y)
    for _ in range(3):
        loc.step()
    x, y, _ = loc.camera_position()
    c = nest.config
    assert nest.origin[0] < x < nest.origin[0] + c.nx * c.dx and nest.origin[1] < y < nest.origin[1] + c.ny * c.dy
