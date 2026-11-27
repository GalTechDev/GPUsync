"""Large-scale weather: compared with exact solutions (rest over terrain, balance, waves, budgets)."""

import numpy as np
import pytest

import gpusync.gfx as gfx
from gpusync.gfx import weather as wx
from gpusync.gfx.atmosphere import lifting_condensation_pressure
from gpusync.gfx.kernels import kwxphys as kp
from gpusync.gfx.kernels.kweather import column_kernel


def _hill(ny, nx, height, radius):
    j, i = np.mgrid[:ny, :nx]
    return height * np.exp(-((i - nx / 2) ** 2 + (j - ny / 2) ** 2) / (2 * radius**2))


@pytest.mark.parametrize("profile", ["reference", "tropopause"])
def test_rest_over_a_mountain(profile):
    """Air at rest over a 2,500 m mountain: the spurious wind stays below 2 cm/s (reference) or 20 cm/s."""
    temp = (lambda y, p: wx.T_REF * (p / 1e5) ** wx.BETA + 0 * y) if profile == "reference" else (lambda y, p: wx.jet_temperature(0 * y, p))
    w = gfx.Weather(gfx.WeatherConfig(nx=48, ny=40, nz=24, dx=20e3, terrain=_hill(40, 48, 2500.0, 4.0), temperature=temp))
    m0 = w.surface_pressure().mean()
    w.run(4 * 3600.0)
    wind = max(np.abs(w.u.numpy()).max(), np.abs(w.v.numpy()).max())
    assert wind < (0.02 if profile == "reference" else 0.2)
    assert abs(w.surface_pressure().mean() / m0 - 1.0) < 1e-7


def test_balanced_jet_is_steady():
    w = gfx.Weather(gfx.WeatherConfig(nx=16, ny=160, nz=24, dx=20e3))
    u0 = w.u.numpy()
    assert u0.max() > 30.0
    w.run(12 * 3600.0)
    low = slice(w.config.sponge_layers, None)
    assert np.abs(w.u.numpy()[low] - u0[low]).max() < 0.6
    assert np.abs(w.v.numpy()).max() < 0.15


def test_external_wave_speed():
    """External mode of a 280 K isothermal atmosphere under a lid at 50 hPa: 308.95 m/s (exact eigenvalue)."""
    t, n, dx = 280.0, 64, 20e3
    w = gfx.Weather(
        gfx.WeatherConfig(nx=n, ny=4, dx=dx, beta=False, latitude=0.01, temperature=lambda y, p: t + 0 * p * y, sponge_layers=0, divergence_damping=0.0)
    )
    k = 2 * np.pi / (n * dx)
    eps = 100.0 / 101325.0 * np.cos(k * w.x)
    ps = 101325.0 * (1 + eps)[None, :] * np.ones((4, 1))
    p = wx.P_TOP + w.sigma[:, None, None] * (ps - wx.P_TOP)
    w.set_state(surface_pressure=ps, theta=t * (1 + wx.KAPPA * eps) * (1e5 / p) ** wx.KAPPA)
    amp, time = [], []
    for _ in range(320):
        w.step()
        amp.append(((w.surface_pressure() - 101325.0).mean(0) * np.cos(k * w.x)).sum())
        time.append(w.time)
    amp, time = np.array(amp), np.array(time)
    z = np.nonzero(amp[1:] * amp[:-1] < 0)[0]
    zero = time[z] + amp[z] / (amp[z] - amp[z + 1]) * w.dt
    speed = 2 * np.pi / k / (2 * np.diff(zero).mean())
    assert speed == pytest.approx(308.95, rel=3e-3)


def test_isa_qnh():
    """Standard atmosphere: the QNH of a field at 1,000 m under 898.75 hPa is 1013.25 hPa."""
    assert wx.sea_level_pressure_isa(89874.6, 1000.0) == pytest.approx(101325.0, abs=15.0)
    assert wx.sea_level_pressure_isa(100000.0, 0.0) == pytest.approx(100000.0)


def _stormy(nz=48):
    """Sixteen times the same unstable air (Weisman & Klemp type), ready for the convection kernel."""
    w = gfx.Weather(gfx.WeatherConfig(nx=16, ny=1, nz=nz))
    p = wx.P_TOP + w.sigma * 95000.0
    z = np.interp(-p, -1e5 * (1 - 0.0065 * np.linspace(0, 2e4, 500) / 288.15).clip(1e-3) ** 5.2559, np.linspace(0, 2e4, 500))
    th = np.where(z < 12000, 300 + 43 * (z / 12000) ** 1.25, 343 * np.exp(9.81 / (1004.64 * 213) * (z - 12000)))
    t = th * (p / 1e5) ** wx.KAPPA
    rh = np.where(z < 12000, 1 - 0.75 * (np.minimum(z, 12000) / 12000) ** 1.25, 0.25)
    q = np.minimum(gfx.Weather._vapor(t, p, rh), 0.0145)
    full = np.ones((nz, 1, 16))
    return w, p, z, th[:, None, None] * full, q[:, None, None] * full


def _convect(w, th, q, lift=8.0, dt=120.0, mass=0.05, **kw):
    nz = w.config.nz
    sfc = np.zeros((1, 16, 4))
    sfc[...] = [0.3, 1e-4, 800.0, 1.5]
    pip = np.full((1, 16), 95000.0) - w.pi0_np
    cells = 4 * nz + kp.CONVECTION_CELLS
    out = column_kernel(kp._convection_columns, cells)(
        w._up(th), w._up(q), w._up(0 * q), w._up(pip), w.pi0, w._up(sfc), w._up(np.full((1, 16), lift)), w._up(np.full((1, 16), mass)), w.lev,
        w._up(wx.plume_table(**kw)), dt, w._columns(cells),
    )  # fmt: skip
    return out.numpy()[0, 0].astype(np.float64)


def test_convection_budgets_close():
    """Water and moist static energy of the column: conserved, up to the rain on the ground."""
    w, p, _, th, q = _stormy()
    out = _convect(w, th, q)
    nz, dt = w.config.nz, 120.0
    mass = (np.diff(w.sigma_half) * 95000.0 / 9.81)[:, None]
    t0, t1 = (x * ((p / 1e5) ** wx.KAPPA)[:, None] for x in (th[:, 0], out[:nz]))
    dq = out[nz : 2 * nz] - q[:, 0]
    dl = out[2 * nz : 3 * nz]
    rain = out[4 * nz + kp.CNV_RAIN]
    assert (out[4 * nz + kp.CNV_KIND] == 2).all() and (rain > 1e-4).all()
    water = ((dq + dl) * mass).sum(0) + rain * dt
    assert np.abs(water).max() < 0.02 * rain.max() * dt
    energy = ((wx.CP * (t1 - t0) + wx.LV * dq) * mass).sum(0)
    assert np.abs(energy).max() < 0.02 * wx.LV * rain.max() * dt
    assert (dl > -1e-9).all()


def test_organized_plume_matches_the_3d_storm():
    """3D storm from the same sounding (500 m mesh, no ice): top 14.6 km, cold air -6.8 K, max w 58 m/s."""
    w, _, _, th, q = _stormy()
    out = _convect(w, th, q, ice=0.0, mass=0.2)
    o = 4 * w.config.nz
    assert out[o + kp.CNV_TOP, 0] == pytest.approx(14600.0, abs=2200.0)
    assert out[o + kp.CNV_COLD, 0] / 9.81 * 300 == pytest.approx(-6.8, abs=2.5)
    assert 30.0 < out[o + kp.CNV_WMAX, 0] < 63.0


def test_mass_flux_has_memory():
    """Mass flux: it is born, grows as long as the air is unstable, then dies out when it no longer is."""
    w, _, _, th, q = _stormy()
    o = 4 * w.config.nz
    small = _convect(w, th, q, mass=0.0)[o + kp.CNV_MASS, 0]
    big = _convect(w, th, q, mass=0.1)[o + kp.CNV_MASS, 0]
    assert 0.0 < small < 0.05 and big > 0.1
    dead = _convect(w, th, 0.2 * q, mass=0.1)
    assert dead[o + kp.CNV_MASS, 0] == 0.0
    assert dead[o + kp.CNV_MEMORY, 0] == pytest.approx(0.1 / (1.0 + 120.0 / (2.0 * wx.PLUME["dissipation"])), rel=1e-4)


_HEIGHT = np.arange(0, 20001.0, 20.0)


def _storm_sounding(q0, rh_factor):
    theta = np.where(_HEIGHT <= 12000, 300 + 43 * (np.minimum(_HEIGHT, 12000) / 12000) ** 1.25, 343 * np.exp(9.81 / (1004.64 * 213) * (_HEIGHT - 12000)))
    q = np.zeros_like(_HEIGHT)
    for _ in range(4):
        tv = theta * (1 + 0.608 * q)
        pi = 1 - np.concatenate([[0], np.cumsum(9.81 / 1004.64 * 20 * 0.5 * (1 / tv[1:] + 1 / tv[:-1]))])
        p = 1e5 * pi ** (1004.64 / 287.04)
        rh = np.where(_HEIGHT <= 12000, 1 - 0.75 * (np.minimum(_HEIGHT, 12000) / 12000) ** 1.25, 0.25)
        rh = rh * (1 - np.clip((_HEIGHT - 1500) / 1500, 0, 1) * (1 - rh_factor))
        es = 611.2 * np.exp(17.67 * (theta * pi - 273.15) / (theta * pi - 29.65))
        q = np.minimum(rh * 0.622 * es / (p - 0.378 * es), q0)
    return p, theta, q


@pytest.mark.parametrize(("q0", "rh_factor", "flux", "rain_3d", "start"), [(0.014, 1.0, 1.0, 4.48, 3.25), (0.0125, 0.85, 1.2, 2.53, 4.25)])
def test_stormy_day_rains_like_the_3d_simulation(q0, rh_factor, flux, rain_3d, start):
    """Thunderstorm day with imposed fluxes: total rain and start time, against the 3D simulation (250 m mesh).
    The first case was used for calibration, not the second."""
    nz = 36
    w = gfx.Weather(gfx.WeatherConfig(nx=16, ny=8, nz=nz, dx=20e3, physics=True, land=np.ones((8, 16)), plume={"ice": 0.0}))
    p, theta, q = _storm_sounding(q0, rh_factor)
    pm = wx.P_TOP + w.sigma * 95000.0
    full = np.ones((nz, 8, 16))
    th0, qv0 = (np.interp(-pm, -p, f)[:, None, None] * full for f in (theta, q))
    w.set_state(u=0 * full, v=0 * full, theta=th0, qv=qv0, qc=0 * full, surface_pressure=np.full((8, 16), 1e5))
    first = None
    for _ in range(210):
        day = np.sin(np.pi * min((w.time + 60) / 36000.0, 1.0))
        w.fixed_fluxes = (0.14 * flux * day, 1.5e-4 * flux * day, 0.2)
        w._physics(120.0)
        w.time += 120.0
        if first is None and w.rain.numpy().mean() > 0.05:
            first = w.time / 3600.0
    assert w.rain.numpy().mean() == pytest.approx(rain_3d, rel=0.25)
    assert first == pytest.approx(start, abs=0.8)
    assert w.convection_field(kp.CNV_MASS).max() < 0.4  # far from the safety limit


def test_dry_air_does_not_convect():
    w, _, _, th, q = _stormy()
    out = _convect(w, th, 0.2 * q)
    nz = w.config.nz
    assert (out[4 * nz + kp.CNV_KIND] == 0).all() and (out[4 * nz + kp.CNV_RAIN] == 0).all()
    assert np.array_equal(out[:nz].astype(np.float32), th[:, 0].astype(np.float32))


def test_condensation_conserves_water_and_energy():
    w, p, _, th, q = _stormy()
    nz, dt = w.config.nz, 120.0
    q = 1.6 * q  # supersaturated
    pip = np.full((1, 16), 95000.0) - w.pi0_np
    cells = 3 * nz + 1
    out = column_kernel(kp._condensation_columns, cells)(w._up(th), w._up(q), w._up(0 * q), w._up(pip), w.pi0, w.lev, dt, w._columns(cells))
    out = out.numpy()[0, 0].astype(np.float64)
    mass = (np.diff(w.sigma_half) * 95000.0 / 9.81)[:, None]
    ex = ((p / 1e5) ** wx.KAPPA)[:, None]
    dq, rain = out[nz : 2 * nz] - q[:, 0], out[3 * nz]
    assert out[2 * nz : 3 * nz].max() > 1e-4 and rain.min() > 0.0
    assert np.abs(((dq + out[2 * nz : 3 * nz]) * mass).sum(0) + rain * dt).max() < 2e-3
    heat = ((wx.CP * (out[:nz] - th[:, 0]) * ex + wx.LV * dq) * mass).sum(0)
    assert np.abs(heat).max() < 0.005 * wx.LV * np.abs((dq * mass).sum(0)).max()


def test_day_heats_the_land_and_water_budget_closes():
    """A summer day over land: warm ground in the afternoon, and water change = evaporation - rain."""
    rh = lambda y, p: np.where(p > 2e4, np.clip(0.7 - 0.5 * (1e5 - p) / 8e4, 0.15, 0.7), 0.02)
    w = gfx.Weather(
        gfx.WeatherConfig(
            nx=16, ny=8, nz=30, dx=50e3, physics=True, land=np.ones((8, 16)), relative_humidity=rh,
            temperature=lambda y, p: wx.jet_temperature(0 * y, p, south=294.0, north=294.0),
        )
    )  # fmt: skip
    t_dawn, w0, evap = w.ts.numpy().mean(), w.water(), 0.0
    for _ in range(int(8 * 3600 / w.dt)):
        before = w.steps // w.physics_every
        w.step()
        if w.steps // w.physics_every != before:
            rho = w.surface_pressure() / (wx.RD * w.temperature()[-1])
            evap += float((w.surface_field(kp.SFC_WQ) * rho).mean()) * w.dt * w.physics_every
    assert w.ts.numpy().mean() > t_dawn + 4.0
    assert w.surface_field(kp.SFC_H).mean() > 600.0
    assert evap > 1.0
    assert w.water() - w0 == pytest.approx(evap - w.rain.numpy().mean(), abs=0.03 * evap)


def test_cold_pool_spreads_at_the_gravity_current_speed():
    """Mass and buoyancy of the cold layer: the front advances, the mass only dilutes (3 h)."""
    n = 96
    h = np.zeros((n, n))
    j, i = np.mgrid[:n, :n]
    r = np.hypot(i - n / 2 + 0.5, j - n / 2 + 0.5)
    h[r < 8] = 800.0
    up = gfx.Weather(gfx.WeatherConfig(nx=16, ny=1))._up
    state = [up(h), up(0 * h), up(0 * h), up(0.15 * h)]
    wind = up(np.zeros((8, 48, 48)))
    cnv = up(np.zeros((48, 3, kp.CONVECTION_CELLS, 16)))
    for _ in range(40):
        state = list(kp.wx_cold_pool(*state, wind, wind, cnv, up(np.zeros((48, 48))), 0, 2, 1 / 2000.0, 45.0, 0))
    out = state[0].numpy()
    total = out.sum() / h.sum()
    assert total == pytest.approx(np.exp(-1800.0 / 10800.0), rel=0.03)
    front = r[out > 20.0].max() - 8.0
    speed = front * 2000.0 / 1800.0
    assert 2.0 < speed < np.sqrt(0.15 * 800.0) * 2.0
    assert np.abs(out - out[::-1]).max() < 1e-2 and np.abs(out - out.T).max() < 1e-2


def test_sounding_feeds_the_3d_simulation():
    rh = lambda y, p: np.where(p > 2e4, np.clip(0.7 - 0.5 * (1e5 - p) / 8e4, 0.15, 0.7), 0.02)
    w = gfx.Weather(gfx.WeatherConfig(nx=32, ny=64, nz=30, dx=40e3, physics=True, relative_humidity=rh))
    w.run(1800.0)
    s = w.sounding(5e5, 2e5)
    # Hydrostatics: the thickness between two levels follows the virtual temperature
    tv = s["t"] * (1 + 0.608 * s["qv"])
    dz = wx.RD / wx.GRAVITY * 0.5 * (tv[1:] + tv[:-1]) * np.log(s["p"][:-1] / s["p"][1:])
    assert np.abs(np.diff(s["z"]) / dz - 1.0).max() < 0.02
    cfg = w.local_config(5e5, 2e5, size=8e3, dx=500.0, dz=400.0, top=16e3)
    a = gfx.Atmosphere(cfg)
    k = np.searchsorted(s["z"], a.z[5])
    assert a.t.numpy()[5].mean() == pytest.approx(np.interp(a.z[5], s["z"], s["t"]), abs=0.5)
    assert a.p0[5] == pytest.approx(np.interp(a.z[5], s["z"][k - 1 : k + 1], s["p"][k - 1 : k + 1]), rel=3e-3)
    a.step(2.0)
    assert np.isfinite(a.w.numpy()).all()


def test_map_and_stability():
    terrain, land, wet = gfx.fictional_geography(64, 48, 40e3, seed=2)
    assert land.min() == 0.0 and land.max() == 1.0 and terrain.max() > 500.0 and (terrain[land < 0.5] == 0.0).all()
    rh = lambda y, p: np.where(p > 2e4, 0.6, 0.02)
    w = gfx.Weather(gfx.WeatherConfig(nx=64, ny=48, nz=30, dx=40e3, terrain=terrain, land=land, wetness=wet, physics=True, perturbation=1.0, relative_humidity=rh))
    with pytest.raises(RuntimeError, match="physics"):
        w.map(64, 48)
    w.run(6 * 3600.0)
    for mode in (0, 1):
        img = w.map(160, 120, mode=mode).numpy()
        assert img.shape == (120, 160) and ((img >> 24) == 255).all() and len(np.unique(img)) > 50
    q = w.qnh() / 100.0
    assert 940.0 < q.min() and q.max() < 1070.0
    assert np.isfinite(w.u.numpy()).all() and w.qv.numpy().min() > -1e-4


def test_bad_config_is_refused():
    with pytest.raises(ValueError, match="multiple"):
        gfx.Weather(gfx.WeatherConfig(nx=50))
    with pytest.raises(ValueError, match="nz"):
        gfx.Weather(gfx.WeatherConfig(nx=16, nz=100))
    with pytest.raises(ValueError, match="unknown"):
        wx.plume_table(bidule=1.0)


def test_parcel_ascent_on_the_storm_sounding():
    """Weisman & Klemp sounding: exact condensation level, pseudo-adiabat at nearly constant energy, and CAPE
    recomputed by an independent integration with a 5 m step."""
    p, theta, q = _storm_sounding(0.014, 1.0)
    t = theta * (p / 1e5) ** wx.KAPPA
    out = wx.parcel_ascent(p[:900], t[:900], q[:900])
    p_lcl = lifting_condensation_pressure(t[0], q[0], p[0])
    assert out["lcl"] == pytest.approx(np.interp(-p_lcl, -p, _HEIGHT), abs=15.0)
    # Independent integration in altitude (5 m step, parcel following dT/dz)
    z = np.arange(0.0, 17e3, 5.0)
    pe, te, re = (np.interp(z, _HEIGHT, f) for f in (p, t, q / (1 - q)))
    tve = te * (1 + re / 0.62197) / (1 + re)
    tp, rp, cape, seen = t[0], re[0], 0.0, False
    for k in range(1, len(z)):
        es = 611.2 * np.exp(17.67 * (tp - 273.15) / (tp - 29.65))
        rs = 0.62197 * es / (pe[k - 1] - es)
        lv = 2.501e6 - 2370.0 * (tp - 273.15)
        dlnp = np.log(pe[k] / pe[k - 1])
        if rs <= rp or seen:
            seen, rp = True, rs
            tp += dlnp * (wx.RD * tp + lv * rs) / (wx.CP + lv**2 * rs * 0.62197 / (wx.RD * tp**2))
            es = 611.2 * np.exp(17.67 * (tp - 273.15) / (tp - 29.65))
            rp = 0.62197 * es / (pe[k] - es)
        else:
            tp *= (pe[k] / pe[k - 1]) ** wx.KAPPA
        tvp = tp * (1 + rp / 0.62197) / (1 + rp)
        if seen:
            cape += max(9.81 * (tvp - tve[k]) / tve[k] * 5.0, 0.0)
    assert out["cape"] == pytest.approx(cape, rel=0.02)
    assert 1500.0 < out["cape"] < 2500.0 and -80.0 < out["cin"] < -20.0
    assert out["lcl"] < out["lfc"] < out["el"]
    assert wx.parcel_ascent(p[:900], t[:900], q[:900], source="unstable")["cape"] >= out["cape"] - 1.0


def test_parcel_ascent_limits():
    """Stable air: no CAPE. Air warmed at the surface: stronger CAPE, weaker CIN. Unknown source rejected."""
    p, theta, q = _storm_sounding(0.014, 1.0)
    t = theta * (p / 1e5) ** wx.KAPPA
    dry = wx.parcel_ascent(p[:900], t[:900], 0.1 * q[:900])
    assert dry["cape"] < 1.0
    base = wx.parcel_ascent(p[:900], t[:900], q[:900])
    warm = t.copy()
    warm[:3] += 2.0
    hot = wx.parcel_ascent(p[:900], warm[:900], q[:900])
    assert hot["cape"] > base["cape"] + 200.0 and hot["cin"] > base["cin"]
    with pytest.raises(ValueError, match="unknown"):
        wx.parcel_ascent(p[:900], t[:900], q[:900], source="bogus")


def test_precipitation_phase_follows_the_ground_temperature():
    """Snow where the air is cold to the ground, rain where it is mild; the snow is part of the precipitation."""
    terrain, land, wet = gfx.fictional_geography(64, 48, 40e3, seed=2)
    rh = lambda y, p: np.where(p > 2e4, 0.85, 0.02)
    w = gfx.Weather(gfx.WeatherConfig(nx=64, ny=48, nz=30, dx=40e3, terrain=terrain, land=land, wetness=wet, physics=True, perturbation=1.0, relative_humidity=rh))
    w.run(18 * 3600.0)
    p = w.precipitation()
    t2 = w.surface_field(wx.SFC_T2) - 273.15
    wet_cells = p["total"] > 0.2
    assert (p["frozen"] <= p["total"] + 1e-9).all() and (w.snow.numpy() <= w.rain.numpy() + 1e-3).all()
    cold, mild = wet_cells & (t2 < -3.0), wet_cells & (t2 > 6.0)
    assert cold.sum() >= 5 and mild.sum() >= 5, (cold.sum(), mild.sum())
    assert (p["frozen"][cold] / p["total"][cold]).mean() > 0.9
    assert (p["frozen"][mild] / p["total"][mild]).mean() < 0.1
