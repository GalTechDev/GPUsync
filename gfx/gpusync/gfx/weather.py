"""Large-scale weather on the GPU: lows, fronts, wind, humidity, surface pressure and QNH.

Hydrostatic primitive equations (sigma coordinate, terrain, prognostic surface pressure), explicit Runge-Kutta 3.
Convective clouds, smaller than a cell, are represented by plumes (see `kernels/kwxphys.py`)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import gpusync as gs

from .kernels.k3d import upload_f32, upload_u32
from .kernels.kweather import (
    BETA,
    CP,
    GRAVITY,
    GROUP,
    KAPPA,
    LEVELS_MAX,
    LV,
    P_TOP,
    RD,
    T_REF,
    _dyn_columns,
    column_kernel,
    wx_momentum,
    wx_pressure,
    wx_scalar,
    wx_unpack,
    wx_unpack_surface,
)

from .kernels.kwxphys import (  # noqa: F401 (the CNV_* and SFC_* names are re-exported)
    CNV_BASE,
    CNV_CAPE,
    CNV_CIN,
    CNV_COLD,
    CNV_DOWN,
    CNV_KIND,
    CNV_LFC,
    CNV_MASS,
    CNV_MEMORY,
    CNV_RAIN,
    CNV_TOP,
    CNV_WMAX,
    CONVECTION_CELLS,
    SFC_CLOUD,
    SFC_H,
    SFC_T2,
    SFC_USTAR,
    SFC_WQ,
    SFC_WTH,
    SURFACE_CELLS,
    _condensation_columns,
    _convection_columns,
    _surface_columns,
    wx_accumulate,
    wx_cold_pool,
    wx_convective_cover,
    wx_front_lift,
    wx_map,
    wx_qnh,
    wx_sample_column,
    wx_sample_surface,
    wx_surface_inputs,
    wx_wind_tendency,
)

OMEGA = 7.292e-5


def sea_level_pressure_isa(pressure, height):
    """QNH (Pa): pressure reduced to sea level in the ICAO standard atmosphere (6.5 K/km, 15 °C)."""
    n = RD * 0.0065 / 9.80665
    return 101325.0 * ((pressure / 101325.0) ** n + 0.0065 * height / 288.15) ** (1.0 / n)


def parcel_ascent(p, t, qv, *, source: str = "surface") -> dict:
    """CAPE, CIN (J/kg) and levels of an undiluted parcel: pseudo-adiabatic ascent (water only, Bolton 1980), with
    virtual temperature.

    `p` (Pa), `t` (K), `qv` (specific humidity) from the ground upward. `source`: "surface" (first level), "mixed"
    (mean of θ and r over 100 hPa) or "unstable" (level of highest CAPE within 300 hPa of the ground). Altitudes are
    above the first level; `lfc`, `el` and `cin` are `nan` if the parcel is never buoyant."""
    p, t, qv = (np.asarray(a, dtype=np.float64) for a in (p, t, qv))
    grid = np.exp(np.arange(math.log(p[0]), math.log(max(p[-1], 1.0e4)), -0.004))
    logp = -np.log(p)
    te = np.interp(-np.log(grid), logp, t)
    r_env = np.interp(-np.log(grid), logp, qv / (1.0 - qv))
    tve = te * (1.0 + r_env / 0.62197) / (1.0 + r_env)
    z = np.concatenate([[0.0], np.cumsum(RD / GRAVITY * 0.5 * (tve[1:] + tve[:-1]) * 0.004)])

    def mixing(temp, pres):
        es = 611.2 * math.exp(17.67 * (temp - 273.15) / (temp - 29.65))
        return 0.62197 * es / max(pres - es, 1.0)

    def slope(temp, pres):  # dT / d(ln p) pseudo-adiabatique
        r = mixing(temp, pres)
        lv = 2.501e6 - 2370.0 * (temp - 273.15)
        return (RD * temp + lv * r) / (CP + lv * lv * r * 0.62197 / (RD * temp * temp))

    def lift(start: int, t0: float, r0: float) -> dict:
        n = len(grid)
        theta = t0 * (1.0e5 / grid[start]) ** KAPPA
        tp = np.full(n, np.nan)
        lcl, wet = None, False
        for m in range(start, n):
            if not wet:
                tp[m] = theta * (grid[m] / 1.0e5) ** KAPPA
                if mixing(tp[m], grid[m]) <= r0:
                    wet = True
                    # Exact condensation level: bisection between this level and the previous one
                    lo, hi = math.log(grid[m]), math.log(grid[max(m - 1, start)])
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if mixing(theta * (math.exp(mid) / 1.0e5) ** KAPPA, math.exp(mid)) <= r0:
                            lo = mid
                        else:
                            hi = mid
                    lcl = math.exp(lo)
                    temp, step = theta * (lcl / 1.0e5) ** KAPPA, math.log(grid[m] / lcl)
                    tp[m] = temp + step * slope(temp + 0.5 * step * slope(temp, lcl), math.exp(math.log(lcl) + 0.5 * step))
            else:
                a, b = tp[m - 1], math.log(grid[m - 1])
                k = slope(a - 0.002 * slope(a, grid[m - 1]), math.exp(b - 0.002))
                tp[m] = a - 0.004 * k
        rp = np.array([r0 if not np.isfinite(x) or lcl is None or q > lcl else mixing(x, q) for x, q in zip(tp, grid)])
        tvp = tp * (1.0 + rp / 0.62197) / (1.0 + rp)
        area = RD * (tvp - tve) * 0.004  # J/kg per level
        out = dict(cape=0.0, cin=math.nan, lcl=math.nan, lfc=math.nan, el=math.nan, pressure=float(grid[start]))
        if lcl is None:
            return out
        out["lcl"] = float(np.interp(-math.log(lcl), -np.log(grid), z))
        positive = np.zeros(n, dtype=bool)
        positive[start:] = (area[start:] > 0.0) & (grid[start:] <= lcl)
        if not positive.any():
            return out
        # Buoyant layers above the condensation level: the one with the largest area gives the equilibrium level
        edges = np.flatnonzero(np.diff(np.concatenate([[0], positive.astype(int), [0]])))
        layers = [(a, b, area[a:b].sum()) for a, b in zip(edges[::2], edges[1::2])]
        best = max(layers, key=lambda q: q[2])
        first = next(a for a, b, v in layers if v > 0.05 * best[2])
        out["cape"] = float(area[first : best[1]][positive[first : best[1]]].sum())
        below = area[start:first]
        out["cin"] = float(below[below < 0.0].sum())
        out["lfc"], out["el"] = float(z[first]), float(z[best[1] - 1])
        return out

    if source == "surface":
        return lift(0, float(te[0]), float(r_env[0]))
    if source == "mixed":
        low = grid >= grid[0] - 1.0e4
        theta = te * (1.0e5 / grid) ** KAPPA
        return lift(0, float(np.average(theta[low], weights=grid[low]) * (grid[0] / 1.0e5) ** KAPPA), float(np.average(r_env[low], weights=grid[low])))
    if source == "unstable":
        starts = np.flatnonzero(grid >= grid[0] - 3.0e4)
        return max((lift(int(m), float(te[m]), float(r_env[m])) for m in starts[::4]), key=lambda q: q["cape"])
    raise ValueError(f"parcel_ascent: unknown source {source!r} (surface, mixed, unstable)")


@dataclass
class WeatherConfig:
    """Domain and initial state of a `Weather`."""

    nx: int = 256
    ny: int = 192
    nz: int = 36
    dx: float = 20_000.0
    dt: float | None = None  # dynamics step (s); None: 0.9 × the gravity-wave limit
    latitude: float = 45.0  # at the domain centre
    beta: bool = True  # variation of f with latitude
    terrain: np.ndarray | None = None  # ground altitude (m), (ny, nx)
    temperature: Callable | None = None  # T(y, p), K; y in m from the centre
    relative_humidity: Callable | None = None  # RH(y, p), 0 to 1
    sea_level_pressure: float = 101_325.0
    surface_wind: float = 0.0  # geostrophic zonal wind at sea level (m/s)
    sponge_layers: int = 4
    divergence_damping: float = 0.01  # en dx² / dt
    perturbation: float = 0.0  # K: waves that seed the lows
    seed: int = 1
    physics: bool = False  # ground, sun, boundary layer, convection, rain
    convection: bool = True
    physics_step: float = 120.0  # s
    land: np.ndarray | None = None  # 1 land, 0 sea; None: wherever the terrain is above zero
    roughness: np.ndarray | float = 0.1  # over land (m)
    albedo: np.ndarray | float = 0.2
    wetness: np.ndarray | float = 0.35  # soil water availability, 0 to 1
    sea_temperature: np.ndarray | Callable | None = None  # K; None: air of the first level + 1 K
    day: int = 172  # day of the year
    start_hour: float = 6.0  # solar time at the west edge
    cooling: float = 1.5  # radiative cooling of the troposphere (K/day)
    cold_pool_refine: int = 4  # fine cold-air cells per coarse cell; 0: none
    front_lift: float = 0.5  # lifting at the edge of the cold air / propagation speed
    resolved_lift: float = 5.0  # factor of the resolved ascent (subgrid variability)
    plume: dict | None = None  # plume settings, see `PLUME`


# Settings of the convective plume, in the order of the table read by the kernel. Calibrated on two days of 3D
# thunderstorms (250 m mesh, no ice), checked on two others
PLUME = {
    "entrainment": 0.00168,  # 1/m, deep: × (1.3 - RH) (qs / qs base)³, form of Bechtold et al. 2008; calibrated on the 3D runs
    "entrainment_shallow": 8e-4,  # 1/m, calibrated on the 3D BOMEX case
    "buoyancy": 0.54,  # share of buoyancy that accelerates the air; calibrated on the 3D runs
    "drag": 1.0,  # braking by entrainment
    "rain_rate": 0.000592,  # 1/m
    "rain_threshold": 0.000946,  # kg/kg
    "downdraft": 0.383,  # downdraft flux / flux at the base
    "ice": 1.0,  # 1: mixed water-ice saturation below 0 °C; 0: water only
    "dissipation": 1880.0,  # s (Pan & Randall 1998, form from memory)
    "inertia": 1.17e+07,  # m⁴/kg
    "w_min": 0.5,  # m/s
    "rain_evaporation": 3.09e-05,
    "downdraft_rh": 0.816,  # relative humidity of downdrafts
    "mass_max": 0.5,  # kg/m²/s
    "downdraft_mix": 0.1,  # share of cloud air at the start of downdrafts
    "growth": 8.79e-06,  # growth of the mass flux (1/m)
    "excess": 2.0,  # excess of the thermals, in flux / w*
    "thermal": 0.6,  # speed of the thermals, in w*
    "sustain": 5.0,  # lifting sustained by existing storms (m/s)
    "narrow": 3.52,  # extra dilution before organization
    "organized": 0.013,  # organization mass flux (kg/m²/s)
    "rain_evaporation_cloud": 2.61e-05,
}


def plume_table(**kw) -> np.ndarray:
    """Table of the plume settings (`PLUME` modified by `kw`)."""
    unknown = set(kw) - set(PLUME)
    if unknown:
        raise ValueError(f"plume: unknown setting {sorted(unknown)}")
    out = np.zeros(24)
    out[: len(PLUME)] = list({**PLUME, **kw}.values())
    return out


def jet_temperature(y, p, *, south=297.0, north=271.0, width=1.1e6, lapse=0.0065, stratosphere=215.0):
    """Zonal temperature of a jet stream: tanh north-south contrast in the troposphere, isothermal stratosphere."""
    surface = 0.5 * (south + north) - 0.5 * (south - north) * np.tanh(y / width)
    tropo = surface * (p / 1.0e5) ** (RD * lapse / GRAVITY)
    soft = 4.0
    return stratosphere + soft * np.logaddexp(0.0, (tropo - stratosphere) / soft)


def fictional_geography(nx: int, ny: int, dx: float, *, seed: int = 3, land_fraction: float = 0.5, mountains: float = 2500.0):
    """Invented geography, periodic in x: `(terrain in m, land mask, soil water availability)`."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(nx, dx)[None, :]
    ky = np.fft.fftfreq(2 * ny, dx)[:, None]
    k = np.sqrt(kx**2 + ky**2)

    def field(scale, slope):
        amp = np.where(k > 0, (1.0 + (k * scale) ** 2) ** (-slope / 2), 0.0)
        f = np.fft.ifft2(np.fft.fft2(rng.normal(size=(2 * ny, nx))) * amp).real[:ny]
        return (f - f.mean()) / f.std()

    base = field(1.5e6, 4.0)
    sea = np.quantile(base, 1.0 - land_fraction)
    land = (base > sea).astype(float)
    inland = np.clip((base - sea) / 1.0, 0.0, 1.0)
    ridge = np.clip(1.0 - np.abs(field(1.0e6, 4.0)) * 2.2, 0.0, 1.0)
    terrain = land * (30.0 + 350.0 * inland + mountains * ridge**1.5 * np.clip(inland * 2.5, 0.0, 1.0))
    # Two smoothing passes: no one-cell step
    for _ in range(2):
        pad = np.pad(terrain, ((1, 1), (0, 0)), mode="edge")
        terrain = 0.5 * terrain + 0.125 * (pad[2:] + pad[:-2] + np.roll(terrain, 1, 1) + np.roll(terrain, -1, 1))
    return terrain * np.maximum(land, 0.0), land, np.clip(0.35 + 0.15 * field(8e5, 4.0), 0.1, 0.8)


class Weather:
    """Large-scale atmosphere (channel periodic in x)."""

    def __init__(self, config: WeatherConfig | None = None, *, device: str | None = None) -> None:
        self.config = cfg = config or WeatherConfig()
        self.device = device
        if cfg.nx % GROUP or cfg.nx < GROUP:
            raise ValueError(f"Weather: nx must be a multiple of {GROUP}, got {cfg.nx}")
        if not 4 <= cfg.nz <= LEVELS_MAX:
            raise ValueError(f"Weather: nz between 4 and {LEVELS_MAX}, got {cfg.nz}")
        self.time, self.steps = 0.0, 0
        self._levels()
        self.dt = cfg.dt or 0.9 * 1.73 / (2.0 * math.sqrt(2.0)) * cfg.dx / 350.0
        self.x = (np.arange(cfg.nx) + 0.5) * cfg.dx
        self.y = (np.arange(cfg.ny) + 0.5 - 0.5 * cfg.ny) * cfg.dx
        f0 = 2.0 * OMEGA * math.sin(math.radians(cfg.latitude))
        slope = 2.0 * OMEGA * math.cos(math.radians(cfg.latitude)) / 6.371e6 if cfg.beta else 0.0
        self.f = f0 + slope * self.y
        self.terrain = np.zeros((cfg.ny, cfg.nx)) if cfg.terrain is None else np.asarray(cfg.terrain, dtype=np.float64)
        p_s0 = 1.0e5 * (1.0 - BETA * GRAVITY * self.terrain / (RD * T_REF)) ** (1.0 / BETA)
        self.pi0_np = p_s0 - P_TOP
        self.pi0 = self._up(self.pi0_np)
        self.coriolis = self._up(self.f)
        self._tpl2 = self._up(np.zeros((cfg.ny, cfg.nx)))
        self._tpl3 = self._up(np.zeros((cfg.nz, cfg.ny, cfg.nx)))
        self._dyn = column_kernel(_dyn_columns, 2 * cfg.nz + 1)
        self._dyn_tpl = self._columns(2 * cfg.nz + 1)
        self._initial()
        if cfg.physics:
            self._physics_setup()

    def _up(self, a) -> gs.Array:
        return upload_f32(np.ascontiguousarray(a, dtype=np.float32), device=self.device)

    def _columns(self, cells: int) -> gs.Array:
        return self._up(np.zeros((self.config.ny, self.config.nx // GROUP, cells, GROUP)))

    # ── Mise en place ──

    def _levels(self) -> None:
        cfg = self.config
        zeta = (np.arange(cfg.nz) + 0.5) / cfg.nz  # 0 at the ground
        w = np.minimum(1.0, 0.1 + 2.6 * zeta) * (1.0 - 0.55 * zeta**2)
        d = (w / w.sum())[::-1]
        half = np.concatenate([[0.0], np.cumsum(d)])
        half[-1] = 1.0
        self.sigma_half, self.sigma = half, 0.5 * (half[1:] + half[:-1])
        table = np.zeros((cfg.nz + 1, 4))
        table[:, 0] = half
        table[:-1, 1], table[:-1, 2] = self.sigma, np.diff(half)
        n = cfg.sponge_layers
        for k in range(n):
            table[k, 3] = 2e-4 * ((n - k) / n) ** 2  # diffusion horizontale (1/s)
        self.lev_np, self.lev = table, self._up(table)

    def _initial(self) -> None:
        """Hydrostatic and geostrophic balance of a temperature `T(y, p)` and a sea-level pressure."""
        cfg = self.config
        temp = cfg.temperature or jet_temperature
        rh = cfg.relative_humidity
        yf = np.concatenate([self.y - 0.5 * cfg.dx, [self.y[-1] + 0.5 * cfg.dx]])  # faces sud
        lnp = np.linspace(math.log(P_TOP * 0.8), math.log(1.09e5), 1200)
        pg = np.exp(lnp)

        def tv_of(y):
            t = temp(y[:, None], pg[None, :])
            return t, (t if rh is None else t * (1.0 + 0.608 * self._vapor(t, pg[None, :], rh(y[:, None], pg[None, :]))))

        def geopotential(y):
            """Φ(y, p) with Φ = 0 at the sea-level pressure."""
            tv = tv_of(y)[1]
            up = np.concatenate([np.zeros((len(y), 1)), np.cumsum(0.5 * (tv[:, 1:] + tv[:, :-1]) * np.diff(lnp), axis=1)], axis=1) * RD
            slp = self._slp(y)
            at = np.array([np.interp(math.log(s), lnp, row) for s, row in zip(slp, up)])
            return at[:, None] - up

        phi_c = geopotential(self.y)
        # Surface pressure: Φ(y, p_surface) = g · terrain
        ps = np.empty((cfg.ny, cfg.nx))
        for j in range(cfg.ny):
            ps[j] = np.exp(np.interp(GRAVITY * self.terrain[j], phi_c[j, ::-1], lnp[::-1]))
        pi = ps - P_TOP
        p = P_TOP + self.sigma[:, None, None] * pi
        t = np.empty_like(p)
        table = temp(self.y[:, None], pg[None, :])
        for j in range(cfg.ny):
            t[:, j] = np.interp(np.log(p[:, j]), lnp, table[j])
        q = np.zeros_like(p) if rh is None else self._vapor(t, p, rh(self.y[None, :, None] * np.ones_like(p), p))
        th = t * (1.0e5 / p) ** KAPPA
        if cfg.perturbation:
            rng = np.random.default_rng(cfg.seed)
            shape = np.exp(-((self.y / 7e5) ** 2))[:, None]
            wave = sum(
                np.cos(2.0 * np.pi * m * self.x / (cfg.nx * cfg.dx) + rng.uniform(0, 2 * np.pi)) / m for m in range(1, 5)
            )[None, :] * shape  # fmt: skip
            t = t + cfg.perturbation * wave * np.exp(-(((self.sigma - 0.75) / 0.3) ** 2))[:, None, None]
            th = t * (1.0e5 / p) ** KAPPA
        # Geostrophic wind on isobaric surfaces
        phi_f = geopotential(yf)
        ug = -(phi_f[1:] - phi_f[:-1]) / cfg.dx / self.f[:, None]
        pu = 0.5 * (p + np.roll(p, 1, axis=2))
        u = np.empty_like(p)
        for j in range(cfg.ny):
            u[:, j] = np.interp(np.log(pu[:, j]), lnp, ug[j])
        self.set_state(u=u, v=np.zeros_like(u), theta=th, qv=q, qc=np.zeros_like(u), surface_pressure=ps)

    def _slp(self, y):
        cfg = self.config
        rho = cfg.sea_level_pressure / (RD * 285.0)
        f = np.interp(y, self.y, self.f)
        out = cfg.sea_level_pressure - rho * f * cfg.surface_wind * y
        return out

    @staticmethod
    def _vapor(t, p, rh):
        es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        e = np.minimum(rh * es, 0.5 * p)
        return 0.62197 * e / (p - 0.37803 * e)

    def set_state(self, **fields) -> None:
        """Replaces fields: `u`, `v`, `theta`, `qv`, `qc` (nz, ny, nx) and `surface_pressure` (ny, nx, Pa)."""
        names = {"u": "u", "v": "v", "theta": "th", "qv": "qv", "qc": "qc"}
        for key, value in fields.items():
            if key == "surface_pressure":
                self.pip = self._up(np.asarray(value, dtype=np.float64) - P_TOP - self.pi0_np)
            else:
                setattr(self, names[key], self._up(value))

    def save(self, path) -> None:
        """Writes the full state (npz), to be resumed later by `load` with the same configuration."""
        data = {n: getattr(self, n).numpy() for n in ("u", "v", "th", "qv", "qc", "pip")}
        data["time"] = self.time
        if self.config.physics:
            data.update({n: getattr(self, n).numpy() for n in ("ts", "rain", "cover", "lift", "memory")})
            data.update({f"pool{n}": f.numpy() for n, f in enumerate(self.pool)})
        np.savez(path, **data)

    def load(self, path) -> None:
        """Resumes a state written by `save` (same grid, same terrain)."""
        with np.load(path) as d:
            if d["th"].shape != self.th.shape or (self.config.physics and "ts" not in d):
                raise ValueError(f"Weather.load: {path} comes from another configuration")
            for n in ("u", "v", "th", "qv", "qc", "pip"):
                setattr(self, n, self._up(d[n]))
            self.time = float(d["time"])
            self.steps = 0
            if self.config.physics:
                for n in ("ts", "rain", "cover", "lift", "memory"):
                    setattr(self, n, self._up(d[n]))
                self.pool = [self._up(d[f"pool{n}"]) for n in range(4)]
                self._dyn_start = (self.th, self.qv, self.qc)
                self._dyn_pair = (self._dyn_start, self._dyn_start)
                # Diagnostics (map, fluxes) available right away: one physics pass, of which only that is kept
                names = ("u", "v", "th", "qv", "qc", "ts", "rain", "cover", "lift", "memory", "pool")
                kept = {n: getattr(self, n) for n in names}
                with gs.batch(self.device):
                    self._physics(self.dt * self.physics_every)
                for n, value in kept.items():
                    setattr(self, n, value)

    # ── Reading ──

    def surface_pressure(self) -> np.ndarray:
        """Surface pressure, QFE (Pa)."""
        return self.pi0_np + P_TOP + self.pip.numpy().astype(np.float64)

    def qnh(self) -> np.ndarray:
        """QNH (Pa)."""
        return sea_level_pressure_isa(self.surface_pressure(), self.terrain)

    def pressure(self) -> np.ndarray:
        """Pressure in the middle of each layer (Pa)."""
        return P_TOP + self.sigma[:, None, None] * (self.surface_pressure() - P_TOP)

    def temperature(self) -> np.ndarray:
        return self.th.numpy() * (self.pressure() / 1.0e5) ** KAPPA

    # ── One step ──

    def _dynamics(self, dt: float) -> None:
        cfg, dev = self.config, self.device
        inv = 1.0 / cfg.dx
        old = (self.u, self.v, self.th, self.qv, self.qc)
        pip_old = self.pip
        u, v, th, qv, qc, pip = *old, pip_old
        damp = cfg.divergence_damping * cfg.dx**2 / self.dt
        for frac in (1.0 / 3.0, 0.5, 1.0):
            h = dt * frac
            col = self._dyn(u, v, th, qv, qc, pip, self.pi0, self.lev, inv, inv, self._dyn_tpl, device=dev)
            new = wx_pressure(pip_old, col, cfg.nz, h, device=dev)
            mom = [
                wx_momentum(b, u, v, th, qv, qc, col, pip, pip_old, new, self.pi0, self.lev, self.coriolis, c, inv, inv, h, damp, device=dev)
                for c, b in ((0, old[0]), (1, old[1]))
            ]
            sca = [
                wx_scalar(b, f, u, v, col, pip, pip_old, new, self.pi0, self.lev, inv, inv, h, lim, device=dev)
                for b, f, lim in ((old[2], th, 0), (old[3], qv, 1), (old[4], qc, 1))
            ]
            (u, v), (th, qv, qc), pip = mom, sca, new
        self.u, self.v, self.th, self.qv, self.qc, self.pip = u, v, th, qv, qc, pip

    # ── Physique ──

    def _physics_setup(self) -> None:
        cfg = self.config
        shape = (cfg.ny, cfg.nx)
        land = (self.terrain > 0.0).astype(float) if cfg.land is None else np.asarray(cfg.land, dtype=float)
        self.land = land
        t1 = self.temperature()[-1]
        sea = t1 + 1.0 if cfg.sea_temperature is None else cfg.sea_temperature
        if callable(sea):
            sea = sea(self.y[:, None] * np.ones(shape))
        ts = np.where(land > 0.5, t1, sea)
        lat = math.radians(cfg.latitude) + self.y / 6.371e6
        surf = np.zeros((*shape, 8))
        surf[..., 0] = land
        surf[..., 1] = np.where(land > 0.5, cfg.roughness, 2e-4)
        surf[..., 2] = np.where(land > 0.5, cfg.albedo, 0.07)
        surf[..., 3] = np.where(land > 0.5, cfg.wetness, 1.0)
        surf[..., 4] = lat[:, None]
        surf[..., 5] = self.x[None, :] / (6.371e6 * math.cos(math.radians(cfg.latitude))) * 12.0 / math.pi
        surf[..., 6] = self.terrain
        surf[..., 7] = ts
        self.surf_np, self.surf = surf, self._up(surf)
        self.ts = self._up(ts)
        self.rain = self._up(np.zeros(shape))  # cumul (mm)
        self.cover = self._up(np.zeros(shape))
        self.lift = self._up(np.zeros(shape))
        r = cfg.cold_pool_refine
        self.pool = [self._up(np.zeros((cfg.ny * max(r, 1), cfg.nx * max(r, 1)))) for _ in range(4)]  # h, hu, hv, hb
        self.terrain_gpu = self._up(self.terrain)
        self.plume = self._up(plume_table(**(cfg.plume or {})))
        nz = cfg.nz
        self._surface = column_kernel(_surface_columns, 5 * nz + SURFACE_CELLS)
        self._convection = column_kernel(_convection_columns, 4 * nz + CONVECTION_CELLS)
        self._condensation = column_kernel(_condensation_columns, 3 * nz + 1)
        self._tpl_s = self._columns(5 * nz + SURFACE_CELLS)
        self._tpl_c = self._columns(4 * nz + CONVECTION_CELLS)
        self._tpl_l = self._columns(3 * nz + 1)
        self._tpl4 = self._up(np.zeros((*shape, 4)))
        self.physics_every = max(1, round(cfg.physics_step / self.dt))
        self._dyn_start = (self.th, self.qv, self.qc)
        self._dyn_pair = (self._dyn_start, self._dyn_start)
        self.surface_out = self.convection_out = None
        self.fixed_fluxes = None  # (K m/s, m/s, friction velocity): replaces the soil budget
        self.memory = self._up(np.zeros(shape))  # convective mass flux, kept from one step to the next

    def _physics(self, dt: float) -> None:
        cfg, dev, nz = self.config, self.device, self.config.nz
        hour = cfg.start_hour + self.time / 3600.0
        decl = -math.radians(23.44) * math.cos(2.0 * math.pi * (cfg.day + 10) / 365.25)
        fixed = (0.0,) * 4 if self.fixed_fluxes is None else (1.0, *self.fixed_fluxes)
        sun = self._up([hour, math.sin(decl), math.cos(decl), *fixed, 0.0])

        def field(col, f):
            return wx_unpack(col, self._tpl3, f, device=dev)

        s = self._surface(
            self.u, self.v, self.th, self.qv, self.qc, self.pip, self.pi0, self.ts, self.surf, self.cover, self.lev, sun,
            cfg.cooling / 86400.0, dt, self._tpl_s, device=dev,
        )  # fmt: skip
        self.th, self.qv, self.qc = (field(s, f) for f in range(3))
        self.u = wx_wind_tendency(self.u, s, 3, 0, device=dev)
        self.v = wx_wind_tendency(self.v, s, 4, 1, device=dev)
        self.ts = wx_unpack_surface(s, self._tpl2, 5 * nz, device=dev)
        self.surface_out = s
        if cfg.convection:
            inp = wx_surface_inputs(s, self._tpl4, 5 * nz, device=dev)
            c = self._convection(self.th, self.qv, self.qc, self.pip, self.pi0, inp, self.lift, self.memory, self.lev, self.plume, dt, self._tpl_c, device=dev)
            self.th, self.qv, self.qc = (field(c, f) for f in range(3))
            self.cover = wx_convective_cover(c, self._tpl2, 4 * nz, device=dev)
            self.convection_out = c
            self.memory = wx_unpack_surface(c, self._tpl2, 4 * nz + CNV_MEMORY, device=dev)
            r = cfg.cold_pool_refine
            if r > 0:
                n = max(1, math.ceil(dt * 60.0 * r / cfg.dx))
                for _ in range(n):
                    self.pool = list(
                        wx_cold_pool(*self.pool, self.u, self.v, c, self.terrain_gpu, 4 * nz, r, r / cfg.dx, dt / n, int(self.time // 3600), device=dev)
                    )
            col = self._dyn(self.u, self.v, self.th, self.qv, self.qc, self.pip, self.pi0, self.lev, 1 / cfg.dx, 1 / cfg.dx, self._dyn_tpl, device=dev)
            self.lift = wx_front_lift(
                *self.pool, col, self.u, self.v, self.pip, self.pi0, self.lev, self.terrain_gpu, max(r, 1), 1.0 / cfg.dx,
                cfg.front_lift, cfg.resolved_lift, device=dev,
            )  # fmt: skip
        m = self._condensation(self.th, self.qv, self.qc, self.pip, self.pi0, self.lev, dt, self._tpl_l, device=dev)
        self.th, self.qv, self.qc = (field(m, f) for f in range(3))
        self.condensation_out = m
        if cfg.convection:
            self.rain = wx_accumulate(self.rain, self.convection_out, 4 * nz, m, 3 * nz, dt, device=dev)

    def surface_field(self, c: int) -> np.ndarray:
        """Diagnostic de surface (`SFC_*`)."""
        return wx_unpack_surface(self.surface_out, self._tpl2, 5 * self.config.nz + c, device=self.device).numpy()

    def convection_field(self, c: int) -> np.ndarray:
        """Diagnostic de convection (`CNV_*`)."""
        return wx_unpack_surface(self.convection_out, self._tpl2, 4 * self.config.nz + c, device=self.device).numpy()

    def sounding(self, x: float, y: float) -> dict:
        """Sounding above the ground at point `(x, y)` (m): nearest cell, from the ground upward."""
        cfg = self.config
        i = int(x / cfg.dx) % cfg.nx
        j = min(max(int(y / cfg.dx + 0.5 * cfg.ny), 0), cfg.ny - 1)
        ps = self.surface_pressure()[j, i]
        p = P_TOP + self.sigma * (ps - P_TOP)
        half = P_TOP + self.sigma_half * (ps - P_TOP)
        th, qv, qc, u, v = (f.numpy()[:, j, i].astype(np.float64) for f in (self.th, self.qv, self.qc, self.u, self.v))
        t = th * (p / 1.0e5) ** KAPPA
        tv = t * (1.0 + 0.608 * qv)
        layer = RD * tv / GRAVITY * np.log(half[1:] / half[:-1])
        z = layer[::-1].cumsum()[::-1] - layer + RD * tv / GRAVITY * np.log(half[1:] / p)
        col = self._dyn(self.u, self.v, self.th, self.qv, self.qc, self.pip, self.pi0, self.lev, 1 / cfg.dx, 1 / cfg.dx, self._dyn_tpl, device=self.device)
        om = col.numpy()[j, i // GROUP, cfg.nz : 2 * cfg.nz, i % GROUP].astype(np.float64)  # pi dσ/dt at the bottom of the layers
        w = -0.5 * (om + np.concatenate([[0.0], om[:-1]])) / (GRAVITY * p / (RD * tv))
        out = dict(z=z, p=p, t=t, theta=th, qv=qv, qc=qc, u=u, v=v, w=w)
        out = {k: a[::-1] for k, a in out.items()}
        out["surface_pressure"], out["height"], out["f"] = float(ps), float(self.terrain[j, i]), float(self.f[j])
        if self.surface_out is not None:
            out["theta_flux"] = float(self.surface_field(SFC_WTH)[j, i])
            out["qt_flux"] = float(self.surface_field(SFC_WQ)[j, i])
            out["friction_velocity"] = float(self.surface_field(SFC_USTAR)[j, i])
        return out

    def sample(self, x: float, y: float, *, width: float = 0.75) -> dict:
        """Smoothed column around `(x, y)` (Gaussian of `width` cells), from the ground upward, for the 3D simulation.

        Profiles: `z` (above the ground), `p`, `t`, `theta`, `qv`, `qc`, `u`, `v`, `w`, and the horizontal advection
        tendencies `theta_adv` (K/s), `q_adv` (1/s) of the last physics interval. Ground: pressure, fluxes, rain…"""
        cfg, dev = self.config, self.device
        if self.surface_out is None or self.convection_out is None:
            raise RuntimeError("Weather.sample: first advance by one physics step (physics=True, convection=True)")
        gx, gy = (x / cfg.dx) % cfg.nx, min(max(y / cfg.dx + 0.5 * cfg.ny, 0.5), cfg.ny - 0.5)
        if getattr(self, "_tpl9", None) is None:
            self._tpl9, self._tpl10 = self._up(np.zeros((cfg.nz, 9))), self._up(np.zeros(10))
        inv = 1.0 / cfg.dx
        col = self._dyn(self.u, self.v, self.th, self.qv, self.qc, self.pip, self.pi0, self.lev, inv, inv, self._dyn_tpl, device=dev)
        a, b = self._dyn_pair
        prof = wx_sample_column(self.u, self.v, self.th, self.qv, self.qc, *a, *b, col, gx, gy, width, self._tpl9, device=dev)
        sfc = wx_sample_surface(
            self.pip, self.pi0, self.surface_out, self.convection_out, self.condensation_out, self.surf, cfg.nz, gx, gy, width, self._tpl10, device=dev
        ).numpy().astype(np.float64)  # fmt: skip
        prof = prof.numpy().astype(np.float64)
        ps = sfc[0] + 1.0e5
        p = P_TOP + self.sigma * (ps - P_TOP)
        half = P_TOP + self.sigma_half * (ps - P_TOP)
        th, qv, qc, u, v, om = prof[:, :6].T
        t = th * (p / 1.0e5) ** KAPPA
        tv = t * (1.0 + 0.608 * qv)
        layer = RD * tv / GRAVITY * np.log(half[1:] / half[:-1])
        z = layer[::-1].cumsum()[::-1] - layer + RD * tv / GRAVITY * np.log(half[1:] / p)
        w = -om / (GRAVITY * p / (RD * tv))
        span = self.dt * self.physics_every
        # Total advection seen by the dynamics, minus the vertical part (which the 3D simulation redoes itself)
        grad = lambda f: np.gradient(f, z)
        out = dict(z=z, p=p, t=t, theta=th, qv=qv, qc=qc, u=u, v=v, w=w)
        out["theta_adv"] = prof[:, 6] / span + w * grad(th)
        out["q_adv"] = prof[:, 7] / span + w * grad(qv + qc)
        out = {k: f[::-1].copy() for k, f in out.items()}
        names = ("theta_flux", "qt_flux", "friction_velocity", "height", "t2", "cloud", "rain", "boundary_layer", "land")
        out.update(zip(names, (float(q) for q in sfc[1:])))
        out["surface_pressure"] = float(ps)
        return out

    def sun_position(self, x: float, y: float) -> tuple[float, float]:
        """Elevation and azimuth of the sun (degrees; 0 = north, 90 = east) at point `(x, y)`."""
        cfg = self.config
        lat = math.radians(cfg.latitude) + y / 6.371e6
        hour = cfg.start_hour + self.time / 3600.0 + (x % (cfg.nx * cfg.dx)) / (6.371e6 * math.cos(math.radians(cfg.latitude))) * 12.0 / math.pi
        decl = -math.radians(23.44) * math.cos(2.0 * math.pi * (cfg.day + 10) / 365.25)
        ha = (hour - 12.0) * math.pi / 12.0
        east = -math.cos(decl) * math.sin(ha)
        north = math.sin(decl) * math.cos(lat) - math.cos(decl) * math.sin(lat) * math.cos(ha)
        up = math.sin(decl) * math.sin(lat) + math.cos(decl) * math.cos(lat) * math.cos(ha)
        return math.degrees(math.asin(up)), math.degrees(math.atan2(east, north)) % 360.0

    def local_config(self, x: float, y: float, *, size: float = 40e3, dx: float = 250.0, dz: float = 200.0, top: float = 18e3, noise: float = 0.3, seed: int = 1):
        """`AtmosphereConfig` of the 3D cloud simulation at point `(x, y)`: sounding, wind, surface fluxes and
        large-scale vertical motion; a `noise` K noise in the first kilometre breaks the symmetry."""
        from .atmosphere import AtmosphereConfig

        s = self.sounding(x, y)
        n = 4 * max(4, round(size / dx / 4))
        nz = round(top / dz)
        h = (np.arange(nz) + 0.5) * dz
        rng = np.random.default_rng(seed)
        bump = rng.uniform(-noise, noise, (nz, n, n)) * np.clip(1.0 - h / 1000.0, 0.0, 1.0)[:, None, None]

        def of(values):
            return lambda a: np.interp(a, s["z"], values)

        mean = (s["z"] < 10e3) & (s["z"] >= 0)
        return AtmosphereConfig(
            nx=n, ny=n, nz=nz, dx=dx, dy=dx, dz=dz, surface_pressure=s["surface_pressure"],
            theta_l=of(s["theta"] - LV / CP * s["qc"] * s["theta"] / s["t"]), qt=of(s["qv"] + s["qc"]), u=of(s["u"]), v=of(s["v"]),
            geostrophic_u=of(s["u"]), geostrophic_v=of(s["v"]), coriolis=s["f"], subsidence=of(s["w"]),
            surface_theta_flux=s.get("theta_flux", 0.0), surface_qt_flux=s.get("qt_flux", 0.0),
            friction_velocity=s.get("friction_velocity", 0.0), frame_u=float(s["u"][mean].mean()), frame_v=float(s["v"][mean].mean()),
            rain=True, ice=True, sponge_depth=0.25 * top, max_dt=min(8.0, 0.02 * dx), cfl=0.8, initial={"theta_l": bump},
        )  # fmt: skip

    def map(self, width: int, height: int, *, mode: int = 0, isobars: bool = True) -> gs.Array:
        """RGBA map (h, w, u32): mode 0 clouds, 1 2-m temperature; rain, QNH isobars every 4 hPa."""
        if self.surface_out is None or self.convection_out is None:
            raise RuntimeError("Weather.map: first advance by one physics step (physics=True, convection=True)")
        if getattr(self, "_map_tpl", None) is None or self._map_tpl.shape != (height, width):
            self._map_tpl = upload_u32(np.zeros((height, width), dtype=np.uint32), device=self.device)
        q = wx_qnh(self.pip, self.pi0, self.terrain_gpu, device=self.device)
        return wx_map(
            q, self.surface_out, self.convection_out, self.condensation_out, self.surf, self.pool[0], self.config.nz, mode, int(isobars),
            self._map_tpl, device=self.device,
        )  # fmt: skip

    def water(self) -> float:
        """Water of the atmosphere (kg/m², domain mean)."""
        mass = np.diff(self.sigma_half)[:, None, None] * (self.surface_pressure() - P_TOP) / GRAVITY
        q = self.qv.numpy().astype(np.float64) + self.qc.numpy()
        return float((q * mass).sum(0).mean())

    def step(self) -> None:
        with gs.batch(self.device):
            self._dynamics(self.dt)
            if self.config.physics and (self.steps + 1) % self.physics_every == 0:
                self._dyn_end = (self.th, self.qv, self.qc)
                self._dyn_pair = (self._dyn_start, self._dyn_end)
                self._physics(self.dt * self.physics_every)
                self._dyn_start = (self.th, self.qv, self.qc)
        self.time += self.dt
        self.steps += 1

    def run(self, seconds: float) -> None:
        end = self.time + seconds - 0.5 * self.dt
        while self.time < end:
            self.step()
            if self.steps % 8 == 0:
                gs.sync(self.device)
