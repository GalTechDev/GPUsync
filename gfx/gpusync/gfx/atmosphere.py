"""Atmosphère humide simulée sur GPU : dynamique anélastique, condensation, pluie chaude.

Équations de Lipps & Hemler (anélastiques) sur grille C, advection d'ordre 9, limitée (MP) pour les scalaires, Runge-Kutta 3 de Wicker & Skamarock, pression par solveur direct (Hartley en x et y,
tridiagonal en z), turbulence sous-maille de Smagorinsky-Lilly, microphysique de Kessler."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import gpusync as gs

from .kernels.k3d import upload_f32
from .kernels.katmo import (
    CP,
    GRAVITY,
    LV,
    RD,
    RV,
    atm_add,
    atm_axpy,
    atm_diffuse_momentum,
    atm_diffuse_scalar,
    atm_divergence,
    atm_eddy_viscosity,
    atm_forcing_scalar,
    atm_columns_hartley_y1,
    atm_hartley_x1,
    atm_ice_fall,
    atm_hartley_x2,
    atm_hartley_y1,
    atm_hartley_y2,
    atm_hartley_y2_columns,
    atm_kessler,
    atm_level_max,
    atm_level_means,
    atm_momentum_stage,
    atm_offset,
    atm_project,
    atm_scalar_stage,
    atm_sediment,
    atm_surface_rain,
    atm_thermo,
    pressure_columns,
)

P00 = 1.0e5
S_REF = 300.0

Profile = Callable[[np.ndarray], np.ndarray]


def saturation_vapor_pressure(t):
    """Pression de vapeur saturante sur eau liquide (Pa), Bolton (1980)."""
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def saturation_mixing(t, p, ice=False):
    """Teneur en vapeur saturante (kg/kg d'air humide) ; `ice` : saturation mixte eau-glace sous 0 °C."""
    es = saturation_vapor_pressure(t)
    if ice:
        mix = np.where(t < 273.15, np.clip((t - 250.16) / 23.0, 0.0, 1.0) ** 2, 1.0)
        es = es * mix + (1.0 - mix) * 611.2 * np.exp(22.46 * (t - 273.15) / (t - 0.55))
    eps = RD / RV
    return eps * es / np.maximum(p - (1.0 - eps) * es, 1.0)


def saturation_adjust(theta_l, qt, p, ice=False):
    """`(T, qc)` d'un air de température potentielle d'eau liquide `theta_l` et d'eau totale `qt` à la pression `p`."""
    exner = (p / P00) ** (RD / CP)
    t = np.array(theta_l * exner, dtype=np.float64)
    tl = t.copy()
    saturated = qt > saturation_mixing(tl, p, ice)
    for _ in range(40):
        # Newton sans seuil dans l'air saturé : qs est convexe, la suite ne peut pas osciller autour du seuil
        qs = saturation_mixing(t, p, ice)
        dqs = (saturation_mixing(t + 1e-3, p, ice) - qs) / 1e-3
        t = t - np.where(saturated, (t - tl - LV / CP * (qt - qs)) / (1.0 + LV / CP * dqs), 0.0)
    return t, np.where(saturated, np.maximum(qt - saturation_mixing(t, p, ice), 0.0), 0.0)


def _adjust_point(theta_l: float, qt: float, p: float, ice: bool) -> tuple[float, float]:
    """`saturation_adjust` d'un seul point, en flottants Python (l'état de référence en demande des milliers)."""

    def qsat(t):
        es = 611.2 * math.exp(17.67 * (t - 273.15) / (t - 29.65))
        if ice and t < 273.15:
            mix = min(max((t - 250.16) / 23.0, 0.0), 1.0) ** 2
            es = es * mix + (1.0 - mix) * 611.2 * math.exp(22.46 * (t - 273.15) / (t - 0.55))
        eps = RD / RV
        return eps * es / max(p - (1.0 - eps) * es, 1.0)

    t = tl = theta_l * (p / P00) ** (RD / CP)
    if qt <= qsat(tl):
        return t, 0.0
    for _ in range(40):
        qs = qsat(t)
        step = (t - tl - LV / CP * (qt - qs)) / (1.0 + LV / CP * (qsat(t + 1e-3) - qs) / 1e-3)
        t -= step
        if abs(step) < 1e-10:
            break
    return t, max(qt - qsat(t), 0.0)


def lifting_condensation_pressure(t, qt, p):
    """Pression (Pa) du niveau de condensation par soulèvement d'une particule non saturée (`s` et `qt` conservés,
    dans un environnement hydrostatique de même température potentielle)."""
    theta = t * (P00 / p) ** (RD / CP)
    lo, hi = 1.0e4, float(p)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if saturation_mixing(theta * (mid / P00) ** (RD / CP), mid) < qt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thermals(nx: int, ny: int, z, dx: float, *, depth: float = 800.0, amplitude: float = 0.4, seed: int = 1) -> np.ndarray:
    """Perturbation de départ (nz, ny, nx) : des bulles de la taille des thermiques (1 à 2 fois `depth`), écart type
    `amplitude`, au lieu d'un bruit d'une maille que la turbulence met une demi-heure à organiser."""
    rng = np.random.default_rng(seed)
    kx, ky = np.fft.fftfreq(nx, dx)[None, :], np.fft.fftfreq(ny, dx)[:, None]
    k = np.sqrt(kx**2 + ky**2)
    k0 = 1.0 / max(2.0 * depth, 4.0 * dx)
    amp = np.where(k > 0.0, np.exp(-(((k - k0) / (0.5 * k0)) ** 2)), 0.0)
    f = np.fft.ifft2(np.fft.fft2(rng.normal(size=(ny, nx))) * amp).real
    f /= max(f.std(), 1e-12)
    shape = np.clip(1.0 - np.asarray(z) / (1.2 * depth), 0.0, 1.0)
    return amplitude * f[None] * shape[:, None, None]


def _split(n: int) -> tuple[int, int]:
    """Facteurs `n1 · n2 = n` les plus proches de la racine (transformée en deux passes ; `(1, n)` si n est premier)."""
    n1 = max(d for d in range(1, math.isqrt(n) + 1) if n % d == 0)
    return n1, n // n1


@dataclass
class AtmosphereConfig:
    """Domaine, sondage et forçages d'une `Atmosphere`. Les profils sont des fonctions de l'altitude (m)."""

    nx: int = 64
    ny: int = 64
    nz: int = 64
    dx: float = 100.0
    dy: float = 100.0
    dz: float = 50.0
    surface_pressure: float = 1.015e5
    theta_l: Profile = lambda z: 300.0 + 0.003 * z
    qt: Profile = lambda z: np.zeros_like(z)
    u: Profile = lambda z: np.zeros_like(z)
    v: Profile = lambda z: np.zeros_like(z)
    geostrophic_u: Profile | None = None
    geostrophic_v: Profile | None = None
    coriolis: float = 0.0
    subsidence: Profile | None = None  # vitesse verticale de grande échelle (m/s)
    theta_tendency: Profile | None = None  # K/s (rayonnement…)
    qt_tendency: Profile | None = None  # 1/s
    surface_theta_flux: float = 0.0  # <w'θ'> en K m/s
    surface_qt_flux: float = 0.0  # <w'qt'> en m/s
    friction_velocity: float = 0.0
    frame_u: float = 0.0  # vitesse de translation du repère (moins de diffusion numérique)
    frame_v: float = 0.0
    smagorinsky: float = 0.18  # 0 : pas de diffusion sous-maille
    prandtl: float = 0.3333
    rain: bool = True
    ice: bool = False  # saturation mixte eau-glace et chute des cristaux (sans neige ni grêle)
    sponge_depth: float = 0.0  # couche amortie sous le sommet (m)
    sponge_time: float = 100.0
    cfl: float = 1.0  # somme des nombres de Courant des trois directions
    max_dt: float = 10.0
    initial: dict = field(default_factory=dict)

    @classmethod
    def trade_cumulus(cls, *, size: float = 6400.0, dx: float = 100.0, dz: float = 40.0, seed: int = 1) -> AtmosphereConfig:
        """Cumulus d'alizé de type BOMEX (Siebesma et al. 2003) : couche mélangée de 520 m, inversion vers 1 500 m.

        Paramètres saisis de mémoire, à contrôler sur l'article ; seuls les flux de surface ont été recoupés."""

        def piecewise(x, y):
            return lambda z: np.interp(z, x, y)

        n = max(4, 4 * round(size / dx / 4))
        nz = round(3000.0 / dz)
        z = (np.arange(nz) + 0.5) * dz
        rng = np.random.default_rng(seed)
        fade = np.clip(1.0 - z / 1600.0, 0.0, 1.0)[:, None, None]

        def noise(amplitude):
            return rng.uniform(-amplitude, amplitude, (nz, n, n)) * fade

        levels = [0.0, 520.0, 1480.0, 2000.0, 3000.0]
        return cls(
            nx=n, ny=n, nz=nz, dx=dx, dy=dx, dz=dz, surface_pressure=1.015e5,
            theta_l=piecewise(levels, [298.7, 298.7, 302.4, 308.2, 311.85]),
            qt=piecewise(levels, [17.0e-3, 16.3e-3, 10.7e-3, 4.2e-3, 3.0e-3]),
            u=lambda z: np.where(z < 700.0, -8.75, -8.75 + 1.8e-3 * (z - 700.0)),
            geostrophic_u=lambda z: -10.0 + 1.8e-3 * z,
            coriolis=0.376e-4,
            subsidence=piecewise([0.0, 1500.0, 2100.0, 3000.0], [0.0, -0.0065, 0.0, 0.0]),
            theta_tendency=piecewise([0.0, 1500.0, 3000.0], [-2.0 / 86400.0, -2.0 / 86400.0, 0.0]),
            qt_tendency=piecewise([0.0, 300.0, 500.0, 3000.0], [-1.2e-8, -1.2e-8, 0.0, 0.0]),
            surface_theta_flux=8e-3, surface_qt_flux=5.2e-5, friction_velocity=0.28,
            frame_u=-7.0, rain=False, max_dt=min(8.0, 0.2 * dx),
            initial={"theta_l": noise(0.1), "qt": noise(2.5e-5)},
        )  # fmt: skip


class Atmosphere:
    """Simulation d'une atmosphère humide ; l'état reste sur GPU (`u, v, w, s, qt, qr`, plus `t, qc, b` diagnostiqués)."""

    def __init__(self, config: AtmosphereConfig | None = None, *, device: str | None = None) -> None:
        self.config = cfg = config or AtmosphereConfig()
        self.device = device
        self.time = 0.0
        self.steps = 0
        if not 4 <= cfg.nz <= 256:
            raise ValueError(f"Atmosphere : nz entre 4 et 256, reçu {cfg.nz}")
        self.z = (np.arange(cfg.nz) + 0.5) * cfg.dz
        self._base_state()
        n = (cfg.nz, cfg.ny, cfg.nx)
        z3 = self.z[:, None, None]
        ones = np.ones(n)
        init = cfg.initial
        self.u = self._up(cfg.u(self.z)[:, None, None] * ones - cfg.frame_u)
        self.v = self._up(cfg.v(self.z)[:, None, None] * ones - cfg.frame_v)
        self.w = self._up(np.zeros(n))
        theta_l = np.asarray(cfg.theta_l(self.z), dtype=np.float64) * np.ones(cfg.nz)
        qt = np.asarray(cfg.qt(self.z), dtype=np.float64) * np.ones(cfg.nz)
        theta_l, qt, p = theta_l[:, None, None], qt[:, None, None], self.p0[:, None, None]
        if init:
            theta_l = theta_l * ones + init.get("theta_l", 0.0)
            qt = qt * ones + init.get("qt", 0.0)
        # Sans perturbation, l'air ne dépend que de l'altitude : un seul ajustement par niveau
        t, qc = saturation_adjust(theta_l, qt, p, cfg.ice)
        self.s = self._up((t + GRAVITY / CP * z3 - LV / CP * qc - S_REF) * ones)
        self.qt = self._up(qt * ones)
        self.qr = self._up(np.zeros(n))
        self._zero = self._up(np.zeros(n))
        self.surface_rain = self._up(np.zeros(n[1:]))  # pluie tombée au sol (mm) par colonne
        self.diagnose()

    # ── Mise en place ──

    def _up(self, a) -> gs.Array:
        return upload_f32(np.ascontiguousarray(a, dtype=np.float32), device=self.device)

    def _base_state(self) -> None:
        """État de référence hydrostatique du sondage, intégré finement puis lu aux niveaux du modèle."""
        cfg = self.config
        sub = 40
        zf = np.linspace(0.0, cfg.nz * cfg.dz, cfg.nz * sub + 1)
        th = np.asarray(cfg.theta_l(zf), dtype=np.float64) * np.ones_like(zf)
        qt = np.asarray(cfg.qt(zf), dtype=np.float64) * np.ones_like(zf)
        pi = np.empty_like(zf)
        pi[0] = (cfg.surface_pressure / P00) ** (RD / CP)

        def slope(m, value):
            value = float(value)
            p = P00 * value ** (CP / RD)
            t, qc = _adjust_point(float(th[m]), float(qt[m]), p, cfg.ice)
            qv = float(qt[m]) - qc
            return -GRAVITY / (CP * (t / value) * (1.0 + (RV / RD - 1.0) * qv - qc))

        h = zf[1] - zf[0]
        for m in range(len(zf) - 1):
            k1 = slope(m, pi[m])
            k2 = slope(m + 1, pi[m] + h * k1)
            pi[m + 1] = pi[m] + 0.5 * h * (k1 + k2)
        p = P00 * pi ** (CP / RD)
        t, qc = saturation_adjust(th, qt, p, cfg.ice)
        qv = qt - qc
        rho = p / (RD * t * (1.0 + (RV / RD - 1.0) * qv - qc))
        mid = np.arange(cfg.nz) * sub + sub // 2
        low = np.arange(cfg.nz) * sub
        self.p0, self.t0, self.qv0, self.qc0 = p[mid], t[mid], qv[mid], qc[mid]
        self.rho0, self.rho_f, self.exner0 = rho[mid], rho[low], pi[mid]
        ref = np.stack([self.z, self.p0, self.t0, self.qv0, self.qc0, self.rho0, self.rho_f, self.exner0], axis=1)
        self.ref = self._up(ref)

        forcing = np.zeros((cfg.nz, 8))
        for col, prof in ((0, cfg.subsidence), (1, cfg.theta_tendency), (2, cfg.qt_tendency)):
            if prof is not None:
                forcing[:, col] = prof(self.z)
        forcing[:, 1] *= self.exner0  # K/s de température potentielle → de température
        forcing[:, 3] = (cfg.geostrophic_u or cfg.u)(self.z)
        forcing[:, 4] = (cfg.geostrophic_v or cfg.v)(self.z)
        if cfg.sponge_depth > 0.0:
            x = np.clip((self.z - (cfg.nz * cfg.dz - cfg.sponge_depth)) / cfg.sponge_depth, 0.0, 1.0)
            forcing[:, 7] = np.sin(0.5 * np.pi * x) ** 2 / cfg.sponge_time
        self._forcing_np, self._forced = forcing, False
        self._forcing = self._up(forcing)
        self._s_ref = self._up(self.t0 + GRAVITY / CP * self.z - LV / CP * self.qc0 - S_REF)
        self._qt_ref = self._up(self.qv0 + self.qc0)

        self._grid_tables()

    def _grid_tables(self) -> None:
        """Tables du solveur de pression."""
        cfg = self.config

        def trig(n):
            angle = 2.0 * np.pi * np.arange(n) / n
            return self._up(np.stack([np.cos(angle), np.sin(angle)], axis=1))

        self._trig_x, self._trig_y = trig(cfg.nx), trig(cfg.ny)
        self._split_x, self._split_y = _split(cfg.nx), _split(cfg.ny)
        # Colonnes : groupes de `g` threads (g divise nx), un thread par colonne
        g = max(d for d in range(1, min(cfg.nx, 128) + 1) if cfg.nx % d == 0)
        self._columns = pressure_columns(g, cfg.nz)
        self._col_tpl = self._up(np.zeros((cfg.ny, cfg.nx // g, cfg.nz, g)))
        self._eig_x = self._up((2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(cfg.nx) / cfg.nx)) / cfg.dx**2)
        self._eig_y = self._up((2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(cfg.ny) / cfg.ny)) / cfg.dy**2)

    # ── Raccord avec une météo de grande échelle ──

    def level_means(self) -> dict:
        """Moyennes horizontales : `s`, `qt`, `u`, `v` (vent réel), `qc`, `qr` (attend le GPU)."""
        if getattr(self, "_tpl_means", None) is None:
            self._tpl_means = self._up(np.zeros((self.config.nz, 6)))
        m = atm_level_means(self.s, self.qt, self.u, self.v, self.qc, self.qr, self._tpl_means, device=self.device).numpy().astype(np.float64)
        out = dict(zip(("s", "qt", "u", "v", "qc", "qr"), m.T))
        out["u"] = out["u"] + self.config.frame_u
        out["v"] = out["v"] + self.config.frame_v
        out["t"] = out["s"] + S_REF - GRAVITY / CP * self.z + LV / CP * (out["qc"] + out["qr"])
        return out

    def offset(self, *, s=None, qt=None, u=None, v=None) -> None:
        """Décale chaque niveau : `s`, `u`, `v` par ajout, `qt` par facteur (jamais négatif)."""
        nz = self.config.nz
        for name, value in (("s", s), ("qt", qt), ("u", u), ("v", v)):
            if value is not None:
                change = np.zeros((nz, 2))
                change[:, 1] = 1.0
                change[:, 1 if name == "qt" else 0] = value
                setattr(self, name, atm_offset(getattr(self, name), self._up(change), device=self.device))
        self.diagnose()

    def refined(self, factor: int = 2, *, noise: float = 0.05, seed: int = 1) -> Atmosphere:
        """La même atmosphère sur une grille horizontale `factor` fois plus fine, au même instant (champs interpolés).

        Un bruit de `noise` K dans le premier kilomètre aide les petits tourbillons à naître."""
        from dataclasses import replace

        from .kernels.knest import nest_fill

        cfg = self.config
        fine = replace(
            cfg, nx=cfg.nx * factor, ny=cfg.ny * factor, dx=cfg.dx / factor, dy=cfg.dy / factor, initial={}, max_dt=cfg.max_dt / factor
        )
        a = Atmosphere.__new__(type(self))
        a.config, a.device = fine, self.device
        a.time, a.steps, a.z = self.time, self.steps, self.z
        zero = a._zero = a._up(np.zeros((fine.nz, fine.ny, fine.nx)))
        for name in ("p0", "t0", "qv0", "qc0", "rho0", "rho_f", "exner0", "ref", "_forcing_np", "_forcing", "_forced", "_s_ref", "_qt_ref"):
            setattr(a, name, getattr(self, name))
        a._grid_tables()
        rng = np.random.default_rng(seed)
        bump = rng.uniform(-noise, noise, (fine.nz, fine.ny, fine.nx)) * np.clip(1.0 - self.z / 1000.0, 0.0, 1.0)[:, None, None]
        with gs.batch(a.device):
            f = [
                nest_fill(q, q, zero, min(c, 3), 1.0, 0.0, 0.0, 1.0 / factor, 1.0 / factor, 1.0, device=a.device)
                for c, q in enumerate((self.u, self.v, self.w, self.s, self.qt, self.qr))
            ]
            f[3] = atm_add(f[3], a._up(bump), device=a.device)
            a.u, a.v, a.w = a._project(*f[:3])
            a.s, a.qt, a.qr = f[3:]
            a.diagnose()
        rain = self.surface_rain.numpy()
        a.surface_rain = a._up(np.repeat(np.repeat(rain, factor, axis=0), factor, axis=1))
        return a

    def set_frame(self, frame_u: float, frame_v: float) -> None:
        """Change la vitesse de translation du repère sans changer le vent réel."""
        cfg = self.config
        self.offset(u=cfg.frame_u - frame_u, v=cfg.frame_v - frame_v)
        cfg.frame_u, cfg.frame_v = float(frame_u), float(frame_v)

    def set_forcing(self, *, w=None, t_tendency=None, qt_tendency=None, u_tendency=None, v_tendency=None) -> None:
        """Forçages de grande échelle par niveau : vitesse verticale (m/s), tendances de T, d'eau totale et de vent."""
        for col, value in ((0, w), (1, t_tendency), (2, qt_tendency), (5, u_tendency), (6, v_tendency)):
            if value is not None:
                self._forcing_np[:, col] = value
        self._forcing = self._up(self._forcing_np)
        self._forced = True

    def rebase(self, t, qv, qc, surface_pressure: float) -> None:
        """Nouvel état de référence hydrostatique (profils aux niveaux du modèle). `s` et `qt` ne dépendent pas de la
        pression : l'air garde sa température ; seules changent la pression et la densité."""
        cfg = self.config
        t, qv, qc = (np.asarray(a, dtype=np.float64) for a in (t, qv, qc))
        tv = t * (1.0 + (RV / RD - 1.0) * qv - qc)
        steps = np.concatenate([[0.5 * cfg.dz / tv[0]], 0.5 * cfg.dz * (1.0 / tv[1:] + 1.0 / tv[:-1])])
        p = surface_pressure * np.exp(-GRAVITY / RD * np.cumsum(steps))
        rho = p / (RD * tv)
        low = np.concatenate([[surface_pressure / (RD * (1.5 * tv[0] - 0.5 * tv[1]))], np.sqrt(rho[1:] * rho[:-1])])
        self.p0, self.t0, self.qv0, self.qc0, self.rho0, self.rho_f = p, t, qv, qc, rho, low
        self.exner0 = (p / P00) ** (RD / CP)
        cfg.surface_pressure = float(surface_pressure)
        self.ref = self._up(np.stack([self.z, p, t, qv, qc, rho, low, self.exner0], axis=1))
        self._s_ref = self._up(t + GRAVITY / CP * self.z - LV / CP * qc - S_REF)
        self._qt_ref = self._up(qv + qc)
        self.diagnose()

    # ── Un pas ──

    def diagnose(self) -> None:
        """Température, eau nuageuse et flottabilité de l'état courant."""
        self.t, self.qc, self.b = atm_thermo(self.s, self.qt, self.qr, self.ref, S_REF, float(self.config.ice), device=self.device)

    # Points d'accroche d'une grille emboîtée (voir `NestedAtmosphere`)
    def _extra_tendencies(self, slow_v, slow_s):
        return slow_v, slow_s

    def _bound_velocity(self, vel):
        return vel

    def _bound_scalars(self, scalars):
        return scalars

    def _project(self, u, v, w):
        cfg, dev = self.config, self.device
        ix, iy, iz = 1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz
        a = atm_divergence(u, v, w, self.ref, ix, iy, iz, device=dev)
        sx, sy = self._split_x, self._split_y
        a = atm_hartley_x2(*atm_hartley_x1(a, self._trig_x, *sx, device=dev), self._trig_x, *sx, device=dev)
        a = atm_hartley_y2_columns(*atm_hartley_y1(a, self._trig_y, *sy, device=dev), self._trig_y, self._col_tpl, *sy, device=dev)
        a = self._columns(a, self.ref, self._eig_x, self._eig_y, iz, device=dev)
        a = atm_hartley_y2(*atm_columns_hartley_y1(a, self._zero, self._trig_y, *sy, device=dev), self._trig_y, *sy, device=dev)
        a = atm_hartley_x2(*atm_hartley_x1(a, self._trig_x, *sx, device=dev), self._trig_x, *sx, device=dev)
        return tuple(atm_project(f, a, c, inv, device=dev) for c, (f, inv) in enumerate(((u, ix), (v, iy), (w, iz))))

    def divergence(self) -> np.ndarray:
        """Divergence du flux de masse (diagnostic, kg/m³/s)."""
        cfg = self.config
        return atm_divergence(self.u, self.v, self.w, self.ref, 1 / cfg.dx, 1 / cfg.dy, 1 / cfg.dz, device=self.device).numpy()

    def stable_dt(self) -> float:
        """Pas de temps d'après la vitesse maximale (attend le GPU)."""
        cfg = self.config
        if getattr(self, "_tpl_max", None) is None:
            self._tpl_max = self._up(np.zeros((cfg.nz, 3)))
        top = atm_level_max(self.u, self.v, self.w, self._tpl_max, device=self.device).numpy().max(axis=0)
        if not np.isfinite(top).all():
            raise FloatingPointError("Atmosphere : la simulation a divergé (vent infini)")
        rate = sum(float(m) / d for m, d in zip(top, (cfg.dx, cfg.dy, cfg.dz)))
        return min(cfg.max_dt, cfg.cfl / max(rate, 1e-6))

    def step(self, dt: float) -> None:
        """Avance l'état de `dt` secondes."""
        with gs.batch(self.device):
            for _ in self._parts(dt):
                pass

    def step_parts(self, dt: float):
        """Le même pas en morceaux (générateur) : un jeu en soumet un par image pour éviter les saccades. L'état visible
        ne change qu'au dernier morceau."""
        parts = self._parts(dt)
        while True:
            with gs.batch(self.device):
                done = next(parts, True)
            yield
            if done:
                return

    def _parts(self, dt: float):
        cfg, dev = self.config, self.device
        inv = (1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz)
        dt = float(dt)
        vel0, sca0 = (self.u, self.v, self.w), (self.s, self.qt, self.qr)
        # Tendances lentes, évaluées une fois par pas
        slow_v = [self._zero] * 3
        slow_s = [self._zero] * 3
        if cfg.smagorinsky > 0.0 or cfg.surface_theta_flux or cfg.surface_qt_flux or cfg.friction_velocity:
            km = atm_eddy_viscosity(
                *vel0, self.t, self.qc, self.qt, self.ref, float(cfg.dx), float(cfg.dy), float(cfg.dz),
                max(float(cfg.smagorinsky), 1e-4), float(cfg.prandtl), device=dev,
            )  # fmt: skip
            slow_v = [
                atm_diffuse_momentum(
                    *vel0, km, self.ref, c, *inv, float(cfg.friction_velocity), float(cfg.frame_u), float(cfg.frame_v), device=dev
                )
                for c in range(3)
            ]
            fluxes = (cfg.surface_theta_flux * float(self.exner0[0]), cfg.surface_qt_flux, 0.0)
            slow_s = [
                atm_diffuse_scalar(f, km, self.ref, *inv, 1.0 / float(cfg.prandtl), float(fx), device=dev)
                for f, fx in zip(sca0 if cfg.rain else sca0[:2], fluxes)
            ] + ([] if cfg.rain else [self._zero])
        if self._forced or any(p is not None for p in (cfg.subsidence, cfg.theta_tendency, cfg.qt_tendency)) or cfg.sponge_depth > 0.0:
            for c, (f, ref_prof) in enumerate(((self.s, self._s_ref), (self.qt, self._qt_ref))):
                forced = atm_forcing_scalar(f, self._forcing, c + 1, inv[2], ref_prof, device=dev)
                slow_s[c] = atm_add(slow_s[c], forced, device=dev)

        slow_v, slow_s = self._extra_tendencies(slow_v, slow_s)
        yield
        vel, sca, b = vel0, sca0, self.b
        for frac in (1.0 / 3.0, 0.5, 1.0):
            h = dt * frac
            new_v = []
            for c in range(3):
                new_v.append(
                    atm_momentum_stage(
                        vel0[c], *vel, b, slow_v[c], self.ref, self._forcing, c, *inv, float(cfg.coriolis),
                        float(cfg.frame_u), float(cfg.frame_v), h, device=dev,
                    )
                )  # fmt: skip
                yield
            new_s = []
            for c in range(3):
                if c == 2 and not cfg.rain:
                    new_s.append(sca0[2])
                    continue
                new_s.append(atm_scalar_stage(sca0[c], sca[c], *vel, slow_s[c], self.ref, *inv, h, device=dev))
                yield
            vel = self._project(*self._bound_velocity(new_v))
            yield
            sca = tuple(self._bound_scalars(new_s))
            t, qc, b = atm_thermo(*sca, self.ref, S_REF, float(self.config.ice), device=dev)
            yield
        self.u, self.v, self.w = vel
        self.s, self.qt, self.qr = sca
        if cfg.rain:
            self.qt, rain, _ = atm_kessler(self.qt, self.qr, qc, t, self.ref, dt, float(cfg.ice), device=dev)
            self.surface_rain = atm_surface_rain(self.surface_rain, rain, self.ref, dt, inv[2], device=dev)
            fallen = atm_sediment(rain, self.ref, dt, inv[2], device=dev)
            # La pluie emporte l'énergie `-L qr / cp` qu'elle porte dans `s`
            minus = atm_axpy(self._zero, fallen, atm_axpy(self._zero, rain, self._zero, -1.0, device=dev), 1.0, device=dev)
            self.s = atm_axpy(self.s, minus, self._zero, -LV / CP, device=dev)
            self.qr = fallen
            self.s, self.qt, self.qr = self._bound_scalars([self.s, self.qt, self.qr])
        if cfg.ice:
            # Les cristaux emportent l'énergie `-L qc / cp` qu'ils portent dans `s`
            change = atm_ice_fall(qc, t, self.ref, dt, inv[2], device=dev)
            self.qt = atm_add(self.qt, change, device=dev)
            self.s = atm_axpy(self.s, change, self._zero, -LV / CP, device=dev)
            self.s, self.qt, self.qr = self._bound_scalars([self.s, self.qt, self.qr])
        self.diagnose()
        self.time += dt
        self.steps += 1

    def run(self, seconds: float, *, every: int = 10) -> None:
        """Avance de `seconds` ; le pas est réévalué tous les `every` pas."""
        end = self.time + seconds
        dt = self.stable_dt()
        k = 0
        while self.time < end - 1e-6:
            if k % every == 0:
                dt = self.stable_dt()
            self.step(min(dt, end - self.time))
            k += 1
