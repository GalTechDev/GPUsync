"""Nuages 3D autour du joueur, pilotés par une `Weather` : la carte fait le temps, la simulation 3D fait les nuages.

Un pavé périodique d'`Atmosphere` suit la colonne de la météo au-dessus du joueur. Toutes les minutes simulées, il en
reçoit l'advection horizontale, la vitesse verticale, les flux de surface, la pression et un rappel lent des moyennes
horizontales. Quand le joueur se déplace, l'air du pavé change de la différence entre les deux colonnes de la carte.
Le raccord est à sens unique : les nuages 3D ne modifient pas la carte."""

from __future__ import annotations

import math

import numpy as np

import gpusync as gs

from .atmosphere import CP, GRAVITY, LV, Atmosphere, AtmosphereConfig, thermals
from .nest import NestedAtmosphere, nearest
from .weather import Weather


class LocalClouds:
    """`Atmosphere` qui suit la météo `weather` au point `(x, y)` (m, coordonnées de la carte).

    `size`, `dx`, `dz`, `top` : pavé 3D. `nudge_time` : rappel de la température et de l'humidité moyennes ;
    `wind_time` : rappel du vent. `every` : intervalle simulé entre deux échanges. `coarse` : départ sur une grille
    `coarse` fois plus large, bien plus rapide ; `refine` passe ensuite à la maille `dx` sans interruption."""

    def __init__(
        self,
        weather: Weather,
        x: float,
        y: float,
        *,
        size: float = 48e3,
        dx: float = 250.0,
        dz: float = 200.0,
        top: float = 18e3,
        nudge_time: float = 5400.0,
        wind_time: float = 1800.0,
        every: float = 60.0,
        coarse: int = 1,
        noise: float = 0.2,
        seed: int = 1,
    ) -> None:
        self.weather = weather
        self.x, self.y = float(x), float(y)
        self.nudge_time, self.wind_time, self.every = float(nudge_time), float(wind_time), float(every)
        self._args = dict(size=size, dx=dx * coarse, dz=dz, top=top, noise=noise, seed=seed)
        self._coarse = int(coarse)
        self.nest: NestedAtmosphere | None = None
        self._nest_args: dict | None = None
        self.start()

    # ── Mise en place ──

    def start(self) -> None:
        """(Re)commence la simulation 3D depuis la colonne de la carte : un ciel sans nuages, à laisser mûrir (`run`)."""
        a = self._args
        s = self.weather.sample(self.x, self.y)
        n = 4 * max(4, round(a["size"] / a["dx"] / 4))
        nz = round(a["top"] / a["dz"])
        h = (np.arange(nz) + 0.5) * a["dz"]
        depth = min(max(s["boundary_layer"], 400.0), 2000.0)
        bump = thermals(n, n, h, a["dx"], depth=depth, amplitude=a["noise"], seed=a["seed"])

        def of(values):
            return lambda z: np.interp(z, s["z"], values)

        deep = (s["z"] > 500.0) & (s["z"] < 6000.0)
        cfg = AtmosphereConfig(
            nx=n, ny=n, nz=nz, dx=a["dx"], dy=a["dx"], dz=a["dz"], surface_pressure=s["surface_pressure"],
            theta_l=of(s["theta"] - LV / CP * s["qc"] * s["theta"] / s["t"]), qt=of(s["qv"] + s["qc"]), u=of(s["u"]), v=of(s["v"]),
            frame_u=float(s["u"][deep].mean()), frame_v=float(s["v"][deep].mean()),
            rain=True, ice=True, sponge_depth=0.25 * a["top"], max_dt=0.02 * a["dx"], cfl=0.8, initial={"theta_l": bump},
        )  # fmt: skip
        self.atmosphere = Atmosphere(cfg, device=self.weather.device)
        self.offset = np.zeros(2)  # position du coin du pavé sur la carte (m)
        self.dt = self.atmosphere.stable_dt()
        # L'horloge du pavé est celle de la carte
        self.time = self.weather.time
        self._next = self._rebased = self.time
        self._adv = self._wind = None
        self._place = (self.x, self.y)
        self._exchange()
        self.nest = None
        if self._nest_args is not None and not self.refinable:
            self.add_nest(**self._nest_args)

    @property
    def refinable(self) -> bool:
        return self._coarse > 1

    def refine(self) -> None:
        """Passe à la maille finale (l'état continue, interpolé). Entre deux pas seulement."""
        if self._coarse > 1:
            self.atmosphere = self.atmosphere.refined(self._coarse)
            self._args["dx"] /= self._coarse
            self._coarse = 1
            self.dt = self.atmosphere.stable_dt()

    def add_nest(self, size: float = 2400.0, dx: float = 25.0, dz: float = 75.0) -> NestedAtmosphere:
        """Grille fine (`NestedAtmosphere`) autour du joueur, dans le pavé ; elle le suit par bonds. Sur toute la hauteur
        du pavé : `dz` doit diviser `top`. Après `refine` seulement."""
        if self.refinable:
            raise RuntimeError("LocalClouds.add_nest : affiner d'abord (refine)")
        self._nest_args = dict(size=size, dx=dx, dz=dz)
        x, y, _ = self.camera_position()
        self.nest = NestedAtmosphere(self.atmosphere, size=size, dx=dx, dz=dz, center=(x, y))
        self.nest.follow()
        self.nest.refresh_steps()
        return self.nest

    @property
    def step_length(self) -> float:
        """Durée d'un pas : celui de la grille fine s'il y en a une."""
        n = self.nest
        return self.dt if n is None or n._dt is None else n._dt[1]

    @property
    def clock(self) -> float:
        """Instant de ce qu'on voit : la grille fine a jusqu'à un pas large de retard sur le pavé."""
        return self.time if self.nest is None else self.time - (self.atmosphere.time - self.nest.time)

    # ── Échange avec la carte ──

    def _levels(self, s: dict) -> dict:
        """Colonne de la carte aux niveaux du pavé."""
        z = self.atmosphere.z
        return {k: np.interp(z, s["z"], s[k]) for k in ("t", "qv", "qc", "u", "v", "w", "theta_adv", "q_adv", "p", "theta")}

    def _exchange(self) -> None:
        atm, weather = self.atmosphere, self.weather
        z = atm.z
        # Déplacement du joueur : l'air du pavé prend la différence des deux colonnes
        moved = math.hypot(*(a - b for a, b in zip(self._place, (self.x, self.y))))
        if moved > 500.0:
            self._shift(self._levels(weather.sample(*self._place)), self._levels(weather.sample(self.x, self.y)))
            self._place = (self.x, self.y)
        s = weather.sample(self.x, self.y)
        c = self._levels(s)
        self.column, self.means = c, atm.level_means()
        m = self.means
        # Advections lissées sur une demi-heure
        # Advections et ascendance prises plus large : un orage résolu par la carte sur une maille n'est pas un forçage
        b = self._levels(weather.sample(self.x, self.y, width=1.5))
        new = np.stack([b["theta_adv"] * b["t"] / b["theta"], b["q_adv"], b["w"]])
        keep = 0.0 if self._adv is None else math.exp(-self.every / 1800.0)
        self._adv = new if self._adv is None else keep * self._adv + (1.0 - keep) * new
        t_adv, q_adv, w = self._adv
        keep_part = np.minimum(1.0, 0.2 / np.maximum(np.abs(w), 1e-6))
        t_adv, q_adv, w = t_adv * keep_part, q_adv * keep_part, w * keep_part * np.clip(z / 1000.0, 0.0, 1.0)
        cooling = -weather.config.cooling / 86400.0 * np.clip((c["p"] - 15000.0) / 10000.0, 0.0, 1.0)
        # Vent : la carte donne sa variation, le rappel rattrape le reste
        wind = np.stack([c["u"], c["v"]])
        before = self._wind if moved <= 500.0 and self._wind is not None else (self.time, wind)
        lapse = self.time - before[0]
        du, dv = (wind - before[1]) / lapse if lapse > 0.0 else 0.0 * wind
        self._wind = (self.time, wind)
        # Rappel : sur l'énergie d'eau liquide et l'eau totale (nuages compris), deux fois plus vif en altitude
        relax = self.nudge_time * (1.0 - 0.5 * np.clip((z - 3000.0) / 3000.0, 0.0, 1.0))
        s_target = c["t"] + GRAVITY / CP * z - LV / CP * c["qc"]
        s_now = m["t"] + GRAVITY / CP * z - LV / CP * (m["qc"] + m["qr"])
        q_now = m["qt"]
        atm.set_forcing(
            w=w,
            t_tendency=t_adv + cooling + (s_target - s_now) / relax,
            qt_tendency=q_adv + (c["qv"] + c["qc"] - q_now) / relax,
            u_tendency=du + (c["u"] - m["u"]) / self.wind_time,
            v_tendency=dv + (c["v"] - m["v"]) / self.wind_time,
        )
        cfg = atm.config
        cfg.surface_theta_flux, cfg.surface_qt_flux = s["theta_flux"], s["qt_flux"]
        cfg.friction_velocity = s["friction_velocity"]
        self.ground = s
        # Repère : suit le vent moyen des nuages quand il a trop changé
        deep = (z > 500.0) & (z < 6000.0)
        fu, fv = float(c["u"][deep].mean()), float(c["v"][deep].mean())
        if math.hypot(fu - cfg.frame_u, fv - cfg.frame_v) > 3.0:
            atm.set_frame(fu, fv)
        # Pression de référence : remise à jour toutes les dix minutes
        if self.time - self._rebased >= 600.0 or moved > 500.0:
            atm.rebase(m["t"] if moved <= 500.0 else atm.level_means()["t"], *self._vapor(atm), s["surface_pressure"])
            self._rebased = self.time
        self.dt = atm.stable_dt()
        self._next = self.time + self.every

    def _vapor(self, atm):
        m = atm.level_means()
        return m["qt"] - m["qc"], m["qc"]

    def _shift(self, old: dict, new: dict) -> None:
        z = self.atmosphere.z
        ds = new["t"] - old["t"] - LV / CP * (new["qc"] - old["qc"])
        ratio = np.clip((new["qv"] + new["qc"]) / np.maximum(old["qv"] + old["qc"], 1e-7), 0.2, 5.0)
        self.atmosphere.offset(s=ds, qt=ratio, u=new["u"] - old["u"], v=new["v"] - old["v"])

    # ── Temps ──

    def parts(self, dt: float | None = None):
        """Un pas du pavé en morceaux (générateur), météo et échanges compris : un morceau par image dans un jeu."""
        atm = self.atmosphere
        if self.nest is not None:
            yield from self._nest_parts()
            return
        dt = self.dt if dt is None else float(dt)
        yield from atm.step_parts(dt)
        self.offset += dt * np.array([atm.config.frame_u, atm.config.frame_v])
        self.time += dt
        while self.weather.time < self.time:
            self.weather.step()
            yield
        if self.time >= self._next:
            self._exchange()
            yield

    def _nest_parts(self):
        """Un pas fin, précédé d'un pas du pavé quand la grille fine l'a rattrapé."""
        atm, nest = self.atmosphere, self.nest
        if nest.steps % 40 == 0:
            nest.refresh_steps()
        before, clock = atm.time, nest.time
        yield from nest.tick_parts()
        # Le repère avance avec ce qu'on voit
        self.offset += (nest.time - clock) * np.array([atm.config.frame_u, atm.config.frame_v])
        if atm.time > before:
            self.time += atm.time - before
            while self.weather.time < self.time:
                self.weather.step()
                yield
            if self.time >= self._next:
                self._exchange()
                nest.follow()
                yield
        # La grille fine se recentre par bonds sur le joueur
        x, y, _ = self.camera_position()
        cfg = nest.config
        cx, cy = (o + 0.5 * cfg.nx * cfg.dx for o in nest.origin)
        if max(abs(x - cx), abs(y - cy)) > 0.15 * cfg.nx * cfg.dx:
            nest.move_to(x, y)

    def step(self, dt: float | None = None) -> None:
        for _ in self.parts(dt):
            pass

    def run(self, seconds: float) -> None:
        end = self.clock + seconds - 1e-3
        k = 0
        while self.clock < end:
            self.step(min(self.dt, end + 1e-3 - self.clock))
            k += 1
            if k % 4 == 0:
                gs.sync(self.weather.device)  # borne la file du GPU

    def move_to(self, x: float, y: float) -> None:
        """Place le joueur ; l'air du pavé suit au prochain échange. Après un grand saut, préférer `start`."""
        self.x, self.y = float(x), float(y)

    # ── Vue ──

    def camera_position(self, height: float = 2.0) -> np.ndarray:
        """Position du joueur dans le repère du pavé (qui se déplace avec le vent), ramenée dans le pavé."""
        cfg = self.atmosphere.config
        side = np.array([cfg.nx * cfg.dx, cfg.ny * cfg.dy])
        x, y = (np.array([self.x, self.y]) - self.offset) % side
        if self.nest is not None:
            x, y = nearest(self.nest, x, y)  # le pavé est périodique : l'image du joueur voisine de la grille fine
        return np.array([x, y, float(height)])

    def sun(self) -> tuple[float, float]:
        """Hauteur et azimut du soleil (degrés) au-dessus du joueur."""
        return self.weather.sun_position(self.x, self.y)
