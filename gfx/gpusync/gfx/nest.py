"""Grille fine emboîtée dans une `Atmosphere` large : le détail près du joueur, la météo tout autour."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import gpusync as gs

from .atmosphere import Atmosphere, _split
from .kernels.katmo import atm_add, atm_axpy, pressure_columns
from .kernels.knest import (
    dct_x1,
    dct_x2,
    dct_y1,
    dct_y2_columns,
    idct_columns_y1,
    idct_x1,
    idct_x2,
    idct_y2,
    nest_divergence,
    nest_fill,
    nest_halo,
    nest_project,
    nest_relax,
    nest_shift,
)

HALO = 6  # mailles imposées au bord (l'advection d'ordre 9 en lit cinq)


class NestedAtmosphere(Atmosphere):
    """Grille fine, sens unique : elle reçoit la grille large `parent` à ses bords et ne lui rend rien.

    `size` : côté utile (m) ; `center` : centre dans les coordonnées du parent (m). Les bords latéraux sont imposés,
    avec une zone de raccord ; la pression est résolue avec des parois de Neumann."""

    def __init__(
        self,
        parent: Atmosphere,
        *,
        size: float = 3200.0,
        dx: float = 25.0,
        dz: float | None = None,
        center: tuple[float, float] | None = None,
        relax_width: float = 10.0,
        relax_time: float = 30.0,
    ) -> None:
        pc = parent.config
        self.parent = parent
        top = pc.nz * pc.dz
        dz = float(dz or dx)
        nz = round(top / dz)
        if abs(nz * dz - top) > 1e-6 * top:
            raise ValueError(f"NestedAtmosphere : dz={dz} ne divise pas la hauteur du domaine ({top} m)")
        inner = 4 * max(4, round(size / dx / 4))
        n = inner + 2 * HALO
        if n * dx > min(pc.nx * pc.dx, pc.ny * pc.dy):
            raise ValueError("NestedAtmosphere : la grille fine doit tenir dans la grille large")
        self._inner = inner
        self._relax = (float(relax_width), 1.0 / float(relax_time))
        cx, cy = center or (0.5 * pc.nx * pc.dx, 0.5 * pc.ny * pc.dy)
        # Coin de la grille fine, halo compris, aligné sur ses propres mailles
        self.origin = (dx * round((cx - 0.5 * n * dx) / dx), dx * round((cy - 0.5 * n * dx) / dx))
        self._targets: list[gs.Array] | None = None
        self._window: tuple | None = None  # (état ancien, nouveau, début, fin) du dernier pas large
        self._dt: tuple[float, float] | None = None
        cfg = replace(pc, nx=n, ny=n, nz=nz, dx=dx, dy=dx, dz=dz, initial={}, max_dt=min(8.0, 0.2 * dx))
        super().__init__(cfg, device=parent.device)
        self.time = parent.time
        fields = self._fill(parent_state(parent), parent_state(parent), 1.0)
        self.u, self.v, self.w, self.s, self.qt, self.qr = fields
        self._targets = fields
        with gs.batch(self.device):
            self.u, self.v, self.w = self._project(self.u, self.v, self.w)
            self.diagnose()

    def _base_state(self) -> None:
        super()._base_state()
        cfg = self.config
        m = self._inner

        def trig(n):
            a, t = 2.0 * np.pi * np.arange(n) / n, 0.5 * np.pi * np.arange(n) / n
            return self._up(np.stack([np.cos(a), np.sin(a), np.cos(t), np.sin(t)], axis=1))

        self._trig, self._split = trig(m), _split(m)
        eig = (2.0 - 2.0 * np.cos(np.pi * np.arange(m) / m)) / cfg.dx**2
        self._eig = self._up(eig)
        g = max(d for d in range(1, min(m, 128) + 1) if m % d == 0)
        self._columns = pressure_columns(g, cfg.nz)
        self._col_tpl = self._up(np.zeros((m, m // g, cfg.nz, g)))
        self._inner_tpl = self._up(np.zeros((cfg.nz, m, m)))

    # ── Raccord ──

    def _fill(self, old, new, blend: float) -> list[gs.Array]:
        cfg, pc = self.config, self.parent.config
        tpl = self._zero
        ratios = (cfg.dx / pc.dx, cfg.dy / pc.dy, cfg.dz / pc.dz)
        corner = (self.origin[0] / pc.dx, self.origin[1] / pc.dy)
        return [
            nest_fill(a, b, tpl, min(c, 3), float(blend), *(float(v) for v in (*corner, *ratios)), device=self.device)
            for c, (a, b) in enumerate(zip(old, new))
        ]

    def _fill_one(self, field, comp: int) -> gs.Array:
        cfg, pc = self.config, self.parent.config
        args = (self.origin[0] / pc.dx, self.origin[1] / pc.dy, cfg.dx / pc.dx, cfg.dy / pc.dy, cfg.dz / pc.dz)
        return nest_fill(field, field, self._zero, min(comp, 3), 1.0, *(float(v) for v in args), device=self.device)

    def _extra_tendencies(self, slow_v, slow_s):
        width, rate = self._relax
        fields = (self.u, self.v, self.w, self.s, self.qt, self.qr)
        out = [
            nest_relax(slow, f, t, HALO, width, rate, device=self.device)
            for slow, f, t in zip((*slow_v, *slow_s), fields, self._targets)
        ]
        return out[:3], out[3:]

    def _bound_velocity(self, vel):
        return [nest_halo(f, t, c, HALO, device=self.device) for c, (f, t) in enumerate(zip(vel, self._targets))]

    def _bound_scalars(self, scalars):
        return [nest_halo(f, t, 3, HALO, device=self.device) for f, t in zip(scalars, self._targets[3:])]

    def _project(self, u, v, w):
        cfg, dev = self.config, self.device
        ix, iy, iz = 1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz
        t, sp = self._trig, self._split
        a = nest_divergence(u, v, w, self.ref, self._inner_tpl, HALO, ix, iy, iz, device=dev)
        a = dct_x2(*dct_x1(a, t, *sp, device=dev), t, *sp, device=dev)
        a = dct_y2_columns(*dct_y1(a, t, *sp, device=dev), t, self._col_tpl, *sp, device=dev)
        a = self._columns(a, self.ref, self._eig, self._eig, iz, device=dev)
        a = idct_y2(*idct_columns_y1(a, self._inner_tpl, t, *sp, device=dev), t, *sp, device=dev)
        a = idct_x2(*idct_x1(a, t, *sp, device=dev), t, *sp, device=dev)
        return tuple(nest_project(f, a, c, HALO, inv, device=dev) for c, (f, inv) in enumerate(((u, ix), (v, iy), (w, iz))))

    def divergence(self) -> np.ndarray:
        cfg = self.config
        return nest_divergence(
            self.u, self.v, self.w, self.ref, self._inner_tpl, HALO, 1 / cfg.dx, 1 / cfg.dy, 1 / cfg.dz, device=self.device
        ).numpy()

    # ── Avance ──

    def run(self, seconds: float, *, every: int = 40) -> None:
        """Avance la grille fine de `seconds` (la grille large peut finir un pas en avance)."""
        end = self.time + seconds
        k = 0
        while self.time < end - 1e-6:
            if k % every == 0:
                self.refresh_steps()
            k += 1
            self.tick(end - self.time)
            if k % 4 == 0:
                gs.sync(self.device)  # borne la file : sinon les tampons en attente remplissent la mémoire du GPU

    def refresh_steps(self) -> None:
        """Réévalue les pas de temps stables des deux grilles (attend le GPU)."""
        self._dt = (self.parent.stable_dt(), self.stable_dt())

    def tick(self, limit: float = math.inf) -> float:
        """Un seul pas fin, précédé d'un pas large s'il le faut ; renvoie sa durée. Régulier, donc adapté à une image."""
        before = self.time
        for _ in self.tick_parts(limit, split=False):
            pass
        return self.time - before

    def tick_parts(self, limit: float = math.inf, *, split: bool = True):
        """Le même pas en petits morceaux (générateur), à répartir sur plusieurs images."""
        if self._dt is None:
            self.refresh_steps()
        self._catch_up()
        if self._window is None or self.time >= self._window[3] - 1e-6:
            old = parent_state(self.parent)
            start = self.parent.time
            if split:
                yield from self.parent.step_parts(self._dt[0])
            else:
                self.parent.step(self._dt[0])
            self._window = (old, parent_state(self.parent), start, self.parent.time)
            # La grille fine repart exactement de l'instant de la grille large
            self.time = start if self.time > start - 1e-6 else self.time
        _, _, start, end = self._window
        left = end - self.time
        dt = min(left / max(1, math.ceil(left / self._dt[1] - 1e-9)), limit)
        with gs.batch(self.device):
            self._targets = self._fill(*self._window[:2], (self.time + dt - start) / (end - start))
        if split:
            yield from self.step_parts(dt)
        else:
            self.step(dt)
            yield

    def _catch_up(self) -> None:
        """La grille large a changé hors de ses pas (air décalé, repère changé) : la fenêtre et la grille fine suivent,
        du même écart interpolé."""
        if self._window is None:
            return
        old, new, start, end = self._window
        now = parent_state(self.parent)
        changed = [c for c, (a, b) in enumerate(zip(now, new)) if a is not b]
        if not changed:
            return
        dev, p = self.device, self.parent
        fields = [self.u, self.v, self.w, self.s, self.qt, self.qr]
        old = list(old)
        with gs.batch(dev):
            for c in changed:
                diff = atm_axpy(now[c], new[c], p._zero, -1.0, device=dev)
                old[c] = atm_add(old[c], diff, device=dev)
                fields[c] = atm_add(fields[c], self._fill_one(diff, c), device=dev)
            self._window = (tuple(old), now, start, end)
            self._targets = self._fill(*self._blend())
            self.u, self.v, self.w = self._project(*self._bound_velocity(fields[:3]))
            self.s, self.qt, self.qr = self._bound_scalars(fields[3:])
            self.diagnose()

    def follow(self) -> None:
        """Reprend de la grille large ce qui a changé entre deux pas : son air (`_catch_up`), son repère, ses flux de
        surface, ses forçages de grande échelle et son état de référence. À appeler entre deux pas fins."""
        p, cfg, pc = self.parent, self.config, self.parent.config
        self._catch_up()
        cfg.frame_u, cfg.frame_v = pc.frame_u, pc.frame_v
        for name in ("surface_theta_flux", "surface_qt_flux", "friction_velocity"):
            setattr(cfg, name, getattr(pc, name))
        z = self.z
        self._forcing_np = np.stack([np.interp(z, p.z, p._forcing_np[:, c]) for c in range(p._forcing_np.shape[1])], axis=1)
        self._forcing, self._forced = self._up(self._forcing_np), p._forced
        self.rebase(*(np.interp(z, p.z, a) for a in (p.t0, p.qv0, p.qc0)), pc.surface_pressure)

    def _blend(self) -> tuple:
        """Grille large vue à l'instant de la grille fine : `(ancien, nouveau, poids)`."""
        if self._window is None:
            state = parent_state(self.parent)
            return state, state, 1.0
        old, new, start, end = self._window
        return old, new, min(max((self.time - start) / (end - start), 0.0), 1.0)

    def move_to(self, x: float, y: float) -> bool:
        """Recentre la grille fine sur `(x, y)` (coordonnées du parent) par mailles entières ; `True` si elle a bougé."""
        cfg, pc = self.config, self.parent.config
        half = 0.5 * cfg.nx * cfg.dx
        # La grille large est périodique : le plus court chemin. Le coin n'est pas ramené dans le domaine (le rendu et
        # le cache le suivent d'un seul tenant) ; `nearest` donne l'image d'un point la plus proche de la grille fine
        sides = (pc.nx * pc.dx, pc.ny * pc.dy)
        sx, sy = (round(((c - half - o + 0.5 * L) % L - 0.5 * L) / cfg.dx) for c, o, L in zip((x, y), self.origin, sides))
        if sx == 0 and sy == 0:
            return False
        self.origin = (self.origin[0] + sx * cfg.dx, self.origin[1] + sy * cfg.dx)
        dev = self.device
        with gs.batch(dev):
            self._targets = self._fill(*self._blend())
            fields = (self.u, self.v, self.w, self.s, self.qt, self.qr)
            moved = [
                nest_shift(f, t, min(c, 3), HALO, int(sx), int(sy), device=dev)
                for c, (f, t) in enumerate(zip(fields, self._targets))
            ]
            self.u, self.v, self.w = self._project(*self._bound_velocity(moved[:3]))
            self.s, self.qt, self.qr = self._bound_scalars(moved[3:])
            self.diagnose()
        return True


def nearest(nest: NestedAtmosphere, x: float, y: float) -> tuple[float, float]:
    """Image périodique de `(x, y)` (coordonnées de la grille large) la plus proche du centre de la grille fine."""
    cfg, pc = nest.config, nest.parent.config
    half = 0.5 * cfg.nx * cfg.dx
    return tuple(
        o + half + ((c - o - half + 0.5 * L) % L - 0.5 * L) for c, o, L in zip((x, y), nest.origin, (pc.nx * pc.dx, pc.ny * pc.dy))
    )


def parent_state(a: Atmosphere) -> tuple[gs.Array, ...]:
    """Champs transmis à une grille emboîtée."""
    return a.u, a.v, a.w, a.s, a.qt, a.qr
