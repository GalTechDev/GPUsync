"""Rendu de référence du ciel et des nuages d'une `Atmosphere` : tracé de chemins dans l'eau liquide simulée."""

from __future__ import annotations

import math

import numpy as np

import gpusync as gs

from .atmosphere import Atmosphere
from .kernels.k3d import upload_f32, upload_i32, upload_u32
from .nest import HALO, NestedAtmosphere
from .kernels.ksky import (
    BLOCK,
    cloud_cache_update,
    cloud_excess,
    cloud_majorant,
    cloud_sun_depth,
    cloud_sun_depth_nested,
    sky_blend,
    sky_decay,
    sky_path_trace,
    sky_ray_march,
    sky_tonemap,
)

EARTH = 6.371e6
TOP = 1.0e5
RAYLEIGH = np.array([5.8e-6, 13.5e-6, 33.1e-6])  # 1/m au niveau de la mer (R, G, B)
H_RAYLEIGH, H_MIE = 8000.0, 1200.0
OZONE = np.array([0.65e-6, 1.881e-6, 0.085e-6])


def draine_parameters(diameter_um: float) -> tuple[float, float, float]:
    """`(g, alpha, poids)` du lobe de Draine pour des gouttes de 5 à 50 µm (Jendersie & d'Eon, 2023).

    Le reste du poids est le pic avant, traité comme non diffusé."""
    d = float(diameter_um)
    g = math.exp(-2.20679 / (d + 3.91029) - 0.428934)
    alpha = math.exp(3.62489 - 8.29288 / (d + 5.52825))
    weight = math.exp(-0.599085 / (d - 0.641583) - 0.665888)
    return g, alpha, weight


_GAUSS = np.polynomial.legendre.leggauss(200)


def legendre_moments(g: float, alpha: float, count: int = 5) -> list[float]:
    """Moments de Legendre de la phase de Draine (le premier vaut 1, le deuxième est le cosinus moyen)."""
    mu, w = _GAUSS
    p = (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu) ** 1.5 * (1.0 + alpha * mu * mu)
    p = p * w / (p * w).sum()
    return [float((p * np.polynomial.legendre.Legendre.basis(n)(mu)).sum()) for n in range(count)]


def peak_exponent(g: float, alpha: float, forward: float) -> float:
    """`n` tel que la part `((1 + mu) / 2)^n` de la phase de Draine (son pic avant) pèse `forward`."""
    mu, w = _GAUSS
    p = (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu) ** 1.5 * (1.0 + alpha * mu * mu) * w
    p /= p.sum()
    lo, hi = 0.0, 200.0
    for _ in range(50):
        n = 0.5 * (lo + hi)
        lo, hi = (n, hi) if (p * (0.5 + 0.5 * mu) ** n).sum() > forward else (lo, n)
    return 0.5 * (lo + hi)


def _densities(h, mie):
    ray = np.exp(-h / H_RAYLEIGH)[..., None] * RAYLEIGH
    aer = np.exp(-h / H_MIE)[..., None] * mie
    oz = np.maximum(0.0, 1.0 - np.abs(h - 25e3) / 15e3)[..., None] * OZONE
    return ray, aer, oz


def _optical_depth(pos, direction, mie, steps=40):
    """Épaisseur optique RGB jusqu'au sommet de l'atmosphère (infinie derrière la Terre)."""
    b = (pos * direction).sum(-1)
    r2 = (pos * pos).sum(-1)
    dist = -b + np.sqrt(np.maximum(b * b - r2 + (EARTH + TOP) ** 2, 0.0))
    hidden = (b < 0) & (b * b - r2 + EARTH**2 > 0)
    x = (np.arange(steps) + 0.5) / steps
    s = dist[..., None] * x**2
    ds = dist[..., None] * 2.0 * x / steps
    p = pos[..., None, :] + direction[..., None, :] * s[..., None]
    h = np.maximum(np.sqrt((p * p).sum(-1)) - EARTH, 0.0)
    ray, aer, oz = _densities(h, mie)
    tau = ((ray + aer * 1.1 + oz) * ds[..., None]).sum(-2)
    return np.where(hidden[..., None], 1e9, tau)


def sun_transmittance(sun, altitude, *, mie=4e-6):
    """Fraction RGB du soleil qui arrive à `altitude` (m)."""
    pos = np.array([0.0, 0.0, EARTH + altitude])
    return np.exp(-_optical_depth(pos, np.asarray(sun, float), mie))


def sky_table(sun, *, altitude=1500.0, width=96, height=48, mie=4e-6, mie_g=0.76, multiple=1.0):
    """Luminance du ciel `(height, width, 3)` pour un soleil d'éclairement 1 : diffusion simple Rayleigh + aérosols + ozone.

    Lignes : `sin(élévation) = (ligne / (height - 1))²` ; colonnes : azimut."""
    sun = np.asarray(sun, float)
    dz = np.maximum((np.arange(height) / (height - 1)) ** 2, 2e-3)
    az = (np.arange(width) / width - 0.5) * 2.0 * np.pi
    rho = np.sqrt(1.0 - np.minimum(dz, 1.0) ** 2)
    d = np.stack([rho[:, None] * np.cos(az), rho[:, None] * np.sin(az), dz[:, None] * np.ones(width)], axis=-1)
    pos0 = np.array([0.0, 0.0, EARTH + altitude])
    b = d @ pos0
    dist = -b + np.sqrt(b * b - pos0 @ pos0 + (EARTH + TOP) ** 2)
    n = 48
    x = (np.arange(n) + 0.5) / n
    mu = d @ sun
    p_ray = 3.0 / (16.0 * np.pi) * (1.0 + mu**2)
    p_mie = (1 - mie_g**2) / (4 * np.pi * (1 + mie_g**2 - 2 * mie_g * mu) ** 1.5)
    out = np.zeros((height, width, 3))
    tau = np.zeros_like(out)
    for k in range(n):
        s = dist * x[k] ** 2
        ds = dist * 2.0 * x[k] / n
        p = pos0 + d * s[..., None]
        hgt = np.maximum(np.sqrt((p * p).sum(-1)) - EARTH, 0.0)
        ray, aer, oz = _densities(hgt, mie)
        ext = ray + aer * 1.1 + oz
        tau += 0.5 * ext * ds[..., None]
        t_sun = np.exp(-np.minimum(_optical_depth(p, np.broadcast_to(sun, p.shape), mie, 24), 60.0))
        # Diffusions suivantes : terme isotrope proportionnel (albédo du ciel), réglé par `multiple`
        src = ray * (p_ray[..., None] + multiple * 0.02) + aer * (p_mie[..., None] + multiple * 0.02)
        out += np.exp(-tau) * t_sun * src * ds[..., None]
        tau += 0.5 * ext * ds[..., None]
    return out


class SkyRenderer:
    """Image de référence des nuages d'une `Atmosphere` (axes du modèle : x est, y nord, z haut)."""

    def __init__(
        self,
        atmosphere: Atmosphere,
        *,
        nested: NestedAtmosphere | None = None,
        sun_elevation: float = 45.0,
        sun_azimuth: float = 180.0,
        droplets: float = 100e6,
        ground_albedo: tuple[float, float, float] = (0.04, 0.05, 0.07),
        aerosol: float = 4e-6,
        irradiance: float = 20.0,
    ) -> None:
        self.atmosphere = atmosphere
        self.nested = nested
        self.device = atmosphere.device
        self.droplets = float(droplets)
        self.ground_albedo = ground_albedo
        self.aerosol = float(aerosol)
        self.irradiance = float(irradiance)
        self.set_sun(sun_elevation, sun_azimuth, update=False)
        self.update()

    def sun_data(self, elevation: float, azimuth: float) -> tuple:
        """Tables du ciel pour un soleil donné (≈ 0,2 s de CPU, sans GPU : peut tourner dans un autre fil)."""
        el, az = math.radians(min(max(elevation, 0.5), 90.0)), math.radians(azimuth)
        sun = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
        e_cloud = self.irradiance * sun_transmittance(sun, 1500.0, mie=self.aerosol)
        return sun, e_cloud, (self.irradiance * sky_table(sun, mie=self.aerosol)).astype(np.float32)

    def set_sun(self, elevation: float = 45.0, azimuth: float = 180.0, *, update: bool = True, data: tuple | None = None) -> None:
        """Place le soleil (degrés ; azimut météorologique : 0 = nord, 90 = est) et recalcule le ciel.

        `data` : résultat de `sun_data`, déjà calculé."""
        self.sun, e_cloud, table = self.sun_data(elevation, azimuth) if data is None else data
        self._sky = upload_f32(table, device=self.device)
        self._sun_gpu = upload_f32(np.array([*self.sun, 0.0, *e_cloud, 0.0], dtype=np.float32), device=self.device)
        if update:
            self.update()

    def set_atmosphere(self, atmosphere: Atmosphere, nested: NestedAtmosphere | None = None) -> None:
        """Change de simulation (grille affinée, grille fine ajoutée…) sans changer le reste du ciel."""
        self.atmosphere, self.nested = atmosphere, nested
        self.update()

    def update(self) -> None:
        """Recalcule extinction, majorants et ombres après un pas de l'atmosphère."""
        a, dev = self.atmosphere, self.device
        cfg = a.config
        grids = [a] if self.nested is None else [a, self.nested]
        number = 4.0 * math.pi * 1000.0 * 0.8 * self.droplets / 3.0
        unit_scale = 1.5e-3 * number ** (1.0 / 3.0)
        excess = [cloud_excess(g.qt, g.qc, g.t, g.ref, float(g.config.ice), unit_scale, device=dev) for g in grids]
        # Majorants à l'échelle 1 (rho q)^(2/3), lus ici : petits, et ils donnent l'eau maximale et les blocs nuageux
        unit = [self._majorants(g, e) for g, e in zip(grids, excess)]
        self._blocks = [np.array(u.numpy()) for u in unit]
        rho_q = max(float(b.max()) for b in self._blocks) ** 1.5
        # re = (3 rho q / (4 pi rho_w k N))^(1/3), k = 0,8
        self.effective_radius_um = 1e6 * (max(rho_q, 1e-7) / number) ** (1.0 / 3.0)
        g, alpha, weight = draine_parameters(min(max(2.0 * 0.75 * self.effective_radius_um, 6.0), 45.0))
        self._phase = (g, alpha)
        self._moments = legendre_moments(g, alpha)
        self._scale = unit_scale * weight
        if cfg.nx % BLOCK or cfg.ny % BLOCK:
            raise ValueError("SkyRenderer : nx et ny multiples de 4")
        inv = (1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz)
        sun = tuple(float(v) for v in self.sun)
        major = [upload_f32(b * np.float32(self._scale), device=dev) for b in self._blocks]
        self._excess, self._majorant = excess[0], major[0]
        self._sun_depth = cloud_sun_depth(excess[0], major[0], a.ref, float(self._scale), *inv, *sun, 0.5 * min(cfg.dx, cfg.dz), device=dev)
        box = np.zeros(16, dtype=np.float32)
        if self.nested is None:
            none = upload_f32(np.zeros((1, 1, 1), dtype=np.float32), device=dev)
            self._fine = (none, none, none, upload_f32(np.zeros((1, 8), dtype=np.float32), device=dev))
        else:
            f, fc = self.nested, self.nested.config
            edge = HALO + 2
            side = fc.nx * fc.dx
            box[0:2] = f.origin
            box[2:6] = (f.origin[0] + edge * fc.dx, f.origin[0] + side - edge * fc.dx, f.origin[1] + edge * fc.dx, f.origin[1] + side - edge * fc.dx)
            box[6:9] = (1.0 / fc.dx, 1.0 / fc.dz, 1.0)
            fine = excess[1]
            depth = cloud_sun_depth_nested(
                fine, major[1], f.ref, self._sun_depth, box, float(self._scale), 1.0 / fc.dx, 1.0 / fc.dz, *inv, *sun, 0.5 * min(fc.dx, fc.dz), device=dev
            )
            self._fine = (fine, depth, major[1], f.ref)
        # Couche nuageuse (m) : hors d'elle, les rayons vont d'un trait
        low, high = 1e9, 0.0
        for g, b in zip(grids, self._blocks):
            levels = np.flatnonzero(b.max(axis=(1, 2)) > 0.0)
            if len(levels):
                low = min(low, (4.0 * levels[0] + 0.5) * g.config.dz)
                high = max(high, (4.0 * levels[-1] + 4.5) * g.config.dz)
        ceiling = (cfg.nz - 0.501) * cfg.dz
        box[9], box[10] = max(min(low, ceiling - 1.0) - 1.0, 0.0), min(max(high, min(low, ceiling) if high else 10.0) + 1.0, ceiling)
        self._box = upload_f32(box, device=dev)

    def _majorants(self, a: Atmosphere, excess: gs.Array) -> gs.Array:
        blocks = [max(1, -(-n // BLOCK)) for n in (a.config.nz, a.config.ny, a.config.nx)]
        tpl = upload_f32(np.zeros(blocks, dtype=np.float32), device=self.device)
        return cloud_majorant(excess, tpl, a.ref, 1.0, device=self.device)

    @staticmethod
    def camera(position, azimuth: float, elevation: float, fov: float = 60.0) -> np.ndarray:
        """Paramètres de la caméra (angles en degrés, azimut météorologique)."""
        az, el = math.radians(azimuth), math.radians(elevation)
        fwd = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
        right = np.array([math.cos(az), -math.sin(az), 0.0])
        cam = np.zeros(16, dtype=np.float32)
        cam[0:3], cam[3:6], cam[6:9], cam[9:12] = position, fwd, right, np.cross(right, fwd)
        cam[12] = math.tan(math.radians(fov) * 0.5)
        return cam

    def start(self, width: int, height: int) -> tuple:
        """Accumulateur vide pour un rendu progressif."""
        zero = np.zeros((height, width), dtype=np.float32)
        return tuple(upload_f32(zero, device=self.device) for _ in range(3))

    def accumulate(self, acc: tuple, cam: np.ndarray, first: int, samples: int, *, max_events: int = 600, keep: float = 1.0) -> tuple:
        """Ajoute `samples` échantillons par pixel (numérotés à partir de `first`) ; `keep` < 1 oublie les anciens."""
        cfg, dev = self.atmosphere.config, self.device
        if keep < 1.0:
            acc = sky_decay(*acc, float(keep), device=dev)
        for s in range(first, first + samples):
            acc = sky_path_trace(
                self._excess, self._sun_depth, self._majorant, self.atmosphere.ref, *self._fine, self._box, self._sky, *acc, cam, self._sun_gpu,
                float(self._scale), 1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz, s, *(float(v) for v in self._phase),
                *(float(v) for v in self.ground_albedo), self.aerosol, int(max_events), device=dev,
            )  # fmt: skip
        return acc

    def tonemap(self, acc: tuple, samples: float, exposure: float = 1.0) -> gs.Array:
        """Image u32 RGBA à afficher."""
        return sky_tonemap(*acc, float(exposure / samples), device=self.device)

    def render(
        self,
        width: int,
        height: int,
        *,
        position: tuple[float, float, float],
        azimuth: float,
        elevation: float,
        fov: float = 60.0,
        samples: int = 64,
        exposure: float = 1.0,
        max_events: int = 600,
    ) -> np.ndarray:
        """Image RGBA `(height, width, 4)` uint8 ; `azimuth` météorologique (0 = nord), angles en degrés."""
        cam = self.camera(position, azimuth, elevation, fov)
        acc = self.start(width, height)
        for first in range(0, samples, 4):
            acc = self.accumulate(acc, cam, first, min(4, samples - first), max_events=max_events)
            gs.sync(self.device)
        self.radiance = np.stack([c.numpy() for c in acc], axis=-1) / samples
        out = self.tonemap(acc, samples, exposure).numpy()
        return np.ascontiguousarray(out).view(np.uint8).reshape(height, width, 4).copy()


class CloudRenderer(SkyRenderer):
    """Rendu temps réel du même ciel : lancer de rayons et cache de diffusion multiple, calé sur `SkyRenderer`.

    Le cache n'occupe que les blocs nuageux (`cells` : mailles du cache par maille de simulation, grille large puis fine).
    `refresh_light` y ajoute un chemin par maille et oublie doucement les anciens (`memory`) ; `march` fait l'image.
    La lumière diffuse est portée par ses moments d'ordre 0, 1 et 2 avec la mise à l'échelle δ-M (`forward`, par défaut
    le troisième moment de la phase) ; le soleil direct garde sa fonction de phase exacte (correction TMS)."""

    def __init__(self, atmosphere: Atmosphere, *, memory: int = 200, forward: float | None = None, cells: tuple[int, int] = (2, 1), **kw) -> None:
        self._cache: gs.Array | None = None
        self._state: list | None = None
        self._peak_key: tuple | None = None
        self._blend = 1.0 - 1.0 / max(int(memory), 1)
        self._seed, self._forward, self._sub = 0, forward, cells
        self._target_size: tuple[int, int] | None = None
        super().__init__(atmosphere, **kw)

    def set_sun(self, elevation: float = 45.0, azimuth: float = 180.0, *, update: bool = True, data: tuple | None = None, keep: bool = False) -> None:
        """Comme `SkyRenderer.set_sun` ; `keep` garde la lumière diffuse déjà calculée (petit déplacement du soleil)."""
        data = self.sun_data(elevation, azimuth) if data is None else data
        super().set_sun(update=update, data=data)
        # Éclairement du ciel sur un sol horizontal, rapporté à la luminance du zénith (une moyenne par composante)
        _, e_cloud, table = data
        sh = table.shape[0]
        dz = (np.arange(sh) / (sh - 1)) ** 2
        weight = np.gradient(dz) * dz * 2.0
        mean = (table * weight[:, None, None]).sum(0).mean(0) / weight.sum()
        self._sun_gpu = upload_f32(np.array([*self.sun, 0.0, *e_cloud, float((mean / table[-1, 0]).mean())], dtype=np.float32), device=self.device)
        if not keep:
            self._cache, self._state = None, None
            if update:
                self._bricks()

    def set_atmosphere(self, atmosphere: Atmosphere, nested: NestedAtmosphere | None = None) -> None:
        self._cache, self._state = None, None
        super().set_atmosphere(atmosphere, nested)

    def update(self) -> None:
        super().update()
        # δ-M à l'ordre 2 : la part avant est le troisième moment de la phase
        self.forward = float(self._moments[3] if self._forward is None else self._forward)
        self._bricks()

    def _bricks(self) -> None:
        """Place les tuiles du cache dans les blocs nuageux et retrouve celles de l'image précédente."""
        grids = [self.atmosphere] + ([self.nested] if self.nested is not None else [])
        layout, index, tiles, source = np.zeros(32, dtype=np.float32), [], [], []
        total, start, state = 0, 0, []
        for g, (grid, sub, blocks) in enumerate(zip(grids, self._sub, self._blocks)):
            c, origin = grid.config, (grid.origin if g else (0.0, 0.0))
            nbz, nby, nbx = blocks.shape
            layout[16 * g : 16 * g + 9] = (c.dx / sub, c.dz / sub, *origin, sub, nbx, nby, nbz, start)
            flat = np.flatnonzero(blocks.ravel() > 0.0)
            per = sub**3
            idx = np.full(blocks.size, -1, dtype=np.int32)
            idx[flat] = total + per * np.arange(len(flat))
            bz, by, bx = np.unravel_index(flat, blocks.shape)
            uz, uy, ux = (u.ravel() for u in np.meshgrid(*[np.arange(sub)] * 3, indexing="ij"))
            own = np.stack([4 * (b[:, None] * sub + u) for b, u in ((bx, ux), (by, uy), (bz, uz))], axis=-1).reshape(-1, 3)
            tiles.append(np.concatenate([own, np.full((len(own), 1), g)], axis=1))
            old = np.full(len(flat), -1, dtype=np.int64)
            if self._state is not None and len(self._state) > g and self._state[g][0].size == idx.size:
                old_idx, old_origin = self._state[g]
                # La grille fine se déplace par mailles entières : on reprend le bloc le plus proche
                px = bx + round((origin[0] - old_origin[0]) / (4.0 * c.dx))
                py = by + round((origin[1] - old_origin[1]) / (4.0 * c.dx))
                ok = (px >= 0) & (px < nbx) & (py >= 0) & (py < nby)
                old[ok] = old_idx[((bz * nby + py) * nbx + px)[ok]]
            source.append(np.where(old[:, None] >= 0, old[:, None] + np.arange(per), -1).ravel())
            index.append(idx)
            state.append((idx, origin))
            total += per * len(flat)
            start += idx.size
        room = max(64, -(-total // 64) * 64)
        pad = np.zeros((room - total, 4), dtype=np.int32)
        pad[:, 3] = -1
        tiles.append(pad)
        source.append(np.full(room - total, -1))
        self._tiles = upload_i32(np.concatenate(tiles).astype(np.int32), device=self.device)
        self._source = upload_i32(np.concatenate(source).astype(np.int32), device=self.device)
        self._same = upload_i32(np.arange(room, dtype=np.int32), device=self.device)
        self._index = upload_i32(np.concatenate(index), device=self.device)
        self._layout = upload_f32(layout, device=self.device)
        self._state, self.cache_cells = state, 64 * total
        if self._cache is None:
            self._cache = upload_f32(np.zeros((1, 12, 64), dtype=np.float32), device=self.device)

    def _peak(self) -> float:
        """Exposant `n` tel que la part `((1 + mu) / 2)^n` de la phase pèse `forward`."""
        key = (*self._phase, self.forward)
        if self._peak_key != key:
            self._peak_key, self._peak_n = key, peak_exponent(*key)
        return self._peak_n

    def refresh_light(self, paths: int = 1, *, parts: int = 1, full: bool = False, max_events: int = 200) -> None:
        """Ajoute `paths` chemins par maille du cache ; avec `parts` > 1, une fraction `1/parts` des mailles à chaque fois.

        Un chemin s'arrête à sa première diffusion et lit la suite dans le cache ; `full` : chemins complets (plus bruités)."""
        cfg = self.atmosphere.config
        for _ in range(paths):
            self._seed += 1
            self._cache = cloud_cache_update(
                self._excess, self._sun_depth, self._majorant, self.atmosphere.ref, *self._fine, self._box, self._sky, self._cache, self._source,
                self._tiles, self._index, self._layout, self._sun_gpu, float(self._scale), 1.0 / cfg.dx, 1.0 / cfg.dy, 1.0 / cfg.dz, self._seed,
                *(float(v) for v in self._phase), *(float(v) for v in self.ground_albedo), self.aerosol, int(max_events),
                float(self._blend), self.forward, self._moments[2], self._peak(), 1000000 if full else 1, self._seed % int(parts), int(parts), device=self.device,
            )  # fmt: skip
            self._source = self._same

    def blend(self, new: tuple, old: tuple, keep: float) -> tuple:
        """Moyenne glissante de l'image `new` avec la précédente."""
        return sky_blend(*new, *old, float(keep), device=self.device)

    def march(self, cam: np.ndarray, width: int, height: int, *, frame: int = 0, quality: float = 1.0) -> tuple:
        """Luminance RGB de l'image (trois plans)."""
        if self._seed == 0:
            self.refresh_light(40)
        cfg = self.atmosphere.config
        if self._target_size != (height, width):
            self._target = upload_u32(np.zeros((height, width), dtype=np.uint32), device=self.device)
            self._target_size = (height, width)
        return sky_ray_march(
            self._excess, self._sun_depth, self._majorant, self.atmosphere.ref, *self._fine, self._box, self._sky,
            self._cache, self._index, self._layout, self._target, cam, self._sun_gpu, float(self._scale), 1.0 / cfg.dx, 1.0 / cfg.dy,
            1.0 / cfg.dz, int(frame), *(float(v) for v in self._phase), *(float(v) for v in self.ground_albedo), self.aerosol,
            float(quality), self.forward, self._moments[2], self._peak(), device=self.device,
        )  # fmt: skip
