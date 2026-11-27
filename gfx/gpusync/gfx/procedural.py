"""Procedural cumulus: a noise-shaped cloud drawn over a clear sky, for real-time rendering.

The shape is not simulated (see `kernels/kclouds.py`): a cloud mask with a flat base and a dome, Perlin-Worley noise
for the mass, Worley noise eroding the edges. The light uses the sun and the sky of `SkyRenderer`."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kernels.k3d import upload_f32
from .kernels.kclouds import cloud_procedural


def _band_noise(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Periodic noise whose spectrum lies between `low` and `high` cycles per period, scaled to [0, 1]."""
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    amp = np.where((kk >= low) & (kk <= high), 1.0 / np.maximum(kk, 1.0), 0.0)
    field = np.fft.ifftn(amp * np.exp(2j * np.pi * rng.random((n, n, n)))).real
    lo, hi = np.percentile(field, [1, 99])
    return np.clip((field - lo) / (hi - lo), 0.0, 1.0)


def _worley(n: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Periodic inverted Worley noise (1 at the feature points, 0 far away), `cells` cells per period."""
    pts = rng.random((cells, cells, cells, 3))  # one feature point in each cell
    x = (np.arange(n) + 0.5) / n * cells
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    best = np.full((n, n, n), 9.0)
    ci, cj, ck = (np.floor(g).astype(int) for g in (gx, gy, gz))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                a, b, c = ci + di, cj + dj, ck + dk
                p = pts[a % cells, b % cells, c % cells]
                d2 = (a + p[..., 0] - gx) ** 2 + (b + p[..., 1] - gy) ** 2 + (c + p[..., 2] - gz) ** 2
                best = np.minimum(best, d2)
    return np.clip(1.0 - np.sqrt(best), 0.0, 1.0).transpose(2, 1, 0)


def cloud_noise(base: int = 96, detail: int = 32, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Periodic noise volumes `(z, y, x)`: mass (Perlin-Worley) and detail (Worley octaves), in [0, 1]."""
    rng = np.random.default_rng(seed)
    perlin = 0.5 * _band_noise(base, 1, 4, rng) + 0.3 * _band_noise(base, 4, 8, rng) + 0.2 * _band_noise(base, 8, 16, rng)
    wfbm = 0.625 * _worley(base, 4, rng) + 0.25 * _worley(base, 8, rng) + 0.125 * _worley(base, 16, rng)
    # Perlin-Worley (Schneider 2015): the billows of Worley inside the large shapes of Perlin
    mass = wfbm + perlin * (1.0 - wfbm)
    mass = (mass - mass.min()) / (mass.max() - mass.min())
    fine = 0.625 * _worley(detail, 4, rng) + 0.25 * _worley(detail, 8, rng) + 0.125 * _worley(detail, 16, rng)
    fine = (fine - fine.min()) / (fine.max() - fine.min())
    return mass.astype(np.float32), fine.astype(np.float32)


@dataclass
class Cumulus:
    """One procedural cumulus. Lengths in m; `liquid_water` (g/m³) sets the extinction with `droplet_radius` (µm).

    The shape is a pile of bubbles, like the thermals that build a cumulus: a first layer covers the footprint of
    `radius` just above the flat `base`, then each bubble carries smaller ones on top of it, fewer and fewer with
    height (`growth`), up to `top`. `shear` (m of horizontal shift per m of height) leans the towers downwind (+x).
    Change `seed` for another cloud of the same kind."""

    x: float = 0.0
    y: float = 0.0
    base: float = 800.0
    top: float = 2800.0
    radius: float = 1500.0
    growth: float = 1.6
    shear: float = 0.0
    seed: int = 1
    coverage: float = 1.0
    liquid_water: float = 0.5
    droplet_radius: float = 10.0
    mass_period: float = 1600.0
    detail_period: float = 350.0
    erosion: float = 0.45
    base_sharpness: float = 0.02

    @property
    def extinction(self) -> float:
        """Maximum extinction (1/m): 3 LWC / (2 rho_w r_e) for droplets much larger than the wavelength."""
        return 1.5 * self.liquid_water * 1e-3 / (1000.0 * self.droplet_radius * 1e-6)

    def bubbles(self, limit: int = 160) -> np.ndarray:
        """`(n, 4)`: x, y, z, radius of each bubble."""
        rng = np.random.default_rng(self.seed)
        depth = self.top - self.base
        r0 = min(0.32 * self.radius, 0.45 * depth)
        count = int(np.clip(1.2 * (self.radius / r0) ** 2, 3, 36))
        out = []
        for _ in range(count):
            a, d = rng.uniform(0.0, 2.0 * np.pi), self.radius * 0.85 * np.sqrt(rng.uniform())
            r = r0 * rng.uniform(0.75, 1.15)
            out.append((self.x + d * np.cos(a), self.y + d * np.sin(a), self.base + 0.45 * r, r))
        layer = list(out)
        while layer and len(out) < limit:
            new = []
            for x, y, z, r in layer:
                h = (z - self.base) / depth
                for _ in range(rng.poisson(max(self.growth * (1.0 - h * h), 0.0))):
                    rc = r * rng.uniform(0.84, 1.02)
                    zc = z + r * rng.uniform(0.5, 0.8)
                    if zc + 0.8 * rc > self.top or len(out) + len(new) >= limit:
                        continue
                    # Drift toward the core, where the updraft is strongest
                    xc = x + 0.2 * (self.x - x) + rng.normal(0.0, 0.25 * r) + self.shear * (zc - z)
                    yc = y + 0.2 * (self.y - y) + rng.normal(0.0, 0.25 * r)
                    new.append((xc, yc, zc, rc))
            # Layers never widen: the most central bubbles carry the next one (the core rises the most)
            if len(new) > len(layer):
                new.sort(key=lambda b: (b[0] - self.x) ** 2 + (b[1] - self.y) ** 2 + rng.uniform(0.0, 0.3) * self.radius**2)
                new = new[: len(layer)]
            out += new
            layer = new
        return np.asarray(out, dtype=np.float32)


class ProceduralClouds:
    """Draws procedural cumulus over the clear-sky image of a `SkyRenderer` (`sky`), lit by its sun and its sky."""

    def __init__(self, sky, *, device: str | None = None, seed: int = 7) -> None:
        self.sky, self.device = sky, device
        mass, fine = cloud_noise(seed=seed)
        self._mass, self._fine = upload_f32(mass, device=device), upload_f32(fine, device=device)
        self.offset = np.zeros(3)
        self.phase = (0.8, -0.3, 0.75)  # forward and backward asymmetry, weight of the forward lobe
        self.aerial = 25e3
        self.steps = 96
        # Octaves of Wrenninge (attenuation, weight, count) and weight of the ambient light, fitted on the path-traced
        # reference (`SkyRenderer`) of the same density: side, front and back light, mean radiance within 12 %
        self.multiple = (0.4, 0.9, 4, 1.5)

    def light(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sun direction and irradiance, sky radiance from above and from the ground (mean over the hemisphere)."""
        sky = self.sky
        table = sky._sky.numpy().astype(np.float64)  # (rows, azimuths, 3), rows: sin(elevation) = (row / (n - 1))²
        n = table.shape[0]
        mu = np.maximum((np.arange(n) / (n - 1)) ** 2, 2e-3)
        # Cosine-weighted mean over the upper hemisphere: irradiance / pi
        weight = np.gradient(mu) * mu
        down = (table.mean(axis=1) * weight[:, None]).sum(0) / weight.sum()
        sun, e_sun = np.asarray(sky.sun, float), np.asarray(sky._sun_gpu.numpy()[4:7], float)
        ground = np.asarray(sky.ground_albedo, float) * (e_sun * max(sun[2], 0.0) + math.pi * down) / math.pi
        return sun, e_sun, down, ground

    def render(self, cam: np.ndarray, background: tuple, clouds: list[Cumulus], *, frame: int = 0) -> tuple:
        """RGB radiance (three planes) of `background` seen through `clouds` (one for now)."""
        out = background
        sun, e_sun, down, ground = self.light()
        cam_gpu = upload_f32(np.asarray(cam, dtype=np.float32), device=self.device)
        for c in clouds:
            bub = c.bubbles()
            lo = (bub[:, :3] - bub[:, 3:]).min(axis=0)
            hi = (bub[:, :3] + bub[:, 3:]).max(axis=0)
            lo[2] = c.base
            prm = np.zeros(48, dtype=np.float32)
            prm[0:3], prm[3:6], prm[6:9], prm[9:12] = sun, e_sun, down, ground
            prm[12:15] = (c.base, hi[2], len(bub))
            prm[36:42] = (*lo, *hi)
            prm[17:22] = (c.coverage, c.extinction, c.mass_period, c.detail_period, c.erosion)
            prm[22:25] = self.offset
            prm[25:28] = self.phase
            prm[28:32] = (frame % 4096, self.aerial, c.base_sharpness, self.steps)
            prm[32:36] = self.multiple
            bub_gpu = upload_f32(bub, device=self.device)
            out = cloud_procedural(*out, self._mass, self._fine, bub_gpu, cam_gpu, upload_f32(prm, device=self.device), device=self.device)
        return out
