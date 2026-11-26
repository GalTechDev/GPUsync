"""Moteurs fluides 100 % GPU et entièrement paramétrables pour `gpusync.gfx` :
- `WaterSurfaceConfig` + `GpuWaterSurface` : surface d'eau interactive 3D (équation des ondes 2D,
  normales lisses réalistes OU facettes nettes low-poly, bords réfléchissants/absorbants, houle ambiante).
- `FluidSPHConfig` + `GpuFluidSPH3D` : fluide particulaire 3D SPH complet (densité, pression, viscosité,
  cohésion, gravité orientable, préréglages eau / low-poly / miel / lave / mercure → `InstancedMesh3D.gpu_trs`).
- `Fluid2DConfig` + `GpuFluid2D` : fluide eulérien 2D Navier-Stokes / Stable Fluids (advection, confinement
  de vorticité, projection Jacobi, flottabilité, rendu lisse ou cel-shaded / pixel-art → `Texture` 0-copie).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

import gpusync as gs

from .kernels.k3d import upload_f32, upload_u32
from .kernels.kfluid import (
    flip_g2p,
    flip_grid_divergence,
    flip_grid_forces,
    flip_grid_pressure_jacobi,
    flip_grid_project,
    flip_p2g,
    fluid2d_advect_dye,
    fluid2d_advect_vel,
    fluid2d_curl,
    fluid2d_divergence,
    fluid2d_pressure_jacobi,
    fluid2d_render_u32,
    fluid2d_splat_dye,
    fluid2d_splat_vel,
    fluid2d_subtract_grad,
    fluid2d_vorticity,
    sph_to_trs,
    water_add_splash,
    water_build_flat_mesh,
    water_build_smooth_mesh,
    water_rescale_prev,
    water_wave_step,
)
from .kernels.kfluid3d import (
    mac_cell_velocity,
    mac_divergence,
    mac_face_forces,
    mac_fill,
    mac_g2p,
    mac_mass_splat,
    mac_p2g,
    mac_pressure_diag,
    mac_pressure_jacobi2,
    mac_project_face,
    mac_shift,
    mac_volume_move,
    mac_volume_source,
    caustic_photons,
    caustic_resolve,
    fluid_velocity_resolve,
    fluid_velocity_splat,
    height_blur,
    vol_blur_axis,
    vol_splat_density,
    vol_unpack_f32,
    vol_wet_walls,
    water_height,
    ww_step,
)
from .kernels.ksph import (
    sort_keys,
    sph_cell_keys,
    sph_cell_ranges,
    sph_density,
    sph_gather,
    sph_integrate,
)
from .kernels.kwater import (
    floating_collide,
    floating_step,
    floating_wakes,
    water_add_wakes,
    water_depth_map,
)
from .material import Material
from .mesh import Mesh
from .texture import Texture


# ───────────────────────── 1. Surface d'Eau 3D (`GpuWaterSurface`) ─────────────────────────


# Lissage des ondes de la taille d'une cellule (fraction par sous-pas, stable sous 0,05)
WAVE_SMOOTHING = 0.01

# Volume de rendu : échelle de densité, et valeur obtenue au cœur de l'eau au repos
DENSITY_SCALE = 0.68
VOLUME_CORE = 0.68


def _make_flat_plane_mesh(size: float, subdivisions: int) -> Mesh:
    """Construit un plan horizontal non-indexé (3 sommets dédiés par triangle) pour le flat-shading low-poly."""
    n = max(1, int(subdivisions))
    half = size * 0.5
    xs = np.linspace(-half, half, n + 1, dtype=np.float32)
    zs = np.linspace(-half, half, n + 1, dtype=np.float32)

    nt = n * n * 2
    pos = np.empty((nt * 3, 3), dtype=np.float32)
    uv = np.empty((nt * 3, 2), dtype=np.float32)
    idx = np.arange(nt * 3, dtype=np.int32).reshape(nt, 3)

    t = 0
    for iz in range(n):
        z0, z1 = float(zs[iz]), float(zs[iz + 1])
        v0, v1 = iz / n, (iz + 1) / n
        for ix in range(n):
            x0, x1 = float(xs[ix]), float(xs[ix + 1])
            u0, u1 = ix / n, (ix + 1) / n
            # Triangle 1 (CCW vu de +Y : (x0,z0) → (x0,z1) → (x1,z1))
            b = t * 3
            pos[b] = (x0, 0.0, z0)
            pos[b + 1] = (x0, 0.0, z1)
            pos[b + 2] = (x1, 0.0, z1)
            uv[b] = (u0, v0)
            uv[b + 1] = (u0, v1)
            uv[b + 2] = (u1, v1)
            t += 1
            # Triangle 2
            b = t * 3
            pos[b] = (x0, 0.0, z0)
            pos[b + 1] = (x1, 0.0, z1)
            pos[b + 2] = (x1, 0.0, z0)
            uv[b] = (u0, v0)
            uv[b + 1] = (u1, v1)
            uv[b + 2] = (u1, v0)
            t += 1

    norms = np.zeros_like(pos)
    norms[:, 1] = 1.0
    return Mesh(positions=pos, normals=norms, uvs=uv, indices=idx)


@dataclass
class WaterSurfaceConfig:
    """Configuration complète d'une surface d'eau 3D (du réaliste miroir au low-poly facetté)."""

    size: float = 12.0
    subdivisions: int = 56
    flat_shading: bool = False  # True = facettes triangulaires nettes (style low-poly / cristal)
    wave_speed: float = 4.8  # vitesse des ondes en m/s, indépendante du maillage (sous-pas automatiques)
    damping: float = 0.985  # vitesse verticale conservée par 1/60 s
    height_scale: float = 0.72
    ambient_waves: float = 0.0  # houle de fond procédurale (0 = bassin calme, >0 = océan)
    ambient_freq: float = 0.72
    ambient_speed: float = 1.25
    boundary: str = "reflect"  # "reflect" (bassin fermé) ou "absorb" (océan ouvert)
    substeps: int = 1
    fixed_dt: float = 1.0 / 60.0  # pas de temps fixe : même physique à 60 Hz, 120 Hz ou 500 FPS
    albedo: tuple[float, float, float] = (0.04, 0.34, 0.56)
    metallic: float = 0.72
    roughness: float = 0.06
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Optique (eau transparente) ; False : surface opaque, rendue avec `albedo`, `metallic`…
    optics: bool = False
    depth: float = 1.6  # profondeur du bassin (m), sous le niveau de l'objet
    ior: float = 1.333
    absorption: tuple[float, float, float] = (0.42, 0.11, 0.06)  # par mètre (R, G, B)
    tint: tuple[float, float, float] = (0.03, 0.34, 0.48)  # lumière diffusée par l'eau
    caustics: float = 0.9
    floor_colors: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.52, 0.64, 0.68), (0.30, 0.46, 0.55))
    walls: bool = True  # murs carrelés (bassin) ; False : le fond continue au-delà (mer)
    tile: float = 0.75  # côté des carreaux du bassin (m)

    @classmethod
    def realistic(cls, size: float = 12.0, subdivisions: int = 56) -> WaterSurfaceConfig:
        """Eau de bassin / lac réaliste lisse, calme et très réfléchissante (idéale avec `cam.ssr = True`)."""
        return cls(
            size=size,
            subdivisions=subdivisions,
            flat_shading=False,
            wave_speed=4.8,
            damping=0.986,
            height_scale=0.68,
            ambient_waves=0.02,
            albedo=(0.03, 0.30, 0.52),
            metallic=0.76,
            roughness=0.05,
            optics=True,
        )

    @classmethod
    def low_poly(cls, size: float = 12.0, subdivisions: int = 16) -> WaterSurfaceConfig:
        """Eau low-poly stylisée à facettes nettes et reflets cristallins."""
        return cls(
            size=size,
            subdivisions=subdivisions,
            flat_shading=True,
            wave_speed=4.5,
            damping=0.982,
            height_scale=0.95,
            ambient_waves=0.08,
            ambient_freq=0.62,
            ambient_speed=1.1,
            albedo=(0.08, 0.62, 0.86),
            metallic=0.22,
            roughness=0.18,
        )

    @classmethod
    def ocean(cls, size: float = 18.0, subdivisions: int = 64) -> WaterSurfaceConfig:
        """Océan ouvert avec houle continue et bords absorbants."""
        return cls(
            size=size,
            subdivisions=subdivisions,
            flat_shading=False,
            wave_speed=5.2,
            damping=0.988,
            height_scale=0.80,
            ambient_waves=0.18,
            ambient_freq=0.52,
            ambient_speed=1.45,
            boundary="absorb",
            albedo=(0.02, 0.22, 0.44),
            metallic=0.75,
            roughness=0.09,
            optics=True,
            depth=6.0,
            absorption=(0.55, 0.20, 0.11),
            tint=(0.015, 0.16, 0.27),
            floor_colors=((0.50, 0.46, 0.36), (0.44, 0.40, 0.31)),
            tile=2.0,
            walls=False,
        )

    @classmethod
    def lava(cls, size: float = 12.0, subdivisions: int = 36) -> WaterSurfaceConfig:
        """Lave visqueuse et incandescente (amortissement fort, émissif chaud)."""
        return cls(
            size=size,
            subdivisions=subdivisions,
            flat_shading=False,
            wave_speed=3.0,
            damping=0.960,
            height_scale=0.85,
            ambient_waves=0.06,
            ambient_speed=0.55,
            albedo=(0.22, 0.04, 0.02),
            metallic=0.05,
            roughness=0.45,
            emissive=(1.65, 0.36, 0.04),
        )

    @classmethod
    def mercury(cls, size: float = 12.0, subdivisions: int = 56) -> WaterSurfaceConfig:
        """Métal liquide / mercure chromé ultra-brillant."""
        return cls(
            size=size,
            subdivisions=subdivisions,
            flat_shading=False,
            wave_speed=5.3,
            damping=0.989,
            height_scale=0.65,
            albedo=(0.92, 0.95, 1.0),
            metallic=0.98,
            roughness=0.04,
        )

    def make_material(self) -> Material:
        return Material(
            albedo=self.albedo,
            metallic=self.metallic,
            roughness=self.roughness,
            emissive=self.emissive,
        )


@dataclass
class FloatingBody:
    """Corps flottant couplé en double sens avec `GpuWaterSurface` (Archimède + surf sur la pente des vagues + sillage)."""

    node: object
    radius: float = 0.48
    buoyancy: float = 24.0
    gravity: float = 9.81
    wave_push: float = 8.5  # force horizontale exercée par la pente des vagues
    drag: float = 0.94  # frottement fluide horizontal
    vertical_drag: float = 0.88  # amortissement du tangage vertical
    wake_strength: float = 0.10  # intensité du sillage laissé dans l'eau
    vel: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vel is None:
            self.vel = np.zeros(3, dtype=np.float32)
        else:
            self.vel = np.asarray(self.vel, dtype=np.float32)


class GpuWaterSurface:
    """Simulation de surface d'eau 3D 100 % GPU (0 copie CPU → `Object3D.gpu_positions/gpu_normals`)."""

    def __init__(self, config: WaterSurfaceConfig | None = None, *, device: str | None = None) -> None:
        self.config = config or WaterSurfaceConfig.realistic()
        self.device = device
        self.t = 0.0
        self._accum = 0.0
        self._h_cpu: np.ndarray | None = None
        self._h_cpu_t: float = -1.0
        self._sub = 1

        self.mesh: Mesh = (
            _make_flat_plane_mesh(self.config.size, self.config.subdivisions)
            if self.config.flat_shading
            else Mesh.plane(self.config.size, subdivisions=self.config.subdivisions)
        )
        self.material: Material = self.config.make_material()

        self.grid_n = self.config.subdivisions + 1
        self._cell_size = self.config.size / max(1, self.config.subdivisions)
        self._base_pos_gpu = upload_f32(self.mesh.positions, device=device)
        zeros = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.h_curr = upload_f32(zeros, device=device)
        self.h_prev = upload_f32(zeros, device=device)
        # Caustiques : deux photons exactement par case et par axe (la grille de photons couvre 1,5 fois le bassin)
        m = min(256, 3 * self.grid_n)
        self._map_tpl = upload_f32(np.zeros((m, m), dtype=np.float32), device=device)
        self._photon_tpl = upload_f32(np.zeros((3 * m, 3 * m), dtype=np.float32), device=device)
        self._caustics: tuple[gs.Array, tuple, gs.Array] | None = None

        self.gpu_positions, self.gpu_normals = self._rebuild_mesh()

    def _rebuild_mesh(self) -> tuple[gs.Array, gs.Array]:
        cfg = self.config
        if cfg.flat_shading:
            return water_build_flat_mesh(
                self._base_pos_gpu,
                self.h_curr,
                float(cfg.size * 0.5),
                float(1.0 / max(cfg.size, 1e-4)),
                float(cfg.height_scale),
                float(cfg.ambient_waves),
                float(cfg.ambient_freq),
                float(self.t * cfg.ambient_speed),
                device=self.device,
            )
        return water_build_smooth_mesh(
            self._base_pos_gpu,
            self.h_curr,
            float(self._cell_size),
            float(cfg.height_scale),
            float(cfg.ambient_waves),
            float(cfg.ambient_freq),
            float(self.t * cfg.ambient_speed),
            device=self.device,
        )

    def splash(self, world_x: float, world_z: float, *, strength: float = -0.55, radius: float = 0.65) -> None:
        """Crée une onde/goutte centrée aux coordonnées monde `(world_x, world_z)`."""
        half = self.config.size * 0.5
        u = (world_x + half) / max(self.config.size, 1e-4)
        v = (world_z + half) / max(self.config.size, 1e-4)
        self.splash_uv(u, v, strength=strength, radius=radius / max(self.config.size, 1e-4))

    def splash_uv(self, u: float, v: float, *, strength: float = -0.55, radius: float = 0.055) -> None:
        """Crée une onde aux coordonnées normalisées `(u, v) ∈ [0, 1]²`."""
        cx = float(np.clip(u, 0.02, 0.98)) * (self.grid_n - 1)
        cy = float(np.clip(v, 0.02, 0.98)) * (self.grid_n - 1)
        r_cells = max(1.0, float(radius) * (self.grid_n - 1))
        inv_r2 = 1.0 / (r_cells * r_cells)
        # Couronne dosée pour que la somme soit nulle sur la grille, même près d'un bord (somme séparable en x et y)
        axis = np.arange(self.grid_n, dtype=np.float64)
        core, wide = (
            float(np.exp(-((axis - cx) ** 2) * k).sum() * np.exp(-((axis - cy) ** 2) * k).sum()) for k in (inv_r2, 0.3 * inv_r2)
        )
        ring = core / wide
        # Les deux instants reçoivent la même forme : la hauteur change, sans vitesse parasite
        self.h_curr, self.h_prev = (
            water_add_splash(h, cx, cy, inv_r2, float(strength), ring, device=self.device) for h in (self.h_curr, self.h_prev)
        )

    def reset(self) -> None:
        zeros = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.h_curr = upload_f32(zeros, device=self.device)
        self.h_prev = upload_f32(zeros, device=self.device)
        self.gpu_positions, self.gpu_normals = self._rebuild_mesh()

    def step(self, dt: float) -> tuple[gs.Array, gs.Array]:
        """Avance la simulation d'ondes à pas fixe sur GPU (indépendant des FPS) et renvoie `(gpu_positions, gpu_normals)`."""
        cfg = self.config
        damp = min(max(float(cfg.damping), 0.85), 1.0)
        absorb = 1 if cfg.boundary == "absorb" else 0
        fixed_dt = max(float(cfg.fixed_dt), 1.0 / 240.0)
        self._accum = min(self._accum + max(float(dt), 0.0), fixed_dt * 3.0)
        n_ticks = int(self._accum / fixed_dt + 1e-6)
        if n_ticks > 0:
            self._accum -= n_ticks * fixed_dt
        # Stable si vitesse · dt / cellule < 0,7 ; le lissage en prend une part
        courant = max(float(cfg.wave_speed), 0.0) * fixed_dt / self._cell_size
        sub = max(1, int(cfg.substeps), math.ceil(courant / 0.6))
        c2 = (courant / sub) ** 2
        # Dans le schéma, une vitesse est une différence entre deux pas : tout se règle par la durée du pas
        if sub != self._sub:
            self._rescale_velocity(self._sub / sub)
            self._sub = sub
        damp_sub = damp ** (fixed_dt * 60.0 / sub)

        with gs.batch(self.device):
            for _ in range(n_ticks * sub):
                h_next = water_wave_step(self.h_curr, self.h_prev, c2, damp_sub, WAVE_SMOOTHING, absorb, device=self.device)
                self.h_prev = self.h_curr
                self.h_curr = h_next
            self.t += float(dt)
            self.gpu_positions, self.gpu_normals = self._rebuild_mesh()
        return self.gpu_positions, self.gpu_normals

    def _rescale_velocity(self, ratio: float) -> None:
        """Le nombre de sous-pas a changé : la même vitesse correspond à un autre écart entre les deux instants."""
        self.h_prev = water_rescale_prev(self.h_curr, self.h_prev, float(ratio), device=self.device)

    def apply_to(self, obj: object) -> None:
        """Branche directement les buffers VRAM sur un `Object3D` (0 copie CPU)."""
        obj.gpu_positions = self.gpu_positions  # type: ignore[attr-defined]
        obj.gpu_normals = self.gpu_normals  # type: ignore[attr-defined]
        obj.water = self if self.config.optics else None  # type: ignore[attr-defined]

    def caustic_map(self, sun_dir) -> gs.Array:
        """Éclairement du fond du bassin par le soleil réfracté (1 = eau plate) ; recalculé si l'eau a changé."""
        cfg = self.config
        sun = (float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2]))
        key = (self.t, sun, float(cfg.depth), float(cfg.ior))
        if self._caustics is not None and self._caustics[0] is self.h_curr and self._caustics[1] == key:
            return self._caustics[2]
        half = 0.5 * float(cfg.size)
        depth = water_depth_map(
            self.h_curr,
            half,
            float(cfg.height_scale),
            float(cfg.depth),
            float(cfg.ambient_waves),
            float(cfg.ambient_freq),
            float(self.t * cfg.ambient_speed),
            device=self.device,
        )
        photons = caustic_photons(
            depth, self._photon_tpl, self._map_tpl, 2, half, 1.0, half, *sun, float(cfg.ior), device=self.device
        )
        expected = (self._photon_tpl.shape[0] / 1.5 / self._map_tpl.shape[0]) ** 2
        out = caustic_resolve(photons, float(expected), device=self.device)
        self._caustics = (self.h_curr, key, out)
        return out

    def sample_height_slope(self, world_x: float, world_z: float) -> tuple[float, float, float]:
        """Renvoie `(hauteur_y, dh_dx, dh_dz)` à la position monde `(world_x, world_z)` pour les objets flottants."""
        if self._h_cpu is None or self._h_cpu_t != self.t:
            self._h_cpu = self.h_curr.numpy()
            self._h_cpu_t = self.t
        cfg = self.config
        half = cfg.size * 0.5
        n = self.grid_n - 1
        gx = float(np.clip((world_x + half) / max(cfg.size, 1e-4) * n, 1.0, n - 1.001))
        gz = float(np.clip((world_z + half) / max(cfg.size, 1e-4) * n, 1.0, n - 1.001))
        ix, iz = int(gx), int(gz)
        tx, tz = gx - ix, gz - iz
        h = self._h_cpu
        h00 = float(h[iz, ix])
        h10 = float(h[iz, ix + 1])
        h01 = float(h[iz + 1, ix])
        h11 = float(h[iz + 1, ix + 1])
        y = ((1.0 - tx) * (1.0 - tz) * h00 + tx * (1.0 - tz) * h10 + (1.0 - tx) * tz * h01 + tx * tz * h11) * cfg.height_scale
        inv_2d = 0.5 * cfg.height_scale / max(self._cell_size, 1e-4)
        dhdx = (float(h[iz, min(ix + 1, n)]) - float(h[iz, max(ix - 1, 0)])) * inv_2d
        dhdz = (float(h[min(iz + 1, n), ix]) - float(h[max(iz - 1, 0), ix])) * inv_2d
        if cfg.ambient_waves > 0.0:
            amb_t = self.t * cfg.ambient_speed
            f = cfg.ambient_freq
            p1 = world_x * f + world_z * (f * 0.55) + amb_t
            p2 = world_x * (-f * 0.72) + world_z * (f * 1.15) - amb_t * 0.82
            y += cfg.ambient_waves * (0.62 * math.sin(p1) + 0.38 * math.cos(p2))
            dhdx += cfg.ambient_waves * (0.62 * f * math.cos(p1) + 0.38 * (f * 0.72) * math.sin(p2))
            dhdz += cfg.ambient_waves * (0.62 * (f * 0.55) * math.cos(p1) - 0.38 * (f * 1.15) * math.sin(p2))
        return y, dhdx, dhdz

    def step_floating(self, bodies: list[FloatingBody], dt: float) -> None:
        """Simule la flottaison (Archimède + poussée par les vagues + roulement + sillage) des `FloatingBody`."""
        dt_c = min(max(float(dt), 0.0), 0.033)
        if dt_c <= 0.0 or not bodies:
            return
        limit = self.config.size * 0.5 - 0.32
        for fb in bodies:
            node = fb.node
            wx = float(node.x)  # type: ignore[attr-defined]
            wy = float(node.y)  # type: ignore[attr-defined]
            wz = float(node.z)  # type: ignore[attr-defined]
            water_y, dhdx, dhdz = self.sample_height_slope(wx, wz)

            # Fraction immergée (0 hors de l'eau, 1 totalement sous l'eau)
            sub = float(np.clip((water_y - (wy - fb.radius)) / max(2.0 * fb.radius, 1e-4), 0.0, 1.0))
            fy = fb.buoyancy * sub - fb.gravity
            # Les vagues poussent horizontalement dans le sens de leur pente (-∇h)
            fx = -dhdx * fb.wave_push * sub
            fz = -dhdz * fb.wave_push * sub

            vel = fb.vel
            assert vel is not None
            vel[0] = (vel[0] + fx * dt_c) * (fb.drag ** (dt_c * 60.0))
            vel[1] = (vel[1] + fy * dt_c) * ((fb.vertical_drag if sub > 0.0 else 0.995) ** (dt_c * 60.0))
            vel[2] = (vel[2] + fz * dt_c) * (fb.drag ** (dt_c * 60.0))

            nx = wx + float(vel[0]) * dt_c
            ny = wy + float(vel[1]) * dt_c
            nz = wz + float(vel[2]) * dt_c

            # Rebond doux sur les margelles du bassin
            max_p = max(0.5, limit - fb.radius)
            if nx < -max_p:
                nx = -max_p
                vel[0] = abs(vel[0]) * 0.65
            elif nx > max_p:
                nx = max_p
                vel[0] = -abs(vel[0]) * 0.65
            if nz < -max_p:
                nz = -max_p
                vel[2] = abs(vel[2]) * 0.65
            elif nz > max_p:
                nz = max_p
                vel[2] = -abs(vel[2]) * 0.65

            node.pos = (nx, ny, nz)  # type: ignore[attr-defined]
            # Roulement + inclinaison naturels avec les vagues
            rot = np.asarray(node.rot, dtype=np.float32)  # type: ignore[attr-defined]
            node.rot = (  # type: ignore[attr-defined]
                float(rot[0]) + float(vel[2]) * dt_c / max(fb.radius, 0.1),
                float(rot[1]),
                float(rot[2]) - float(vel[0]) * dt_c / max(fb.radius, 0.1),
            )

            # Couplage retour (sillage subtil si le corps plonge ou glisse vite)
            if fb.wake_strength > 0.0 and sub > 0.05:
                horiz_sp = math.hypot(float(vel[0]), float(vel[2]))
                vert_imp = -float(vel[1]) if vel[1] < -0.35 else 0.0
                # Sillage par 1/60 s : même trace dans l'eau quel que soit le nombre d'images par seconde
                imp = (vert_imp * 0.12 + horiz_sp * 0.025) * fb.wake_strength
                if imp > 0.004:
                    self.splash(nx, nz, strength=-min(imp, 0.18) * dt_c * 60.0, radius=fb.radius * 1.15)

        # Collisions élastiques douces entre bouées flottantes
        for i in range(len(bodies)):
            a = bodies[i]
            for j in range(i + 1, len(bodies)):
                b = bodies[j]
                pa = np.array(a.node.pos, dtype=np.float32)  # type: ignore[attr-defined]
                pb = np.array(b.node.pos, dtype=np.float32)  # type: ignore[attr-defined]
                d = pb - pa
                dist = float(np.linalg.norm(d))
                rsum = a.radius + b.radius
                if 1e-5 < dist < rsum:
                    n_dir = d / dist
                    ov = (rsum - dist) * 0.5
                    pa -= n_dir * ov
                    pb += n_dir * ov
                    a.node.pos = (float(pa[0]), float(pa[1]), float(pa[2]))  # type: ignore[attr-defined]
                    b.node.pos = (float(pb[0]), float(pb[1]), float(pb[2]))  # type: ignore[attr-defined]
                    assert a.vel is not None and b.vel is not None
                    rv = float(np.dot(b.vel - a.vel, n_dir))
                    if rv < 0.0:
                        imp_v = -0.8 * rv * n_dir
                        a.vel -= imp_v
                        b.vel += imp_v


class GpuFloaters:
    """Flotteurs simulés entièrement sur GPU : leurs objets sont dessinés là où le calcul les place.

    Même physique que `GpuWaterSurface.step_floating`, sans aucune lecture vers le CPU. En contrepartie,
    `node.pos` n'est plus à jour : appelez `read()` quand le programme a vraiment besoin des positions
    (cet appel attend le GPU)."""

    def __init__(self, water: GpuWaterSurface, bodies: list[FloatingBody]) -> None:
        if not bodies:
            raise ValueError("GpuFloaters : au moins un flotteur attendu")
        self.water = water
        self.bodies = list(bodies)
        m = len(self.bodies)
        state = np.zeros((m, 16), dtype=np.float32)
        for k, fb in enumerate(self.bodies):
            state[k, 0:3] = fb.node.pos  # type: ignore[attr-defined]
            state[k, 3:6] = fb.node.rot  # type: ignore[attr-defined]
            state[k, 6:8] = 1.0
            state[k, 8:11] = fb.vel
        self.state = upload_f32(state, device=water.device)
        self._kicks = np.zeros((m, 8), dtype=np.float32)
        self._kicks[:, 4:7] = 1.0
        self._params: gs.Array | None = None
        self._params_key: tuple | None = None
        self._bind()

    def _bind(self) -> None:
        for k, fb in enumerate(self.bodies):
            fb.node.bind_transform(self.state, k)  # type: ignore[attr-defined]

    def _params_gpu(self) -> gs.Array:
        key = tuple(
            (b.radius, b.buoyancy, b.gravity, b.wave_push, b.drag, b.vertical_drag, b.wake_strength, 0.0) for b in self.bodies
        )
        if key != self._params_key:
            self._params = upload_f32(np.asarray(key, dtype=np.float32), device=self.water.device)
            self._params_key = key
        return self._params

    def set_velocity(self, body: FloatingBody | int, *, x: float | None = None, y: float | None = None, z: float | None = None) -> None:
        """Impose une ou plusieurs composantes de vitesse (m/s) au prochain pas, sans lire l'état."""
        k = body if isinstance(body, int) else self.bodies.index(body)
        for axis, value in enumerate((x, y, z)):
            if value is not None:
                self._kicks[k, axis] = float(value)
                self._kicks[k, 4 + axis] = 0.0

    def add_velocity(self, body: FloatingBody | int, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Ajoute une vitesse (m/s) au prochain pas, sans lire l'état."""
        k = body if isinstance(body, int) else self.bodies.index(body)
        self._kicks[k, 0:3] += (dx, dy, dz)

    def step(self, dt: float) -> None:
        """Avance les flotteurs et leur sillage (après `water.step`)."""
        dt_c = min(max(float(dt), 0.0), 0.033)
        if dt_c <= 0.0:
            return
        w = self.water
        cfg = w.config
        dev = w.device
        params = self._params_gpu()
        with gs.batch(dev):
            state = floating_step(
                self.state,
                params,
                upload_f32(self._kicks.copy(), device=dev),
                w.h_curr,
                float(cfg.size),
                float(cfg.height_scale),
                float(cfg.ambient_waves),
                float(cfg.ambient_freq),
                float(w.t * cfg.ambient_speed),
                dt_c,
                device=dev,
            )
            if any(b.wake_strength > 0.0 for b in self.bodies):
                wakes = floating_wakes(state, params, w.h_curr, float(cfg.size), dt_c, device=dev)
                w.h_curr, w.h_prev = water_add_wakes(w.h_curr, w.h_prev, wakes, device=dev)
            if len(self.bodies) > 1:
                state = floating_collide(state, params, device=dev)
            self.state = state
        self._kicks[:, 0:4] = 0.0
        self._kicks[:, 4:7] = 1.0
        self._bind()

    def read(self) -> np.ndarray:
        """Rapatrie l'état `(N, 16)` et met à jour `node.pos`, `node.rot` et `vel` (attend le GPU)."""
        state = np.array(self.state.numpy())
        for k, fb in enumerate(self.bodies):
            fb.node.pos = tuple(float(v) for v in state[k, 0:3])  # type: ignore[attr-defined]
            fb.node.rot = tuple(float(v) for v in state[k, 3:6])  # type: ignore[attr-defined]
            fb.vel = state[k, 8:11].copy()
        return state


# ───────────────────────── 2. Fluide Particulaire 3D SPH (`GpuFluidSPH3D`) ─────────────────────────


@dataclass
class FluidSPHConfig:
    """Configuration complète du fluide volumétrique 3D (MAC FLIP Blender + Raytracing 3D 2-rebonds + Whitewater)."""

    count: int = 20000
    solver: str = "mac_flip"  # "mac_flip" (grille MAC décalée Blender), "flip" (colocalisé), "sph" (grille triée)
    grid_res: tuple[int, int, int] = (38, 26, 22)  # résolution grille physique MAC 3D
    vol_res: tuple[int, int, int] = (88, 56, 48)  # résolution volume 3D haute finesse pour raytracing
    vol_blur_iters: int = 2  # passes de flou gaussien 3D séparable (nappes fines et lisses)
    whitewater_count: int = 6000  # micro-particules GPU d'embruns (spray), écume (foam) et bulles
    pressure_iters: int = 26  # itérations Jacobi 3D MAC d'incompressibilité (faites par deux)
    flip_ratio: float = 0.93  # 0.93 = éclaboussures vives (eau), 0.25 = visqueux (miel/lave)
    density_push: float = 1.55  # force de maintien de volume 3D
    volume_iters: int = 4  # itérations Jacobi du maintien de volume (faites par deux)
    shift: float = 0.03  # redistribution des particules au cœur de l'eau (évite les poches creuses)
    bounds: tuple[float, float, float] = (3.4, 4.2, 2.2)  # (half_x, height_y, half_z)
    floor_y: float = 0.0
    floor_colors: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.68, 0.72, 0.78),
        (0.48, 0.54, 0.62),
    )
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    # Solveur "sph" (grille triée) ; l'espacement au repos vaut la moitié du rayon
    smoothing_radius: float = 0.0  # 0 : choisi pour que l'eau remplisse 28 % du bassin
    sound_speed: float = 0.0  # vitesse du son numérique ; 0 : trois fois la vitesse de chute dans le bassin
    viscosity: float = 0.24
    cohesion: float = 0.07
    damping: float = 0.998
    wall_restitution: float = 0.14
    max_speed: float = 13.5
    particle_scale: float = 0.075
    velocity_stretch: float = 0.015
    spin_rate: float = 0.0
    substeps: int = 1
    # Mode de rendu : "raymarch" (raytracing 3D 2-rebonds Snell+TIR), "surface" (SSFR), "particles" (low-poly)
    render_mode: str = "raymarch"
    iso_level: float = 0.34  # seuil d'isosurface dans le volume 3D continu
    ior: float = 1.333  # indice de réfraction physique (1.333 eau, 1.49 miel, 1.52 verre)
    caustics: float = 0.85  # intensité des caustiques animées projetées sur le fond
    surface_scale: float = 2.15  # utilisé en mode "surface" (SSFR)
    smooth_iters: int = 10
    smooth_radius: float = 4.2
    smooth_depth_sigma: float = 0.48
    absorption: tuple[float, float, float] = (0.42, 0.11, 0.04)  # absorption Beer-Lambert par mètre (R, G, B)
    refraction: float = 14.0
    foam_rate: float = 30.0  # particules diffuses par seconde et par particule d'eau très agitée
    foam: float = 0.78  # intensité visuelle des micro-particules whitewater (spray/écume/bulles)
    fixed_dt: float = 1.0 / 60.0  # pas de temps fixe : même physique à 60 Hz, 120 Hz ou 300 FPS
    mesh_style: str = "sphere"  # utilisé en mode "particles" : "sphere", "crystal", "cube"
    albedo: tuple[float, float, float] = (0.02, 0.22, 0.44)
    metallic: float = 0.0
    roughness: float = 0.02
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def water(cls, count: int = 20000) -> FluidSPHConfig:
        """Eau volumétrique réaliste : 20 000 particules MAC FLIP 3D + Raytracing 2-rebonds Snell/TIR + 6 000 whitewater."""
        grid_res = (38, 26, 22) if count >= 14000 else ((30, 20, 18) if count >= 8000 else (18, 12, 12))
        vol_res = (88, 56, 48) if count >= 14000 else ((64, 42, 36) if count >= 8000 else (40, 26, 22))
        ww_n = 6000 if count >= 10000 else 1500
        return cls(
            count=count,
            solver="mac_flip",
            grid_res=grid_res,
            vol_res=vol_res,
            vol_blur_iters=2,
            whitewater_count=ww_n,
            flip_ratio=0.93,
            density_push=1.55,
            render_mode="raymarch",
            iso_level=0.34,
            ior=1.333,
            caustics=0.85,
            absorption=(0.42, 0.11, 0.04),
            foam=0.78,
            albedo=(0.02, 0.22, 0.44),
            metallic=0.0,
            roughness=0.02,
        )

    @classmethod
    def water_sph(cls, count: int = 20000) -> FluidSPHConfig:
        """Eau simulée en SPH sur grille triée (sans grille de pression), même rendu que `water`."""
        return replace(cls.water(count), solver="sph")

    @classmethod
    def low_poly(cls, count: int = 360, *, mesh_style: str = "cube") -> FluidSPHConfig:
        """Fluide 3D low-poly / stylisé (cubes ou cristaux facettés qui tournent dans le courant)."""
        return cls(
            count=count,
            solver="sph",
            smoothing_radius=0.66,
            render_mode="particles",
            particle_scale=0.25,
            spin_rate=1.6,
            mesh_style=mesh_style,
            albedo=(0.14, 0.78, 0.98),
            metallic=0.18,
            roughness=0.22,
            emissive=(0.02, 0.10, 0.16),
        )

    @classmethod
    def honey(cls, count: int = 10000) -> FluidSPHConfig:
        """Miel / sirop doré volumétrique épais et translucide (IOR 1.49)."""
        return cls(
            count=count,
            solver="mac_flip",
            flip_ratio=0.22,
            density_push=1.15,
            damping=0.984,
            wall_restitution=0.05,
            render_mode="raymarch",
            iso_level=0.36,
            ior=1.49,
            caustics=0.45,
            absorption=(0.12, 0.55, 2.20),
            foam=0.0,
            albedo=(0.98, 0.60, 0.05),
            metallic=0.0,
            roughness=0.06,
            emissive=(0.06, 0.02, 0.0),
        )

    @classmethod
    def lava(cls, count: int = 10000) -> FluidSPHConfig:
        """Magma volumétrique incandescent (haute viscosité, croûte sombre + cœur brûlant)."""
        return cls(
            count=count,
            solver="mac_flip",
            flip_ratio=0.20,
            density_push=1.10,
            damping=0.982,
            wall_restitution=0.05,
            render_mode="raymarch",
            iso_level=0.38,
            ior=1.50,
            caustics=0.0,
            absorption=(4.5, 6.0, 8.0),
            foam=0.0,
            mesh_style="crystal",
            spin_rate=0.6,
            albedo=(0.25, 0.04, 0.02),
            metallic=0.05,
            roughness=0.30,
            emissive=(1.95, 0.42, 0.05),
        )

    @classmethod
    def mercury(cls, count: int = 12000) -> FluidSPHConfig:
        """Mercure liquide volumétrique chromé (métal liquide miroir 100 % réfléchissant)."""
        return cls(
            count=count,
            solver="mac_flip",
            flip_ratio=0.92,
            density_push=1.45,
            wall_restitution=0.25,
            render_mode="raymarch",
            iso_level=0.36,
            ior=1.60,
            caustics=0.0,
            absorption=(8.0, 8.0, 8.0),
            foam=0.0,
            albedo=(0.94, 0.96, 1.0),
            metallic=1.0,
            roughness=0.025,
            emissive=(0.0, 0.0, 0.0),
        )

    def make_mesh(self) -> Mesh:
        if self.mesh_style == "cube":
            return Mesh.box(1.5, 1.5, 1.5, sub=1)
        if self.mesh_style == "crystal":
            return Mesh.crystal(radius=0.95, height=1.7, sides=6)
        return Mesh.sphere(1.0, rings=7, sectors=10)

    def make_material(self) -> Material:
        return Material(
            albedo=self.albedo,
            metallic=self.metallic,
            roughness=self.roughness,
            emissive=self.emissive,
        )


class GpuFluidSPH3D:
    """Solveur de fluide 3D SPH 100 % GPU (0 copie CPU → `InstancedMesh3D.gpu_trs`)."""

    def __init__(
        self,
        config: FluidSPHConfig | None = None,
        *,
        layout: str = "dam_break",
        device: str | None = None,
    ) -> None:
        self.config = config or FluidSPHConfig.water()
        self.device = device
        self.visible: bool = True
        self.t = 0.0
        self._accum = 0.0
        self._pulse = [0.0, 0.0]

        n = int(self.config.count)
        self._trs_tpl = upload_f32(np.zeros((n, 8), dtype=np.float32), device=device)
        gx_n, gy_n, gz_n = self.config.grid_res
        self._ncells = int(gx_n * gy_n * gz_n)
        self._grid_tpl = upload_f32(np.zeros((self._ncells,), dtype=np.float32), device=device)
        self._p_grid = upload_f32(np.zeros((self._ncells,), dtype=np.float32), device=device)
        self._q_grid = upload_f32(np.zeros((self._ncells,), dtype=np.float32), device=device)
        # Templates pour la grille MAC décalée (faces U, V, W) et le volume 3D de raymarching
        self._u_tpl = upload_f32(np.zeros(((gx_n + 1) * gy_n * gz_n,), dtype=np.float32), device=device)
        self._v_tpl = upload_f32(np.zeros((gx_n * (gy_n + 1) * gz_n,), dtype=np.float32), device=device)
        self._w_tpl = upload_f32(np.zeros((gx_n * gy_n * (gz_n + 1),), dtype=np.float32), device=device)
        dx_v, dy_v, dz_v = self.config.vol_res
        self._vol_tpl = upload_f32(np.zeros((dx_v * dy_v * dz_v,), dtype=np.float32), device=device)
        self.vol_dens_gpu: gs.Array | None = None
        self._raw_dens_gpu: gs.Array | None = None
        self._c4_tpl = upload_f32(np.zeros((4,), dtype=np.float32), device=device)
        self._height_tpl = upload_f32(np.zeros((dz_v, dx_v), dtype=np.float32), device=device)
        self._photon_tpl = upload_f32(np.zeros((dz_v * 6, dx_v * 6), dtype=np.float32), device=device)
        self._map_tpl = upload_f32(np.zeros((dz_v * 3, dx_v * 3), dtype=np.float32), device=device)
        self._caustics: tuple[gs.Array, tuple[float, float, float], gs.Array] | None = None
        self._ww_frame: int = 0
        ww_n = max(1, int(self.config.whitewater_count))
        self.ww_pos_gpu = upload_f32(np.zeros((ww_n, 4), dtype=np.float32), device=device)
        self.ww_vel_gpu = upload_f32(np.zeros((ww_n, 4), dtype=np.float32), device=device)
        if self.config.solver == "sph":
            self._init_sph()
        self.reset(layout=layout)

    def _init_sph(self) -> None:
        """Grandeurs fixes du solveur SPH : espacement, vitesse du son, grille et taille du tri."""
        cfg = self.config
        n = int(cfg.count)
        bx, by, bz = cfg.bounds
        if n > 1 << 18:
            raise ValueError(f'solver="sph" : 262 144 particules au plus, reçu {n}')
        if cfg.smoothing_radius > 0.0:
            self._sph_h = float(cfg.smoothing_radius)
        else:
            self._sph_h = 2.0 * (4.0 * bx * by * bz * 0.28 / n) ** (1.0 / 3.0)
        self._sph_spacing = 0.5 * self._sph_h
        grav = math.sqrt(sum(g * g for g in cfg.gravity))
        self._sph_sound = float(cfg.sound_speed) or max(3.0 * math.sqrt(2.0 * grav * by * 0.72), 1.0)
        # Cellules un peu plus larges que h : au plus 16 384, la clé de tri ne prévoit pas plus
        cell = self._sph_h * 1.001
        while True:
            dims = tuple(max(1, math.ceil(d / cell)) for d in (2.0 * bx, by, 2.0 * bz))
            if dims[0] * dims[1] * dims[2] <= 1 << 14:
                break
            cell *= 1.05
        self._sph_cell, self._sph_dims = cell, dims
        self._sph_keys = 1 << max(0, (n - 1).bit_length())
        self._sph_key_tpl = upload_u32(np.zeros((self._sph_keys,), dtype=np.uint32), device=self.device)
        self._sph_cell_tpl = upload_f32(np.zeros((dims[0] * dims[1] * dims[2],), dtype=np.float32), device=self.device)

    def _sph_substeps(self, tick: float) -> int:
        """Sous-pas d'un pas de simulation : condition CFL sur la vitesse du son."""
        limit = 0.4 * self._sph_h / self._sph_sound
        return max(1, int(self.config.substeps), math.ceil(tick / limit))

    def _sph_lattice(self, layout: str) -> np.ndarray:
        """Positions de départ sur un réseau régulier : un tirage au hasard superposerait des particules."""
        cfg = self.config
        n = int(cfg.count)
        bx, by, bz = cfg.bounds
        d = self._sph_spacing
        rng = np.random.default_rng(1337)
        kz = max(1, int(2.0 * bz / d))
        if layout == "column":
            # Deux fois plus haute que large, dans la limite du bassin
            side = max(2, round((n / 2.0) ** (1.0 / 3.0)), math.ceil(math.sqrt(n / max(1, int(by * 0.9 / d)))))
            kx, kz = min(side, int(2.0 * bx / d)), min(side, kz)
        else:
            ky = max(1, int(by * 0.72 / d))
            kx = min(max(1, math.ceil(n / (ky * kz))), int(2.0 * bx / d))
        idx = np.arange(n)
        ix, iz, iy = idx % kx, (idx // kx) % kz, idx // (kx * kz)
        p = np.empty((n, 3), dtype=np.float32)
        p[:, 0] = (ix + 0.5) * d - (bx if layout != "column" else 0.5 * kx * d)
        if layout == "dual_wave":
            right = ix >= kx // 2
            p[right, 0] += 2.0 * bx - kx * d
        p[:, 1] = cfg.floor_y + (iy + 0.5) * d
        p[:, 2] = (iz + 0.5) * d - 0.5 * kz * d
        return p + rng.uniform(-0.02 * d, 0.02 * d, p.shape).astype(np.float32)

    def reset(self, *, layout: str = "dam_break") -> gs.Array:
        """Réinitialise les particules (`"dam_break"`, `"column"`, `"rain"`, `"dual_wave"`)."""
        cfg = self.config
        n = int(cfg.count)
        bx, by, bz = cfg.bounds
        rng = np.random.default_rng(1337)

        p4 = np.zeros((n, 4), dtype=np.float32)
        v4 = np.zeros((n, 4), dtype=np.float32)
        rad = cfg.particle_scale

        if layout == "column":
            # Colonne d'eau qui s'effondre au centre
            side = max(2, int(round(n ** (1.0 / 3.0))))
            spacing = rad * 1.45
            idx = 0
            for iy in range(side * 2):
                for ix in range(side):
                    for iz in range(side):
                        if idx >= n:
                            break
                        p4[idx, 0] = (ix - side * 0.5 + 0.5) * spacing + rng.uniform(-0.02, 0.02)
                        p4[idx, 1] = cfg.floor_y + rad + 0.4 + iy * spacing
                        p4[idx, 2] = (iz - side * 0.5 + 0.5) * spacing + rng.uniform(-0.02, 0.02)
                        idx += 1
        elif layout == "rain":
            p4[:, 0] = rng.uniform(-bx + rad, bx - rad, n)
            p4[:, 1] = rng.uniform(cfg.floor_y + 0.5, cfg.floor_y + by - rad, n)
            p4[:, 2] = rng.uniform(-bz + rad, bz - rad, n)
            v4[:, 1] = rng.uniform(-2.0, 0.0, n)
        elif layout == "dual_wave":
            half_n = n // 2
            p4[:half_n, 0] = rng.uniform(-bx + rad, -bx * 0.25, half_n)
            p4[half_n:, 0] = rng.uniform(bx * 0.25, bx - rad, n - half_n)
            p4[:, 1] = rng.uniform(cfg.floor_y + rad + 0.1, cfg.floor_y + by * 0.72, n)
            p4[:, 2] = rng.uniform(-bz + rad, bz - rad, n)
            v4[:half_n, 0] = 2.8
            v4[half_n:, 0] = -2.8
        else:
            # Dam-break classique : grand bloc d'eau dense sur le côté gauche qui déferle vers la droite
            vol_x = bx * 0.92
            vol_z = bz * 1.80
            vol_y = by * 0.72
            p4[:, 0] = rng.uniform(-bx + rad + 0.06, -bx + rad + 0.06 + vol_x, n)
            p4[:, 1] = rng.uniform(cfg.floor_y + rad + 0.06, cfg.floor_y + rad + 0.06 + vol_y, n)
            p4[:, 2] = rng.uniform(-vol_z * 0.5, vol_z * 0.5, n)

        if cfg.solver == "sph" and layout != "rain":
            p4[:, :3] = self._sph_lattice(layout)
            if layout == "dual_wave":
                v4[:, 0] = np.where(p4[:, 0] < 0.0, 2.8, -2.8)
        p4[:, 3] = rad
        self.pos_gpu = upload_f32(p4, device=self.device)
        self.vel_gpu = upload_f32(v4, device=self.device)
        self._p_grid = upload_f32(np.zeros((self._ncells,), dtype=np.float32), device=self.device)
        self._q_grid = upload_f32(np.zeros((self._ncells,), dtype=np.float32), device=self.device)
        ww_n = max(1, int(cfg.whitewater_count))
        self.ww_pos_gpu = upload_f32(np.zeros((ww_n, 4), dtype=np.float32), device=self.device)
        self.ww_vel_gpu = upload_f32(np.zeros((ww_n, 4), dtype=np.float32), device=self.device)
        self.trs_gpu = sph_to_trs(
            self.pos_gpu,
            self.vel_gpu,
            self._trs_tpl,
            float(cfg.particle_scale),
            float(cfg.velocity_stretch),
            float(cfg.spin_rate),
            float(self.t),
            device=self.device,
        )
        if cfg.render_mode == "raymarch":
            with gs.batch(self.device):
                self._rebuild_volume_in_batch()
        return self.trs_gpu

    def _sph_step(self, dt: float, accel: tuple[float, float, float]) -> None:
        """Un sous-pas SPH : tri par cellule, densités, forces et intégration."""
        cfg = self.config
        dev = self.device
        bx, by, bz = (float(b) for b in cfg.bounds)
        fy = float(cfg.floor_y)
        dims, cell, h = self._sph_dims, float(self._sph_cell), float(self._sph_h)
        mass = self._sph_spacing**3
        keys = sph_cell_keys(self.pos_gpu, self._sph_key_tpl, *dims, cell, bx, bz, fy, device=dev)
        keys = sort_keys(keys, self._sph_keys, device=dev)
        start, count = sph_cell_ranges(keys, self._sph_cell_tpl, device=dev)
        spos = sph_gather(self.pos_gpu, keys, device=dev)
        svel = sph_gather(self.vel_gpu, keys, device=dev)
        rho = sph_density(spos, start, count, *dims, cell, h, mass, bx, by, bz, fy, device=dev)
        self.pos_gpu, self.vel_gpu = sph_integrate(
            self.pos_gpu,
            self.vel_gpu,
            spos,
            svel,
            rho,
            keys,
            start,
            count,
            *dims,
            float(dt),
            cell,
            h,
            mass,
            float(self._sph_sound),
            float(cfg.viscosity * h * self._sph_sound * 0.02),
            float(cfg.cohesion * self._sph_sound**2 * 0.02 / h),
            *(float(a) for a in accel),
            bx,
            by,
            bz,
            fy,
            float(cfg.wall_restitution),
            float(cfg.damping) ** (dt * 60.0),  # `damping` vaut pour 1/60 s
            float(cfg.max_speed),
            device=dev,
        )

    def _rebuild_volume_in_batch(self) -> None:
        """Construit et lisse le volume 3D continu (densité) directement en VRAM."""
        cfg = self.config
        dx_v, dy_v, dz_v = cfg.vol_res
        bx, by, bz = cfg.bounds
        # Normalise la densité pour que le cœur liquide atteigne ~1.0 quel que soit `count`
        expected_per_vox = max(0.05, float(cfg.count) / max(float(dx_v * dy_v * dz_v) * 0.25, 1.0))
        dens_scale = DENSITY_SCALE / expected_per_vox
        du = vol_splat_density(
            self.pos_gpu,
            self._vol_tpl,
            int(dx_v),
            int(dy_v),
            int(dz_v),
            float(bx),
            float(by),
            float(bz),
            float(cfg.floor_y),
            device=self.device,
        )
        d_f = vol_unpack_f32(du, float(dens_scale), device=self.device)
        for _ in range(max(0, int(cfg.vol_blur_iters))):
            d_f = vol_blur_axis(d_f, int(dx_v), int(dy_v), int(dz_v), 0, device=self.device)
            d_f = vol_blur_axis(d_f, int(dx_v), int(dy_v), int(dz_v), 1, device=self.device)
            d_f = vol_blur_axis(d_f, int(dx_v), int(dy_v), int(dz_v), 2, device=self.device)
        self._raw_dens_gpu = d_f  # sans la retouche des parois, pour `mac_shift`
        self.vol_dens_gpu = vol_wet_walls(d_f, int(dx_v), int(dy_v), int(dz_v), device=self.device)

    def caustic_map(self, sun_dir) -> gs.Array:
        """Éclairement du fond par le soleil réfracté (1 = eau plate) ; recalculé si l'eau ou le soleil a changé."""
        sun = (float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2]))
        dens = self.vol_dens_gpu
        if self._caustics is not None and self._caustics[0] is dens and self._caustics[1] == sun:
            return self._caustics[2]
        cfg = self.config
        dev = self.device
        dx_v, dy_v, dz_v = cfg.vol_res
        bx, by, bz = cfg.bounds
        height = water_height(dens, self._height_tpl, int(dy_v), float(cfg.iso_level), device=dev)
        for _ in range(2):
            height = height_blur(height, 0, device=dev)
            height = height_blur(height, 1, device=dev)
        photons = caustic_photons(
            height,
            self._photon_tpl,
            self._map_tpl,
            int(dy_v),
            float(bx),
            float(by),
            float(bz),
            *sun,
            float(cfg.ior),
            device=dev,
        )
        # Photons par case pour une surface plate : la grille couvre 1,5 × 1,5 fois le bassin
        expected = (6 / 1.5 / 3) ** 2  # grille ×6, carte ×3
        out = caustic_resolve(photons, float(expected), device=dev)
        self._caustics = (dens, sun, out)
        return out

    def extract_surface_mesh(self, *, iso_level: float | None = None) -> Mesh:
        """Extrait un maillage 3D facetté (style Blender / Low-Poly) depuis le volume de densité 3D."""
        if self.vol_dens_gpu is None:
            with gs.batch(self.device):
                self._rebuild_volume_in_batch()
        assert self.vol_dens_gpu is not None
        dx_v, dy_v, dz_v = self.config.vol_res
        bx, by, bz = self.config.bounds
        iso = float(self.config.iso_level if iso_level is None else iso_level)
        vol = self.vol_dens_gpu.numpy().reshape(dz_v, dy_v, dx_v)
        # Extraction rapide des faces frontières (Surface Nets / Voxel-Mesh propre, 0 dépendance)
        occ = vol >= iso
        sx = (2.0 * bx) / max(dx_v - 1, 1)
        sy = by / max(dy_v - 1, 1)
        sz = (2.0 * bz) / max(dz_v - 1, 1)
        verts: list[tuple[float, float, float]] = []
        norms: list[tuple[float, float, float]] = []
        # 6 directions : (+x, -x, +y, -y, +z, -z)
        padded = np.pad(occ, 1, mode="constant", constant_values=False)
        zs, ys, xs = np.nonzero(occ)
        for iz, iy, ix in zip(zs.tolist(), ys.tolist(), xs.tolist()):
            x0 = -bx + ix * sx
            y0 = self.config.floor_y + iy * sy
            z0 = -bz + iz * sz
            x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz
            pz, py, px = iz + 1, iy + 1, ix + 1
            if not padded[pz, py + 1, px]:  # +Y
                verts.extend([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x0, y1, z0), (x1, y1, z1), (x1, y1, z0)])
                norms.extend([(0.0, 1.0, 0.0)] * 6)
            if not padded[pz, py - 1, px]:  # -Y
                verts.extend([(x0, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1)])
                norms.extend([(0.0, -1.0, 0.0)] * 6)
            if not padded[pz, py, px + 1]:  # +X
                verts.extend([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z0), (x1, y1, z1), (x1, y0, z1)])
                norms.extend([(1.0, 0.0, 0.0)] * 6)
            if not padded[pz, py, px - 1]:  # -X
                verts.extend([(x0, y0, z0), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)])
                norms.extend([(-1.0, 0.0, 0.0)] * 6)
            if not padded[pz + 1, py, px]:  # +Z
                verts.extend([(x0, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1)])
                norms.extend([(0.0, 0.0, 1.0)] * 6)
            if not padded[pz - 1, py, px]:  # -Z
                verts.extend([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x0, y0, z0), (x1, y1, z0), (x1, y0, z0)])
                norms.extend([(0.0, 0.0, -1.0)] * 6)
        if not verts:
            return Mesh.box(0.1, 0.1, 0.1, sub=1)
        pos_arr = np.asarray(verts, dtype=np.float32)
        nrm_arr = np.asarray(norms, dtype=np.float32)
        uv_arr = np.zeros((pos_arr.shape[0], 2), dtype=np.float32)
        idx_arr = np.arange(pos_arr.shape[0], dtype=np.int32).reshape(-1, 3)
        return Mesh(positions=pos_arr, normals=nrm_arr, uvs=uv_arr, indices=idx_arr)

    def push_wave(self, force_x: float = 28.0, force_z: float = 0.0) -> None:
        """Envoie une impulsion de vague latérale au prochain `step()`."""
        self._pulse[0] += float(force_x)
        self._pulse[1] += float(force_z)

    def step(self, dt: float) -> gs.Array:
        """Exécute la simulation 3D à pas fixe 100 % GPU (`flip` O(N) ou `sph`) et renvoie `trs_gpu` `(N, 8)`."""
        cfg = self.config
        fixed_dt = max(float(cfg.fixed_dt), 1.0 / 240.0)
        self._accum = min(self._accum + max(float(dt), 0.0), fixed_dt * 2.0)
        n_ticks = int(self._accum / fixed_dt)
        if n_ticks <= 0:
            self.t += float(dt)
            return self.trs_gpu
        self._accum -= n_ticks * fixed_dt

        steps = n_ticks * max(1, int(cfg.substeps))
        sub_dt = fixed_dt / max(1, int(cfg.substeps))
        bx, by, bz = cfg.bounds
        gx, gy, gz = cfg.gravity
        px_f, pz_f = self._pulse[0], self._pulse[1]
        self._pulse = [0.0, 0.0]

        mac_faces = None
        with gs.batch(self.device):
            if cfg.solver == "mac_flip":
                gx_n, gy_n, gz_n = cfg.grid_res
                rest_cnt = max(0.8, float(cfg.count) / max(float(gx_n * gy_n * gz_n) * 0.28, 1.0))
                spacing = (4.0 * bx * by * bz * 0.28 / cfg.count) ** (1.0 / 3.0)
                if self._raw_dens_gpu is not None and cfg.shift > 0.0:
                    self.pos_gpu = mac_shift(
                        self.pos_gpu,
                        self._raw_dens_gpu,
                        *(int(d) for d in cfg.vol_res),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        VOLUME_CORE,
                        float(cfg.shift),
                        float(spacing),
                        device=self.device,
                    )
                for s in range(steps):
                    fill = mac_fill(
                        mac_mass_splat(
                            self.pos_gpu,
                            self._grid_tpl,
                            int(gx_n),
                            int(gy_n),
                            int(gz_n),
                            float(bx),
                            float(by),
                            float(bz),
                            float(cfg.floor_y),
                            device=self.device,
                        ),
                        float(rest_cnt),
                        device=self.device,
                    )
                    u_sum, u_wt, v_sum, v_wt, w_sum, w_wt, c_cnt = mac_p2g(
                        self.pos_gpu,
                        self.vel_gpu,
                        self._u_tpl,
                        self._v_tpl,
                        self._w_tpl,
                        self._grid_tpl,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        device=self.device,
                    )
                    ax_dt = float((gx + (px_f if s == 0 else 0.0)) * sub_dt)
                    ay_dt = float(gy * sub_dt)
                    az_dt = float((gz + (pz_f if s == 0 else 0.0)) * sub_dt)
                    u_old, u_star = mac_face_forces(u_sum, u_wt, int(gx_n + 1), int(gy_n), 0, int(gx_n), ax_dt, device=self.device)
                    v_old, v_star = mac_face_forces(v_sum, v_wt, int(gx_n), int(gy_n + 1), 1, int(gy_n), ay_dt, device=self.device)
                    w_old, w_star = mac_face_forces(w_sum, w_wt, int(gx_n), int(gy_n), 2, int(gz_n), az_dt, device=self.device)

                    div = mac_divergence(
                        u_star, v_star, w_star, fill, int(gx_n), int(gy_n), int(gz_n), device=self.device
                    )
                    p = self._p_grid
                    diag = mac_pressure_diag(fill, int(gx_n), int(gy_n), int(gz_n), device=self.device)
                    for _ in range(max(1, (int(cfg.pressure_iters) + 1) // 2)):
                        p = mac_pressure_jacobi2(p, div, diag, int(gx_n), int(gy_n), int(gz_n), device=self.device)
                    self._p_grid = p

                    u_new = mac_project_face(u_star, p, fill, int(gx_n), int(gy_n), int(gz_n), 0, device=self.device)
                    v_new = mac_project_face(v_star, p, fill, int(gx_n), int(gy_n), int(gz_n), 1, device=self.device)
                    w_new = mac_project_face(w_star, p, fill, int(gx_n), int(gy_n), int(gz_n), 2, device=self.device)

                    mac_faces, mac_rest = (u_new, v_new, w_new, fill), rest_cnt
                    self.pos_gpu, self.vel_gpu = mac_g2p(
                        self.pos_gpu,
                        self.vel_gpu,
                        u_new,
                        v_new,
                        w_new,
                        u_old,
                        v_old,
                        w_old,
                        c_cnt,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        sub_dt,
                        float(cfg.flip_ratio),
                        float(cfg.damping),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        float(cfg.wall_restitution),
                        float(cfg.max_speed),
                        device=self.device,
                    )
                    # Volume : corrigé par les positions, sans toucher aux vitesses
                    src = mac_volume_source(fill, int(gx_n), int(gy_n), int(gz_n), float(cfg.density_push), device=self.device)
                    q = self._q_grid
                    for _ in range(max(1, (int(cfg.volume_iters) + 1) // 2)):
                        q = mac_pressure_jacobi2(q, src, diag, int(gx_n), int(gy_n), int(gz_n), device=self.device)
                    self._q_grid = q
                    self.pos_gpu = mac_volume_move(
                        self.pos_gpu,
                        q,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        sub_dt,
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        device=self.device,
                    )
            elif cfg.solver == "flip":
                gx_n, gy_n, gz_n = cfg.grid_res
                # Masse au repos calibrée sur le volume occupé (~1/3 de la grille)
                rest_mass = max(1.5, float(cfg.count) / max(float(gx_n * gy_n * gz_n) * 0.28, 1.0))
                for s in range(steps):
                    gm_u, gvx_u, gvy_u, gvz_u = flip_p2g(
                        self.pos_gpu,
                        self.vel_gpu,
                        self._grid_tpl,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        device=self.device,
                    )
                    vx_o, vy_o, vz_o, mass_f = flip_grid_forces(
                        gm_u,
                        gvx_u,
                        gvy_u,
                        gvz_u,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        sub_dt,
                        float(gx),
                        float(gy),
                        float(gz),
                        float(px_f if s == 0 else 0.0),
                        float(pz_f if s == 0 else 0.0),
                        device=self.device,
                    )
                    div = flip_grid_divergence(
                        vx_o,
                        vy_o,
                        vz_o,
                        mass_f,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        float(rest_mass),
                        float(cfg.density_push),
                        device=self.device,
                    )
                    p = self._p_grid
                    for _ in range(max(1, int(cfg.pressure_iters))):
                        p = flip_grid_pressure_jacobi(p, div, mass_f, int(gx_n), int(gy_n), int(gz_n), device=self.device)
                    self._p_grid = p
                    vx_n, vy_n, vz_n = flip_grid_project(
                        vx_o, vy_o, vz_o, p, mass_f, int(gx_n), int(gy_n), int(gz_n), device=self.device
                    )
                    self.pos_gpu, self.vel_gpu = flip_g2p(
                        self.pos_gpu,
                        self.vel_gpu,
                        vx_n,
                        vy_n,
                        vz_n,
                        vx_o,
                        vy_o,
                        vz_o,
                        int(gx_n),
                        int(gy_n),
                        int(gz_n),
                        sub_dt,
                        float(gx),
                        float(gy),
                        float(gz),
                        float(px_f if s == 0 else 0.0),
                        float(pz_f if s == 0 else 0.0),
                        float(cfg.flip_ratio),
                        float(cfg.damping),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        float(cfg.wall_restitution),
                        float(cfg.max_speed),
                        device=self.device,
                    )
            else:
                n_sub = n_ticks * self._sph_substeps(fixed_dt)
                for s in range(n_sub):
                    # L'impulsion des vagues vaut une accélération pendant un pas entier
                    pulse = s < n_sub // n_ticks
                    self._sph_step(
                        fixed_dt * n_ticks / n_sub,
                        (gx + (px_f if pulse else 0.0), gy, gz + (pz_f if pulse else 0.0)),
                    )
            self.t += float(dt)
            if cfg.render_mode == "raymarch":
                self._rebuild_volume_in_batch()
                if cfg.foam > 0.0 and cfg.whitewater_count > 0 and self.vol_dens_gpu is not None:
                    self._ww_frame += 1
                    wnx, wny, wnz = cfg.grid_res
                    if mac_faces is not None:
                        cells = mac_cell_velocity(*mac_faces, self._c4_tpl, float(mac_rest), int(wnx), int(wny), int(wnz), device=self.device)
                    else:
                        cells = fluid_velocity_resolve(
                            *fluid_velocity_splat(
                                self.pos_gpu,
                                self.vel_gpu,
                                self._grid_tpl,
                                int(wnx),
                                int(wny),
                                int(wnz),
                                float(bx),
                                float(by),
                                float(bz),
                                float(cfg.floor_y),
                                device=self.device,
                            ),
                            self._c4_tpl,
                            device=self.device,
                        )
                    rest_mass = max(0.5, float(cfg.count) / max(float(wnx * wny * wnz) * 0.40, 1.0))
                    dx_v, dy_v, dz_v = cfg.vol_res
                    self.ww_pos_gpu, self.ww_vel_gpu = ww_step(
                        self.ww_pos_gpu,
                        self.ww_vel_gpu,
                        self.pos_gpu,
                        self.vel_gpu,
                        cells,
                        self.vol_dens_gpu,
                        int(wnx),
                        int(wny),
                        int(wnz),
                        int(dx_v),
                        int(dy_v),
                        int(dz_v),
                        float(bx),
                        float(by),
                        float(bz),
                        float(cfg.floor_y),
                        float(cfg.iso_level),
                        rest_mass,
                        float(abs(gy)),
                        fixed_dt * n_ticks,
                        int(self._ww_frame),
                        float(cfg.foam_rate),
                        device=self.device,
                    )
            elif cfg.render_mode == "particles":
                self.trs_gpu = sph_to_trs(
                    self.pos_gpu,
                    self.vel_gpu,
                    self._trs_tpl,
                    float(cfg.particle_scale),
                    float(cfg.velocity_stretch),
                    float(cfg.spin_rate),
                    float(self.t),
                    device=self.device,
                )
        return self.trs_gpu


# Alias ergonomiques pour l'API fluide 3D nouvelle génération
Fluid3DConfig = FluidSPHConfig
GpuFluid3D = GpuFluidSPH3D


# ───────────────────────── 3. Fluide 2D Eulérien Navier-Stokes (`GpuFluid2D`) ─────────────────────────


@dataclass
class Fluid2DConfig:
    """Configuration complète du fluide 2D Navier-Stokes (du soyeux réaliste au pixel-art / cel-shaded)."""

    width: int = 192
    height: int = 128
    velocity_dissipation: float = 0.992  # vitesse conservée par 1/60 s
    dye_dissipation: float = 0.991  # encre conservée par 1/60 s
    pressure_iters: int = 16
    curl_strength: float = 3.2  # confinement de vorticité (0 = laminaire, 3-6 = tourbillons riches)
    buoyancy: float = 0.0  # > 0 pour fumée / feu qui monte
    bg_color: tuple[float, float, float] = (0.02, 0.025, 0.05)
    exposure: float = 1.25
    posterize_levels: int = 0  # 0 = dégradé lisse réaliste ; 4..6 = bandes cel-shaded / low-poly
    bilinear: bool = True  # False = rendu pixel-art net
    fixed_dt: float = 1.0 / 60.0  # pas de temps fixe : même résultat à 30, 60 ou 240 FPS

    @classmethod
    def ink(cls, width: int = 192, height: int = 128) -> Fluid2DConfig:
        """Encre / plasma fluide réaliste aux tourbillons fins et soyeux."""
        return cls(width=width, height=height, curl_strength=3.6, dye_dissipation=0.993, bilinear=True)

    @classmethod
    def smoke_fire(cls, width: int = 160, height: int = 128) -> Fluid2DConfig:
        """Fumée / flammes montantes avec flottabilité thermique."""
        return cls(
            width=width,
            height=height,
            velocity_dissipation=0.985,
            dye_dissipation=0.978,
            curl_strength=4.5,
            buoyancy=26.0,
            bg_color=(0.015, 0.01, 0.02),
            exposure=1.4,
        )

    @classmethod
    def low_poly_pixel(cls, width: int = 64, height: int = 44) -> Fluid2DConfig:
        """Fluide stylisé low-poly / pixel-art cel-shaded (basse résolution + quantification de teintes)."""
        return cls(
            width=width,
            height=height,
            velocity_dissipation=0.990,
            dye_dissipation=0.992,
            pressure_iters=12,
            curl_strength=3.0,
            posterize_levels=5,
            bilinear=False,
            exposure=1.35,
        )


class GpuFluid2D:
    """Solveur Navier-Stokes 2D (Stable Fluids) 100 % GPU avec `Texture` 0-copie (2D pur ou écran 3D)."""

    def __init__(self, config: Fluid2DConfig | None = None, *, device: str | None = None) -> None:
        self.config = config or Fluid2DConfig.ink()
        self.device = device
        self.t = 0.0
        self._accum = 0.0

        h, w = int(self.config.height), int(self.config.width)
        z = np.zeros((h, w), dtype=np.float32)
        self.vx = upload_f32(z, device=device)
        self.vy = upload_f32(z, device=device)
        self.p = upload_f32(z, device=device)
        self.dr = upload_f32(z, device=device)
        self.dg = upload_f32(z, device=device)
        self.db = upload_f32(z, device=device)

        self.frame_gpu = self._pack_frame()
        self.texture = Texture.from_gpu(self.frame_gpu, bilinear=self.config.bilinear)

    def _pack_frame(self) -> gs.Array:
        cfg = self.config
        return fluid2d_render_u32(
            self.dr,
            self.dg,
            self.db,
            float(cfg.bg_color[0]),
            float(cfg.bg_color[1]),
            float(cfg.bg_color[2]),
            float(cfg.exposure),
            float(cfg.posterize_levels),
            device=self.device,
        )

    def clear(self) -> None:
        h, w = int(self.config.height), int(self.config.width)
        z = np.zeros((h, w), dtype=np.float32)
        self.vx = upload_f32(z, device=self.device)
        self.vy = upload_f32(z, device=self.device)
        self.p = upload_f32(z, device=self.device)
        self.dr = upload_f32(z, device=self.device)
        self.dg = upload_f32(z, device=self.device)
        self.db = upload_f32(z, device=self.device)
        self.frame_gpu = self._pack_frame()
        self.texture.update(self.frame_gpu)

    def splat(
        self,
        u: float,
        v: float,
        dx: float,
        dy: float,
        *,
        color: tuple[float, float, float] = (0.2, 0.85, 1.0),
        radius: float = 0.05,
        force: float = 120.0,
    ) -> None:
        """Injecte un jet d'encre + vitesse aux coordonnées normalisées `(u, v) ∈ [0, 1]²`."""
        w = float(self.config.width)
        h = float(self.config.height)
        cx = float(np.clip(u, 0.02, 0.98)) * (w - 1.0)
        cy = float(np.clip(v, 0.02, 0.98)) * (h - 1.0)
        r_px = max(1.5, float(radius) * min(w, h))
        inv_r2 = 1.0 / (r_px * r_px)
        with gs.batch(self.device):
            self.vx, self.vy = fluid2d_splat_vel(
                self.vx, self.vy, cx, cy, float(dx * force), float(dy * force), inv_r2, device=self.device
            )
            self.dr, self.dg, self.db = fluid2d_splat_dye(
                self.dr,
                self.dg,
                self.db,
                cx,
                cy,
                float(color[0]),
                float(color[1]),
                float(color[2]),
                inv_r2,
                device=self.device,
            )

    def step(self, dt: float) -> gs.Array:
        """Avance Navier-Stokes 2D sur GPU (advection → vorticité → Jacobi → teinture) et met à jour `self.texture`."""
        cfg = self.config
        dt_phys = max(float(cfg.fixed_dt), 1.0 / 240.0)
        dt_s = dt_phys * 60.0  # vitesses en cellules par 1/60 s
        self._accum = min(self._accum + max(float(dt), 0.0), dt_phys * 3.0)
        n_ticks = int(self._accum / dt_phys + 1e-6)
        self._accum -= n_ticks * dt_phys
        with gs.batch(self.device):
            for _ in range(n_ticks):
                self._tick(dt_phys, dt_s)
            self.frame_gpu = self._pack_frame()
        self.t += float(dt)
        self.texture.bilinear = bool(cfg.bilinear)
        self.texture.update(self.frame_gpu)
        return self.frame_gpu

    def _tick(self, dt_phys: float, dt_s: float) -> None:
        """Un pas fixe : advection, tourbillons, pression, teinture."""
        cfg = self.config
        self.vx, self.vy = fluid2d_advect_vel(
            self.vx,
            self.vy,
            self.dr,
            self.dg,
            self.db,
            dt_s,
            float(cfg.velocity_dissipation) ** dt_s,
            float(cfg.buoyancy * dt_phys),
            device=self.device,
        )
        if cfg.curl_strength > 0.0:
            curl = fluid2d_curl(self.vx, self.vy, device=self.device)
            self.vx, self.vy = fluid2d_vorticity(
                self.vx, self.vy, curl, float(cfg.curl_strength), dt_phys, device=self.device
            )
        div = fluid2d_divergence(self.vx, self.vy, device=self.device)
        p = self.p
        for _ in range(max(1, int(cfg.pressure_iters))):
            p = fluid2d_pressure_jacobi(p, div, device=self.device)
        self.p = p
        self.vx, self.vy = fluid2d_subtract_grad(self.vx, self.vy, self.p, device=self.device)
        self.dr, self.dg, self.db = fluid2d_advect_dye(
            self.dr,
            self.dg,
            self.db,
            self.vx,
            self.vy,
            dt_s,
            float(cfg.dye_dissipation) ** dt_s,
            device=self.device,
        )
