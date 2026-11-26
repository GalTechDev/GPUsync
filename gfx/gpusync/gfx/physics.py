"""Physique modulaire `gpusync.gfx` :
- Étage 1 : `PhysicsWorld` léger branché sur les `Object3D` (`SphereCollider`, `BoxCollider`,
  `PlaneCollider`, impulsions, rebonds, triggers, `raycast`).
- Étage 2 : `GpuSpherePhysics` — solveur de collisions de sphères 3D 100 % GPU (`@gs.kernel`)
  écrivant directement dans `InstancedMesh3D.gpu_trs` sans aucune copie CPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import gpusync as gs

from .kernels.k3d import upload_f32
from .math3d import normalize
from .node import Node


# ───────────────────────── Colliders & Raycast ─────────────────────────


@dataclass
class SphereCollider:
    radius: float = 0.5


@dataclass
class BoxCollider:
    half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class PlaneCollider:
    y: float = 0.0


Collider = SphereCollider | BoxCollider | PlaneCollider


@dataclass
class RayHit:
    body: RigidBody
    distance: float
    point: np.ndarray
    normal: np.ndarray


@dataclass
class RigidBody:
    """Corps rigide léger attaché à un `Node` / `Object3D`."""

    node: Node
    collider: Collider
    mass: float = 1.0
    restitution: float = 0.65
    damping: float = 0.995
    static: bool = False
    is_trigger: bool = False
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    on_collision: Callable[[RigidBody, RigidBody], Any] | None = None

    def __post_init__(self) -> None:
        self.vel = np.asarray(self.vel, dtype=np.float32)

    @property
    def inv_mass(self) -> float:
        return 0.0 if (self.static or self.mass <= 0.0) else 1.0 / self.mass


class PhysicsWorld:
    """Monde physique 3D léger (gravité, collisions sphère/boîte/plan, triggers, raycasts)."""

    def __init__(self, gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)) -> None:
        self.gravity = np.asarray(gravity, dtype=np.float32)
        self.bodies: list[RigidBody] = []

    def add(self, body: RigidBody) -> RigidBody:
        self.bodies.append(body)
        return body

    def remove(self, body: RigidBody) -> None:
        if body in self.bodies:
            self.bodies.remove(body)

    def add_ground(self, node: Node, y: float = 0.0, restitution: float = 0.6) -> RigidBody:
        return self.add(RigidBody(node, PlaneCollider(y=y), mass=0.0, restitution=restitution, static=True))

    def step(self, dt: float, *, substeps: int = 2) -> None:
        if dt <= 0.0:
            return
        sub_dt = min(dt, 0.05) / max(1, substeps)
        for _ in range(max(1, substeps)):
            self._step_once(sub_dt)

    def _step_once(self, dt: float) -> None:
        # 1. Intégration semi-implicite d'Euler
        for b in self.bodies:
            if b.static:
                continue
            b.vel = (b.vel + self.gravity * dt) * (b.damping ** (dt * 60.0))
            p = np.asarray(b.node.pos, dtype=np.float32) + b.vel * dt
            b.node.pos = (float(p[0]), float(p[1]), float(p[2]))

        # 2. Résolution des paires
        n = len(self.bodies)
        for i in range(n):
            a = self.bodies[i]
            for j in range(i + 1, n):
                b = self.bodies[j]
                if a.static and b.static:
                    continue
                self._resolve_pair(a, b)

    def _notify(self, a: RigidBody, b: RigidBody) -> None:
        if a.on_collision is not None:
            a.on_collision(a, b)
        if b.on_collision is not None:
            b.on_collision(b, a)

    def _apply_impulse(self, a: RigidBody, b: RigidBody, normal: np.ndarray, pen: float) -> None:
        if a.is_trigger or b.is_trigger:
            self._notify(a, b)
            return
        inv_sum = a.inv_mass + b.inv_mass
        if inv_sum <= 0.0:
            return
        # Correction de position
        corr = normal * (pen / inv_sum)
        if not a.static:
            pa = np.asarray(a.node.pos, dtype=np.float32) - corr * a.inv_mass
            a.node.pos = (float(pa[0]), float(pa[1]), float(pa[2]))
        if not b.static:
            pb = np.asarray(b.node.pos, dtype=np.float32) + corr * b.inv_mass
            b.node.pos = (float(pb[0]), float(pb[1]), float(pb[2]))
        # Impulsion de vitesse le long de la normale (de A vers B)
        rv = b.vel - a.vel
        vn = float(np.dot(rv, normal))
        if vn < 0.0:
            e = min(a.restitution, b.restitution)
            j_imp = -(1.0 + e) * vn / inv_sum
            imp = normal * j_imp
            if not a.static:
                a.vel = a.vel - imp * a.inv_mass
            if not b.static:
                b.vel = b.vel + imp * b.inv_mass
        self._notify(a, b)

    def _resolve_pair(self, a: RigidBody, b: RigidBody) -> None:
        ca, cb = a.collider, b.collider
        # Sphere vs Plane
        if isinstance(ca, SphereCollider) and isinstance(cb, PlaneCollider):
            pen = (cb.y + ca.radius) - a.node.y
            if pen > 0.0:
                self._apply_impulse(a, b, np.array([0.0, -1.0, 0.0], dtype=np.float32), pen)
            return
        if isinstance(ca, PlaneCollider) and isinstance(cb, SphereCollider):
            pen = (ca.y + cb.radius) - b.node.y
            if pen > 0.0:
                self._apply_impulse(a, b, np.array([0.0, 1.0, 0.0], dtype=np.float32), pen)
            return
        # Box vs Plane
        if isinstance(ca, BoxCollider) and isinstance(cb, PlaneCollider):
            pen = (cb.y + ca.half_extents[1]) - a.node.y
            if pen > 0.0:
                self._apply_impulse(a, b, np.array([0.0, -1.0, 0.0], dtype=np.float32), pen)
            return
        if isinstance(ca, PlaneCollider) and isinstance(cb, BoxCollider):
            pen = (ca.y + cb.half_extents[1]) - b.node.y
            if pen > 0.0:
                self._apply_impulse(a, b, np.array([0.0, 1.0, 0.0], dtype=np.float32), pen)
            return
        # Sphere vs Sphere
        if isinstance(ca, SphereCollider) and isinstance(cb, SphereCollider):
            d = np.asarray(b.node.pos, dtype=np.float32) - np.asarray(a.node.pos, dtype=np.float32)
            dist = float(np.linalg.norm(d))
            rsum = ca.radius + cb.radius
            if dist < rsum:
                n = d / dist if dist > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                self._apply_impulse(a, b, n, rsum - dist)
            return
        # Sphere vs Box (AABB)
        if isinstance(ca, SphereCollider) and isinstance(cb, BoxCollider):
            pa = np.asarray(a.node.pos, dtype=np.float32)
            pb = np.asarray(b.node.pos, dtype=np.float32)
            he = np.asarray(cb.half_extents, dtype=np.float32)
            closest = np.clip(pa, pb - he, pb + he)
            d = closest - pa
            dist = float(np.linalg.norm(d))
            if dist < ca.radius:
                n = d / dist if dist > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                self._apply_impulse(a, b, n, ca.radius - dist)
            return
        if isinstance(ca, BoxCollider) and isinstance(cb, SphereCollider):
            self._resolve_pair(b, a)
            return
        # Box vs Box (AABB)
        if isinstance(ca, BoxCollider) and isinstance(cb, BoxCollider):
            pa = np.asarray(a.node.pos, dtype=np.float32)
            pb = np.asarray(b.node.pos, dtype=np.float32)
            diff = pb - pa
            ov = np.asarray(ca.half_extents, dtype=np.float32) + np.asarray(cb.half_extents, dtype=np.float32) - np.abs(diff)
            if np.all(ov > 0.0):
                axis = int(np.argmin(ov))
                n = np.zeros(3, dtype=np.float32)
                n[axis] = 1.0 if diff[axis] >= 0.0 else -1.0
                self._apply_impulse(a, b, n, float(ov[axis]))

    def raycast(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        max_dist: float = 100.0,
    ) -> RayHit | None:
        """Lancer de rayon analytique contre tous les colliders du monde physique."""
        ro = np.asarray(origin, dtype=np.float32)
        rd = normalize(direction)
        best: RayHit | None = None
        best_t = float(max_dist)

        for b in self.bodies:
            c = b.collider
            if isinstance(c, PlaneCollider):
                if abs(float(rd[1])) > 1e-6:
                    t = (c.y - float(ro[1])) / float(rd[1])
                    if 1e-4 < t < best_t:
                        best_t = t
                        n = np.array([0.0, 1.0 if rd[1] < 0 else -1.0, 0.0], dtype=np.float32)
                        best = RayHit(b, t, ro + rd * t, n)
            elif isinstance(c, SphereCollider):
                oc = ro - np.asarray(b.node.pos, dtype=np.float32)
                b_term = float(np.dot(oc, rd))
                c_term = float(np.dot(oc, oc)) - c.radius * c.radius
                disc = b_term * b_term - c_term
                if disc >= 0.0:
                    t = -b_term - math.sqrt(disc)
                    if 1e-4 < t < best_t:
                        best_t = t
                        pt = ro + rd * t
                        best = RayHit(b, t, pt, normalize(pt - np.asarray(b.node.pos, dtype=np.float32)))
            elif isinstance(c, BoxCollider):
                pb = np.asarray(b.node.pos, dtype=np.float32)
                he = np.asarray(c.half_extents, dtype=np.float32)
                bmin, bmax = pb - he, pb + he
                tmin, tmax = 1e-4, best_t
                hit_axis = 0
                hit_sign = 1.0
                hit = True
                for ax in range(3):
                    if abs(float(rd[ax])) < 1e-7:
                        if ro[ax] < bmin[ax] or ro[ax] > bmax[ax]:
                            hit = False
                            break
                    else:
                        inv_d = 1.0 / float(rd[ax])
                        t1 = (float(bmin[ax]) - float(ro[ax])) * inv_d
                        t2 = (float(bmax[ax]) - float(ro[ax])) * inv_d
                        s = -1.0 if t1 < t2 else 1.0
                        if t1 > t2:
                            t1, t2 = t2, t1
                        if t1 > tmin:
                            tmin = t1
                            hit_axis = ax
                            hit_sign = s
                        tmax = min(tmax, t2)
                        if tmin > tmax:
                            hit = False
                            break
                if hit and tmin < best_t:
                    best_t = tmin
                    n = np.zeros(3, dtype=np.float32)
                    n[hit_axis] = hit_sign
                    best = RayHit(b, tmin, ro + rd * tmin, n)
        return best


# ───────────────────────── Étage 2 : Solveur de sphères 100 % GPU ─────────────────────────


@gs.kernel
def _step_gpu_spheres_k(
    pos: gs.f32["n", "c4"],  # [x, y, z, radius]
    vel: gs.f32["n", "c4"],  # [vx, vy, vz, spin]
    dt: gs.f32,
    gravity: gs.f32,
    box_half: gs.f32,
    floor_y: gs.f32,
    restitution: gs.f32,
) -> (gs.f32["n", "c4"], gs.f32["n", "c4"]):
    """Simule N sphères en collision élastique 100 % GPU (reste en VRAM)."""
    i, c = gs.index()
    n = pos.shape[0]

    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    rad = pos[i, 3]

    vx = vel[i, 0]
    vy = vel[i, 1] - gravity * dt
    vz = vel[i, 2]
    spin = vel[i, 3]

    # Interactions sphère-sphère (répulsion élastique douce + amortissement)
    for j in range(n):
        if j != i:
            dx = px - pos[j, 0]
            dy = py - pos[j, 1]
            dz = pz - pos[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            rsum = rad + pos[j, 3]
            if d2 < rsum * rsum and d2 > 1e-8:
                dist = gs.sqrt(d2)
                inv_d = 1.0 / dist
                nx = dx * inv_d
                ny = dy * inv_d
                nz = dz * inv_d
                overlap = rsum - dist
                # Correction de position + impulsion relative
                px += nx * overlap * 0.45
                py += ny * overlap * 0.45
                pz += nz * overlap * 0.45
                rvn = (vx - vel[j, 0]) * nx + (vy - vel[j, 1]) * ny + (vz - vel[j, 2]) * nz
                if rvn < 0.0:
                    imp = -(1.0 + restitution) * rvn * 0.5
                    vx += nx * imp
                    vy += ny * imp
                    vz += nz * imp

    # Intégration + rebonds sur le sol et les parois de l'arène
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    if py - rad < floor_y:
        py = floor_y + rad
        if vy < 0.0:
            vy = -vy * restitution
        vx = vx * 0.992
        vz = vz * 0.992
    if px - rad < -box_half:
        px = -box_half + rad
        if vx < 0.0:
            vx = -vx * restitution
    elif px + rad > box_half:
        px = box_half - rad
        if vx > 0.0:
            vx = -vx * restitution
    if pz - rad < -box_half:
        pz = -box_half + rad
        if vz < 0.0:
            vz = -vz * restitution
    elif pz + rad > box_half:
        pz = box_half - rad
        if vz > 0.0:
            vz = -vz * restitution

    out_p = px
    out_v = vx
    if c == 1:
        out_p = py
        out_v = vy
    elif c == 2:
        out_p = pz
        out_v = vz
    elif c == 3:
        out_p = rad
        out_v = spin

    return out_p, out_v


@gs.kernel
def _pos_to_trs_k(pos: gs.f32["n", "c4"], trs_tpl: gs.f32["n", "c8"], t: gs.f32) -> gs.f32["n", "c8"]:
    """Convertit `pos[n, 4]` (`x, y, z, radius`) en `trs[n, 8]` directement en VRAM pour `InstancedMesh3D`."""
    i, c = gs.index()
    val = pos[i, 0] + trs_tpl[i, c] * 0.0
    if c == 1:
        val = pos[i, 1]
    elif c == 2:
        val = pos[i, 2]
    elif c == 3:
        val = t * 0.5 + float(i) * 0.13
    elif c == 4:
        val = t * 0.7 + float(i) * 0.21
    elif c == 5:
        val = 0.0
    elif c == 6:
        val = pos[i, 3]
    elif c == 7:
        val = 1.0
    return val


class GpuSpherePhysics:
    """Solveur de collisions 3D 100 % GPU (zéro copie CPU) connecté à `InstancedMesh3D.gpu_trs`."""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray | float = 0.3,
        *,
        gravity: float = 9.81,
        box_half: float = 5.0,
        floor_y: float = 0.0,
        restitution: float = 0.78,
        device: str | None = None,
    ) -> None:
        n = int(positions.shape[0])
        p4 = np.zeros((n, 4), dtype=np.float32)
        p4[:, :3] = positions
        p4[:, 3] = radii
        v4 = np.zeros((n, 4), dtype=np.float32)
        v4[:, :3] = velocities

        self.gravity = float(gravity)
        self.box_half = float(box_half)
        self.floor_y = float(floor_y)
        self.restitution = float(restitution)
        self.device = device
        self.t = 0.0

        self.pos_gpu = upload_f32(p4, device=device)
        self.vel_gpu = upload_f32(v4, device=device)
        self._trs_tpl = upload_f32(np.zeros((n, 8), dtype=np.float32), device=device)
        self.trs_gpu = _pos_to_trs_k(self.pos_gpu, self._trs_tpl, 0.0, device=device)

    def step(self, dt: float, *, substeps: int = 2) -> gs.Array:
        """Avance la simulation sur GPU et renvoie le `gs.Array` `trs` `(N, 8)` prêt pour `InstancedMesh3D`."""
        sub_dt = min(float(dt), 0.04) / max(1, substeps)
        with gs.batch(self.device):
            for _ in range(max(1, substeps)):
                self.pos_gpu, self.vel_gpu = _step_gpu_spheres_k(
                    self.pos_gpu,
                    self.vel_gpu,
                    sub_dt,
                    self.gravity,
                    self.box_half,
                    self.floor_y,
                    self.restitution,
                    device=self.device,
                )
            self.t += dt
            self.trs_gpu = _pos_to_trs_k(self.pos_gpu, self._trs_tpl, float(self.t), device=self.device)
        return self.trs_gpu
