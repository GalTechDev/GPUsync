"""Mathématiques 3D pour `gpusync.gfx` (vecteurs, matrices 4×4, transforms hiérarchiques en batch)."""

from __future__ import annotations

import math

import numpy as np


def normalize(v: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.shape == (3,):  # cas courant, plusieurs fois par image : sans np.linalg
        x, y, z = arr.tolist()
        n = math.sqrt(x * x + y * y + z * z)
    else:
        n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-8 else arr


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produit vectoriel de deux vecteurs (3,) (bien plus rapide que np.cross)."""
    ax, ay, az = (float(x) for x in a)
    bx, by, bz = (float(x) for x in b)
    return np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dtype=np.float32)


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=np.float32)


def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def mat4_translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def mat4_scale(sx: float, sy: float | None = None, sz: float | None = None) -> np.ndarray:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotate_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def mat4_rotate_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def mat4_rotate_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def mat4_trs(
    pos: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    scale: float | tuple[float, float, float] | np.ndarray = 1.0,
) -> np.ndarray:
    """Matrice locale `T @ Ry @ Rx @ Rz @ S` (float32)."""
    if isinstance(scale, (int, float)):
        sx = sy = sz = float(scale)
    else:
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    rx, ry, rz = float(rot[0]), float(rot[1]), float(rot[2])
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    snx, sny, snz = math.sin(rx), math.sin(ry), math.sin(rz)

    r00 = cy * cz + sny * snx * snz
    r01 = -cy * snz + sny * snx * cz
    r02 = sny * cx
    r10 = cx * snz
    r11 = cx * cz
    r12 = -snx
    r20 = -sny * cz + cy * snx * snz
    r21 = sny * snz + cy * snx * cz
    r22 = cy * cx

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0], m[0, 1], m[0, 2], m[0, 3] = r00 * sx, r01 * sy, r02 * sz, float(pos[0])
    m[1, 0], m[1, 1], m[1, 2], m[1, 3] = r10 * sx, r11 * sy, r12 * sz, float(pos[1])
    m[2, 0], m[2, 1], m[2, 2], m[2, 3] = r20 * sx, r21 * sy, r22 * sz, float(pos[2])
    m[3, 3] = 1.0
    return m


def batch_trs(pos: np.ndarray, rot: np.ndarray, scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule `(models, normals)` en batch NumPy `(N, 4, 4)` float32 sans appel `linalg.inv`."""
    n = pos.shape[0]
    cx, cy, cz = np.cos(rot[:, 0]), np.cos(rot[:, 1]), np.cos(rot[:, 2])
    sx, sy, sz = np.sin(rot[:, 0]), np.sin(rot[:, 1]), np.sin(rot[:, 2])

    r00 = cy * cz + sy * sx * sz
    r01 = -cy * sz + sy * sx * cz
    r02 = sy * cx
    r10 = cx * sz
    r11 = cx * cz
    r12 = -sx
    r20 = -sy * cz + cy * sx * sz
    r21 = sy * sz + cy * sx * cz
    r22 = cy * cx

    m4 = np.zeros((n, 4, 4), dtype=np.float32)
    m4[:, 0, 0] = r00 * scl[:, 0]
    m4[:, 0, 1] = r01 * scl[:, 1]
    m4[:, 0, 2] = r02 * scl[:, 2]
    m4[:, 0, 3] = pos[:, 0]
    m4[:, 1, 0] = r10 * scl[:, 0]
    m4[:, 1, 1] = r11 * scl[:, 1]
    m4[:, 1, 2] = r12 * scl[:, 2]
    m4[:, 1, 3] = pos[:, 1]
    m4[:, 2, 0] = r20 * scl[:, 0]
    m4[:, 2, 1] = r21 * scl[:, 1]
    m4[:, 2, 2] = r22 * scl[:, 2]
    m4[:, 2, 3] = pos[:, 2]
    m4[:, 3, 3] = 1.0

    inv_s = 1.0 / np.maximum(np.abs(scl), 1e-8)
    n4 = np.zeros((n, 4, 4), dtype=np.float32)
    n4[:, 0, 0] = r00 * inv_s[:, 0]
    n4[:, 0, 1] = r01 * inv_s[:, 1]
    n4[:, 0, 2] = r02 * inv_s[:, 2]
    n4[:, 1, 0] = r10 * inv_s[:, 0]
    n4[:, 1, 1] = r11 * inv_s[:, 1]
    n4[:, 1, 2] = r12 * inv_s[:, 2]
    n4[:, 2, 0] = r20 * inv_s[:, 0]
    n4[:, 2, 1] = r21 * inv_s[:, 1]
    n4[:, 2, 2] = r22 * inv_s[:, 2]
    n4[:, 3, 3] = 1.0
    return m4, n4


def batch_normal_from_models(m4: np.ndarray) -> np.ndarray:
    """Inverse transposée du bloc 3×3 en batch (utilisé uniquement si hiérarchie parent/enfant active)."""
    n = m4.shape[0]
    n4 = np.zeros((n, 4, 4), dtype=np.float32)
    n4[:, 3, 3] = 1.0
    a = m4[:, :3, :3]
    # Cofacteur 3×3 vectorisé (sans np.linalg.inv, robuste même pour des matrices d'échelle nulle)
    c00 = a[:, 1, 1] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 1]
    c01 = a[:, 1, 2] * a[:, 2, 0] - a[:, 1, 0] * a[:, 2, 2]
    c02 = a[:, 1, 0] * a[:, 2, 1] - a[:, 1, 1] * a[:, 2, 0]
    c10 = a[:, 0, 2] * a[:, 2, 1] - a[:, 0, 1] * a[:, 2, 2]
    c11 = a[:, 0, 0] * a[:, 2, 2] - a[:, 0, 2] * a[:, 2, 0]
    c12 = a[:, 0, 1] * a[:, 2, 0] - a[:, 0, 0] * a[:, 2, 1]
    c20 = a[:, 0, 1] * a[:, 1, 2] - a[:, 0, 2] * a[:, 1, 1]
    c21 = a[:, 0, 2] * a[:, 1, 0] - a[:, 0, 0] * a[:, 1, 2]
    c22 = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    det = a[:, 0, 0] * c00 + a[:, 0, 1] * c01 + a[:, 0, 2] * c02
    inv_det = np.where(np.abs(det) > 1e-10, 1.0 / np.where(np.abs(det) > 1e-10, det, 1.0), 0.0)
    n4[:, 0, 0], n4[:, 0, 1], n4[:, 0, 2] = c00 * inv_det, c01 * inv_det, c02 * inv_det
    n4[:, 1, 0], n4[:, 1, 1], n4[:, 1, 2] = c10 * inv_det, c11 * inv_det, c12 * inv_det
    n4[:, 2, 0], n4[:, 2, 1], n4[:, 2, 2] = c20 * inv_det, c21 * inv_det, c22 * inv_det
    return n4


def mat4_look_at(
    eye: tuple[float, float, float] | np.ndarray,
    target: tuple[float, float, float] | np.ndarray,
    up: tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Matrice vue main droite : caméra regarde vers -Z."""
    eye_v = np.asarray(eye, dtype=np.float32)
    target_v = np.asarray(target, dtype=np.float32)
    up_v = np.asarray(up, dtype=np.float32)

    f = normalize(target_v - eye_v)
    side = cross3(f, up_v)
    if math.hypot(*side.tolist()) < 1e-5:
        side = cross3(f, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    r = normalize(side)
    u = cross3(r, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(r, eye_v))
    m[1, 3] = -float(np.dot(u, eye_v))
    m[2, 3] = float(np.dot(f, eye_v))
    return m


def mat4_perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Projection perspective (NDC x,y dans [-1, 1], profondeur linéaire gérée côté shader)."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-4)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def mat4_ortho(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Projection orthographique (NDC x,y dans [-1, 1], z dans [0, 1])."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -1.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -near / (far - near)
    m[3, 3] = 1.0
    return m
