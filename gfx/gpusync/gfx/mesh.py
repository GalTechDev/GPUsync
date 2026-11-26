"""Maillages 3D procéduraux, chargeur Wavefront `.obj` et recettes sérialisables `.scene.json`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Mesh:
    """Maillage indexé : positions `(N, 3)`, normales `(N, 3)`, UV `(N, 2)`, triangles `(T, 3)`."""

    positions: np.ndarray  # float32 (N, 3)
    normals: np.ndarray  # float32 (N, 3)
    uvs: np.ndarray  # float32 (N, 2)
    indices: np.ndarray  # int32 (T, 3)
    recipe: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def num_vertices(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])

    # ── Constructeurs ergonomiques (avec recette sérialisable) ──

    @classmethod
    def plane(cls, size: float = 16.0, subdivisions: int = 16) -> Mesh:
        return make_plane_grid(size=size, subdivisions=subdivisions)

    @classmethod
    def quad(cls, width: float = 2.0, height: float = 1.2) -> Mesh:
        """Rectangle vertical face +Z (idéal pour écrans 3D, miroirs, affiches, panneaux UI)."""
        return make_quad(width=width, height=height)

    @classmethod
    def box(cls, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, sub: int = 2) -> Mesh:
        return make_box(sx=sx, sy=sy, sz=sz, sub=sub)

    @classmethod
    def cube(cls, size: float = 1.0, sub: int = 2) -> Mesh:
        return make_box(sx=size, sy=size, sz=size, sub=sub)

    @classmethod
    def sphere(cls, radius: float = 1.0, rings: int = 24, sectors: int = 36) -> Mesh:
        return make_uv_sphere(radius=radius, rings=rings, sectors=sectors)

    @classmethod
    def torus(
        cls, major_r: float = 1.0, minor_r: float = 0.32, major_seg: int = 44, minor_seg: int = 22
    ) -> Mesh:
        return make_torus(major_r=major_r, minor_r=minor_r, major_seg=major_seg, minor_seg=minor_seg)

    @classmethod
    def cylinder(cls, radius: float = 0.5, height: float = 2.0, sectors: int = 28) -> Mesh:
        return make_cylinder(radius=radius, height=height, sectors=sectors)

    @classmethod
    def crystal(cls, radius: float = 0.7, height: float = 2.0, sides: int = 8) -> Mesh:
        return make_crystal(radius=radius, height=height, sides=sides)

    @classmethod
    def from_obj(cls, path: str | Path, *, normalize_size: float | None = None) -> Mesh:
        return load_obj(path, normalize_size=normalize_size)

    @classmethod
    def from_gltf(cls, path: str | Path, *, normalize_size: float | None = None) -> Mesh:
        """Charge un maillage glTF 2.0 / `.glb` (zéro dépendance externe)."""
        from .gltf import load_gltf_mesh

        return load_gltf_mesh(path, normalize_size=normalize_size)

    # ── Sérialisation légère (.scene.json) ──

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        if self.recipe is not None:
            d = dict(self.recipe)
            if d.get("kind") in ("obj", "gltf") and base_dir is not None and "path" in d:
                try:
                    d["path"] = str(Path(d["path"]).resolve().relative_to(base_dir.resolve()))
                except ValueError:
                    pass
            return d
        return {
            "kind": "custom",
            "positions": self.positions.tolist(),
            "normals": self.normals.tolist(),
            "uvs": self.uvs.tolist(),
            "indices": self.indices.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Mesh:
        kind = data["kind"]
        if kind == "plane":
            return cls.plane(size=float(data.get("size", 16.0)), subdivisions=int(data.get("subdivisions", 16)))
        if kind == "quad":
            return cls.quad(width=float(data.get("width", 2.0)), height=float(data.get("height", 1.2)))
        if kind == "box":
            return cls.box(
                sx=float(data.get("sx", 1.0)),
                sy=float(data.get("sy", 1.0)),
                sz=float(data.get("sz", 1.0)),
                sub=int(data.get("sub", 2)),
            )
        if kind == "sphere":
            return cls.sphere(
                radius=float(data.get("radius", 1.0)),
                rings=int(data.get("rings", 24)),
                sectors=int(data.get("sectors", 36)),
            )
        if kind == "torus":
            return cls.torus(
                major_r=float(data.get("major_r", 1.0)),
                minor_r=float(data.get("minor_r", 0.32)),
                major_seg=int(data.get("major_seg", 44)),
                minor_seg=int(data.get("minor_seg", 22)),
            )
        if kind == "cylinder":
            return cls.cylinder(
                radius=float(data.get("radius", 0.5)),
                height=float(data.get("height", 2.0)),
                sectors=int(data.get("sectors", 28)),
            )
        if kind == "crystal":
            return cls.crystal(
                radius=float(data.get("radius", 0.7)),
                height=float(data.get("height", 2.0)),
                sides=int(data.get("sides", 8)),
            )
        if kind == "obj":
            p = Path(data["path"])
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            ns = data.get("normalize_size")
            return cls.from_obj(p, normalize_size=float(ns) if ns is not None else None)
        if kind == "gltf":
            p = Path(data["path"])
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            ns = data.get("normalize_size")
            return cls.from_gltf(p, normalize_size=float(ns) if ns is not None else None)
        if kind == "custom":
            return cls(
                positions=np.ascontiguousarray(data["positions"], dtype=np.float32),
                normals=np.ascontiguousarray(data["normals"], dtype=np.float32),
                uvs=np.ascontiguousarray(data["uvs"], dtype=np.float32),
                indices=np.ascontiguousarray(data["indices"], dtype=np.int32),
            )
        raise ValueError(f"Type de maillage inconnu dans le fichier scène : {kind!r}")


def make_plane_grid(size: float = 16.0, subdivisions: int = 16) -> Mesh:
    """Plan horizontal Y=0 subdivisé en petits triangles (optimisé pour le rasterizer GPU)."""
    n = subdivisions + 1
    half = size * 0.5
    coords = np.linspace(-half, half, n, dtype=np.float32)
    gx, gz = np.meshgrid(coords, coords, indexing="xy")
    positions = np.stack([gx.ravel(), np.zeros(n * n, dtype=np.float32), gz.ravel()], axis=-1)
    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n * n, 1))
    u = np.linspace(0.0, size * 0.5, n, dtype=np.float32)
    gu, gv = np.meshgrid(u, u, indexing="xy")
    uvs = np.stack([gu.ravel(), gv.ravel()], axis=-1)

    tris = []
    for iz in range(subdivisions):
        for ix in range(subdivisions):
            a = iz * n + ix
            b = a + 1
            c = a + n
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return Mesh(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        indices=np.ascontiguousarray(tris, dtype=np.int32),
        recipe={"kind": "plane", "size": float(size), "subdivisions": int(subdivisions)},
    )


def make_quad(width: float = 2.0, height: float = 1.2) -> Mesh:
    """Quad vertical face +Z avec UV `[0, 1] × [0, 1]` (v=0 en bas, v=1 en haut)."""
    hw, hh = width * 0.5, height * 0.5
    positions = np.array(
        [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]],
        dtype=np.float32,
    )
    normals = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Mesh(
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=indices,
        recipe={"kind": "quad", "width": float(width), "height": float(height)},
    )


def make_box(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, sub: int = 2) -> Mesh:
    """Boîte à normales nettes (faces subdivisées pour garder des triangles compacts)."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    faces = [
        ((0, 0, 1), (hx, 0, 0), (0, hy, 0), (0, 0, hz)),
        ((0, 0, -1), (-hx, 0, 0), (0, hy, 0), (0, 0, -hz)),
        ((1, 0, 0), (0, 0, -hz), (0, hy, 0), (hx, 0, 0)),
        ((-1, 0, 0), (0, 0, hz), (0, hy, 0), (-hx, 0, 0)),
        ((0, 1, 0), (hx, 0, 0), (0, 0, -hz), (0, hy, 0)),
        ((0, -1, 0), (hx, 0, 0), (0, 0, hz), (0, -hy, 0)),
    ]
    pos_list, norm_list, uv_list, idx_list = [], [], [], []
    base = 0
    for normal, u_ax, v_ax, center in faces:
        n_v = np.array(normal, dtype=np.float32)
        u_v = np.array(u_ax, dtype=np.float32)
        v_v = np.array(v_ax, dtype=np.float32)
        c_v = np.array(center, dtype=np.float32)
        for iy in range(sub + 1):
            ty = -1.0 + 2.0 * iy / sub
            for ix in range(sub + 1):
                tx = -1.0 + 2.0 * ix / sub
                pos_list.append(c_v + tx * u_v + ty * v_v)
                norm_list.append(n_v)
                uv_list.append((ix / sub, iy / sub))
        row = sub + 1
        for iy in range(sub):
            for ix in range(sub):
                a = base + iy * row + ix
                b = a + 1
                c = a + row
                d = c + 1
                idx_list.append((a, b, d))
                idx_list.append((a, d, c))
        base += row * row
    return Mesh(
        positions=np.ascontiguousarray(pos_list, dtype=np.float32),
        normals=np.ascontiguousarray(norm_list, dtype=np.float32),
        uvs=np.ascontiguousarray(uv_list, dtype=np.float32),
        indices=np.ascontiguousarray(idx_list, dtype=np.int32),
        recipe={"kind": "box", "sx": float(sx), "sy": float(sy), "sz": float(sz), "sub": int(sub)},
    )


def make_uv_sphere(radius: float = 1.0, rings: int = 24, sectors: int = 36) -> Mesh:
    """Sphère UV lisse."""
    pos_list, norm_list, uv_list, idx_list = [], [], [], []
    for ir in range(rings + 1):
        v = ir / rings
        phi = v * math.pi
        sp, cp = math.sin(phi), math.cos(phi)
        for isec in range(sectors + 1):
            u = isec / sectors
            theta = u * 2.0 * math.pi
            st, ct = math.sin(theta), math.cos(theta)
            nx, ny, nz = sp * ct, cp, sp * st
            pos_list.append((radius * nx, radius * ny, radius * nz))
            norm_list.append((nx, ny, nz))
            uv_list.append((u * 2.0, v))
    row = sectors + 1
    for ir in range(rings):
        for isec in range(sectors):
            a = ir * row + isec
            b = a + 1
            c = a + row
            d = c + 1
            if ir > 0:
                idx_list.append((a, b, c))
            if ir < rings - 1:
                idx_list.append((b, d, c))
    return Mesh(
        positions=np.ascontiguousarray(pos_list, dtype=np.float32),
        normals=np.ascontiguousarray(norm_list, dtype=np.float32),
        uvs=np.ascontiguousarray(uv_list, dtype=np.float32),
        indices=np.ascontiguousarray(idx_list, dtype=np.int32),
        recipe={"kind": "sphere", "radius": float(radius), "rings": int(rings), "sectors": int(sectors)},
    )


def make_torus(
    major_r: float = 1.0, minor_r: float = 0.32, major_seg: int = 44, minor_seg: int = 22
) -> Mesh:
    """Tore lisse centré à l'origine, dans le plan XZ."""
    pos_list, norm_list, uv_list, idx_list = [], [], [], []
    for i in range(major_seg + 1):
        u = i / major_seg
        a = u * 2.0 * math.pi
        ca, sa = math.cos(a), math.sin(a)
        for j in range(minor_seg + 1):
            v = j / minor_seg
            b = v * 2.0 * math.pi
            cb, sb = math.cos(b), math.sin(b)
            r = major_r + minor_r * cb
            x, y, z = r * ca, minor_r * sb, r * sa
            nx, ny, nz = cb * ca, sb, cb * sa
            pos_list.append((x, y, z))
            norm_list.append((nx, ny, nz))
            uv_list.append((u * 4.0, v * 2.0))
    row = minor_seg + 1
    for i in range(major_seg):
        for j in range(minor_seg):
            a = i * row + j
            b = a + 1
            c = a + row
            d = c + 1
            idx_list.append((a, b, c))
            idx_list.append((b, d, c))
    return Mesh(
        positions=np.ascontiguousarray(pos_list, dtype=np.float32),
        normals=np.ascontiguousarray(norm_list, dtype=np.float32),
        uvs=np.ascontiguousarray(uv_list, dtype=np.float32),
        indices=np.ascontiguousarray(idx_list, dtype=np.int32),
        recipe={
            "kind": "torus",
            "major_r": float(major_r),
            "minor_r": float(minor_r),
            "major_seg": int(major_seg),
            "minor_seg": int(minor_seg),
        },
    )


def make_cylinder(radius: float = 0.5, height: float = 2.0, sectors: int = 28) -> Mesh:
    """Cylindre fermé avec normales propres sur les flancs et les couvercles."""
    pos_list, norm_list, uv_list, idx_list = [], [], [], []
    hy = height * 0.5
    for iy, y in enumerate((-hy, hy)):
        for i in range(sectors + 1):
            u = i / sectors
            a = u * 2.0 * math.pi
            ca, sa = math.cos(a), math.sin(a)
            pos_list.append((radius * ca, y, radius * sa))
            norm_list.append((ca, 0.0, sa))
            uv_list.append((u * 2.0, float(iy)))
    row = sectors + 1
    for i in range(sectors):
        a = i
        b = i + 1
        c = row + i
        d = row + i + 1
        idx_list.append((a, c, b))
        idx_list.append((b, c, d))

    for sign, y in ((1.0, hy), (-1.0, -hy)):
        center_idx = len(pos_list)
        pos_list.append((0.0, y, 0.0))
        norm_list.append((0.0, sign, 0.0))
        uv_list.append((0.5, 0.5))
        ring_start = len(pos_list)
        for i in range(sectors + 1):
            a = (i / sectors) * 2.0 * math.pi
            ca, sa = math.cos(a), math.sin(a)
            pos_list.append((radius * ca, y, radius * sa))
            norm_list.append((0.0, sign, 0.0))
            uv_list.append((0.5 + 0.5 * ca, 0.5 + 0.5 * sa))
        for i in range(sectors):
            if sign > 0:
                idx_list.append((center_idx, ring_start + i + 1, ring_start + i))
            else:
                idx_list.append((center_idx, ring_start + i, ring_start + i + 1))

    return Mesh(
        positions=np.ascontiguousarray(pos_list, dtype=np.float32),
        normals=np.ascontiguousarray(norm_list, dtype=np.float32),
        uvs=np.ascontiguousarray(uv_list, dtype=np.float32),
        indices=np.ascontiguousarray(idx_list, dtype=np.int32),
        recipe={"kind": "cylinder", "radius": float(radius), "height": float(height), "sectors": int(sectors)},
    )


def make_crystal(radius: float = 0.7, height: float = 2.0, sides: int = 8) -> Mesh:
    """Cristal facetté (normales plates par face pour des reflets spéculaires francs)."""
    top = np.array([0.0, height * 0.5, 0.0], dtype=np.float32)
    bot = np.array([0.0, -height * 0.5, 0.0], dtype=np.float32)
    ring_hi = []
    ring_lo = []
    for i in range(sides):
        a = (i / sides) * 2.0 * math.pi
        ring_hi.append(np.array([radius * math.cos(a), height * 0.16, radius * math.sin(a)], dtype=np.float32))
        ring_lo.append(
            np.array(
                [radius * 0.85 * math.cos(a + 0.2), -height * 0.18, radius * 0.85 * math.sin(a + 0.2)],
                dtype=np.float32,
            )
        )

    pos_list, norm_list, uv_list, idx_list = [], [], [], []

    def add_tri(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> None:
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        n = n / ln if ln > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base = len(pos_list)
        pos_list.extend([p0, p1, p2])
        norm_list.extend([n, n, n])
        uv_list.extend([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        idx_list.append((base, base + 1, base + 2))

    for i in range(sides):
        j = (i + 1) % sides
        add_tri(top, ring_hi[j], ring_hi[i])
        add_tri(ring_hi[i], ring_hi[j], ring_lo[i])
        add_tri(ring_hi[j], ring_lo[j], ring_lo[i])
        add_tri(ring_lo[i], ring_lo[j], bot)

    return Mesh(
        positions=np.ascontiguousarray(pos_list, dtype=np.float32),
        normals=np.ascontiguousarray(norm_list, dtype=np.float32),
        uvs=np.ascontiguousarray(uv_list, dtype=np.float32),
        indices=np.ascontiguousarray(idx_list, dtype=np.int32),
        recipe={"kind": "crystal", "radius": float(radius), "height": float(height), "sides": int(sides)},
    )


def load_obj(path: str | Path, *, normalize_size: float | None = None) -> Mesh:
    """Charge un maillage Wavefront `.obj` (triangles ou quads, normales calculées si absentes)."""
    p = Path(path)
    raw_v: list[tuple[float, float, float]] = []
    raw_vn: list[tuple[float, float, float]] = []
    raw_vt: list[tuple[float, float]] = []
    pos_list, norm_list, uv_list, idx_list = [], [], [], []
    vert_map: dict[tuple[int, int, int], int] = {}

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            raw_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag == "vn" and len(parts) >= 4:
            raw_vn.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag == "vt" and len(parts) >= 3:
            raw_vt.append((float(parts[1]), float(parts[2])))
        elif tag == "f" and len(parts) >= 4:
            face = []
            for tok in parts[1:]:
                subs = tok.split("/")
                vi = int(subs[0]) - 1 if subs[0] else -1
                ti = int(subs[1]) - 1 if len(subs) > 1 and subs[1] else -1
                ni = int(subs[2]) - 1 if len(subs) > 2 and subs[2] else -1
                key = (vi, ti, ni)
                if key not in vert_map:
                    vert_map[key] = len(pos_list)
                    pos_list.append(raw_v[vi])
                    uv_list.append(raw_vt[ti] if 0 <= ti < len(raw_vt) else (0.0, 0.0))
                    norm_list.append(raw_vn[ni] if 0 <= ni < len(raw_vn) else (0.0, 0.0, 0.0))
                face.append(vert_map[key])
            for k in range(1, len(face) - 1):
                idx_list.append((face[0], face[k], face[k + 1]))

    positions = np.ascontiguousarray(pos_list, dtype=np.float32)
    normals = np.ascontiguousarray(norm_list, dtype=np.float32)
    uvs = np.ascontiguousarray(uv_list, dtype=np.float32)
    indices = np.ascontiguousarray(idx_list, dtype=np.int32)

    if normalize_size is not None and len(positions) > 0:
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        span = float(np.max(hi - lo))
        if span > 1e-6:
            positions = np.ascontiguousarray((positions - 0.5 * (lo + hi)) * (normalize_size / span), dtype=np.float32)

    if not raw_vn and len(indices) > 0:
        normals[:] = 0.0
        p0 = positions[indices[:, 0]]
        p1 = positions[indices[:, 1]]
        p2 = positions[indices[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        for c in range(3):
            np.add.at(normals, indices[:, c], fn)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lens > 1e-8, normals / np.maximum(lens, 1e-8), np.array([0.0, 1.0, 0.0], np.float32))

    recipe: dict[str, Any] = {"kind": "obj", "path": str(p)}
    if normalize_size is not None:
        recipe["normalize_size"] = float(normalize_size)
    return Mesh(positions=positions, normals=normals, uvs=uvs, indices=indices, recipe=recipe)
