"""Pure-Python glTF 2.0 / `.glb` loader (no external dependency, Pillow optional for textures).

Supports:
- binary `.glb` files (magic `glTF` v2) and JSON `.gltf` (external `.bin` buffers or `data:...;base64,...`),
- `POSITION`, `NORMAL` (computed if missing), `TEXCOORD_0` attributes, `u8`/`u16`/`u32` indices,
- `pbrMetallicRoughness` materials (`baseColorFactor`, `metallicFactor`, `roughnessFactor`, `emissiveFactor`,
  embedded `baseColorTexture`),
- node hierarchy (`translation`, quaternion `rotation`, `scale`, `matrix`, `children`).
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .material import Material
from .mesh import Mesh
from .node import Node, Object3D
from .texture import Texture

_COMPONENT_DTYPES: dict[int, np.dtype] = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}

_TYPE_COUNTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _parse_gltf_container(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    raw = path.read_bytes()
    if len(raw) >= 12 and raw[:4] == b"glTF":
        _, version, _ = struct.unpack_from("<III", raw, 0)
        if version != 2:
            raise ValueError(f"Unsupported glTF version: {version} (only v2 is supported)")
        pos = 12
        gltf_json: dict[str, Any] | None = None
        bin_chunk: bytes = b""
        while pos + 8 <= len(raw):
            chunk_len, chunk_type = struct.unpack_from("<II", raw, pos)
            pos += 8
            data = raw[pos : pos + chunk_len]
            pos += chunk_len
            if chunk_type == 0x4E4F534A:  # JSON
                gltf_json = json.loads(data.decode("utf-8"))
            elif chunk_type == 0x004E4942:  # BIN
                bin_chunk = data
        if gltf_json is None:
            raise ValueError(f"Invalid GLB file (JSON chunk missing): {path}")
        buffers = [bin_chunk]
        for bi, bdesc in enumerate(gltf_json.get("buffers", [])):
            if bi == 0 and "uri" not in bdesc:
                continue
            buffers.append(_load_buffer_uri(bdesc.get("uri", ""), path.parent))
        return gltf_json, buffers

    gltf_json = json.loads(raw.decode("utf-8"))
    buffers = [_load_buffer_uri(b.get("uri", ""), path.parent) for b in gltf_json.get("buffers", [])]
    return gltf_json, buffers


def _load_buffer_uri(uri: str, base_dir: Path) -> bytes:
    if uri.startswith("data:"):
        _, b64 = uri.split(",", 1)
        return base64.b64decode(b64)
    return (base_dir / uri).read_bytes()


def _read_accessor(gltf: dict[str, Any], buffers: list[bytes], acc_idx: int) -> np.ndarray:
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    buf = buffers[bv["buffer"]]
    dtype = _COMPONENT_DTYPES[acc["componentType"]]
    ncomp = _TYPE_COUNTS[acc["type"]]
    count = int(acc["count"])
    offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    stride = int(bv.get("byteStride", 0))
    elem_bytes = dtype.itemsize * ncomp

    if stride == 0 or stride == elem_bytes:
        arr = np.frombuffer(buf, dtype=dtype, count=count * ncomp, offset=offset)
        arr = arr.reshape((count, ncomp)) if ncomp > 1 else arr.reshape((count,))
        return np.array(arr, copy=True)

    # Access with a custom stride
    out = np.empty((count, ncomp) if ncomp > 1 else (count,), dtype=dtype)
    for i in range(count):
        v = np.frombuffer(buf, dtype=dtype, count=ncomp, offset=offset + i * stride)
        out[i] = v if ncomp > 1 else v[0]
    return out


def _load_image_texture(gltf: dict[str, Any], buffers: list[bytes], tex_idx: int, base_dir: Path) -> Texture | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    tex_desc = gltf.get("textures", [])[tex_idx]
    src_idx = tex_desc.get("source")
    if src_idx is None:
        return None
    img_desc = gltf.get("images", [])[src_idx]
    if "bufferView" in img_desc:
        bv = gltf["bufferViews"][img_desc["bufferView"]]
        off = int(bv.get("byteOffset", 0))
        ln = int(bv["byteLength"])
        raw = buffers[bv["buffer"]][off : off + ln]
        rgba = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)
        return Texture.from_array(rgba)
    if "uri" in img_desc:
        uri = img_desc["uri"]
        if uri.startswith("data:"):
            _, b64 = uri.split(",", 1)
            rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA"), dtype=np.uint8)
            return Texture.from_array(rgba)
        return Texture.from_file(base_dir / uri)
    return None


def _compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    if len(indices) == 0:
        normals[:, 1] = 1.0
        return normals
    p0 = positions[indices[:, 0]]
    p1 = positions[indices[:, 1]]
    p2 = positions[indices[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    for c in range(3):
        np.add.at(normals, indices[:, c], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.ascontiguousarray(
        np.where(lens > 1e-8, normals / np.maximum(lens, 1e-8), np.array([0.0, 1.0, 0.0], np.float32)),
        dtype=np.float32,
    )


def _quat_to_euler_yxz(q: list[float]) -> tuple[float, float, float]:
    """Converts a glTF quaternion `[x, y, z, w]` into angles `(rx, ry, rz)` compatible with `Ry @ Rx @ Rz`."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 1e-8:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    # 3×3 rotation matrix
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Ry @ Rx @ Rz decomposition: r12 = -sin(rx), r02 = sin(ry)cos(rx), r22 = cos(ry)cos(rx)
    sin_x = max(-1.0, min(1.0, -r12))
    rx = math.asin(sin_x)
    if abs(math.cos(rx)) > 1e-5:
        ry = math.atan2(r02, r22)
        rz = math.atan2(r10, r11)
    else:
        ry = math.atan2(-r20, r00)
        rz = 0.0
    return (rx, ry, rz)


def _extract_primitive(gltf: dict[str, Any], buffers: list[bytes], prim: dict[str, Any]) -> Mesh:
    attrs = prim["attributes"]
    positions = _read_accessor(gltf, buffers, attrs["POSITION"]).astype(np.float32, copy=False)
    if "indices" in prim:
        idx_raw = _read_accessor(gltf, buffers, prim["indices"]).astype(np.int32, copy=False).ravel()
        indices = idx_raw.reshape(-1, 3)
    else:
        n = (len(positions) // 3) * 3
        indices = np.arange(n, dtype=np.int32).reshape(-1, 3)

    if "NORMAL" in attrs:
        normals = _read_accessor(gltf, buffers, attrs["NORMAL"]).astype(np.float32, copy=False)
    else:
        normals = _compute_normals(positions, indices)

    if "TEXCOORD_0" in attrs:
        uvs = _read_accessor(gltf, buffers, attrs["TEXCOORD_0"]).astype(np.float32, copy=False)
        # glTF uses v=0 at the top, our engine uses v=0 at the bottom → flip V
        uvs = np.ascontiguousarray(np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=-1), dtype=np.float32)
    else:
        uvs = np.zeros((len(positions), 2), dtype=np.float32)

    return Mesh(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        uvs=uvs,
        indices=np.ascontiguousarray(indices, dtype=np.int32),
    )


def _extract_material(
    gltf: dict[str, Any],
    buffers: list[bytes],
    mat_idx: int | None,
    base_dir: Path,
    tex_cache: dict[int, Texture | None],
) -> Material:
    if mat_idx is None:
        return Material()
    md = gltf.get("materials", [])[mat_idx]
    pbr = md.get("pbrMetallicRoughness", {})
    bc = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    metallic = float(pbr.get("metallicFactor", 1.0 if "metallicRoughnessTexture" in pbr else 0.0))
    roughness = float(pbr.get("roughnessFactor", 0.5))
    em = md.get("emissiveFactor", [0.0, 0.0, 0.0])

    albedo_map = None
    if "baseColorTexture" in pbr:
        ti = int(pbr["baseColorTexture"]["index"])
        if ti not in tex_cache:
            tex_cache[ti] = _load_image_texture(gltf, buffers, ti, base_dir)
        albedo_map = tex_cache[ti]

    return Material(
        albedo=(float(bc[0]), float(bc[1]), float(bc[2])),
        metallic=metallic,
        roughness=roughness,
        emissive=(float(em[0]), float(em[1]), float(em[2])),
        albedo_map=albedo_map,
    )


def load_gltf_mesh(path: str | Path, *, normalize_size: float | None = None) -> Mesh:
    """Loads every triangle of a `.glb` / `.gltf` file into a single merged `Mesh`."""
    p = Path(path)
    gltf, buffers = _parse_gltf_container(p)
    pos_l, norm_l, uv_l, idx_l = [], [], [], []
    v_off = 0
    for mesh_desc in gltf.get("meshes", []):
        for prim in mesh_desc.get("primitives", []):
            if prim.get("mode", 4) != 4:  # TRIANGLES uniquement
                continue
            m = _extract_primitive(gltf, buffers, prim)
            pos_l.append(m.positions)
            norm_l.append(m.normals)
            uv_l.append(m.uvs)
            idx_l.append(m.indices + v_off)
            v_off += m.num_vertices
    if not pos_l:
        raise ValueError(f"No triangle mesh found in {p}")

    positions = np.ascontiguousarray(np.concatenate(pos_l, axis=0), dtype=np.float32)
    normals = np.ascontiguousarray(np.concatenate(norm_l, axis=0), dtype=np.float32)
    uvs = np.ascontiguousarray(np.concatenate(uv_l, axis=0), dtype=np.float32)
    indices = np.ascontiguousarray(np.concatenate(idx_l, axis=0), dtype=np.int32)

    if normalize_size is not None and len(positions) > 0:
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        span = float(np.max(hi - lo))
        if span > 1e-6:
            positions = np.ascontiguousarray((positions - 0.5 * (lo + hi)) * (normalize_size / span), dtype=np.float32)

    recipe: dict[str, Any] = {"kind": "gltf", "path": str(p)}
    if normalize_size is not None:
        recipe["normalize_size"] = float(normalize_size)
    return Mesh(positions=positions, normals=normals, uvs=uvs, indices=indices, recipe=recipe)


def load_gltf_nodes(path: str | Path) -> list[Node]:
    """Imports the full hierarchy (`Node` / `Object3D` + PBR materials + textures) of a `.glb`/`.gltf`."""
    p = Path(path)
    gltf, buffers = _parse_gltf_container(p)
    tex_cache: dict[int, Texture | None] = {}
    nodes_raw = gltf.get("nodes", [])
    meshes_raw = gltf.get("meshes", [])

    def build_node(ni: int) -> Node:
        nd = nodes_raw[ni]
        name = nd.get("name", f"gltf_node_{ni}")
        pos = tuple(float(x) for x in nd.get("translation", (0.0, 0.0, 0.0)))
        rot = _quat_to_euler_yxz(nd["rotation"]) if "rotation" in nd else (0.0, 0.0, 0.0)
        scl = tuple(float(x) for x in nd.get("scale", (1.0, 1.0, 1.0)))

        if "mesh" in nd:
            mdesc = meshes_raw[int(nd["mesh"])]
            prims = [pr for pr in mdesc.get("primitives", []) if pr.get("mode", 4) == 4]
            if len(prims) == 1:
                m = _extract_primitive(gltf, buffers, prims[0])
                mat = _extract_material(gltf, buffers, prims[0].get("material"), p.parent, tex_cache)
                node: Node = Object3D(name, m, mat, pos=pos, rot=rot, scale=scl)
            else:
                node = Node(name, pos=pos, rot=rot, scale=scl)
                for pi, pr in enumerate(prims):
                    m = _extract_primitive(gltf, buffers, pr)
                    mat = _extract_material(gltf, buffers, pr.get("material"), p.parent, tex_cache)
                    node.add_child(Object3D(f"{name}_p{pi}", m, mat))
        else:
            node = Node(name, pos=pos, rot=rot, scale=scl)

        for ci in nd.get("children", []):
            node.add_child(build_node(int(ci)))
        return node

    scenes = gltf.get("scenes", [])
    if scenes:
        si = int(gltf.get("scene", 0))
        root_ids = [int(i) for i in scenes[si].get("nodes", [])]
    else:
        referenced = {int(c) for nd in nodes_raw for c in nd.get("children", [])}
        root_ids = [i for i in range(len(nodes_raw)) if i not in referenced]

    return [build_node(i) for i in root_ids]


def write_glb_from_mesh(path: str | Path, mesh: Mesh, material: Material | None = None) -> Path:
    """Writes a `Mesh` (+ PBR `Material`) in the standard binary `.glb` v2 format (useful for export & tests)."""
    p = Path(path)
    pos_b = np.ascontiguousarray(mesh.positions, dtype=np.float32).tobytes()
    norm_b = np.ascontiguousarray(mesh.normals, dtype=np.float32).tobytes()
    uvs_flip = np.ascontiguousarray(np.stack([mesh.uvs[:, 0], 1.0 - mesh.uvs[:, 1]], axis=-1), dtype=np.float32)
    uv_b = uvs_flip.tobytes()
    idx_b = np.ascontiguousarray(mesh.indices.ravel(), dtype=np.uint32).tobytes()

    bin_parts = [pos_b, norm_b, uv_b, idx_b]
    offsets = []
    cur = 0
    for part in bin_parts:
        offsets.append((cur, len(part)))
        cur += len(part)
    bin_blob = b"".join(bin_parts)
    pad_bin = (-len(bin_blob)) % 4
    bin_blob += b"\x00" * pad_bin

    lo = mesh.positions.min(axis=0).tolist()
    hi = mesh.positions.max(axis=0).tolist()
    mat = material or Material()

    doc = {
        "asset": {"version": "2.0", "generator": "gpusync.gfx"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "mesh", "mesh": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [float(mat.albedo[0]), float(mat.albedo[1]), float(mat.albedo[2]), 1.0],
                    "metallicFactor": float(mat.metallic),
                    "roughnessFactor": float(mat.roughness),
                },
                "emissiveFactor": [float(mat.emissive[0]), float(mat.emissive[1]), float(mat.emissive[2])],
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0][0], "byteLength": offsets[0][1]},
            {"buffer": 0, "byteOffset": offsets[1][0], "byteLength": offsets[1][1]},
            {"buffer": 0, "byteOffset": offsets[2][0], "byteLength": offsets[2][1]},
            {"buffer": 0, "byteOffset": offsets[3][0], "byteLength": offsets[3][1]},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": mesh.num_vertices, "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": 1, "componentType": 5126, "count": mesh.num_vertices, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": mesh.num_vertices, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": mesh.num_triangles * 3, "type": "SCALAR"},
        ],
    }

    json_blob = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    pad_json = (-len(json_blob)) % 4
    json_blob += b" " * pad_json

    total_len = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    out = (
        struct.pack("<III", 0x46546C67, 2, total_len)
        + struct.pack("<II", len(json_blob), 0x4E4F534A)
        + json_blob
        + struct.pack("<II", len(bin_blob), 0x004E4942)
        + bin_blob
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(out)
    return p
