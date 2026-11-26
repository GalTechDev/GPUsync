"""Hierarchical nodes (`Node`, `Object3D`) for the `gpusync.gfx` scene graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .material import Material
from .math3d import mat4_trs
from .mesh import Mesh


class Node:
    """Hierarchical transform node (parent/child, position, rotation, scale, visibility)."""

    def __init__(
        self,
        name: str = "node",
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
        visible: bool = True,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self._pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        self._rot = [float(rot[0]), float(rot[1]), float(rot[2])]
        if isinstance(scale, (int, float)):
            self._scale = [float(scale), float(scale), float(scale)]
        else:
            self._scale = [float(scale[0]), float(scale[1]), float(scale[2])]
        self.visible = bool(visible)
        self.tags: set[str] = set(tags) if tags else set()
        self.parent: Node | None = None
        self.children: list[Node] = []

    # ── Convenient transform properties ──

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self._pos[0], self._pos[1], self._pos[2])

    @pos.setter
    def pos(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self._pos = [float(value[0]), float(value[1]), float(value[2])]

    @property
    def x(self) -> float:
        return self._pos[0]

    @x.setter
    def x(self, v: float) -> None:
        self._pos[0] = float(v)

    @property
    def y(self) -> float:
        return self._pos[1]

    @y.setter
    def y(self, v: float) -> None:
        self._pos[1] = float(v)

    @property
    def z(self) -> float:
        return self._pos[2]

    @z.setter
    def z(self, v: float) -> None:
        self._pos[2] = float(v)

    @property
    def rot(self) -> tuple[float, float, float]:
        return (self._rot[0], self._rot[1], self._rot[2])

    @rot.setter
    def rot(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self._rot = [float(value[0]), float(value[1]), float(value[2])]

    @property
    def rot_x(self) -> float:
        return self._rot[0]

    @rot_x.setter
    def rot_x(self, v: float) -> None:
        self._rot[0] = float(v)

    @property
    def rot_y(self) -> float:
        return self._rot[1]

    @rot_y.setter
    def rot_y(self, v: float) -> None:
        self._rot[1] = float(v)

    @property
    def rot_z(self) -> float:
        return self._rot[2]

    @rot_z.setter
    def rot_z(self, v: float) -> None:
        self._rot[2] = float(v)

    @property
    def scale(self) -> tuple[float, float, float]:
        return (self._scale[0], self._scale[1], self._scale[2])

    @scale.setter
    def scale(self, value: float | tuple[float, float, float] | np.ndarray) -> None:
        if isinstance(value, (int, float)):
            s = float(value)
            self._scale = [s, s, s]
        else:
            self._scale = [float(value[0]), float(value[1]), float(value[2])]

    # ── Hierarchy ──

    def add_child(self, child: Node) -> Node:
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)
        return child

    def remove_child(self, child: Node) -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def is_visible_in_tree(self) -> bool:
        cur: Node | None = self
        while cur is not None:
            if not cur.visible:
                return False
            cur = cur.parent
        return True

    def local_matrix(self) -> np.ndarray:
        return mat4_trs(self._pos, self._rot, self._scale)

    def world_matrix(self) -> np.ndarray:
        m = self.local_matrix()
        cur = self.parent
        while cur is not None:
            m = cur.local_matrix() @ m
            cur = cur.parent
        return m.astype(np.float32)

    def iter_tree(self):
        """Depth-first traversal (parent before children)."""
        yield self
        for ch in self.children:
            yield from ch.iter_tree()

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "node",
            "name": self.name,
            "pos": list(self._pos),
            "rot": list(self._rot),
            "scale": list(self._scale),
            "visible": self.visible,
        }
        if self.tags:
            d["tags"] = sorted(self.tags)
        if self.children:
            d["children"] = [c.to_dict(base_dir=base_dir) for c in self.children]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Node:
        kind = data.get("kind", "node")
        if kind == "object3d":
            node: Node = Object3D.from_dict(data, base_dir=base_dir)
        else:
            node = Node(
                name=data.get("name", "node"),
                pos=tuple(data.get("pos", (0.0, 0.0, 0.0))),
                rot=tuple(data.get("rot", (0.0, 0.0, 0.0))),
                scale=tuple(data.get("scale", (1.0, 1.0, 1.0))),
                visible=bool(data.get("visible", True)),
                tags=data.get("tags"),
            )
            for ch in data.get("children", []):
                node.add_child(Node.from_dict(ch, base_dir=base_dir))
        return node

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} pos={self.pos}>"


class Object3D(Node):
    """Renderable 3D object: ties a `Mesh` and a `Material` to a node of the scene graph.

    Compute↔render bridge: assign `obj.gpu_positions` (and optionally `obj.gpu_normals`) to a
    `gs.Array` `f32[num_vertices, 3]` produced by your `@gs.kernel` to deform the mesh in
    real time without any CPU copy.
    """

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        material: Material | None = None,
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
        visible: bool = True,
        cast_shadow: bool = True,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name, pos=pos, rot=rot, scale=scale, visible=visible, tags=tags)
        self.mesh = mesh
        self.material = material if material is not None else Material()
        self.cast_shadow = bool(cast_shadow)
        self.gpu_positions: Any = None  # optional gs.Array f32[num_vertices, 3] (GPU deformation)
        self.gpu_normals: Any = None  # gs.Array f32[num_vertices, 3] optionnel
        self.gpu_transform: Any = None  # (gs.Array, row): see `bind_transform`
        self.water: Any = None  # `GpuWaterSurface`: transparent rendering (set by `apply_to`)

    def bind_transform(self, array: Any, row: int = 0) -> None:
        """Binds the object's position to a GPU computation: the object is drawn where `array[row]` puts it.

        `array` is a `gs.Array` f32 `(N, 8)` or wider: `[px, py, pz, rx, ry, rz, scale, visible, …]`,
        in world coordinates. Nothing goes back through the CPU: `pos` and `rot` are therefore no longer up to
        date, and the object's children do not follow it. `array=None` hands control back to the CPU."""
        if array is not None and (len(array.shape) != 2 or array.shape[1] < 8 or not 0 <= row < array.shape[0]):
            raise ValueError(f"bind_transform: array (N, 8) or wider and a valid row expected, got {tuple(array.shape)}, row {row}")
        self.gpu_transform = None if array is None else (array, int(row))

    def base_positions_gpu(self, device: str | None = None):
        """Uploads the rest vertices to VRAM once (`gs.Array` f32) for your `@gs.kernel`."""
        from .kernels.k3d import upload_f32

        return upload_f32(self.mesh.positions, device=device)

    def base_normals_gpu(self, device: str | None = None):
        from .kernels.k3d import upload_f32

        return upload_f32(self.mesh.normals, device=device)

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        d = super().to_dict(base_dir=base_dir)
        d["kind"] = "object3d"
        d["mesh"] = self.mesh.to_dict(base_dir=base_dir)
        d["material"] = self.material.to_dict(base_dir=base_dir)
        d["cast_shadow"] = self.cast_shadow
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Object3D:
        obj = cls(
            name=data.get("name", "object"),
            mesh=Mesh.from_dict(data["mesh"], base_dir=base_dir),
            material=Material.from_dict(data.get("material", {}), base_dir=base_dir),
            pos=tuple(data.get("pos", (0.0, 0.0, 0.0))),
            rot=tuple(data.get("rot", (0.0, 0.0, 0.0))),
            scale=tuple(data.get("scale", (1.0, 1.0, 1.0))),
            visible=bool(data.get("visible", True)),
            cast_shadow=bool(data.get("cast_shadow", True)),
            tags=data.get("tags"),
        )
        for ch in data.get("children", []):
            obj.add_child(Node.from_dict(ch, base_dir=base_dir))
        return obj


class InstancedMesh3D(Object3D):
    """Massive GPU instancing: renders `count` copies of the same `Mesh` driven by a GPU buffer.

    Feed `inst.gpu_trs` (`gs.Array` `f32[count, 8]`: `[px, py, pz, rx, ry, rz, scale, visible]`)
    or `inst.gpu_matrices` (`gs.Array` `f32[count, 16]`) directly from a `@gs.kernel`:
    the `model`, `normal`, `MVP` and `light_MVP` matrices are computed 100 % on the GPU.
    """

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        material: Material | None = None,
        count: int = 1,
        *,
        trs: np.ndarray | None = None,
        gpu_trs: Any = None,
        gpu_matrices: Any = None,
        visible: bool = True,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name, mesh, material, visible=visible, tags=tags)
        self.count = max(1, int(count))
        if trs is not None:
            self.trs = np.ascontiguousarray(trs, dtype=np.float32)
        else:
            self.trs = np.zeros((self.count, 8), dtype=np.float32)
            self.trs[:, 6] = 1.0  # scale = 1
            self.trs[:, 7] = 1.0  # visible = 1
        self.gpu_trs: Any = gpu_trs
        self.gpu_matrices: Any = gpu_matrices

    def base_trs_gpu(self, device: str | None = None):
        """Uploads the initial `(count, 8)` TRS array to VRAM for an animation `@gs.kernel`."""
        from .kernels.k3d import upload_f32

        return upload_f32(self.trs, device=device)
