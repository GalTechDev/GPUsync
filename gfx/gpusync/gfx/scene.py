"""`Scene` base class: inheritable multi-scenes (`load`/`update`/`unload`) and `.scene.json` export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera import Camera2D, Camera3D
from .material import Environment, Light, PointLight, SpotLight, light_from_dict
from .math3d import batch_normal_from_models, batch_trs, mat4_look_at, mat4_ortho
from .node import InstancedMesh3D, Node, Object3D
from .objects2d import Object2D, Particles2D
from .texture import RenderTarget, Texture
from .ui import UILayer


class Scene:
    """Inheritable 3D/2D/UI scene with a life cycle (`load`, `update`, `on_event`, `unload`) and `.scene.json` export.

    Example:
        class MyLevel(gfx.Scene):
            source = "levels/base.scene.json"  # optional

            def load(self):
                self.orb = self.add(gfx.Object3D("orb", gfx.Mesh.sphere(0.8), pos=(0, 1.5, 0)))

            def update(self, dt: float):
                self.orb.rot_y += 1.5 * dt
    """

    source: str | Path | None = None

    def __init__(self) -> None:
        self.app: Any = None
        self.roots: list[Node] = []
        self.objects_2d: list[Object2D] = []
        self.particles_2d: list[Particles2D] = []
        self.lights: list[Light] = []
        self.cameras: list[Camera3D] = []
        self.cameras_2d: list[Camera2D] = []
        self.env: Environment = Environment()
        self.ui: UILayer = UILayer()
        self.clear_color: tuple[int, int, int, int] = (14, 18, 28, 255)

        self._loaded: bool = False
        self._geom_dirty: bool = True
        self._geom_version: int = 0

        # Flattened geometry buffers (recomputed only when the topology changes)
        self._flat_nodes: list[Node] = []
        self._node_parent_idx: list[int] = []
        self._obj_node_indices: list[int] = []
        self._objects: list[Object3D] = []
        self._v_offsets: list[int] = []
        self._obj_mat_ranges: list[tuple[int, int]] = []
        self._total_model_slots: int = 0
        self._textures: list[Texture] = []
        self._render_targets: list[RenderTarget] = []
        self.fluid_volumes: list[Any] = []
        self._has_interactive_rt: bool = False
        self._sky_tid: int = -1
        self._has_hierarchy: bool = False

        self.v_pos = np.zeros((0, 3), dtype=np.float32)
        self.v_norm = np.zeros((0, 3), dtype=np.float32)
        self.v_uv = np.zeros((0, 2), dtype=np.float32)
        self.v_obj = np.zeros((0,), dtype=np.int32)
        self.tri_idx = np.zeros((0, 3), dtype=np.int32)
        self.tri_mat = np.zeros((0,), dtype=np.int32)
        self.tri_obj = np.zeros((0,), dtype=np.int32)
        self.materials = np.zeros((0, 16), dtype=np.float32)

    # ── Life cycle (to override in subclasses) ──

    def load(self) -> None:
        """Called once when the scene becomes active (after `source` is loaded, if any)."""

    def update(self, dt: float) -> None:
        """Called every frame with the elapsed time `dt` (in seconds)."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Called for each window event not consumed by the UI."""

    def unload(self) -> None:
        """Called when leaving the scene (cleanup, if any)."""

    def _ensure_loaded(self, app: Any = None) -> None:
        if app is not None:
            self.app = app
        if self._loaded:
            return
        if self.source is not None:
            self.load_file(self.source)
        self.load()
        if not self.cameras:
            self.cameras.append(Camera3D.orbit())
        self._loaded = True
        self.compile_geometry()

    # ── Main camera & multiple cameras ──

    @property
    def camera(self) -> Camera3D:
        if not self.cameras:
            self.cameras.append(Camera3D.orbit())
        return self.cameras[0]

    @camera.setter
    def camera(self, cam: Camera3D) -> None:
        if self.cameras:
            self.cameras[0] = cam
        else:
            self.cameras.append(cam)

    @property
    def camera_2d(self) -> Camera2D:
        if not self.cameras_2d:
            self.cameras_2d.append(Camera2D())
        return self.cameras_2d[0]

    @camera_2d.setter
    def camera_2d(self, cam: Camera2D) -> None:
        if self.cameras_2d:
            self.cameras_2d[0] = cam
        else:
            self.cameras_2d.append(cam)

    @property
    def has_2d(self) -> bool:
        return bool(self.objects_2d or self.particles_2d)

    def add_camera(self, cam: Camera3D | Camera2D) -> Camera3D | Camera2D:
        if isinstance(cam, Camera2D):
            self.cameras_2d.append(cam)
        else:
            self.cameras.append(cam)
        return cam

    def active_cameras(self) -> list[Camera3D]:
        if not self.cameras:
            self.cameras.append(Camera3D.orbit())
        return sorted((c for c in self.cameras if c.enabled), key=lambda c: c.order)

    # ── Managing nodes, 3D/2D objects and lights ──

    @property
    def objects(self) -> list[Object3D]:
        if self._geom_dirty:
            self._rebuild_node_lists()
        return self._objects

    def add(self, item: Any, *, parent: Node | None = None):
        """Adds a `Node`/`Object3D`, `GpuFluidSPH3D`, `Object2D` or `Particles2D` to the scene."""
        if isinstance(item, Object2D):
            self.objects_2d.append(item)
            return item
        if isinstance(item, Particles2D):
            self.particles_2d.append(item)
            return item
        if hasattr(item, "pos_gpu") and hasattr(item, "config") and hasattr(item.config, "render_mode"):
            self.fluid_volumes.append(item)
            return item
        if parent is not None:
            parent.add_child(item)
        else:
            if item.parent is not None:
                item.parent.remove_child(item)
            self.roots.append(item)
        self.mark_dirty()
        return item

    def remove(self, item: Any) -> None:
        if isinstance(item, Object2D):
            if item in self.objects_2d:
                self.objects_2d.remove(item)
            return
        if isinstance(item, Particles2D):
            if item in self.particles_2d:
                self.particles_2d.remove(item)
            return
        if item in self.fluid_volumes:
            self.fluid_volumes.remove(item)
            return
        if item.parent is not None:
            item.parent.remove_child(item)
        elif item in self.roots:
            self.roots.remove(item)
        self.mark_dirty()

    def add_light(self, light: Light) -> Light:
        self.lights.append(light)
        return light

    def import_gltf(self, path: str | Path, *, parent: Node | None = None) -> list[Node]:
        """Imports every node, PBR material and texture of a `.glb` / `.gltf` file into the scene."""
        from .gltf import load_gltf_nodes

        nodes = load_gltf_nodes(path)
        for n in nodes:
            self.add(n, parent=parent)
        return nodes

    def get(self, name: str) -> Node | None:
        """Looks for the first node named `name` in the whole graph."""
        for root in self.roots:
            for n in root.iter_tree():
                if n.name == name:
                    return n
        return None

    def find_by_tag(self, tag: str) -> list[Node]:
        out: list[Node] = []
        for root in self.roots:
            for n in root.iter_tree():
                if tag in n.tags:
                    out.append(n)
        return out

    def mark_dirty(self) -> None:
        """Signals that a mesh, a material or the hierarchy changed (VRAM recompilation)."""
        self._geom_dirty = True

    def clear(self) -> None:
        self.roots.clear()
        self.objects_2d.clear()
        self.particles_2d.clear()
        self.lights.clear()
        self.cameras.clear()
        self.cameras_2d.clear()
        self.ui.clear()
        self.mark_dirty()

    # ── Geometry compilation & vectorized transforms ──

    def _rebuild_node_lists(self) -> None:
        flat: list[Node] = []
        parents: list[int] = []
        objs: list[Object3D] = []
        obj_node_idx: list[int] = []
        has_hier = False

        def visit(node: Node, p_idx: int) -> None:
            nonlocal has_hier
            idx = len(flat)
            flat.append(node)
            parents.append(p_idx)
            if p_idx >= 0:
                has_hier = True
            if isinstance(node, Object3D):
                objs.append(node)
                obj_node_idx.append(idx)
            for ch in node.children:
                visit(ch, idx)

        for r in self.roots:
            visit(r, -1)

        self._flat_nodes = flat
        self._node_parent_idx = parents
        self._objects = objs
        self._obj_node_indices = obj_node_idx
        self._has_hierarchy = has_hier

    def compile_geometry(self) -> None:
        """Flattens every `Object3D` of the graph into contiguous flat buffers ready for the GPU."""
        if not self._geom_dirty:
            return
        self._rebuild_node_lists()
        objs = self._objects
        self._textures = []
        self._render_targets = []
        self._has_interactive_rt = False
        self._sky_tid = -1
        self._v_offsets = []
        self._obj_mat_ranges = []
        self._total_model_slots = 0
        if not objs:
            self.v_pos = np.zeros((1, 3), dtype=np.float32)
            self.v_norm = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            self.v_uv = np.zeros((1, 2), dtype=np.float32)
            self.v_obj = np.zeros((1,), dtype=np.int32)
            self.tri_idx = np.zeros((0, 3), dtype=np.int32)
            self.tri_mat = np.zeros((0,), dtype=np.int32)
            self.tri_obj = np.zeros((0,), dtype=np.int32)
            self.materials = np.zeros((1, 16), dtype=np.float32)
            self._geom_dirty = False
            self._geom_version += 1
            return

        tex_map: dict[int, int] = {}

        def reg_tex(tex: Texture | None) -> int:
            if tex is None:
                return -1
            k = id(tex)
            if k not in tex_map:
                tex_map[k] = len(self._textures)
                self._textures.append(tex)
                if isinstance(tex, RenderTarget):
                    self._render_targets.append(tex)
                    if tex.interactive:
                        self._has_interactive_rt = True
            return tex_map[k]

        pos_l, norm_l, uv_l, obj_l = [], [], [], []
        idx_l, mat_l, mats = [], [], []
        v_offset = 0
        m_slot = 0
        for oi, obj in enumerate(objs):
            m = obj.mesh
            mat = obj.material
            self._v_offsets.append(v_offset)
            mats.append(
                mat.pack(
                    albedo_tid=reg_tex(mat.albedo_map),
                    emissive_tid=reg_tex(mat.emissive_map),
                    roughness_tid=reg_tex(mat.roughness_map),
                )
            )
            if isinstance(obj, InstancedMesh3D):
                cnt = obj.count
                self._obj_mat_ranges.append((m_slot, cnt))
                nv_m = m.num_vertices
                nt_m = m.num_triangles
                pos_l.append(np.tile(m.positions, (cnt, 1)))
                norm_l.append(np.tile(m.normals, (cnt, 1)))
                uv_l.append(np.tile(m.uvs, (cnt, 1)))
                obj_l.append(np.repeat(np.arange(m_slot, m_slot + cnt, dtype=np.int32), nv_m))
                inst_offsets = (np.arange(cnt, dtype=np.int32) * nv_m + v_offset)[:, None, None]
                idx_l.append((m.indices[None, :, :] + inst_offsets).reshape(-1, 3))
                mat_l.append(np.full(nt_m * cnt, oi, dtype=np.int32))
                v_offset += nv_m * cnt
                m_slot += cnt
            else:
                self._obj_mat_ranges.append((m_slot, 1))
                pos_l.append(m.positions)
                norm_l.append(m.normals)
                uv_l.append(m.uvs)
                obj_l.append(np.full(m.num_vertices, m_slot, dtype=np.int32))
                idx_l.append(m.indices + v_offset)
                mat_l.append(np.full(m.num_triangles, oi, dtype=np.int32))
                v_offset += m.num_vertices
                m_slot += 1

        if self.env.skybox is not None:
            self._sky_tid = reg_tex(self.env.skybox)

        self._total_model_slots = m_slot
        self.v_pos = np.ascontiguousarray(np.concatenate(pos_l, axis=0), dtype=np.float32)
        self.v_norm = np.ascontiguousarray(np.concatenate(norm_l, axis=0), dtype=np.float32)
        self.v_uv = np.ascontiguousarray(np.concatenate(uv_l, axis=0), dtype=np.float32)
        self.v_obj = np.ascontiguousarray(np.concatenate(obj_l, axis=0), dtype=np.int32)
        self.tri_idx = np.ascontiguousarray(np.concatenate(idx_l, axis=0), dtype=np.int32)
        self.tri_mat = np.ascontiguousarray(np.concatenate(mat_l, axis=0), dtype=np.int32)
        self.tri_obj = self.tri_mat
        self.materials = np.ascontiguousarray(np.stack(mats, axis=0), dtype=np.float32)
        self._geom_dirty = False
        self._geom_version += 1

    def build_world_transforms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes `(models_flat, normals_flat, light_mvps_flat, light_vp_flat)` for the whole scene.

        Shared by every camera of a frame (avoids recomputing `vertex_world` and the
        shadow map several times with multiple cameras).
        """
        self.compile_geometry()
        sun = self.env.sun
        sun_eye = np.asarray(sun.shadow_target, dtype=np.float32) + sun.dir_normalized * sun.shadow_distance
        light_view = mat4_look_at(sun_eye, sun.shadow_target, (0.0, 1.0, 0.0))
        ext = sun.shadow_extent
        light_proj = mat4_ortho(-ext, ext, -ext, ext, 1.0, sun.shadow_distance * 2.2)
        light_vp = (light_proj @ light_view).astype(np.float32)

        n_objs = len(self._objects)
        if n_objs == 0:
            eye = np.eye(4, dtype=np.float32).reshape(1, 16)
            return eye, eye, eye, np.ascontiguousarray(light_vp.ravel())

        n_nodes = len(self._flat_nodes)
        pos = np.empty((n_nodes, 3), dtype=np.float32)
        rot = np.empty((n_nodes, 3), dtype=np.float32)
        scl = np.empty((n_nodes, 3), dtype=np.float32)
        vis = np.empty((n_nodes,), dtype=np.bool_)
        for i, node in enumerate(self._flat_nodes):
            pos[i] = node._pos
            rot[i] = node._rot
            scl[i] = node._scale
            vis[i] = node.visible

        if not self._has_hierarchy:
            scl_eff = np.where(vis[:, None], scl, 0.0)
            m4, n4 = batch_trs(pos, rot, scl_eff)
            obj_m4 = m4[self._obj_node_indices]
            obj_n4 = n4[self._obj_node_indices]
        else:
            local_m4, _ = batch_trs(pos, rot, scl)
            world_m4 = local_m4.copy()
            vis_tree = vis.copy()
            for i, p_idx in enumerate(self._node_parent_idx):
                if p_idx >= 0:
                    world_m4[i] = world_m4[p_idx] @ local_m4[i]
                    vis_tree[i] = vis_tree[p_idx] and vis[i]
            obj_idx = self._obj_node_indices
            obj_m4 = world_m4[obj_idx]
            obj_vis = vis_tree[obj_idx]
            if not np.all(obj_vis):
                obj_m4 = obj_m4.copy()
                obj_m4[~obj_vis, :3, :3] = 0.0
            obj_n4 = batch_normal_from_models(obj_m4)

        total_slots = self._total_model_slots
        if total_slots == n_objs:
            full_m4 = obj_m4
            full_n4 = obj_n4
        else:
            full_m4 = np.zeros((total_slots, 4, 4), dtype=np.float32)
            full_n4 = np.zeros((total_slots, 4, 4), dtype=np.float32)
            full_m4[:, 3, 3] = 1.0
            full_n4[:, 3, 3] = 1.0
            for oi, (m_start, cnt) in enumerate(self._obj_mat_ranges):
                obj = self._objects[oi]
                if isinstance(obj, InstancedMesh3D):
                    if obj.gpu_trs is None and obj.gpu_matrices is None:
                        t_arr = obj.trs
                        sc_eff = np.where(t_arr[:, 7:8] > 0.0, t_arr[:, 6:7], 0.0)
                        im4, in4 = batch_trs(t_arr[:, 0:3], t_arr[:, 3:6], np.repeat(sc_eff, 3, axis=1))
                        full_m4[m_start : m_start + cnt] = im4
                        full_n4[m_start : m_start + cnt] = in4
                else:
                    full_m4[m_start] = obj_m4[oi]
                    full_n4[m_start] = obj_n4[oi]

        light_mvps = np.matmul(light_vp, full_m4).reshape(total_slots, 16)
        return (
            np.ascontiguousarray(full_m4.reshape(total_slots, 16)),
            np.ascontiguousarray(full_n4.reshape(total_slots, 16)),
            np.ascontiguousarray(light_mvps),
            np.ascontiguousarray(light_vp.ravel()),
        )

    def build_camera_mvps(self, models_flat: np.ndarray, cam: Camera3D, aspect: float) -> np.ndarray:
        """Computes `MVP = P @ V @ M` as a batch for a given camera."""
        vp = (cam.proj_matrix(aspect) @ cam.view_matrix()).astype(np.float32)
        m4 = models_flat.reshape(-1, 4, 4)
        return np.ascontiguousarray(np.matmul(vp, m4).reshape(m4.shape[0], 16))

    def pack_lights(self) -> np.ndarray:
        """Packs the active lights into a `(max(1, N), 16)` f32 array (PointLight + SpotLight)."""
        if not self.lights:
            return np.zeros((1, 16), dtype=np.float32)
        out = np.zeros((len(self.lights), 16), dtype=np.float32)
        for i, light in enumerate(self.lights):
            out[i] = light.pack()
        return out

    def pack_materials(self) -> np.ndarray:
        """Updates the material array (if a material was changed on the fly)."""
        if not self._objects:
            return self.materials
        return np.ascontiguousarray(np.stack([o.material.pack() for o in self._objects], axis=0), dtype=np.float32)

    # ── Export / Import `.scene.json` ──

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        return {
            "version": 1,
            "environment": self.env.to_dict(base_dir=base_dir),
            "cameras": [c.to_dict() for c in self.cameras],
            "cameras_2d": [c.to_dict() for c in self.cameras_2d],
            "lights": [l.to_dict() for l in self.lights],
            "nodes": [r.to_dict(base_dir=base_dir) for r in self.roots],
        }

    def save(self, path: str | Path) -> Path:
        """Exports the scene as readable JSON (`.scene.json`)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(base_dir=p.parent)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return p

    def load_dict(self, data: dict[str, Any], *, base_dir: Path | None = None) -> None:
        """Loads the data of a scene dictionary into `self`."""
        self.clear()
        if "environment" in data:
            self.env = Environment.from_dict(data["environment"], base_dir=base_dir)
        self.cameras = [Camera3D.from_dict(c) for c in data.get("cameras", [])]
        self.cameras_2d = [Camera2D.from_dict(c) for c in data.get("cameras_2d", [])]
        self.lights = [light_from_dict(l) for l in data.get("lights", [])]
        for nd in data.get("nodes", []):
            self.roots.append(Node.from_dict(nd, base_dir=base_dir))
        self.mark_dirty()

    def load_file(self, path: str | Path) -> None:
        """Loads a `.scene.json` file into `self`."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        self.load_dict(data, base_dir=p.parent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Scene:
        sc = cls()
        sc.load_dict(data, base_dir=base_dir)
        sc._loaded = True
        sc.compile_geometry()
        return sc

    @classmethod
    def from_file(cls, path: str | Path) -> Scene:
        """Creates a `Scene` directly from a `.scene.json` file."""
        sc = cls()
        sc.load_file(path)
        sc._loaded = True
        sc.compile_geometry()
        return sc
