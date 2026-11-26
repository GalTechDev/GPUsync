"""Multi-viewport 3D rendering pipeline + O(1) 3D picking and GPU UI compositor (`Renderer3D`)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import gpusync as gs

from .camera import Camera3D
from .input import PickResult
from .kernels.k3d import (
    BLOOM_GROUP,
    EMPTY_VIS,
    bind_trs_to_matrices,
    blit_viewport,
    bloom_down_h,
    bloom_v,
    clear_u32,
    fxaa_pass,
    mul_vp_matrices,
    patch_matrices,
    patch_trs_to_matrices,
    patch_vertices,
    rasterize_shadow,
    rasterize_triangles,
    shade_pbr,
    ssr_pass,
    tonemap_pack,
    tonemap_pack_nobloom,
    upload_f32,
    upload_i32,
    upload_u32,
    upscale_sharp,
    vertex_project,
    vertex_project_all,
    vertex_world,
)
from .kernels.kfluid import (
    fluid_bilateral_blur_h,
    fluid_bilateral_blur_v,
    fluid_curvature_flow,
    fluid_shade_volumetric,
    fluid_splat_spheres,
    fluid_unpack_dt,
)
from .kernels.kfluid3d import fluid_raymarch_3d, ww_splat_screen
from .kernels.kui import ui_composite
from .kernels.kwater import water_merge, water_optics
from .node import InstancedMesh3D
from .renderer2d import Renderer2D, _pack_rgba_u32
from .scene import Scene
from .texture import RenderTarget, TexturePool

_ALIGN = 16384


def _aligned_u32(shape: tuple[int, ...]) -> np.ndarray:
    nbytes = int(np.prod(shape, dtype=np.int64)) * 4
    raw = np.empty(nbytes + _ALIGN, dtype=np.uint8)
    off = (-raw.ctypes.data) % _ALIGN
    return raw[off : off + nbytes].view(np.uint32).reshape(shape)


def _align_dim_w(width: int) -> int:
    step = BLOOM_GROUP * 2  # 128 for bloom_down_h (group=64 at half resolution)
    return max(step, ((width + step // 2) // step) * step)


def _align_dim_h(height: int) -> int:
    return max(16, (height // 2) * 2)


class _GpuSceneCache:
    """Static scene buffers resident in VRAM (avoids any re-upload per frame)."""

    def __init__(self, scene: Scene, device: str | None) -> None:
        scene.compile_geometry()
        self.scene_id = id(scene)
        self.geom_version = scene._geom_version
        self.device = device
        self.n_tris = int(scene.tri_idx.shape[0])
        self.v_pos = upload_f32(scene.v_pos, device=device)
        self.v_norm = upload_f32(scene.v_norm, device=device)
        self.v_uv = upload_f32(scene.v_uv, device=device)
        self.v_obj = upload_i32(scene.v_obj, device=device)
        self.tri_idx = upload_i32(scene.tri_idx, device=device) if self.n_tris > 0 else None
        self.tri_mat = upload_i32(scene.tri_mat, device=device) if self.n_tris > 0 else None
        self.materials = upload_f32(scene.materials, device=device)
        self._waters: tuple[tuple[int, ...], gs.Array, list] | None = None
        self._slot_rows: dict[tuple, gs.Array] = {}

    def slot_rows(self, slots: tuple[tuple[int, int], ...], n_slots: int) -> gs.Array:
        """Table from matrix slot → row of the bound array (-1: object not bound)."""
        key = (slots, n_slots)
        if key not in self._slot_rows:
            if len(self._slot_rows) > 16:
                self._slot_rows.clear()
            table = np.full(n_slots, -1, dtype=np.int32)
            for slot, row in slots:
                table[slot] = row
            self._slot_rows[key] = upload_i32(table, device=self.device)
        return self._slot_rows[key]

    def split_water(self, scene: Scene) -> tuple[gs.Array, list]:
        """Opaque triangles (the water ones rendered degenerate: the numbering does not change) and, per water surface,
        `(object, first triangle, triangles)`."""
        which = tuple(i for i, o in enumerate(scene._objects) if o.water is not None and o.visible)
        if not which:
            return self.tri_idx, []
        if self._waters is None or self._waters[0] != which:
            opaque = scene.tri_idx.copy()
            parts = []
            for oi in which:
                tris = np.flatnonzero(scene.tri_obj == oi)
                first, last = int(tris[0]), int(tris[-1]) + 1
                parts.append((oi, first, upload_i32(np.ascontiguousarray(scene.tri_idx[first:last]), device=self.device)))
                opaque[first:last] = opaque[first, 0]
            self._waters = (which, upload_i32(opaque, device=self.device), parts)
        return self._waters[1], [(scene._objects[oi], first, tri) for oi, first, tri in self._waters[2]]


@dataclass
class _ViewportCapture:
    """Minimal state kept after rendering for exact 3D picking `(object, world_pos, uv)`."""

    camera: Camera3D
    rect: tuple[int, int, int, int]  # (ox, oy, dw, dh) in the screen framebuffer
    render_size: tuple[int, int]  # (rw, rh) of the visbuf
    visbuf: gs.Array
    models_flat: np.ndarray | gs.Array  # on the GPU if objects are placed there: read back only on click


class Renderer3D:
    """GPUsync 3D renderer (ultra-fast single camera, multi-viewport, GPU UI, 3D picking)."""

    def __init__(
        self,
        width: int = 1024,
        height: int = 576,
        *,
        shadow_size: int = 512,
        device: str | None = None,
        render_scale: float = 1.0,
    ) -> None:
        # Share of the resolution computed in 3D (single full-screen camera), scaled up afterwards; 2D and interface untouched
        self.render_scale = float(render_scale)
        self.width = _align_dim_w(width)
        self.height = _align_dim_h(height)
        self.shadow_size = int(shadow_size)
        self.device = device

        self._fb_buf = _aligned_u32((self.height, self.width))
        self._cache: _GpuSceneCache | None = None
        self._tex_pool = TexturePool()
        self._renderer2d = Renderer2D()
        self._vis_empty_cache: dict[tuple[int, int, str | None], gs.Array] = {}
        self._half_ref_cache: dict[tuple[int, int, str | None], gs.Array] = {}
        self._shadow_empty: gs.Array | None = None
        self._shadow_empty_dev: str | None = "__none__"
        self._dummy_u32: gs.Array | None = None
        self._dummy_u32_dev: str | None = "__none__"
        self._last_views: list[_ViewportCapture] = []
        self.last_timings_ms: dict[str, float] = {}
        self.active_device: str = device or gs.default_device().id

    def resize(self, width: int, height: int) -> None:
        w = _align_dim_w(width)
        h = _align_dim_h(height)
        if w != self.width or h != self.height:
            self.width = w
            self.height = h
            self._fb_buf = _aligned_u32((h, w))

    def invalidate_scene(self) -> None:
        self._cache = None
        self._last_views.clear()

    def _scene_buffers(self, scene: Scene, dev: str | None) -> _GpuSceneCache:
        scene.compile_geometry()
        c = self._cache
        if (
            c is None
            or c.scene_id != id(scene)
            or c.geom_version != scene._geom_version
            or c.device != dev
        ):
            self._cache = _GpuSceneCache(scene, dev)
        return self._cache

    def _get_vis_empty(self, w: int, h: int, dev: str | None) -> gs.Array:
        key = (w, h, dev)
        arr = self._vis_empty_cache.get(key)
        if arr is None:
            if len(self._vis_empty_cache) > 8:
                self._vis_empty_cache.clear()
            buf = _aligned_u32((h, w))
            buf.fill(EMPTY_VIS)
            arr = upload_u32(buf, device=dev)
            self._vis_empty_cache[key] = arr
        return arr

    def _get_half_ref(self, w: int, h: int, dev: str | None) -> gs.Array:
        key = (w, h, dev)
        arr = self._half_ref_cache.get(key)
        if arr is None:
            if len(self._half_ref_cache) > 8:
                self._half_ref_cache.clear()
            arr = upload_f32(np.empty((h // 2, w // 2), dtype=np.float32), device=dev)
            self._half_ref_cache[key] = arr
        return arr

    def _get_shadow_empty(self, dev: str | None) -> gs.Array:
        if self._shadow_empty is None or self._shadow_empty_dev != dev:
            buf = _aligned_u32((self.shadow_size, self.shadow_size))
            buf.fill(EMPTY_VIS)
            self._shadow_empty = upload_u32(buf, device=dev)
            self._shadow_empty_dev = dev
        return self._shadow_empty

    def _get_dummy_u32(self, dev: str | None) -> gs.Array:
        if self._dummy_u32 is None or self._dummy_u32_dev != dev:
            self._dummy_u32 = upload_u32(np.zeros((1, 1), dtype=np.uint32), device=dev)
            self._dummy_u32_dev = dev
        return self._dummy_u32

    def _apply_dynamic_gpu_buffers(
        self,
        scene: Scene,
        gpu: _GpuSceneCache,
        models_flat: np.ndarray,
        normals_flat: np.ndarray,
        light_mvps: np.ndarray,
        light_vp: np.ndarray,
        dev: str | None,
    ) -> tuple[gs.Array, gs.Array, np.ndarray | gs.Array, np.ndarray | gs.Array, np.ndarray | gs.Array, bool]:
        """Applies dynamic GPU vertices and GPU instances directly in VRAM (no CPU copy)."""
        v_pos = gpu.v_pos
        v_norm = gpu.v_norm
        total_v = int(v_pos.shape[0])

        for oi, obj in enumerate(scene._objects):
            if obj.gpu_positions is not None:
                off = scene._v_offsets[oi]
                if off == 0 and int(obj.gpu_positions.shape[0]) == total_v:
                    v_pos = obj.gpu_positions
                else:
                    v_pos = patch_vertices(v_pos, obj.gpu_positions, off, device=dev)
            if obj.gpu_normals is not None:
                off = scene._v_offsets[oi]
                if off == 0 and int(obj.gpu_normals.shape[0]) == total_v:
                    v_norm = obj.gpu_normals
                else:
                    v_norm = patch_vertices(v_norm, obj.gpu_normals, off, device=dev)

        has_gpu_inst = any(
            isinstance(o, InstancedMesh3D) and (o.gpu_trs is not None or o.gpu_matrices is not None)
            for o in scene._objects
        )
        # Objects bound to a GPU computation, grouped by source array (one kernel per array)
        bound: dict[int, tuple[gs.Array, list[tuple[int, int]]]] = {}
        for oi, obj in enumerate(scene._objects):
            if obj.gpu_transform is not None and not isinstance(obj, InstancedMesh3D):
                arr, row = obj.gpu_transform
                bound.setdefault(id(arr), (arr, []))[1].append((scene._obj_mat_ranges[oi][0], row))
        if not has_gpu_inst and not bound:
            return v_pos, v_norm, models_flat, normals_flat, light_mvps, False

        models_g = upload_f32(models_flat, device=dev)
        norms_g = upload_f32(normals_flat, device=dev)
        for oi, obj in enumerate(scene._objects):
            if isinstance(obj, InstancedMesh3D):
                m_start, _ = scene._obj_mat_ranges[oi]
                if obj.gpu_trs is not None:
                    models_g, norms_g = patch_trs_to_matrices(models_g, norms_g, obj.gpu_trs, m_start, device=dev)
                elif obj.gpu_matrices is not None:
                    models_g = patch_matrices(models_g, obj.gpu_matrices, m_start, device=dev)
                    norms_g = patch_matrices(norms_g, obj.gpu_matrices, m_start, device=dev)
        for arr, slots in bound.values():
            models_g, norms_g = bind_trs_to_matrices(
                models_g, norms_g, arr, gpu.slot_rows(tuple(slots), int(models_flat.shape[0])), device=dev
            )
        light_mvps_g = mul_vp_matrices(models_g, light_vp, device=dev)
        return v_pos, v_norm, models_g, norms_g, light_mvps_g, True

    def _render_camera_view(
        self,
        scene: Scene,
        gpu: _GpuSceneCache,
        cam: Camera3D,
        rw: int,
        rh: int,
        v_pos: gs.Array,
        wpos: gs.Array,
        wnorm: gs.Array,
        models_flat: np.ndarray | gs.Array,
        models_on_gpu: bool,
        light_mvps: np.ndarray | gs.Array,
        light_vp: np.ndarray,
        lights: np.ndarray,
        tex_pool_gpu: gs.Array,
        tex_meta_gpu: gs.Array,
        dyn_atlas_gpu: gs.Array,
        sky_tid: int,
        smap: gs.Array | None,
        compute_shadow: bool,
        dev: str | None,
        final_out: np.ndarray | None,
        timings: dict[str, float],
        profile: bool,
    ) -> tuple[gs.Array, gs.Array, gs.Array | None]:
        """Renders a camera view `(rw, rh)` and returns `(fb_cam, visbuf, smap)`."""

        def stamp(name: str, t0: float, arr=None) -> float:
            if profile:
                if arr is not None:
                    gs.sync(dev)
                now = time.perf_counter()
                timings[name] = timings.get(name, 0.0) + (now - t0) * 1000.0
                return now
            return t0

        aspect = rw / max(rh, 1)
        vp_flat = np.ascontiguousarray((cam.proj_matrix(aspect) @ cam.view_matrix()).astype(np.float32).ravel())
        if models_on_gpu:
            mvps: np.ndarray | gs.Array = mul_vp_matrices(models_flat, vp_flat, device=dev)
        else:
            mvps = scene.build_camera_mvps(models_flat, cam, aspect)
        fwd, right, up = cam.basis()
        tan_hfov = math.tan(math.radians(cam.fov_deg) * 0.5)
        sun = scene.env.sun
        sun_d = sun.dir_normalized
        is_ortho = 1 if cam.ortho else 0

        # Transparent water is drawn after the opaque scene (in perspective only)
        tri_opaque, waters = gpu.split_water(scene) if not cam.ortho else (gpu.tri_idx, [])

        t0 = time.perf_counter()
        if compute_shadow:
            sx, sy, inv_w, sh_x, sh_y, sh_z = vertex_project_all(
                v_pos,
                gpu.v_obj,
                mvps,
                light_mvps,
                float(rw),
                float(rh),
                float(cam.near),
                float(cam.far),
                is_ortho,
                float(self.shadow_size),
                float(self.shadow_size),
                device=dev,
            )
            t0 = stamp("vertex", t0, sx)
            smap = rasterize_shadow(
                tri_opaque, sh_x, sh_y, sh_z, device=dev, out=self._get_shadow_empty(dev)
            )
            t0 = stamp("shadow_map", t0, smap)
        else:
            sx, sy, inv_w = vertex_project(
                v_pos,
                gpu.v_obj,
                mvps,
                float(rw),
                float(rh),
                float(cam.near),
                float(cam.far),
                is_ortho,
                device=dev,
            )
            t0 = stamp("vertex", t0, sx)

        smap_arg = smap if (cam.shadows and smap is not None) else self._get_dummy_u32(dev)

        inv_depth_range = 1.0 / max(cam.far - cam.near, 1e-4)
        # Depth (24 bits), then number of the visible triangle (32 bits): two passes of the same kernel
        tri_args = (tri_opaque, sx, sy, inv_w)
        clip = (float(cam.near), float(inv_depth_range))
        empty = self._get_vis_empty(rw, rh, dev)
        zbuf = rasterize_triangles(*tri_args, self._get_dummy_u32(dev), *clip, 0, device=dev, out=empty)
        visbuf = rasterize_triangles(*tri_args, zbuf, *clip, 1, device=dev, out=empty)
        t0 = stamp("vis_raster", t0, visbuf)

        hdr_r, hdr_g, hdr_b = shade_pbr(
            visbuf,
            gpu.tri_idx,
            gpu.tri_mat,
            sx,
            sy,
            inv_w,
            wpos,
            wnorm,
            gpu.v_uv,
            gpu.materials,
            lights,
            smap_arg,
            light_vp,
            tex_pool_gpu,
            tex_meta_gpu,
            dyn_atlas_gpu,
            float(cam.pos[0]),
            float(cam.pos[1]),
            float(cam.pos[2]),
            float(fwd[0]),
            float(fwd[1]),
            float(fwd[2]),
            float(right[0]),
            float(right[1]),
            float(right[2]),
            float(up[0]),
            float(up[1]),
            float(up[2]),
            float(tan_hfov),
            float(sun_d[0]),
            float(sun_d[1]),
            float(sun_d[2]),
            float(sun.color[0]),
            float(sun.color[1]),
            float(sun.color[2]),
            float(sun.intensity),
            float(scene.env.fog_density),
            float(scene.env.fog_color[0]),
            float(scene.env.fog_color[1]),
            float(scene.env.fog_color[2]),
            int(sky_tid),
            float(scene.env.skybox_intensity),
            1 if (cam.shadows and smap is not None) else 0,
            1 if cam.wireframe else 0,
            float(cam.near),
            device=dev,
        )
        t0 = stamp("shade_pbr", t0, hdr_r)

        # Optional screen-space reflections (SSR) on shiny / metallic surfaces
        if cam.ssr and not cam.ortho:
            hdr_r, hdr_g, hdr_b = ssr_pass(
                hdr_r,
                hdr_g,
                hdr_b,
                visbuf,
                zbuf,
                gpu.tri_idx,
                gpu.tri_mat,
                sx,
                sy,
                inv_w,
                wpos,
                wnorm,
                gpu.materials,
                vp_flat,
                float(cam.pos[0]),
                float(cam.pos[1]),
                float(cam.pos[2]),
                float(cam.near),
                float(cam.far),
                float(cam.ssr_strength),
                int(cam.ssr_steps),
                device=dev,
            )
            t0 = stamp("ssr", t0, hdr_r)

        # Transparent water surfaces (refraction, absorption, caustics, reflections)
        for obj, first_tri, tri_w in waters:
            surf = obj.water
            cfg = surf.config
            w_args = (tri_w, sx, sy, inv_w)
            zbuf_w = rasterize_triangles(*w_args, self._get_dummy_u32(dev), *clip, 0, device=dev, out=empty)
            vis_w = rasterize_triangles(*w_args, zbuf_w, *clip, 1, device=dev, out=empty)
            center = obj.world_matrix()[:3, 3]
            hdr_r, hdr_g, hdr_b = water_optics(
                hdr_r,
                hdr_g,
                hdr_b,
                zbuf,
                zbuf_w,
                vis_w,
                tri_w,
                sx,
                sy,
                inv_w,
                wnorm,
                surf.caustic_map(sun_d),
                smap_arg,
                light_vp,
                tex_pool_gpu,
                tex_meta_gpu,
                dyn_atlas_gpu,
                int(sky_tid),
                float(scene.env.skybox_intensity),
                1 if (cam.shadows and smap is not None) else 0,
                *(float(v) for v in (*cam.pos, *fwd, *right, *up)),
                float(tan_hfov),
                float(cam.near),
                float(cam.far),
                *(float(v) for v in (*sun_d, *sun.color)),
                float(sun.intensity),
                float(scene.env.fog_density),
                *(float(v) for v in scene.env.fog_color),
                float(center[0]),
                float(center[2]),
                float(center[1] - cfg.depth),
                0.5 * float(cfg.size),
                1 if cfg.walls else 0,
                float(cfg.ior),
                *(float(v) for v in (*cfg.absorption, *cfg.tint, *cfg.floor_colors[0], *cfg.floor_colors[1])),
                float(cfg.tile),
                float(cfg.roughness),
                float(cfg.caustics),
                device=dev,
            )
            zbuf, visbuf = water_merge(zbuf, visbuf, zbuf_w, vis_w, int(first_tri), device=dev)
            t0 = stamp("water", t0, hdr_r)

        # Volumetric 3D fluids (3D Snell / Beer-Lambert raymarching OR screen-space SSFR)
        if scene.fluid_volumes and not cam.ortho:
            for fl in scene.fluid_volumes:
                if not getattr(fl, "visible", True):
                    continue
                cfg = fl.config
                if cfg.render_mode == "raymarch":
                    if fl.vol_dens_gpu is None:
                        fl._rebuild_volume_in_batch()
                    dx_v, dy_v, dz_v = cfg.vol_res
                    bx_f, by_f, bz_f = cfg.bounds
                    ww_a, ww_d = ww_splat_screen(
                        fl.ww_pos_gpu,
                        fl.ww_vel_gpu,
                        zbuf,
                        vp_flat,
                        float(cam.near),
                        float(cam.far),
                        device=dev,
                    )
                    fc1, fc2 = cfg.floor_colors
                    hdr_r, hdr_g, hdr_b = fluid_raymarch_3d(
                        hdr_r,
                        hdr_g,
                        hdr_b,
                        zbuf,
                        fl.vol_dens_gpu,
                        fl.caustic_map(sun_d),
                        ww_a,
                        ww_d,
                        tex_pool_gpu,
                        tex_meta_gpu,
                        dyn_atlas_gpu,
                        int(sky_tid),
                        float(scene.env.skybox_intensity),
                        int(dx_v),
                        int(dy_v),
                        int(dz_v),
                        float(bx_f),
                        float(by_f),
                        float(bz_f),
                        float(cfg.floor_y),
                        float(fc1[0]),
                        float(fc1[1]),
                        float(fc1[2]),
                        float(fc2[0]),
                        float(fc2[1]),
                        float(fc2[2]),
                        float(cam.pos[0]),
                        float(cam.pos[1]),
                        float(cam.pos[2]),
                        float(fwd[0]),
                        float(fwd[1]),
                        float(fwd[2]),
                        float(right[0]),
                        float(right[1]),
                        float(right[2]),
                        float(up[0]),
                        float(up[1]),
                        float(up[2]),
                        float(tan_hfov),
                        float(cam.near),
                        float(cam.far),
                        float(sun_d[0]),
                        float(sun_d[1]),
                        float(sun_d[2]),
                        float(sun.color[0]),
                        float(sun.color[1]),
                        float(sun.color[2]),
                        float(sun.intensity),
                        float(cfg.iso_level),
                        float(cfg.ior),
                        float(cfg.absorption[0]),
                        float(cfg.absorption[1]),
                        float(cfg.absorption[2]),
                        float(cfg.albedo[0]),
                        float(cfg.albedo[1]),
                        float(cfg.albedo[2]),
                        float(cfg.emissive[0]),
                        float(cfg.emissive[1]),
                        float(cfg.emissive[2]),
                        float(cfg.metallic),
                        float(cfg.roughness),
                        float(cfg.foam),
                        float(cfg.caustics),
                        device=dev,
                    )
                    continue
                if cfg.render_mode != "surface":
                    continue
                d_u32, t_u32 = fluid_splat_spheres(
                    fl.pos_gpu,
                    zbuf,
                    vp_flat,
                    float(tan_hfov),
                    float(cam.near),
                    float(cam.far),
                    float(cfg.surface_scale),
                    device=dev,
                )
                depth_f, thick_f = fluid_unpack_dt(d_u32, t_u32, float(cam.near), float(cam.far), device=dev)
                # Curvature flow (van der Laan / NVIDIA FleX): smooth glassy surface without staircase artefacts
                n_smooth = max(0, int(cfg.smooth_iters))
                for _ in range(n_smooth):
                    depth_f, thick_f = fluid_curvature_flow(
                        depth_f, thick_f, float(cfg.smooth_depth_sigma), device=dev
                    )
                if n_smooth > 0:
                    depth_f, thick_f = fluid_bilateral_blur_h(
                        depth_f, thick_f, float(cfg.smooth_radius), float(cfg.smooth_depth_sigma), device=dev
                    )
                    depth_f, thick_f = fluid_bilateral_blur_v(
                        depth_f, thick_f, float(cfg.smooth_radius), float(cfg.smooth_depth_sigma), device=dev
                    )
                hdr_r, hdr_g, hdr_b = fluid_shade_volumetric(
                    hdr_r,
                    hdr_g,
                    hdr_b,
                    depth_f,
                    thick_f,
                    tex_pool_gpu,
                    tex_meta_gpu,
                    dyn_atlas_gpu,
                    int(sky_tid),
                    float(scene.env.skybox_intensity),
                    float(fwd[0]),
                    float(fwd[1]),
                    float(fwd[2]),
                    float(right[0]),
                    float(right[1]),
                    float(right[2]),
                    float(up[0]),
                    float(up[1]),
                    float(up[2]),
                    float(tan_hfov),
                    float(sun_d[0]),
                    float(sun_d[1]),
                    float(sun_d[2]),
                    float(sun.color[0]),
                    float(sun.color[1]),
                    float(sun.color[2]),
                    float(sun.intensity),
                    float(cfg.absorption[0]),
                    float(cfg.absorption[1]),
                    float(cfg.absorption[2]),
                    float(cfg.albedo[0]),
                    float(cfg.albedo[1]),
                    float(cfg.albedo[2]),
                    float(cfg.emissive[0]),
                    float(cfg.emissive[1]),
                    float(cfg.emissive[2]),
                    float(cfg.refraction),
                    float(cfg.metallic),
                    float(cfg.roughness),
                    float(cfg.foam),
                    device=dev,
                )
            t0 = stamp("fluid_surface", t0, hdr_r)

        # Effets HDR pluggables (stage="hdr")
        hdr_fx = []
        ldr_fx = []
        for fx in cam.effects:
            if isinstance(fx, tuple):
                st, fn = fx
                if st == "hdr":
                    hdr_fx.append(fn)
                else:
                    ldr_fx.append(fn)
            elif getattr(fx, "enabled", True):
                if getattr(fx, "stage", "ldr") == "hdr":
                    hdr_fx.append(fx)
                else:
                    ldr_fx.append(fx)

        for fx in hdr_fx:
            if hasattr(fx, "apply"):
                hdr_r, hdr_g, hdr_b = fx.apply((hdr_r, hdr_g, hdr_b), device=dev)
            else:
                hdr_r, hdr_g, hdr_b = fx(hdr_r, hdr_g, hdr_b, device=dev)

        has_ldr_fx = len(ldr_fx) > 0
        post_out = None if has_ldr_fx else final_out
        tm_out = None if cam.fxaa else post_out
        can_bloom = cam.bloom and (rw % (BLOOM_GROUP * 2) == 0) and (rh % 2 == 0)
        if can_bloom:
            half_ref = self._get_half_ref(rw, rh, dev)
            bh_r, bh_g, bh_b = bloom_down_h(
                hdr_r, hdr_g, hdr_b, half_ref, float(cam.bloom_threshold), device=dev
            )
            bl_r, bl_g, bl_b = bloom_v(bh_r, bh_g, bh_b, device=dev)
            t0 = stamp("bloom", t0, bl_r)
            fb = tonemap_pack(
                hdr_r,
                hdr_g,
                hdr_b,
                bl_r,
                bl_g,
                bl_b,
                float(cam.bloom_strength),
                float(cam.exposure),
                device=dev,
                out=tm_out,
            )
        else:
            fb = tonemap_pack_nobloom(hdr_r, hdr_g, hdr_b, float(cam.exposure), device=dev, out=tm_out)
        t0 = stamp("tonemap", t0, fb)

        if cam.fxaa:
            fb = fxaa_pass(fb, device=dev, out=post_out)
            t0 = stamp("fxaa", t0, fb)

        # Pluggable LDR / screen effects (`PostEffect` or user `@gs.kernel`)
        for fi, fx in enumerate(ldr_fx):
            fx_out = final_out if fi == len(ldr_fx) - 1 else None
            if hasattr(fx, "apply"):
                fb = fx.apply(fb, device=dev, out=fx_out)
            else:
                fb = fx(fb, device=dev, out=fx_out)
        if has_ldr_fx:
            t0 = stamp("post_fx", t0, fb)

        return fb, visbuf, smap

    def _update_render_target(
        self,
        rt: RenderTarget,
        parent_scene: Scene,
        parent_gpu: _GpuSceneCache,
        parent_vpos: gs.Array | None,
        parent_wpos: gs.Array | None,
        parent_wnorm: gs.Array | None,
        parent_models: np.ndarray | gs.Array | None,
        parent_models_gpu: bool,
        parent_light_mvps: np.ndarray | gs.Array | None,
        parent_light_vp: np.ndarray | None,
        parent_lights: np.ndarray | None,
        dev: str | None,
    ) -> None:
        """Renders a `RenderTarget` (3D camera, 2D/3D scene or interactive UI) directly in VRAM."""
        rw = max(16, (rt.width // 2) * 2)
        rh = max(16, (rt.height // 2) * 2)
        if rt.match_viewport_aspect and self.width >= 64 and self.height >= 64:
            # Match the mirror's aspect ratio exactly to the main viewport (half resolution)
            rw = max(64, ((self.width // 2) // 2) * 2)
            rh = max(64, ((self.height // 2) // 2) * 2)
        fb: gs.Array | None = None

        if rt.camera is not None:
            # 3D camera (e.g. security camera, mirror, portal)
            target_scene = rt.scene if rt.scene is not None else parent_scene
            target_scene._ensure_loaded()
            if (
                target_scene is parent_scene
                and parent_vpos is not None
                and parent_wpos is not None
                and parent_wnorm is not None
                and parent_models is not None
            ):
                t_gpu = parent_gpu
                t_vpos, t_wpos, t_wnorm = parent_vpos, parent_wpos, parent_wnorm
                t_models, t_mgpu, t_lmvps, t_lvp, t_lights = (
                    parent_models,
                    parent_models_gpu,
                    parent_light_mvps,
                    parent_light_vp,
                    parent_lights,
                )
            else:
                t_gpu = _GpuSceneCache(target_scene, dev)
                t_models, t_norms, t_lmvps, t_lvp = target_scene.build_world_transforms()
                t_lights = target_scene.pack_lights()
                t_vpos, t_vnorm, t_models, t_norms, t_lmvps, t_mgpu = self._apply_dynamic_gpu_buffers(
                    target_scene, t_gpu, t_models, t_norms, t_lmvps, t_lvp, dev
                )
                t_wpos, t_wnorm = vertex_world(t_vpos, t_vnorm, t_gpu.v_obj, t_models, t_norms, device=dev)

            # Keep every static texture (the HDRI skybox included!) at the same indices,
            # and replace only the RenderTargets with a 2×2 black texture to avoid any recursion.
            has_rt_tex = any(isinstance(t, RenderTarget) for t in target_scene._textures)
            if has_rt_tex:
                if not hasattr(self, "_rt_dummy_tex"):
                    from .texture import Texture

                    self._rt_dummy_tex = Texture.solid((10, 12, 18, 255), width=2, height=2)
                rt_ids = {i for i, t in enumerate(target_scene._textures) if isinstance(t, RenderTarget)}
                safe_textures = [self._rt_dummy_tex if i in rt_ids else t for i, t in enumerate(target_scene._textures)]
                tp_gpu, tm_gpu, da_gpu = self._tex_pool.build(safe_textures, dev)
                rt_mats = target_scene.materials.copy()
                for col in (13, 14, 15):
                    mask = np.isin(rt_mats[:, col].astype(np.int32), list(rt_ids))
                    if col == 14:
                        rt_mats[mask, 7:10] = 0.04
                    rt_mats[mask, col] = -1.0
                t_gpu_mats = upload_f32(rt_mats, device=dev)
                old_mats = t_gpu.materials
                t_gpu.materials = t_gpu_mats
                rt_sky_tid = -1 if target_scene._sky_tid in rt_ids else target_scene._sky_tid
            else:
                tp_gpu, tm_gpu, da_gpu = self._tex_pool.build(target_scene._textures, dev)
                old_mats = None
                rt_sky_tid = target_scene._sky_tid

            cam = rt.camera
            fb, _, _ = self._render_camera_view(
                target_scene,
                t_gpu,
                cam,
                rw,
                rh,
                t_vpos,
                t_wpos,
                t_wnorm,
                t_models,
                t_mgpu,
                t_lmvps,
                t_lvp,
                t_lights,
                tp_gpu,
                tm_gpu,
                da_gpu,
                sky_tid=rt_sky_tid,
                smap=None,
                compute_shadow=False,
                dev=dev,
                final_out=None,
                timings={},
                profile=False,
            )
            if old_mats is not None:
                t_gpu.materials = old_mats

        elif rt.scene is not None:
            # 2D scene (or 2D + UI) rendered into the texture
            sub = rt.scene
            sub._ensure_loaded()
            fb = self._renderer2d.render(
                rw,
                rh,
                sub.objects_2d,
                sub.particles_2d,
                sub.camera_2d if sub.cameras_2d else None,
                base=None,
                clear_color=rt.clear_color,
                device=dev,
            )
            if sub.ui.has_visible():
                uc = sub.ui.compile(rw, rh, dev)
                if uc is not None:
                    fb = ui_composite(fb, uc[0], uc[1], uc[2], device=dev)
        else:
            # Standalone UI on a plain background
            fb = clear_u32(self._get_vis_empty(rw, rh, dev), _pack_rgba_u32(rt.clear_color), device=dev)

        # Extra UI attached directly to the RenderTarget
        if rt.ui is not None and rt.ui.has_visible() and fb is not None:
            uc = rt.ui.compile(rw, rh, dev)
            if uc is not None:
                fb = ui_composite(fb, uc[0], uc[1], uc[2], device=dev)

        if fb is not None:
            rt.set_gpu_frame(fb)

    def render_frame(self, scene: Scene, *, profile: bool = False, host_out: bool = False) -> gs.Array:
        """Runs the 3D layers (if any), 2D (if any) and UI in a single `gs.batch()`."""
        scene._ensure_loaded()
        dev = self.device
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        gpu = self._scene_buffers(scene, dev)
        has_3d = gpu.n_tris > 0
        has_2d = scene.has_2d
        has_ui = scene.ui.has_visible()
        ui_compiled = scene.ui.compile(self.width, self.height, dev) if has_ui else None

        # Pure 2D path (or empty scene / UI): NO 3D pass launched!
        if not has_3d:
            with gs.batch(dev):
                if has_2d:
                    fb = self._renderer2d.render(
                        self.width,
                        self.height,
                        scene.objects_2d,
                        scene.particles_2d,
                        scene.camera_2d if scene.cameras_2d else None,
                        base=None,
                        clear_color=scene.clear_color,
                        device=dev,
                        out=self._fb_buf if (host_out and ui_compiled is None) else None,
                    )
                else:
                    fb = clear_u32(
                        self._get_vis_empty(self.width, self.height, dev),
                        _pack_rgba_u32(scene.clear_color),
                        device=dev,
                        out=self._fb_buf if (host_out and ui_compiled is None) else None,
                    )
                if ui_compiled is not None:
                    boxes_gpu, glyphs_gpu, n_boxes = ui_compiled
                    fb = ui_composite(
                        fb, boxes_gpu, glyphs_gpu, n_boxes, device=dev, out=self._fb_buf if host_out else None
                    )
            self._last_views = []
            self.active_device = fb.device
            if profile:
                timings["2d_ui"] = (time.perf_counter() - t0) * 1000.0
                self.last_timings_ms = timings
            return fb

        cams = scene.active_cameras()
        models_flat, normals_flat, light_mvps, light_vp = scene.build_world_transforms()
        lights = scene.pack_lights()
        if profile:
            timings["cpu_prep"] = (time.perf_counter() - t0) * 1000.0

        need_any_shadow = any(c.shadows for c in cams)
        single_full = len(cams) == 1 and cams[0].viewport.is_full
        final_3d_out = self._fb_buf if (host_out and not has_2d and ui_compiled is None) else None
        views_capture: list[_ViewportCapture] = []

        with gs.batch(dev):
            # 1. Dynamic GPU vertices + GPU instancing + world-space transform (once for everything)
            v_pos_f, v_norm_f, models_f, norms_f, light_mvps_f, models_on_gpu = self._apply_dynamic_gpu_buffers(
                scene, gpu, models_flat, normals_flat, light_mvps, light_vp, dev
            )
            wpos, wnorm = vertex_world(v_pos_f, v_norm_f, gpu.v_obj, models_f, norms_f, device=dev)

            # 2. Update the RenderTargets (cameras on 3D screens, 2D scenes / UI on 3D surfaces)
            for rt in scene._render_targets:
                self._update_render_target(
                    rt, scene, gpu, v_pos_f, wpos, wnorm, models_f, models_on_gpu, light_mvps_f, light_vp, lights, dev
                )
            tex_pool_gpu, tex_meta_gpu, dyn_atlas_gpu = self._tex_pool.build(scene._textures, dev)

            smap: gs.Array | None = None
            shadow_done = False

            if single_full:
                cam = cams[0]
                scale = min(max(self.render_scale, 0.25), 1.0)
                rw, rh = self.width, self.height
                if scale < 1.0:
                    rw, rh = _align_dim_w(int(self.width * scale)), _align_dim_h(int(self.height * scale))
                scaled = (rw, rh) != (self.width, self.height)
                fb, visbuf, smap = self._render_camera_view(
                    scene,
                    gpu,
                    cam,
                    rw,
                    rh,
                    v_pos_f,
                    wpos,
                    wnorm,
                    models_f,
                    models_on_gpu,
                    light_mvps_f,
                    light_vp,
                    lights,
                    tex_pool_gpu,
                    tex_meta_gpu,
                    dyn_atlas_gpu,
                    scene._sky_tid,
                    smap,
                    compute_shadow=need_any_shadow,
                    dev=dev,
                    final_out=None if scaled else final_3d_out,
                    timings=timings,
                    profile=profile,
                )
                if scaled:
                    fb = upscale_sharp(self._get_vis_empty(self.width, self.height, dev), fb, device=dev, out=final_3d_out)
                views_capture.append(_ViewportCapture(cam, (0, 0, self.width, self.height), (rw, rh), visbuf, models_f))
            else:
                fb: gs.Array | None = None
                for ci, cam in enumerate(cams):
                    ox, oy, dw, dh = cam.viewport.pixel_rect(self.width, self.height)
                    rw = _align_dim_w(dw) if cam.bloom else max(16, (dw // 2) * 2)
                    rh = _align_dim_h(dh)
                    compute_shadow = need_any_shadow and not shadow_done
                    if compute_shadow:
                        shadow_done = True

                    is_last_cam = ci == len(cams) - 1
                    cam_is_full = cam.viewport.is_full and rw == self.width and rh == self.height
                    cam_out = final_3d_out if (is_last_cam and cam_is_full) else None

                    sub_fb, visbuf, smap = self._render_camera_view(
                        scene,
                        gpu,
                        cam,
                        rw,
                        rh,
                        v_pos_f,
                        wpos,
                        wnorm,
                        models_f,
                        models_on_gpu,
                        light_mvps_f,
                        light_vp,
                        lights,
                        tex_pool_gpu,
                        tex_meta_gpu,
                        dyn_atlas_gpu,
                        scene._sky_tid,
                        smap,
                        compute_shadow=compute_shadow,
                        dev=dev,
                        final_out=cam_out,
                        timings=timings,
                        profile=profile,
                    )
                    views_capture.append(_ViewportCapture(cam, (ox, oy, dw, dh), (rw, rh), visbuf, models_f))

                    if fb is None and cam_is_full:
                        fb = sub_fb
                    else:
                        if fb is None:
                            fb = clear_u32(
                                self._get_vis_empty(self.width, self.height, dev),
                                _pack_rgba_u32(scene.clear_color),
                                device=dev,
                            )
                        bc = cam.border_color
                        border_u32 = (
                            (bc[0] & 255) | ((bc[1] & 255) << 8) | ((bc[2] & 255) << 16) | 0xFF000000
                            if bc is not None
                            else 0
                        )
                        blit_out = final_3d_out if is_last_cam else None
                        fb = blit_viewport(fb, sub_fb, ox, oy, dw, dh, border_u32, device=dev, out=blit_out)

            # 3. 2D layer, if any, composited over the 3D
            if has_2d and fb is not None:
                fb = self._renderer2d.render(
                    self.width,
                    self.height,
                    scene.objects_2d,
                    scene.particles_2d,
                    scene.camera_2d if scene.cameras_2d else None,
                    base=fb,
                    clear_color=scene.clear_color,
                    device=dev,
                    out=self._fb_buf if (host_out and ui_compiled is None) else None,
                )

            # 4. Screen UI layer (last, over every camera and the 2D)
            if ui_compiled is not None and fb is not None:
                t_ui = time.perf_counter()
                boxes_gpu, glyphs_gpu, n_boxes = ui_compiled
                fb = ui_composite(
                    fb, boxes_gpu, glyphs_gpu, n_boxes, device=dev, out=self._fb_buf if host_out else None
                )
                if profile:
                    gs.sync(dev)
                    timings["ui"] = (time.perf_counter() - t_ui) * 1000.0

        assert fb is not None
        self._last_views = views_capture
        self.active_device = fb.device
        if profile:
            self.last_timings_ms = timings
        return fb

    def render_rgba(self, scene: Scene, *, profile: bool = False) -> np.ndarray:
        """Renders the scene and returns a NumPy view `(H, W, 4)` uint8."""
        fb = self.render_frame(scene, profile=profile, host_out=True)
        return np.asarray(fb).view(np.uint8).reshape(self.height, self.width, 4)

    def pick(self, scene: Scene, x: float, y: float) -> PickResult | None:
        """Exact O(1) 3D picking from the Visibility Buffer of the last frame.

        Returns `PickResult(object, triangle, world_pos, uv, camera)` or `None` if the pixel is sky.
        """
        if not self._last_views or not scene.objects:
            return None
        ix, iy = int(x), int(y)
        # Go in reverse drawing order: the camera drawn on top wins
        for cap in reversed(self._last_views):
            ox, oy, dw, dh = cap.rect
            if not (ox <= ix < ox + dw and oy <= iy < oy + dh):
                continue
            rw, rh = cap.render_size
            vx = min(max(0, int((ix - ox) * rw / max(dw, 1))), rw - 1)
            vy = min(max(0, int((iy - oy) * rh / max(dh, 1))), rh - 1)
            vis_arr = np.asarray(cap.visbuf)
            vis = int(vis_arr[vy, vx])
            if vis == EMPTY_VIS:
                return None
            tri_id = vis
            if tri_id >= len(scene.tri_idx):
                return None
            obj_idx = int(scene.tri_obj[tri_id])
            obj = scene.objects[obj_idx]

            # Exact computation of (world_pos, uv): homogeneous barycentrics (valid even when the triangle
            # crosses the near plane). The matrix is the vertex's (an instance occupies several).
            i0, i1, i2 = int(scene.tri_idx[tri_id, 0]), int(scene.tri_idx[tri_id, 1]), int(scene.tri_idx[tri_id, 2])
            models = np.asarray(cap.models_flat)
            m = models[int(scene.v_obj[i0])].reshape(4, 4).astype(np.float64)
            p_h = np.concatenate([scene.v_pos[[i0, i1, i2]], np.ones((3, 1))], axis=1)
            w_pts = (p_h @ m.T)[:, :3]

            vp = (cap.camera.proj_matrix(rw / max(rh, 1)) @ cap.camera.view_matrix()).astype(np.float64)
            c_pts = np.concatenate([w_pts, np.ones((3, 1))], axis=1) @ vp.T
            hw = np.ones(3) if cap.camera.ortho else c_pts[:, 3]
            gx = (c_pts[:, 0] * 0.5 + 0.5 * hw) * rw - (vx + 0.5) * hw
            gy = (0.5 * hw - c_pts[:, 1] * 0.5) * rh - (vy + 0.5) * hw
            e = np.array([gx[1] * gy[2] - gx[2] * gy[1], gx[2] * gy[0] - gx[0] * gy[2], gx[0] * gy[1] - gx[1] * gy[0]])
            s_e = float(e.sum())
            b = e / s_e if abs(s_e) > 1e-12 else np.full(3, 1.0 / 3.0)

            world_pos = (b[0] * w_pts[0] + b[1] * w_pts[1] + b[2] * w_pts[2]).astype(np.float32)
            uvs = scene.v_uv[[i0, i1, i2]]
            uv = (float(b[0] * uvs[0, 0] + b[1] * uvs[1, 0] + b[2] * uvs[2, 0]),
                  float(b[0] * uvs[0, 1] + b[1] * uvs[1, 1] + b[2] * uvs[2, 1]))
            return PickResult(object=obj, triangle=tri_id, world_pos=world_pos, uv=uv, camera=cap.camera)
        return None
