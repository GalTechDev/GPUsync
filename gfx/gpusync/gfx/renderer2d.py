"""Moteur de rendu 2D pur 100 % GPUsync (`Renderer2D`) : zéro surcoût 3D, sprites, formes, particules GPU."""

from __future__ import annotations

import math

import numpy as np

import gpusync as gs

from .camera import Camera2D
from .kernels.k2d import composite_particles_2d, render_2d_shapes, scatter_particles_2d
from .kernels.k3d import clear_u32, upload_u32
from .objects2d import Object2D, Particles2D
from .texture import Texture, TexturePool


def _pack_rgba_u32(color: tuple[int, int, int, int]) -> int:
    r = max(0, min(255, int(color[0])))
    g = max(0, min(255, int(color[1])))
    b = max(0, min(255, int(color[2])))
    a = max(0, min(255, int(color[3]))) if len(color) > 3 else 255
    return r | (g << 8) | (b << 16) | (a << 24)


class Renderer2D:
    """Rend une liste d'`Object2D` et `Particles2D` via une `Camera2D` (seul ou sur un framebuffer 3D)."""

    def __init__(self) -> None:
        self._tex_pool = TexturePool()
        self._zero_cache: dict[tuple[int, int, str | None], gs.Array] = {}

    def _get_zero_layer(self, w: int, h: int, dev: str | None) -> gs.Array:
        key = (w, h, dev)
        arr = self._zero_cache.get(key)
        if arr is None:
            if len(self._zero_cache) > 8:
                self._zero_cache.clear()
            arr = upload_u32(np.zeros((h, w), dtype=np.uint32), device=dev)
            self._zero_cache[key] = arr
        return arr

    def render(
        self,
        width: int,
        height: int,
        objects_2d: list[Object2D],
        particles_2d: list[Particles2D],
        camera: Camera2D | None = None,
        *,
        base: gs.Array | None = None,
        clear_color: tuple[int, int, int, int] = (14, 18, 28, 255),
        device: str | None = None,
        out: np.ndarray | None = None,
    ) -> gs.Array:
        """Exécute la couche 2D sur GPU.

        Si `base is None`, part de `clear_color` (scène 2D pure ou `RenderTarget`).
        Sinon, compose directement par-dessus `base`.
        """
        cam = camera or Camera2D()
        do_clear = 1 if base is None else 0
        base_arg = base if base is not None else self._get_zero_layer(width, height, device)
        clear_u32_val = _pack_rgba_u32(clear_color)

        visible_shapes = sorted((o for o in objects_2d if o.visible), key=lambda o: o.z)
        visible_parts = [p for p in particles_2d if p.visible]

        has_particles = len(visible_parts) > 0
        shapes_out = None if has_particles else out

        if visible_shapes:
            textures: list[Texture] = []
            tex_map: dict[int, int] = {}

            def get_tid(tex: Texture | None) -> int:
                if tex is None:
                    return -1
                k = id(tex)
                if k not in tex_map:
                    tex_map[k] = len(textures)
                    textures.append(tex)
                return tex_map[k]

            sf_list, si_list = [], []
            for o in visible_shapes:
                f_row, i_row = o.pack(get_tid)
                sf_list.append(f_row)
                si_list.append(i_row)
            shapes_f = np.asarray(sf_list, dtype=np.float32)
            shapes_i = np.asarray(si_list, dtype=np.int32)
            tex_pool_gpu, tex_meta_gpu, dyn_atlas_gpu = self._tex_pool.build(textures, device)

            inv_zoom = 1.0 / max(float(cam.zoom), 1e-4)
            cam_cos = math.cos(cam.rotation)
            cam_sin = math.sin(cam.rotation)

            fb = render_2d_shapes(
                base_arg,
                shapes_f,
                shapes_i,
                tex_pool_gpu,
                tex_meta_gpu,
                dyn_atlas_gpu,
                len(visible_shapes),
                float(cam.pos[0]),
                float(cam.pos[1]),
                float(inv_zoom),
                float(cam_cos),
                float(cam_sin),
                clear_u32_val,
                do_clear,
                device=device,
                out=shapes_out,
            )
        elif do_clear:
            fb = clear_u32(base_arg, clear_u32_val, device=device, out=shapes_out)
        else:
            fb = base_arg

        if has_particles:
            cam_cos = math.cos(cam.rotation)
            cam_sin = math.sin(cam.rotation)
            layer = self._get_zero_layer(width, height, device)
            for p in visible_parts:
                layer = scatter_particles_2d(
                    p.px,
                    p.py,
                    p.colors,
                    float(cam.pos[0]),
                    float(cam.pos[1]),
                    float(cam.zoom),
                    float(cam_cos),
                    float(cam_sin),
                    int(p.point_size),
                    device=device,
                    out=layer,
                )
            fb = composite_particles_2d(fb, layer, device=device, out=out)

        return fb
