"""Unified GPU textures (`Texture`, `RenderTarget`, `TexturePool`) for 3D, 2D, video and dynamic surfaces.

Everything rests on a `u32[H, W]` buffer (packed RGBA8) resident in VRAM:
- `Texture.from_file("wood.png")` / `Texture.from_array(rgba)` / `tex.update(frame)` (video),
- `Texture.from_gpu(gs_array)` (direct output of a `@gs.kernel`, zero-copy),
- `RenderTarget(w, h, camera=..., scene=..., ui=...)`: a 3D camera, a 2D scene or an interactive UI
  rendered live onto any 3D material (`albedo_map` / `emissive_map`) or `Sprite2D`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import gpusync as gs

from .kernels.k3d import blit_viewport, upload_i32, upload_u32


def pack_rgba_u8_to_u32(arr: np.ndarray) -> np.ndarray:
    """Converts a `(H, W, 3|4)` uint8 or `(H, W)` uint32 array into a contiguous `(H, W)` uint32 RGBA8 array."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.dtype == np.uint32:
        return np.ascontiguousarray(a, dtype=np.uint32)
    if a.ndim == 3 and a.shape[2] == 4 and a.dtype == np.uint8:
        return np.ascontiguousarray(a, dtype=np.uint8).view(np.uint32).reshape(a.shape[0], a.shape[1])
    if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
        h, w, _ = a.shape
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[:, :, :3] = a
        out[:, :, 3] = 255
        return out.view(np.uint32).reshape(h, w)
    raise TypeError(f"Texture expects (H, W) uint32 or (H, W, 3|4) uint8, got {a.shape} {a.dtype}")


class Texture:
    """RGBA8 2D texture (`u32[H, W]`) usable in 3D (`Material`) and in 2D (`Sprite2D`)."""

    def __init__(
        self,
        pixels: np.ndarray | gs.Array,
        *,
        bilinear: bool = True,
        recipe: dict[str, Any] | None = None,
    ) -> None:
        self.bilinear: bool = bool(bilinear)
        self.recipe: dict[str, Any] | None = recipe
        self._version: int = 1
        self._gpu_array: gs.Array | None = None
        self._cpu_pixels: np.ndarray | None = None

        if isinstance(pixels, gs.Array):
            if pixels.ndim != 2 or pixels.dtype != np.uint32:
                raise TypeError(f"GPU texture expects a 2-D u32 gs.Array, got {pixels.shape} {pixels.dtype}")
            self._gpu_array = pixels
            self.height, self.width = int(pixels.shape[0]), int(pixels.shape[1])
        else:
            u32 = pack_rgba_u8_to_u32(pixels)
            self._cpu_pixels = u32
            self.height, self.width = int(u32.shape[0]), int(u32.shape[1])

    @property
    def is_dynamic(self) -> bool:
        return self._gpu_array is not None

    @property
    def gpu_array(self) -> gs.Array | None:
        return self._gpu_array

    @property
    def cpu_pixels(self) -> np.ndarray | None:
        return self._cpu_pixels

    def update(self, pixels: np.ndarray | gs.Array) -> None:
        """Updates the texture in place (e.g. a new video frame, a CPU canvas or a `@gs.kernel` output)."""
        if isinstance(pixels, gs.Array):
            if pixels.ndim != 2 or pixels.dtype != np.uint32:
                raise TypeError("Texture.update(gs.Array) expects a 2-D u32 array")
            self._gpu_array = pixels
            self._cpu_pixels = None
            self.height, self.width = int(pixels.shape[0]), int(pixels.shape[1])
        else:
            u32 = pack_rgba_u8_to_u32(pixels)
            self._cpu_pixels = u32
            self._gpu_array = None
            self.height, self.width = int(u32.shape[0]), int(u32.shape[1])
        self._version += 1

    # ── Constructeurs ──

    @classmethod
    def from_array(cls, arr: np.ndarray, *, bilinear: bool = True) -> Texture:
        return cls(arr, bilinear=bilinear)

    @classmethod
    def from_gpu(cls, arr: gs.Array, *, bilinear: bool = True) -> Texture:
        """Wraps a `gs.Array` `u32[H, W]` produced by a `@gs.kernel` directly (zero-copy)."""
        return cls(arr, bilinear=bilinear)

    @classmethod
    def from_file(cls, path: str | Path, *, bilinear: bool = True) -> Texture:
        from PIL import Image

        p = Path(path)
        rgba = np.asarray(Image.open(p).convert("RGBA"), dtype=np.uint8)
        return cls(rgba, bilinear=bilinear, recipe={"kind": "file", "path": str(p), "bilinear": bilinear})

    @classmethod
    def solid(
        cls,
        color: tuple[int, int, int] | tuple[int, int, int, int],
        width: int = 4,
        height: int = 4,
    ) -> Texture:
        c = (color[0], color[1], color[2], color[3] if len(color) > 3 else 255)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :] = c
        return cls(
            rgba,
            bilinear=False,
            recipe={"kind": "solid", "color": list(c), "width": width, "height": height},
        )

    @classmethod
    def checker(
        cls,
        c1: tuple[int, int, int, int] = (230, 235, 245, 255),
        c2: tuple[int, int, int, int] = (45, 55, 75, 255),
        *,
        size: int = 128,
        cells: int = 8,
        bilinear: bool = True,
    ) -> Texture:
        yy, xx = np.indices((size, size))
        step = max(1, size // cells)
        mask = (((xx // step) + (yy // step)) & 1) == 0
        rgba = np.where(mask[:, :, None], np.array(c1, np.uint8), np.array(c2, np.uint8))
        return cls(
            rgba,
            bilinear=bilinear,
            recipe={"kind": "checker", "c1": list(c1), "c2": list(c2), "size": size, "cells": cells, "bilinear": bilinear},
        )

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any] | None:
        if self.recipe is None:
            return None
        d = dict(self.recipe)
        if d.get("kind") == "file" and base_dir is not None and "path" in d:
            try:
                d["path"] = str(Path(d["path"]).resolve().relative_to(base_dir.resolve()))
            except ValueError:
                pass
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Texture:
        kind = data["kind"]
        if kind == "file":
            p = Path(data["path"])
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            return cls.from_file(p, bilinear=bool(data.get("bilinear", True)))
        if kind == "solid":
            return cls.solid(
                tuple(int(x) for x in data["color"]),
                width=int(data.get("width", 4)),
                height=int(data.get("height", 4)),
            )
        if kind == "checker":
            return cls.checker(
                tuple(int(x) for x in data["c1"]),
                tuple(int(x) for x in data["c2"]),
                size=int(data.get("size", 128)),
                cells=int(data.get("cells", 8)),
                bilinear=bool(data.get("bilinear", True)),
            )
        if kind == "studio_skybox":
            from .material import Environment

            return Environment.studio_skybox(int(data.get("width", 512)), int(data.get("height", 256)))
        if kind == "daylight_skybox":
            from .material import Environment

            return Environment.daylight_skybox(int(data.get("width", 512)), int(data.get("height", 256)))
        raise ValueError(f"Unknown texture recipe: {kind!r}")


class RenderTarget(Texture):
    """Dynamic texture rendered live on the GPU every frame.

    It can be fed by:
    - a `camera` (`Camera3D`) looking at the current scene (or another `scene`),
    - a complete 2D / 3D `scene` (e.g. a 2D mini-game shown on a 3D screen),
    - an interactive `ui` (`UILayer`): with `interactive=True`, clicking or hovering the 3D surface
      that carries this `RenderTarget` automatically forwards the UV coordinates `(u*W, (1-v)*H)`!
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 256,
        *,
        scene: Any = None,
        camera: Any = None,
        ui: Any = None,
        clear_color: tuple[int, int, int, int] = (14, 18, 28, 255),
        interactive: bool = True,
        bilinear: bool = True,
        match_viewport_aspect: bool = False,
    ) -> None:
        placeholder = np.zeros((height, width), dtype=np.uint32)
        super().__init__(placeholder, bilinear=bilinear, recipe=None)
        self.width = int(width)
        self.height = int(height)
        self.scene = scene
        self.camera = camera
        self.ui = ui
        self.clear_color = clear_color
        self.interactive = bool(interactive)
        self.match_viewport_aspect = bool(match_viewport_aspect)

    @property
    def is_dynamic(self) -> bool:
        return True

    def set_gpu_frame(self, fb: gs.Array) -> None:
        self._gpu_array = fb
        self._cpu_pixels = None
        self.height, self.width = int(fb.shape[0]), int(fb.shape[1])
        self._version += 1

    def handle_uv_event(self, event: dict[str, Any], uv: tuple[float, float]) -> bool:
        """Translates a 3D mouse event at `uv=(u, v)` into pixel coordinates `(x, y)` of the surface."""
        if not self.interactive:
            return False
        u = float(uv[0]) % 1.0
        v = float(uv[1]) % 1.0
        sx = u * self.width
        sy = (1.0 - v) * self.height
        sub_ev = {**event, "x": sx, "y": sy}
        consumed = False
        if self.ui is not None:
            consumed = self.ui.handle_event(sub_ev, self.width, self.height)
        if self.scene is not None:
            if not consumed and self.scene.ui is not None:
                consumed = self.scene.ui.handle_event(sub_ev, self.width, self.height)
            if not consumed:
                self.scene.on_event(sub_ev)
        return consumed


class PlanarReflection:
    """Exact optical planar reflection (vertical mirror `z=plane_z` or horizontal floor/water `y=plane_y`).

    Wraps a mirror `Camera3D` and its `RenderTarget` matched to the viewport's aspect ratio:
    call `mirror.sync(scene.camera)` in `update()` and use `Material.mirror(mirror)` on the surface.
    """

    def __init__(
        self,
        *,
        plane_y: float | None = None,
        plane_z: float | None = None,
        width: int = 512,
        height: int = 288,
        match_viewport_aspect: bool = True,
    ) -> None:
        from .camera import Camera3D

        if plane_y is None and plane_z is None:
            plane_z = 0.0
        self.plane_y = plane_y
        self.plane_z = plane_z
        self.camera = Camera3D.fixed((0.0, 2.0, 0.0), (0.0, 2.0, 1.0), bloom=False, fxaa=True)
        self.target = RenderTarget(
            width,
            height,
            camera=self.camera,
            interactive=False,
            match_viewport_aspect=match_viewport_aspect,
        )

    def sync(self, main_cam: Any) -> None:
        """Synchronizes the mirror camera with the main camera."""
        p = np.asarray(main_cam.pos, dtype=np.float32).copy()
        t = np.asarray(main_cam.target, dtype=np.float32).copy()
        if self.plane_y is not None:
            p[1] = 2.0 * self.plane_y - p[1]
            t[1] = 2.0 * self.plane_y - t[1]
        if self.plane_z is not None:
            p[2] = 2.0 * self.plane_z - p[2]
            t[2] = 2.0 * self.plane_z - t[2]
        self.camera.pos = p
        self.camera.target = t
        self.camera.fov_deg = float(main_cam.fov_deg)


class TexturePool:
    """Bindless manager: gathers static textures (1 upload) and dynamic ones (zero-copy on the GPU).

    Each texture gets an integer identifier `tid >= 0` indexing `meta[tid]`:
    `[offset_or_x, y_or_0, width, flags_and_height]`
    - `meta[tid, 0]`: linear offset in `static_pool` (if static) or `ox` in `dyn_atlas` (if dynamic)
    - `meta[tid, 1]`: `0` (static) or `oy` in `dyn_atlas` (dynamic)
    - `meta[tid, 2]`: `width`
    - `meta[tid, 3]`: `height | (1 << 20 if dynamic) | (1 << 21 if bilinear)`
    """

    FLAG_DYNAMIC = 1 << 20
    FLAG_BILINEAR = 1 << 21

    def __init__(self) -> None:
        self._static_sig: Any = None
        self._static_dev: str | None = "__none__"
        self._static_pool_gpu: gs.Array | None = None
        self._dummy_pool_gpu: gs.Array | None = None
        self._dummy_dyn_gpu: gs.Array | None = None
        self._dummy_meta_gpu: gs.Array | None = None
        self._dummy_dev: str | None = "__none__"
        self._dyn_bg_cache: dict[tuple[int, int, str | None], gs.Array] = {}

    def _ensure_dummies(self, dev: str | None) -> None:
        if self._dummy_dev != dev or self._dummy_pool_gpu is None:
            self._dummy_pool_gpu = upload_u32(np.zeros((1,), dtype=np.uint32), device=dev)
            self._dummy_dyn_gpu = upload_u32(np.zeros((1, 1), dtype=np.uint32), device=dev)
            self._dummy_meta_gpu = upload_i32(np.zeros((1, 4), dtype=np.int32), device=dev)
            self._dummy_dev = dev

    def _get_dyn_bg(self, w: int, h: int, dev: str | None) -> gs.Array:
        key = (w, h, dev)
        arr = self._dyn_bg_cache.get(key)
        if arr is None:
            if len(self._dyn_bg_cache) > 6:
                self._dyn_bg_cache.clear()
            arr = upload_u32(np.zeros((h, w), dtype=np.uint32), device=dev)
            self._dyn_bg_cache[key] = arr
        return arr

    def build(
        self,
        textures: list[Texture],
        device: str | None,
    ) -> tuple[gs.Array, gs.Array, gs.Array]:
        """Returns `(static_pool_gpu, meta_gpu, dyn_atlas_gpu)` ready for `shade_pbr` or `render_2d`."""
        self._ensure_dummies(device)
        if not textures:
            assert self._dummy_pool_gpu is not None
            assert self._dummy_meta_gpu is not None
            assert self._dummy_dyn_gpu is not None
            return self._dummy_pool_gpu, self._dummy_meta_gpu, self._dummy_dyn_gpu

        # 1. Separate static textures (CPU) from dynamic ones (already resident in VRAM)
        static_items: list[tuple[int, Texture]] = []
        dyn_items: list[tuple[int, Texture]] = []
        for idx, tex in enumerate(textures):
            if tex.gpu_array is not None:
                dyn_items.append((idx, tex))
            else:
                static_items.append((idx, tex))

        meta = np.zeros((len(textures), 4), dtype=np.int32)

        # 2. Static pool (cached as long as the static textures / versions do not change)
        static_sig = tuple((id(t), t._version) for _, t in static_items)
        offset = 0
        for tid, tex in static_items:
            flags = tex.height | (self.FLAG_BILINEAR if tex.bilinear else 0)
            meta[tid] = (offset, 0, tex.width, flags)
            offset += tex.width * tex.height

        if not static_items:
            static_pool_gpu = self._dummy_pool_gpu
        elif static_sig == self._static_sig and self._static_dev == device and self._static_pool_gpu is not None:
            static_pool_gpu = self._static_pool_gpu
        else:
            chunks = [t.cpu_pixels.ravel() for _, t in static_items if t.cpu_pixels is not None]
            flat = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.uint32) if chunks else np.zeros((1,), dtype=np.uint32)
            static_pool_gpu = upload_u32(flat, device=device)
            self._static_pool_gpu = static_pool_gpu
            self._static_sig = static_sig
            self._static_dev = device

        # 3. Dynamic atlas, 100 % GPU (zero-copy with a single dynamic texture, GPU side-by-side blit if N > 1)
        if not dyn_items:
            dyn_atlas_gpu = self._dummy_dyn_gpu
        elif len(dyn_items) == 1:
            tid, tex = dyn_items[0]
            assert tex.gpu_array is not None
            flags = tex.height | self.FLAG_DYNAMIC | (self.FLAG_BILINEAR if tex.bilinear else 0)
            meta[tid] = (0, 0, tex.width, flags)
            dyn_atlas_gpu = tex.gpu_array
        else:
            total_w = sum(t.width for _, t in dyn_items)
            max_h = max(t.height for _, t in dyn_items)
            atlas = self._get_dyn_bg(total_w, max_h, device)
            cur_x = 0
            for tid, tex in dyn_items:
                assert tex.gpu_array is not None
                atlas = blit_viewport(atlas, tex.gpu_array, cur_x, 0, tex.width, tex.height, 0, device=device)
                flags = tex.height | self.FLAG_DYNAMIC | (self.FLAG_BILINEAR if tex.bilinear else 0)
                meta[tid] = (cur_x, 0, tex.width, flags)
                cur_x += tex.width
            dyn_atlas_gpu = atlas

        meta_gpu = upload_i32(meta, device=device)
        assert static_pool_gpu is not None and dyn_atlas_gpu is not None
        return static_pool_gpu, meta_gpu, dyn_atlas_gpu
