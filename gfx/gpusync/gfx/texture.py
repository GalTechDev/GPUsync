"""Textures GPU unifiées (`Texture`, `RenderTarget`, `TexturePool`) pour 3D, 2D, vidéo et surfaces dynamiques.

Tout repose sur un buffer `u32[H, W]` (RGBA8 empaqueté) résident en VRAM :
- `Texture.from_file("wood.png")` / `Texture.from_array(rgba)` / `tex.update(frame)` (vidéo),
- `Texture.from_gpu(gs_array)` (sortie directe d'un `@gs.kernel`, 0-copie),
- `RenderTarget(w, h, camera=..., scene=..., ui=...)` : caméra 3D, scène 2D ou UI interactive
  rendue en direct sur n'importe quel matériau 3D (`albedo_map` / `emissive_map`) ou `Sprite2D`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import gpusync as gs

from .kernels.k3d import blit_viewport, upload_i32, upload_u32


def pack_rgba_u8_to_u32(arr: np.ndarray) -> np.ndarray:
    """Convertit un tableau `(H, W, 3|4)` uint8 ou `(H, W)` uint32 en `(H, W)` uint32 contigu RGBA8."""
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
    raise TypeError(f"Texture attend (H, W) uint32 ou (H, W, 3|4) uint8, reçu {a.shape} {a.dtype}")


class Texture:
    """Texture 2D RGBA8 (`u32[H, W]`) utilisable en 3D (`Material`) et en 2D (`Sprite2D`)."""

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
                raise TypeError(f"Texture GPU attend un gs.Array u32 2D, reçu {pixels.shape} {pixels.dtype}")
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
        """Met à jour la texture en place (ex. nouvelle frame vidéo, canvas CPU ou sortie `@gs.kernel`)."""
        if isinstance(pixels, gs.Array):
            if pixels.ndim != 2 or pixels.dtype != np.uint32:
                raise TypeError("Texture.update(gs.Array) attend un tableau u32 2D")
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
        """Enveloppe directement un `gs.Array` `u32[H, W]` produit par un `@gs.kernel` (0-copie)."""
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
        raise ValueError(f"Recette de texture inconnue : {kind!r}")


class RenderTarget(Texture):
    """Texture dynamique rendue en direct sur GPU chaque image.

    Peut être alimentée par :
    - une `camera` (`Camera3D`) regardant la scène courante (ou une autre `scene`),
    - une `scene` 2D / 3D complète (ex. mini-jeu 2D affiché sur un écran 3D),
    - une `ui` (`UILayer`) interactive : si `interactive=True`, cliquer ou survoler la surface 3D
      qui porte ce `RenderTarget` transmet automatiquement les coordonnées UV `(u*W, (1-v)*H)` !
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
        """Traduit un événement souris 3D à `uv=(u, v)` en coordonnées pixels `(x, y)` de la surface."""
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
    """Reflet planaire optique exact (miroir vertical `z=plane_z` ou sol/eau horizontal `y=plane_y`).

    Encapsule une `Camera3D` miroir et son `RenderTarget` calibré sur le ratio d'aspect du viewport :
    appelez `mirror.sync(scene.camera)` dans `update()` et utilisez `Material.mirror(mirror)` sur la surface.
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
        """Synchronise la caméra miroir avec la caméra principale."""
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
    """Gestionnaire bindless : regroupe les textures statiques (1 upload) et dynamiques (0-copie GPU).

    Chaque texture reçoit un identifiant entier `tid >= 0` indexant `meta[tid]` :
    `[offset_or_x, y_or_0, width, flags_and_height]`
    - `meta[tid, 0]` : offset linéaire dans `static_pool` (si statique) ou `ox` dans `dyn_atlas` (si dynamique)
    - `meta[tid, 1]` : `0` (statique) ou `oy` dans `dyn_atlas` (dynamique)
    - `meta[tid, 2]` : `width`
    - `meta[tid, 3]` : `height | (1 << 20 si dynamique) | (1 << 21 si bilinéaire)`
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
        """Renvoie `(static_pool_gpu, meta_gpu, dyn_atlas_gpu)` prêts pour `shade_pbr` ou `render_2d`."""
        self._ensure_dummies(device)
        if not textures:
            assert self._dummy_pool_gpu is not None
            assert self._dummy_meta_gpu is not None
            assert self._dummy_dyn_gpu is not None
            return self._dummy_pool_gpu, self._dummy_meta_gpu, self._dummy_dyn_gpu

        # 1. Séparer textures statiques (CPU) et dynamiques (déjà résidentes en VRAM)
        static_items: list[tuple[int, Texture]] = []
        dyn_items: list[tuple[int, Texture]] = []
        for idx, tex in enumerate(textures):
            if tex.gpu_array is not None:
                dyn_items.append((idx, tex))
            else:
                static_items.append((idx, tex))

        meta = np.zeros((len(textures), 4), dtype=np.int32)

        # 2. Pool statique (mis en cache tant que les textures/versions statiques ne changent pas)
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

        # 3. Atlas dynamique 100 % GPU (0-copie si 1 seule texture dynamique, blit GPU côte à côte si N > 1)
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
