"""Objects of the pure 2D layer (`Rect2D`, `Circle2D`, `Line2D`, `Sprite2D`, `Particles2D`)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import gpusync as gs

from .texture import Texture
from .ui import pack_rgba_i32

SHAPE_RECT = 0
SHAPE_CIRCLE = 1
SHAPE_LINE = 2
SHAPE_SPRITE = 3


@dataclass
class Object2D:
    """Base class of the 2D primitives (sorted by `z`, shown through `Camera2D`)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visible: bool = True
    name: str = ""

    def pack(self, tex_id_fn) -> tuple[list[float], list[int]]:
        raise NotImplementedError


@dataclass
class Rect2D(Object2D):
    """Filled or outlined 2D rectangle, with optional rotation (radians)."""

    w: float = 100.0
    h: float = 60.0
    color: tuple[int, int, int, int] = (70, 175, 235, 255)
    border_color: tuple[int, int, int, int] | None = None
    border_width: float = 1.5
    rotation: float = 0.0

    def pack(self, tex_id_fn) -> tuple[list[float], list[int]]:
        # f32: [cx, cy, hw, hh, cos_r, sin_r, border_w, 0]
        c = math.cos(self.rotation)
        s = math.sin(self.rotation)
        f = [float(self.x), float(self.y), float(self.w) * 0.5, float(self.h) * 0.5, c, s, float(self.border_width), 0.0]
        i = [SHAPE_RECT, pack_rgba_i32(self.color), pack_rgba_i32(self.border_color), -1, 0, 0, 0, 0]
        return f, i


@dataclass
class Circle2D(Object2D):
    """Anti-aliased 2D disc with an optional border."""

    radius: float = 32.0
    color: tuple[int, int, int, int] = (240, 120, 75, 255)
    border_color: tuple[int, int, int, int] | None = None
    border_width: float = 1.5

    def pack(self, tex_id_fn) -> tuple[list[float], list[int]]:
        f = [float(self.x), float(self.y), float(self.radius), float(self.radius), 1.0, 0.0, float(self.border_width), 0.0]
        i = [SHAPE_CIRCLE, pack_rgba_i32(self.color), pack_rgba_i32(self.border_color), -1, 0, 0, 0, 0]
        return f, i


@dataclass
class Line2D(Object2D):
    """Anti-aliased 2D segment between `(x, y)` and `(x2, y2)`."""

    x2: float = 100.0
    y2: float = 100.0
    thickness: float = 2.5
    color: tuple[int, int, int, int] = (230, 240, 255, 255)

    def pack(self, tex_id_fn) -> tuple[list[float], list[int]]:
        f = [float(self.x), float(self.y), float(self.x2), float(self.y2), float(self.thickness) * 0.5, 0.0, 0.0, 0.0]
        i = [SHAPE_LINE, pack_rgba_i32(self.color), 0, -1, 0, 0, 0, 0]
        return f, i


@dataclass
class Sprite2D(Object2D):
    """Textured 2D sprite (supports a static `Texture`, a `RenderTarget` or a GPU `@gs.kernel` texture)."""

    w: float = 128.0
    h: float = 128.0
    texture: Texture | None = None
    tint: tuple[int, int, int, int] = (255, 255, 255, 255)
    rotation: float = 0.0

    def pack(self, tex_id_fn) -> tuple[list[float], list[int]]:
        tid = tex_id_fn(self.texture) if self.texture is not None else -1
        c = math.cos(self.rotation)
        s = math.sin(self.rotation)
        f = [float(self.x), float(self.y), float(self.w) * 0.5, float(self.h) * 0.5, c, s, 0.0, 0.0]
        i = [SHAPE_SPRITE, pack_rgba_i32(self.tint), 0, tid, 0, 0, 0, 0]
        return f, i


@dataclass
class Particles2D:
    """Particle cloud, 100 % GPU (positions and colours as `gs.Array` or NumPy, no CPU copy).

    The `px`, `py` (`float32[N]`) and `colors` (`uint32[N]` RGBA8) arrays can be updated
    every frame by a user `@gs.kernel` and rendered directly without ever going back through RAM!
    """

    px: np.ndarray | gs.Array
    py: np.ndarray | gs.Array
    colors: np.ndarray | gs.Array
    point_size: int = 2
    visible: bool = True
    name: str = "particles"
    meta: dict[str, Any] = field(default_factory=dict)
