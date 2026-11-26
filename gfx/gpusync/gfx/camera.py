"""Cameras (`Camera3D`, `Camera2D`) and normalized display areas (`Viewport`)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .math3d import cross3, mat4_look_at, mat4_ortho, mat4_perspective, normalize


@dataclass
class Viewport:
    """Normalized rectangle `[0, 1] × [0, 1]` on the screen (origin at the top left)."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    @property
    def is_full(self) -> bool:
        return (
            abs(self.x) < 1e-5
            and abs(self.y) < 1e-5
            and abs(self.w - 1.0) < 1e-5
            and abs(self.h - 1.0) < 1e-5
        )

    def pixel_rect(self, fb_w: int, fb_h: int) -> tuple[int, int, int, int]:
        """Converts the viewport into `(px, py, pw, ph)` in the framebuffer `(fb_w, fb_h)`."""
        px = max(0, min(fb_w - 1, int(round(self.x * fb_w))))
        py = max(0, min(fb_h - 1, int(round(self.y * fb_h))))
        pw = max(16, min(fb_w - px, int(round(self.w * fb_w))))
        ph = max(16, min(fb_h - py, int(round(self.h * fb_h))))
        return px, py, pw, ph

    def contains_pixel(self, x: float, y: float, fb_w: int, fb_h: int) -> bool:
        px, py, pw, ph = self.pixel_rect(fb_w, fb_h)
        return px <= x < px + pw and py <= y < py + ph

    def to_dict(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]

    @classmethod
    def from_any(cls, v: Viewport | tuple[float, float, float, float] | list[float]) -> Viewport:
        if isinstance(v, Viewport):
            return v
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


@dataclass
class Camera3D:
    """3D camera (perspective or orthographic, orbit / free-fly / fixed control, viewport & post-processing)."""

    name: str = "main"
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 3.4, 8.8], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.5, 0.0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fov_deg: float = 52.0
    near: float = 0.2
    far: float = 60.0
    ortho: bool = False
    ortho_size: float = 8.0

    # Built-in controller
    mode: str = "orbit"  # "orbit" | "free" | "fixed"
    yaw: float = 0.35
    pitch: float = 0.30
    distance: float = 9.2

    # Multiple cameras / screen display
    viewport: Viewport = field(default_factory=Viewport)
    order: int = 0
    enabled: bool = True
    border_color: tuple[int, int, int] | None = None

    # Rendering options per camera
    shadows: bool = True
    bloom: bool = True
    bloom_threshold: float = 0.95
    bloom_strength: float = 0.65
    exposure: float = 1.15
    fxaa: bool = True
    wireframe: bool = False
    ssr: bool = False
    ssr_strength: float = 0.65
    ssr_steps: int = 24
    effects: list[Any] = field(default_factory=list)

    def add_effect(self, effect: Any, *, stage: str | None = None) -> Any:
        """Adds a post-processing effect (`PostEffect` or user `@gs.kernel`) to this camera."""
        if stage is not None and not hasattr(effect, "stage"):
            self.effects.append((stage, effect))
        else:
            self.effects.append(effect)
        return effect

    def clear_effects(self) -> None:
        self.effects.clear()

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.up = np.asarray(self.up, dtype=np.float32)
        if not isinstance(self.viewport, Viewport):
            self.viewport = Viewport.from_any(self.viewport)

    @classmethod
    def orbit(
        cls,
        target: tuple[float, float, float] = (0.0, 1.5, 0.0),
        *,
        distance: float = 9.2,
        yaw: float = 0.35,
        pitch: float = 0.30,
        fov_deg: float = 52.0,
        viewport: Viewport | tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        order: int = 0,
        name: str = "main",
        **kwargs: Any,
    ) -> Camera3D:
        cam = cls(
            name=name,
            target=np.asarray(target, dtype=np.float32),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            fov_deg=fov_deg,
            mode="orbit",
            viewport=Viewport.from_any(viewport),
            order=order,
            **kwargs,
        )
        cam.update_orbit()
        return cam

    @classmethod
    def free_fly(
        cls,
        pos: tuple[float, float, float] = (0.0, 2.0, 6.0),
        *,
        yaw: float = 0.0,
        pitch: float = -0.15,
        fov_deg: float = 60.0,
        viewport: Viewport | tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        order: int = 0,
        name: str = "free",
        **kwargs: Any,
    ) -> Camera3D:
        cam = cls(
            name=name,
            pos=np.asarray(pos, dtype=np.float32),
            yaw=yaw,
            pitch=pitch,
            fov_deg=fov_deg,
            mode="free",
            viewport=Viewport.from_any(viewport),
            order=order,
            **kwargs,
        )
        cam.update_free_fly()
        return cam

    @classmethod
    def fixed(
        cls,
        pos: tuple[float, float, float],
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        fov_deg: float = 52.0,
        viewport: Viewport | tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        order: int = 0,
        name: str = "fixed",
        **kwargs: Any,
    ) -> Camera3D:
        return cls(
            name=name,
            pos=np.asarray(pos, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            fov_deg=fov_deg,
            mode="fixed",
            viewport=Viewport.from_any(viewport),
            order=order,
            **kwargs,
        )

    @classmethod
    def top_down(
        cls,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        height: float = 16.0,
        fov_deg: float = 45.0,
        viewport: Viewport | tuple[float, float, float, float] = (0.74, 0.02, 0.24, 0.28),
        order: int = 1,
        name: str = "minimap",
        border_color: tuple[int, int, int] | None = (70, 175, 230),
        **kwargs: Any,
    ) -> Camera3D:
        t = np.asarray(target, dtype=np.float32)
        return cls(
            name=name,
            pos=np.array([t[0], t[1] + height, t[2] + 0.01], dtype=np.float32),
            target=t,
            fov_deg=fov_deg,
            mode="fixed",
            viewport=Viewport.from_any(viewport),
            order=order,
            border_color=border_color,
            bloom=kwargs.pop("bloom", False),
            fxaa=kwargs.pop("fxaa", False),
            **kwargs,
        )

    def update_orbit(self) -> None:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        offset = np.array([self.distance * cp * sy, self.distance * sp, self.distance * cp * cy], dtype=np.float32)
        self.pos = self.target + offset

    def update_free_fly(self) -> None:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        fwd = np.array([-cp * sy, sp, -cp * cy], dtype=np.float32)
        self.target = self.pos + fwd

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fwd = normalize(self.target - self.pos)
        up_ref = self.up
        side = cross3(fwd, up_ref)
        if math.hypot(*side.tolist()) < 1e-5:
            side = cross3(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        right = normalize(side)
        up = cross3(right, fwd)
        return fwd, right, up

    def view_matrix(self) -> np.ndarray:
        return mat4_look_at(self.pos, self.target, self.up)

    def proj_matrix(self, aspect: float) -> np.ndarray:
        if self.ortho:
            h = self.ortho_size
            w = h * max(aspect, 1e-4)
            return mat4_ortho(-w, w, -h, h, self.near, self.far)
        return mat4_perspective(self.fov_deg, aspect, self.near, self.far)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pos": [float(x) for x in self.pos],
            "target": [float(x) for x in self.target],
            "up": [float(x) for x in self.up],
            "fov_deg": float(self.fov_deg),
            "near": float(self.near),
            "far": float(self.far),
            "ortho": bool(self.ortho),
            "ortho_size": float(self.ortho_size),
            "mode": self.mode,
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "distance": float(self.distance),
            "viewport": self.viewport.to_dict(),
            "order": int(self.order),
            "enabled": bool(self.enabled),
            "border_color": list(self.border_color) if self.border_color is not None else None,
            "shadows": bool(self.shadows),
            "bloom": bool(self.bloom),
            "bloom_threshold": float(self.bloom_threshold),
            "bloom_strength": float(self.bloom_strength),
            "exposure": float(self.exposure),
            "fxaa": bool(self.fxaa),
            "wireframe": bool(self.wireframe),
            "ssr": bool(self.ssr),
            "ssr_strength": float(self.ssr_strength),
            "ssr_steps": int(self.ssr_steps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera3D:
        bc = data.get("border_color")
        cam = cls(
            name=data.get("name", "main"),
            pos=np.asarray(data.get("pos", [0.0, 3.4, 8.8]), dtype=np.float32),
            target=np.asarray(data.get("target", [0.0, 1.5, 0.0]), dtype=np.float32),
            up=np.asarray(data.get("up", [0.0, 1.0, 0.0]), dtype=np.float32),
            fov_deg=float(data.get("fov_deg", 52.0)),
            near=float(data.get("near", 0.2)),
            far=float(data.get("far", 60.0)),
            ortho=bool(data.get("ortho", False)),
            ortho_size=float(data.get("ortho_size", 8.0)),
            mode=str(data.get("mode", "orbit")),
            yaw=float(data.get("yaw", 0.35)),
            pitch=float(data.get("pitch", 0.30)),
            distance=float(data.get("distance", 9.2)),
            viewport=Viewport.from_any(data.get("viewport", [0.0, 0.0, 1.0, 1.0])),
            order=int(data.get("order", 0)),
            enabled=bool(data.get("enabled", True)),
            border_color=tuple(int(c) for c in bc) if bc is not None else None,
            shadows=bool(data.get("shadows", True)),
            bloom=bool(data.get("bloom", True)),
            bloom_threshold=float(data.get("bloom_threshold", 0.95)),
            bloom_strength=float(data.get("bloom_strength", 0.65)),
            exposure=float(data.get("exposure", 1.15)),
            fxaa=bool(data.get("fxaa", True)),
            wireframe=bool(data.get("wireframe", False)),
            ssr=bool(data.get("ssr", False)),
            ssr_strength=float(data.get("ssr_strength", 0.65)),
            ssr_steps=int(data.get("ssr_steps", 24)),
        )
        return cam


@dataclass
class Camera2D:
    """2D camera (pan, zoom, rotation, viewport) — ready for the dedicated 2D layer (milestone 2)."""

    name: str = "cam2d"
    pos: tuple[float, float] = (0.0, 0.0)
    zoom: float = 1.0
    rotation: float = 0.0
    viewport: Viewport = field(default_factory=Viewport)
    order: int = 10
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "zoom": float(self.zoom),
            "rotation": float(self.rotation),
            "viewport": self.viewport.to_dict(),
            "order": int(self.order),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera2D:
        p = data.get("pos", [0.0, 0.0])
        return cls(
            name=data.get("name", "cam2d"),
            pos=(float(p[0]), float(p[1])),
            zoom=float(data.get("zoom", 1.0)),
            rotation=float(data.get("rotation", 0.0)),
            viewport=Viewport.from_any(data.get("viewport", [0.0, 0.0, 1.0, 1.0])),
            order=int(data.get("order", 10)),
            enabled=bool(data.get("enabled", True)),
        )
