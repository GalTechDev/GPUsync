"""Pluggable GPU post-processing effects (`camera.add_effect(...)`) written in pure `@gs.kernel`.

Each effect can be:
- an instance of a built-in effect (`ChromaticAberration`, `Scanlines`, `ColorGrade`, `Pixelate`),
- or directly a user `@gs.kernel` `fn(fb: gs.u32["h", "w"]) -> gs.u32["h", "w"]`
  (or an HDR effect `stage="hdr"` taking `(hr, hg, hb)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import gpusync as gs


@gs.kernel
def _chromatic_aberration_k(src: gs.u32["h", "w"], strength: gs.f32) -> gs.u32["h", "w"]:
    py, px = gs.index()
    h = src.shape[0]
    w = src.shape[1]
    ux = (float(px) + 0.5) / float(w) - 0.5
    uy = (float(py) + 0.5) / float(h) - 0.5
    rad = gs.sqrt(ux * ux + uy * uy)
    dx = int(ux * rad * strength * 10.0)
    rx = gs.min(gs.max(px + dx, 0), w - 1)
    bx = gs.min(gs.max(px - dx, 0), w - 1)

    mask = gs.u32(255)
    cr = src[py, rx] & mask
    cg = (src[py, px] >> gs.u32(8)) & mask
    cb = (src[py, bx] >> gs.u32(16)) & mask
    return cr | (cg << gs.u32(8)) | (cb << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def _scanlines_k(src: gs.u32["h", "w"], intensity: gs.f32, pitch: gs.i32) -> gs.u32["h", "w"]:
    py, px = gs.index()
    c = src[py, px]
    if (py % gs.max(pitch, 2)) != 0:
        return c
    mask = gs.u32(255)
    k = 1.0 - intensity
    r = int(float(c & mask) * k)
    g = int(float((c >> gs.u32(8)) & mask) * k)
    b = int(float((c >> gs.u32(16)) & mask) * k)
    return gs.u32(r) | (gs.u32(g) << gs.u32(8)) | (gs.u32(b) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def _color_grade_k(
    src: gs.u32["h", "w"],
    saturation: gs.f32,
    contrast: gs.f32,
    tr: gs.f32,
    tg: gs.f32,
    tb: gs.f32,
) -> gs.u32["h", "w"]:
    py, px = gs.index()
    c = src[py, px]
    mask = gs.u32(255)
    r = float(c & mask) / 255.0
    g = float((c >> gs.u32(8)) & mask) / 255.0
    b = float((c >> gs.u32(16)) & mask) / 255.0
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r = lum + (r - lum) * saturation
    g = lum + (g - lum) * saturation
    b = lum + (b - lum) * saturation
    r = ((r - 0.5) * contrast + 0.5) * tr
    g = ((g - 0.5) * contrast + 0.5) * tg
    b = ((b - 0.5) * contrast + 0.5) * tb
    ri = int(gs.min(gs.max(r, 0.0), 1.0) * 255.0 + 0.5)
    gi = int(gs.min(gs.max(g, 0.0), 1.0) * 255.0 + 0.5)
    bi = int(gs.min(gs.max(b, 0.0), 1.0) * 255.0 + 0.5)
    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def _pixelate_k(src: gs.u32["h", "w"], block: gs.i32) -> gs.u32["h", "w"]:
    py, px = gs.index()
    b = gs.max(block, 1)
    sy = (py // b) * b
    sx = (px // b) * b
    return src[sy, sx]


class PostEffect:
    """Base class of a GPU post-processing effect (`stage='ldr'` or `'hdr'`)."""

    stage: str = "ldr"
    enabled: bool = True

    def apply(self, fb, *, device=None, out=None):
        raise NotImplementedError


@dataclass
class ChromaticAberration(PostEffect):
    """Radial chromatic aberration at the edges of the screen."""

    strength: float = 2.5
    enabled: bool = True

    def apply(self, fb, *, device=None, out=None):
        return _chromatic_aberration_k(fb, float(self.strength), device=device, out=out)


@dataclass
class Scanlines(PostEffect):
    """Retro CRT scanlines."""

    intensity: float = 0.28
    pitch: int = 3
    enabled: bool = True

    def apply(self, fb, *, device=None, out=None):
        return _scanlines_k(fb, float(self.intensity), int(self.pitch), device=device, out=out)


@dataclass
class ColorGrade(PostEffect):
    """Colour grading (saturation, contrast, RGB tint)."""

    saturation: float = 1.15
    contrast: float = 1.08
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled: bool = True

    def apply(self, fb, *, device=None, out=None):
        return _color_grade_k(
            fb,
            float(self.saturation),
            float(self.contrast),
            float(self.tint[0]),
            float(self.tint[1]),
            float(self.tint[2]),
            device=device,
            out=out,
        )


@dataclass
class Pixelate(PostEffect):
    """Pixel-art effect with square blocks."""

    block_size: int = 4
    enabled: bool = True

    def apply(self, fb, *, device=None, out=None):
        return _pixelate_k(fb, int(self.block_size), device=device, out=out)
