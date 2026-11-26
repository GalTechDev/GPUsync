"""100 % GPU UI layer (`UILayer`, `Panel`, `Label`, `Button`, `ProgressBar`, `FPSOverlay`).

Every widget is packed into two small GPU buffers (`boxes` i32 and `glyphs` u32), cached
in VRAM as long as nothing changes, then rendered in a single `ui_composite` dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kernels.k3d import upload_i32, upload_u32

# 5×6 bitmap font (6 rows × 5 bits → 30 bits packed into a u32)
_FONT_ROWS: dict[str, tuple[str, ...]] = {
    " ": (".....", ".....", ".....", ".....", ".....", "....."),
    "0": (".###.", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "...#.", "..#..", ".#...", "#####"),
    "3": ("####.", "....#", ".###.", "....#", "#...#", ".###."),
    "4": ("#..#.", "#..#.", "#####", "...#.", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "#...#", ".###."),
    "6": (".###.", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#..."),
    "8": (".###.", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#..##", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "###..", "#..#.", "#...#", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "####.", "#..#.", "#...#", "#...#"),
    "S": (".####", "#....", ".###.", "....#", "#...#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "...#.", "..#..", ".#...", "#....", "#####"),
    ":": (".....", "..#..", ".....", ".....", "..#..", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", "..#.."),
    ",": (".....", ".....", ".....", ".....", "..#..", ".#..."),
    "-": (".....", ".....", "#####", ".....", ".....", "....."),
    "+": (".....", "..#..", "#####", "..#..", ".....", "....."),
    "=": (".....", "#####", ".....", "#####", ".....", "....."),
    "/": ("....#", "...#.", "..#..", ".#...", "#....", "....."),
    "|": ("..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "(": ("...#.", "..#..", "..#..", "..#..", "..#..", "...#."),
    ")": (".#...", "..#..", "..#..", "..#..", "..#..", ".#..."),
    "[": (".###.", ".#...", ".#...", ".#...", ".#...", ".###."),
    "]": (".###.", "...#.", "...#.", "...#.", "...#.", ".###."),
    "%": ("##..#", "##.#.", "..#..", ".#.##", "#..##", "....."),
    "*": (".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#"),
    "!": ("..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "..##.", "..#..", ".....", "..#.."),
    "_": (".....", ".....", ".....", ".....", ".....", "#####"),
    "#": (".#.#.", "#####", ".#.#.", "#####", ".#.#.", "....."),
    "<": ("...#.", "..#..", ".#...", "..#..", "...#.", "....."),
    ">": (".#...", "..#..", "...#.", "..#..", ".#...", "....."),
}


def _pack_glyph(rows: tuple[str, ...]) -> int:
    bits = 0
    for y, line in enumerate(rows):
        for x, ch in enumerate(line):
            if ch == "#":
                bits |= 1 << (y * 5 + x)
    return bits


_GLYPHS: dict[str, int] = {k: _pack_glyph(v) for k, v in _FONT_ROWS.items()}


def _normalize_text(text: str) -> str:
    return (
        text.upper()
        .replace("É", "E")
        .replace("È", "E")
        .replace("Ê", "E")
        .replace("À", "A")
        .replace("Â", "A")
        .replace("Ô", "O")
        .replace("Û", "U")
        .replace("Ù", "U")
        .replace("Ç", "C")
        .replace("×", "X")
        .replace("·", "|")
        .replace("→", ">")
    )


def pack_rgba_i32(color: tuple[int, int, int] | tuple[int, int, int, int] | None) -> int:
    """Packs `(r, g, b[, a])` into the bit pattern of an `int32` for `ui_composite`."""
    if color is None:
        return 0
    r = max(0, min(255, int(color[0])))
    g = max(0, min(255, int(color[1])))
    b = max(0, min(255, int(color[2])))
    a = max(0, min(255, int(color[3]))) if len(color) > 3 else 255
    u = r | (g << 8) | (b << 16) | (a << 24)
    return int(np.uint32(u).view(np.int32))


def _resolve_rect(x: int, y: int, w: int, h: int, anchor: str, fb_w: int, fb_h: int) -> tuple[int, int, int, int]:
    if anchor == "top_right":
        ax = fb_w - w - x
        ay = y
    elif anchor == "bottom_left":
        ax = x
        ay = fb_h - h - y
    elif anchor == "bottom_right":
        ax = fb_w - w - x
        ay = fb_h - h - y
    elif anchor == "center":
        ax = (fb_w - w) // 2 + x
        ay = (fb_h - h) // 2 + y
    else:  # "top_left"
        ax = x
        ay = y
    return ax, ay, ax + w, ay + h


@dataclass
class Widget:
    x: int = 10
    y: int = 10
    w: int = 160
    h: int = 36
    anchor: str = "top_left"
    visible: bool = True

    def rect(self, fb_w: int, fb_h: int) -> tuple[int, int, int, int]:
        return _resolve_rect(self.x, self.y, self.w, self.h, self.anchor, fb_w, fb_h)

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        raise NotImplementedError


@dataclass
class Panel(Widget):
    """Translucent rectangular panel with an optional border."""

    bg: tuple[int, int, int, int] = (12, 16, 26, 190)
    border: tuple[int, int, int, int] | None = (55, 135, 185, 255)

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        x0, y0, x1, y1 = self.rect(fb_w, fb_h)
        boxes.append([x0, y0, x1, y1, pack_rgba_i32(self.bg), pack_rgba_i32(self.border), -1, 0, 0, 0, 0, 0])


@dataclass
class Label(Widget):
    """GPU bitmap text (with optional background / border)."""

    text: str = ""
    color: tuple[int, int, int, int] = (238, 246, 255, 255)
    scale: int = 2
    bg: tuple[int, int, int, int] | None = None
    border: tuple[int, int, int, int] | None = None
    pad_x: int = 8
    pad_y: int = 6

    def rect(self, fb_w: int, fb_h: int) -> tuple[int, int, int, int]:
        lines = self.text.splitlines() or [""]
        max_cols = max(len(l) for l in lines)
        w = max(self.w, max_cols * 6 * self.scale + self.pad_x * 2) if self.bg else max_cols * 6 * self.scale + self.pad_x * 2
        h = len(lines) * 8 * self.scale + self.pad_y * 2
        return _resolve_rect(self.x, self.y, w, h, self.anchor, fb_w, fb_h)

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        lines = self.text.splitlines() or [""]
        x0, y0, x1, y1 = self.rect(fb_w, fb_h)
        if self.bg is not None or self.border is not None:
            boxes.append([x0, y0, x1, y1, pack_rgba_i32(self.bg), pack_rgba_i32(self.border), -1, 0, 0, 0, 0, 0])
        tc = pack_rgba_i32(self.color)
        for li, line in enumerate(lines):
            if not line:
                continue
            row_idx = len(texts)
            texts.append(line)
            tx = x0 + self.pad_x
            ty = y0 + self.pad_y + li * 8 * self.scale
            boxes.append([x0, y0, x1, y1, 0, 0, row_idx, len(line), self.scale, tx, ty, tc])


@dataclass
class Button(Widget):
    """Interactive GPU button with hover / click states and an `on_click()` callback."""

    text: str = "BUTTON"
    on_click: Callable[[], Any] | None = None
    scale: int = 2
    bg: tuple[int, int, int, int] = (24, 34, 52, 215)
    hover_bg: tuple[int, int, int, int] = (38, 66, 98, 235)
    pressed_bg: tuple[int, int, int, int] = (55, 115, 165, 245)
    border: tuple[int, int, int, int] = (80, 175, 225, 255)
    text_color: tuple[int, int, int, int] = (240, 248, 255, 255)
    hovered: bool = field(default=False, init=False, repr=False)
    pressed: bool = field(default=False, init=False, repr=False)

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        x0, y0, x1, y1 = self.rect(fb_w, fb_h)
        cur_bg = self.pressed_bg if self.pressed else (self.hover_bg if self.hovered else self.bg)
        row_idx = len(texts)
        texts.append(self.text)
        tw = len(self.text) * 6 * self.scale
        th = 6 * self.scale
        tx = x0 + max(4, (x1 - x0 - tw) // 2)
        ty = y0 + max(2, (y1 - y0 - th) // 2)
        boxes.append(
            [
                x0,
                y0,
                x1,
                y1,
                pack_rgba_i32(cur_bg),
                pack_rgba_i32(self.border),
                row_idx,
                len(self.text),
                self.scale,
                tx,
                ty,
                pack_rgba_i32(self.text_color),
            ]
        )


@dataclass
class ProgressBar(Widget):
    """Progress / health bar, 100 % GPU (2 boxes: background + fill)."""

    value: float = 0.5  # [0.0, 1.0]
    bg: tuple[int, int, int, int] = (14, 18, 28, 200)
    fill_color: tuple[int, int, int, int] = (65, 195, 240, 240)
    border: tuple[int, int, int, int] = (70, 145, 190, 255)
    label: str = ""
    scale: int = 2
    text_color: tuple[int, int, int, int] = (240, 248, 255, 255)

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        x0, y0, x1, y1 = self.rect(fb_w, fb_h)
        boxes.append([x0, y0, x1, y1, pack_rgba_i32(self.bg), pack_rgba_i32(self.border), -1, 0, 0, 0, 0, 0])
        v = max(0.0, min(1.0, float(self.value)))
        fx1 = x0 + 2 + int(round((x1 - x0 - 4) * v))
        if fx1 > x0 + 2:
            boxes.append([x0 + 2, y0 + 2, fx1, y1 - 2, pack_rgba_i32(self.fill_color), 0, -1, 0, 0, 0, 0, 0])
        if self.label:
            row_idx = len(texts)
            texts.append(self.label)
            tw = len(self.label) * 6 * self.scale
            th = 6 * self.scale
            tx = x0 + max(4, (x1 - x0 - tw) // 2)
            ty = y0 + max(2, (y1 - y0 - th) // 2)
            boxes.append(
                [x0, y0, x1, y1, 0, 0, row_idx, len(self.label), self.scale, tx, ty, pack_rgba_i32(self.text_color)]
            )


class FPSOverlay(Widget):
    """Ready-made HUD panel showing GPU FPS, screen FPS and scene info."""

    def __init__(self, x: int = 10, y: int = 10, scale: int = 2, anchor: str = "top_left") -> None:
        super().__init__(x=x, y=y, w=0, h=0, anchor=anchor, visible=True)
        self.scale = scale
        self.line1 = "GPUSYNC GFX"
        self.line2 = "INITIALISATION..."

    def set_lines(self, line1: str, line2: str = "") -> None:
        self.line1 = line1
        self.line2 = line2

    def emit(self, fb_w: int, fb_h: int, boxes: list[list[int]], texts: list[str]) -> None:
        lines = [self.line1] + ([self.line2] if self.line2 else [])
        max_cols = max(len(l) for l in lines)
        w = max_cols * 6 * self.scale + 20
        h = len(lines) * 8 * self.scale + 16
        x0, y0, x1, y1 = _resolve_rect(self.x, self.y, w, h, self.anchor, fb_w, fb_h)
        boxes.append(
            [
                x0,
                y0,
                x1,
                y1,
                pack_rgba_i32((10, 14, 24, 190)),
                pack_rgba_i32((55, 135, 185, 255)),
                -1,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        colors = [pack_rgba_i32((240, 246, 255, 255)), pack_rgba_i32((125, 225, 255, 255))]
        for li, line in enumerate(lines):
            row_idx = len(texts)
            texts.append(line)
            boxes.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    0,
                    0,
                    row_idx,
                    len(line),
                    self.scale,
                    x0 + 10,
                    y0 + 8 + li * 8 * self.scale,
                    colors[min(li, len(colors) - 1)],
                ]
            )


class UILayer:
    """Screen widget manager: intercepts mouse clicks + cached GPU compilation."""

    def __init__(self) -> None:
        self.widgets: list[Widget] = []
        self.visible: bool = True
        self._cache_sig: Any = None
        self._cache_dev: str | None = "__none__"
        self._boxes_gpu: Any = None
        self._glyphs_gpu: Any = None
        self._n_boxes: int = 0

    def add(self, widget: Widget) -> Widget:
        self.widgets.append(widget)
        self.invalidate()
        return widget

    def remove(self, widget: Widget) -> None:
        if widget in self.widgets:
            self.widgets.remove(widget)
            self.invalidate()

    def clear(self) -> None:
        self.widgets.clear()
        self.invalidate()

    def invalidate(self) -> None:
        self._cache_sig = None

    def has_visible(self) -> bool:
        return self.visible and any(w.visible for w in self.widgets)

    def handle_event(self, event: dict[str, Any], fb_w: int, fb_h: int) -> bool:
        """Handles mouse events on the UI buttons. Returns `True` if the UI consumes the event."""
        if not self.visible:
            return False
        t = event.get("type")
        if t == "mouse_move":
            mx, my = float(event["x"]), float(event["y"])
            for w in self.widgets:
                if isinstance(w, Button) and w.visible:
                    x0, y0, x1, y1 = w.rect(fb_w, fb_h)
                    inside = x0 <= mx < x1 and y0 <= my < y1
                    if w.hovered != inside:
                        w.hovered = inside
                        self.invalidate()
            return False
        if t == "mouse_down" and event.get("button") == "left":
            mx, my = float(event.get("x", -1)), float(event.get("y", -1))
            # Test from the last widget drawn (on top) down to the first
            for w in reversed(self.widgets):
                if isinstance(w, Button) and w.visible:
                    x0, y0, x1, y1 = w.rect(fb_w, fb_h)
                    if x0 <= mx < x1 and y0 <= my < y1:
                        w.pressed = True
                        w.hovered = True
                        self.invalidate()
                        return True
            return False
        if t == "mouse_up" and event.get("button") == "left":
            mx, my = float(event.get("x", -1)), float(event.get("y", -1))
            consumed = False
            for w in reversed(self.widgets):
                if isinstance(w, Button) and w.visible and w.pressed:
                    w.pressed = False
                    self.invalidate()
                    x0, y0, x1, y1 = w.rect(fb_w, fb_h)
                    if x0 <= mx < x1 and y0 <= my < y1:
                        consumed = True
                        if w.on_click is not None:
                            w.on_click()
            return consumed
        return False

    def compile(self, fb_w: int, fb_h: int, device: str | None) -> tuple[Any, Any, int] | None:
        """Builds and caches the GPU buffers `(boxes_gpu, glyphs_gpu, n_boxes)`."""
        if not self.has_visible():
            return None

        boxes_list: list[list[int]] = []
        texts: list[str] = []
        for w in self.widgets:
            if w.visible:
                w.emit(fb_w, fb_h, boxes_list, texts)
        if not boxes_list:
            return None

        sig = (fb_w, fb_h, tuple(tuple(b) for b in boxes_list), tuple(texts))
        if sig == self._cache_sig and self._cache_dev == device and self._boxes_gpu is not None:
            return self._boxes_gpu, self._glyphs_gpu, self._n_boxes

        boxes_np = np.asarray(boxes_list, dtype=np.int32)
        if texts:
            max_cols = max(1, max(len(t) for t in texts))
            glyphs_np = np.zeros((len(texts), max_cols), dtype=np.uint32)
            for ri, text in enumerate(texts):
                norm = _normalize_text(text)
                for ci, ch in enumerate(norm):
                    glyphs_np[ri, ci] = _GLYPHS.get(ch, 0)
        else:
            glyphs_np = np.zeros((1, 1), dtype=np.uint32)

        self._boxes_gpu = upload_i32(boxes_np, device=device)
        self._glyphs_gpu = upload_u32(glyphs_np, device=device)
        self._n_boxes = len(boxes_list)
        self._cache_sig = sig
        self._cache_dev = device
        return self._boxes_gpu, self._glyphs_gpu, self._n_boxes
