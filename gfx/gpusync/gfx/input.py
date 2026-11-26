"""Entrées unifiées (`Input`) et résultat de picking 3D exact (`PickResult`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PickResult:
    """Résultat d'un picking 3D exact depuis le Visibility Buffer."""

    object: Any  # Object3D
    triangle: int
    world_pos: np.ndarray  # float32 (3,)
    uv: tuple[float, float]
    camera: Any  # Camera3D


class Input:
    """État des entrées clavier / souris pour l'image courante, avec priorité UI et picking 3D."""

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.keys_pressed: set[str] = set()
        self.keys_released: set[str] = set()
        self.mouse_pos: tuple[float, float] = (0.0, 0.0)
        self.mouse_delta: tuple[float, float] = (0.0, 0.0)
        self.mouse_buttons: set[str] = set()
        self.mouse_pressed: set[str] = set()
        self.mouse_released: set[str] = set()
        self.mouse_wheel: tuple[float, float] = (0.0, 0.0)
        self.ui_captured: bool = False
        self._picker: Any = None

    def begin_frame(self) -> None:
        self.keys_pressed.clear()
        self.keys_released.clear()
        self.mouse_pressed.clear()
        self.mouse_released.clear()
        self.mouse_delta = (0.0, 0.0)
        self.mouse_wheel = (0.0, 0.0)
        self.ui_captured = False

    def feed(self, ev: dict[str, Any], *, consumed_by_ui: bool = False) -> None:
        t = ev.get("type")
        if consumed_by_ui:
            self.ui_captured = True
            return
        if t == "key_down":
            k = ev["key"]
            self.keys.add(k)
            self.keys_pressed.add(k)
        elif t == "key_up":
            k = ev["key"]
            self.keys.discard(k)
            self.keys_released.add(k)
        elif t == "mouse_move":
            self.mouse_pos = (float(ev["x"]), float(ev["y"]))
            self.mouse_delta = (self.mouse_delta[0] + float(ev["dx"]), self.mouse_delta[1] + float(ev["dy"]))
        elif t == "mouse_down":
            b = ev["button"]
            self.mouse_buttons.add(b)
            self.mouse_pressed.add(b)
        elif t == "mouse_up":
            b = ev["button"]
            self.mouse_buttons.discard(b)
            self.mouse_released.add(b)
        elif t == "mouse_wheel":
            self.mouse_wheel = (self.mouse_wheel[0] + float(ev["dx"]), self.mouse_wheel[1] + float(ev["dy"]))

    def key_down(self, key: str) -> bool:
        return key.lower() in self.keys

    def key_pressed(self, key: str) -> bool:
        return key.lower() in self.keys_pressed

    def mouse_down(self, button: str = "left") -> bool:
        return button in self.mouse_buttons

    def mouse_clicked(self, button: str = "left") -> bool:
        return button in self.mouse_pressed and not self.ui_captured

    def pick(self, x: float | None = None, y: float | None = None) -> PickResult | None:
        """Picking 3D exact à `(x, y)` (ou à la position souris courante) via le Visibility Buffer."""
        if self._picker is None:
            return None
        px = self.mouse_pos[0] if x is None else x
        py = self.mouse_pos[1] if y is None else y
        return self._picker(px, py)

    def clicked_object(self, button: str = "left") -> Any | None:
        """Renvoie l'`Object3D` cliqué sur cette image (ou `None`)."""
        if not self.mouse_clicked(button):
            return None
        hit = self.pick()
        return hit.object if hit is not None else None
