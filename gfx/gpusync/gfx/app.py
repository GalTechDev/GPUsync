"""Application temps réel (`App`) : fenêtre native `gs.Window` zéro-copie, multi-scènes et mode headless."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import gpusync as gs

from .camera import Camera3D
from .input import Input, PickResult
from .renderer3d import Renderer3D
from .scene import Scene
from .texture import RenderTarget
from .ui import FPSOverlay


class App:
    """Boucle d'application `gpusync.gfx` (interactive sur `gs.Window` ou headless pour rendu/tests/ML).

    Exemple :
        app = gfx.App(1024, 576, title="Ma démo")
        app.run(MaScene())
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 576,
        *,
        title: str = "GPUsync GFX",
        vsync: bool = False,
        device: str | None = None,
        shadow_size: int = 512,
        camera_controls: bool = True,
        show_fps: bool = True,
        render_scale: float = 1.0,
    ) -> None:
        self.title = title
        self.vsync = vsync
        self.camera_controls = camera_controls
        self.show_fps = show_fps
        self.renderer = Renderer3D(width, height, shadow_size=shadow_size, device=device, render_scale=render_scale)
        self.input = Input()
        self.scene: Scene | None = None
        self.window: gs.Window | None = None
        self._running = False
        self._fps_overlay: FPSOverlay | None = None
        self.fps_gpu: float = 0.0
        self.gpu_ms: float = 0.0
        self.fps_display: float = 0.0

    @property
    def width(self) -> int:
        return self.renderer.width

    @property
    def height(self) -> int:
        return self.renderer.height

    @property
    def device(self) -> str:
        return self.renderer.active_device

    # ── Gestion multi-scènes ──

    def set_scene(self, scene: Scene) -> Scene:
        """Bascule vers une nouvelle `Scene` (décharge l'ancienne et libère son cache VRAM)."""
        if self.scene is not None and self.scene is not scene:
            self.scene.unload()
            self.scene.app = None
        self.renderer.invalidate_scene()
        self.scene = scene
        scene._ensure_loaded(self)
        if self.show_fps and not any(isinstance(w, FPSOverlay) for w in scene.ui.widgets):
            self._fps_overlay = FPSOverlay()
            scene.ui.add(self._fps_overlay)
        else:
            self._fps_overlay = next((w for w in scene.ui.widgets if isinstance(w, FPSOverlay)), None)
        self.input._picker = lambda x, y: self.renderer.pick(scene, x, y)
        return scene

    def quit(self) -> None:
        self._running = False
        if self.window is not None:
            self.window.close()

    # ── Picking & captures (interactif ou headless) ──

    def pick(self, x: float, y: float, scene: Scene | None = None) -> PickResult | None:
        sc = scene or self.scene
        if sc is None:
            return None
        return self.renderer.pick(sc, x, y)

    def dispatch_surface_event(self, ev: dict[str, Any], scene: Scene | None = None) -> bool:
        """Transmet un événement souris vers une surface 3D interactive (écran UI/2D dans le monde)."""
        sc = scene or self.scene
        if sc is None or not sc._has_interactive_rt:
            return False
        et = ev.get("type")
        if et not in ("mouse_move", "mouse_down", "mouse_up"):
            return False
        mx = float(ev.get("x", self.input.mouse_pos[0]))
        my = float(ev.get("y", self.input.mouse_pos[1]))
        hit = self.renderer.pick(sc, mx, my)
        if hit is None:
            return False
        mat = hit.object.material
        for tex in (mat.emissive_map, mat.albedo_map):
            if isinstance(tex, RenderTarget) and tex.interactive:
                if tex.handle_uv_event(ev, hit.uv):
                    return True
        return False

    def render_frame(self, scene: Scene | None = None, *, profile: bool = False, host_out: bool = False) -> gs.Array:
        """Rend une image de `scene` sur GPU (fonctionne sans fenêtre, mode headless)."""
        sc = scene if scene is not None else self.scene
        if sc is None:
            raise RuntimeError("Aucune scène active : passez `scene` ou appelez `app.set_scene(...)`.")
        if sc is not self.scene:
            self.set_scene(sc)
        return self.renderer.render_frame(sc, profile=profile, host_out=host_out)

    def render_rgba(self, scene: Scene | None = None, *, profile: bool = False) -> np.ndarray:
        """Rend une image et renvoie une copie NumPy `(H, W, 4)` uint8 (mode headless inclus)."""
        sc = scene if scene is not None else self.scene
        if sc is None:
            raise RuntimeError("Aucune scène active : passez `scene` ou appelez `app.set_scene(...)`.")
        if sc is not self.scene:
            self.set_scene(sc)
        return self.renderer.render_rgba(sc, profile=profile).copy()

    def screenshot(self, path: str | Path, scene: Scene | None = None) -> Path:
        """Enregistre une image PNG de la scène (nécessite Pillow)."""
        from PIL import Image

        p = Path(path)
        rgba = self.render_rgba(scene)
        Image.fromarray(rgba[:, :, :3], mode="RGB").save(p)
        return p

    def benchmark(self, scene: Scene, *, frames: int = 120, dt: float = 1.0 / 60.0) -> dict[str, Any]:
        """Exécute un benchmark GPU pur en continu (sans attente écran) et renvoie les stats."""
        self.set_scene(scene)
        # Échauffement
        self.renderer.render_frame(scene)
        gs.sync(self.renderer.device)

        acc: dict[str, float] = {}
        for _ in range(5):
            scene.update(dt)
            self.renderer.render_frame(scene, profile=True)
            for k, v in self.renderer.last_timings_ms.items():
                acc[k] = acc.get(k, 0.0) + v / 5.0

        t0 = time.perf_counter()
        fb = None
        for _ in range(frames):
            scene.update(dt)
            fb = self.renderer.render_frame(scene)
        gs.sync(self.renderer.device)
        elapsed = time.perf_counter() - t0
        return {
            "fps": frames / max(elapsed, 1e-6),
            "ms_per_frame": (elapsed / frames) * 1000.0,
            "device": fb.device if fb is not None else self.device,
            "width": self.width,
            "height": self.height,
            "triangles": int(scene.tri_idx.shape[0]),
            "passes_ms": acc,
        }

    # ── Contrôleur de caméra par défaut (orbit & free-fly) ──

    def _apply_camera_controls(self, cam: Camera3D, dt: float) -> None:
        inp = self.input
        if cam.mode == "orbit":
            dx, dy = inp.mouse_delta
            if inp.mouse_down("left") and not inp.ui_captured and (dx != 0.0 or dy != 0.0):
                cam.yaw -= dx * 0.006
                cam.pitch = max(-1.25, min(1.35, cam.pitch + dy * 0.006))
                cam.update_orbit()
            elif inp.mouse_down("right") and not inp.ui_captured and (dx != 0.0 or dy != 0.0):
                fwd = cam.target - cam.pos
                fwd[1] = 0.0
                fn = float(np.linalg.norm(fwd))
                if fn > 1e-6:
                    fwd /= fn
                right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                pan = (-right * dx + np.array([0.0, 1.0, 0.0], dtype=np.float32) * dy) * (cam.distance * 0.0018)
                cam.target = cam.target + pan
                cam.update_orbit()
            wy = inp.mouse_wheel[1]
            if wy != 0.0 and not inp.ui_captured:
                cam.distance = max(1.5, min(40.0, cam.distance * (1.0 - 0.08 * wy)))
                cam.update_orbit()
            if inp.keys:
                speed = 5.0 * dt
                fwd = cam.target - cam.pos
                fwd[1] = 0.0
                fn = float(np.linalg.norm(fwd))
                if fn > 1e-6:
                    fwd /= fn
                right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                move = np.zeros(3, dtype=np.float32)
                if inp.key_down("w") or inp.key_down("z") or inp.key_down("up"):
                    move += fwd * speed
                if inp.key_down("s") or inp.key_down("down"):
                    move -= fwd * speed
                if inp.key_down("d") or inp.key_down("right"):
                    move += right * speed
                if inp.key_down("a") or inp.key_down("q") or inp.key_down("left"):
                    move -= right * speed
                if inp.key_down("e"):
                    move[1] += speed
                if inp.key_down("c"):
                    move[1] -= speed
                if np.any(move != 0):
                    cam.target = cam.target + move
                    cam.update_orbit()
        elif cam.mode == "free":
            dx, dy = inp.mouse_delta
            if inp.mouse_down("left") and not inp.ui_captured and (dx != 0.0 or dy != 0.0):
                cam.yaw -= dx * 0.004
                cam.pitch = max(-1.45, min(1.45, cam.pitch - dy * 0.004))
                cam.update_free_fly()
            if inp.keys:
                fwd, right, up = cam.basis()
                speed = 6.0 * dt
                move = np.zeros(3, dtype=np.float32)
                if inp.key_down("w") or inp.key_down("z") or inp.key_down("up"):
                    move += fwd * speed
                if inp.key_down("s") or inp.key_down("down"):
                    move -= fwd * speed
                if inp.key_down("d") or inp.key_down("right"):
                    move += right * speed
                if inp.key_down("a") or inp.key_down("q") or inp.key_down("left"):
                    move -= right * speed
                if inp.key_down("e"):
                    move += up * speed
                if inp.key_down("c"):
                    move -= up * speed
                if np.any(move != 0):
                    cam.pos = cam.pos + move
                    cam.update_free_fly()

    # ── Boucle interactive native `gs.Window` ──

    def run(self, scene: Scene) -> None:
        """Ouvre la fenêtre native `gs.Window` et lance la boucle temps réel zéro-copie."""
        req_dev = self.renderer.device
        win_dev = req_dev if req_dev and req_dev.split(":")[0] in ("metal", "vulkan", "dx12") else None

        with gs.Window(self.width, self.height, title=self.title, vsync=self.vsync, device=win_dev) as win:
            self.window = win
            if self.renderer.device is None:
                self.renderer.device = win.device
            self.set_scene(scene)
            self._running = True

            # Échauffement GPU avant le chrono
            fb0 = self.renderer.render_frame(self.scene)
            win.present(fb0)

            last_t = time.perf_counter()
            stats_next = 0.0
            last_zc = win.zero_copy

            try:
                while win.open and self._running:
                    now = time.perf_counter()
                    dt = min(now - last_t, 0.1)
                    last_t = now

                    cur_scene = self.scene
                    assert cur_scene is not None
                    self.input.begin_frame()

                    win_w, win_h = win.size
                    sx_in = self.width / max(win_w, 1)
                    sy_in = self.height / max(win_h, 1)
                    for ev in win.poll():
                        et = ev["type"]
                        if et == "resize":
                            self.renderer.resize(ev["width"], ev["height"])
                            cur_scene.ui.invalidate()
                            win_w, win_h = max(ev["width"], 1), max(ev["height"], 1)
                            sx_in = self.width / win_w
                            sy_in = self.height / win_h
                        elif et == "key_down" and ev["key"] == "escape":
                            self.quit()
                            break
                        elif et == "mouse_move":
                            ev = {
                                **ev,
                                "x": float(ev["x"]) * sx_in,
                                "y": float(ev["y"]) * sy_in,
                                "dx": float(ev["dx"]) * sx_in,
                                "dy": float(ev["dy"]) * sy_in,
                            }
                        elif et in ("mouse_down", "mouse_up") and "x" not in ev:
                            ev = {**ev, "x": self.input.mouse_pos[0], "y": self.input.mouse_pos[1]}
                        consumed = cur_scene.ui.handle_event(ev, self.width, self.height)
                        if not consumed and cur_scene._has_interactive_rt and et in ("mouse_down", "mouse_up"):
                            consumed = self.dispatch_surface_event(ev, cur_scene)
                        self.input.feed(ev, consumed_by_ui=consumed)
                        if not consumed:
                            cur_scene.on_event(ev)

                    if not win.open or not self._running:
                        break

                    # Peut avoir changé de scène pendant on_event / clic bouton UI
                    cur_scene = self.scene
                    assert cur_scene is not None
                    if self.camera_controls and cur_scene.cameras and cur_scene.objects:
                        self._apply_camera_controls(cur_scene.camera, dt)
                    cur_scene.update(dt)
                    # Mettre à jour les sous-scènes éventuelles rendues dans des RenderTargets
                    for rt in cur_scene._render_targets:
                        if rt.scene is not None and rt.scene is not cur_scene:
                            rt.scene._ensure_loaded(self)
                            rt.scene.update(dt)

                    cur_scene = self.scene
                    assert cur_scene is not None

                    if now >= stats_next:
                        # Quatre images d'affilée : une seule, isolée, mesure surtout le réveil du GPU
                        gs.sync(self.renderer.device)
                        t_g0 = time.perf_counter()
                        for _ in range(4):
                            fb = self.renderer.render_frame(cur_scene)
                        gs.sync(self.renderer.device)
                        g_ms = (time.perf_counter() - t_g0) * 250.0
                        self.gpu_ms = g_ms if self.gpu_ms == 0.0 else self.gpu_ms * 0.7 + g_ms * 0.3
                        self.fps_gpu = 1000.0 / max(self.gpu_ms, 0.05)
                        win.present(fb)
                        last_zc = win.zero_copy
                        stats_next = now + 0.25

                        zc = "0-COPIE" if last_zc else "UPLOAD"
                        if self._fps_overlay is not None:
                            self._fps_overlay.set_lines(
                                f"GPU {self.fps_gpu:5.0f} FPS ({self.gpu_ms:4.2f} MS) | ECRAN {self.fps_display:4.0f} FPS | {self.renderer.active_device} ({zc})",
                                f"{self.width}X{self.height} | {cur_scene.tri_idx.shape[0]} TRIS | CAMS:{len(cur_scene.active_cameras())}",
                            )
                            cur_scene.ui.invalidate()
                        win.set_title(
                            f"{self.title} | GPU {self.fps_gpu:5.0f} FPS ({self.gpu_ms:4.2f} ms) · "
                            f"Écran {self.fps_display:4.0f} FPS | {self.renderer.active_device}"
                        )
                    else:
                        fb = self.renderer.render_frame(cur_scene)
                        win.present(fb)
                        last_zc = win.zero_copy

                    inst = 1.0 / max(dt, 1e-5)
                    self.fps_display = inst if self.fps_display == 0.0 else self.fps_display * 0.92 + inst * 0.08
            finally:
                if self.scene is not None:
                    self.scene.unload()
                self.window = None
