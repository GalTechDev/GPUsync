"""Tests of the `gpusync.gfx` graphics framework (milestone 1):
- `Scene` life cycle (`load`, `update`, `unload`, multiple scenes with `app.set_scene`),
- `Node` / `Object3D` hierarchy (world matrices, inherited visibility),
- `.scene.json` export / import (primitives, `.obj`, materials, `PointLight`, `SpotLight`, cameras / viewports),
- Multiple cameras (split-screen `Viewport` + minimap + ortho projection),
- `SpotLight` (lit cone vs outside the cone),
- Exact `O(1)` 3D picking (`object`, `world_pos`, `uv`) + multi-viewport priority,
- GPU `UILayer` (`Panel`, `Label`, `Button`, `ProgressBar`, click & event consumption),
- GPU / CPU parity of the RGBA rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpusync as gs
import gpusync.gfx as gfx


def _gpu_device() -> str | None:
    return next((d.id for d in gs.devices() if d.backend != "cpu"), None)


class SimpleScene(gfx.Scene):
    def __init__(self) -> None:
        super().__init__()
        self.events_log: list[str] = []

    def load(self) -> None:
        self.events_log.append("load")
        self.camera = gfx.Camera3D.orbit(target=(0.0, 0.5, 0.0), distance=5.0, yaw=0.0, pitch=0.15)
        self.floor = self.add(gfx.Object3D("floor", gfx.Mesh.plane(10.0, subdivisions=6), gfx.Material.checker()))
        self.rig = self.add(gfx.Node("rig", pos=(0.0, 1.0, 0.0)))
        self.box = self.add(
            gfx.Object3D("box", gfx.Mesh.cube(1.2, sub=2), gfx.Material.solid((0.9, 0.3, 0.2), roughness=0.3)),
            parent=self.rig,
        )
        self.add_light(gfx.PointLight(pos=(2.0, 3.0, 2.0), color=(1.0, 0.9, 0.8), intensity=14.0))

    def update(self, dt: float) -> None:
        self.events_log.append("update")
        self.rig.rot_y += dt

    def unload(self) -> None:
        self.events_log.append("unload")


def test_scene_lifecycle_and_switching():
    app = gfx.App(256, 144, device="cpu:0", show_fps=False)
    s1 = SimpleScene()
    s2 = SimpleScene()
    app.set_scene(s1)
    assert s1.events_log == ["load"]
    assert s1.get("box") is s1.box
    assert s1.box.parent is s1.rig

    app.set_scene(s2)
    assert "unload" in s1.events_log
    assert s2.events_log == ["load"]


def test_hierarchy_transforms_and_visibility():
    sc = gfx.Scene()
    parent = sc.add(gfx.Node("parent", pos=(2.0, 0.0, 0.0), rot=(0.0, np.pi / 2, 0.0)))
    child = sc.add(gfx.Object3D("child", gfx.Mesh.cube(1.0), pos=(0.0, 0.0, 3.0)), parent=parent)
    models, _, _, _ = sc.build_world_transforms()
    m = models[0].reshape(4, 4)
    # A +90° rotation around Y sends (0, 0, 3) to (3, 0, 0), then a +2 translation → (5, 0, 0)
    np.testing.assert_allclose(m[:3, 3], [5.0, 0.0, 0.0], atol=1e-4)

    # Hiding the parent zeroes the child's 3×3 block (triangles ignored without recompiling)
    parent.visible = False
    models2, _, _, _ = sc.build_world_transforms()
    np.testing.assert_allclose(models2[0].reshape(4, 4)[:3, :3], 0.0, atol=1e-6)


def test_scene_json_roundtrip(tmp_path):
    # Mini .obj to also check the relative serialization of external assets
    obj_path = tmp_path / "tri.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    sc = SimpleScene()
    sc._ensure_loaded()
    sc.add(gfx.Object3D("custom_obj", gfx.Mesh.from_obj(obj_path), gfx.Material.glow((0.2, 0.9, 0.6)), pos=(1, 1, 0)))
    spot = gfx.SpotLight(pos=(0.0, 4.0, 2.0), color=(0.4, 0.8, 1.0), intensity=20.0, inner_deg=12.0, outer_deg=25.0)
    spot.look_at((0.0, 0.5, 0.0))
    sc.add_light(spot)
    sc.add_camera(gfx.Camera3D.top_down(viewport=(0.75, 0.0, 0.25, 0.25), order=1))

    scene_file = sc.save(tmp_path / "level1.scene.json")
    loaded = gfx.Scene.from_file(scene_file)

    assert len(loaded.objects) == len(sc.objects)
    assert loaded.get("box") is not None
    assert loaded.get("box").parent is loaded.get("rig")
    assert loaded.get("custom_obj") is not None
    assert len(loaded.cameras) == 2
    assert loaded.cameras[1].viewport.w == pytest.approx(0.25)
    assert len(loaded.lights) == 2
    assert isinstance(loaded.lights[1], gfx.SpotLight)
    assert loaded.lights[1].outer_deg == pytest.approx(25.0)

    # Hybrid class with `source = ...`
    class LoadedLevel(gfx.Scene):
        source = scene_file

        def load(self) -> None:
            self.hero = self.get("box")

    lvl = LoadedLevel()
    lvl._ensure_loaded()
    assert lvl.hero is not None and lvl.hero.name == "box"


def test_picking_3d_and_uv():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0))
    target_box = sc.add(
        gfx.Object3D("target", gfx.Mesh.cube(2.0, sub=2), gfx.Material.solid((0.8, 0.4, 0.2)), pos=(0.0, 0.0, 0.0))
    )
    app.render_frame(sc)

    # Screen centre: must hit the front face (z = +1.0) of `target`
    hit = app.pick(app.width // 2, app.height // 2)
    assert hit is not None
    assert hit.object is target_box
    assert hit.world_pos[2] == pytest.approx(1.0, abs=0.1)
    assert 0.0 <= hit.uv[0] <= 1.0 and 0.0 <= hit.uv[1] <= 1.0

    # Top-left corner: empty sky
    assert app.pick(2, 2) is None


def test_multi_camera_split_screen_and_ortho():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(
        pos=(0.0, 1.0, 5.0), target=(0.0, 0.5, 0.0), viewport=(0.0, 0.0, 0.5, 1.0), order=0, bloom=False, fxaa=False
    )
    sc.add_camera(
        gfx.Camera3D(
            name="ortho_right",
            pos=np.array([0.0, 8.0, 0.01], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ortho=True,
            ortho_size=3.0,
            viewport=(0.5, 0.0, 0.5, 1.0),
            order=1,
            border_color=(255, 80, 40),
            bloom=False,
            fxaa=False,
        )
    )
    sc.add(gfx.Object3D("cube", gfx.Mesh.cube(1.8, sub=2), gfx.Material.solid((0.2, 0.8, 0.9)), pos=(0, 0.5, 0)))

    rgba = app.render_rgba(sc)
    assert rgba.shape == (app.height, app.width, 4)
    # Coloured border on the left edge of the right viewport (x = app.width // 2)
    bx = app.width // 2
    assert tuple(int(c) for c in rgba[app.height // 2, bx, :3]) == (255, 80, 40)

    # Picking in each half returns the right camera!
    hit_l = app.pick(app.width // 4, app.height // 2)
    hit_r = app.pick((3 * app.width) // 4, app.height // 2)
    assert hit_l is not None and hit_l.camera is sc.cameras[0]
    assert hit_r is not None and hit_r.camera is sc.cameras[1]


def test_spotlight_illuminates_cone():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.env.sun.intensity = 0.0
    sc.env.fog_density = 0.0
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 6.0, 6.0), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)
    sc.add(gfx.Object3D("floor", gfx.Mesh.plane(12.0, subdivisions=12), gfx.Material.solid((0.8, 0.8, 0.8), roughness=0.6)))
    dark = app.render_rgba(sc).copy()

    spot = gfx.SpotLight(pos=(0.0, 4.0, 0.0), direction=(0.0, -1.0, 0.0), color=(1.0, 0.9, 0.6), intensity=45.0, inner_deg=15.0, outer_deg=25.0)
    sc.add_light(spot)
    lit = app.render_rgba(sc)

    # The centre under the spot must be clearly brighter than without the spot
    cy, cx = app.height // 2, app.width // 2
    assert float(lit[cy, cx, :3].mean()) > float(dark[cy, cx, :3].mean()) + 30.0


def test_ui_layer_rendering_and_button_click():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = SimpleScene()
    clicked = []
    btn = sc.ui.add(
        gfx.Button(
            x=12,
            y=12,
            w=110,
            h=32,
            text="GO",
            bg=(200, 30, 30, 255),
            border=(255, 255, 255, 255),
            on_click=lambda: clicked.append(True),
        )
    )
    sc.ui.add(gfx.ProgressBar(x=12, y=54, w=110, h=18, value=0.75))

    rgba = app.render_rgba(sc)
    # White border of the button at (12, 12) and bright red background at (15, 15)
    assert tuple(int(c) for c in rgba[12, 12, :3]) == (255, 255, 255)
    assert rgba[15, 15, 0] > 180 and rgba[15, 15, 1] < 60

    # Simulated click in the button: must trigger on_click AND consume the event
    assert sc.ui.handle_event({"type": "mouse_move", "x": 30, "y": 25}, app.width, app.height) is False
    assert btn.hovered is True
    assert sc.ui.handle_event({"type": "mouse_down", "button": "left", "x": 30, "y": 25}, app.width, app.height) is True
    assert sc.ui.handle_event({"type": "mouse_up", "button": "left", "x": 30, "y": 25}, app.width, app.height) is True
    assert clicked == [True]


def test_gpu_matches_cpu_reference():
    gpu = _gpu_device()
    if gpu is None:
        pytest.skip("no GPU available")
    sc = SimpleScene()
    sc._ensure_loaded()
    sc.camera.bloom = True
    sc.camera.fxaa = True

    app_gpu = gfx.App(256, 144, device=gpu, shadow_size=128, show_fps=False)
    app_cpu = gfx.App(256, 144, device="cpu:0", shadow_size=128, show_fps=False)

    img_gpu = app_gpu.render_rgba(sc)
    img_cpu = app_cpu.render_rgba(sc)
    diff = np.abs(img_gpu.astype(np.int16) - img_cpu.astype(np.int16))
    assert float(diff.mean()) < 2.5
    assert float((diff > 10).mean()) < 0.02


# ───────────────────────── Milestone 2 tests: pure 2D, textures, RenderTarget & UI on 3D ─────────────────────────


def test_pure_2d_scene_shapes_sprites_and_particles():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.clear_color = (10, 12, 20, 255)
    sc.camera_2d = gfx.Camera2D(pos=(0.0, 0.0), zoom=1.0)
    sc.add(gfx.Rect2D(0.0, 0.0, 80.0, 60.0, color=(220, 40, 40, 255)))
    sc.add(gfx.Circle2D(-70.0, 0.0, radius=20.0, color=(40, 220, 80, 255)))
    sc.add(gfx.Line2D(-100.0, -50.0, 100.0, -50.0, thickness=4.0, color=(60, 140, 255, 255)))
    sc.add(gfx.Sprite2D(80.0, 0.0, 36.0, 36.0, texture=gfx.Texture.solid((240, 220, 40, 255))))

    px = np.array([0.0, 10.0], dtype=np.float32)
    py = np.array([45.0, 45.0], dtype=np.float32)
    col = np.array([0xFFFFFFFF, 0xFFFF00FF], dtype=np.uint32)
    sc.add(gfx.Particles2D(px, py, col, point_size=3))

    rgba = app.render_rgba(sc)
    cy, cx = app.height // 2, app.width // 2
    # Centre : rectangle rouge
    assert rgba[cy, cx, 0] > 200 and rgba[cy, cx, 1] < 60
    # Gauche : cercle vert
    assert rgba[cy, cx - 70, 1] > 200 and rgba[cy, cx - 70, 0] < 60
    # Droite : sprite jaune
    assert rgba[cy, cx + 80, 0] > 210 and rgba[cy, cx + 80, 1] > 190 and rgba[cy, cx + 80, 2] < 60
    # Bas centre (y = cy + 45) : particule blanche
    assert rgba[cy + 45, cx, 0] == 255 and rgba[cy + 45, cx, 1] == 255 and rgba[cy + 45, cx, 2] == 255


def test_pbr_textures_live_update_and_json_roundtrip(tmp_path):
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)

    tex = gfx.Texture.solid((240, 40, 40, 255), width=8, height=8)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 2.2), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)
    sc.add(gfx.Object3D("panel", gfx.Mesh.quad(2.0, 2.0), gfx.Material.screen(tex, brightness=1.0)))

    rgba_red = app.render_rgba(sc).copy()
    cy, cx = app.height // 2, app.width // 2
    assert rgba_red[cy, cx, 0] > 180 and rgba_red[cy, cx, 0] > int(rgba_red[cy, cx, 2]) + 100

    # Dynamic in-place update (like a video frame) → turns blue!
    blue_frame = np.zeros((8, 8, 4), dtype=np.uint8)
    blue_frame[:, :] = (30, 50, 245, 255)
    tex.update(blue_frame)
    rgba_blue = app.render_rgba(sc)
    assert rgba_blue[cy, cx, 2] > 180 and rgba_blue[cy, cx, 2] > int(rgba_blue[cy, cx, 0]) + 100

    # `.scene.json` export / import with a texture recipe
    sc.objects[0].material.emissive_map = gfx.Texture.checker(size=32, cells=4)
    p = sc.save(tmp_path / "textured.scene.json")
    sc2 = gfx.Scene.from_file(p)
    assert sc2.objects[0].material.emissive_map is not None
    assert sc2.objects[0].material.emissive_map.width == 32


def test_render_target_2d_and_camera_on_3d_surfaces():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)

    # Green 2D scene shown on a 3D quad on the left
    sub2d = gfx.Scene()
    sub2d.add(gfx.Rect2D(0.0, 0.0, 200.0, 200.0, color=(30, 235, 70, 255)))
    rt_2d = gfx.RenderTarget(128, 128, scene=sub2d, interactive=False)

    # 3D camera looking at a red sphere, shown on a 3D quad on the right
    cam_sec = gfx.Camera3D.fixed(pos=(0.0, 0.0, 6.0), target=(0.0, 0.0, 9.0), bloom=False, fxaa=False, shadows=False)
    rt_cam = gfx.RenderTarget(128, 128, camera=cam_sec, interactive=False)

    sc = gfx.Scene()
    sc.env.sun.intensity = 0.0
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 3.2), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)
    sc.add(gfx.Object3D("scr_2d", gfx.Mesh.quad(1.6, 1.6), gfx.Material.screen(rt_2d, brightness=1.0), pos=(-1.1, 0.0, 0.0)))
    sc.add(gfx.Object3D("scr_cam", gfx.Mesh.quad(1.6, 1.6), gfx.Material.screen(rt_cam, brightness=1.0), pos=(1.1, 0.0, 0.0)))
    sc.add(gfx.Object3D("hidden_red", gfx.Mesh.sphere(1.6, rings=12, sectors=16), gfx.Material.glow((1.0, 0.05, 0.05), 1.4, roughness=0.9), pos=(0.0, 0.0, 9.0)))

    rgba = app.render_rgba(sc)
    cy = app.height // 2
    # Left screen: green (2D scene)
    left_px = rgba[cy, app.width // 4]
    assert left_px[1] > 160 and left_px[1] > int(left_px[0]) + 80
    # Right screen: red (3D camera filming the red sphere behind the main camera!)
    right_px = rgba[cy, (3 * app.width) // 4]
    assert right_px[0] > 160 and right_px[0] > int(right_px[1]) + 80


def test_interactive_ui_on_3d_surface_click():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)

    clicked = []
    surface_ui = gfx.UILayer()
    # Button covering the whole centre of the 200×200 RenderTarget
    surface_ui.add(gfx.Button(20, 20, 160, 160, text="OK", bg=(40, 120, 240, 255), on_click=lambda: clicked.append("3d_ui")))
    rt = gfx.RenderTarget(200, 200, ui=surface_ui, interactive=True)

    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 2.2), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)
    sc.add(gfx.Object3D("terminal", gfx.Mesh.quad(2.0, 2.0), gfx.Material.screen(rt, brightness=1.0)))

    app.render_frame(sc)
    cx, cy = app.width // 2, app.height // 2
    assert app.dispatch_surface_event({"type": "mouse_down", "button": "left", "x": cx, "y": cy}, sc) is True
    assert app.dispatch_surface_event({"type": "mouse_up", "button": "left", "x": cx, "y": cy}, sc) is True
    assert clicked == ["3d_ui"]


# ───────────────────────── Milestone 3 tests: compute↔render, GPU instancing, post-effects & glTF/GLB ─────────────────────────


@gs.kernel
def _shift_verts_up(base: gs.f32["nv", "c3"], dy: gs.f32) -> gs.f32["nv", "c3"]:
    v, c = gs.index()
    val = base[v, c]
    if c == 1:
        val = val + dy
    return val


@gs.kernel
def _invert_fb(src: gs.u32["h", "w"]) -> gs.u32["h", "w"]:
    py, px = gs.index()
    c = src[py, px]
    m = gs.u32(255)
    r = gs.u32(255) - (c & m)
    g = gs.u32(255) - ((c >> gs.u32(8)) & m)
    b = gs.u32(255) - ((c >> gs.u32(16)) & m)
    return r | (g << gs.u32(8)) | (b << gs.u32(16)) | gs.u32(4278190080)


def test_dynamic_gpu_vertices_zero_copy():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 4.5), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)
    cube = sc.add(gfx.Object3D("dyn", gfx.Mesh.cube(1.4, sub=1), gfx.Material.glow((0.2, 0.9, 1.0), 1.0)))

    before = app.render_rgba(sc).copy()
    # Move the vertices directly in VRAM to +1.5 in Y without touching the CPU!
    base_gpu = cube.base_positions_gpu(device=dev)
    cube.gpu_positions = _shift_verts_up(base_gpu, 1.6, device=dev)
    after = app.render_rgba(sc)

    # The centre was bright (cube) and becomes dark sky again after moving up
    cy, cx = app.height // 2, app.width // 2
    assert float(before[cy, cx, :3].mean()) > float(after[cy, cx, :3].mean()) + 30.0
    assert float(after[cy - 35, cx, :3].mean()) > float(before[cy - 35, cx, :3].mean()) + 30.0


def test_gpu_instanced_mesh_trs():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed(pos=(0.0, 0.0, 6.0), target=(0.0, 0.0, 0.0), bloom=False, fxaa=False, shadows=False)

    # 2 instances placed left (-2, 0, 0) and right (+2, 0, 0) through gpu_trs
    trs = np.zeros((2, 8), dtype=np.float32)
    trs[0] = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 1.0]
    trs[1] = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 1.0]
    inst = sc.add(
        gfx.InstancedMesh3D("pair", gfx.Mesh.cube(1.0, sub=1), gfx.Material.glow((1.0, 0.5, 0.1), 1.2), count=2, trs=trs)
    )
    inst.gpu_trs = inst.base_trs_gpu(device=dev)

    rgba = app.render_rgba(sc)
    cy = app.height // 2
    # Left and right contain the 2 orange instances, the centre is empty (sky)
    assert rgba[cy, app.width // 4, 0] > 180 and rgba[cy, app.width // 4, 0] > int(rgba[cy, app.width // 4, 2]) + 60
    assert rgba[cy, (3 * app.width) // 4, 0] > 180 and rgba[cy, (3 * app.width) // 4, 0] > int(rgba[cy, (3 * app.width) // 4, 2]) + 60
    assert app.pick(app.width // 2, cy) is None
    assert app.pick(app.width // 4, cy) is not None
    assert app.pick((3 * app.width) // 4, cy) is not None


def test_pluggable_post_effects_builtin_and_custom_kernel():
    dev = _gpu_device() or "cpu:0"
    app = gfx.App(256, 144, device=dev, show_fps=False)
    sc = SimpleScene()
    sc._ensure_loaded()
    sc.camera.bloom = False
    sc.camera.fxaa = False
    base = app.render_rgba(sc).copy()

    # Built-in Scanlines + ChromaticAberration effect
    sc.camera.add_effect(gfx.ChromaticAberration(strength=2.0))
    sc.camera.add_effect(gfx.Scanlines(intensity=0.5, pitch=2))
    styled = app.render_rgba(sc).copy()
    assert not np.array_equal(base, styled)
    # Even rows must be clearly darkened by Scanlines
    assert float(styled[0::2, :, :3].mean()) < float(styled[1::2, :, :3].mean()) * 0.75

    # Direct custom @gs.kernel effect (colour inversion)
    sc.camera.clear_effects()
    sc.camera.add_effect(_invert_fb)
    inv = app.render_rgba(sc)
    np.testing.assert_allclose(inv[:, :, :3].astype(np.int16), 255 - base[:, :, :3].astype(np.int16), atol=1)


def test_gltf_glb_roundtrip_mesh_and_scene_import(tmp_path):
    mesh_orig = gfx.Mesh.crystal(radius=0.8, height=1.8, sides=6)
    mat_orig = gfx.Material.solid((0.2, 0.75, 0.9), metallic=0.8, roughness=0.25)
    glb_path = gfx.write_glb_from_mesh(tmp_path / "crystal.glb", mesh_orig, mat_orig)

    # 1) Direct loading as a Mesh + .scene.json recipe
    loaded_mesh = gfx.Mesh.from_gltf(glb_path)
    assert loaded_mesh.num_vertices == mesh_orig.num_vertices
    assert loaded_mesh.num_triangles == mesh_orig.num_triangles
    np.testing.assert_allclose(loaded_mesh.positions, mesh_orig.positions, atol=1e-5)

    # 2) Full import as nodes + PBR materials into a Scene
    sc = gfx.Scene()
    nodes = sc.import_gltf(glb_path)
    assert len(nodes) == 1
    obj = sc.objects[0]
    assert obj.material.metallic == pytest.approx(0.8, abs=1e-4)
    assert obj.material.albedo[2] == pytest.approx(0.9, abs=1e-4)

    # 3) `.scene.json` serialization keeps the relative glTF recipe
    sc2 = gfx.Scene()
    sc2.add(gfx.Object3D("c", loaded_mesh, mat_orig))
    sfile = sc2.save(tmp_path / "with_glb.scene.json")
    sc3 = gfx.Scene.from_file(sfile)
    assert sc3.objects[0].mesh.num_triangles == mesh_orig.num_triangles


# ───────────────────────── Milestone 4: advanced reflections (HDRI / SSR / mirror) & physics ─────────────────────────


def test_hdri_skybox_rendering_and_json(tmp_path: Path):
    """An equirectangular skybox colours the sky background, lights the IBL reflections and survives JSON."""
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 1.0, 5.0), (0.0, 1.0, 0.0), bloom=False, fxaa=False)
    # Plain green skybox for a sharp contrast with the default procedural blue sky
    green_sky = gfx.Texture.from_array(np.full((32, 64, 4), [20, 235, 45, 255], dtype=np.uint8))
    sc.env = gfx.Environment(skybox=green_sky, skybox_intensity=1.2)
    sc.add(gfx.Object3D("sphere", gfx.Mesh.sphere(0.8), gfx.Material.metal((1.0, 1.0, 1.0), roughness=0.05), pos=(0.0, 1.0, 0.0)))

    app = gfx.App(160, 120, show_fps=False)
    img = app.render_rgba(sc)
    # Top-left corner = skybox sky → green clearly dominant
    sky_px = img[10, 10]
    assert int(sky_px[1]) > int(sky_px[0]) + 80
    assert int(sky_px[1]) > int(sky_px[2]) + 80
    # Centre = mirror sphere reflecting the green sky
    center_px = img[60, 80]
    assert int(center_px[1]) > int(center_px[0]) + 25

    # Procedural studio skybox + JSON save / reload
    sc_studio = gfx.Scene()
    sc_studio.env = gfx.Environment(skybox=gfx.Environment.studio_skybox(64, 32), skybox_intensity=1.0)
    sc_studio.add(gfx.Object3D("b", gfx.Mesh.box(1.0)))
    sfile = sc_studio.save(tmp_path / "studio_sky.scene.json")
    loaded = gfx.Scene.from_file(sfile)
    assert loaded.env.skybox is not None
    assert loaded.env.skybox.width == 64


def test_ssr_reflects_bright_object_on_glossy_floor():
    """With `cam.ssr = True`, a bright emissive object above a shiny floor is reflected in it."""
    sc = gfx.Scene()
    sc.env = gfx.Environment(sun=gfx.DirectionalLight(intensity=0.2), fog_density=0.0)
    sc.camera = gfx.Camera3D.fixed((0.0, 2.2, 5.2), (0.0, 0.7, 0.0), bloom=False, fxaa=False, ssr=False)
    sc.add(
        gfx.Object3D(
            "floor",
            gfx.Mesh.plane(12.0, subdivisions=12),
            gfx.Material(albedo=(0.04, 0.04, 0.05), metallic=0.85, roughness=0.06),
        )
    )
    sc.add(
        gfx.Object3D(
            "neon",
            gfx.Mesh.box(2.0, 1.8, 0.5),
            gfx.Material.glow((1.0, 0.15, 0.15), intensity=3.5),
            pos=(0.0, 1.2, 0.0),
        )
    )

    app = gfx.App(256, 160, show_fps=False)
    img_off = app.render_rgba(sc).copy()
    sc.camera.ssr = True
    sc.camera.ssr_strength = 0.85
    sc.camera.ssr_steps = 32
    img_on = app.render_rgba(sc).copy()

    # Floor area just in front of the red neon box (bottom centre of the image)
    floor_off = img_off[108:148, 96:160, 0].astype(np.float32).mean()
    floor_on = img_on[108:148, 96:160, 0].astype(np.float32).mean()
    assert floor_on > floor_off + 20.0


def test_planar_reflection_mirror_sync():
    """`PlanarReflection` + `Material.mirror` reflects the camera and projects the reflection 1:1 on the mirror."""
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 1.5, 4.5), (0.0, 1.5, 0.0), bloom=False, fxaa=False)
    mirror = gfx.PlanarReflection(plane_z=-2.0, width=128, height=96, match_viewport_aspect=False)
    mirror.sync(sc.camera)
    # Symmetry z' = 2*(-2) - 4.5 = -8.5
    assert mirror.camera.pos[2] == pytest.approx(-8.5, abs=1e-4)
    assert mirror.camera.target[2] == pytest.approx(-4.0, abs=1e-4)

    # A magenta cube between the camera and the mirror must appear in the mirror
    sc.add(gfx.Object3D("mirror", gfx.Mesh.quad(5.0, 3.5), gfx.Material.mirror(mirror), pos=(0.0, 1.5, -2.0)))
    sc.add(
        gfx.Object3D(
            "cube",
            gfx.Mesh.box(1.2, 1.2, 1.2),
            gfx.Material.glow((1.0, 0.1, 0.9), intensity=2.2),
            pos=(0.0, 1.5, 1.2),
        )
    )

    app = gfx.App(256, 140, show_fps=False)
    img = app.render_rgba(sc)
    # The mirror's RenderTarget was filled on the GPU and holds bright magenta pixels
    assert mirror.target._gpu_array is not None
    m_rgba = mirror.target._gpu_array.numpy().view(np.uint8).reshape(96, 128, 4)
    center = m_rgba[48, 64]
    assert int(center[0]) > 150 and int(center[2]) > 130
    assert img.shape == (140, 256, 4)


def test_physics_world_collisions_triggers_and_raycast():
    """`PhysicsWorld` handles gravity, bouncing on the ground, sphere-sphere collision, triggers and raycasts."""
    world = gfx.PhysicsWorld(gravity=(0.0, -9.81, 0.0))
    ground_node = gfx.Node("ground")
    world.add_ground(ground_node, y=0.0, restitution=0.8)

    ball = gfx.Node("ball", pos=(0.0, 2.0, 0.0))
    rb_ball = world.add(gfx.RigidBody(ball, gfx.SphereCollider(0.5), mass=1.0, restitution=0.8))

    # Trigger zone crossed by the ball as it falls
    triggered: list[str] = []
    zone = gfx.Node("zone", pos=(0.0, 1.2, 0.0))
    world.add(
        gfx.RigidBody(
            zone,
            gfx.SphereCollider(0.4),
            static=True,
            is_trigger=True,
            on_collision=lambda _a, b: triggered.append(b.node.name),
        )
    )

    for _ in range(90):
        world.step(1.0 / 60.0)

    # The ball never goes below the ground (y >= radius) and fired the trigger
    assert ball.y >= 0.45
    assert "ball" in triggered

    # Analytical raycast: hits a sphere, a box and the ground
    box_node = gfx.Node("box", pos=(3.0, 1.0, 0.0))
    rb_box = world.add(gfx.RigidBody(box_node, gfx.BoxCollider((0.5, 0.5, 0.5)), static=True))
    hit_box = world.raycast((3.0, 1.0, 5.0), (0.0, 0.0, -1.0))
    assert hit_box is not None and hit_box.body is rb_box
    assert hit_box.distance == pytest.approx(4.5, abs=1e-3)

    hit_ground = world.raycast((-5.0, 3.0, 0.0), (0.0, -1.0, 0.0))
    assert hit_ground is not None and hit_ground.distance == pytest.approx(3.0, abs=1e-3)
    assert rb_ball in world.bodies


def test_gpu_sphere_physics_to_instanced_mesh():
    """`GpuSpherePhysics` simulates collisions 100 % on the GPU and feeds `InstancedMesh3D.gpu_trs` directly."""
    pos = np.array([[-0.3, 1.5, 0.0], [0.3, 1.5, 0.0], [0.0, 0.8, 0.0]], dtype=np.float32)
    vel = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    phys = gfx.GpuSpherePhysics(pos, vel, radii=0.35, gravity=9.81, box_half=3.0, floor_y=0.0, restitution=0.8)

    for _ in range(40):
        trs_gpu = phys.step(1.0 / 60.0, substeps=2)

    p_after = phys.pos_gpu.numpy()
    # Every sphere stays above the ground (y >= r) and within the box bounds
    assert np.all(p_after[:, 1] >= 0.34)
    assert np.all(np.abs(p_after[:, 0]) <= 3.0)
    # The first two spheres rushing toward each other pushed each other back
    assert float(p_after[1, 0] - p_after[0, 0]) > 0.5

    # Direct connection to `InstancedMesh3D.gpu_trs` and full GPU rendering
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 1.5, 5.5), (0.0, 0.8, 0.0), bloom=False, fxaa=False)
    inst = sc.add(
        gfx.InstancedMesh3D("s", gfx.Mesh.sphere(1.0, rings=8, sectors=10), gfx.Material.metal((1.0, 0.7, 0.2)), count=3)
    )
    inst.gpu_trs = trs_gpu

    app = gfx.App(128, 120, show_fps=False)
    img = app.render_rgba(sc)
    assert img.shape == (120, 128, 4)
    assert float(img[:, :, :3].std()) > 5.0


# ───────────────────────── 100 % GPU fluids (3D water / 3D SPH / 2D Navier-Stokes) ─────────────────────────


def test_gpu_water_surface_realistic_and_low_poly_flat():
    """`GpuWaterSurface` simulates 3D waves on the GPU in smooth realistic mode AND flat-shaded low-poly."""
    # 1) Realistic mode (smooth indexed mesh)
    w_real = gfx.GpuWaterSurface(gfx.WaterSurfaceConfig.realistic(size=10.0, subdivisions=16))
    w_real.splash(0.0, 0.0, strength=-0.8, radius=0.9)
    for _ in range(8):
        pos_g, norm_g = w_real.step(1.0 / 60.0)
    pos_np = pos_g.numpy()
    # The waves propagated (non-zero, bounded height)
    assert float(np.abs(pos_np[:, 1]).max()) > 0.05
    assert float(np.abs(pos_np[:, 1]).max()) < 3.0

    # 2) Two-way flotation (`FloatingBody`): a nearby wave sets the sphere moving
    buoy_node = gfx.Node("buoy", pos=(1.0, 0.1, 0.0))
    fb = gfx.FloatingBody(buoy_node, radius=0.45, wave_push=18.0)
    w_real.reset()
    w_real.splash(-0.2, 0.0, strength=0.65, radius=0.9)
    excursion = 0.0
    for _ in range(35):
        w_real.step(1.0 / 60.0)
        w_real.step_floating([fb], 1.0 / 60.0)
        excursion = max(excursion, abs(buoy_node.x - 1.0))
    # The wave creates no water: the buoy goes and comes back, only the excursion counts
    assert excursion > 0.015

    # 3) Low-poly mode (non-indexed flat-shaded mesh: the 3 vertices of a triangle share the same normal)
    w_lp = gfx.GpuWaterSurface(gfx.WaterSurfaceConfig.low_poly(size=10.0, subdivisions=8))
    w_lp.splash(0.5, -0.5, strength=-0.9, radius=1.2)
    pos_lp, norm_lp = w_lp.step(1.0 / 60.0)
    n_lp = norm_lp.numpy().reshape(-1, 3, 3)
    np.testing.assert_allclose(n_lp[:, 0, :], n_lp[:, 1, :], atol=1e-5)
    np.testing.assert_allclose(n_lp[:, 0, :], n_lp[:, 2, :], atol=1e-5)
    assert np.all(n_lp[:, :, 1] > 0.0)

    # Full GPU rendering without CPU copy
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 4.5, 6.5), (0.0, 0.0, 0.0), bloom=False, fxaa=False)
    obj = sc.add(gfx.Object3D("water_lp", w_lp.mesh, w_lp.material))
    w_lp.apply_to(obj)
    app = gfx.App(128, 96, show_fps=False)
    img = app.render_rgba(sc)
    assert float(img[:, :, :3].std()) > 5.0


def test_gpu_fluid_sph3d_presets_and_instancing():
    """`GpuFluid3D` simulates O(N) 3D MAC FLIP + 3D Snell / Beer-Lambert ray marching + mesh extraction + low-poly."""
    cfg_w = gfx.Fluid3DConfig.water(count=2000)
    assert cfg_w.solver == "mac_flip"
    assert cfg_w.render_mode == "raymarch"
    sph_w = gfx.GpuFluid3D(cfg_w, layout="dam_break")
    init_mean_x = float(sph_w.pos_gpu.numpy()[:, 0].mean())
    sph_w.push_wave(force_x=35.0)
    for _ in range(45):
        sph_w.step(1.0 / 60.0)

    p = sph_w.pos_gpu.numpy()
    bx, by, bz = cfg_w.bounds
    assert np.all(p[:, 1] >= cfg_w.floor_y)
    assert np.all(np.abs(p[:, 0]) <= bx + 1e-3)
    assert np.all(np.abs(p[:, 2]) <= bz + 1e-3)
    # The dam-break block clearly moved to the right (+X)
    assert float(p[:, 0].mean()) > init_mean_x + 0.15

    # 1) Volumetric 3D ray-marching mode (Snell + Beer-Lambert + caustics)
    sc_surf = gfx.Scene()
    sc_surf.camera = gfx.Camera3D.fixed((0.0, 2.2, 7.0), (0.0, 1.0, 0.0), bloom=False, fxaa=False)
    sc_surf.add(gfx.Object3D("floor", gfx.Mesh.plane(12.0, subdivisions=8), gfx.Material.checker()))
    app = gfx.App(128, 96, show_fps=False)
    img_empty = app.render_rgba(sc_surf)
    sc_surf.add(sph_w)
    img_fluid = app.render_rgba(sc_surf)
    assert float(np.abs(img_fluid.astype(np.float32) - img_empty.astype(np.float32)).mean()) > 1.5

    # 2) 3D mesh extraction from the density volume (Blender-like, exportable low-poly)
    m_surf = sph_w.extract_surface_mesh()
    assert m_surf.num_triangles > 20

    # 3) Low-poly preset (spinning cubes) + zero-copy instanced 3D rendering
    cfg_lp = gfx.FluidSPHConfig.low_poly(count=48, mesh_style="cube")
    sph_lp = gfx.GpuFluidSPH3D(cfg_lp, layout="column")
    trs_lp = sph_lp.step(1.0 / 60.0)
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 2.0, 7.0), (0.0, 1.0, 0.0), bloom=False, fxaa=False)
    inst = sc.add(gfx.InstancedMesh3D("sph", cfg_lp.make_mesh(), cfg_lp.make_material(), count=cfg_lp.count))
    inst.gpu_trs = trs_lp
    img = app.render_rgba(sc)
    assert float(img[:, :, :3].std()) > 5.0


def test_gpu_fluid2d_navier_stokes_and_pixel_cel():
    """`GpuFluid2D` simule Navier-Stokes 2D (encre lisse + cel-shaded pixel-art) en 0-copie 2D & 3D."""
    # 1) Realistic pure 2D ink (`Sprite2D`)
    f_ink = gfx.GpuFluid2D(gfx.Fluid2DConfig.ink(width=64, height=48))
    f_ink.splat(0.3, 0.5, 1.0, 0.15, color=(2.0, 0.2, 0.8), radius=0.12, force=3.5)
    for _ in range(8):
        f_ink.step(1.0 / 60.0)
    # The ink moved to the right by advection
    dye_r = f_ink.dr.numpy()
    assert float(dye_r[:, 22:48].max()) > 0.25

    sc2d = gfx.Scene()
    sc2d.camera_2d = gfx.Camera2D()
    sc2d.add(gfx.Sprite2D(name="ink", x=0.0, y=0.0, w=128.0, h=96.0, texture=f_ink.texture))
    app = gfx.App(128, 96, show_fps=False)
    img2d = app.render_rgba(sc2d)
    assert int(img2d[:, :, 0].max()) > 120

    # 2) Low-poly pixel-art cel-shaded mode (`posterize_levels > 1`, `bilinear=False`) mapped on a 3D quad
    f_pix = gfx.GpuFluid2D(gfx.Fluid2DConfig.low_poly_pixel(width=40, height=32))
    assert f_pix.texture.bilinear is False
    f_pix.splat(0.5, 0.5, 0.0, -1.0, color=(0.2, 1.8, 2.0), radius=0.16, force=30.0)
    f_pix.step(1.0 / 60.0)
    sc3d = gfx.Scene()
    sc3d.camera = gfx.Camera3D.fixed((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), bloom=False, fxaa=False)
    sc3d.add(gfx.Object3D("screen", gfx.Mesh.quad(3.0, 2.0), gfx.Material.screen(f_pix.texture)))
    img3d = app.render_rgba(sc3d)
    assert int(img3d[48, 64, 2]) > 80
