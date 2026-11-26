"""Objects placed by a GPU computation (`bind_transform`) and floaters simulated on the GPU (`GpuFloaters`)."""

import numpy as np
import pytest

import gpusync as gs
import gpusync.gfx as gfx
from gpusync.gfx.kernels.k3d import upload_f32


def _scene(*objects):
    sc = gfx.Scene()
    sc.camera = gfx.Camera3D.fixed((0.0, 3.0, 9.0), (0.0, 0.5, 0.0), bloom=False, fxaa=False)
    sc.add(gfx.Object3D("floor", gfx.Mesh.plane(14.0, subdivisions=2), gfx.Material.checker()))
    for o in objects:
        sc.add(o)
    return sc


def _box(name="box", pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scale=1.0):
    return gfx.Object3D(name, gfx.Mesh.box(1.0, 0.6, 1.4), gfx.Material(albedo=(0.9, 0.2, 0.1)), pos=pos, rot=rot, scale=scale)


def _img(sc, app=None):
    app = app or gfx.App(256, 144, show_fps=False)
    return app.render_rgba(sc)[:, :, :3].astype(np.float32)


@gs.kernel
def move_row(src: gs.f32["n", "c8"], dx: gs.f32) -> gs.f32["n", "c8"]:
    i, c = gs.index()
    if c == 0:
        return src[i, c] + dx
    return src[i, c]


@pytest.mark.parametrize("scale", [1.0, (0.6, 1.5, 0.9)])
def test_bound_object_matches_cpu_placement(scale):
    """Same image as with `pos`/`rot`: shadows, lighting and the object's own scale included."""
    pos, rot = (1.2, 0.9, -0.4), (0.3, 0.8, -0.5)
    expected = _img(_scene(_box(pos=pos, rot=rot, scale=scale)))
    obj = _box(scale=scale)
    trs = np.array([[9, 9, 9, 0, 0, 0, 1, 1], [*pos, *rot, 1, 1]], dtype=np.float32)
    obj.bind_transform(upload_f32(trs), 1)
    got = _img(_scene(obj))
    assert np.abs(got - expected).max() <= 2
    assert np.abs(expected - _img(_scene())).max() > 30


def test_bound_position_follows_a_kernel_without_cpu():
    obj = _box(pos=(-2.0, 0.5, 0.0))
    sc = _scene(obj)
    app = gfx.App(256, 144, show_fps=False)
    trs = upload_f32(np.array([[-2.0, 0.5, 0, 0, 0, 0, 1, 1]], dtype=np.float32))
    centers = []
    for _ in range(3):
        obj.bind_transform(trs, 0)
        img = _img(sc, app)
        red = np.argwhere((img[:, :, 0] - img[:, :, 2]) > 60)
        centers.append(red[:, 1].mean())
        trs = move_row(trs, 1.5)
    assert centers[0] + 15 < centers[1] < centers[2] - 15
    assert obj.x == -2.0  # the CPU knew nothing


def test_bound_scale_visibility_and_release():
    obj = _box(pos=(0.0, 0.6, 0.0))
    sc, app = _scene(obj), gfx.App(256, 144, show_fps=False)
    cpu, empty = _img(sc, app), _img(_scene())
    base = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    obj.bind_transform(upload_f32(np.array([base], dtype=np.float32)))
    assert np.abs(_img(sc, app) - cpu).max() <= 2
    hidden = base[:7] + [0.0]
    obj.bind_transform(upload_f32(np.array([hidden], dtype=np.float32)))
    assert np.abs(_img(sc, app) - empty).max() <= 2
    obj.bind_transform(upload_f32(np.array([base], dtype=np.float32)))
    obj.visible = False
    assert np.abs(_img(sc, app) - empty).max() <= 2
    obj.visible = True
    obj.bind_transform(None)
    assert np.abs(_img(sc, app) - cpu).max() <= 2
    for bad in (np.zeros((2, 4), np.float32), np.zeros(8, np.float32)):
        with pytest.raises(ValueError, match="bind_transform"):
            obj.bind_transform(upload_f32(bad))
    with pytest.raises(ValueError, match="row 3"):
        obj.bind_transform(upload_f32(np.zeros((2, 8), np.float32)), 3)


def test_bound_objects_from_several_arrays_with_instances():
    a, b = _box("a"), _box("b")
    inst = gfx.InstancedMesh3D("inst", gfx.Mesh.box(0.4, 0.4, 0.4), gfx.Material(), count=3)
    inst.gpu_trs = upload_f32(np.array([[x, 2.0, 0, 0, 0, 0, 1, 1] for x in (-2, 0, 2)], dtype=np.float32))
    a.bind_transform(upload_f32(np.array([[-2.5, 0.5, 0, 0, 0, 0, 1, 1]], dtype=np.float32)))
    b.bind_transform(upload_f32(np.array([[2.5, 0.5, 0, 0, 0, 0, 1, 1]], dtype=np.float32)))
    expected = _scene(inst, _box("a", (-2.5, 0.5, 0.0)), _box("b", (2.5, 0.5, 0.0)))
    want = _img(expected)
    assert np.abs(_img(_scene(inst, a, b)) - want).max() <= 2


def test_picking_uses_the_gpu_position():
    obj = _box(pos=(50.0, 0.0, 0.0))  # CPU position off screen
    obj.bind_transform(upload_f32(np.array([[0.0, 0.5, 0.0, 0, 0, 0, 1, 1]], dtype=np.float32)))
    app = gfx.App(256, 144, show_fps=False)
    sc = _scene(obj)
    app.set_scene(sc)
    _img(sc, app)
    hit = app.pick(128, 72)
    assert hit is not None and hit.object is obj
    assert np.abs(np.asarray(hit.world_pos) - (0.0, 0.5, 0.0)).max() < 0.8


# ───────────────────────── Flotteurs ─────────────────────────

BODIES = [(0.0, 0.25, 0.0, 0.56), (-2.1, 0.22, 1.4, 0.42), (0.7, 0.2, 0.3, 0.38)]


def _floating(gpu, n, steps, *, kick=None, fps=60):
    cfg = gfx.WaterSurfaceConfig.realistic()
    cfg.ambient_waves = 0.05
    w = gfx.GpuWaterSurface(cfg)
    bodies = [
        gfx.FloatingBody(_box(f"b{i}", (x, y, z)), radius=r, wave_push=16.5, drag=0.96, wake_strength=0.14)
        for i, (x, y, z, r) in enumerate(BODIES[:n])
    ]
    sim = gfx.GpuFloaters(w, bodies) if gpu else None
    w.splash(-1.6, -1.0, strength=-0.9, radius=0.85)
    for k in range(steps):
        w.step(1.0 / fps)
        if k == kick:
            if gpu:
                sim.set_velocity(bodies[0], y=-3.8)
            else:
                bodies[0].vel[1] = -3.8
        if gpu:
            sim.step(1.0 / fps)
        else:
            w.step_floating(bodies, 1.0 / fps)
    if gpu:
        sim.read()
    return np.array([[*b.node.pos, *b.node.rot, *b.vel] for b in bodies]), w.h_curr.numpy()


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gpu_floaters_match_cpu_floaters(n):
    """Same trajectory, rotation, speed and wake as the CPU version, collisions and dive included."""
    cpu, h_cpu = _floating(False, n, 150, kick=60)
    gpu, h_gpu = _floating(True, n, 150, kick=60)
    np.testing.assert_allclose(gpu, cpu, atol=2e-3)
    np.testing.assert_allclose(h_gpu, h_cpu, atol=2e-3)
    start = np.array([b[:3] for b in BODIES[:n]])
    assert np.abs(cpu[:, :3] - start).max() > 0.02


def test_gpu_floaters_do_not_touch_the_cpu_until_read():
    w = gfx.GpuWaterSurface(gfx.WaterSurfaceConfig.realistic())
    body = gfx.FloatingBody(_box("b", (1.0, 1.5, 0.0)), radius=0.5)
    sim = gfx.GpuFloaters(w, [body])
    for _ in range(60):
        w.step(1.0 / 60.0)
        sim.step(1.0 / 60.0)
    assert body.node.pos == (1.0, 1.5, 0.0)
    sim.read()
    assert body.node.y < 0.8
    assert w._h_cpu is None  # the grid was never copied back


def test_gpu_floaters_kicks_are_applied_once():
    w = gfx.GpuWaterSurface(gfx.WaterSurfaceConfig.realistic())
    body = gfx.FloatingBody(_box("b", (0.0, 5.0, 0.0)), radius=0.5, gravity=0.0, drag=1.0)
    sim = gfx.GpuFloaters(w, [body])
    sim.set_velocity(0, x=2.0)
    sim.add_velocity(body, dz=1.0)
    for _ in range(30):
        sim.step(1.0 / 60.0)
    sim.read()
    assert abs(body.node.x - 1.0) < 1e-3 and abs(body.node.z - 0.5) < 1e-3
    with pytest.raises(ValueError):
        gfx.GpuFloaters(w, [])
