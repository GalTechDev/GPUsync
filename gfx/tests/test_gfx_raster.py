"""Rasterizer of gpusync.gfx: more than 65,536 triangles, depth precision, near-plane clipping,
exact picking. On every GPU."""

from __future__ import annotations

import numpy as np
import pytest

import gpusync as gs
import gpusync.gfx as gfx

GPUS = [d.id for d in gs.devices() if d.backend != "cpu"]
W, H = 256, 144
EMPTY = 0xFFFFFFFF


def _app(dev):
    return gfx.App(W, H, device=dev, show_fps=False)


def _cam(pos, target, **kw):
    return gfx.Camera3D.fixed(pos=pos, target=target, bloom=False, fxaa=False, shadows=False, **kw)


def _covered(app):
    return np.asarray(app.renderer._last_views[0].visbuf) != EMPTY


def _ballast_scene(n_ballast):
    """One visible red ball, preceded by `n_ballast` spheres out of view (smaller triangle numbers)."""
    sc = gfx.Scene()
    sc.camera = _cam((0.0, 0.8, 4.0), (0.0, 0.0, 0.0))
    for k in range(n_ballast):
        sc.add(gfx.Object3D(f"lest{k}", gfx.Mesh.sphere(0.5), gfx.Material.solid((0, 1, 0)), pos=(0, 500.0 + 3 * k, 0)))
    sc.add(gfx.Object3D("ball", gfx.Mesh.sphere(1.0), gfx.Material.solid((1.0, 0.1, 0.1), roughness=0.9)))
    return sc


@pytest.mark.parametrize("dev", GPUS)
@pytest.mark.parametrize("n_ballast", [45, 90])
def test_more_than_65536_triangles(dev, n_ballast):
    ref = _app(dev).render_rgba(_ballast_scene(0))
    app = _app(dev)
    sc = _ballast_scene(n_ballast)
    img = app.render_rgba(sc)
    assert sc.tri_idx.shape[0] > 65_536
    assert np.abs(img.astype(int) - ref.astype(int)).max() <= 2
    hit = app.pick(W // 2, H // 2)
    assert hit is not None and hit.object.name == "ball"
    assert hit.triangle >= n_ballast * gfx.Mesh.sphere(0.5).num_triangles


def _floor_scene(subdivisions, pos=(0.0, 1.2, 3.0), target=(0.0, 0.3, -3.0), **kw):
    sc = gfx.Scene()
    sc.camera = _cam(pos, target, **kw)
    sc.add(gfx.Object3D("floor", gfx.Mesh.plane(40.0, subdivisions=subdivisions), gfx.Material.checker()))
    return sc


VIEWS = [
    dict(),  # vertices behind the camera
    dict(pos=(0.0, 0.3, 0.0), target=(5.0, 0.0, 1.0)),  # skimming the ground, at an angle
    dict(pos=(1.0, 2.0, -2.0), target=(1.0, 0.0, -2.2)),  # looking down, camera in the middle of the ground
    dict(near=1.0),  # far near plane
]


@pytest.mark.parametrize("dev", GPUS)
@pytest.mark.parametrize("view", VIEWS)
def test_near_plane_clipping(dev, view):
    """A 2-triangle floor and the same floor heavily subdivided give the same image."""
    fine_app, coarse_app = _app(dev), _app(dev)
    fine = fine_app.render_rgba(_floor_scene(200, **view)).astype(int)
    coarse = coarse_app.render_rgba(_floor_scene(1, **view)).astype(int)
    mask_f, mask_c = _covered(fine_app), _covered(coarse_app)
    assert mask_c.mean() > 0.3
    assert (mask_f != mask_c).mean() < 0.02  # floor edge within one pixel
    both = mask_f & mask_c
    diff = np.abs(fine - coarse).max(axis=2)[both]
    assert (diff > 12).mean() < 0.02  # checkerboard edges within one pixel


@pytest.mark.parametrize("dev", GPUS)
def test_pick_on_clipped_triangle(dev):
    app = _app(dev)
    sc = _floor_scene(1)
    app.render_rgba(sc)
    cam = sc.camera
    fwd, right, up = cam.basis()
    t = np.tan(np.radians(cam.fov_deg) * 0.5)
    checked = 0
    for x, y in [(128, 140), (20, 130), (240, 100), (128, 90), (60, 120)]:
        hit = app.pick(x, y)
        ray = fwd + right * (((x + 0.5) / W * 2 - 1) * W / H * t) + up * ((1 - (y + 0.5) / H * 2) * t)
        if ray[1] >= -1e-3:
            continue
        expected = cam.pos + ray * (-cam.pos[1] / ray[1])
        if np.abs(expected[[0, 2]]).max() > 19:
            continue
        assert hit is not None
        np.testing.assert_allclose(hit.world_pos, expected, atol=0.02 * np.linalg.norm(expected - cam.pos))
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("dev", GPUS)
def test_depth_precision(dev):
    """Two panels 2 mm apart, depth up to 400 m: the front one wins everywhere."""
    sc = gfx.Scene()
    sc.camera = _cam((0.0, 0.0, 6.0), (0.0, 0.0, 0.0), far=400.0)
    back = gfx.Object3D("back", gfx.Mesh.quad(40.0, 40.0), gfx.Material.solid((0, 1, 0)), rot=(0.0, 0.3, 0.0))
    front = gfx.Object3D("front", gfx.Mesh.quad(40.0, 40.0), gfx.Material.solid((1, 0, 0)), rot=(0.0, 0.3, 0.0))
    front.pos = np.array([0.002 * np.sin(0.3), 0.0, 0.002 * np.cos(0.3)], dtype=np.float32)
    sc.add(back)
    sc.add(front)
    app = _app(dev)
    app.render_rgba(sc)
    vis = np.asarray(app.renderer._last_views[0].visbuf)
    seen = sc.tri_obj[vis[vis != EMPTY]]
    assert seen.size > 0.9 * W * H
    assert (seen == 1).mean() > 0.999


@pytest.mark.parametrize("dev", GPUS)
def test_pick_after_instanced_mesh(dev):
    """An instance before the object shifts the matrices: picking must take the one of the vertex hit."""
    trs = np.zeros((3, 8), dtype=np.float32)
    trs[:] = [0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    trs[0, :3] = [-3.0, 0.0, 0.0]
    sc = gfx.Scene()
    sc.camera = _cam((0.0, 0.0, 8.0), (0.0, 0.0, 0.0))
    sc.add(gfx.InstancedMesh3D("inst", gfx.Mesh.cube(1.0, sub=1), gfx.Material.solid((0, 1, 0)), count=3, trs=trs))
    sc.add(gfx.Object3D("box", gfx.Mesh.cube(1.0, sub=1), gfx.Material.solid((1, 0, 0)), pos=(2.0, 0.0, 0.0)))
    app = _app(dev)
    app.render_rgba(sc)
    vis = np.asarray(app.renderer._last_views[0].visbuf)
    mask = np.where(vis == EMPTY, -1, sc.tri_obj[np.where(vis == EMPTY, 0, vis & 0x7FFFFFFF) % len(sc.tri_obj)])
    for obj_idx, name, pos in [(0, "inst", (-3.0, 0.0, 0.0)), (1, "box", (2.0, 0.0, 0.0))]:
        ys, xs = np.nonzero(mask == obj_idx)
        x, y = int(np.median(xs)), int(np.median(ys))
        hit = app.pick(x, y)
        assert hit.object.name == name
        assert np.abs(hit.world_pos - np.array(pos)).max() <= 0.52
