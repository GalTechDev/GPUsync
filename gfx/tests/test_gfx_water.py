"""3D water: simulation at rest, caustics, submerged objects, spray and foam."""

import numpy as np
import pytest

import gpusync.gfx as gfx
from gpusync.gfx.fluids import upload_f32

SUN = np.array([0.48, 0.78, 0.40]) / np.linalg.norm([0.48, 0.78, 0.40])


def _flat_volume(fl, level):
    dx, dy, dz = fl.config.vol_res
    dens = np.zeros((dz, dy, dx), dtype=np.float32)
    dens[:, :level, :] = 1.0
    return upload_f32(dens.ravel(), device=fl.device)


def _surface(fl):
    """Surface height per column (m), edges excluded, and number of hollow voxels below the surface."""
    dx, dy, dz = fl.config.vol_res
    dens = fl.vol_dens_gpu.numpy().reshape(dz, dy, dx)
    iso = fl.config.iso_level
    wet = dens >= iso
    holes = ~wet & (np.cumsum(wet[:, ::-1, :], axis=1)[:, ::-1, :] > 0)
    top = dy - 1 - np.argmax(wet[:, ::-1, :], axis=1)
    z, x = np.indices(top.shape)
    a, b = dens[z, top, x], dens[z, np.minimum(top + 1, dy - 1), x]
    height = (top + (a - iso) / np.maximum(a - b, 1e-5)) * fl.config.bounds[1] / (dy - 1)
    return height[4:-4, 4:-4], int(holes[4:-4, :, 4:-4].sum())


@pytest.mark.parametrize("make", [gfx.Fluid3DConfig.water, gfx.Fluid3DConfig.water_sph])
def test_water_comes_to_rest(make):
    """At rest: motionless, centred, flat, without air pockets, at the starting volume."""
    fl = gfx.GpuFluid3D(make(count=14000), layout="dam_break")
    for _ in range(3600):
        fl.step(1.0 / 60.0)
    speed = np.linalg.norm(fl.vel_gpu.numpy()[:, :3], axis=1).mean()
    assert speed < 0.08
    center = fl.pos_gpu.numpy()[:, :3].mean(0)
    assert abs(center[0]) < 0.15 and abs(center[2]) < 0.1
    # 28 % of the basin at the start: neither swelling nor compacting
    assert abs(center[1] - 0.14 * fl.config.bounds[1]) < 0.06
    height, holes = _surface(fl)
    assert holes == 0
    assert height.std() < 0.02
    # More agitation: more foam
    assert (fl.ww_pos_gpu.numpy()[:, 3] > 0).sum() < 60


@pytest.mark.parametrize("level", [0, 20])
def test_caustics_of_flat_water_are_uniform(level):
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water(count=2000), layout="column")
    fl.vol_dens_gpu = _flat_volume(fl, level)
    light = fl.caustic_map(SUN).numpy()
    inner = light[8:-8, 8:-8]
    assert abs(inner.mean() - 1.0) < 0.03
    assert inner.min() > 0.85 and inner.max() < 1.15


def test_caustics_are_cached_until_water_changes():
    fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water(count=2000), layout="column")
    first = fl.caustic_map(SUN)
    assert fl.caustic_map(SUN) is first
    assert fl.caustic_map((0.0, 1.0, 0.0)) is not first
    fl.vol_dens_gpu = _flat_volume(fl, 8)
    assert fl.caustic_map(SUN) is not first


def test_submerged_object_is_visible():
    """An object fully under water can be seen through the glass and the surface."""

    def render(with_object):
        fl = gfx.GpuFluid3D(gfx.Fluid3DConfig.water(count=2000), layout="column")
        fl.vol_dens_gpu = _flat_volume(fl, fl.config.vol_res[1] // 2)
        sc = gfx.Scene()
        sc.camera = gfx.Camera3D.fixed((0.0, 1.6, 8.0), (0.0, 1.0, 0.0), bloom=False, fxaa=False)
        sc.add(gfx.Object3D("floor", gfx.Mesh.plane(16.0, subdivisions=4), gfx.Material.checker()))
        if with_object:
            sc.add(gfx.Object3D("ball", gfx.Mesh.sphere(0.6), gfx.Material(albedo=(0.95, 0.05, 0.05)), pos=(0.0, 0.9, 0.5)))
        sc.add(fl)
        return gfx.App(256, 144, show_fps=False).render_rgba(sc)[:, :, :3].astype(np.float32)

    diff = np.abs(render(True) - render(False)).max(axis=2)
    assert (diff > 25).sum() > 150


def test_whitewater_is_born_and_bounded():
    cfg = gfx.Fluid3DConfig.water(count=10000)
    fl = gfx.GpuFluid3D(cfg, layout="dam_break")
    seen = set()
    for k in range(200):
        fl.step(1.0 / 60.0)
        if k % 25 == 24:
            p, v = fl.ww_pos_gpu.numpy(), fl.ww_vel_gpu.numpy()
            assert np.isfinite(p).all() and np.isfinite(v).all()
            alive = p[:, 3] > 0
            seen |= set(v[alive, 3].astype(int).tolist())
            bx, _, bz = cfg.bounds
            assert np.all(np.abs(p[alive, 0]) <= bx) and np.all(np.abs(p[alive, 2]) <= bz)
    assert {0, 1} <= seen


# ───────────────────────── Wave surface and 2D ink ─────────────────────────


@pytest.mark.parametrize("boundary_point", [(0.0, 0.0), (5.0, -4.5)])
def test_wave_surface_keeps_its_volume(boundary_point):
    """A splash moves the water without creating any, even near an edge."""
    cfg = gfx.WaterSurfaceConfig.realistic()
    cfg.ambient_waves = 0.0
    w = gfx.GpuWaterSurface(cfg)
    w.splash(*boundary_point, strength=-0.9, radius=0.9)
    assert w.h_curr.numpy().min() < -0.2
    for _ in range(300):
        w.step(1.0 / 60.0)
        assert abs(float(w.h_curr.numpy().mean())) < 1e-4


@pytest.mark.parametrize("subdivisions", [56, 112, 224])
def test_wave_speed_does_not_depend_on_the_mesh(subdivisions):
    cfg = gfx.WaterSurfaceConfig.realistic(size=24.0, subdivisions=subdivisions)
    cfg.ambient_waves, cfg.damping = 0.0, 1.0
    w = gfx.GpuWaterSurface(cfg)
    w.splash(0.0, 0.0, strength=-0.2, radius=1.2)
    trough = []
    for k in range(1, 97):
        w.step(1.0 / 60.0)
        if k in (48, 96):
            row = w.h_curr.numpy()[w.grid_n // 2]
            x = np.linspace(-12.0, 12.0, w.grid_n)
            right = x > 2.0
            trough.append(x[right][np.argmin(row[right])])
    assert abs((trough[1] - trough[0]) / 0.8 - cfg.wave_speed) < 0.5


def test_wave_surface_is_stable_and_calms_down():
    for make in ("realistic", "low_poly", "ocean", "lava", "mercury"):
        cfg = getattr(gfx.WaterSurfaceConfig, make)()
        cfg.ambient_waves, cfg.wave_speed = 0.0, 12.0
        w = gfx.GpuWaterSurface(cfg)
        w.splash(0.3, 0.2, strength=-1.5, radius=0.5)
        for _ in range(900):
            w.step(1.0 / 60.0)
        h = w.h_curr.numpy()
        assert np.isfinite(h).all() and np.abs(h).max() < 0.05, make


def test_wave_speed_can_change_while_running():
    """The number of sub-steps changes with the speed: the wave must neither jump nor diverge."""
    cfg = gfx.WaterSurfaceConfig.realistic()
    cfg.ambient_waves = 0.0
    w = gfx.GpuWaterSurface(cfg)
    w.splash(0.0, 0.0, strength=-0.6, radius=0.9)
    for k in range(120):
        cfg.wave_speed = 4.8 if (k // 20) % 2 == 0 else 16.0
        w.step(1.0 / 60.0)
        assert np.abs(w.h_curr.numpy()).max() < 1.0


@pytest.mark.parametrize("make", [gfx.Fluid2DConfig.ink, gfx.Fluid2DConfig.smoke_fire])
def test_ink_does_not_depend_on_the_frame_rate(make):
    dyes = []
    for fps in (30, 60, 240):
        f = gfx.GpuFluid2D(make(width=96, height=64))
        f.splat(0.5, 0.7, 0.4, -0.5, color=(2.0, 1.0, 0.4), radius=0.1, force=4.0)
        for _ in range(fps):
            f.step(1.0 / fps)
        dyes.append(f.dr.numpy())
    assert dyes[1].sum() > 1.0
    np.testing.assert_allclose(dyes[0], dyes[1], atol=1e-5)
    np.testing.assert_allclose(dyes[2], dyes[1], atol=1e-5)


def test_ink_fixed_step_can_be_refined():
    """A step twice as fine barely changes the amount of ink: the coefficients really are per second."""
    total = []
    for fixed_dt in (1.0 / 60.0, 1.0 / 120.0):
        cfg = gfx.Fluid2DConfig.ink(96, 64)
        cfg.fixed_dt = fixed_dt
        f = gfx.GpuFluid2D(cfg)
        f.splat(0.5, 0.5, 0.3, 0.0, color=(2.0, 1.0, 0.4), radius=0.1, force=3.0)
        for _ in range(60):
            f.step(1.0 / 60.0)
        total.append(float(f.dr.numpy().sum()))
    assert abs(total[0] - total[1]) < 0.12 * total[0]


# ───────────────────────── Optics of the water surface ─────────────────────────


def _pool(objects=(), *, pos=(0.0, 0.0, 0.0), cam=None, subdivisions=24, **cfg_changes):
    cfg = gfx.WaterSurfaceConfig.realistic(subdivisions=subdivisions)
    cfg.ambient_waves = 0.0
    for name, value in cfg_changes.items():
        setattr(cfg, name, value)
    water = gfx.GpuWaterSurface(cfg)
    sc = gfx.Scene()
    sc.camera = cam or gfx.Camera3D.fixed((pos[0], pos[1] + 7.0, pos[2] + 6.0), pos, bloom=False, fxaa=False)
    obj = sc.add(gfx.Object3D("water", water.mesh, water.material, pos=pos))
    water.apply_to(obj)
    for o in objects:
        sc.add(o)
    return sc, water, obj


def _ball(y, pos=(0.0, 0.0, 0.0)):
    return gfx.Object3D("ball", gfx.Mesh.sphere(0.6), gfx.Material(albedo=(0.95, 0.05, 0.05)), pos=(pos[0], pos[1] + y, pos[2]))


def _img(sc, size=(256, 144)):
    app = gfx.App(*size, show_fps=False)
    return app, app.render_rgba(sc)[:, :, :3].astype(np.float32)


@pytest.mark.parametrize("pos", [(0.0, 0.0, 0.0), (20.0, 3.0, -5.0)])
def test_pool_shows_submerged_objects(pos):
    _, empty = _img(_pool(pos=pos)[0])
    _, seen = _img(_pool([_ball(-0.9, pos)], pos=pos)[0])
    red = (seen[:, :, 0] - seen[:, :, 2]) - (empty[:, :, 0] - empty[:, :, 2])
    assert (red > 40).sum() > 120
    # The same ball under an opaque surface stays invisible
    _, hidden = _img(_pool([_ball(-0.9, pos)], pos=pos, optics=False)[0])
    _, opaque = _img(_pool(pos=pos, optics=False)[0])
    assert np.abs(hidden - opaque).max() < 8


def test_deeper_water_is_darker_in_red():
    """Absorption: the deeper the water, the less red comes back, and blue holds up better."""
    mid = []
    for depth in (0.3, 1.5, 5.0):
        _, img = _img(_pool(depth=depth, walls=False, tint=(0.0, 0.0, 0.0), caustics=0.0)[0])
        mid.append(img[60:90, 100:156].mean(axis=(0, 1)))
    assert mid[0][0] > mid[1][0] > mid[2][0]
    assert mid[2][2] / mid[0][2] > mid[2][0] / mid[0][0]


def test_pool_picking_hits_water_then_objects():
    sc, _, obj = _pool([_ball(0.2)])
    app, _ = _img(sc)
    assert app.pick(128, 72).object.name == "ball"
    hit = app.pick(40, 72)
    assert hit.object is obj
    assert abs(float(hit.world_pos[1])) < 0.05


def test_optics_can_be_switched_and_objects_removed():
    sc, water, obj = _pool([_ball(-0.9)])
    app, clear = _img(sc)
    water.config.optics = False
    water.apply_to(obj)
    opaque = app.render_rgba(sc)[:, :, :3].astype(np.float32)
    assert np.abs(opaque - clear).mean() > 5
    water.config.optics = True
    water.apply_to(obj)
    again = app.render_rgba(sc)[:, :, :3].astype(np.float32)
    assert np.abs(again - clear).max() < 2
    sc.remove(obj)
    assert np.isfinite(app.render_rgba(sc)).all()


def test_pool_caustics_of_flat_water_are_uniform():
    _, water, _ = _pool()
    light = water.caustic_map(SUN).numpy()[4:-4, 4:-4]
    assert abs(light.mean() - 1.0) < 0.01 and np.abs(light - 1.0).max() < 0.02


def test_pool_caustics_match_geometric_optics():
    """Under a trough, sun at the zenith: irradiance = 1 / (1 + (1 - 1/n) · depth · curvature)²."""
    _, water, _ = _pool(subdivisions=56)
    cfg = water.config
    water.splash(0.0, 0.0, strength=-0.6, radius=1.5)
    h = water.h_curr.numpy() * cfg.height_scale
    m, cell = water.grid_n // 2, cfg.size / cfg.subdivisions
    lap = (h[m, m - 1] + h[m, m + 1] + h[m - 1, m] + h[m + 1, m] - 4.0 * h[m, m]) / cell**2
    expected = 1.0 / (1.0 + (1.0 - 1.0 / cfg.ior) * (cfg.depth + h[m, m]) * lap / 2.0) ** 2
    light = water.caustic_map((0.0, 1.0, 0.0)).numpy()
    k = light.shape[0] // 2
    assert expected < 0.9
    assert abs(light[k - 1 : k + 2, k - 1 : k + 2].mean() - expected) < 0.05
    assert abs(light[4:-4, 4:-4].mean() - 1.0) < 0.02


def test_two_pools_and_orthographic_camera():
    sc, _, _ = _pool([_ball(-0.9)])
    other = gfx.GpuWaterSurface(gfx.WaterSurfaceConfig.ocean(size=6.0, subdivisions=16))
    obj = sc.add(gfx.Object3D("sea", other.mesh, other.material, pos=(0.0, 1.0, -4.0)))
    other.apply_to(obj)
    app, img = _img(sc)
    assert np.isfinite(img).all() and img.std() > 5
    names = {h.object.name for h in (app.pick(128, 100), app.pick(128, 20)) if h}
    assert "water" in names or "ball" in names
    sc.camera.ortho = True
    assert np.isfinite(app.render_rgba(sc)).all()


# ───────────────────────── Optimizations: same results ─────────────────────────


def test_fused_pressure_matches_plain_jacobi():
    """Two iterations per launch = two ordinary Jacobi iterations, air, walls and surface included."""
    from gpusync.gfx.kernels.kfluid3d import mac_pressure_diag, mac_pressure_jacobi2

    nx, ny, nz = 9, 7, 6
    rng = np.random.default_rng(3)
    fill = rng.uniform(0.0, 1.3, (nz, ny, nx)).astype(np.float32)
    div = rng.normal(size=fill.shape).astype(np.float32)
    p = rng.normal(size=fill.shape).astype(np.float32)
    wet = fill >= 0.5

    def neighbours(a, value):
        padded = np.pad(a.astype(np.float64), 1, constant_values=value)
        return [np.roll(padded, s, ax)[1:-1, 1:-1, 1:-1] for ax in range(3) for s in (1, -1)]

    count = np.zeros(fill.shape)
    for fj in neighbours(fill, np.nan):
        theta = np.clip((fill - 0.5) / np.maximum(fill - fj, 1e-4), 0.25, 1.0)
        count += np.where(np.isnan(fj), 0.0, np.where(fj >= 0.5, 1.0, 1.0 / theta))
    expected = p.astype(np.float64)
    for _ in range(6):
        around = sum(neighbours(np.where(wet, expected, 0.0), 0.0))
        expected = np.where(wet, (around - div) / np.maximum(count, 1.0), 0.0)

    got = p.ravel()
    diag = mac_pressure_diag(fill.ravel(), nx, ny, nz)
    for _ in range(3):
        got = mac_pressure_jacobi2(got, div.ravel(), diag, nx, ny, nz)
    np.testing.assert_allclose(got.numpy(), expected.ravel(), atol=2e-5)


def test_render_scale_keeps_image_picking_and_interface():
    sc, _, obj = _pool([_ball(0.2)])
    sc.ui.add(gfx.Button(x=10, y=10, w=60, h=24, text="OK", on_click=lambda: None))
    full_app = gfx.App(512, 288, show_fps=False)
    full = full_app.render_rgba(sc)[:, :, :3].astype(np.float32)
    app = gfx.App(512, 288, show_fps=False, render_scale=0.75)
    img = app.render_rgba(sc)[:, :, :3].astype(np.float32)
    assert img.shape == full.shape
    assert np.abs(img - full).mean() < 4
    # The interface stays at full resolution (semi-transparent: the scaled-up background counts a little there)
    assert np.abs(img[10:34, 10:70] - full[10:34, 10:70]).max() <= 3
    assert app.pick(256, 144).object.name == "ball"
    assert app.pick(80, 144).object is obj
    app.renderer.render_scale = 1.0
    np.testing.assert_array_equal(app.render_rgba(sc)[:, :, :3], full_app.render_rgba(sc)[:, :, :3])
