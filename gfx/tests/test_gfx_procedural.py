"""Procedural cumulus: shape bounds and a finite image over the clear sky."""

import numpy as np

import gpusync.gfx as gfx
from gpusync.gfx.procedural import Cumulus, ProceduralClouds


def test_bubbles_stay_between_base_and_top():
    for kw in (dict(top=1600.0, radius=1600.0, growth=1.2), dict(top=4500.0, radius=1300.0, growth=2.2, shear=0.15)):
        c = Cumulus(base=800.0, **kw)
        b = c.bubbles()
        assert 3 <= len(b) <= 160
        assert (b[:, 2] + 0.8 * b[:, 3] <= c.top + 1e-3).all() and (b[:, 2] > c.base).all()
        assert np.array_equal(b, c.bubbles())  # same seed, same cloud


def test_cloud_is_drawn_over_the_sky():
    atm = gfx.Atmosphere(gfx.AtmosphereConfig(nx=16, ny=16, nz=40, dx=1000.0, dy=1000.0, dz=150.0, rain=False))
    sky = gfx.CloudRenderer(atm, sun_elevation=40.0, sun_azimuth=160.0)
    pc = ProceduralClouds(sky)
    cam = sky.camera(np.array([2000.0, 2000.0, 20.0]), 45.0, 12.0)
    bg = sky.march(cam, 96, 54)
    out = pc.render(cam, bg, [Cumulus(x=6000.0, y=6000.0)])
    img, clear = (np.stack([c.numpy() for c in a], -1) for a in (out, bg))
    assert np.isfinite(img).all() and img.min() >= 0.0
    changed = np.abs(img - clear).sum(-1) > 0.05 * clear.sum(-1)
    assert 0.01 < changed.mean() < 0.6
