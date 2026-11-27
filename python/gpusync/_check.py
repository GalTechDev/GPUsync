"""Self-test of `python -m gpusync --check`: small kernels on every device, compared with NumPy.

Each test runs on one device given explicitly (`device=`), so nothing falls back elsewhere: a failure is the
device's own. A test the device cannot run (f64 without support) is reported as skipped, not failed.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

import gpusync as gs

BINS = 16
GROUP = 64


@gs.kernel
def _scale(x: gs.f32[:], m: gs.f32, s: gs.f32) -> gs.f32[:]:
    return (x - m) / s


@gs.kernel
def _mirror(a: gs.f32["n", "m"], b: gs.f32["n", "m"]) -> gs.f32["n", "m"]:
    i, j = gs.index()
    return a[i, j] + b[i, a.shape[1] - 1 - j]


@gs.kernel(grid="x")
def _histogram(x: gs.f32["n"], counts: gs.Out[gs.i32[BINS]]):
    i = gs.index()
    gs.atomic_add(counts, gs.min(gs.max(int(x[i] * BINS), 0), BINS - 1), 1)


@gs.kernel(group=GROUP)
def _group_sum(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, GROUP)
    t[r] = x[i]
    gs.barrier()
    s = GROUP // 2
    while s > 0:
        if r < s:
            t[r] = t[r] + t[r + s]
        gs.barrier()
        s = s // 2
    return t[0]


@gs.kernel
def _square_f64(x: gs.f64[:]) -> gs.f64[:]:
    return x * x + 1.0


class _Skip(Exception):
    pass


def _elementwise(dev, rng):
    x = rng.random(10_007, dtype=np.float32)
    got = np.asarray(_scale(x, 0.5, 0.25, device=dev))
    np.testing.assert_allclose(got, (x - np.float32(0.5)) / np.float32(0.25), rtol=1e-6, atol=1e-6)


def _indexed(dev, rng):
    a = rng.random((37, 53), dtype=np.float32)
    b = rng.random((37, 53), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(_mirror(a, b, device=dev)), a + b[:, ::-1], rtol=1e-6)


def _atomics(dev, rng):
    x = rng.random(20_011, dtype=np.float32)
    want = np.bincount(np.minimum((x * BINS).astype(np.int64), BINS - 1), minlength=BINS)
    np.testing.assert_array_equal(np.asarray(_histogram(x, device=dev)), want.astype(np.int32))


def _groups(dev, rng):
    x = rng.random(GROUP * 32, dtype=np.float32)
    got = np.asarray(_group_sum(x, device=dev)).reshape(32, GROUP)
    np.testing.assert_allclose(got, np.repeat(x.reshape(32, GROUP).sum(1, keepdims=True), GROUP, 1), rtol=1e-5)


def _f64(dev, rng):
    if not _device(dev).f64:
        raise _Skip("not supported")
    x = 1.0 + 1e-10 * np.arange(4099, dtype=np.float64)  # differences lost in f32
    np.testing.assert_allclose(np.asarray(_square_f64(x, device=dev)), x * x + 1.0, rtol=1e-14)


def _chained(dev, rng):
    x = rng.random(10_007, dtype=np.float32)
    y = _scale(x, 0.5, 0.25, device=dev)  # not read: the second kernel takes it as it is
    z = _scale(y, 1.0, 2.0, device=dev)
    want = ((x - np.float32(0.5)) / np.float32(0.25) - np.float32(1.0)) / np.float32(2.0)
    np.testing.assert_allclose(np.asarray(z), want, rtol=1e-6, atol=1e-6)


TESTS = [
    ("elementwise", _elementwise),
    ("indexed", _indexed),
    ("atomics", _atomics),
    ("groups", _groups),
    ("f64", _f64),
    ("chained", _chained),
]


def _describe(e: Exception) -> str:
    """One line: the exception type and the useful part of its message (NumPy's mismatch summary for wrong values)."""
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    if isinstance(e, AssertionError):
        keep = [ln for ln in lines if ln.startswith(("Mismatched", "Max absolute", "Max relative"))]
        return "wrong result: " + "; ".join(keep or lines[:1])
    return f"{type(e).__name__}: {lines[0] if lines else ''}"


def _device(dev) -> gs.Device:
    return next(d for d in gs.devices() if d.id == dev)


@dataclass
class Result:
    """Outcome of the self-test on one device: `status[name]` is "ok", "skipped" or "failed"."""

    device: gs.Device
    status: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(s != "failed" for s in self.status.values())


def check(devices=None) -> list[Result]:
    """Runs every test on each device (`devices`: identifiers, default all detected devices)."""
    found = gs.devices()
    if devices:
        known = {d.id for d in found}
        unknown = [d for d in devices if d not in known]
        if unknown:
            raise gs.DeviceError(f"unknown device(s): {', '.join(unknown)}; detected: {', '.join(sorted(known))}")
        found = [d for d in found if d.id in devices]
    results = []
    for d in found:
        res = Result(d)
        start = time.perf_counter()
        for name, test in TESTS:
            rng = np.random.default_rng(7)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", gs.CompatibilityWarning)
                    test(d.id, rng)
                res.status[name] = "ok"
            except _Skip as e:
                res.status[name] = "skipped"
                res.errors[name] = str(e)
            except Exception as e:  # noqa: BLE001 (every failure is reported, none stops the others)
                res.status[name] = "failed"
                res.errors[name] = _describe(e)
        res.seconds = time.perf_counter() - start
        results.append(res)
    return results
