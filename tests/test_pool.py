"""Réutilisation des buffers du périphérique (réserve par backend) : un buffer libéré resservant
alors que des calculs en file le lisent encore ne doit jamais fausser un résultat."""

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices() if not d.id.startswith("cpu")]  # réserves des GPU


@gs.kernel
def plus1(x: gs.u32[:]) -> gs.u32[:]:
    return x + 1


@gs.kernel
def slow(x: gs.f32[:]) -> gs.f32[:]:
    s = x
    for k in range(2000):
        s = s * 0.5 + 1.0
    return s + x


@gs.kernel
def fill(x: gs.f32[:], v: gs.f32) -> gs.f32[:]:
    return x * 0 + v


@pytest.mark.parametrize("dev", DEVICES)
def test_long_chain(dev):
    """y = f(y) 300 fois sans lecture : les résultats intermédiaires sont libérés et réutilisés."""
    z = plus1(np.zeros(1 << 20, np.uint32), device=dev)
    for _ in range(299):
        z = plus1(z, device=dev)
    check(z, np.full(1 << 20, 300, np.uint32))


@pytest.mark.parametrize("dev", DEVICES)
def test_input_released_during_compute(dev):
    """L'entrée d'un calcul long est lâchée aussitôt ; ses buffers resservent à d'autres calculs
    (qui les écrasent) pendant que le premier peut encore les lire."""
    n = 1 << 18
    ref = np.zeros(n, np.float32)
    for _ in range(2000):
        ref = ref * np.float32(0.5) + np.float32(1.0)
    for _ in range(3):
        xin = fill(np.ones(n, np.float32), 0.0, device=dev)  # zéros, restés sur le périphérique
        y = slow(xin, device=dev)
        del xin  # buffer rendu à la réserve pendant le calcul
        others = [fill(np.zeros(n, np.float32), float(k + 7), device=dev) for k in range(8)]
        check(y, ref)
        for k, a in enumerate(others):
            check(a, np.full(n, k + 7, np.float32))


@pytest.mark.parametrize("dev", DEVICES)
def test_independent_calls_kept_and_dropped(dev):
    rng = np.random.default_rng(5)
    xs = [rng.integers(0, 1 << 30, 50_000, dtype=np.uint32) for _ in range(40)]
    kept = []
    for k, x in enumerate(xs):
        y = plus1(plus1(x, device=dev), device=dev)
        if k % 3 == 0:
            kept.append((x, y))
    for x, y in kept:
        check(y, x + 2)
