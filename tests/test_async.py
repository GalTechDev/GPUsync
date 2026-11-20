"""Jalon M4 : attente asynchrone, appels depuis plusieurs threads, DLPack, y.save()."""

import asyncio
import threading
import time

import numpy as np
import pytest

import gpusync as gs
from gpusync._device import _set_transfer_limits

from test_index import check, dtw, dtw_ref, moments, pairwise

DEVICES = [d.id for d in gs.devices()]
GPUS = [d for d in DEVICES if d != "cpu:0"]
RNG = np.random.default_rng(404)


@gs.kernel
def axpy(x: gs.f32[:], y: gs.f32[:], a: gs.f32) -> gs.f32[:]:
    return a * x + y


@pytest.mark.parametrize("dev", DEVICES)
def test_await(dev):
    x = RNG.random(5000, dtype=np.float32)
    y = RNG.random(5000, dtype=np.float32)

    async def main():
        z = await axpy(x, y, 2.0, device=dev)
        assert isinstance(z, gs.Array) and z.ready
        w = await axpy(z, y, -1.0, device=dev)  # chaînage puis attente
        m, s, a = await moments(RNG.random((40, 7), dtype=np.float32), device=dev)
        assert m.ready and s.ready and a.ready
        return z, w

    z, w = asyncio.run(main())
    check(z, 2.0 * x + y)
    check(w, -2.0 * x, atol=1e-6)


@pytest.mark.parametrize("dev", GPUS)
def test_boucle_libre_pendant_l_attente(dev):
    """Pendant `await y`, la boucle asyncio continue de tourner (attente sur un thread)."""
    x = RNG.random((1200, 48), dtype=np.float32)

    async def main():
        ticks = 0
        stop = False

        async def ticker():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.001)

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        y = dtw(x, 5, device=dev)
        t0 = time.perf_counter()
        before = ticks
        await y
        waited = time.perf_counter() - t0
        stop = True
        await t
        return y, ticks - before, waited

    y, ticks, waited = asyncio.run(main())
    if waited > 0.03:  # le calcul n'était pas fini au retour de l'appel : la boucle a dû tourner
        assert ticks >= 3, (ticks, waited)
    check(np.asarray(y)[:20, :20], dtw_ref(x[:20], 5).astype(np.float32), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_appel_dans_un_thread(dev):
    """`asyncio.to_thread(kernel, …)` : l'appel entier (même par blocs) hors de la boucle, GIL libéré."""
    x = RNG.random((300, 8), dtype=np.float32)
    ref = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).astype(np.float32)

    async def main():
        ys = await asyncio.gather(*(asyncio.to_thread(pairwise, x, device=dev) for _ in range(6)))
        return [np.asarray(y) for y in ys]

    for y in asyncio.run(main()):
        check(y, ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_plusieurs_threads(dev):
    """Le même kernel lancé et lu en parallèle depuis 8 threads, sur le même périphérique."""
    xs = [RNG.random(20_000, dtype=np.float32) for _ in range(8)]
    out = [None] * 8
    errors = []

    def work(i):
        try:
            for _ in range(5):
                out[i] = np.asarray(axpy(xs[i], xs[i], float(i), device=dev)).copy()
        except Exception as e:  # pragma: no cover - remonté plus bas
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for i in range(8):
        check(out[i], float(i) * xs[i] + xs[i])


def test_lecture_concurrente_du_meme_resultat():
    x = RNG.random((500, 6), dtype=np.float32)
    y = pairwise(x)
    got = []
    threads = [threading.Thread(target=lambda: got.append(y.numpy())) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(g is got[0] for g in got)


class Tensor:
    """Objet tiers qui n'expose que le protocole DLPack (comme un tenseur PyTorch ou JAX sur CPU)."""

    def __init__(self, a):
        self._a = a

    def __dlpack__(self, **kw):
        return self._a.__dlpack__(**kw)

    def __dlpack_device__(self):
        return self._a.__dlpack_device__()


@pytest.mark.parametrize("dev", DEVICES)
def test_dlpack(dev):
    x = RNG.random(3000, dtype=np.float32)
    y = axpy(Tensor(x), Tensor(x), 3.0, device=dev)  # import DLPack
    v = np.from_dlpack(y)  # export DLPack : la mémoire du résultat, sans copie
    assert np.shares_memory(v, y.numpy())
    check(v, 4.0 * x)


def test_dlpack_pytorch():
    torch = pytest.importorskip("torch")
    x = torch.rand(1000)
    y = axpy(x, x, 1.0)
    t = torch.from_dlpack(y)
    assert torch.allclose(t, 2 * x)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("budget", [None, 10_000])
def test_save(dev, budget, tmp_path):
    x = RNG.random((120, 5), dtype=np.float32)
    _set_transfer_limits(chunk_bytes=4000, budget_bytes=budget)
    try:
        y = pairwise(x, device=dev)
        y.save(tmp_path / "d.npy")
    finally:
        _set_transfer_limits()
    ref = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).astype(np.float32)
    check(np.load(tmp_path / "d.npy"), ref)
    check(y, ref)  # toujours lisible après l'écriture
