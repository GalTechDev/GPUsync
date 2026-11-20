"""Jalon M3 : kernels avec index découpés par blocs de lignes (ARCHITECTURE.md §9.7).

Les réglages du moteur sont réduits pour forcer, sur de petits tableaux, de nombreux blocs dont le
dernier est incomplet, dans chaque mode (Direct, Résident, par blocs), et des lancements cadencés.
"""

import numpy as np
import pytest

import gpusync as gs
from gpusync import _core
from gpusync._device import _set_transfer_limits

from test_index import check, dtw, dtw_ref, matvec, moments, pairwise

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(31)

# (taille de bloc, budget, taille de bloc Direct)
CONFIGS = {
    "blocs": dict(chunk_bytes=4000, budget_bytes=200_000, direct_chunk_bytes=12_000),
    "budget_minimal": dict(chunk_bytes=4000, budget_bytes=1, direct_chunk_bytes=12_000),
}


@pytest.fixture(params=list(CONFIGS))
def limits(request):
    _set_transfer_limits(**CONFIGS[request.param])
    yield request.param
    _set_transfer_limits()


@gs.kernel
def gather(x: gs.f32["n"], idx: gs.i32["m"]) -> gs.f32["m"]:
    """1 dimension : `idx` est découpé, `x` (lu à n'importe quel indice) reste entier."""
    i = gs.index()
    return x[idx[i]] * 2.0


@gs.kernel
def rescale(x: gs.f32["n", "t"]) -> gs.f32["n", "t"]:
    """Une ligne lue en entier, mais toujours la ligne courante : découpée."""
    i, j = gs.index()
    m = x[i, 0]
    for k in range(x.shape[1]):
        m = gs.max(m, x[i, k])
    return x[i, j] / m


@gs.kernel
def shifted(x: gs.f32["n", "t"]) -> gs.f32["n", "t"]:
    """Lit la ligne suivante : `x` ne peut pas être découpé (reste entier)."""
    i, j = gs.index()
    nxt = i + 1 if i + 1 < x.shape[0] else i
    return x[nxt, j] - x[i, j]


def needs_whole(dev):
    """Le budget peut-il s'appliquer ? Pas en mode Direct (cpu:0, OpenCL sur mémoire unifiée) ;
    sur Metal, le mode Direct dépend de l'alignement des tableaux NumPy (aléatoire)."""
    return not (dev == "cpu:0" or (dev.startswith("opencl") and _core.device_zero_copy(dev)))


@pytest.mark.parametrize("dev", DEVICES)
def test_lignes_decoupees(dev, limits):
    x = RNG.standard_normal((300, 9)).astype(np.float32)
    mean, std, arg = moments(x, device=dev)
    check(mean, x.mean(1), rtol=1e-5)
    check(std, x.std(1), rtol=1e-4, atol=1e-4)
    check(arg, x.argmax(1).astype(np.int32))
    a = RNG.random((257, 33), dtype=np.float32) + 0.5
    check(rescale(a, device=dev), a / a.max(1, keepdims=True))


@pytest.mark.parametrize("dev", DEVICES)
def test_entree_entiere_et_sortie_decoupee(dev, limits):
    x = RNG.random((150, 6), dtype=np.float32)
    ref = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).astype(np.float32)
    v = RNG.random(6, dtype=np.float32)
    s = RNG.random((150, 6), dtype=np.float32)
    nxt = np.vstack([s[1:], s[-1:]])
    if limits == "budget_minimal" and needs_whole(dev) and not dev.startswith("metal"):
        # `x` (lu aussi à la ligne j) ne tient pas dans un budget d'un octet : erreur qui le nomme.
        with pytest.raises(gs.UnsupportedFeature, match="l'entrée 'x'.*indexez-la par la ligne courante"):
            pairwise(x, device=dev)
        with pytest.raises(gs.UnsupportedFeature, match="l'entrée 'v'"):
            matvec(x, v, device=dev)
        return
    xs = RNG.random(1000, dtype=np.float32)
    idx = RNG.integers(0, 1000, 5003, dtype=np.int32)
    for f, args, want in [
        (gather, (xs, idx), xs[idx] * 2.0),
        (pairwise, (x,), ref),
        (matvec, (x, v), (x @ v).astype(np.float32)),
        (shifted, (s,), nxt - s),
    ]:
        try:
            got = f(*args, device=dev)
        except gs.UnsupportedFeature:
            assert dev.startswith("metal") and limits == "budget_minimal"
            continue
        check(got, want, rtol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_dtw_par_blocs(dev, limits):
    x = RNG.random((90, 12), dtype=np.float32)
    try:
        y = dtw(x, 3, device=dev)
    except gs.UnsupportedFeature as e:
        assert limits == "budget_minimal" and needs_whole(dev), e
        return
    check(y, dtw_ref(x, 3).astype(np.float32), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_entree_restee_sur_le_peripherique(dev):
    """Une sortie restée sur le GPU, relue par blocs de lignes par le kernel suivant."""
    x = RNG.random((200, 5), dtype=np.float32)
    d = pairwise(x, device=dev)  # réglages par défaut : Résident sur GPU dédié
    ref = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).astype(np.float32)
    _set_transfer_limits(chunk_bytes=4000, budget_bytes=10_000, direct_chunk_bytes=12_000)
    try:
        mean, std, arg = moments(d, device=dev)
        check(mean, ref.mean(1), rtol=1e-5)
        check(arg, ref.argmax(1).astype(np.int32))
    finally:
        _set_transfer_limits()


@pytest.mark.parametrize("dev", [d for d in DEVICES if d != "cpu:0"])
def test_lancements_cadences(dev):
    """Grille de plus de 65 536 cases : plusieurs lancements chronométrés, résultat identique."""
    x = RNG.random((700, 4), dtype=np.float32)
    ref = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).astype(np.float32)
    check(pairwise(x, device=dev), ref)
    y = RNG.random((3000, 40), dtype=np.float32)
    check(moments(y, device=dev)[0], y.mean(1), rtol=1e-5)


def test_mode_par_blocs():
    """Sur GPU dédié, un appel au-delà du budget passe en mode « flux » (par blocs de lignes)."""
    dedicated = [d.id for d in gs.devices() if not d.unified_memory]
    if not dedicated:
        pytest.skip("pas de GPU dédié")
    x = RNG.random((400, 8), dtype=np.float32)
    _set_transfer_limits(chunk_bytes=4000, budget_bytes=10_000)
    try:
        for dev in dedicated:
            y = moments(x, device=dev)[0]
            assert y.mode == "flux", (dev, y.mode)
            check(y, x.mean(1), rtol=1e-5)
    finally:
        _set_transfer_limits()
