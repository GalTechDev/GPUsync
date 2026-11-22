"""Groupes de threads (`@gs.kernel(group=n)`) : mémoire partagée (gs.shared), position dans le groupe
(gs.lane), barrières (gs.barrier) ; comparés à NumPy sur chaque périphérique. Règles vérifiées à la
compilation (barrières atteintes par tout le groupe) et accès concurrents détectés sur cpu:0."""

import importlib.util

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(42)
G = 256


@gs.kernel(group=G)
def reverse(x: gs.f32["b", "n"]) -> gs.f32["b", "n"]:
    """Chaque groupe renverse ses G cases (échange par la mémoire partagée)."""
    i, j = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, G)
    t[r] = x[i, j]
    gs.barrier()
    return t[G - 1 - r]


@gs.kernel(group=G)
def group_sum(x: gs.f32["n"]) -> gs.f32["n"]:
    """Somme du groupe par réduction en arbre ; chaque case reçoit la somme de son groupe."""
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, G)
    t[r] = x[i]
    gs.barrier()
    s = G // 2
    while s > 0:
        if r < s:
            t[r] = t[r] + t[r + s]
        gs.barrier()
        s = s // 2
    return t[0]


@gs.kernel(group=G)
def prefix_sum(x: gs.i32["n"]) -> gs.i32["n"]:
    """Somme préfixe dans chaque groupe (Hillis-Steele : lecture, barrière, écriture, barrière)."""
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.i32, G)
    t[r] = x[i]
    gs.barrier()
    d = 1
    for _ in range(8):  # log2(256)
        v = t[r]
        if r >= d:
            v = v + t[r - d]
        gs.barrier()
        t[r] = v
        gs.barrier()
        d = d * 2
    return t[r]


@gs.kernel(group=128)
def across_rows(x: gs.i32["b", "n"]) -> gs.i32["b", "n"]:
    """Groupes de 128 cases sur des lignes de 64 : un groupe couvre deux lignes."""
    i, j = gs.index()
    r = gs.lane()
    t = gs.shared(gs.i32, 128)
    t[r] = x[i, j]
    gs.barrier()
    return t[(r + 64) % 128]


@gs.kernel(group=64)
def zeros_and_two_arrays(x: gs.f32["n"]) -> gs.f32["n"]:
    """Deux tableaux partagés, dont un jamais écrit (il doit valoir zéro, comme gs.local)."""
    i = gs.index()
    r = gs.lane()
    a = gs.shared(gs.f32, 64)
    unused = gs.shared(gs.f32, 64)
    a[r] = x[i] * 2.0
    gs.barrier()
    return a[63 - r] + unused[r]


def groups(x, g):
    return x.reshape(-1, g)


@pytest.mark.parametrize("dev", DEVICES)
def test_reverse(dev):
    x = RNG.random((6, 4 * G), dtype=np.float32)
    ref = groups(x, G)[:, ::-1].reshape(x.shape)
    check(reverse(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_group_sum(dev):
    x = RNG.integers(-50, 50, 20 * G).astype(np.float32)  # entiers : somme exacte en f32
    ref = np.repeat(groups(x, G).sum(axis=1), G)
    check(group_sum(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_prefix_sum(dev):
    x = RNG.integers(-1000, 1000, 12 * G, dtype=np.int32)
    ref = np.cumsum(groups(x, G), axis=1).astype(np.int32).reshape(-1)
    check(prefix_sum(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_group_spanning_rows(dev):
    x = RNG.integers(0, 1 << 20, (10, 64), dtype=np.int32)
    flat = groups(x.reshape(-1), 128)
    ref = np.roll(flat, -64, axis=1).reshape(x.shape)
    check(across_rows(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_shared_starts_at_zero(dev):
    x = RNG.random(64 * 7, dtype=np.float32)
    ref = groups(x * 2, 64)[:, ::-1].reshape(-1)
    check(zeros_and_two_arrays(x, device=dev), ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_many_groups_chained(dev):
    """Beaucoup de groupes (lancements découpés), résultat repris par un second kernel avec groupe."""
    x = RNG.random((64, 16 * G), dtype=np.float32)
    y = reverse(reverse(x, device=dev), device=dev)
    check(y, x)


@pytest.mark.parametrize("dev", DEVICES)
def test_largest_group(dev):
    """Taille de groupe maximale du périphérique (jusqu'à 1024), mémoire partagée à la mesure."""
    info = next(d for d in gs.devices() if d.id == dev)
    g = min(1024, info.max_group)
    g = 1 << (g.bit_length() - 1)
    path = _write(f"""
import gpusync as gs

@gs.kernel(group={g})
def rev(x: gs.u32["n"]) -> gs.u32["n"]:
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.u32, {g})
    t[r] = x[i]
    gs.barrier()
    return t[{g - 1} - r]
""")
    rev = _load(path).rev
    x = RNG.integers(0, 1 << 30, 9 * g, dtype=np.uint32)
    check(rev(x, device=dev), groups(x, g)[:, ::-1].reshape(-1))


def test_shape_not_a_multiple_of_the_group():
    with pytest.raises(gs.ShapeMismatch, match="tuiles complètes de 256"):
        reverse(np.zeros((3, 100), np.float32), device="cpu")


# ───────────────────────────── erreurs ─────────────────────────────

_counter = [0]


def _write(src):
    import tempfile
    from pathlib import Path

    _counter[0] += 1
    d = Path(tempfile.mkdtemp())
    p = d / f"kgroup{_counter[0]}.py"
    p.write_text(src)
    return p


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile(body, group="group=64", out="gs.f32[\"n\"]"):
    return _load(_write(
        "import gpusync as gs\n\n"
        f"@gs.kernel({group})\n"
        f"def k(x: gs.f32[\"n\"]) -> {out}:\n" + "".join(f"    {line}\n" for line in body)
    ))


def test_barrier_under_thread_condition():
    with pytest.raises(gs.KernelSyntaxError, match=r"(?s)ligne 9.*dépend du thread"):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)", "t[gs.lane()] = x[i]",
                  "if x[i] > 0.0:", "    gs.barrier()", "return t[0]"])


def test_barrier_in_loop_with_thread_bound():
    with pytest.raises(gs.KernelSyntaxError, match="dépend du thread"):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)",
                  "for k in range(gs.lane()):", "    gs.barrier()", "return t[0]"])


def test_barrier_after_partial_return():
    with pytest.raises(gs.KernelSyntaxError, match="return"):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)",
                  "if gs.lane() == 0:", "    return 1.0", "gs.barrier()", "return t[0]"])


def test_barrier_after_thread_break():
    with pytest.raises(gs.KernelSyntaxError, match="dépend du thread"):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)",
                  "for k in range(4):", "    if x[i] > 0.5:", "        break", "    gs.barrier()",
                  "return t[0]"])


def test_value_read_from_shared_is_per_thread():
    with pytest.raises(gs.KernelSyntaxError, match="dépend du thread"):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)", "t[gs.lane()] = x[i]", "gs.barrier()",
                  "v = t[0]", "if v > 0.0:", "    gs.barrier()", "return v"])


def test_uniform_conditions_are_accepted():
    k = _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)", "n = x.shape[0]",
                  "for s in range(3):", "    if n > 64:", "        gs.barrier()",
                  "t[gs.lane()] = x[i]", "gs.barrier()", "return t[63 - gs.lane()]"])
    x = RNG.random(128, dtype=np.float32)
    check(k.k(x, device="cpu"), groups(x, 64)[:, ::-1].reshape(-1))


def test_group_requires_index_and_declaration():
    with pytest.raises(gs.KernelSyntaxError, match="gs.index"):
        _load(_write("import gpusync as gs\n\n@gs.kernel(group=64)\ndef k(x: gs.f32[:]) -> gs.f32[:]:\n    return x\n"))
    with pytest.raises(gs.KernelSyntaxError, match="group="):
        _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)", "return x[i]"], group="")
    with pytest.raises(gs.KernelSyntaxError, match="group="):
        _compile(["i = gs.index()", "return x[i] + gs.lane()"], group="")
    with pytest.raises(ValueError, match="group"):
        gs.kernel(group=0)


def test_race_detected_on_cpu():
    k = _compile(["i = gs.index()", "r = gs.lane()", "t = gs.shared(gs.f32, 64)", "t[r] = x[i]",
                  "return t[(r + 1) % 64]"])
    with pytest.raises(gs.DeviceError, match=r"accès concurrent à t\[\d+\].*gs.barrier"):
        k.k(np.ones(64, np.float32), device="cpu")


def test_write_write_race_detected_on_cpu():
    k = _compile(["i = gs.index()", "t = gs.shared(gs.f32, 64)", "t[0] = x[i]", "gs.barrier()",
                  "return t[0]"])
    with pytest.raises(gs.DeviceError, match=r"accès concurrent à t\[0\].*l'écrivent"):
        k.k(np.ones(64, np.float32), device="cpu")


def test_group_larger_than_device():
    small = [d for d in gs.devices() if d.max_group < 1024]
    if not small:
        pytest.skip("tous les périphériques acceptent 1024 threads par groupe")
    k = _compile(["i = gs.index()", "return x[i]"], group="group=1024")
    with pytest.raises(gs.UnsupportedFeature, match="au plus"):
        k.k(np.ones(1024, np.float32), device=small[0].id)
