"""gs.matmul_u8 : produit de matrices d'octets collectif d'un groupe (Tensor Cores en CUDA, boucle
portable ailleurs), comparé à NumPy sur chaque périphérique, pour des formes alignées sur les tuiles des
Tensor Cores (16 × 8 × 32) et quelconques."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

import gpusync as gs

from test_index import check

DEVICES = [d.id for d in gs.devices()]
RNG = np.random.default_rng(8)


def make_kernel(group, m, n, k, reps=1):
    """Kernel : chaque groupe charge a (m × k octets) et b (n × k), c = 0, puis reps produits ;
    rend c (m·n mots) répété sur les cases du groupe."""
    kw = k // 4
    src = f'''import gpusync as gs

@gs.kernel(group={group})
def mm(a: gs.u32["g", "ka"], b: gs.u32["g", "kb"], tmpl: gs.u32["g", "w"]) -> gs.u32["g", "w"]:
    i, j = gs.index()
    r = gs.lane()
    A = gs.shared(gs.u32, {m * kw})
    B = gs.shared(gs.u32, {n * kw})
    C = gs.shared(gs.u32, {m * n})
    for t in range(r, {m * kw}, {group}):
        A[t] = a[i, t]
    for t in range(r, {n * kw}, {group}):
        B[t] = b[i, t]
    gs.barrier()
    for rep in range({reps}):
        gs.matmul_u8(C, A, B, {m}, {n}, {k})
    return C[j % {m * n}]
'''
    path = Path(tempfile.mkdtemp()) / f"mm_{group}_{m}_{n}_{k}_{reps}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mm


def reference(a, b, m, n, k, reps):
    ab = a.view(np.uint8).reshape(len(a), m, k).astype(np.uint64)
    bb = b.view(np.uint8).reshape(len(b), n, k).astype(np.uint64)
    prod = np.einsum("gik,gjk->gij", ab, bb) * reps
    return (prod & 0xFFFFFFFF).astype(np.uint32).reshape(len(a), m * n)


CASES = [
    (128, 32, 16, 64, 1),    # tuiles des Tensor Cores
    (256, 64, 64, 128, 2),   # plusieurs tuiles par warp, deux produits accumulés
    (64, 5, 7, 12, 1),       # forme quelconque : boucle portable partout
]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("group,m,n,k,reps", CASES)
def test_matmul_u8(dev, group, m, n, k, reps):
    info = next(d for d in gs.devices() if d.id == dev)
    if group > info.max_group:
        pytest.skip(f"groupes de {group} threads non disponibles")
    f = make_kernel(group, m, n, k, reps)
    groups = 6
    w = group * max(1, -(-(m * n) // group))
    a = RNG.integers(0, 1 << 32, (groups, m * k // 4), dtype=np.uint32)
    b = RNG.integers(0, 1 << 32, (groups, n * k // 4), dtype=np.uint32)
    if dev.startswith("cpu"):
        a, b = a[:2], b[:2]
    out = np.asarray(f(a, b, np.zeros((len(a), w), np.uint32), device=dev))
    ref = reference(a, b, m, n, k, reps)
    idx = np.arange(w) % (m * n)
    check(out, ref[:, idx])


def test_python_function_without_device():
    c = np.zeros(4 * 8, np.uint32)
    a = RNG.integers(0, 1 << 32, 4 * 8 // 4 * 8, dtype=np.uint32)[: 4 * 32 // 4]
    b = RNG.integers(0, 1 << 32, 8 * 32 // 4, dtype=np.uint32)
    gs.matmul_u8(c, a, b, 4, 8, 32)
    check(c, reference(a[None], b[None], 4, 8, 32, 1)[0])


def test_errors(tmp_path):
    def load(body, group="group=64"):
        p = tmp_path / f"k{abs(hash(tuple(body) + (group,)))}.py"
        p.write_text("import gpusync as gs\n\n"
                     f"@gs.kernel({group})\ndef k(x: gs.u32[\"n\"]) -> gs.u32[\"n\"]:\n"
                     + "".join(f"    {line}\n" for line in body))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    base = ["i = gs.index()", "A = gs.shared(gs.u32, 64)", "C = gs.shared(gs.u32, 64)"]
    with pytest.raises(gs.TypeMismatch, match="multiple de 4"):
        load(base + ["gs.matmul_u8(C, A, A, 4, 4, 6)", "return x[i]"])
    with pytest.raises(gs.TypeMismatch, match="il en faut"):
        load(base + ["gs.matmul_u8(C, A, A, 16, 16, 32)", "return x[i]"])
    with pytest.raises(gs.KernelSyntaxError, match="dépend du thread"):
        load(base + ["if x[i] > 0:", "    gs.matmul_u8(C, A, A, 4, 4, 16)", "return x[i]"])
    with pytest.raises(gs.KernelSyntaxError, match="group="):
        load(["i = gs.index()", "gs.matmul_u8(x, x, x, 4, 4, 16)", "return x[i]"], group="")
