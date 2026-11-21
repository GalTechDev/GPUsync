"""Indices 64 bits : un tableau de plus de 2³¹ − 1 éléments dans un kernel avec index.

Sans eux, l'indice aplati `x[i, k]` (32 bits) déborderait et le GPU lirait au mauvais endroit, sans
erreur. La variante à indices 64 bits n'est compilée qu'à la demande ; ici, elle est forcée sur de
petits tableaux (seuil abaissé par `GPUSYNC_WIDE_INDEX_LIMIT`) pour être testée sur tous les
périphériques. Le vrai cas (8 Gio) est en option : `GS_BIG=1`.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import gpusync as gs

HERE = os.path.dirname(os.path.abspath(__file__))
WGPU = ("metal", "vulkan", "dx12")


def test_generated_code():
    from test_index import pairwise

    narrow = pairwise.source("opencl")
    assert "(long)" not in narrow  # variante par défaut : indices 32 bits, plus rapide


def test_series_with_wide_indices():
    """Les tests des kernels avec index et du mode explicite, avec indices 64 bits forcés (hors wgpu)."""
    # Tests sans `device=` : sur un périphérique qui a les indices 64 bits.
    env = dict(os.environ, GPUSYNC_WIDE_INDEX_LIMIT="16", GPUSYNC_DEVICE=next(d.id for d in gs.devices() if d.id.split(":")[0] not in WGPU))
    # test_block_mode parcourt lui-même les GPU dédiés, Vulkan compris : exclu ici.
    skip = " and ".join([f"not {b}" for b in WGPU] + ["not block_mode"])
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-k", skip,
         "test_index.py", "test_rows.py", "test_explicit.py"],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


SCRIPT = textwrap.dedent('''
    import numpy as np, gpusync as gs, sys
    @gs.kernel
    def last(x: gs.f32["n", "t"]) -> gs.f32["n"]:
        i = gs.index()
        return x[x.shape[0] - 1 - i, x.shape[1] - 1] + 1.0
    x = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    want = x[::-1, -1] + 1
    for d in gs.devices():
        try:
            y = np.asarray(last(x, device=d.id))
            print(d.id, "ok" if np.array_equal(y, want) else "FAUX")
        except gs.UnsupportedFeature as e:
            print(d.id, "refus", "64 bits" in str(e))
''')


def test_wgpu_refuses_clearly(tmp_path):
    env = dict(os.environ, GPUSYNC_WIDE_INDEX_LIMIT="16")
    script = tmp_path / "large.py"  # dans un fichier : le source d'un kernel doit être lisible
    script.write_text(SCRIPT)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        dev, verdict = line.split()[:2]
        if dev.split(":")[0] in WGPU:
            assert line.endswith("refus True"), line
        else:
            assert verdict == "ok", line


@pytest.mark.skipif(os.environ.get("GS_BIG") != "1", reason="8 Gio de données : GS_BIG=1 pour l'activer")
@pytest.mark.parametrize("dev", [d.id for d in gs.devices() if d.id.split(":")[0] in ("cuda", "opencl", "cpu")])
def test_more_than_2_31_elements(dev):
    """Tableau de 2³¹ + 2048 éléments f32 (8 Gio), gardé entier : lu à la ligne `n − 1 − i`."""
    n, t = (1 << 20) + 1, 2048
    x = np.empty((n, t), dtype=np.float32)
    x[:, -1] = np.arange(n, dtype=np.float32)
    x[:, 0] = 0.0

    @gs.kernel
    def last(x: gs.f32["n", "t"]) -> gs.f32["n"]:
        i = gs.index()
        return x[x.shape[0] - 1 - i, x.shape[1] - 1]

    gs.set_memory_budget(15 << 30)
    try:
        y = np.asarray(last(x, device=dev))
    except gs.UnsupportedFeature as e:  # ne tient pas entier sur ce périphérique : refus clair
        assert "doit donc" in str(e) or "tenir" in str(e), e
        pytest.skip(f"{dev} : {str(e)[:80]}")
    finally:
        gs.set_memory_budget(None)
    np.testing.assert_array_equal(y[:4], np.arange(n - 1, n - 5, -1, dtype=np.float32))
    np.testing.assert_array_equal(y[-4:], np.arange(3, -1, -1, dtype=np.float32))
