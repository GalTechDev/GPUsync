"""64-bit indices: an array of more than 2³¹ − 1 elements in an indexed kernel.

Without them, the flattened index `x[i, k]` (32 bits) would overflow and the GPU would read at the wrong
place, without error. The 64-bit index variant is compiled only on demand; here it is forced on small
arrays (threshold lowered with `GPUSYNC_WIDE_INDEX_LIMIT`) to be tested on every device. The real case
(8 GiB) is optional: `GS_BIG=1`.
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
    assert "(long)" not in narrow  # default variant: 32-bit indices, faster


def test_series_with_wide_indices():
    """The tests of indexed kernels and explicit mode, with 64-bit indices forced (except wgpu)."""
    # Tests without `device=`: on a device that has 64-bit indices.
    env = dict(os.environ, GPUSYNC_WIDE_INDEX_LIMIT="16", GPUSYNC_DEVICE=next(d.id for d in gs.devices() if d.id.split(":")[0] not in WGPU))
    # test_block_mode itself goes through the discrete GPUs, Vulkan included: excluded here.
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
            print(d.id, "ok" if np.array_equal(y, want) else "WRONG")
        except gs.UnsupportedFeature as e:
            print(d.id, "refused", "64-bit" in str(e))
''')


def test_wgpu_refuses_clearly(tmp_path):
    env = dict(os.environ, GPUSYNC_WIDE_INDEX_LIMIT="16")
    script = tmp_path / "large.py"  # in a file: a kernel's source must be readable
    script.write_text(SCRIPT)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        dev, verdict = line.split()[:2]
        if dev.split(":")[0] in WGPU:
            assert line.endswith("refused True"), line
        else:
            assert verdict == "ok", line


@pytest.mark.skipif(os.environ.get("GS_BIG") != "1", reason="8 GiB of data: GS_BIG=1 to enable it")
@pytest.mark.parametrize("dev", [d.id for d in gs.devices() if d.id.split(":")[0] in ("cuda", "opencl", "cpu")])
def test_more_than_2_31_elements(dev):
    """Array of 2³¹ + 2048 f32 elements (8 GiB), kept whole: read at row `n − 1 − i`."""
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
    except gs.UnsupportedFeature as e:  # does not fit whole on this device: clear refusal
        assert "so it must" in str(e) or "fit whole" in str(e), e
        pytest.skip(f"{dev}: {str(e)[:80]}")
    finally:
        gs.set_memory_budget(None)
    np.testing.assert_array_equal(y[:4], np.arange(n - 1, n - 5, -1, dtype=np.float32))
    np.testing.assert_array_equal(y[-4:], np.arange(3, -1, -1, dtype=np.float32))
