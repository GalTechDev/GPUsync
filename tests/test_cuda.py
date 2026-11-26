"""Milestone M6: CUDA backend. The generated code is checked everywhere; the rest needs an NVIDIA GPU.

Parity with the other backends is covered by the differential tests (every test
parametrized by device also runs on `cuda:0`); here: what is specific to CUDA.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import gpusync as gs

from test_explicit import per_bin
from test_index import dtw

HAS_CUDA = any(d.backend == "cuda" for d in gs.devices())
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no NVIDIA GPU with NVRTC")

SCRIPT = textwrap.dedent('''
    import numpy as np, gpusync as gs
    st = {s.name: s for s in gs.backends()}["cuda"]
    print("CUDA", "ok" if st.available else "indisponible", st.error or "")
    if st.available:
        @gs.kernel
        def axpy(x: gs.f32[:], a: gs.f32) -> gs.f32[:]:
            return a * x + 1.0
        x = np.arange(1000, dtype=np.float32)
        print("RESULT", bool((np.asarray(axpy(x, 2.0, device="cuda:0")) == 2 * x + 1).all()))
''')


def run(tmp_path, env):
    path = tmp_path / "k.py"
    path.write_text(SCRIPT)
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def without_toolkit(**extra):
    """Environment without the CUDA toolkit (neither CUDA_PATH nor its folders in the PATH)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_PATH") and k != "CUDA_HOME"}
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if "CUDA" not in p.upper())
    env.update(extra)
    return env


def test_code_cuda():
    src = dtw.source("cuda")
    assert 'extern "C" __global__ void gs_dtw(' in src
    assert "blockIdx.x * blockDim.x + threadIdx.x" in src and "__restrict__" in src
    atom = per_bin.source("cuda")
    assert "atomicAdd(&" in atom and "gs_atomic_add_f32" not in atom  # addition flottante native
    assert "atomicCAS" in atom and "fminf(" in atom  # float min / max emulated


@needs_cuda
def test_cache_ptx(tmp_path):
    cache = tmp_path / "cache"
    env = dict(os.environ, GPUSYNC_CACHE_DIR=str(cache))
    assert "RESULT True" in run(tmp_path, env)
    files = list((cache / "cuda").glob("*.ptx"))
    assert len(files) == 1 and files[0].read_bytes().startswith(b"//")
    assert "RESULT True" in run(tmp_path, env)  # 2nd process: PTX read back from the cache
    files[0].write_bytes(b"illisible")  # damaged file: recompiled, no error
    assert "RESULT True" in run(tmp_path, env)


@needs_cuda
def test_without_nvrtc(tmp_path):
    out = run(tmp_path, without_toolkit(GPUSYNC_NVRTC_DIRS="", GPUSYNC_CACHE="0"))
    assert "CUDA indisponible" in out and "pip install nvidia-cuda-nvrtc-cu12" in out


@needs_cuda
def test_nvrtc_from_pip(tmp_path):
    import glob

    if not any(glob.glob(os.path.join(p, "nvidia", "cuda_nvrtc", "bin", "*nvrtc*")) or
               glob.glob(os.path.join(p, "nvidia", "cuda_nvrtc", "lib", "*nvrtc*")) for p in sys.path):
        pytest.skip("nvidia-cuda-nvrtc-cu12 wheel not installed")
    out = run(tmp_path, without_toolkit(GPUSYNC_CACHE="0"))
    assert "CUDA ok" in out and "RESULT True" in out


@needs_cuda
def test_f64_full():
    """CUDA provides the whole math library in f64 (unlike Vulkan)."""
    @gs.kernel
    def f(x: gs.f64[:]) -> gs.f64[:]:
        return gs.exp(x) * gs.log(x + 2.0) + gs.sin(x)

    x = np.linspace(0, 3, 10_001)
    np.testing.assert_allclose(np.asarray(f(x, device="cuda:0")), np.exp(x) * np.log(x + 2) + np.sin(x), rtol=1e-14)
