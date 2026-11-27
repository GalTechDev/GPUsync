# GPUsync

Write GPU kernels in Python and run them on any GPU: CUDA, Metal, Vulkan, DX12 (through wgpu) and OpenCL ≥ 1.2.
The core is written in Rust; the only runtime dependency is NumPy.

```bash
pip install gpusync
```

```python
import numpy as np
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

x = np.random.rand(1_000_000).astype(np.float32)
y = normalize(x, x.mean(), x.std())     # runs on the best available GPU
print(y[:5])                            # waits for the result and copies it back
z = normalize(y, 0.0, 2.0)              # chaining: y stays on the GPU
```

- **Write one cell, run them all.** Element-wise kernels, indexed kernels (`gs.index()`, named dimensions, several
  outputs, local arrays) and an explicit mode with atomic writes for histograms and scattering.
- **Every GPU, the same code.** The device is picked from what the kernel needs (CUDA, then Vulkan, OpenCL…); when
  an operation is missing (f64 on Metal, for instance), the whole kernel moves to a capable device, with a
  warning. A reference interpreter (`cpu:0`) is always available.
- **No size limit.** Inputs and outputs larger than GPU memory, or than main memory (`np.memmap`, `out=`,
  `kernel.stream()`): the computation is split and transfers overlap with compute.
- **Results stay on the GPU** until read: chaining, `await`, Jupyter, threads, zero-copy DLPack, `y.save()`.
- **Thread groups**: shared memory, barriers checked at compile time, several cells per thread, byte matrix
  products on Tensor Cores (CUDA).
- **Native window** (`gs.Window`) that shows GPU results without a copy.

## Documentation

Full documentation in [docs/markdown](docs/markdown/README.md), also published as a website (generated into
`docs/`): [installation](docs/markdown/installation.md), [writing a kernel](docs/markdown/kernels.md),
[results and large data](docs/markdown/results.md), [devices](docs/markdown/devices.md),
[atomics](docs/markdown/atomics.md), [thread groups](docs/markdown/groups.md), [window](docs/markdown/window.md),
[errors and performance](docs/markdown/errors.md), [reference](docs/markdown/reference.md). Every Python example in
it is run by the tests.

The internal design (IR, code generators, transfer engine, measurements per milestone) is described in
[ARCHITECTURE.md](ARCHITECTURE.md) (in French).

The graphics and simulation engine `gpusync.gfx` (2D / 3D rendering, fluids, atmosphere, weather) is a separate
package, [gpusync-gfx](gfx/README.md), installed on top of the library.

## Status

Version 0.1. Tested on macOS (Apple M5 Pro: Metal, OpenCL) and Windows (RTX 4080 SUPER and Intel
UHD 770: CUDA, Vulkan, OpenCL); Linux is built by the CI but not tested on a GPU yet.

## Development

Requirements: Rust (rustup), Python ≥ 3.10 and a C compiler (Xcode Command Line Tools on macOS, Visual Studio Build
Tools "C++" on Windows).

```bash
python3 tools/dev.py setup      # .venv + maturin, pytest, NumPy, then builds both packages
python3 tools/dev.py test       # rebuilds, then cargo test, pytest and the device diagnostic
python3 tools/build_pages.py    # regenerates the documentation website
```

`GS_EXPECT=metal,opencl` (or `vulkan,opencl`) makes the tests fail if these backends have no device on the
machine. To run on every test machine at once, see [tools/remote](tools/remote/README.md):

```bash
python3 tools/remote/gs_remote.py run all -- @python tools/dev.py test
```

Benchmarks of each milestone are in `tools/bench_m*.py` (see ARCHITECTURE.md for their results).

## Layout

```text
crates/
  gpusync-ir/               typed IR, semantics, reference interpreter
  gpusync-frontend/         Python AST (JSON) → IR: typing, errors
  gpusync-codegen/          IR → OpenCL C, CUDA C, WGSL
  gpusync-runtime/          shared types (DeviceInfo, Backend trait), discovery
  gpusync-backend-wgpu/     Metal, Vulkan, DX12: discovery, execution, window
  gpusync-backend-opencl/   OpenCL (loaded dynamically): discovery, execution, zero-copy
  gpusync-backend-cuda/     CUDA (driver and NVRTC loaded dynamically): discovery, execution
  gpusync-core/             compilation and launch (no Python); multi-GPU differential test
  gpusync-py/               PyO3 binding → gpusync._core module
python/gpusync/             Python package: @gs.kernel, gs.Array, types, devices, gs.Window
gfx/                        gpusync-gfx package: the gpusync.gfx engine and its tests
docs/                       GitHub Pages website, generated from docs/markdown by tools/build_pages.py
docs/markdown/              library documentation (examples checked by tests/test_docs.py)
tests/                      Python tests (differential on every device, errors, docs)
tools/                      dev tasks (dev.py), website generator, benchmarks, remote test machines
```

## Licence

MIT, see [LICENSE](LICENSE).
