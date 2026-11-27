# Installation

```bash
pip install gpusync              # the library
pip install gpusync-gfx          # optional: the graphics and simulation engine
```

Precompiled wheels exist for Windows x64, macOS (Apple Silicon and Intel) and Linux x86_64 / aarch64
(glibc ≥ 2.28), for every Python ≥ 3.10: no Rust or C compiler is needed. The only runtime dependency is NumPy
≥ 1.22. No GPU SDK is needed either: drivers (CUDA, OpenCL, Metal, Vulkan, DX12) are loaded at startup when
present, and a missing one only disables the matching backend.

On another platform, pip builds from the source archive, which requires the tools below.

## From source

To work on GPUsync, build it from the repository. The build produces a native module (`gpusync._core`, written in
Rust) and installs the Python package in editable mode.

### Requirements

| | macOS | Windows | Linux |
|---|---|---|---|
| Rust | `rustup` | `rustup` | `rustup` |
| Python | ≥ 3.10 | ≥ 3.10 | ≥ 3.10 |
| C compiler | Xcode Command Line Tools | Visual Studio Build Tools, "C++" workload | `gcc` or `clang` |

No GPU library is needed to build: drivers (OpenCL, CUDA) are loaded at startup when present, and a missing one
only disables the matching backend.

### Build

```bash
python3 tools/dev.py setup      # creates .venv, installs maturin, pytest, NumPy, then builds
```

After changing Rust code:

```bash
python3 tools/dev.py build      # rebuilds the extension in .venv
python3 tools/dev.py test       # rebuilds, then cargo test, pytest and the diagnostic
```

Run scripts with the Python from `.venv` (`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows), or with
`python3 tools/dev.py run script.py`.

The only runtime dependency is NumPy ≥ 1.22.

`setup` and `build` also install the graphics engine `gpusync.gfx`, which is a separate package
([gpusync-gfx](../gfx/README.md), folder `gfx/`, pure Python). If you only want the library, ignore it: everything
works without it, and `gs.gfx` raises an `ImportError` that gives the install command. To add or remove it by
hand:

```bash
.venv/bin/python -m pip install --no-build-isolation --no-deps -e gfx
.venv/bin/python -m pip uninstall gpusync-gfx
```

## Check what is detected

```bash
.venv/bin/python -m gpusync
```

lists each backend, its devices and, when a backend is unavailable, why (here on the Mac):

```text
gpusync 0.1.0 — Darwin arm64, Python 3.15.2

[cuda] unavailable: no CUDA driver on macOS (no NVIDIA GPU)
[wgpu] 1 device(s)
  metal:0    Apple M5 Pro
             integrated_gpu, Apple, metal | memory ?, buffer max 4.0 GB | unified memory
[opencl] 1 device(s)
  opencl:0   Apple M5 Pro
             integrated_gpu, Apple, OpenCL 1.2 | memory 17.8 GB, buffer max 3.3 GB | unified memory
[cpu] 1 device(s)
  cpu:0      GPUsync interpreter (CPU, slow)
             cpu, GPUsync, 0.1.0 | memory ?, buffer max ? | f64, unified memory

Default device: Device('Apple M5 Pro', backend='metal', f64=False, unified_memory=True)
```

From Python:

```python
import gpusync as gs

for st in gs.backends():                       # one status per backend: cuda, wgpu, opencl, cpu
    print(st.name, st.available, st.error)
for d in gs.devices():                         # usable devices
    print(d.id, d.name, d.kind, "f64" if d.f64 else "no f64")
```

`cpu:0` always exists: it is GPUsync's reference interpreter, exact but slow, useful to check a result or to
debug.

## Check that everything works

A detected device can still fail at the first kernel (old driver, broken OpenCL install). `--check` runs a few
small kernels on every device — element-wise, indexed, atomics, thread groups, f64, chained results — and compares
each result with NumPy:

```bash
python -m gpusync --check
python -m gpusync --check -d opencl:0      # only this device (repeatable)
```

```text
Self-test (each result compared with NumPy):

  metal:0    Apple M5 Pro                     ✓ elementwise  ✓ indexed  ✓ atomics  ✓ groups  – f64  ✓ chained  (0.2 s)
             f64: not supported
  opencl:0   Apple M5 Pro                     ✓ elementwise  ✓ indexed  ✓ atomics  ✓ groups  – f64  ✓ chained  (0.1 s)
             f64: not supported
  cpu:0      GPUsync interpreter (CPU, slow)  ✓ elementwise  ✓ indexed  ✓ atomics  ✓ groups  ✓ f64  ✓ chained  (0.0 s)

All 3 device(s) passed.
```

`✓` passed, `–` skipped (the device does not support it), `✗` failed, with the error below the line. The exit code
is 1 when a test fails, so it can be used in a script or a CI job. On a console that cannot show these marks (Windows code page
1252, output redirected to a file), they become `+`, `-` and `X`.

## Backends

| Backend | Devices | Notes |
|---|---|---|
| CUDA | NVIDIA GPUs | preferred on an NVIDIA GPU; needs the driver and NVRTC (see below) |
| wgpu | Metal (macOS), Vulkan, DX12 | Metal on a Mac; Vulkan on discrete GPUs without CUDA |
| OpenCL ≥ 1.2 | any installed OpenCL driver | zero-copy on unified-memory GPUs; fallback when an operation is missing elsewhere |
| cpu | interpreter | always available |

### CUDA and NVRTC

The CUDA backend compiles kernels with NVRTC, NVIDIA's runtime compilation library. Without it, `gs.backends()`
says so and Vulkan takes over on the same GPU. Two ways to get it:

```bash
pip install nvidia-cuda-nvrtc-cu12      # a library, not a compiler to install
```

or the CUDA toolkit. GPUsync looks for NVRTC in the `nvidia-*` pip wheels visible to Python, then in the toolkit
and on the `PATH`. Compiled code (PTX) is cached on disk.

## Environment variables

| Variable | Effect |
|---|---|
| `GPUSYNC_DEVICE` | default device, e.g. `opencl`, `cuda:0`, `cpu` (see [Devices](devices.md)) |
| `GPUSYNC_CACHE_DIR` | folder of the compiled CUDA code cache |
| `GPUSYNC_CACHE=0` | disables that cache |
| `GPUSYNC_NVRTC` | exact path of the NVRTC library |
| `GPUSYNC_NVRTC_DIRS` | folders where NVRTC is looked for (separated by `:`, or `;` on Windows); empty: do not look in pip wheels |
| `GPUSYNC_SPECIALIZE=0` | disables size specialization (see [Errors, debugging, performance](errors.md)) |
| `GPUSYNC_DUMP_DIR` | writes every generated code to this folder when it is compiled |

## Tests

```bash
python3 tools/dev.py test
```

Differential tests run each kernel on every device of the machine and compare the results with the interpreter.
`GS_EXPECT=metal,opencl` (or `vulkan,opencl`…) makes the suite fail if one of these backends has no device, so
that nothing is believed tested when it is not. The engine tests (`gfx/tests`) run with the others when the
engine is installed, and are skipped otherwise.
