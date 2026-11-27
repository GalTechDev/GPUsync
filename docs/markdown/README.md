# GPUsync documentation

GPUsync runs ordinary Python functions on the GPU: you write how one cell of the result is computed, and the
library translates it for CUDA, Metal, Vulkan, DX12 or OpenCL, picks the device, moves the data and splits large
computations. The same code runs on a Mac, on a PC with an NVIDIA card or on a laptop with an integrated GPU.

```py
import numpy as np
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

x = np.random.rand(1_000_000).astype(np.float32)
y = normalize(x, x.mean(), x.std())     # runs on the GPU
print(y[:5])                            # waits for the result and copies it back
```

This documentation covers the library alone. The graphics engine `gpusync.gfx`, built on top of it, is a separate
package ([gpusync-gfx](../../gfx/README.md)) and is not described here.

## Contents

| Page | Contents |
|---|---|
| [Installation](installation.md) | build, check the detected GPUs, CUDA, environment variables |
| [Writing a kernel](kernels.md) | the three kinds of kernel, types, what you can write inside one |
| [Results and large data](results.md) | `gs.Array`, chaining, `await`, `out=`, files, streams, several kernels in one submission |
| [Devices](devices.md) | automatic choice, manual choice, fallback when an operation is missing, memory |
| [Explicit mode and atomics](atomics.md) | histograms, scattering: one thread per input, `gs.atomic_*` writes |
| [Thread groups](groups.md) | shared memory, barriers, several cells per thread, byte matrix products |
| [Window](window.md) | `gs.Window`: show a GPU result without going through main memory |
| [Errors, debugging, performance](errors.md) | exceptions, generated code, pitfalls, known limits |
| [Reference](reference.md) | every public name, on one page |

For the internal design (IR, code generators, transfer engine, measurements), see
[ARCHITECTURE.md](../../ARCHITECTURE.md).

## Conventions

- Examples marked `python` are run by `tests/test_docs.py`: they work as written. Those marked `py` (window,
  PyTorch, multi-gigabyte files) are not.
- `gs` stands for `gpusync`, `np` for NumPy.
- The documentation website (`docs/`, published on GitHub Pages) is generated from these pages by
  `python3 tools/build_pages.py`; a test checks that it is up to date.
- Measurements come from two machines: a Mac with an Apple M5 Pro chip and a PC with an RTX 4080 SUPER.
