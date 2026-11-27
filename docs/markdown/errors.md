# Errors, debugging, performance

## Exceptions

All derive from `gs.GpusyncError`. Messages say what is wrong, where, and how to fix it.

| Exception | When |
|---|---|
| `gs.KernelSyntaxError` | at decoration: construct rejected in the kernel, missing return type, source not found |
| `gs.TypeMismatch` | at call time: an argument's type differs from the signature |
| `gs.ShapeMismatch` | at call time: inconsistent shapes, unknown output dimension |
| `gs.UnsupportedFeature` | the device cannot do an operation of the kernel (with `compat="strict"` or an explicit `device=`), or exceeds its group limits |
| `gs.DeviceError` | device not found, driver error, GPU lost |
| `gs.OutOfMemory` | the output cannot be allocated in main memory: give `out=` |
| `gs.OutOfBounds` | explicit mode: out-of-bounds atomic write (also an `IndexError`) |
| `gs.CompatibilityWarning` | warning: the kernel runs somewhere other than planned (see [Devices](devices.md)) |

```python
import numpy as np
import gpusync as gs

@gs.kernel
def norm(x: gs.f32[:], m: gs.f32) -> gs.f32[:]:
    return x - m

try:
    norm(np.ones(3), 0.0)                    # float64 instead of float32
except gs.TypeMismatch as e:
    print(e)
# argument 'x': gs.f32[:] expected, got a float64 array. Use x.astype(np.float32),
# or @gs.kernel(cast='auto') to convert

@gs.kernel
def add(a: gs.f32["n"], b: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    return a[i] + b[i]

try:
    add(np.ones(3, np.float32), np.ones(4, np.float32))
except gs.ShapeMismatch as e:
    print(e)                                 # add(): inconsistent dimension 'n': 'a' → 3, 'b' → 4
```

An error in the kernel body is reported at decoration time, with the file and the line:

```text
gs.KernelSyntaxError: kernel 'k' (calc.py), line 8: call to "foo" not supported: only gs and math functions
can be used in a kernel
```

## Debugging

- **Compare with the interpreter.** `device="cpu"` runs the kernel with the reference interpreter: same semantics
  as the GPUs, exact result, slow. A GPU / `cpu:0` difference reveals a race between threads or a precision
  issue. For group kernels, the interpreter also reports concurrent accesses to shared memory.
- **Compare with NumPy.** An element-wise kernel written with `gs.*` functions also runs in NumPy
  (`kernel.py_func`).
- **Read the generated code.** `kernel.source("cuda" | "opencl" | "wgsl" | "ir")`; `GPUSYNC_DUMP_DIR=folder` writes
  each code when it is compiled, specialized variants included.
- **Machine diagnostic.** `python -m gpusync`, `gs.backends()`.

```python
x = np.random.rand(1000).astype(np.float32)
assert np.allclose(norm(x, 0.5), norm(x, 0.5, device="cpu"))
print(norm.source("ir"))
```

## Pitfalls

| Pitfall | What happens | What to do |
|---|---|---|
| `y + 1`, `y == 2` on a `gs.Array` inside a compute loop | each operation waits for the GPU and computes on the host | a kernel, to stay on the GPU |
| `if y:` on a `gs.Array` | true as soon as it has elements, whatever their values (NumPy would raise) | `if (y > 0).all():` |
| modifying a NumPy input before reading the result | on unified memory, the GPU reads the modified memory | read or await the result first |
| kernel defined in the console or `python -c` | rejected: source not found | put it in a file or a notebook cell |
| changing a global constant after decoration | the kernel keeps the old value | pass it as a parameter |
| one name for an integer and then a float in a kernel | rejection or truncated value | two names |
| long loop of kernels that never reads nor calls `gs.sync()` | pending work piles up in GPU memory | `gs.sync()` every few iterations |
| `math.sqrt` in a kernel called through `py_func` | `TypeError` (math does not accept arrays) | `gs.sqrt` |
| f64 on a Mac | falls back to `cpu:0`, exact but slow | f32, or OpenCL / CUDA on a machine with f64 |

## Performance

- **The first call compiles** the kernel for the device (from a few milliseconds to a few hundred for CUDA, cached
  on disk). Measure from the second call on.
- **Each call has a fixed cost** (around 0.1 ms on Vulkan, more with OpenCL). A kernel doing little work per cell
  must process many cells; for a series of small kernels on Metal, Vulkan or DX12, `gs.batch()` submits them at
  once.
- **Keep results on the GPU**: chain kernels on `gs.Array` values and read only at the end.
- **f32 rather than f64**: consumer GPUs compute f64 16 to 64 times slower, when they do at all.
- **64-bit integers**: division and remainder are slow everywhere (emulated), very slow on Metal; `gs.mulhi`
  often lets you stay in 32 bits (Montgomery modular multiplication).
- **Size specialization.** In an indexed kernel, a dimension that is not an axis of the output (the length `t` of
  a series, for instance) and is at most 1,024 becomes a constant of the compiled code: unrolled loops, local
  arrays in registers. Measured gain on a DTW: ×1.5 to ×40 depending on the GPU. At most 8 variants per kernel and
  device; changing the size of the output axes recompiles nothing. `GPUSYNC_SPECIALIZE=0` disables it.
- **Groups and shared memory** (see [Thread groups](groups.md)) for anything that reuses neighbouring data.

## Known limits

- No slices (`x[a:b]`) and no calls to your own functions inside a kernel.
- No broadcasting between arrays of different shapes in an element-wise kernel.
- Non-contiguous arrays (transposed or strided NumPy views): copied to contiguous memory at call time.
- 64-bit `gs.Out` not allowed; `grid=` cannot be combined with `group=`.
- Beyond 2³¹ − 1 elements in one array kept whole on the device: handled by OpenCL, CUDA and `cpu:0` (a 64-bit
  index variant compiled on demand), rejected by wgpu (Metal, Vulkan, DX12).
- A GPU lost by the driver (`gs.DeviceError`) is reopened on the next call, but the running computation is lost.
