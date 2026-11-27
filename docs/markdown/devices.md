# Devices

## Listing

```python
import numpy as np
import gpusync as gs

for d in gs.devices():
    print(d.id, d.name, d.backend, d.kind, d.memory_bytes, d.f64, d.unified_memory)
```

Attributes of a `gs.Device`:

| Attribute | Meaning |
|---|---|
| `id` | identifier, e.g. `"metal:0"`, `"cuda:0"`, `"opencl:1"`, `"cpu:0"` |
| `name`, `vendor`, `version` | hardware name, vendor, driver or API version |
| `backend` | `"cuda"`, `"metal"`, `"vulkan"`, `"dx12"`, `"opencl"` or `"cpu"` |
| `kind` | `"discrete_gpu"`, `"integrated_gpu"`, `"cpu"`… |
| `index` | number within its backend |
| `memory_bytes`, `max_buffer_bytes` | device memory, largest possible buffer (`None` if unknown) |
| `unified_memory` | the GPU shares main memory (Apple Silicon, integrated GPUs) |
| `f64` | `gs.f64` computation is possible |
| `f64_math` | math functions (`exp`, `sin`…) in f64 are possible |
| `int64` | `gs.i64` / `gs.u64` computation is possible |
| `max_group`, `shared_bytes` | largest thread group and its shared memory (see [Thread groups](groups.md)) |

Discovery runs once; `gs.devices(refresh=True)` runs it again. `gs.backends()` gives the status of each backend
and, when it is unavailable, the reason (`st.error`).

## Automatic choice

Without any instruction, GPUsync takes the first device of this ranking:

| Rank | Category | Backends, in order |
|---|---|---|
| 1 | discrete GPU | CUDA, Vulkan, OpenCL, DX12, Metal |
| 2 | integrated GPU | Metal, OpenCL, Vulkan, DX12 |
| 3 | OpenCL on a CPU (POCL…) | |
| 4 | interpreter | `cpu:0` |

On a Mac: `metal:0`. On a PC with an NVIDIA card: `cuda:0` if NVRTC is present, otherwise `vulkan:0`.

## Choosing yourself

A device is named by a `gs.Device`, its identifier (`"opencl:1"`), a backend name (`"cuda"`, `"opencl"`,
`"metal"`, `"vulkan"`, `"dx12"`, `"cpu"`: the first of that backend) or `"gpu"` (the first GPU of the ranking).

```python
@gs.kernel
def square(x: gs.f32[:]) -> gs.f32[:]:
    return x * x

x = np.arange(4, dtype=np.float32)

square(x, device="cpu")                 # for one call

with gs.device("cpu"):                  # for a block of code (per thread)
    y = square(x)
assert y.device == "cpu:0"

gs.set_default_device("cpu")            # for the rest of the program
gs.set_default_device(None)             # back to the ranking
```

Priority: the call's `device=`, then `with gs.device(...)`, then `gs.set_default_device(...)`, then the
`GPUSYNC_DEVICE` environment variable, then the ranking. A name that cannot be found raises `gs.DeviceError`,
which lists the available devices.

`y.device` always tells where the computation actually ran, and `gs.default_device(kernel)` where it would run
now, given what the kernel needs.

## When an operation is missing: `compat`

Not every device can do everything: no f64 on Metal or on most integrated GPUs, no f64 math functions on Vulkan,
64-bit integers missing in places. GPUsync works out what each kernel requires, and if the planned device is not
suitable:

| `@gs.kernel(compat=...)` | Behaviour |
|---|---|
| `"auto"` (default) | a `gs.CompatibilityWarning` (once per kernel), then **the whole kernel** runs elsewhere: CUDA or OpenCL on the same GPU if possible, otherwise another capable device, otherwise `cpu:0` (exact but slow, and the warning says so) |
| `"strict"` | `gs.UnsupportedFeature`, with the list of compatible devices |

```python
import warnings

@gs.kernel
def decay(x: gs.f64[:]) -> gs.f64[:]:
    return gs.exp(-x)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = decay(np.ones(3))
print(r.device, [str(w.message) for w in caught])   # on a Mac: cpu:0, with the explanation
assert np.allclose(r, np.exp(-1.0))
```

Nothing is emulated: an f64 is never simulated with two f32, and a kernel is never split across two devices. With
an explicit `device=`, the requested device is used as is, or the call is rejected.

On the test PC, for instance: an f32 kernel → `cuda:0`; f64 arithmetic under Vulkan → `vulkan:0`; f64 with
`gs.exp` under Vulkan → `opencl:0` (same card).

## Memory

A device's **budget** is the memory a call may take at once before switching to `flux` mode (chunk by chunk):
50 % of the device memory by default.

```python
gs.set_memory_budget(512 * 2**20)       # 512 MiB
gs.set_memory_budget(None)              # back to the default
```

Lowering it leaves room for other programs (or for rendering); raising it avoids splitting computations that
fit. It does not limit data size: beyond it, the computation is merely split.

## When a GPU drops out

If the driver loses the device (on Windows, a computation longer than about 2 s is interrupted by the system),
the call raises `gs.DeviceError` with an explanation, and the device is reopened on the next call. Split the work
into shorter calls.
