# Reference

Every public name of `gpusync` (imported as `gs`). The graphics engine `gs.gfx` is not described here.

## Decorator and kernel

### `@gs.kernel` / `@gs.kernel(cast=None, compat="auto", grid=None, group=None, cells=None)`

Turns an annotated function into a `gs.Kernel`. See [Writing a kernel](kernels.md).

| Parameter | Values | Effect |
|---|---|---|
| `cast` | `None`, `"auto"` | `"auto"`: converts arrays of another numeric type (one copy) instead of raising `gs.TypeMismatch` |
| `compat` | `"auto"`, `"strict"` | operation missing on the device: fallback with `gs.CompatibilityWarning`, or `gs.UnsupportedFeature` |
| `grid` | name of an input, or tuple of dimensions | explicit mode: one thread per grid cell, `gs.Out` outputs ([Explicit mode](atomics.md)) |
| `group` | integer from 1 to 1024 | thread groups: `gs.shared`, `gs.lane`, `gs.barrier` ([Thread groups](groups.md)) |
| `cells` | integer ≥ 1, with `group` | several output cells per thread, written with `gs.store` |

### `gs.Kernel`

| | |
|---|---|
| `k(*args, device=None, out=None)` | runs the computation; returns a `gs.Array`, or a tuple of `gs.Array` (several outputs or several `gs.Out`). `out=`: NumPy destination(s), or starting contents of the `gs.Out` |
| `k.stream(source, *rest, device=None)` | iterator of NumPy results, one per chunk of `source`; the next chunk is launched before the current one is returned |
| `k.source(target="opencl")` | generated code: `"cuda"`, `"opencl"`, `"wgsl"`, `"ir"` |
| `k.py_func` | the original Python function |
| `repr(k)` | the signature, e.g. `<gs.kernel f(x: gs.f32[:]) -> gs.f32[:]>` |

## Types

| | |
|---|---|
| `gs.f32`, `gs.f64`, `gs.i32`, `gs.u32`, `gs.i64`, `gs.u64` | scalar types: parameter annotation, conversion inside a kernel (`gs.i32(v)`), conversion outside (`gs.f32(3)` → `numpy.float32(3.0)`); `.np` gives the NumPy type |
| `gs.f32[:]` | array of any shape (element-wise) |
| `gs.f32["n", "t"]`, `gs.f32[:, :]`, `gs.f32["n", 3]` | array with named, unnamed or fixed-size dimensions |
| `gs.Out[gs.i32[10]]`, `gs.Out[gs.f32["m"], gs.inf]` | explicit-mode output, with its start value (0 by default) |

## Inside a kernel

Outside a kernel, the math functions are NumPy's; `gs.index`, `gs.lane`, `gs.barrier` and `gs.store` raise an
error.

| Name | Where | Effect |
|---|---|---|
| `gs.index()` | indexed, explicit mode | coordinates of the cell (an integer, or a tuple to unpack: `i, j = gs.index()`) |
| `gs.local(type, n)` | everywhere | array private to the thread, zero-filled, constant size |
| `gs.shared(type, n)` | `group=` | array shared by the group, zero-filled, constant size |
| `gs.lane()` | `group=` | rank of the thread in the group |
| `gs.barrier()` | `group=` | waits for the whole group |
| `gs.store(k, v)` | `cells=` | writes the thread's k-th cell |
| `gs.matmul_u8(c, a, b, m, n, k)` | `group=` | byte matrix product, collective |
| `gs.atomic_add(out, i, v)`, `gs.atomic_min`, `gs.atomic_max` | `grid=` | atomic write into a `gs.Out`; `i` integer or tuple |
| `gs.sqrt`, `exp`, `exp2`, `log`, `log2`, `pow`, `sin`, `cos`, `tan`, `tanh`, `atan2`, `floor`, `ceil`, `abs` | everywhere | native GPU functions (also through `math.*`) |
| `gs.min(a, b, …)`, `gs.max(a, b, …)` | everywhere | two values or more |
| `gs.mulhi(a, b)` | everywhere | high half of the full product of two integers of the same type |
| `gs.pi`, `gs.e`, `gs.tau`, `gs.inf`, `gs.nan` | everywhere | constants |
| `x.shape[k]`, `len(x)` | indexed | sizes of an array |
| `float()`, `int()`, `gs.f32()`… | everywhere | conversions (`int` truncates toward zero) |

## `gs.Array`

Result of a kernel. See [Results and large data](results.md).

| | |
|---|---|
| `shape`, `dtype`, `ndim`, `size`, `nbytes`, `device` | immediate |
| `mode` | `"direct"`, `"resident"`, `"flux"` or `"cpu"` (`"host"` for an empty result) |
| `zero_copy` | `True` in `direct` mode |
| `ready` | the result has already been read |
| `numpy()`, `np.asarray(y)`, `y[...]`, `len(y)`, `iter(y)`, `print(y)` | wait and copy back |
| `sum`, `mean`, `min`, `max`, `std` (`axis=`, NumPy options) | reductions by NumPy after reading |
| `save(path)` | `.npy` file, written chunk by chunk |
| `await y` | waits without blocking the asyncio loop; returns `y` |
| `__dlpack__`, `__dlpack_device__` | DLPack export (host memory) |

Operators (`+`, `*`, `==`, `<`, `-y`…) and NumPy ufuncs: computed on the host after reading, they return a NumPy
array. A `gs.Array` stays hashable (by identity).

A kernel with several outputs returns a tuple of `gs.Array` that can be awaited at once: `a, b = await k(x)`.

## Devices

| | |
|---|---|
| `gs.devices(refresh=False)` | list of `gs.Device` |
| `gs.backends(refresh=False)` | list of `gs.BackendStatus` (`name`, `available`, `error`, `devices`) |
| `gs.device(dev)` | `with` context: device of the block (per thread); yields the `gs.Device` |
| `gs.set_default_device(dev)` | default device (`None`: the ranking) |
| `gs.default_device(kernel=None)` | device that would be used now |
| `gs.set_memory_budget(nbytes=None)` | budget of `resident` mode (`None`: 50 % of the device memory) |
| `gs.batch(device=None)` | `with` context: a single submission for the block's kernels (wgpu) |
| `gs.sync(device=None)` | waits for submitted work, without copying anything back |
| `gs.Device` | `id`, `name`, `backend`, `kind`, `index`, `vendor`, `version`, `memory_bytes`, `max_buffer_bytes`, `unified_memory`, `f64`, `f64_math`, `int64`, `max_group`, `shared_bytes` |

`dev`: a `gs.Device`, an identifier (`"cuda:0"`), a backend name (`"opencl"`) or `"gpu"`. See
[Devices](devices.md).

## Window

`gs.Window(width, height, title="GPUsync", *, vsync=False, resizable=True, device=None)`

| | |
|---|---|
| `poll()` | list of events (dictionaries, key `type`) |
| `present(image)` | shows a `gs.Array` `uint32 (H, W)` or a NumPy array `uint32 (H, W)`, `uint8 (H, W, 4)`, `uint8 (H, W, 3)` |
| `open`, `device`, `size`, `width`, `height`, `physical_size`, `zero_copy` | state |
| `vsync` | readable and writable |
| `mouse_pos`, `keys`, `mouse_buttons`, `key_down(key)` | current input |
| `set_title(title)`, `close()` | |

Usable with `with`. See [Window](window.md).

## Exceptions and warnings

`gs.GpusyncError` (base), `gs.KernelSyntaxError`, `gs.TypeMismatch`, `gs.ShapeMismatch`, `gs.UnsupportedFeature`,
`gs.DeviceError`, `gs.OutOfMemory`, `gs.OutOfBounds`; warning `gs.CompatibilityWarning`. See
[Errors](errors.md).

## Miscellaneous

| | |
|---|---|
| `gs.__version__` | library version |
| `python -m gpusync` | diagnostic of backends and devices |
| `python -m gpusync --check [-d ID]` | also runs a self-test on every device (exit code 1 on failure) |
