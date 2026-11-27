# Results and large data

## `gs.Array`: a result that stays on the GPU

Calling a kernel **queues the work and returns immediately**. The result is a `gs.Array`: a handle to data that
is still on the device.

```python
import numpy as np
import gpusync as gs

@gs.kernel
def scale(x: gs.f32[:], k: gs.f32) -> gs.f32[:]:
    return x * k

@gs.kernel
def offset(x: gs.f32[:], c: gs.f32) -> gs.f32[:]:
    return x + c

x = np.arange(6, dtype=np.float32)
y = scale(x, 2.0)              # returns at once
z = offset(y, 1.0)             # chaining: y does not go back through main memory
print(z.shape, z.dtype, z.device)   # known without waiting for the computation
print(z[:3])                   # reading: waits for the end and copies back
```

| On a `gs.Array` | Effect |
|---|---|
| passing it to another kernel | no wait, no copy if it is the same device |
| `y.shape`, `y.dtype`, `y.ndim`, `y.size`, `y.nbytes`, `y.device` | immediate |
| `np.asarray(y)`, `y.numpy()`, `y[...]`, `len(y)`, `print(y)`, iteration | waits for the computation, then copies back (once) |
| `y.sum()`, `y.mean()`, `y.min()`, `y.max()`, `y.std()` (with `axis=`) | reductions computed by NumPy after reading |
| `y + 1`, `y == 2`, `np.sqrt(y)`… | NumPy operators and functions, on the host after reading: they return a NumPy array |
| `await y` | waits without blocking the asyncio loop |
| `y.save("r.npy")` | writes a `.npy` file, chunk by chunk from the GPU |
| `np.from_dlpack(y)`, `torch.from_dlpack(y)` | DLPack export of the host memory, without copy |
| `y.ready` | `True` if the result has already been read |
| `y.mode` | how the call was executed (see below) |

**Operators read the result.** `z + 1`, `z * z`, `z == 2`, `-z`, `np.maximum(z, 0)` wait for the computation,
copy back, and return a **NumPy array** computed on the host. It is handy to check or post-process; to keep the
computation on the GPU, write a kernel.

```python
r = z + 1                      # np.ndarray
assert isinstance(r, np.ndarray) and r[0] == 2.0
assert (z == 1.0).sum() == 1   # value-by-value comparison
```

**Do not modify an input array before reading the result.** On a unified-memory GPU (Mac, integrated GPUs), the
GPU reads the NumPy array's memory directly, without copying: changing it during the computation changes the
result.

## Writing into a given array: `out=`

```python
dest = np.empty(6, np.float32)
r = scale(x, 3.0, out=dest)    # the result is written into dest
r.numpy()                      # waits for the end
assert dest[5] == 15.0
```

A kernel with several outputs takes a tuple: `out=(a, b)`. The destination must have the exact shape and type of
the output, and be contiguous and writable.

## No size limit

Inputs and outputs can be larger than the GPU's memory, or than main memory. GPUsync then splits the computation
into chunks and overlaps transfers with computation; the kernel does not change.

```python
n = 1_000_000
big = np.lib.format.open_memmap("input.npy", mode="w+", dtype=np.float32, shape=(n,))
big[:] = 1.0
out = np.lib.format.open_memmap("output.npy", mode="w+", dtype=np.float32, shape=(n,))
scale(big, 2.0, out=out).numpy()       # file → GPU → file, chunk by chunk
assert out[-1] == 2.0
```

For a result larger than main memory, always give an `out=` destination (an `np.memmap`): without it, the output
has to be allocated in memory, and `gs.OutOfMemory` says so when that is impossible.

In an indexed kernel, the output is split into blocks of rows (its first axis). An input read only at the current
row (`x[i, k]` in a kernel where `i` is the row) is split along with it; other inputs stay whole on the GPU and
must fit there. Measured: 19.3 GiB of pairwise distances computed on a 16 GB RTX in 3.8 s.

`y.save()` writes a result still on the GPU straight to a file, chunk by chunk:

```python
scale(x, 2.0).save("double.npy")
assert np.load("double.npy")[5] == 10.0
```

### Streams of chunks: `kernel.stream()`

For data that arrives in pieces (a sensor, a split file), `stream` takes an iterable of chunks and returns an
iterator of NumPy results. The next chunk is launched before the current chunk's result is returned: computation
and your own processing overlap.

```python
def chunks():
    for k in range(4):
        yield np.full(1000, k, np.float32)

total = 0.0
for res in scale.stream(chunks(), 0.5):    # the remaining arguments complete each call
    total += float(res.sum())
assert total == 0.5 * 1000 * (0 + 1 + 2 + 3)
```

An item of the stream can be a tuple: it then gives the first arguments of the kernel.

## Execution modes

Each call picks the most economical mode by itself, shown in `y.mode`:

| Mode | When | Extra memory |
|---|---|---|
| `direct` | unified-memory GPU, NumPy inputs: the GPU works in the arrays' own memory | ≈ 0 |
| `resident` | the call fits in the device budget (50 % of its memory by default); or an input is already a result on the GPU | bounded by the budget |
| `flux` | beyond that: chunk by chunk (streaming), transfers overlapped with computation | ≈ 0.3 to 1.2 GiB |
| `cpu` | `cpu:0` interpreter | |

The budget is set with `gs.set_memory_budget(nbytes)` (see [Devices](devices.md)).

## Asynchronous: `await`, threads, Jupyter

`await y` waits for the computation on a thread, without blocking the asyncio loop; a kernel with several outputs
is awaited in one go. The same code works in a script, in Jupyter (which already has a loop) and in an
asynchronous server.

```python
import asyncio

@gs.kernel
def minmax(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    lo = x[i, 0]
    hi = x[i, 0]
    for k in range(x.shape[1]):
        lo = gs.min(lo, x[i, k])
        hi = gs.max(hi, x[i, k])
    return lo, hi

async def main():
    y = await scale(x, 2.0)                     # y is ready
    lo, hi = await minmax(np.random.rand(8, 50).astype(np.float32))
    # a large chunked call blocks its thread: move it out of the loop
    big2 = await asyncio.to_thread(scale, big, 3.0, out=out)
    return y, lo, hi

y, lo, hi = asyncio.run(main())
assert (np.asarray(lo) <= np.asarray(hi)).all()
```

Kernel calls are safe from several threads.

## Interoperability

`gs.Array` implements DLPack (host-memory export, without copy); a kernel accepts as input any DLPack object in
host memory (CPU PyTorch tensor, JAX array…).

```py
import torch
t = torch.from_dlpack(scale(x, 2.0))              # CPU tensor, no copy
scale(torch.ones(10), 2.0)                        # PyTorch input accepted
scale(torch.ones(10, device="cuda"), 2.0)         # rejected: move it to host memory first (.cpu())
```

## Several kernels in one submission: `gs.batch` and `gs.sync`

On Metal, Vulkan and DX12, each call is one submission to the GPU, which costs a few tens of microseconds. For a
loop of dozens of small kernels (a simulation, a renderer), `gs.batch()` groups them into a single submission,
sent when the block exits:

```python
a = np.zeros(100_000, np.float32)
y = scale(a, 1.0)
with gs.batch():
    for _ in range(20):
        y = offset(y, 1.0)                     # recorded, not submitted yet
gs.sync()                                      # waits for the end, without copying anything back
assert np.asarray(y)[0] == 20.0
```

- A read (`y.numpy()`…) inside the block submits what came before.
- On OpenCL, CUDA and `cpu:0`, the block has no effect (the code stays correct everywhere).
- `gs.sync(device)` waits for everything submitted to a device. In a long loop that never reads its results, call
  it now and then (every few iterations): otherwise pending work, and the buffers it holds, pile up in GPU
  memory.
