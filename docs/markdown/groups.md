# Thread groups and shared memory

On a GPU, threads are grouped: threads of the same group can share a small fast memory and wait for one another.
This is what makes reductions, transforms (FFT, NTT) and matrix-product tiles possible. By default GPUsync forms
the groups itself; `group=` makes them visible.

## `group=`, `gs.shared`, `gs.lane`, `gs.barrier`

`@gs.kernel(group=g)` (indexed kernels, 1 ≤ g ≤ 1024) groups `g` **consecutive** cells of the output (in NumPy
order, a group may span several rows). Inside the kernel:

| | |
|---|---|
| `t = gs.shared(type, size)` | array shared by the group, zero-filled, of constant size |
| `gs.lane()` | rank of the thread in its group, from 0 to g − 1 |
| `gs.barrier()` | waits for the whole group; its writes to `gs.shared` are then visible |

```python
import numpy as np
import gpusync as gs

@gs.kernel(group=256)
def group_sum(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, 256)
    t[r] = x[i]
    gs.barrier()
    s = 128
    while s > 0:                     # tree reduction: 8 steps for 256 values
        if r < s:
            t[r] = t[r] + t[r + s]
        gs.barrier()
        s = s // 2
    return t[0]                      # every cell gets the sum of its group

x = np.random.rand(4096).astype(np.float32)
sums = np.asarray(group_sum(x)).reshape(16, 256)
assert np.allclose(sums[:, 0], x.reshape(16, 256).sum(axis=1), rtol=1e-5)
```

Rules:

- **The output has a multiple of g cells**; otherwise the call is rejected with a clear message.
- **A barrier is reached by the whole group.** GPUsync checks it at compile time: a barrier under a condition or
  inside a loop that depends on the thread (on `gs.index()`, `gs.lane()`, a read of `gs.shared` or anything
  computed from them), or after a `return` taken by only some threads, is rejected with its line. On a GPU the
  mistake would cause a hang or a wrong result.
- **No concurrent access without a barrier.** Between two barriers, a `gs.shared` cell written by one thread must
  not be read or written by another. On a GPU the mistake goes unnoticed; on `cpu:0`, the interpreter reports it
  ("concurrent access to t[12]: thread 3 writes it and thread 7 reads it…"). Run a new group kernel once on
  `cpu:0` (`device="cpu"`).
- Limits depend on the device: `gs.Device.max_group` (threads per group) and `shared_bytes` (shared memory per
  group). Examples: RTX 4080 SUPER 1,024 threads and 48 KiB; Apple M5 Pro 1,024 threads with Metal, 256 with
  OpenCL, 32 KiB. Exceeding them raises `gs.UnsupportedFeature`.

## Several cells per thread: `cells=` and `gs.store`

`@gs.kernel(group=g, cells=c)` launches c times fewer threads. A group owns a tile of g × c consecutive cells; the
thread of rank r writes its cells r, r + g, r + 2g… with `gs.store(k, value)` (k from 0 to c − 1), and returns
nothing. `gs.index()` gives the coordinates of its first cell. This lets a group process more points in shared
memory than it has threads (an 8,192-point transform with 1,024 threads).

```python
@gs.kernel(group=128, cells=4)
def reverse_tiles(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()                   # first cell of the thread: start of the tile + rank
    r = gs.lane()
    start = i - r
    t = gs.shared(gs.f32, 512)
    for k in range(4):
        t[r + k * 128] = x[start + r + k * 128]
    gs.barrier()
    for k in range(4):
        gs.store(k, t[511 - (r + k * 128)])   # each tile of 512 values, reversed

x = np.arange(1024, dtype=np.float32)
y = np.asarray(reverse_tiles(x))
assert np.array_equal(y, x.reshape(2, 512)[:, ::-1].ravel())
```

`cells=` requires a single output, whose number of cells is a multiple of g × c.

## Byte matrix product: `gs.matmul_u8`

`gs.matmul_u8(C, A, B, m, n, k)` is a **collective** operation (the whole group calls it, like a barrier) on
`gs.shared` arrays of `gs.u32`: `A` holds m rows of k bytes, `B` n columns of k bytes, packed 4 bytes per word
(least significant byte first), and `C[i·n + j] += Σₜ A[i, t] · B[j, t]`, modulo 2³².

With CUDA, on an Ampere card or newer, with m, n, k multiples of 16, 8 and 32 and a group of 32 threads, the
computation goes through the **Tensor Cores** (measured: 23.6 TOPS on an RTX 4080 SUPER, against 1.4 to 1.7 with
the portable loop). Elsewhere, a portable loop gives exactly the same result.

```python
@gs.kernel(group=32)
def byte_matmul(a: gs.u32["ma"], b: gs.u32["nb"], ref: gs.u32["out"]) -> gs.u32["out"]:
    i = gs.index()
    r = gs.lane()
    sa = gs.shared(gs.u32, 128)      # 16 rows × 32 bytes = 128 words
    sb = gs.shared(gs.u32, 64)       # 8 columns × 32 bytes = 64 words
    sc = gs.shared(gs.u32, 128)      # 16 × 8 results
    for q in range(4):
        sa[r + 32 * q] = a[r + 32 * q]
    for q in range(2):
        sb[r + 32 * q] = b[r + 32 * q]
    gs.matmul_u8(sc, sa, sb, 16, 8, 32)
    return sc[i % 128]

A = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
B = np.random.randint(0, 256, (8, 32), dtype=np.uint8)
C = byte_matmul(A.view(np.uint32).ravel(), B.view(np.uint32).ravel(), np.zeros(128, np.uint32))
assert np.array_equal(np.asarray(C).reshape(16, 8), (A.astype(np.uint64) @ B.T.astype(np.uint64)).astype(np.uint32))
```

The `ref` parameter only gives the size of the output (128 cells, i.e. 4 groups that each redo the same product).
