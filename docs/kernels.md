# Writing a kernel

A kernel is a Python function decorated with `@gs.kernel`. You do not write a loop over the data: you write how
**one cell of the result** is computed, and the GPU runs it for every cell at once, each on its own thread.

There are three kinds of kernel:

| Kind | You write | For |
|---|---|---|
| element-wise | a formula over arrays of the same shape | point-wise transformations |
| indexed (`gs.index()`) | how cell `(i, j…)` of the output is computed, reading anywhere in the inputs | distances, convolutions, per-row reductions, everything else |
| explicit (`grid=`) | what one thread per **input** does, with atomic writes | histograms, scattering: see [Explicit mode](atomics.md) |

A kernel must be defined in a `.py` file or a notebook cell: GPUsync reads its source code when it is decorated.
Defined in `python -c` or in the interactive console, it is rejected with a message that says so.

## Element-wise

```python
import numpy as np
import gpusync as gs

@gs.kernel
def blend(a: gs.f32[:], b: gs.f32[:], t: gs.f32) -> gs.f32[:]:
    return a * (1.0 - t) + b * t

a = np.zeros((480, 640), np.float32)
b = np.ones((480, 640), np.float32)
c = blend(a, b, 0.25)
assert c.shape == (480, 640) and np.allclose(c, 0.25)
```

- `gs.f32[:]` accepts an array of **any shape**; all arrays of a call must have the same shape, which is the shape
  of the output. There is no NumPy-style broadcasting: a value shared by every cell is passed as a scalar
  (`t: gs.f32`).
- The return type is required (`-> gs.f32[:]`).
- In the body, an array stands for **its value in the current cell**: `a * (1.0 - t)` multiplies two numbers.

You can use variables, conditions and loops:

```python
@gs.kernel
def clamp_or_log(x: gs.f32[:], limit: gs.f32) -> gs.f32[:]:
    if x > limit:
        return limit
    y = 0.0
    for k in range(3):
        y += gs.log(1.0 + x * (k + 1))
    return y

r = clamp_or_log(np.array([0.5, 2.0, 50.0], np.float32), 10.0)
assert np.allclose(r[2], 10.0)
```

## Indexed

`gs.index()` gives the coordinates of the output cell being computed. Array dimensions have names; the output
shape is written with these names, and GPUsync works out its size:

```python
@gs.kernel
def pairwise_dist(x: gs.f32["n", "d"]) -> gs.f32["n", "n"]:
    i, j = gs.index()                      # cell (i, j) of the output
    acc = 0.0
    for k in range(x.shape[1]):
        diff = x[i, k] - x[j, k]
        acc += diff * diff
    return gs.sqrt(acc)

pts = np.random.rand(500, 3).astype(np.float32)
dist = pairwise_dist(pts)                  # (500, 3) → (500, 500)
ref = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
assert np.allclose(dist, ref, atol=1e-5)
```

Dimension rules:

| Written | Meaning |
|---|---|
| `gs.f32["n", "t"]` | two named dimensions; a name means the same size everywhere it appears |
| `gs.f32[:, :]` | two unnamed dimensions, any size, readable through `x.shape[k]` |
| `gs.f32["n", 3]` | fixed size: the call is rejected if the axis does not have 3 elements |
| `-> gs.f32["n"]` | the output: only input dimension names or integer sizes |

A mismatch (`'a' → 3, 'b' → 4` for the same dimension `n`) raises `gs.ShapeMismatch` before anything runs.

In the body you index with integers (`x[i, k]`), and read `x.shape[k]` and `len(x)` (the first dimension). No
slices (`x[i, :]`): write the loop.

```python
@gs.kernel
def row_mean(x: gs.f32["n", "t"]) -> gs.f32["n"]:
    i = gs.index()
    s = 0.0
    for k in range(x.shape[1]):
        s += x[i, k]
    return s / x.shape[1]

m = np.arange(12, dtype=np.float32).reshape(3, 4)
assert np.allclose(row_mean(m), m.mean(axis=1))
```

### Several outputs

A tuple of types as the return type, a tuple of values in `return`; the call returns a tuple of results:

```python
@gs.kernel
def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    t = x.shape[1]
    s = 0.0
    for k in range(t):
        s += x[i, k]
    mean = s / t
    v = 0.0
    for k in range(t):
        d = x[i, k] - mean
        v += d * d
    return mean, gs.sqrt(v / t)

series = np.random.rand(100, 365).astype(np.float32)
mean, std = stats(series)
assert np.allclose(std, series.std(axis=1), atol=1e-5)
```

All outputs have the same shape.

### Local arrays

`gs.local(type, size)` creates an array private to each cell, zero-filled, of constant size:

```python
@gs.kernel
def histogram_per_row(x: gs.f32["n", "t"]) -> gs.i32["n"]:
    i = gs.index()
    bins = gs.local(gs.i32, 10)
    for k in range(x.shape[1]):
        b = gs.min(int(x[i, k] * 10.0), 9)
        bins[b] += 1
    best = 0
    for b in range(10):
        if bins[b] > bins[best]:
            best = b
    return best                            # the most populated bin of the row

data = np.random.rand(4, 1000).astype(np.float32)
assert histogram_per_row(data).shape == (4,)
```

A local array lives in the thread's private memory: a few tens to a few hundred elements at most; beyond that it
gets slow.

## Types

| Type | NumPy | Notes |
|---|---|---|
| `gs.f32` | `float32` | the usual GPU compute type |
| `gs.f64` | `float64` | missing on Metal and on many consumer GPUs: see [Devices](devices.md) |
| `gs.i32`, `gs.u32` | `int32`, `uint32` | |
| `gs.i64`, `gs.u64` | `int64`, `uint64` | 64-bit division and remainder are slow on Metal (prefer OpenCL on a Mac for these) |

Arrays must have **exactly** the declared type. A `float64` array passed to a `gs.f32[:]` parameter is rejected
(`gs.TypeMismatch`, with the fix to apply); `@gs.kernel(cast="auto")` allows the conversion, which costs a copy.

```python
@gs.kernel(cast="auto")
def twice(x: gs.f32[:]) -> gs.f32[:]:
    return x * 2.0

assert twice(np.arange(3)).dtype == np.float32      # 64-bit integers converted to float32
```

Inside a kernel:

- **literals take the type of their context**: `0.0` in an f32 computation is an f32 (it never forces f64);
- mixed operations follow `integer op float → float`, never to f64 unless asked; between integers, lossless
  widening is implicit (i32 → i64, u32 → i64, u32 → u64), the rest is converted by hand;
- conversions: `float(v)`, `int(v)` (truncation toward zero, as in Python), `gs.f32(v)`, `gs.i32(v)`, `gs.u32(v)`,
  `gs.i64(v)`…;
- `//` and `%` behave as in Python, negatives included (`-7 // 3 == -3`, `-7 % 3 == 2`);
- variables are typed at their first assignment; giving one name values of different types (an integer counter,
  then a float) is rejected or causes errors: use two names.

```python
@gs.kernel
def divmod_like_python(x: gs.i32[:]) -> gs.i32[:]:
    return x // 3 * 100 + x % 3

assert list(divmod_like_python(np.array([-7, 7], np.int32))) == [-298, 201]
```

## What you can write in a kernel

| Allowed | Rejected |
|---|---|
| arithmetic, comparisons, `and` / `or` / `not` | lists, tuples (except `gs.index()` and `return`), dicts, strings |
| bitwise `&` `\|` `^` `~` `<<` `>>` on integers; `&` `\|` `^` on booleans | calls to Python functions (including your own) |
| `a if c else b`, `if` / `elif` / `else` | recursion, `lambda`, generators |
| `for k in range(...)`, `while`, `break`, `continue` | `try`, `with` |
| scalar local variables, `gs.local` | slices `x[a:b]`, variable-size allocation |
| numeric global constants of the module | objects, attributes other than `x.shape` |
| the `gs.*` and `math.*` functions listed below | |

A rejected construct raises `gs.KernelSyntaxError` at decoration time, with the file, the line and the reason.

Math functions (`gs.name` or `math.name`):

| | |
|---|---|
| roots, exponentials | `sqrt`, `exp`, `exp2`, `log`, `log2`, `pow` |
| trigonometry | `sin`, `cos`, `tan`, `tanh`, `atan2` |
| rounding | `floor`, `ceil`, `abs` |
| extremes | `gs.min(a, b, …)`, `gs.max(a, b, …)` (2 values or more) |
| integers | `gs.mulhi(a, b)`: high half of the full product (e.g. `(a * b) >> 32` without overflow, for u32) |
| constants | `gs.pi`, `gs.e`, `gs.tau`, `gs.inf`, `gs.nan` (and `math.pi`…) |

```python
import math

GAIN = 3.0                                  # global constant: read at decoration time

@gs.kernel
def gained(x: gs.f32[:]) -> gs.f32[:]:
    return math.sqrt(x) * GAIN + gs.max(x, 0.0, -1.0)

assert np.allclose(gained(np.array([4.0], np.float32)), 10.0)
```

A global constant is frozen when the kernel is decorated: changing it afterwards does not change the kernel.

## The Python function stays usable

Outside a kernel, `gs.sqrt`, `gs.exp`… are the matching NumPy functions, and `kernel.py_func` is the original
function: an element-wise kernel written with `gs.*` functions also runs in plain NumPy, which is handy to
compare.

```python
assert np.allclose(blend.py_func(a, b, 0.25), blend(a, b, 0.25))
```

This does not hold with `math.*` (which does not accept arrays), nor for indexed kernels: `gs.index()` has no
meaning outside the GPU.

## Compilation

A kernel is translated once when it is decorated (type checking, internal representation), then compiled for a
device on the first call that runs it there. Later calls reuse the compiled code. `kernel.source(target)` shows
the generated code (`"cuda"`, `"opencl"`, `"wgsl"` or `"ir"`):

```python
print(blend.source("opencl"))
```
