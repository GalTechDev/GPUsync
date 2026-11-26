# Explicit mode and atomic writes

## When do you need it?

In an ordinary kernel, each thread computes **its own** cell of the output, and nobody else touches it. Some
computations do not work that way: the cell to modify **depends on the data**.

- A histogram: each measurement adds 1 to the bin it falls in.
- A per-cell accumulation: each raindrop adds its mass to the grid cell where it reaches the ground.
- A per-zone extreme: each station keeps the lowest temperature of its zone.

Thousands of threads run at the same time on a GPU, and many will target the same cell. But "add 1" happens in
three steps: read the value, add, write it back. If two threads do it together:

| Step | Thread A | Thread B | Value of the cell |
|---|---|---|---|
| 1 | reads 7 | | 7 |
| 2 | | reads 7 | 7 |
| 3 | writes 8 | | 8 |
| 4 | | writes 8 | **8** instead of 9 |

One measurement is lost, and not always the same one from run to run. An **atomic write** does the three steps
as one block: the hardware guarantees that no other thread gets in between. If two threads arrive together, one
waits for the other, and the cell does end up at 9.

## Start from plain Python

Here is a histogram of temperatures, in 10 °C bins from −20 to +40 °C, written in plain Python:

```python
import numpy as np
import gpusync as gs

def histo_python(t):
    counts = np.zeros(6, np.int32)              # 1. create the output, zero-filled
    for i in range(len(t)):                     # 2. go through the measurements
        bin_ = int((t[i] + 20.0) / 10.0)        # 3. compute the bin
        bin_ = min(max(bin_, 0), 5)
        counts[bin_] += 1                       # 4. add 1
    return counts                               # 5. return the result
```

And the same thing as a kernel:

```python
@gs.kernel(grid="t")
def histo(t: gs.f32["n"], counts: gs.Out[gs.i32[6]]):
    i = gs.index()
    bin_ = gs.min(gs.max(int((t[i] + 20.0) / 10.0), 0), 5)
    gs.atomic_add(counts, bin_, 1)

temperatures = np.random.uniform(-25.0, 45.0, 10_000).astype(np.float32)
counts = histo(temperatures)                    # a single argument: see below
assert np.array_equal(counts, histo_python(temperatures))
```

Line by line:

| Plain Python | Kernel | Explanation |
|---|---|---|
| `for i in range(len(t)):` | `grid="t"` and `i = gs.index()` | there is no loop: **each iteration becomes a thread**. `grid="t"` asks for one thread per element of `t`; `gs.index()` gives each thread its number `i` |
| `counts = np.zeros(6, np.int32)` | `counts: gs.Out[gs.i32[6]]` | the output is **declared among the parameters**: "an array of 6 integers, zero-filled". GPUsync creates it |
| `min(...)`, `max(...)` | `gs.min(...)`, `gs.max(...)` | inside a kernel, functions are written with `gs.` |
| `counts[bin_] += 1` | `gs.atomic_add(counts, bin_, 1)` | same meaning, but safe against simultaneous writes. `+=` is rejected on an output |
| `return counts` | nothing | GPUsync returns the outputs itself at the end |
| `histo_python(temperatures)` | `histo(temperatures)` | when calling, you pass **only the inputs** |

### The two surprising points

**1. The output is a parameter you do not pass.** The definition says `histo(t, counts)`, but you call
`histo(temperatures)`, with a single argument. Every parameter of type `gs.Out[...]` is an output: it is left out
of the call, and it is what the call returns. The other parameters are the inputs, in order.

```py
@gs.kernel(grid="t")
def k(t: gs.f32["n"], threshold: gs.f32, counts: gs.Out[gs.i32[6]]): ...

counts = k(temperatures, 0.5)       # inputs: t, threshold; returned: counts
```

**2. No `return` value.** The kernel does not compute a cell: it acts, through its atomic writes. A bare `return`
stops the thread early ("this measurement does not count").

## What you write in the kernel

**The grid** says how many threads to launch:

- `grid="t"`: one thread per element of input `t` (for a 2-D array, one per cell, and `gs.index()` returns two
  numbers: `i, j = gs.index()`);
- `grid=("n",)`: one thread per value of dimension `n`, useful when the input has other axes (for instance
  `xy: gs.f32["n", 2]`, points with two coordinates: one thread per point, not one per coordinate).

**The three writes:**

| | Effect | For |
|---|---|---|
| `gs.atomic_add(output, cell, value)` | `output[cell] += value` | counting, accumulating |
| `gs.atomic_min(output, cell, value)` | `output[cell] = min(output[cell], value)` | the smallest per cell |
| `gs.atomic_max(output, cell, value)` | `output[cell] = max(output[cell], value)` | the largest per cell |

- Allowed types: `gs.i32`, `gs.u32`, `gs.f32`.
- `cell` is an integer, or a tuple for a multi-dimensional output: `(row, column)`.
- You do not **read** an output inside the kernel: a thread cannot know how far the others have got.

## Two complete examples

**Rain accumulated per grid cell.** Each drop (position in [0, 1)², mass) adds its mass to its cell of a 4 × 4
grid. A two-dimensional output, so the cell is a tuple:

```python
@gs.kernel(grid=("n",))                                  # one thread per drop
def rain_map(xy: gs.f32["n", 2], mass: gs.f32["n"], total: gs.Out[gs.f32[4, 4]]):
    i = gs.index()
    col = gs.min(int(xy[i, 0] * 4.0), 3)
    row = gs.min(int(xy[i, 1] * 4.0), 3)
    gs.atomic_add(total, (row, col), mass[i])

xy = np.random.rand(5000, 2).astype(np.float32)
mass = np.random.rand(5000).astype(np.float32)
total = rain_map(xy, mass)

expected = np.zeros((4, 4), np.float32)
np.add.at(expected, ((xy[:, 1] * 4).astype(int), (xy[:, 0] * 4).astype(int)), mass)
assert np.allclose(total, expected, rtol=1e-4)
```

**Lowest temperature per zone.** For a minimum, the output must not start at 0 (the minimum would be wrong as
soon as every value is positive) but at +∞: that is the **start value**, given second in
`gs.Out[type, start]`.

```python
@gs.kernel(grid="t")
def coldest(t: gs.f32["n"], zone: gs.i32["n"], low: gs.Out[gs.f32[3], gs.inf]):
    i = gs.index()
    gs.atomic_min(low, zone[i], t[i])

t = np.array([12.0, 3.5, 8.0, -1.0, 20.0], np.float32)
zone = np.array([0, 0, 1, 1, 2], np.int32)
assert np.array_equal(coldest(t, zone), [3.5, -1.0, 20.0])
```

With several `gs.Out`, the call returns a tuple, in parameter order: `low, high = k(t, zone)`.

### An output whose size depends on the data

Above, sizes are written as constants (`gs.f32[3]`). To make them follow the data, give them the name of an
**input** dimension: the size is read from that input at each call. When there is no natural input of that size,
add one that is only there for it:

```python
@gs.kernel(grid="t")
def coldest_n(t: gs.f32["n"], zone: gs.i32["n"], zones: gs.i32["m"], low: gs.Out[gs.f32["m"], gs.inf]):
    i = gs.index()
    gs.atomic_min(low, zone[i], t[i])

low = coldest_n(t, zone, np.zeros(3, np.int32))    # `zones` is not read: its size gives m = 3
assert np.array_equal(low, [3.5, -1.0, 20.0])
```

## Accumulating over several calls: `out=`

For data that arrives in batches, pass an existing array with `out=`: the output starts from its contents instead
of zero, and successive calls add up.

```python
acc = np.zeros(6, np.int32)
for _ in range(3):                                   # three batches of measurements
    batch = np.random.uniform(-25.0, 45.0, 10_000).astype(np.float32)
    histo(batch, out=acc).numpy()                    # .numpy(): wait for the end before the next batch
assert acc.sum() == 30_000
```

## A cell outside the output

If a thread targets a cell that does not exist (bin 10 of a 10-bin histogram), the write **is not done**: nothing
is written elsewhere in memory, and the other writes are done normally. Reading the result then raises
`gs.OutOfBounds`, naming the output and the faulty cell:

```python
@gs.kernel(grid="t")
def careless(t: gs.f32["n"], counts: gs.Out[gs.i32[6]]):
    i = gs.index()
    gs.atomic_add(counts, int((t[i] + 20.0) / 10.0), 1)     # 45 °C gives bin 6: out of bounds

try:
    careless(np.array([5.0, 45.0], np.float32)).numpy()
    raised = False
except gs.OutOfBounds as e:
    print(e)                                                 # … index 6 for counts of shape (6,) …
    raised = True
assert raised
```

Clamp the cell (`gs.min(bin_, 5)`) or skip the data point with an `if` followed by a `return`.

## Cost and precision

- **The more threads target the same cell, the slower it gets**, since they go one at a time: six bins contend
  much more than a thousand. Integer writes are native everywhere; in `gs.f32`, OpenCL 1.2 and WGSL (Metal,
  Vulkan, DX12) emulate them with a loop, slower still when threads crowd in.
- **In `gs.f32`, the last digit may vary from run to run**: the threads' arrival order changes, and (a + b) + c is
  not exactly a + (b + c) in floating point. For an identical result every time, accumulate in integers (for
  instance tenths of a millimetre instead of millimetres as floats).
- Not allowed yet: 64-bit outputs (`gs.i64`, `gs.f64`), and `grid=` together with `group=`.
