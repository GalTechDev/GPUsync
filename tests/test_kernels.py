"""Differential tests (milestone M1): each kernel, on every available device, compared with NumPy.

A kernel's Python function runs as is on NumPy arrays (gs.sqrt = np.sqrt…):
it is the reference when the kernel is vectorizable; otherwise a NumPy reference is written next to it.
"""

import math
import warnings

import numpy as np
import pytest

import gpusync as gs

DEVICES = [d.id for d in gs.devices()]
N = 4099  # not a multiple of the group size
RNG = np.random.default_rng(1234)


def f32(a):
    return np.asarray(a, dtype=np.float32)


def check(y, ref, rtol=2e-6, atol=1e-6):
    got = np.asarray(y)
    ref = np.asarray(ref)
    assert got.shape == ref.shape and got.dtype == ref.dtype, (got.shape, got.dtype, ref.shape, ref.dtype)
    if got.dtype.kind in "iu":
        bad = np.flatnonzero(got != ref)
    else:
        bad = np.flatnonzero(~np.isclose(got, ref, rtol=rtol, atol=atol, equal_nan=True))
    assert bad.size == 0, f"{bad.size} mismatch(es), e.g. index {bad[0]}: {got.flat[bad[0]]} instead of {ref.flat[bad[0]]}"


# ───────────────────────────── test kernels ─────────────────────────────


@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std


@gs.kernel
def horner(x: gs.f32[:], degree: gs.i32) -> gs.f32[:]:
    acc = 0  # becomes f32: assigned a float further down
    for _ in range(degree):
        acc = acc * x + 1.0
    return acc


@gs.kernel
def branches(x: gs.f32[:], y: gs.f32[:]) -> gs.f32[:]:
    if x > 0.5 and y > 0.5:
        return x + y
    elif x < -0.5 or not y < 0.0:
        return x - y
    else:
        big = 0.0 < x < 0.25
        return 100.0 if big else -100.0


@gs.kernel
def int_ops(a: gs.i32[:], b: gs.i32[:]) -> gs.i32[:]:
    return a // b + 1000 * (a % b) - abs(a) + min(a, b) * 3


@gs.kernel
def uint_wrap(a: gs.u32[:], b: gs.u32[:]) -> gs.u32[:]:
    return (a - b) * 3 + a // 7 + b % 5


@gs.kernel
def maths(x: gs.f32[:], y: gs.f32[:]) -> gs.f32[:]:
    s = gs.sqrt(abs(x)) + gs.exp(-x * x) + gs.log(1.0 + x * x) + gs.tanh(x)
    t = gs.sin(x) * gs.cos(y) + gs.atan2(x, y) + gs.pow(abs(x), 1.5)
    u = gs.floor(x) + gs.ceil(y) + gs.max(x, y, 0.25) - gs.min(x, y) + math.pi
    return s + t + u


@gs.kernel
def py_mod_div(x: gs.f32[:], y: gs.f32[:]) -> gs.f32[:]:
    return (x % 1.7) + (y % -0.9) * 10.0 + (x // 0.3) * 100.0


@gs.kernel
def negative_pow(x: gs.f32[:]) -> gs.f32[:]:
    return x**2 + x**3 * 0.5 + (-x) ** 2.0


@gs.kernel
def casts(x: gs.f32[:], k: gs.i32[:]) -> gs.i32[:]:
    t = int(x * 10.0)  # truncation toward zero, as in Python
    return t + gs.i32(float(k) * 0.5) + int(-2.7)


@gs.kernel
def collatz(n: gs.u32[:]) -> gs.u32[:]:
    v = n
    steps = gs.u32(0)
    while v != 1:
        if steps >= 500:
            break
        if v % 2 == 0:
            v = v // 2
            steps += 1
            continue
        v = 3 * v + 1
        steps += 1
    return steps


@gs.kernel
def precise(x: gs.f64[:]) -> gs.f64[:]:
    return x * 1e-12 + 1.0


@gs.kernel
def f64_ops(x: gs.f64[:], s: gs.f64) -> gs.f64[:]:
    """f64 operations available on Vulkan: arithmetic, //, %, sqrt, floor, abs, min, max, comparisons."""
    y = (x % 0.7) + (x // 0.3) * s + gs.sqrt(x) + gs.floor(x * 3.0) + abs(x - 0.5) + max(x, 0.2)
    return y if x > 0.5 else -y


@gs.kernel(compat="strict")
def precise_strict(x: gs.f64[:]) -> gs.f64[:]:
    return x * 1e-12 + 1.0


@gs.kernel
def f64_math(x: gs.f64[:]) -> gs.f64[:]:
    return gs.exp(x) + gs.log(1.0 + x * x)


@gs.kernel(compat="strict")
def f64_math_strict(x: gs.f64[:]) -> gs.f64[:]:
    return gs.exp(x) + gs.log(1.0 + x * x)


# ───────────────────────────── differential tests ─────────────────────────────


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def test_normalize(device):
    x = f32(RNG.random(N))
    check(normalize(x, 0.5, 0.25, device=device), (x - np.float32(0.5)) / np.float32(0.25))


def test_loop_and_retyped_local(device):
    x = f32(RNG.uniform(-1, 1, N))
    ref = np.zeros_like(x)
    for _ in range(7):
        ref = ref * x + np.float32(1.0)
    check(horner(x, 7, device=device), ref, rtol=1e-5)
    check(horner(x, 0, device=device), np.zeros_like(x))


def test_branches_and_or_not_chained_comparisons(device):
    x, y = f32(RNG.uniform(-1, 1, N)), f32(RNG.uniform(-1, 1, N))
    ref = np.where((x > 0.5) & (y > 0.5), x + y,
                   np.where((x < -0.5) | ~(y < 0.0), x - y,
                            np.where((x > 0) & (x < 0.25), np.float32(100), np.float32(-100))))
    check(branches(x, y, device=device), ref.astype(np.float32))


def test_integers_like_python(device):
    a = RNG.integers(-1000, 1000, N).astype(np.int32)
    b = RNG.integers(-50, 50, N).astype(np.int32)
    b[::17] = 0  # division by zero: 0, like NumPy
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = a // b + 1000 * (a % b) - np.abs(a) + np.minimum(a, b) * 3
    check(int_ops(a, b, device=device), ref.astype(np.int32))


def test_unsigned_integers_wraparound(device):
    a = RNG.integers(0, 2**31, N, dtype=np.uint32)
    b = RNG.integers(0, 2**32 - 1, N, dtype=np.uint32)
    ref = (a - b) * np.uint32(3) + a // np.uint32(7) + b % np.uint32(5)
    check(uint_wrap(a, b, device=device), ref)


def test_math_functions(device):
    x, y = f32(RNG.uniform(-3, 3, N)), f32(RNG.uniform(-3, 3, N))
    check(maths(x, y, device=device), maths.py_func(x, y).astype(np.float32), rtol=2e-5, atol=2e-5)


def test_float_modulo_and_floor_division(device):
    x, y = f32(RNG.uniform(-10, 10, N)), f32(RNG.uniform(-10, 10, N))
    ref = np.remainder(x, np.float32(1.7)) + np.remainder(y, np.float32(-0.9)) * 10 + np.floor(x / np.float32(0.3)) * 100
    check(py_mod_div(x, y, device=device), ref.astype(np.float32), rtol=1e-5, atol=1e-4)


def test_power_negative_base(device):
    x = f32(RNG.uniform(-4, 4, N))
    check(negative_pow(x, device=device), negative_pow.py_func(x), rtol=1e-5, atol=1e-5)


def test_conversions(device):
    x = f32(RNG.uniform(-50, 50, N))
    k = RNG.integers(-1000, 1000, N).astype(np.int32)
    ref = np.trunc(x * np.float32(10)).astype(np.int32) + np.trunc(k.astype(np.float32) * 0.5).astype(np.int32) - 2
    check(casts(x, k, device=device), ref)


def test_while_break_continue(device):
    n = np.arange(1, 2001, dtype=np.uint32)

    def steps(v):
        s = 0
        while v != 1 and s < 500:
            v = v // 2 if v % 2 == 0 else 3 * v + 1
            s += 1
        return s

    check(collatz(n, device=device), np.array([steps(int(v)) for v in n], dtype=np.uint32))


def test_f64(device):
    """f64 arithmetic: exact where available; otherwise an error (strict) or a warned fallback (auto)."""
    x = RNG.random(N)
    dev = next(d for d in gs.devices() if d.id == device)
    if dev.f64:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no fallback expected
            check(precise(x, device=device), x * 1e-12 + 1.0, rtol=1e-15, atol=0)
        return
    with pytest.raises(gs.UnsupportedFeature, match="f64.*Compatible devices"):
        precise_strict(x, device=device)
    with pytest.warns(gs.CompatibilityWarning, match=f"f64 is not available on {device}"):
        y = precise(x, device=device)
    assert y.device != device
    check(y, x * 1e-12 + 1.0, rtol=1e-15, atol=0)


def test_f64_ops_without_transcendental_functions(device):
    x = RNG.uniform(-3, 3, N)
    x = np.abs(x)  # sqrt
    y = (np.remainder(x, 0.7) + np.floor(x / 0.3) * 1.25 + np.sqrt(x) + np.floor(x * 3.0) + np.abs(x - 0.5)
         + np.maximum(x, 0.2))
    ref = np.where(x > 0.5, y, -y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gs.CompatibilityWarning)
        check(f64_ops(x, 1.25, device=device), ref, rtol=1e-14, atol=1e-14)


def test_f64_math_functions(device):
    """exp / log in f64: never on Vulkan (silent f32 precision or a driver crash, measured)."""
    x = RNG.random(N)
    ref = np.exp(x) + np.log(1.0 + x * x)
    dev = next(d for d in gs.devices() if d.id == device)
    if dev.f64_math:
        check(f64_math(x, device=device), ref, rtol=1e-14, atol=0)
        return
    with pytest.raises(gs.UnsupportedFeature, match="gs.exp|f64"):
        f64_math_strict(x, device=device)
    with pytest.warns(gs.CompatibilityWarning, match=device):
        y = f64_math(x, device=device)
    assert next(d for d in gs.devices() if d.id == y.device).f64_math
    check(y, ref, rtol=1e-14, atol=0)


def test_automatic_choice_by_requirements():
    """Without device=: first of the ranking if suitable, otherwise a warned fallback to a capable device."""
    from gpusync._device import capable, ranked

    first = ranked()[0]
    x32, x64 = f32(RNG.random(64)), RNG.random(64)
    assert normalize(x32, 0.0, 1.0).device == first.id
    for kern, req in ((precise, precise._req), (f64_math, f64_math._req)):
        if capable(first, req):
            assert kern(x64).device == first.id
        else:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                y = kern(x64)
            chosen = next(d for d in gs.devices() if d.id == y.device)
            assert capable(chosen, req)
            if chosen.backend != "cpu":
                assert chosen.backend == "opencl"


def test_arbitrary_shapes_and_empty_array(device):
    x = f32(RNG.random((37, 3, 5)))
    y = normalize(x, 0.0, 2.0, device=device)
    assert y.shape == (37, 3, 5)
    check(y, x / np.float32(2))
    empty = normalize(np.zeros((0, 4), np.float32), 0.0, 1.0, device=device)
    assert empty.shape == (0, 4) and np.asarray(empty).size == 0


def test_chaining_and_non_contiguous_view(device):
    x = f32(RNG.random((64, 64)))
    view = x[:, ::2]  # non-contiguous: copied to contiguous memory
    y = normalize(normalize(view, 1.0, 2.0, device=device), -0.5, 0.5, device=device)
    check(y, ((view - np.float32(1)) / np.float32(2) + np.float32(0.5)) / np.float32(0.5))


def test_chaining_across_devices():
    x = f32(RNG.random(N))
    y = normalize(x, 0.0, 2.0, device=DEVICES[0])
    z = normalize(y, 0.0, 0.5, device=DEVICES[-1])
    check(z, x)


def test_zero_copy_opencl_unified_memory():
    unified = [d for d in gs.devices() if d.backend == "opencl" and d.unified_memory]
    if not unified:
        pytest.skip("no OpenCL device with unified memory")
    x = f32(RNG.random(1 << 20))
    y = normalize(x, 0.0, 1.0, device=unified[0])
    assert y.zero_copy, "the output should be computed directly in the NumPy array"
    out = np.asarray(y)
    assert np.asarray(y) is out  # no copy on read
    check(out, x)


def test_unused_result_and_released_inputs(device):
    # The result is dropped and the input freed before the computation ends: nothing must crash.
    for _ in range(20):
        normalize(f32(RNG.random(1 << 16)), 0.0, 1.0, device=device)
    y = normalize(f32(np.ones(8)), 0.0, 1.0, device=device)
    check(y, np.ones(8, np.float32))


def test_device_choice():
    from gpusync._device import ranked

    assert gs.default_device().id == ranked()[0].id
    with gs.device("cpu") as d:
        assert d.id == "cpu:0"
        assert normalize(f32([2.0]), 0.0, 2.0).device == "cpu:0"
    assert normalize(f32([2.0]), 0.0, 2.0).device == ranked()[0].id
    with pytest.raises(gs.DeviceError, match="not found"):
        normalize(f32([1.0]), 0.0, 1.0, device="opencl:42")


def test_array_larger_than_a_wgpu_buffer():
    # On the UHD 770 with Vulkan, a bound buffer is limited to 1 GiB − 4 B. Before M1.5: wrong result without
    # error, then a clear error; since M1.5: processed chunk by chunk (Flux mode), without limit.
    small = [d for d in gs.devices() if d.backend in ("metal", "vulkan", "dx12")
             and d.max_buffer_bytes is not None and d.max_buffer_bytes < (1 << 30) + 4096]
    if not small:
        pytest.skip("no wgpu device limited to ~1 GiB per buffer")
    n = small[0].max_buffer_bytes // 4 + 1
    x = np.arange(n, dtype=np.float32)
    y = normalize(x, 1.0, 2.0, device=small[0])
    assert y.mode == "flux"
    out = np.asarray(y)
    for i in (0, 1, n // 2, n - 2, n - 1):  # ends and chunk boundaries
        assert out[i] == (np.float32(i) - 1) / 2


@gs.kernel
def loop_var(x: gs.f32["n", "t"], m: gs.i32) -> (gs.i32["n"], gs.i32["n"], gs.i32["n"]):
    """Loop variable as in Python: last value after the loop, unchanged if no iteration,
    and reassigning it in the body does not change the iteration."""
    i = gs.index()
    k = -1
    for k in range(x.shape[1]):
        pass
    j = -1
    for j in range(m, 0):  # no iteration if m >= 0
        pass
    count = 0
    for q in range(x.shape[1]):
        q = q + 100
        count += 1
    return k, j, count


@pytest.mark.parametrize("dev", DEVICES)
def test_loop_variable_like_python(dev):
    x = np.zeros((5, 7), np.float32)
    k, j, count = loop_var(x, 3, device=dev)
    check(k, np.full(5, 6, np.int32))
    check(j, np.full(5, -1, np.int32))
    check(count, np.full(5, 7, np.int32))
