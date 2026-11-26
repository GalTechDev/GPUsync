"""Error messages (goal O6): what is wrong, where, and how to fix it. Milestone M1."""

import numpy as np
import pytest

import gpusync as gs


def compile_error(src: str, tmp_path, exc=gs.KernelSyntaxError):
    """Writes the kernel to a file (the source must be readable) and returns the error message."""
    path = tmp_path / "k.py"
    path.write_text("import math\nimport gpusync as gs\n\nLIMIT = 3\n\n@gs.kernel\n" + src)
    ns = {}
    with pytest.raises(exc) as info:
        code = compile(path.read_text(), str(path), "exec")
        import linecache

        linecache.cache[str(path)] = (len(path.read_text()), None, path.read_text().splitlines(True), str(path))
        exec(code, ns)
    return str(info.value)


CASES = [
    # (kernel source, exception, fragments expected in the message)
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    l = [1, 2]\n    return x\n", gs.KernelSyntaxError, ["line 8", "lists"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x[0]\n", gs.KernelSyntaxError, ["line 8", "current element"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x + unknown\n", gs.KernelSyntaxError, ["line 8", "unknown"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    x = x + 1.0\n    return x\n", gs.KernelSyntaxError, ["cannot be modified"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    if x > 0.0:\n        return x\n", gs.KernelSyntaxError, ["every branch"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    if x:\n        return x\n    return -x\n", gs.TypeMismatch, ["line 8", "x != 0"]),
    ("def k(x: gs.i32[:]) -> gs.i32[:]:\n    return x ** 2\n", gs.TypeMismatch, ["**", "floats"]),
    ("def k(x: gs.i32[:]) -> gs.i32[:]:\n    y = x + 1\n    y = 2.5\n    return gs.i32(y)\n", None, []),
    ("def k(x: gs.i32[:], s: gs.u32) -> gs.i32[:]:\n    return x + s\n", gs.TypeMismatch, ["i32", "u32", "explicit"]),
    ("def k(x: gs.f32[:]) -> gs.i32[:]:\n    return x * 2.0\n", gs.TypeMismatch, ["returned value", "f32"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    while x > 0.0:\n        break\n    else:\n        pass\n    return x\n", gs.KernelSyntaxError, ["while ... else"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    for i in [1, 2]:\n        pass\n    return x\n", gs.KernelSyntaxError, ["range"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    print(x)\n    return x\n", gs.KernelSyntaxError, ["expression without effect"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return len(x)\n", gs.KernelSyntaxError, ["line 8", "gs.index()"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x, x\n", gs.TypeMismatch, ["line 8", "2 value(s)", "1 declared output"]),
    # Milestone M2: indexes, named dimensions, local arrays.
    ('def k(x: gs.f32[:]) -> gs.f32["n"]:\n    i = gs.index()\n    return x[i]\n', gs.TypeMismatch, ['"n"', "is not among the inputs"]),
    ('def k(x: gs.f32["n"]) -> gs.f32["m"]:\n    i = gs.index()\n    return x[i]\n', gs.TypeMismatch, ['"m"', "is not among the inputs"]),
    ('def k(x: gs.f32["n"]) -> gs.f32[:]:\n    i = gs.index()\n    return x[i]\n', gs.TypeMismatch, ["name the output dimensions"]),
    ('def k(x: gs.f32["n"]) -> gs.f32["n"]:\n    i = gs.index()\n    return x\n', gs.KernelSyntaxError, ["line 9"]),
    ('def k(x: gs.f32["n", "t"]) -> gs.f32["n"]:\n    i = gs.index()\n    return x[i]\n', gs.TypeMismatch, ["line 9", "2 dimension(s), 1 index(es)"]),
    ('def k(x: gs.f32["n"], s: gs.i32) -> gs.f32["n"]:\n    i = gs.index()\n    b = gs.local(gs.f32, s)\n    return x[i]\n', gs.TypeMismatch, ["line 9", "integer constant"]),
    ('def k(x: gs.f32["n"]) -> gs.f32["n"]:\n    i = gs.index()\n    b = gs.local(gs.f32, 4)\n    b[0] = x[i]\n    return b[0] + x[i]\n', None, []),
    ("def k(x) -> gs.f32[:]:\n    return x\n", gs.KernelSyntaxError, ["'x' has no type", "gs.f32[:]"]),
    ("def k(x: gs.f32[:]):\n    return x\n", gs.KernelSyntaxError, ["output type"]),
    ("def k(s: gs.f32) -> gs.f32[:]:\n    return s\n", gs.TypeMismatch, ["at least one array"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x * LIMIT + math.pi\n", None, []),
]


@pytest.mark.parametrize("src,exc,fragments", CASES)
def test_compile_errors(src, exc, fragments, tmp_path):
    if exc is None:  # valid cases (local retyped as f32 then converted, global constant)
        path = tmp_path / "ok.py"
        path.write_text("import math\nimport gpusync as gs\n\nLIMIT = 3\n\n@gs.kernel\n" + src)
        import importlib.util

        spec = importlib.util.spec_from_file_location("ok", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return
    msg = compile_error(src, tmp_path, exc)
    assert "kernel 'k'" in msg
    for f in fragments:
        assert f in msg, f"« {f} » absent de : {msg}"


@gs.kernel
def add(a: gs.f32[:], b: gs.f32[:]) -> gs.f32[:]:
    return a + b


@gs.kernel(cast="auto")
def add_auto(a: gs.f32[:], b: gs.f32[:]) -> gs.f32[:]:
    return a + b


@gs.kernel
def scale(a: gs.i32[:], s: gs.u32) -> gs.i32[:]:
    return a * gs.i32(s)


def test_array_type():
    with pytest.raises(gs.TypeMismatch, match=r"'a': gs.f32\[:\] expected, got a float64 array.*astype"):
        add(np.ones(3), np.ones(3, np.float32), device="cpu")


def test_cast_auto():
    y = add_auto(np.ones(3), np.arange(3), device="cpu")
    assert np.asarray(y).tolist() == [1.0, 2.0, 3.0]


def test_incompatible_shapes():
    with pytest.raises(gs.ShapeMismatch, match=r"'a' \(3,\), 'b' \(4,\)"):
        add(np.ones(3, np.float32), np.ones(4, np.float32), device="cpu")


def test_scalars():
    a = np.arange(4, dtype=np.int32)
    assert np.asarray(scale(a, 3, device="cpu")).tolist() == [0, 3, 6, 9]
    with pytest.raises(gs.TypeMismatch, match="out of the range of u32"):
        scale(a, -1, device="cpu")
    with pytest.raises(gs.TypeMismatch, match="integer expected"):
        scale(a, 1.5, device="cpu")
    with pytest.raises(gs.TypeMismatch, match="scalar expected"):
        scale(a, np.ones(3), device="cpu")
    with pytest.raises(TypeError, match="expects 2 argument"):
        scale(a)


def test_generated_code_visible():
    assert "__kernel void gs_add" in add.source("opencl")
    assert "@compute" in add.source("wgsl")
    assert "Kernel" in add.source("ir")
    assert repr(add) == "<gs.kernel add(a: gs.f32[:], b: gs.f32[:]) -> gs.f32[:]>"


def test_original_python_function_usable():
    a = np.array([1.0, 2.0], np.float32)
    assert add.py_func(a, a).tolist() == [2.0, 4.0]
