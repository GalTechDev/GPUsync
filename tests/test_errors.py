"""Messages d'erreur (objectif O6) : ce qui ne va pas, où, et comment corriger. Jalon M1."""

import numpy as np
import pytest

import gpusync as gs


def compile_error(src: str, tmp_path, exc=gs.KernelSyntaxError):
    """Écrit le kernel dans un fichier (le source doit être lisible) et renvoie le message d'erreur."""
    path = tmp_path / "k.py"
    path.write_text("import math\nimport gpusync as gs\n\nLIMITE = 3\n\n@gs.kernel\n" + src)
    ns = {}
    with pytest.raises(exc) as info:
        code = compile(path.read_text(), str(path), "exec")
        import linecache

        linecache.cache[str(path)] = (len(path.read_text()), None, path.read_text().splitlines(True), str(path))
        exec(code, ns)
    return str(info.value)


CASES = [
    # (source du kernel, exception, fragments attendus dans le message)
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    l = [1, 2]\n    return x\n", gs.KernelSyntaxError, ["ligne 8", "listes"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x[0]\n", gs.KernelSyntaxError, ["ligne 8", "élément courant"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x + inconnu\n", gs.KernelSyntaxError, ["ligne 8", "inconnu"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    x = x + 1.0\n    return x\n", gs.KernelSyntaxError, ["ne peut pas être modifié"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    if x > 0.0:\n        return x\n", gs.KernelSyntaxError, ["toutes les branches"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    if x:\n        return x\n    return -x\n", gs.TypeMismatch, ["ligne 8", "x != 0"]),
    ("def k(x: gs.i32[:]) -> gs.i32[:]:\n    return x ** 2\n", gs.TypeMismatch, ["**", "flottants"]),
    ("def k(x: gs.i32[:]) -> gs.i32[:]:\n    y = x + 1\n    y = 2.5\n    return gs.i32(y)\n", None, []),
    ("def k(x: gs.i32[:], s: gs.u32) -> gs.i32[:]:\n    return x + s\n", gs.TypeMismatch, ["i32", "u32", "explicite"]),
    ("def k(x: gs.f32[:]) -> gs.i32[:]:\n    return x * 2.0\n", gs.TypeMismatch, ["valeur renvoyée", "f32"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    while x > 0.0:\n        break\n    else:\n        pass\n    return x\n", gs.KernelSyntaxError, ["while ... else"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    for i in [1, 2]:\n        pass\n    return x\n", gs.KernelSyntaxError, ["range"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    print(x)\n    return x\n", gs.KernelSyntaxError, ["expression sans effet"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return len(x)\n", gs.KernelSyntaxError, ["M2"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x, x\n", gs.KernelSyntaxError, ["M2"]),
    ("def k(x) -> gs.f32[:]:\n    return x\n", gs.KernelSyntaxError, ["'x' sans type", "gs.f32[:]"]),
    ("def k(x: gs.f32[:]):\n    return x\n", gs.KernelSyntaxError, ["type de sortie"]),
    ("def k(s: gs.f32) -> gs.f32[:]:\n    return s\n", gs.TypeMismatch, ["au moins un tableau"]),
    ("def k(x: gs.f32[:]) -> gs.f32[:]:\n    return x * LIMITE + math.pi\n", None, []),
]


@pytest.mark.parametrize("src,exc,fragments", CASES)
def test_erreurs_de_compilation(src, exc, fragments, tmp_path):
    if exc is None:  # cas valides (locale retypée en f32 puis convertie, constante globale)
        path = tmp_path / "ok.py"
        path.write_text("import math\nimport gpusync as gs\n\nLIMITE = 3\n\n@gs.kernel\n" + src)
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


def test_type_de_tableau():
    with pytest.raises(gs.TypeMismatch, match=r"'a' : gs.f32\[:\] attendu, reçu un tableau float64.*astype"):
        add(np.ones(3), np.ones(3, np.float32), device="cpu")


def test_cast_auto():
    y = add_auto(np.ones(3), np.arange(3), device="cpu")
    assert np.asarray(y).tolist() == [1.0, 2.0, 3.0]


def test_formes_incompatibles():
    with pytest.raises(gs.ShapeMismatch, match=r"'a' \(3,\), 'b' \(4,\)"):
        add(np.ones(3, np.float32), np.ones(4, np.float32), device="cpu")


def test_scalaires():
    a = np.arange(4, dtype=np.int32)
    assert np.asarray(scale(a, 3, device="cpu")).tolist() == [0, 3, 6, 9]
    with pytest.raises(gs.TypeMismatch, match="hors de la plage de u32"):
        scale(a, -1, device="cpu")
    with pytest.raises(gs.TypeMismatch, match="entier attendu"):
        scale(a, 1.5, device="cpu")
    with pytest.raises(gs.TypeMismatch, match="scalaire attendu"):
        scale(a, np.ones(3), device="cpu")
    with pytest.raises(TypeError, match="attend 2 argument"):
        scale(a)


def test_code_genere_visible():
    assert "__kernel void gs_add" in add.source("opencl")
    assert "@compute" in add.source("wgsl")
    assert "Kernel" in add.source("ir")
    assert repr(add) == "<gs.kernel add(a: gs.f32[:], b: gs.f32[:]) -> gs.f32[:]>"


def test_fonction_python_d_origine_utilisable():
    a = np.array([1.0, 2.0], np.float32)
    assert add.py_func(a, a).tolist() == [2.0, 4.0]
