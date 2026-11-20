"""Fonction Python → description JSON pour le frontend Rust (voir crates/gpusync-frontend).

Le Python fournit ce que lui seul connaît : le source, la signature (annotations évaluées), les
constantes globales et les modules utilisés. Le typage et la validation sont faits en Rust.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import math
import textwrap
import types
import typing

from ._types import ArrayType, DType, OutType


class KernelDeclarationError(Exception):
    """Erreur dans la déclaration (signature) d'un kernel : convertie en gs.KernelSyntaxError."""


def _dump(node):
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            kind, rep = "bool", str(v)
        elif isinstance(v, int):
            kind, rep = "int", repr(v)
        elif isinstance(v, float):
            kind, rep = "float", repr(v)
        elif isinstance(v, str):
            kind, rep = "str", ""
        else:
            kind, rep = "other", ""
        return {"_type": "Constant", "value": {"kind": kind, "repr": rep}, "lineno": getattr(node, "lineno", 0)}
    if isinstance(node, ast.AST):
        d = {"_type": type(node).__name__}
        for field, value in ast.iter_fields(node):
            d[field] = _dump(value)
        if hasattr(node, "lineno"):
            d["lineno"] = node.lineno
        return d
    if isinstance(node, list):
        return [_dump(x) for x in node]
    return node


def _const_json(value):
    if isinstance(value, bool):
        return {"kind": "bool", "repr": str(value)}
    if isinstance(value, int):
        return {"kind": "int", "repr": repr(value)}
    if isinstance(value, float):
        return {"kind": "float", "repr": repr(value)}
    return None


def _scope(func) -> dict:
    """Noms visibles depuis `func` : intégrés, globaux, puis variables de fermeture (comme
    inspect.getclosurevars, sans désassembler le code : 40 % du temps d'analyse en moins)."""
    g = func.__globals__
    b = g.get("__builtins__", builtins)
    scope = dict(b.__dict__ if isinstance(b, types.ModuleType) else b)
    scope.update(g)
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            scope[name] = cell.cell_contents
        except ValueError:  # variable de fermeture pas encore affectée
            pass
    return scope


def return_types(func, ret) -> tuple[list, bool]:
    """Sorties déclarées : ([ArrayType], tuple ?) pour `-> gs.f32[:]` ou `-> (gs.f32[:], gs.i32[:])`."""
    if isinstance(ret, ArrayType):
        return [ret], False
    items = None
    if isinstance(ret, tuple):
        items = ret
    elif typing.get_origin(ret) is tuple:
        items = typing.get_args(ret)
    if items and all(isinstance(r, ArrayType) for r in items):
        return list(items), True
    raise KernelDeclarationError(
        f"'{func.__name__}' : précisez le type de sortie, ex. -> gs.f32[:] (élément par élément), "
        '-> gs.f32["n", "n"] (avec gs.index()) ou -> (gs.f32[:], gs.f32[:]) (plusieurs sorties)'
    )


def kernel_spec(func, grid=None) -> tuple[dict, list]:
    """Renvoie (spec JSON pour le frontend, liste des paramètres [(nom, kind, DType, dims)])."""
    try:
        lines, start = inspect.getsourcelines(func)
    except (OSError, TypeError) as e:
        raise KernelDeclarationError(
            f"source de '{func.__name__}' introuvable ({e}) : un kernel doit être défini dans un fichier ou un notebook"
        ) from None
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    fdef = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not isinstance(fdef, ast.FunctionDef):
        raise KernelDeclarationError("@gs.kernel s'applique à une fonction (def), pas async def")

    a = fdef.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
        raise KernelDeclarationError(
            f"'{func.__name__}' : paramètres simples uniquement (pas de *args, **kwargs, valeurs par défaut ni paramètres nommés)"
        )
    annotations = inspect.get_annotations(func, eval_str=True)
    params = []
    fills = {}  # gs.Out : valeur de départ
    for arg in a.args:
        ann = annotations.get(arg.arg)
        if isinstance(ann, OutType):
            if grid is None:
                raise KernelDeclarationError(
                    f"paramètre '{arg.arg}' : gs.Out s'utilise avec une grille explicite, ex. @gs.kernel(grid=\"x\")"
                )
            params.append((arg.arg, "out", ann.array.dtype, ann.array.dims))
            fills[arg.arg] = ann.fill
        elif isinstance(ann, ArrayType):
            params.append((arg.arg, "array", ann.dtype, ann.dims))
        elif isinstance(ann, DType):
            params.append((arg.arg, "scalar", ann, None))
        elif ann is None:
            raise KernelDeclarationError(
                f"paramètre '{arg.arg}' sans type : annotez-le, ex. {arg.arg}: gs.f32[:] (tableau) ou {arg.arg}: gs.f32 (scalaire)"
            )
        else:
            raise KernelDeclarationError(f"paramètre '{arg.arg}' : type {ann!r} non supporté (gs.f32[:], gs.f32, gs.i32…)")
    if grid is None:
        outs, _ = return_types(func, annotations.get("return"))
    else:
        outs = []
        if annotations.get("return") is not None:
            raise KernelDeclarationError(
                f"'{func.__name__}' : avec grid=, les résultats sont les paramètres gs.Out ; retirez le type de retour (-> ...)"
            )
        if not any(k == "out" for _, k, _, _ in params):
            raise KernelDeclarationError(
                f"'{func.__name__}' : avec grid=, déclarez au moins une sortie, ex. counts: gs.Out[gs.i32[10]]"
            )

    # Noms libres : constantes numériques globales et modules gs / math.
    import gpusync

    names = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    scope = _scope(func)
    constants, modules = {}, {}
    for name in names:
        if name not in scope:
            continue
        value = scope[name]
        if isinstance(value, types.ModuleType):
            if value is gpusync:
                modules[name] = "gpusync"
            elif value is math:
                modules[name] = "math"
        elif (c := _const_json(value)) is not None:
            constants[name] = c

    spec = {
        "name": func.__name__,
        "filename": inspect.getsourcefile(func) or "<inconnu>",
        "line_offset": start - 1,
        "params": [
            {"name": n, "kind": k, "dtype": d.name, **({"dims": list(dims)} if k != "scalar" else {})}
            for n, k, d, dims in params
        ],
        "outputs": [{"dtype": o.dtype.name, "dims": list(o.dims)} for o in outs],
        "grid": None if grid is None else (grid if isinstance(grid, str) else list(grid)),
        "body": _dump(fdef.body),
        "constants": constants,
        "modules": modules,
    }
    spec["_fills"] = fills
    return spec, params


def kernel_json(func, grid=None) -> tuple[str, list, list, bool, dict]:
    """(spec JSON, paramètres, sorties [ArrayType], la sortie est-elle un tuple ?, valeur de départ
    de chaque gs.Out).

    Mode explicite (`grid=`) : les sorties sont les paramètres gs.Out (tuple s'il y en a plusieurs).
    """
    spec, params = kernel_spec(func, grid)
    fills = spec.pop("_fills")
    if grid is not None:
        outs = [ArrayType(d, dims) for _, k, d, dims in params if k == "out"]
        return json.dumps(spec), params, outs, len(outs) > 1, fills
    outs, is_tuple = return_types(func, inspect.get_annotations(func, eval_str=True).get("return"))
    return json.dumps(spec), params, outs, is_tuple, fills
