"""Python function → JSON description for the Rust frontend (see crates/gpusync-frontend).

Python provides what only it knows: the source, the signature (evaluated annotations), the
global constants and the modules used. Typing and validation are done in Rust.
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
    """Error in a kernel's declaration (signature): converted into gs.KernelSyntaxError."""


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
    """Names visible from `func`: builtins, globals, then closure variables (like
    inspect.getclosurevars, without disassembling the code: 40 % less analysis time)."""
    g = func.__globals__
    b = g.get("__builtins__", builtins)
    scope = dict(b.__dict__ if isinstance(b, types.ModuleType) else b)
    scope.update(g)
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            scope[name] = cell.cell_contents
        except ValueError:  # closure variable not assigned yet
            pass
    return scope


def return_types(func, ret) -> tuple[list, bool]:
    """Declared outputs: ([ArrayType], tuple?) for `-> gs.f32[:]` or `-> (gs.f32[:], gs.i32[:])`."""
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
        f"'{func.__name__}': give the output type, e.g. -> gs.f32[:] (element-wise), "
        '-> gs.f32["n", "n"] (with gs.index()) or -> (gs.f32[:], gs.f32[:]) (several outputs)'
    )


def kernel_spec(func, grid=None, group=None, cells=None) -> tuple[dict, list]:
    """Returns (JSON spec for the frontend, list of parameters [(name, kind, DType, dims)])."""
    try:
        lines, start = inspect.getsourcelines(func)
    except (OSError, TypeError) as e:
        raise KernelDeclarationError(
            f"source of '{func.__name__}' not found ({e}): a kernel must be defined in a file or a notebook"
        ) from None
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    fdef = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not isinstance(fdef, ast.FunctionDef):
        raise KernelDeclarationError("@gs.kernel applies to a function (def), not async def")

    a = fdef.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
        raise KernelDeclarationError(
            f"'{func.__name__}': plain parameters only (no *args, **kwargs, default values or keyword-only parameters)"
        )
    annotations = inspect.get_annotations(func, eval_str=True)
    params = []
    fills = {}  # gs.Out: start value
    for arg in a.args:
        ann = annotations.get(arg.arg)
        if isinstance(ann, OutType):
            if grid is None:
                raise KernelDeclarationError(
                    f"parameter '{arg.arg}': gs.Out is used with an explicit grid, e.g. @gs.kernel(grid=\"x\")"
                )
            params.append((arg.arg, "out", ann.array.dtype, ann.array.dims))
            fills[arg.arg] = ann.fill
        elif isinstance(ann, ArrayType):
            params.append((arg.arg, "array", ann.dtype, ann.dims))
        elif isinstance(ann, DType):
            params.append((arg.arg, "scalar", ann, None))
        elif ann is None:
            raise KernelDeclarationError(
                f"parameter '{arg.arg}' has no type: annotate it, e.g. {arg.arg}: gs.f32[:] (array) or {arg.arg}: gs.f32 (scalar)"
            )
        else:
            raise KernelDeclarationError(f"parameter '{arg.arg}': type {ann!r} not supported (gs.f32[:], gs.f32, gs.i32…)")
    if grid is None:
        outs, _ = return_types(func, annotations.get("return"))
    else:
        outs = []
        if annotations.get("return") is not None:
            raise KernelDeclarationError(
                f"'{func.__name__}': with grid=, the results are the gs.Out parameters; remove the return type (-> ...)"
            )
        if not any(k == "out" for _, k, _, _ in params):
            raise KernelDeclarationError(
                f"'{func.__name__}': with grid=, declare at least one output, e.g. counts: gs.Out[gs.i32[10]]"
            )

    # Free names: numeric global constants and the gs / math modules.
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
        "filename": inspect.getsourcefile(func) or "<unknown>",
        "line_offset": start - 1,
        "params": [
            {"name": n, "kind": k, "dtype": d.name, **({"dims": list(dims)} if k != "scalar" else {})}
            for n, k, d, dims in params
        ],
        "outputs": [{"dtype": o.dtype.name, "dims": list(o.dims)} for o in outs],
        "grid": None if grid is None else (grid if isinstance(grid, str) else list(grid)),
        "group": group,
        "cells": cells,
        "body": _dump(fdef.body),
        "constants": constants,
        "modules": modules,
    }
    spec["_fills"] = fills
    return spec, params


def kernel_json(func, grid=None, group=None, cells=None) -> tuple[str, list, list, bool, dict]:
    """(JSON spec, parameters, outputs [ArrayType], is the output a tuple?, start value
    of each gs.Out).

    Explicit mode (`grid=`): the outputs are the gs.Out parameters (a tuple if there are several).
    """
    spec, params = kernel_spec(func, grid, group, cells)
    fills = spec.pop("_fills")
    if grid is not None:
        outs = [ArrayType(d, dims) for _, k, d, dims in params if k == "out"]
        return json.dumps(spec), params, outs, len(outs) > 1, fills
    outs, is_tuple = return_types(func, inspect.get_annotations(func, eval_str=True).get("return"))
    return json.dumps(spec), params, outs, is_tuple, fills
