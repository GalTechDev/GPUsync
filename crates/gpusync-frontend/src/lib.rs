//! Frontend: Python AST (serialized to JSON by the Python package) → typed IR (ARCHITECTURE.md §5).
//!
//! Input (JSON):
//! ```json
//! { "name": "normalize", "output": "f32", "line_offset": 41, "filename": "demo.py",
//!   "params": [{"name": "x", "kind": "array", "dtype": "f32"}, {"name": "mean", "kind": "scalar", "dtype": "f32"}],
//!   "body": [ <ast.stmt nodes> ],
//!   "constants": {"N": {"kind": "int", "repr": "3"}},
//!   "modules": {"gs": "gpusync", "math": "math"} }
//! ```
//! Each AST node is `{"_type": "BinOp", <ast fields>, "lineno": n}`; an `ast.Constant` carries
//! `"value": {"kind": "int|float|bool|str|none", "repr": "..."}`.
//!
//! Typing (§5.3): parameters are typed by the signature; a literal takes the type of its
//! context (`x * 2` stays in f32); a local takes the type of its assignments (`acc = 0` then
//! `acc += x` with a float x → float acc, as in Python). Never an implicit promotion to f64.

mod uniform;
use std::collections::HashMap;
use uniform::check_barriers;

use gpusync_ir::*;
use serde_json::Value as J;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Python construct not supported in a kernel.
    Syntax,
    /// Types incompatibles.
    Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrontendError {
    pub kind: ErrorKind,
    pub message: String,
    pub line: Option<u32>,
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(l) => write!(f, "line {l}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for FrontendError {}

type R<T> = Result<T, FrontendError>;

/// Entry point: JSON (see the module header) → typed kernel.
pub fn lower_json(spec: &str) -> R<Kernel> {
    let v: J = serde_json::from_str(spec)
        .map_err(|e| err(ErrorKind::Syntax, format!("invalid JSON: {e}"), None))?;
    lower(&v)
}

pub fn lower(spec: &J) -> R<Kernel> {
    let name = spec["name"].as_str().unwrap_or("kernel").to_string();
    // Outputs: list of {"dtype", "dims"} (or the old "output" field: one output).
    let out_specs: Vec<J> = match spec["outputs"].as_array() {
        Some(o) => o.clone(),
        None => vec![serde_json::json!({ "dtype": spec["output"].clone(), "dims": null })],
    };
    let mut outputs = Vec::new();
    for o in &out_specs {
        let t = parse_dtype(&o["dtype"], "return type")?;
        if !t.is_numeric() {
            return Err(err(
                ErrorKind::Type,
                "the return type must be numeric (f32, f64, i32, u32, i64, u64)".into(),
                None,
            ));
        }
        outputs.push(t);
    }
    let modules: HashMap<String, String> = spec["modules"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                .collect()
        })
        .unwrap_or_default();
    let body_json = spec["body"].as_array().cloned().unwrap_or_default();
    // Explicit mode (§4.5): `grid=` (name of an input or list of dimensions) and gs.Out outputs.
    let explicit = !spec["grid"].is_null();
    let indexed = explicit || uses_index(&J::Array(body_json.clone()), &modules);
    if explicit && !outputs.is_empty() {
        return Err(err(
            ErrorKind::Type,
            "with grid=, the results are the gs.Out parameters: remove the return type (-> ...)".into(),
            None,
        ));
    }
    let mut out_params: Vec<usize> = Vec::new();

    let mut params = Vec::new();
    let mut dim_names: Vec<String> = Vec::new(); // named dimensions, in order of appearance
    let mut param_dims: Vec<Vec<String>> = Vec::new();
    for p in spec["params"].as_array().cloned().unwrap_or_default() {
        let pname = p["name"].as_str().unwrap_or("?").to_string();
        let ty = parse_dtype(&p["dtype"], &format!("parameter '{pname}'"))?;
        if !ty.is_numeric() {
            return Err(err(
                ErrorKind::Type,
                format!(
                    "parameter '{pname}': type {ty} not supported (f32, f64, i32, u32, i64, u64)"
                ),
                None,
            ));
        }
        let kind = match p["kind"].as_str() {
            Some("array") => ParamKind::Array,
            Some("out") => {
                if !explicit {
                    return Err(err(
                        ErrorKind::Type,
                        format!("parameter '{pname}': gs.Out is used with @gs.kernel(grid=...)"),
                        None,
                    ));
                }
                if ty == Scalar::F64 || ty.is_int64() {
                    return Err(err(
                        ErrorKind::Type,
                        format!(
                            "parameter '{pname}': gs.Out output in {ty} not supported (atomic operations exist in i32, u32 and f32)"
                        ),
                        None,
                    ));
                }
                out_params.push(params.len());
                ParamKind::Array
            }
            _ => ParamKind::Scalar,
        };
        let mut dims = Vec::new();
        if kind == ParamKind::Array {
            match p["dims"].as_array() {
                Some(ds) => {
                    for (a, d) in ds.iter().enumerate() {
                        // Unnamed axis (`gs.f32[:, :]`): a name of the parameter's own; integer size
                        // (`gs.Out[gs.i32[10]]`): its decimal form, supplied by the caller.
                        dims.push(match (d.as_str(), d.as_u64()) {
                            (Some(n), _) => n.to_string(),
                            (None, Some(v)) => v.to_string(),
                            _ if p["kind"] == "out" => {
                                return Err(err(
                                    ErrorKind::Type,
                                    format!("output '{pname}': the size of each dimension is required, as an integer or named after an input (e.g. gs.Out[gs.i32[10]], gs.Out[gs.f32[\"n\"]])"),
                                    None,
                                ));
                            }
                            _ => format!("{pname}.{a}"),
                        });
                    }
                }
                None if indexed => {
                    return Err(err(
                        ErrorKind::Type,
                        format!(
                            "parameter '{pname}': in a kernel with gs.index(), give the dimensions, e.g. {pname}: gs.f32[\"n\", \"t\"]"
                        ),
                        None,
                    ));
                }
                None => {}
            }
        }
        for d in &dims {
            if !dim_names.contains(d) {
                dim_names.push(d.clone());
            }
        }
        param_dims.push(dims);
        params.push(Param {
            name: pname,
            kind,
            ty,
            dims: vec![],
            hidden: false,
        });
    }
    if !params.iter().any(|p| p.kind == ParamKind::Array) {
        return Err(err(
            ErrorKind::Type,
            "a kernel needs at least one array parameter (e.g. x: gs.f32[:])".into(),
            None,
        ));
    }
    // Grid dimensions in explicit mode: those of an input, or a list of names.
    let grid_dims: Option<Vec<String>> = if !explicit {
        None
    } else if let Some(g) = spec["grid"].as_str() {
        let Some(p) = params.iter().position(|p| {
            p.name == g
                && p.kind == ParamKind::Array
                && !out_params.contains(&params.iter().position(|q| q.name == g).unwrap())
        }) else {
            return Err(err(
                ErrorKind::Type,
                format!("grid=\"{g}\": '{g}' is not an input array of the kernel"),
                None,
            ));
        };
        Some(param_dims[p].clone())
    } else {
        let names: Vec<String> = spec["grid"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|d| match (d.as_str(), d.as_u64()) {
                (Some(n), _) => n.to_string(),
                (None, Some(v)) => v.to_string(),
                _ => String::new(),
            })
            .collect();
        for d in &names {
            if d.is_empty() || (!dim_names.contains(d) && d.parse::<u64>().is_err()) {
                return Err(err(
                    ErrorKind::Type,
                    format!(
                        "grid=(…): dimension \"{d}\" is not among the inputs (name it in an annotation, e.g. x: gs.f32[\"{d}\"])"
                    ),
                    None,
                ));
            }
        }
        Some(names)
    };
    if let Some(g) = &grid_dims {
        if g.is_empty() {
            return Err(err(
                ErrorKind::Type,
                "grid=: at least one dimension".into(),
                None,
            ));
        }
        for d in g.iter().chain(std::iter::once(&"2".to_string())) {
            if !dim_names.contains(d) {
                dim_names.push(d.clone()); // integer sizes (including that of gs_err)
            }
        }
    }
    let grid = if indexed {
        // Output dimensions: named, all identical, present in the inputs.
        let mut out_dims: Option<Vec<String>> = grid_dims.clone();
        for o in &out_specs {
            let ds: Vec<String> = match o["dims"].as_array() {
                Some(ds) if ds.iter().all(|d| d.is_string()) => {
                    ds.iter().map(|d| d.as_str().unwrap().to_string()).collect()
                }
                _ => {
                    return Err(err(
                        ErrorKind::Type,
                        "in a kernel with gs.index(), name the output dimensions, e.g. -> gs.f32[\"n\", \"n\"]".into(),
                        None,
                    ));
                }
            };
            for d in &ds {
                if !dim_names.contains(d) {
                    return Err(err(
                        ErrorKind::Type,
                        format!(
                            "output dimension \"{d}\" is not among the inputs: its size is unknown"
                        ),
                        None,
                    ));
                }
            }
            match &out_dims {
                Some(prev) if prev != &ds => {
                    return Err(err(
                        ErrorKind::Type,
                        "all outputs must have the same dimensions".into(),
                        None,
                    ));
                }
                _ => out_dims = Some(ds),
            }
        }
        // Hidden parameters: size of each dimension (i32), supplied by the runtime.
        let base = params.len();
        for d in &dim_names {
            params.push(Param {
                name: format!("gs_dim_{d}"),
                kind: ParamKind::Scalar,
                ty: Scalar::I32,
                dims: vec![],
                hidden: true,
            });
        }
        let idx = |d: &String| base + dim_names.iter().position(|x| x == d).unwrap();
        for (p, ds) in params.iter_mut().zip(&param_dims) {
            p.dims = ds.iter().map(idx).collect();
        }
        // Engine parameters for launches in blocks of rows (§9.7).
        let engine = params.len();
        for (name, ty) in [
            ("gs_row0", Scalar::I32),
            ("gs_inrow", Scalar::I32),
            ("gs_obase", Scalar::U32),
        ] {
            params.push(Param {
                name: name.into(),
                kind: ParamKind::Scalar,
                ty,
                dims: vec![],
                hidden: true,
            });
        }
        let out_dims = out_dims.unwrap_or_default();
        // Explicit mode: array of the rejected (out-of-bounds) writes, u32 × 2.
        let mut writable = out_params.clone();
        if explicit {
            writable.push(params.len());
            params.push(Param {
                name: "gs_err".into(),
                kind: ParamKind::Array,
                ty: Scalar::U32,
                dims: vec![idx(&"2".to_string())],
                hidden: true,
            });
            param_dims.push(vec!["2".into()]);
        }
        let readers = row_readers(&J::Array(body_json.clone()), &modules);
        let chunked = param_dims
            .iter()
            .enumerate()
            .filter(|(p, ds)| {
                params[*p].kind == ParamKind::Array
                    && !writable.contains(p)
                    && !ds.is_empty()
                    && ds.first() == out_dims.first()
                    && !readers.other.contains(&params[*p].name)
                    && readers.ok
            })
            .map(|(p, _)| p)
            .collect();
        Grid::Indexed {
            axes: out_dims.iter().map(idx).collect(),
            chunked,
            row0: engine,
            inrow: engine + 1,
            obase: engine + 2,
            writable,
        }
    } else {
        Grid::Elementwise
    };

    let mut constants = HashMap::new();
    if let Some(obj) = spec["constants"].as_object() {
        for (k, v) in obj {
            if let Some(l) = parse_lit(v) {
                constants.insert(k.clone(), l);
            }
        }
    }
    let line_offset = spec["line_offset"].as_u64().unwrap_or(0) as u32;
    // Group size (`@gs.kernel(group=n)`): kernels with gs.index(), outside explicit mode.
    let group: Option<u32> = match &spec["group"] {
        J::Null => None,
        g => {
            let n = g.as_u64().filter(|n| (1..=1024).contains(n)).ok_or_else(|| {
                err(ErrorKind::Type, format!("group={g}: number of threads per group between 1 and 1024 (e.g. group=256)"), None)
            })?;
            if !indexed {
                return Err(err(
                    ErrorKind::Syntax,
                    "group=: a group computes neighbouring cells of the output, the kernel must use gs.index()".into(),
                    None,
                ));
            }
            if explicit {
                return Err(err(
                    ErrorKind::Syntax,
                    "group= with grid= is not supported yet".into(),
                    None,
                ));
            }
            Some(n as u32)
        }
    };
    // Cells per thread (`cells=`): kernels with group=, a single output.
    let cells: u32 = match &spec["cells"] {
        J::Null => 1,
        c => {
            let n = c.as_u64().filter(|n| (1..=1024).contains(n)).ok_or_else(|| {
                err(
                    ErrorKind::Type,
                    format!("cells={c}: cells per thread between 1 and 1024 (e.g. cells=8)"),
                    None,
                )
            })?;
            if n > 1 && group.is_none() {
                return Err(err(
                    ErrorKind::Syntax,
                    "cells=: a thread's cells belong to its group's tile, declare group= (e.g. @gs.kernel(group=1024, cells=8))".into(),
                    None,
                ));
            }
            if n > 1 && outputs.len() != 1 {
                return Err(err(
                    ErrorKind::Type,
                    "cells=: a single output".into(),
                    None,
                ));
            }
            n as u32
        }
    };
    let (rank, chunked, inrow, writable) = match &grid {
        Grid::Indexed {
            axes,
            chunked,
            inrow,
            writable,
            ..
        } => (
            Some(axes.len()),
            chunked.clone(),
            Some(*inrow),
            writable.clone(),
        ),
        Grid::Elementwise => (None, vec![], None, vec![]),
    };

    let mut hints: HashMap<String, Scalar> = HashMap::new();
    for _pass in 0..16 {
        let (body, locals, restart, barrier_lines) = {
            let mut lw = Lowerer {
                params: &params,
                outputs: &outputs,
                rank,
                chunked: &chunked,
                inrow,
                writable: &writable,
                constants: &constants,
                modules: &modules,
                line_offset,
                locals: vec![],
                local_index: HashMap::new(),
                weak: vec![],
                hints: &mut hints,
                restart: false,
                loop_depth: 0,
                line: None,
                group,
                barrier_lines: vec![],
                cells,
            };
            let body = lw.block(&body_json, true)?;
            (
                body,
                std::mem::take(&mut lw.locals),
                lw.restart,
                std::mem::take(&mut lw.barrier_lines),
            )
        };
        if restart {
            continue;
        }
        if !outputs.is_empty() && cells == 1 && !always_returns(&body) {
            return Err(err(
                ErrorKind::Syntax,
                "the kernel must end with `return` in every branch".into(),
                None,
            ));
        }
        let k = Kernel {
            name,
            params,
            outputs,
            locals,
            body,
            grid,
            group,
            cells,
        };
        if group.is_some() {
            check_barriers(&k, &barrier_lines)?;
        }
        return Ok(k);
    }
    Err(err(
        ErrorKind::Type,
        "cannot determine the type of the local variables".into(),
        None,
    ))
}

/// Reading arrays by row (for the split into blocks of rows, §9.7).
struct RowReaders {
    /// A single assignment `i, … = gs.index()`, and `i` is never reassigned.
    ok: bool,
    /// Arrays indexed at least once with a first index other than `i`.
    other: Vec<String>,
}

/// Which arrays are only read at the current row (`x[i, …]`, `i` first axis of `gs.index()`)?
fn row_readers(body: &J, modules: &HashMap<String, String>) -> RowReaders {
    fn is_index_call(n: &J, modules: &HashMap<String, String>) -> bool {
        n["_type"] == "Call"
            && n["func"]["_type"] == "Attribute"
            && n["func"]["attr"] == "index"
            && n["func"]["value"]["_type"] == "Name"
            && modules
                .get(n["func"]["value"]["id"].as_str().unwrap_or(""))
                .map(String::as_str)
                == Some("gpusync")
    }
    fn names(t: &J, out: &mut Vec<String>) {
        match t["_type"].as_str() {
            Some("Name") => out.push(t["id"].as_str().unwrap_or("").to_string()),
            Some("Tuple") | Some("List") => {
                for e in t["elts"].as_array().into_iter().flatten() {
                    names(e, out);
                }
            }
            _ => {}
        }
    }
    fn walk(n: &J, f: &mut dyn FnMut(&J)) {
        match n {
            J::Object(o) => {
                f(n);
                for v in o.values() {
                    walk(v, f);
                }
            }
            J::Array(a) => a.iter().for_each(|v| walk(v, f)),
            _ => {}
        }
    }
    let mut row_vars = Vec::new();
    let mut assigned: Vec<String> = Vec::new();
    walk(body, &mut |n| match n["_type"].as_str() {
        Some("Assign") => {
            for t in n["targets"].as_array().into_iter().flatten() {
                if is_index_call(&n["value"], modules) {
                    let first = if t["_type"] == "Tuple" {
                        t["elts"][0].clone()
                    } else {
                        t.clone()
                    };
                    names(&first, &mut row_vars);
                }
                names(t, &mut assigned);
            }
        }
        Some("AugAssign") | Some("AnnAssign") | Some("NamedExpr") => {
            names(&n["target"], &mut assigned)
        }
        Some("For") => names(&n["target"], &mut assigned),
        _ => {}
    });
    let ok = row_vars.len() == 1 && assigned.iter().filter(|a| **a == row_vars[0]).count() == 1;
    let mut other = Vec::new();
    walk(body, &mut |n| {
        if n["_type"] == "Subscript" && n["value"]["_type"] == "Name" {
            let first = if n["slice"]["_type"] == "Tuple" {
                &n["slice"]["elts"][0]
            } else {
                &n["slice"]
            };
            let is_row = first["_type"] == "Name"
                && row_vars.first().is_some_and(|r| first["id"] == r.as_str());
            if !is_row {
                other.push(n["value"]["id"].as_str().unwrap_or("").to_string());
            }
        }
    });
    RowReaders { ok, other }
}

/// Does the body call `gs.index()`?
fn uses_index(node: &J, modules: &HashMap<String, String>) -> bool {
    match node {
        J::Object(o) => {
            if o.get("_type").and_then(|t| t.as_str()) == Some("Call") {
                let f = &node["func"];
                if f["_type"] == "Attribute"
                    && f["attr"] == "index"
                    && f["value"]["_type"] == "Name"
                    && modules
                        .get(f["value"]["id"].as_str().unwrap_or(""))
                        .map(String::as_str)
                        == Some("gpusync")
                {
                    return true;
                }
            }
            o.values().any(|v| uses_index(v, modules))
        }
        J::Array(a) => a.iter().any(|v| uses_index(v, modules)),
        _ => false,
    }
}

fn err(kind: ErrorKind, message: String, line: Option<u32>) -> FrontendError {
    FrontendError {
        kind,
        message,
        line,
    }
}

fn parse_dtype(v: &J, what: &str) -> R<Scalar> {
    v.as_str()
        .and_then(Scalar::parse)
        .ok_or_else(|| err(ErrorKind::Type, format!("{what}: unknown type {v}"), None))
}

/// Lossless implicit conversion between integers (like Python's unbounded integers):
/// i32 → i64, u32 → i64, u32 → u64.
fn widens(from: Scalar, to: Scalar) -> bool {
    matches!(
        (from, to),
        (Scalar::I32, Scalar::I64) | (Scalar::U32, Scalar::I64) | (Scalar::U32, Scalar::U64)
    )
}

/// Literal not typed yet ("weak"): takes the type of its context.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lit {
    Int(i64),
    Float(f64),
    Bool(bool),
}

fn parse_lit(v: &J) -> Option<Lit> {
    let repr = v["repr"].as_str()?;
    match v["kind"].as_str()? {
        "int" => repr.parse().ok().map(Lit::Int),
        "float" => match repr {
            "inf" => Some(Lit::Float(f64::INFINITY)),
            "-inf" => Some(Lit::Float(f64::NEG_INFINITY)),
            "nan" => Some(Lit::Float(f64::NAN)),
            r => r.parse().ok().map(Lit::Float),
        },
        "bool" => Some(Lit::Bool(repr == "True")),
        _ => None,
    }
}

/// Value during lowering: weak literal or typed expression.
#[derive(Debug, Clone)]
enum Val {
    Lit(Lit),
    E(Expr),
}

impl Val {
    fn ty(&self) -> Scalar {
        match self {
            Val::Lit(Lit::Int(_)) => Scalar::I32,
            Val::Lit(Lit::Float(_)) => Scalar::F32,
            Val::Lit(Lit::Bool(_)) => Scalar::Bool,
            Val::E(e) => e.ty,
        }
    }
    fn is_lit(&self) -> bool {
        matches!(self, Val::Lit(_))
    }
}

fn ex(ty: Scalar, kind: ExprKind) -> Expr {
    Expr { ty, kind }
}

fn cast(e: Expr, to: Scalar) -> Expr {
    if e.ty == to {
        e
    } else {
        ex(to, ExprKind::Cast(Box::new(e)))
    }
}

struct Lowerer<'a> {
    params: &'a [Param],
    outputs: &'a [Scalar],
    /// Rank of the grid for an indexed kernel; `None`: element-wise.
    rank: Option<usize>,
    /// Arrays read only at the current row: indexed starting from row `inrow`.
    chunked: &'a [usize],
    inrow: Option<usize>,
    /// Explicit mode: gs.Out arrays, then gs_err last.
    writable: &'a [usize],
    constants: &'a HashMap<String, Lit>,
    modules: &'a HashMap<String, String>,
    line_offset: u32,
    locals: Vec<Local>,
    local_index: HashMap<String, usize>,
    /// Was the local created from a literal (type still "soft")?
    weak: Vec<bool>,
    hints: &'a mut HashMap<String, Scalar>,
    restart: bool,
    loop_depth: u32,
    line: Option<u32>,
    /// Group size (`group=`) and line of each `gs.barrier()`, in code order.
    group: Option<u32>,
    barrier_lines: Vec<Option<u32>>,
    cells: u32,
}

impl Lowerer<'_> {
    fn e(&self, kind: ErrorKind, msg: impl Into<String>) -> FrontendError {
        err(kind, msg.into(), self.line)
    }

    fn set_line(&mut self, node: &J) {
        if let Some(l) = node["lineno"].as_u64() {
            self.line = Some(l as u32 + self.line_offset);
        }
    }

    // ───────────────────────── conversions ─────────────────────────

    /// Literal → constant of the requested type (checks the range).
    fn lit_to(&self, l: Lit, ty: Scalar) -> R<Expr> {
        let c = match (l, ty) {
            (Lit::Bool(b), Scalar::Bool) => Const::Bool(b),
            (Lit::Int(v), Scalar::I32) => Const::I32(
                i32::try_from(v)
                    .map_err(|_| self.e(ErrorKind::Type, format!("{v} exceeds the range of i32")))?,
            ),
            (Lit::Int(v), Scalar::U32) => Const::U32(u32::try_from(v).map_err(|_| {
                self.e(
                    ErrorKind::Type,
                    format!("{v} out of the range of u32 (positive integers < 2³²)"),
                )
            })?),
            (Lit::Int(v), Scalar::I64) => Const::I64(v),
            (Lit::Int(v), Scalar::U64) => Const::U64(u64::try_from(v).map_err(|_| {
                self.e(
                    ErrorKind::Type,
                    format!("{v} out of the range of u64 (positive integers)"),
                )
            })?),
            (Lit::Int(v), Scalar::F32) => Const::F32(v as f32),
            (Lit::Int(v), Scalar::F64) => Const::F64(v as f64),
            (Lit::Float(v), Scalar::F32) => Const::F32(v as f32),
            (Lit::Float(v), Scalar::F64) => Const::F64(v),
            (Lit::Float(v), t) if t.is_int() => {
                return Err(self.e(
                    ErrorKind::Type,
                    format!("float literal {v} in an integer context ({t}): use int(...) or an integer"),
                ));
            }
            (l, t) => {
                return Err(self.e(
                    ErrorKind::Type,
                    format!("{l:?} incompatible with type {t}"),
                ));
            }
        };
        Ok(ex(ty, ExprKind::Const(c)))
    }

    /// Value → expression of the exact type `ty`, with Python's implicit conversions
    /// (integer → float, f32 → f64, 32-bit integer → 64-bit integer without loss). Other
    /// conversions must be explicit.
    fn to_ty(&self, v: Val, ty: Scalar, what: &str) -> R<Expr> {
        match v {
            Val::Lit(l) => self.lit_to(l, ty),
            Val::E(e) if e.ty == ty => Ok(e),
            Val::E(e) if widens(e.ty, ty) => Ok(cast(e, ty)),
            Val::E(e) if e.ty.is_int() && ty.is_float() => Ok(cast(e, ty)),
            Val::E(e) if e.ty == Scalar::F32 && ty == Scalar::F64 => Ok(cast(e, ty)),
            Val::E(e) => Err(self.e(
                ErrorKind::Type,
                format!("{what}: {} instead of {ty} — explicit conversion needed (e.g. gs.{ty}(...))", e.ty),
            )),
        }
    }

    fn materialize(&self, v: Val) -> R<Expr> {
        let ty = v.ty();
        self.to_ty(v, ty, "value")
    }

    /// Common type of two numeric operands (rules of §5.3).
    fn unify(&self, a: &Val, b: &Val, op: &str) -> R<Scalar> {
        let (ta, tb) = (a.ty(), b.ty());
        if ta == Scalar::Bool || tb == Scalar::Bool {
            return Err(self.e(
                ErrorKind::Type,
                format!("operation \"{op}\" on a boolean"),
            ));
        }
        Ok(match (a.is_lit(), b.is_lit()) {
            (true, true) => {
                if ta.is_float() || tb.is_float() {
                    Scalar::F32
                } else {
                    Scalar::I32
                }
            }
            (true, false) => {
                if tb.is_int() && matches!(a, Val::Lit(Lit::Float(_))) {
                    Scalar::F32
                } else {
                    tb
                }
            }
            (false, true) => {
                if ta.is_int() && matches!(b, Val::Lit(Lit::Float(_))) {
                    Scalar::F32
                } else {
                    ta
                }
            }
            (false, false) => {
                if ta == tb {
                    ta
                } else if ta.is_float() && tb.is_float() {
                    Scalar::F64
                } else if ta.is_float() {
                    ta
                } else if tb.is_float() || widens(ta, tb) {
                    tb
                } else if widens(tb, ta) {
                    ta
                } else {
                    return Err(self.e(
                        ErrorKind::Type,
                        format!("\"{op}\" between {ta} and {tb}: explicit conversion needed (e.g. gs.{ta}(...) or gs.{tb}(...))"),
                    ));
                }
            }
        })
    }

    // ───────────────────────── instructions ─────────────────────────

    fn block(&mut self, stmts: &[J], top: bool) -> R<Block> {
        let mut out = Vec::new();
        for (i, s) in stmts.iter().enumerate() {
            self.set_line(s);
            let t = s["_type"].as_str().unwrap_or("");
            // Docstring: first statement, a lone string.
            if top && i == 0 && t == "Expr" && s["value"]["value"]["kind"] == "str" {
                continue;
            }
            match t {
                "Assign" => {
                    let targets = s["targets"].as_array().cloned().unwrap_or_default();
                    if targets.len() != 1 {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "multiple assignment a = b = ... not supported",
                        ));
                    }
                    let target = &targets[0];
                    let value = &s["value"];
                    if self.is_gs_call(value, "index") {
                        out.extend(self.assign_index(target)?);
                    } else if self.is_gs_call(value, "local") {
                        self.declare_local_array(target, value, false)?;
                    } else if self.is_gs_call(value, "shared") {
                        self.declare_local_array(target, value, true)?;
                    } else if target["_type"] == "Subscript" {
                        let v = self.expr(value)?;
                        out.push(self.store(target, v)?);
                    } else if target["_type"] == "Name" {
                        let v = self.expr(value)?;
                        out.push(self.assign(target["id"].as_str().unwrap_or(""), v, None)?);
                    } else {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "assignment not supported: a variable (a = ...) or a cell of a local array (buf[k] = ...) on the left; a, b = ... only with gs.index()",
                        ));
                    }
                }
                "AnnAssign" => {
                    if s["target"]["_type"] != "Name" || s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "annotation without a value or on something other than a variable",
                        ));
                    }
                    let ty = self.annotation_type(&s["annotation"])?;
                    let v = self.expr(&s["value"])?;
                    out.push(self.assign(s["target"]["id"].as_str().unwrap_or(""), v, Some(ty))?);
                }
                "AugAssign" if s["target"]["_type"] == "Subscript" => {
                    let current = self.expr(&s["target"])?;
                    let rhs = self.expr(&s["value"])?;
                    let v = self.binop(op_name(&s["op"]), current, rhs)?;
                    out.push(self.store(&s["target"], v)?);
                }
                "AugAssign" => {
                    if s["target"]["_type"] != "Name" {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "x op= ...: only plain variables are supported",
                        ));
                    }
                    let name = s["target"]["id"].as_str().unwrap_or("").to_string();
                    let current = self.name(&name)?;
                    let rhs = self.expr(&s["value"])?;
                    let v = self.binop(op_name(&s["op"]), current, rhs)?;
                    out.push(self.assign(&name, v, None)?);
                }
                "If" => {
                    let cond = self.condition(&s["test"])?;
                    let then = self.block(
                        s["body"].as_array().map(|v| v.as_slice()).unwrap_or(&[]),
                        false,
                    )?;
                    let otherwise = self.block(
                        s["orelse"].as_array().map(|v| v.as_slice()).unwrap_or(&[]),
                        false,
                    )?;
                    out.push(Stmt::If {
                        cond,
                        then,
                        otherwise,
                    });
                }
                "For" => out.push(self.for_loop(s)?),
                "While" => {
                    if s["orelse"].as_array().is_some_and(|a| !a.is_empty()) {
                        return Err(self.e(ErrorKind::Syntax, "while ... else is not supported"));
                    }
                    let cond = self.condition(&s["test"])?;
                    self.loop_depth += 1;
                    let body = self.block(
                        s["body"].as_array().map(|v| v.as_slice()).unwrap_or(&[]),
                        false,
                    );
                    self.loop_depth -= 1;
                    out.push(Stmt::While { cond, body: body? });
                }
                "Break" | "Continue" => {
                    if self.loop_depth == 0 {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            format!("{} outside a loop", t.to_lowercase()),
                        ));
                    }
                    out.push(if t == "Break" {
                        Stmt::Break
                    } else {
                        Stmt::Continue
                    });
                }
                "Return" if self.cells > 1 => {
                    if !s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "with cells=, each cell is written with gs.store(k, value): return without a value",
                        ));
                    }
                    out.push(Stmt::Return(vec![]));
                }
                "Return" if self.outputs.is_empty() => {
                    if !s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "with grid=, the results are written to the gs.Out (gs.atomic_add…): return without a value",
                        ));
                    }
                    out.push(Stmt::Return(vec![]));
                }
                "Return" => {
                    if s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "return without a value: the kernel must return the output element",
                        ));
                    }
                    let values: Vec<J> = if s["value"]["_type"] == "Tuple" {
                        s["value"]["elts"].as_array().cloned().unwrap_or_default()
                    } else {
                        vec![s["value"].clone()]
                    };
                    if values.len() != self.outputs.len() {
                        return Err(self.e(
                            ErrorKind::Type,
                            format!(
                                "{} value(s) returned for {} declared output(s) (-> ({})…)",
                                values.len(),
                                self.outputs.len(),
                                self.outputs
                                    .iter()
                                    .map(|t| format!("gs.{t}[...]"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        ));
                    }
                    let mut es = Vec::new();
                    for (v, t) in values.iter().zip(self.outputs.to_vec()) {
                        let v = self.expr(v)?;
                        es.push(self.to_ty(v, t, "returned value")?);
                    }
                    out.push(Stmt::Return(es));
                }
                "Pass" => {}
                "Expr"
                    if ["atomic_add", "atomic_min", "atomic_max"]
                        .iter()
                        .any(|f| self.is_gs_call(&s["value"], f)) =>
                {
                    out.push(self.atomic(&s["value"])?);
                }
                "Expr" if self.is_gs_call(&s["value"], "matmul_u8") => {
                    let st = self.matmul(&s["value"])?;
                    self.barrier_lines.push(self.line);
                    out.push(st);
                }
                "Expr" if self.is_gs_call(&s["value"], "store") => {
                    if self.cells <= 1 {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "gs.store(k, value): several cells per thread, declare them: @gs.kernel(group=1024, cells=8)",
                        ));
                    }
                    let args = s["value"]["args"].as_array().cloned().unwrap_or_default();
                    if args.len() != 2 {
                        return Err(
                            self.e(ErrorKind::Syntax, "gs.store(k, value): two arguments")
                        );
                    }
                    let k = self.index_expr(&args[0])?;
                    let v = self.expr(&args[1])?;
                    let value = self.to_ty(v, self.outputs[0], "gs.store")?;
                    out.push(Stmt::Store { k, value });
                }
                "Expr" if self.is_gs_call(&s["value"], "barrier") => {
                    if self.group.is_none() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "gs.barrier() synchronizes the threads of a group: declare the group size, @gs.kernel(group=256)",
                        ));
                    }
                    if s["value"]["args"].as_array().is_some_and(|a| !a.is_empty()) {
                        return Err(
                            self.e(ErrorKind::Syntax, "gs.barrier() takes no argument")
                        );
                    }
                    self.barrier_lines.push(self.line);
                    out.push(Stmt::Barrier);
                }
                "Expr" => {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        "expression without effect (a lone function call is not supported)",
                    ));
                }
                other => {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        format!(
                            "Python statement not supported in a kernel: {}",
                            stmt_label(other)
                        ),
                    ));
                }
            }
        }
        Ok(out)
    }

    fn annotation_type(&self, a: &J) -> R<Scalar> {
        let name = match a["_type"].as_str() {
            Some("Name") => a["id"].as_str().unwrap_or(""),
            Some("Attribute") if self.module_of(&a["value"]) == Some("gpusync") => {
                a["attr"].as_str().unwrap_or("")
            }
            _ => "",
        };
        match name {
            "float" => Ok(Scalar::F32),
            "int" => Ok(Scalar::I32),
            "bool" => Ok(Scalar::Bool),
            n => Scalar::parse(n).ok_or_else(|| {
                self.e(
                    ErrorKind::Type,
                    "variable annotation: use gs.f32, gs.i32, gs.u32, gs.f64 or bool",
                )
            }),
        }
    }

    /// Is `node` a call `gs.<fname>(...)`?
    fn is_gs_call(&self, node: &J, fname: &str) -> bool {
        node["_type"] == "Call"
            && node["func"]["_type"] == "Attribute"
            && node["func"]["attr"] == fname
            && self.module_of(&node["func"]["value"]) == Some("gpusync")
    }

    /// `i, j = gs.index()`: one i32 local per grid axis.
    fn assign_index(&mut self, target: &J) -> R<Vec<Stmt>> {
        let Some(rank) = self.rank else {
            unreachable!("gs.index() implies an indexed grid")
        };
        let names: Vec<String> = match target["_type"].as_str() {
            Some("Name") => vec![target["id"].as_str().unwrap_or("").to_string()],
            Some("Tuple") => target["elts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    if e["_type"] == "Name" {
                        Ok(e["id"].as_str().unwrap_or("").to_string())
                    } else {
                        Err(())
                    }
                })
                .collect::<Result<_, _>>()
                .map_err(|_| {
                    self.e(
                        ErrorKind::Syntax,
                        "gs.index() is assigned to variables: i, j = gs.index()",
                    )
                })?,
            _ => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.index() is assigned to variables: i, j = gs.index()",
                ));
            }
        };
        if names.len() != rank {
            return Err(self.e(
                ErrorKind::Type,
                format!("gs.index() returns {rank} coordinate(s) (one per output dimension), {} variable(s) on the left", names.len()),
            ));
        }
        let mut out = Vec::new();
        for (a, n) in names.iter().enumerate() {
            out.push(self.assign(
                n,
                Val::E(ex(Scalar::I32, ExprKind::GridIndex(a))),
                Some(Scalar::I32),
            )?);
        }
        Ok(out)
    }

    /// `gs.atomic_add(output, index, value)` (or `atomic_min`, `atomic_max`) on a gs.Out;
    /// integer index, or tuple of indices for a multi-dimensional output.
    fn atomic(&mut self, call: &J) -> R<Stmt> {
        let fname = call["func"]["attr"].as_str().unwrap_or("").to_string();
        let op = match fname.as_str() {
            "atomic_add" => AtomicOp::Add,
            "atomic_min" => AtomicOp::Min,
            _ => AtomicOp::Max,
        };
        let args = call["args"].as_array().cloned().unwrap_or_default();
        let Some(&err_param) = self.writable.last() else {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("gs.{fname}: reserved for kernels with grid= and gs.Out outputs (@gs.kernel(grid=\"x\"))"),
            ));
        };
        if args.len() != 3 || !call["keywords"].as_array().is_none_or(|k| k.is_empty()) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("gs.{fname}(output, index, value): e.g. gs.{fname}(counts, k, 1)"),
            ));
        }
        let target = args[0]["id"].as_str().unwrap_or("");
        let Some(p) = self
            .params
            .iter()
            .position(|q| q.name == target && !q.hidden)
            .filter(|p| self.writable.contains(p))
        else {
            return Err(self.e(
                ErrorKind::Type,
                format!("gs.{fname}: the first argument must be a gs.Out output of the kernel"),
            ));
        };
        let dims = self.params[p].dims.clone();
        let idx_nodes: Vec<J> = if args[1]["_type"] == "Tuple" {
            args[1]["elts"].as_array().cloned().unwrap_or_default()
        } else {
            vec![args[1].clone()]
        };
        if idx_nodes.len() != dims.len() {
            return Err(self.e(
                ErrorKind::Type,
                format!(
                    "'{target}' has {} dimension(s), {} index(es) given (several: a tuple, e.g. ({}))",
                    dims.len(),
                    idx_nodes.len(),
                    (0..dims.len()).map(|k| format!("i{k}")).collect::<Vec<_>>().join(", ")
                ),
            ));
        }
        let mut flat: Option<Expr> = None;
        let mut ok: Option<Expr> = None;
        for (k, n) in idx_nodes.iter().enumerate() {
            let i = self.index_expr(n)?;
            let d = ex(Scalar::I32, ExprKind::Param(dims[k]));
            let zero = ex(Scalar::I32, ExprKind::Const(Const::I32(0)));
            let ge = ex(
                Scalar::Bool,
                ExprKind::Compare(CmpOp::Ge, Box::new(i.clone()), Box::new(zero)),
            );
            let lt = ex(
                Scalar::Bool,
                ExprKind::Compare(CmpOp::Lt, Box::new(i.clone()), Box::new(d.clone())),
            );
            let both = ex(
                Scalar::Bool,
                ExprKind::Binary(BinOp::And, Box::new(ge), Box::new(lt)),
            );
            ok = Some(match ok {
                None => both,
                Some(a) => ex(
                    Scalar::Bool,
                    ExprKind::Binary(BinOp::And, Box::new(a), Box::new(both)),
                ),
            });
            flat = Some(match flat {
                None => i,
                Some(acc) => {
                    let mul = ex(
                        Scalar::I32,
                        ExprKind::Binary(BinOp::Mul, Box::new(acc), Box::new(d)),
                    );
                    ex(
                        Scalar::I32,
                        ExprKind::Binary(BinOp::Add, Box::new(mul), Box::new(i)),
                    )
                }
            });
        }
        let ty = self.params[p].ty;
        let v = self.expr(&args[2])?;
        let value = self.to_ty(
            v,
            ty,
            &format!("value of gs.{fname} on '{target}' (gs.{ty})"),
        )?;
        let code = self.writable.iter().position(|w| *w == p).unwrap() as u32 + 1;
        Ok(Stmt::Atomic {
            op,
            param: p,
            index: flat.unwrap(),
            in_bounds: ok.unwrap(),
            value,
            err: err_param,
            code,
        })
    }

    /// `buf = gs.local(gs.f32, N)`: local array of N elements (N constant).
    fn declare_local_array(&mut self, target: &J, call: &J, shared: bool) -> R<()> {
        let f = if shared { "gs.shared" } else { "gs.local" };
        if shared && self.group.is_none() {
            return Err(self.e(
                ErrorKind::Syntax,
                "gs.shared(...): memory shared by the threads of a group, declare the group size: @gs.kernel(group=256)",
            ));
        }
        if target["_type"] != "Name" {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("{f}(...) is assigned to a variable: buf = {f}(gs.f32, 16)"),
            ));
        }
        let name = target["id"].as_str().unwrap_or("").to_string();
        let args = call["args"].as_array().cloned().unwrap_or_default();
        if args.len() != 2 {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("{f}(type, size): e.g. {f}(gs.f32, 16)"),
            ));
        }
        let ty = self.annotation_type(&args[0])?;
        if !ty.is_numeric() {
            return Err(self.e(
                ErrorKind::Type,
                format!("{f}: numeric type expected (gs.f32, gs.i32…)"),
            ));
        }
        let len = match self.expr(&args[1])? {
            Val::Lit(Lit::Int(v)) if (1..=1 << 16).contains(&v) => v as u32,
            _ => {
                return Err(self.e(
                    ErrorKind::Type,
                    format!("{f}: the size must be an integer constant between 1 and 65536 (literal or global constant)"),
                ));
            }
        };
        if self.local_index.contains_key(&name) || self.params.iter().any(|p| p.name == name) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("\"{name}\" already exists: a local array is declared only once"),
            ));
        }
        let i = self.locals.len();
        self.locals.push(Local {
            name: name.clone(),
            ty,
            len: Some(len),
            shared,
        });
        self.weak.push(false);
        self.local_index.insert(name, i);
        Ok(())
    }

    /// `gs.matmul_u8(c, a, b, m, n, k)`: gs.shared u32 arrays, constant dimensions.
    fn matmul(&mut self, call: &J) -> R<Stmt> {
        const USAGE: &str = "gs.matmul_u8(c, a, b, m, n, k): c, a, b gs.shared(gs.u32, …) arrays, \
                             m, n, k constant integers (k a multiple of 4); c[i·n + j] += Σ a[i, t]·b[j, t] \
                             on bytes (a by rows, b by columns, 4 bytes per word)";
        if self.group.is_none() {
            return Err(self.e(
                ErrorKind::Syntax,
                "gs.matmul_u8: collective operation of a thread group, declare the group size: @gs.kernel(group=256)",
            ));
        }
        let args = call["args"].as_array().cloned().unwrap_or_default();
        if args.len() != 6 {
            return Err(self.e(ErrorKind::Syntax, USAGE));
        }
        let mut arrays = Vec::new();
        for a in &args[..3] {
            let name = a["id"].as_str().unwrap_or("");
            let l = (a["_type"] == "Name")
                .then(|| self.local_index.get(name).copied())
                .flatten()
                .filter(|l| self.locals[*l].shared && self.locals[*l].ty == Scalar::U32)
                .ok_or_else(|| self.e(ErrorKind::Type, format!("« {name} » : {USAGE}")))?;
            arrays.push(l);
        }
        let mut dims = Vec::new();
        for a in &args[3..] {
            match self.expr(a)? {
                Val::Lit(Lit::Int(v)) if (1..=1 << 16).contains(&v) => dims.push(v as u32),
                _ => return Err(self.e(ErrorKind::Type, USAGE)),
            }
        }
        let (m, n, k) = (dims[0], dims[1], dims[2]);
        if k % 4 != 0 {
            return Err(self.e(
                ErrorKind::Type,
                format!("gs.matmul_u8: k = {k} must be a multiple of 4 (4 bytes per word)"),
            ));
        }
        for (l, need, what) in [
            (arrays[0], m * n, "c (m·n mots)"),
            (arrays[1], m * k / 4, "a (m·k/4 mots)"),
            (arrays[2], n * k / 4, "b (n·k/4 mots)"),
        ] {
            let len = self.locals[l].len.unwrap_or(0);
            if len < need {
                return Err(self.e(
                    ErrorKind::Type,
                    format!(
                        "gs.matmul_u8: \"{}\" has {len} words, {need} are needed for {what}",
                        self.locals[l].name
                    ),
                ));
            }
        }
        Ok(Stmt::MatMul {
            c: arrays[0],
            a: arrays[1],
            b: arrays[2],
            m,
            n,
            k,
        })
    }

    /// i32 integer index (u32, i64, u64 and literals are converted).
    fn index_expr(&mut self, node: &J) -> R<Expr> {
        let v = self.expr(node)?;
        match v.ty() {
            Scalar::I32 => self.to_ty(v, Scalar::I32, "index"),
            Scalar::U32 | Scalar::I64 | Scalar::U64 => Ok(cast(self.materialize(v)?, Scalar::I32)),
            t => Err(self.e(
                ErrorKind::Type,
                format!("index of type {t}: an index must be an integer (int(...) if needed)"),
            )),
        }
    }

    /// `x[i, k]` (input array) or `buf[k]` (local array) → (array, flattened index).
    fn subscript(&mut self, node: &J) -> R<(Option<usize>, Option<usize>, Expr)> {
        let base = &node["value"];
        let id = if base["_type"] == "Name" {
            base["id"].as_str().unwrap_or("").to_string()
        } else {
            String::new()
        };
        let idx_nodes: Vec<J> = if node["slice"]["_type"] == "Tuple" {
            node["slice"]["elts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            vec![node["slice"].clone()]
        };
        if idx_nodes.iter().any(|n| n["_type"] == "Slice") {
            return Err(self.e(
                ErrorKind::Syntax,
                "slices x[a:b] are not supported in a kernel: index cell by cell",
            ));
        }
        if let Some(&l) = self.local_index.get(&id) {
            if self.locals[l].len.is_some() {
                if idx_nodes.len() != 1 {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        format!("local array '{id}' has one dimension: {id}[k]"),
                    ));
                }
                let i = self.index_expr(&idx_nodes[0])?;
                return Ok((None, Some(l), i));
            }
        }
        if let Some(p) = self
            .params
            .iter()
            .position(|p| p.name == id && p.kind == ParamKind::Array)
        {
            if self.writable.contains(&p) {
                return Err(self.out_only(&id));
            }
            if self.rank.is_none() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("indexing {id}[...]: in an element-wise kernel, `{id}` already stands for the current element (to index, use gs.index() and named dimensions)"),
                ));
            }
            let dims = self.params[p].dims.clone();
            if idx_nodes.len() != dims.len() {
                return Err(self.e(
                    ErrorKind::Type,
                    format!(
                        "'{id}' has {} dimension(s), {} index(es) given",
                        dims.len(),
                        idx_nodes.len()
                    ),
                ));
            }
            // Flattened index, C order: ((i0 * d1 + i1) * d2 + i2)…
            let mut flat: Option<Expr> = None;
            for (k, n) in idx_nodes.iter().enumerate() {
                let mut i = self.index_expr(n)?;
                if k == 0 && self.chunked.contains(&p) {
                    // Array split by rows: only the block's part is on the device.
                    let base = ex(Scalar::I32, ExprKind::Param(self.inrow.unwrap()));
                    i = ex(
                        Scalar::I32,
                        ExprKind::Binary(BinOp::Sub, Box::new(i), Box::new(base)),
                    );
                }
                flat = Some(match flat {
                    None => i,
                    Some(acc) => {
                        let d = ex(Scalar::I32, ExprKind::Param(dims[k]));
                        let mul = ex(
                            Scalar::I32,
                            ExprKind::Binary(BinOp::Mul, Box::new(acc), Box::new(d)),
                        );
                        ex(
                            Scalar::I32,
                            ExprKind::Binary(BinOp::Add, Box::new(mul), Box::new(i)),
                        )
                    }
                });
            }
            return Ok((Some(p), None, flat.unwrap()));
        }
        Err(self.e(ErrorKind::Syntax, "indexing not supported: only input arrays and local arrays (gs.local) can be indexed".to_string()))
    }

    /// `buf[k] = v`: write into a local array (inputs are read-only).
    fn store(&mut self, target: &J, v: Val) -> R<Stmt> {
        let (param, local, index) = self.subscript(target)?;
        match (param, local) {
            (_, Some(l)) => {
                let ty = self.locals[l].ty;
                let value = self.to_ty(v, ty, "write into a local array")?;
                Ok(Stmt::LocalStore {
                    local: l,
                    index,
                    value,
                })
            }
            (Some(p), _) => Err(self.e(
                ErrorKind::Syntax,
                format!(
                    "'{}' is an input, read-only: the result is written with return",
                    self.params[p].name
                ),
            )),
            _ => unreachable!(),
        }
    }

    /// `x.shape[k]` / `len(x)`: size of an axis (i32).
    fn shape_of(&self, id: &str, axis: i64) -> R<Val> {
        if let Some(&l) = self.local_index.get(id) {
            if let Some(n) = self.locals[l].len {
                if axis == 0 || axis == -1 {
                    return Ok(Val::Lit(Lit::Int(n as i64)));
                }
            }
        }
        let Some(p) = self
            .params
            .iter()
            .find(|p| p.name == id && p.kind == ParamKind::Array)
        else {
            return Err(self.e(ErrorKind::Syntax, format!("\"{id}\" is not an array")));
        };
        if self.rank.is_none() {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("{id}.shape: available in kernels with gs.index() and named dimensions"),
            ));
        }
        let rank = p.dims.len() as i64;
        let a = if axis < 0 { axis + rank } else { axis };
        if !(0..rank).contains(&a) {
            return Err(self.e(
                ErrorKind::Type,
                format!("{id}.shape[{axis}] : '{id}' a {rank} dimension(s)"),
            ));
        }
        Ok(Val::E(ex(Scalar::I32, ExprKind::Param(p.dims[a as usize]))))
    }

    fn assign(&mut self, name: &str, v: Val, declared: Option<Scalar>) -> R<Stmt> {
        if self.params.iter().any(|p| p.name == name) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!(
                    "parameter '{name}' cannot be modified: use a local variable"
                ),
            ));
        }
        if self.constants.contains_key(name) || self.modules.contains_key(name) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("'{name}' shadows a global constant or module"),
            ));
        }
        let idx = match self.local_index.get(name) {
            Some(&i) => {
                if self.locals[i].len.is_some() {
                    return Err(self.e(ErrorKind::Syntax, format!("'{name}' is a local array: write into its cells ({name}[k] = ...)")));
                }
                if let Some(d) = declared {
                    if d != self.locals[i].ty {
                        return Err(self.e(
                            ErrorKind::Type,
                            format!("'{name}' redeclared with another type"),
                        ));
                    }
                }
                let cur = self.locals[i].ty;
                let vt = v.ty();
                // Python is dynamic: an integer local that receives a float becomes
                // a float; a local born from a literal takes the type of the next value.
                let int_to_float = cur.is_int() && vt.is_float();
                let weak_retype = self.weak[i]
                    && vt.is_numeric()
                    && cur.is_numeric()
                    && !(cur.is_float() && vt.is_int());
                if !v.is_lit() && vt != cur && (int_to_float || weak_retype) {
                    self.hints.insert(name.to_string(), vt);
                    self.restart = true;
                } else if matches!(v, Val::Lit(Lit::Float(_))) && cur.is_int() {
                    self.hints.insert(name.to_string(), Scalar::F32);
                    self.restart = true;
                }
                i
            }
            None => {
                let ty = declared
                    .or_else(|| self.hints.get(name).copied())
                    .unwrap_or(v.ty());
                let i = self.locals.len();
                self.locals.push(Local {
                    len: None,
                    name: name.to_string(),
                    ty,
                    shared: false,
                });
                self.weak
                    .push(declared.is_none() && v.is_lit() && !self.hints.contains_key(name));
                self.local_index.insert(name.to_string(), i);
                i
            }
        };
        let ty = self.locals[idx].ty;
        if self.restart {
            // The type will be fixed on the next pass: provisional value.
            return Ok(Stmt::Assign {
                local: idx,
                value: ex(ty, ExprKind::Const(gpusync_ir::interp::zero(ty))),
            });
        }
        let value = self.to_ty(v, ty, &format!("assignment of '{name}'"))?;
        Ok(Stmt::Assign { local: idx, value })
    }

    fn for_loop(&mut self, s: &J) -> R<Stmt> {
        if s["target"]["_type"] != "Name" {
            return Err(self.e(
                ErrorKind::Syntax,
                "for loop: a single variable (for i in range(...))",
            ));
        }
        if s["orelse"].as_array().is_some_and(|a| !a.is_empty()) {
            return Err(self.e(ErrorKind::Syntax, "for ... else is not supported"));
        }
        let it = &s["iter"];
        let is_range =
            it["_type"] == "Call" && it["func"]["_type"] == "Name" && it["func"]["id"] == "range";
        if !is_range {
            return Err(self.e(
                ErrorKind::Syntax,
                "for loop: only `for i in range(...)` is supported in a kernel",
            ));
        }
        let args = it["args"].as_array().cloned().unwrap_or_default();
        if args.is_empty()
            || args.len() > 3
            || it["keywords"].as_array().is_some_and(|k| !k.is_empty())
        {
            return Err(self.e(
                ErrorKind::Syntax,
                "range() expects 1 to 3 positional arguments",
            ));
        }
        let mut vals = args.iter().map(|a| self.expr(a)).collect::<R<Vec<_>>>()?;
        for v in &vals {
            if !v.ty().is_int() {
                return Err(self.e(
                    ErrorKind::Type,
                    "range(): the bounds must be integers",
                ));
            }
        }
        let ty = vals
            .iter()
            .find(|v| !v.is_lit())
            .map(|v| v.ty())
            .unwrap_or(Scalar::I32);
        let step = if vals.len() == 3 {
            vals.pop().unwrap()
        } else {
            Val::Lit(Lit::Int(1))
        };
        let stop = vals.pop().unwrap();
        let start = vals.pop().unwrap_or(Val::Lit(Lit::Int(0)));
        let (start, stop, step) = (
            self.to_ty(start, ty, "range")?,
            self.to_ty(stop, ty, "range")?,
            self.to_ty(step, ty, "range")?,
        );
        let name = s["target"]["id"].as_str().unwrap_or("");
        if let Some(&i) = self.local_index.get(name) {
            if self.locals[i].ty != ty {
                return Err(self.e(
                    ErrorKind::Type,
                    format!(
                        "loop variable '{name}' already used with type {}",
                        self.locals[i].ty
                    ),
                ));
            }
        }
        let Stmt::Assign { local: var, .. } = self.assign(name, Val::E(start.clone()), Some(ty))?
        else {
            unreachable!()
        };
        self.loop_depth += 1;
        let body = self.block(
            s["body"].as_array().map(|v| v.as_slice()).unwrap_or(&[]),
            false,
        );
        self.loop_depth -= 1;
        Ok(Stmt::For {
            var,
            start,
            stop,
            step,
            body: body?,
        })
    }

    fn condition(&mut self, node: &J) -> R<Expr> {
        let v = self.expr(node)?;
        if v.ty() != Scalar::Bool {
            return Err(self.e(
                ErrorKind::Type,
                format!(
                    "condition of type {}: write an explicit comparison (e.g. x != 0)",
                    v.ty()
                ),
            ));
        }
        self.materialize(v)
    }

    // ───────────────────────── expressions ─────────────────────────

    fn module_of(&self, node: &J) -> Option<&str> {
        if node["_type"] == "Name" {
            self.modules.get(node["id"].as_str()?).map(|s| s.as_str())
        } else {
            None
        }
    }

    fn out_only(&self, id: &str) -> FrontendError {
        self.e(
            ErrorKind::Syntax,
            format!("'{id}' is a gs.Out output, write-only: write it with gs.atomic_add({id}, index, value) (or atomic_min, atomic_max)"),
        )
    }

    fn name(&self, id: &str) -> R<Val> {
        if let Some(i) = self.params.iter().position(|p| p.name == id && !p.hidden) {
            if self.writable.contains(&i) {
                return Err(self.out_only(id));
            }
            if self.params[i].kind == ParamKind::Array && self.rank.is_some() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("'{id}' is an array: index it ({id}[i, ...]) with the coordinates from gs.index()"),
                ));
            }
            return Ok(Val::E(ex(self.params[i].ty, ExprKind::Param(i))));
        }
        if let Some(&i) = self.local_index.get(id) {
            if self.locals[i].len.is_some() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("'{id}' is a local array: index it ({id}[k])"),
                ));
            }
            return Ok(Val::E(ex(self.locals[i].ty, ExprKind::Local(i))));
        }
        if let Some(l) = self.constants.get(id) {
            return Ok(Val::Lit(*l));
        }
        Err(self.e(
            ErrorKind::Syntax,
            format!("unknown name \"{id}\": only parameters, local variables and int/float/bool global constants can be used"),
        ))
    }

    fn expr(&mut self, node: &J) -> R<Val> {
        self.set_line(node);
        match node["_type"].as_str().unwrap_or("") {
            "Constant" => match parse_lit(&node["value"]) {
                Some(l) => Ok(Val::Lit(l)),
                None => Err(self.e(
                    ErrorKind::Syntax,
                    "only numeric and boolean constants are supported",
                )),
            },
            "Name" => self.name(node["id"].as_str().unwrap_or("")),
            "BinOp" => {
                let a = self.expr(&node["left"])?;
                let b = self.expr(&node["right"])?;
                self.binop(op_name(&node["op"]), a, b)
            }
            "UnaryOp" => {
                let a = self.expr(&node["operand"])?;
                match node["op"]["_type"].as_str().unwrap_or("") {
                    "UAdd" => Ok(a),
                    "USub" => match a {
                        Val::Lit(Lit::Int(v)) => Ok(Val::Lit(Lit::Int(v.wrapping_neg()))),
                        Val::Lit(Lit::Float(v)) => Ok(Val::Lit(Lit::Float(-v))),
                        Val::E(e) if e.ty.is_numeric() => Ok(Val::E(ex(e.ty, ExprKind::Unary(UnOp::Neg, Box::new(e))))),
                        _ => Err(self.e(ErrorKind::Type, "negation of a non-numeric value")),
                    },
                    "Not" => match a {
                        Val::Lit(Lit::Bool(b)) => Ok(Val::Lit(Lit::Bool(!b))),
                        Val::E(e) if e.ty == Scalar::Bool => Ok(Val::E(ex(Scalar::Bool, ExprKind::Unary(UnOp::Not, Box::new(e))))),
                        _ => Err(self.e(ErrorKind::Type, "`not` expects a boolean: write an explicit comparison (e.g. x == 0)")),
                    },
                    "Invert" => match a {
                        Val::Lit(Lit::Int(v)) => Ok(Val::Lit(Lit::Int(!v))),
                        Val::E(e) if e.ty.is_int() => Ok(Val::E(ex(e.ty, ExprKind::Unary(UnOp::BitNot, Box::new(e))))),
                        _ => Err(self.e(ErrorKind::Type, "~ expects an integer (use `not` for a boolean)")),
                    },
                    o => Err(self.e(ErrorKind::Syntax, format!("unary operator \"{o}\" not supported"))),
                }
            }
            "BoolOp" => {
                let op = if node["op"]["_type"] == "And" {
                    BinOp::And
                } else {
                    BinOp::Or
                };
                let values = node["values"].as_array().cloned().unwrap_or_default();
                let mut acc: Option<Expr> = None;
                for v in &values {
                    let v = self.expr(v)?;
                    if v.ty() != Scalar::Bool {
                        return Err(self.e(
                            ErrorKind::Type,
                            "`and` / `or` expect booleans (comparisons)",
                        ));
                    }
                    let e = self.materialize(v)?;
                    acc = Some(match acc {
                        None => e,
                        Some(a) => ex(Scalar::Bool, ExprKind::Binary(op, Box::new(a), Box::new(e))),
                    });
                }
                Ok(Val::E(acc.ok_or_else(|| {
                    self.e(ErrorKind::Syntax, "empty boolean expression")
                })?))
            }
            "Compare" => {
                let ops = node["ops"].as_array().cloned().unwrap_or_default();
                let comps = node["comparators"].as_array().cloned().unwrap_or_default();
                let mut left = self.expr(&node["left"])?;
                let mut acc: Option<Expr> = None;
                for (op, rhs) in ops.iter().zip(&comps) {
                    let right = self.expr(rhs)?;
                    let cmp = match op["_type"].as_str().unwrap_or("") {
                        "Eq" => CmpOp::Eq,
                        "NotEq" => CmpOp::Ne,
                        "Lt" => CmpOp::Lt,
                        "LtE" => CmpOp::Le,
                        "Gt" => CmpOp::Gt,
                        "GtE" => CmpOp::Ge,
                        o => {
                            return Err(self.e(
                                ErrorKind::Syntax,
                                format!("comparison \"{o}\" not supported (is, in…)"),
                            ));
                        }
                    };
                    let ty = if left.ty() == Scalar::Bool
                        && right.ty() == Scalar::Bool
                        && matches!(cmp, CmpOp::Eq | CmpOp::Ne)
                    {
                        Scalar::Bool
                    } else {
                        self.unify(&left, &right, "comparison")?
                    };
                    let (a, b) = (
                        self.to_ty(left, ty, "comparison")?,
                        self.to_ty(right.clone(), ty, "comparison")?,
                    );
                    let c = ex(
                        Scalar::Bool,
                        ExprKind::Compare(cmp, Box::new(a), Box::new(b)),
                    );
                    acc = Some(match acc {
                        None => c,
                        Some(prev) => ex(
                            Scalar::Bool,
                            ExprKind::Binary(BinOp::And, Box::new(prev), Box::new(c)),
                        ),
                    });
                    left = right;
                }
                Ok(Val::E(acc.ok_or_else(|| {
                    self.e(ErrorKind::Syntax, "empty comparison")
                })?))
            }
            "IfExp" => {
                let cond = self.condition(&node["test"])?;
                let a = self.expr(&node["body"])?;
                let b = self.expr(&node["orelse"])?;
                let ty = if a.ty() == Scalar::Bool && b.ty() == Scalar::Bool {
                    Scalar::Bool
                } else {
                    self.unify(&a, &b, "a if c else b")?
                };
                let (a, b) = (
                    self.to_ty(a, ty, "conditional expression")?,
                    self.to_ty(b, ty, "conditional expression")?,
                );
                Ok(Val::E(ex(
                    ty,
                    ExprKind::Select(Box::new(cond), Box::new(a), Box::new(b)),
                )))
            }
            "Call" => self.call(node),
            "Attribute" => {
                let attr = node["attr"].as_str().unwrap_or("");
                match (self.module_of(&node["value"]), attr) {
                    (Some("math" | "gpusync"), "pi") => {
                        Ok(Val::Lit(Lit::Float(std::f64::consts::PI)))
                    }
                    (Some("math" | "gpusync"), "e") => {
                        Ok(Val::Lit(Lit::Float(std::f64::consts::E)))
                    }
                    (Some("math" | "gpusync"), "tau") => {
                        Ok(Val::Lit(Lit::Float(std::f64::consts::TAU)))
                    }
                    (Some("math" | "gpusync"), "inf") => Ok(Val::Lit(Lit::Float(f64::INFINITY))),
                    (Some("math" | "gpusync"), "nan") => Ok(Val::Lit(Lit::Float(f64::NAN))),
                    _ => Err(self.e(
                        ErrorKind::Syntax,
                        format!("attribute \".{attr}\" not supported in a kernel"),
                    )),
                }
            }
            "Subscript" => {
                let base = &node["value"];
                if base["_type"] == "Attribute"
                    && base["attr"] == "shape"
                    && base["value"]["_type"] == "Name"
                {
                    let axis = match parse_lit(&node["slice"]["value"]) {
                        Some(Lit::Int(a)) if node["slice"]["_type"] == "Constant" => a,
                        _ => match &node["slice"] {
                            n if n["_type"] == "UnaryOp" && n["op"]["_type"] == "USub" => {
                                match parse_lit(&n["operand"]["value"]) {
                                    Some(Lit::Int(a)) => -a,
                                    _ => {
                                        return Err(self.e(
                                            ErrorKind::Syntax,
                                            "x.shape[k]: k must be a constant integer",
                                        ));
                                    }
                                }
                            }
                            _ => {
                                return Err(self.e(
                                    ErrorKind::Syntax,
                                    "x.shape[k]: k must be a constant integer",
                                ));
                            }
                        },
                    };
                    return self.shape_of(base["value"]["id"].as_str().unwrap_or(""), axis);
                }
                let (param, local, index) = self.subscript(node)?;
                Ok(Val::E(match (param, local) {
                    (Some(p), _) => ex(self.params[p].ty, ExprKind::Load(p, Box::new(index))),
                    (_, Some(l)) => ex(self.locals[l].ty, ExprKind::LocalLoad(l, Box::new(index))),
                    _ => unreachable!(),
                }))
            }
            "List" | "ListComp" | "Dict" | "Set" | "Tuple" => Err(self.e(
                ErrorKind::Syntax,
                "lists, tuples and dictionaries are not supported in a kernel",
            )),
            "Lambda" => Err(self.e(ErrorKind::Syntax, "lambda not supported in a kernel")),
            other => Err(self.e(
                ErrorKind::Syntax,
                format!("Python expression not supported in a kernel: {other}"),
            )),
        }
    }

    fn binop(&mut self, op: &str, a: Val, b: Val) -> R<Val> {
        let bop = match op {
            "Add" => BinOp::Add,
            "Sub" => BinOp::Sub,
            "Mult" => BinOp::Mul,
            "Div" => BinOp::Div,
            "FloorDiv" => BinOp::FloorDiv,
            "Mod" => BinOp::Mod,
            "Pow" => BinOp::Pow,
            "BitAnd" => BinOp::BitAnd,
            "BitOr" => BinOp::BitOr,
            "BitXor" => BinOp::BitXor,
            "LShift" => BinOp::Shl,
            "RShift" => BinOp::Shr,
            o => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("operator \"{o}\" not supported (@…)"),
                ));
            }
        };
        if let (Val::Lit(x), Val::Lit(y)) = (&a, &b) {
            if let Some(l) = fold(bop, *x, *y) {
                return Ok(Val::Lit(l));
            }
        }
        if matches!(
            bop,
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
        ) {
            return self.bitop(bop, a, b);
        }
        let mut ty = self.unify(&a, &b, op_symbol(bop))?;
        if bop == BinOp::Div && ty.is_int() {
            ty = Scalar::F32; // `/` is a true division, as in Python
        }
        if bop == BinOp::Pow && ty.is_int() {
            return Err(self.e(
                ErrorKind::Type,
                "** on integers is not supported: use floats (float(x) ** 2) or a multiplication",
            ));
        }
        let (x, y) = (
            self.to_ty(a, ty, op_symbol(bop))?,
            self.to_ty(b, ty, op_symbol(bop))?,
        );
        Ok(Val::E(ex(
            ty,
            ExprKind::Binary(bop, Box::new(x), Box::new(y)),
        )))
    }

    /// `&`, `|`, `^` (integers, or booleans for all three), `<<` and `>>` (integers).
    fn bitop(&mut self, op: BinOp, a: Val, b: Val) -> R<Val> {
        let sym = op_symbol(op);
        if a.ty() == Scalar::Bool
            && b.ty() == Scalar::Bool
            && !matches!(op, BinOp::Shl | BinOp::Shr)
        {
            let (x, y) = (self.materialize(a)?, self.materialize(b)?);
            return Ok(Val::E(match op {
                BinOp::BitAnd => ex(
                    Scalar::Bool,
                    ExprKind::Binary(BinOp::And, Box::new(x), Box::new(y)),
                ),
                BinOp::BitOr => ex(
                    Scalar::Bool,
                    ExprKind::Binary(BinOp::Or, Box::new(x), Box::new(y)),
                ),
                _ => ex(
                    Scalar::Bool,
                    ExprKind::Compare(CmpOp::Ne, Box::new(x), Box::new(y)),
                ),
            }));
        }
        let not_int = |v: &Val| !v.ty().is_int() || matches!(v, Val::Lit(Lit::Float(_)));
        if not_int(&a) || not_int(&b) {
            return Err(self.e(
                ErrorKind::Type,
                format!(
                    "\"{sym}\" expects integers ({} and {} received)",
                    a.ty(),
                    b.ty()
                ),
            ));
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            // Type of the shifted value; the count is converted to that type.
            let ty = match (&a, &b) {
                (Val::Lit(_), Val::E(e)) => e.ty,
                _ => a.ty(),
            };
            let x = self.to_ty(a, ty, sym)?;
            let y = match b {
                Val::Lit(l) => self.lit_to(l, ty)?,
                Val::E(e) => cast(e, ty),
            };
            return Ok(Val::E(ex(
                ty,
                ExprKind::Binary(op, Box::new(x), Box::new(y)),
            )));
        }
        let ty = self.unify(&a, &b, sym)?;
        let (x, y) = (self.to_ty(a, ty, sym)?, self.to_ty(b, ty, sym)?);
        Ok(Val::E(ex(
            ty,
            ExprKind::Binary(op, Box::new(x), Box::new(y)),
        )))
    }

    fn call(&mut self, node: &J) -> R<Val> {
        if node["keywords"].as_array().is_some_and(|k| !k.is_empty()) {
            return Err(self.e(
                ErrorKind::Syntax,
                "keyword arguments are not supported in kernel calls",
            ));
        }
        let func = &node["func"];
        let (module_owned, fname) = match func["_type"].as_str() {
            Some("Name") => (None, func["id"].as_str().unwrap_or("")),
            Some("Attribute") => (
                self.module_of(&func["value"]).map(str::to_string),
                func["attr"].as_str().unwrap_or(""),
            ),
            _ => return Err(self.e(ErrorKind::Syntax, "function call not supported")),
        };
        let module = module_owned.as_deref();
        let args = node["args"].as_array().cloned().unwrap_or_default();
        // Calls whose arguments are not values: handled before evaluating the arguments.
        match (module, fname) {
            (None, "len") => {
                let a = args
                    .first()
                    .filter(|a| a["_type"] == "Name" && args.len() == 1);
                return match a {
                    Some(a) => self.shape_of(a["id"].as_str().unwrap_or(""), 0),
                    None => Err(self.e(ErrorKind::Syntax, "len(x): x must be an array")),
                };
            }
            (Some("gpusync"), "index") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.index() is used alone in an assignment: i, j = gs.index()",
                ));
            }
            (Some("gpusync"), "local") => {
                return Err(self.e(ErrorKind::Syntax, "gs.local(...) is used alone in an assignment: buf = gs.local(gs.f32, 16)"));
            }
            (Some("gpusync"), "shared") => {
                return Err(self.e(ErrorKind::Syntax, "gs.shared(...) is used alone in an assignment: t = gs.shared(gs.f32, 256)"));
            }
            (Some("gpusync"), "store") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.store(k, value) is used alone, as a statement",
                ));
            }
            (Some("gpusync"), "matmul_u8") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.matmul_u8(...) is used alone, as a statement",
                ));
            }
            (Some("gpusync"), "barrier") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.barrier() is used alone, as a statement",
                ));
            }
            (Some("gpusync"), "lane") => {
                if self.group.is_none() {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        "gs.lane(): position of the thread in its group, declare the group size: @gs.kernel(group=256)",
                    ));
                }
                if !args.is_empty() {
                    return Err(self.e(ErrorKind::Syntax, "gs.lane() takes no argument"));
                }
                return Ok(Val::E(ex(Scalar::I32, ExprKind::Lane)));
            }
            _ => {}
        }
        let mut vals = args.iter().map(|a| self.expr(a)).collect::<R<Vec<_>>>()?;

        // Conversions explicites
        let cast_to = match (module, fname) {
            (None, "float") => Some(Scalar::F32),
            (None, "int") => Some(Scalar::I32),
            (None, "bool") => Some(Scalar::Bool),
            (Some("gpusync"), n) => Scalar::parse(n).filter(|s| *s != Scalar::Bool),
            _ => None,
        };
        if let Some(to) = cast_to {
            if vals.len() != 1 {
                return Err(self.e(ErrorKind::Syntax, format!("{fname}() expects one argument")));
            }
            let v = vals.pop().unwrap();
            return Ok(Val::E(match v {
                Val::Lit(l) => match (l, to) {
                    (Lit::Float(f), t) if t.is_int() => {
                        self.lit_to(Lit::Int(f.trunc() as i64), t)?
                    }
                    (Lit::Bool(b), t) if t != Scalar::Bool => self.lit_to(Lit::Int(b as i64), t)?,
                    (l, t) => self.lit_to(l, t)?,
                },
                Val::E(e) => cast(e, to),
            }));
        }

        let f = match (module, fname) {
            (None, "abs") | (Some("gpusync"), "abs") => Intrinsic::Abs,
            (None, "min") | (Some("gpusync"), "min") => Intrinsic::Min,
            (None, "max") | (Some("gpusync"), "max") => Intrinsic::Max,
            (Some("math"), "fabs") => Intrinsic::Abs,
            (Some("gpusync"), "mulhi") => Intrinsic::MulHi,
            (Some("math" | "gpusync"), n) => match n {
                "sqrt" => Intrinsic::Sqrt,
                "exp" => Intrinsic::Exp,
                "exp2" => Intrinsic::Exp2,
                "log" => Intrinsic::Log,
                "log2" => Intrinsic::Log2,
                "sin" => Intrinsic::Sin,
                "cos" => Intrinsic::Cos,
                "tan" => Intrinsic::Tan,
                "tanh" => Intrinsic::Tanh,
                "atan2" => Intrinsic::Atan2,
                "floor" => Intrinsic::Floor,
                "ceil" => Intrinsic::Ceil,
                "pow" => Intrinsic::Pow,
                _ => {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        format!("function \"{fname}\" not available in a kernel"),
                    ));
                }
            },
            (None, "range") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "range() can only be used in `for i in range(...)`",
                ));
            }

            _ => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("call to \"{fname}\" not supported: only gs and math functions can be used in a kernel"),
                ));
            }
        };

        // min / max with more than two arguments: folded pairwise.
        if matches!(f, Intrinsic::Min | Intrinsic::Max) && vals.len() > 2 {
            let mut it = vals.into_iter();
            let mut acc = it.next().unwrap();
            for v in it {
                acc = self.intrinsic(f, vec![acc, v], fname)?;
            }
            return Ok(acc);
        }
        let v = self.intrinsic(f, vals, fname)?;
        // math.floor / math.ceil return an integer in Python.
        if module == Some("math") && matches!(f, Intrinsic::Floor | Intrinsic::Ceil) {
            let e = self.materialize(v)?;
            return Ok(Val::E(cast(e, Scalar::I32)));
        }
        Ok(v)
    }

    fn intrinsic(&mut self, f: Intrinsic, vals: Vec<Val>, fname: &str) -> R<Val> {
        if vals.len() != f.arity() {
            return Err(self.e(
                ErrorKind::Syntax,
                format!(
                    "{fname}() expects {} argument(s), {} received",
                    f.arity(),
                    vals.len()
                ),
            ));
        }
        let mut ty = if vals.len() == 2 {
            self.unify(&vals[0], &vals[1], fname)?
        } else {
            vals[0].ty()
        };
        if !ty.is_numeric() {
            return Err(self.e(ErrorKind::Type, format!("{fname}() expects a number")));
        }
        if f == Intrinsic::MulHi && !ty.is_int() {
            return Err(self.e(
                ErrorKind::Type,
                format!("gs.mulhi() expects two integers of the same type ({ty} received)"),
            ));
        }
        if ty.is_int() && !f.accepts_int() {
            ty = Scalar::F32; // math.sqrt(4) works in Python
        }
        let args = vals
            .into_iter()
            .map(|v| self.to_ty(v, ty, fname))
            .collect::<R<Vec<_>>>()?;
        Ok(Val::E(ex(ty, ExprKind::Call(f, args))))
    }
}

/// Folding of operations between literals, with Python semantics.
fn fold(op: BinOp, a: Lit, b: Lit) -> Option<Lit> {
    use Lit::*;
    Some(match (a, b) {
        (Int(x), Int(y)) => match op {
            BinOp::Add => Int(x.checked_add(y)?),
            BinOp::Sub => Int(x.checked_sub(y)?),
            BinOp::Mul => Int(x.checked_mul(y)?),
            BinOp::Div => Float(x as f64 / y as f64),
            BinOp::FloorDiv if y != 0 => Int(x.div_euclid(y)
                - if (x.rem_euclid(y) != 0) && y < 0 {
                    1
                } else {
                    0
                }),
            BinOp::Mod if y != 0 => Int(((x % y) + y) % y),
            BinOp::Pow if (0..=62).contains(&y) => Int(x.checked_pow(y as u32)?),
            BinOp::BitAnd => Int(x & y),
            BinOp::BitOr => Int(x | y),
            BinOp::BitXor => Int(x ^ y),
            BinOp::Shl if (0..64).contains(&y) => Int(i64::try_from((x as i128) << y)
                .ok()
                .filter(|_| (x as i128) << y >> y == x as i128)?),
            BinOp::Shr if y >= 0 => Int(x >> y.min(63)),
            _ => return None,
        },
        (Int(_) | Float(_), Int(_) | Float(_)) => {
            let (x, y) = (as_f(a), as_f(b));
            Float(match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::FloorDiv => (x / y).floor(),
                BinOp::Mod => mod_f64(x, y),
                BinOp::Pow => x.powf(y),
                _ => return None,
            })
        }
        _ => return None,
    })
}

fn as_f(l: Lit) -> f64 {
    match l {
        Lit::Int(v) => v as f64,
        Lit::Float(v) => v,
        Lit::Bool(b) => b as i64 as f64,
    }
}

fn op_name(op: &J) -> &str {
    op["_type"].as_str().unwrap_or("")
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::FloorDiv => "//",
        BinOp::Mod => "%",
        BinOp::Pow => "**",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

fn stmt_label(t: &str) -> String {
    match t {
        "FunctionDef" => "nested function definition".into(),
        "With" => "with".into(),
        "Try" => "try / except".into(),
        "Raise" => "raise".into(),
        "Import" | "ImportFrom" => "import".into(),
        "Global" | "Nonlocal" => "global / nonlocal".into(),
        "Delete" => "del".into(),
        "Assert" => "assert".into(),
        other => other.to_string(),
    }
}

/// Does the block end with a `return` on every path?
fn always_returns(block: &Block) -> bool {
    match block.last() {
        Some(Stmt::Return(_)) => true,
        Some(Stmt::If {
            then, otherwise, ..
        }) => always_returns(then) && always_returns(otherwise),
        _ => false,
    }
}
