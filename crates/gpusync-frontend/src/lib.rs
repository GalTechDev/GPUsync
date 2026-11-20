//! Frontend : AST Python (sérialisé en JSON par le package Python) → IR typée (ARCHITECTURE.md §5).
//!
//! Entrée (JSON) :
//! ```json
//! { "name": "normalize", "output": "f32", "line_offset": 41, "filename": "demo.py",
//!   "params": [{"name": "x", "kind": "array", "dtype": "f32"}, {"name": "mean", "kind": "scalar", "dtype": "f32"}],
//!   "body": [ <nœuds ast.stmt> ],
//!   "constants": {"N": {"kind": "int", "repr": "3"}},
//!   "modules": {"gs": "gpusync", "math": "math"} }
//! ```
//! Chaque nœud AST est `{"_type": "BinOp", <champs ast>, "lineno": n}` ; un `ast.Constant` porte
//! `"value": {"kind": "int|float|bool|str|none", "repr": "..."}`.
//!
//! Typage (§5.3) : les paramètres sont typés par la signature ; un littéral prend le type de son
//! contexte (`x * 2` reste en f32) ; une locale prend le type de ses affectations (`acc = 0` puis
//! `acc += x` avec x flottant → acc flottant, comme en Python). Jamais de promotion implicite vers f64.

use std::collections::HashMap;

use gpusync_ir::*;
use serde_json::Value as J;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Construction Python non supportée dans un kernel.
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
            Some(l) => write!(f, "ligne {l} : {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for FrontendError {}

type R<T> = Result<T, FrontendError>;

/// Point d'entrée : JSON (voir l'en-tête du module) → kernel typé.
pub fn lower_json(spec: &str) -> R<Kernel> {
    let v: J = serde_json::from_str(spec)
        .map_err(|e| err(ErrorKind::Syntax, format!("JSON invalide : {e}"), None))?;
    lower(&v)
}

pub fn lower(spec: &J) -> R<Kernel> {
    let name = spec["name"].as_str().unwrap_or("kernel").to_string();
    // Sorties : liste {"dtype", "dims"} (ou ancien champ "output" : une sortie).
    let out_specs: Vec<J> = match spec["outputs"].as_array() {
        Some(o) => o.clone(),
        None => vec![serde_json::json!({ "dtype": spec["output"].clone(), "dims": null })],
    };
    let mut outputs = Vec::new();
    for o in &out_specs {
        let t = parse_dtype(&o["dtype"], "type de retour")?;
        if !t.is_numeric() {
            return Err(err(
                ErrorKind::Type,
                "le type de retour doit être numérique (f32, f64, i32, u32)".into(),
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
    let indexed = uses_index(&J::Array(body_json.clone()), &modules);

    let mut params = Vec::new();
    let mut dim_names: Vec<String> = Vec::new(); // dimensions nommées, dans l'ordre d'apparition
    let mut param_dims: Vec<Vec<String>> = Vec::new();
    for p in spec["params"].as_array().cloned().unwrap_or_default() {
        let pname = p["name"].as_str().unwrap_or("?").to_string();
        let ty = parse_dtype(&p["dtype"], &format!("paramètre '{pname}'"))?;
        if !ty.is_numeric() {
            return Err(err(
                ErrorKind::Type,
                format!("paramètre '{pname}' : type {ty} non supporté (f32, f64, i32, u32)"),
                None,
            ));
        }
        let kind = match p["kind"].as_str() {
            Some("array") => ParamKind::Array,
            _ => ParamKind::Scalar,
        };
        let mut dims = Vec::new();
        if kind == ParamKind::Array {
            match p["dims"].as_array() {
                Some(ds) => {
                    for (a, d) in ds.iter().enumerate() {
                        // Axe sans nom (`gs.f32[:, :]`) : nom propre au paramètre.
                        dims.push(
                            d.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| format!("{pname}.{a}")),
                        );
                    }
                }
                None if indexed => {
                    return Err(err(
                        ErrorKind::Type,
                        format!(
                            "paramètre '{pname}' : dans un kernel avec gs.index(), précisez les dimensions, ex. {pname}: gs.f32[\"n\", \"t\"]"
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
            "un kernel a besoin d'au moins un tableau en paramètre (ex. x: gs.f32[:])".into(),
            None,
        ));
    }
    let grid = if indexed {
        // Dimensions des sorties : nommées, toutes identiques, présentes dans les entrées.
        let mut out_dims: Option<Vec<String>> = None;
        for o in &out_specs {
            let ds: Vec<String> = match o["dims"].as_array() {
                Some(ds) if ds.iter().all(|d| d.is_string()) => {
                    ds.iter().map(|d| d.as_str().unwrap().to_string()).collect()
                }
                _ => {
                    return Err(err(
                        ErrorKind::Type,
                        "dans un kernel avec gs.index(), nommez les dimensions de la sortie, ex. -> gs.f32[\"n\", \"n\"]".into(),
                        None,
                    ));
                }
            };
            for d in &ds {
                if !dim_names.contains(d) {
                    return Err(err(
                        ErrorKind::Type,
                        format!(
                            "dimension « {d} » de la sortie absente des entrées : sa taille est inconnue"
                        ),
                        None,
                    ));
                }
            }
            match &out_dims {
                Some(prev) if prev != &ds => {
                    return Err(err(
                        ErrorKind::Type,
                        "toutes les sorties doivent avoir les mêmes dimensions".into(),
                        None,
                    ));
                }
                _ => out_dims = Some(ds),
            }
        }
        // Paramètres cachés : taille de chaque dimension (i32), fournis par le runtime.
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
        // Paramètres du moteur pour le lancement par blocs de lignes (§9.7).
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
        let readers = row_readers(&J::Array(body_json.clone()), &modules);
        let chunked = param_dims
            .iter()
            .enumerate()
            .filter(|(p, ds)| {
                params[*p].kind == ParamKind::Array
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
    let (rank, chunked, inrow) = match &grid {
        Grid::Indexed {
            axes,
            chunked,
            inrow,
            ..
        } => (Some(axes.len()), chunked.clone(), Some(*inrow)),
        Grid::Elementwise => (None, vec![], None),
    };

    let mut hints: HashMap<String, Scalar> = HashMap::new();
    for _pass in 0..16 {
        let (body, locals, restart) = {
            let mut lw = Lowerer {
                params: &params,
                outputs: &outputs,
                rank,
                chunked: &chunked,
                inrow,
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
            };
            let body = lw.block(&body_json, true)?;
            (body, std::mem::take(&mut lw.locals), lw.restart)
        };
        if restart {
            continue;
        }
        if !always_returns(&body) {
            return Err(err(
                ErrorKind::Syntax,
                "le kernel doit se terminer par `return` dans toutes les branches".into(),
                None,
            ));
        }
        return Ok(Kernel {
            name,
            params,
            outputs,
            locals,
            body,
            grid,
        });
    }
    Err(err(
        ErrorKind::Type,
        "impossible de déterminer le type des variables locales".into(),
        None,
    ))
}

/// Lecture des tableaux par ligne (pour le découpage par blocs de lignes, §9.7).
struct RowReaders {
    /// Une seule affectation `i, … = gs.index()`, et `i` n'est jamais réaffecté.
    ok: bool,
    /// Tableaux indexés au moins une fois avec un premier indice autre que `i`.
    other: Vec<String>,
}

/// Quels tableaux ne sont lus qu'à la ligne courante (`x[i, …]`, `i` premier axe de `gs.index()`) ?
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

/// Le corps appelle-t-il `gs.index()` ?
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
        .ok_or_else(|| err(ErrorKind::Type, format!("{what} : type inconnu {v}"), None))
}

/// Littéral non encore typé (« faible ») : prend le type de son contexte.
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

/// Valeur pendant l'abaissement : littéral faible ou expression typée.
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
    /// Rang de la grille pour un kernel avec index ; `None` : élément par élément.
    rank: Option<usize>,
    /// Tableaux lus seulement à la ligne courante : indexés à partir de la ligne `inrow`.
    chunked: &'a [usize],
    inrow: Option<usize>,
    constants: &'a HashMap<String, Lit>,
    modules: &'a HashMap<String, String>,
    line_offset: u32,
    locals: Vec<Local>,
    local_index: HashMap<String, usize>,
    /// La locale a-t-elle été créée à partir d'un littéral (type encore « souple ») ?
    weak: Vec<bool>,
    hints: &'a mut HashMap<String, Scalar>,
    restart: bool,
    loop_depth: u32,
    line: Option<u32>,
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

    /// Littéral → constante du type demandé (vérifie la plage).
    fn lit_to(&self, l: Lit, ty: Scalar) -> R<Expr> {
        let c = match (l, ty) {
            (Lit::Bool(b), Scalar::Bool) => Const::Bool(b),
            (Lit::Int(v), Scalar::I32) => Const::I32(
                i32::try_from(v)
                    .map_err(|_| self.e(ErrorKind::Type, format!("{v} dépasse la plage de i32")))?,
            ),
            (Lit::Int(v), Scalar::U32) => Const::U32(u32::try_from(v).map_err(|_| {
                self.e(
                    ErrorKind::Type,
                    format!("{v} hors de la plage de u32 (entiers positifs < 2³²)"),
                )
            })?),
            (Lit::Int(v), Scalar::F32) => Const::F32(v as f32),
            (Lit::Int(v), Scalar::F64) => Const::F64(v as f64),
            (Lit::Float(v), Scalar::F32) => Const::F32(v as f32),
            (Lit::Float(v), Scalar::F64) => Const::F64(v),
            (Lit::Float(v), t) if t.is_int() => {
                return Err(self.e(
                    ErrorKind::Type,
                    format!("littéral flottant {v} dans un contexte entier ({t}) : utilisez int(...) ou un entier"),
                ));
            }
            (l, t) => {
                return Err(self.e(
                    ErrorKind::Type,
                    format!("{l:?} incompatible avec le type {t}"),
                ));
            }
        };
        Ok(ex(ty, ExprKind::Const(c)))
    }

    /// Valeur → expression du type exact `ty`, avec les conversions implicites de Python
    /// (entier → flottant, f32 → f64). Les autres conversions doivent être explicites.
    fn to_ty(&self, v: Val, ty: Scalar, what: &str) -> R<Expr> {
        match v {
            Val::Lit(l) => self.lit_to(l, ty),
            Val::E(e) if e.ty == ty => Ok(e),
            Val::E(e) if e.ty.is_int() && ty.is_float() => Ok(cast(e, ty)),
            Val::E(e) if e.ty == Scalar::F32 && ty == Scalar::F64 => Ok(cast(e, ty)),
            Val::E(e) => Err(self.e(
                ErrorKind::Type,
                format!("{what} : {} au lieu de {ty} — conversion explicite nécessaire (ex. gs.{ty}(...))", e.ty),
            )),
        }
    }

    fn materialize(&self, v: Val) -> R<Expr> {
        let ty = v.ty();
        self.to_ty(v, ty, "valeur")
    }

    /// Type commun de deux opérandes numériques (règles du §5.3).
    fn unify(&self, a: &Val, b: &Val, op: &str) -> R<Scalar> {
        let (ta, tb) = (a.ty(), b.ty());
        if ta == Scalar::Bool || tb == Scalar::Bool {
            return Err(self.e(
                ErrorKind::Type,
                format!("opération « {op} » sur un booléen"),
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
                } else if tb.is_float() {
                    tb
                } else {
                    return Err(self.e(
                        ErrorKind::Type,
                        format!("« {op} » entre {ta} et {tb} : conversion explicite nécessaire (gs.i32(...) ou gs.u32(...))"),
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
            // Docstring : première instruction, chaîne seule.
            if top && i == 0 && t == "Expr" && s["value"]["value"]["kind"] == "str" {
                continue;
            }
            match t {
                "Assign" => {
                    let targets = s["targets"].as_array().cloned().unwrap_or_default();
                    if targets.len() != 1 {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "affectation multiple a = b = ... non supportée",
                        ));
                    }
                    let target = &targets[0];
                    let value = &s["value"];
                    if self.is_gs_call(value, "index") {
                        out.extend(self.assign_index(target)?);
                    } else if self.is_gs_call(value, "local") {
                        self.declare_local_array(target, value)?;
                    } else if target["_type"] == "Subscript" {
                        let v = self.expr(value)?;
                        out.push(self.store(target, v)?);
                    } else if target["_type"] == "Name" {
                        let v = self.expr(value)?;
                        out.push(self.assign(target["id"].as_str().unwrap_or(""), v, None)?);
                    } else {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "affectation non supportée : une variable (a = ...) ou une case de tableau local (buf[k] = ...) à gauche ; a, b = ... seulement avec gs.index()",
                        ));
                    }
                }
                "AnnAssign" => {
                    if s["target"]["_type"] != "Name" || s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "annotation sans valeur ou sur autre chose qu'une variable",
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
                            "x op= ... : seules les variables simples sont supportées",
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
                        return Err(self.e(ErrorKind::Syntax, "while ... else n'est pas supporté"));
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
                            format!("{} hors d'une boucle", t.to_lowercase()),
                        ));
                    }
                    out.push(if t == "Break" {
                        Stmt::Break
                    } else {
                        Stmt::Continue
                    });
                }
                "Return" => {
                    if s["value"].is_null() {
                        return Err(self.e(
                            ErrorKind::Syntax,
                            "return sans valeur : le kernel doit renvoyer l'élément de sortie",
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
                                "{} valeur(s) renvoyée(s) pour {} sortie(s) déclarée(s) (-> ({})…)",
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
                        es.push(self.to_ty(v, t, "valeur renvoyée")?);
                    }
                    out.push(Stmt::Return(es));
                }
                "Pass" => {}
                "Expr" => {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        "expression sans effet (un appel de fonction seul n'est pas supporté)",
                    ));
                }
                other => {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        format!(
                            "instruction Python non supportée dans un kernel : {}",
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
                    "annotation de variable : utilisez gs.f32, gs.i32, gs.u32, gs.f64 ou bool",
                )
            }),
        }
    }

    /// `node` est-il un appel `gs.<fname>(...)` ?
    fn is_gs_call(&self, node: &J, fname: &str) -> bool {
        node["_type"] == "Call"
            && node["func"]["_type"] == "Attribute"
            && node["func"]["attr"] == fname
            && self.module_of(&node["func"]["value"]) == Some("gpusync")
    }

    /// `i, j = gs.index()` : une locale i32 par axe de la grille.
    fn assign_index(&mut self, target: &J) -> R<Vec<Stmt>> {
        let Some(rank) = self.rank else {
            unreachable!("gs.index() implique une grille indexée")
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
                        "gs.index() s'affecte à des variables : i, j = gs.index()",
                    )
                })?,
            _ => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.index() s'affecte à des variables : i, j = gs.index()",
                ));
            }
        };
        if names.len() != rank {
            return Err(self.e(
                ErrorKind::Type,
                format!("gs.index() renvoie {rank} coordonnée(s) (une par dimension de la sortie), {} variable(s) à gauche", names.len()),
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

    /// `buf = gs.local(gs.f32, N)` : tableau local de N éléments (N constant).
    fn declare_local_array(&mut self, target: &J, call: &J) -> R<()> {
        if target["_type"] != "Name" {
            return Err(self.e(
                ErrorKind::Syntax,
                "gs.local(...) s'affecte à une variable : buf = gs.local(gs.f32, 16)",
            ));
        }
        let name = target["id"].as_str().unwrap_or("").to_string();
        let args = call["args"].as_array().cloned().unwrap_or_default();
        if args.len() != 2 {
            return Err(self.e(
                ErrorKind::Syntax,
                "gs.local(type, taille) : ex. gs.local(gs.f32, 16)",
            ));
        }
        let ty = self.annotation_type(&args[0])?;
        if !ty.is_numeric() {
            return Err(self.e(
                ErrorKind::Type,
                "gs.local : type numérique attendu (gs.f32, gs.i32…)",
            ));
        }
        let len = match self.expr(&args[1])? {
            Val::Lit(Lit::Int(v)) if (1..=1 << 16).contains(&v) => v as u32,
            _ => {
                return Err(self.e(
                    ErrorKind::Type,
                    "gs.local : la taille doit être une constante entière entre 1 et 65536 (littéral ou constante globale)",
                ));
            }
        };
        if self.local_index.contains_key(&name) || self.params.iter().any(|p| p.name == name) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("« {name} » existe déjà : un tableau local se déclare une seule fois"),
            ));
        }
        let i = self.locals.len();
        self.locals.push(Local {
            name: name.clone(),
            ty,
            len: Some(len),
        });
        self.weak.push(false);
        self.local_index.insert(name, i);
        Ok(())
    }

    /// Indice entier i32 (les u32 et les littéraux sont convertis).
    fn index_expr(&mut self, node: &J) -> R<Expr> {
        let v = self.expr(node)?;
        match v.ty() {
            Scalar::I32 => self.to_ty(v, Scalar::I32, "indice"),
            Scalar::U32 => Ok(cast(self.materialize(v)?, Scalar::I32)),
            t => Err(self.e(
                ErrorKind::Type,
                format!("indice de type {t} : un indice doit être entier (int(...) si besoin)"),
            )),
        }
    }

    /// `x[i, k]` (tableau d'entrée) ou `buf[k]` (tableau local) → (tableau, indice aplati).
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
                "tranches x[a:b] non supportées dans un kernel : indexez case par case",
            ));
        }
        if let Some(&l) = self.local_index.get(&id) {
            if self.locals[l].len.is_some() {
                if idx_nodes.len() != 1 {
                    return Err(self.e(
                        ErrorKind::Syntax,
                        format!("le tableau local '{id}' a une dimension : {id}[k]"),
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
            if self.rank.is_none() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("indexation {id}[...] : dans un kernel élément par élément, `{id}` désigne déjà l'élément courant (pour indexer, utilisez gs.index() et des dimensions nommées)"),
                ));
            }
            let dims = self.params[p].dims.clone();
            if idx_nodes.len() != dims.len() {
                return Err(self.e(
                    ErrorKind::Type,
                    format!(
                        "'{id}' a {} dimension(s), {} indice(s) donné(s)",
                        dims.len(),
                        idx_nodes.len()
                    ),
                ));
            }
            // Indice aplati, ordre C : ((i0 * d1 + i1) * d2 + i2)…
            let mut flat: Option<Expr> = None;
            for (k, n) in idx_nodes.iter().enumerate() {
                let mut i = self.index_expr(n)?;
                if k == 0 && self.chunked.contains(&p) {
                    // Tableau découpé par lignes : seule la partie du bloc est sur le périphérique.
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
        Err(self.e(ErrorKind::Syntax, "indexation non supportée : seuls les tableaux d'entrée et les tableaux locaux (gs.local) s'indexent".to_string()))
    }

    /// `buf[k] = v` : écriture dans un tableau local (les entrées sont en lecture seule).
    fn store(&mut self, target: &J, v: Val) -> R<Stmt> {
        let (param, local, index) = self.subscript(target)?;
        match (param, local) {
            (_, Some(l)) => {
                let ty = self.locals[l].ty;
                let value = self.to_ty(v, ty, "écriture dans un tableau local")?;
                Ok(Stmt::LocalStore {
                    local: l,
                    index,
                    value,
                })
            }
            (Some(p), _) => Err(self.e(
                ErrorKind::Syntax,
                format!(
                    "'{}' est une entrée, en lecture seule : le résultat s'écrit avec return",
                    self.params[p].name
                ),
            )),
            _ => unreachable!(),
        }
    }

    /// `x.shape[k]` / `len(x)` : taille d'un axe (i32).
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
            return Err(self.e(ErrorKind::Syntax, format!("« {id} » n'est pas un tableau")));
        };
        if self.rank.is_none() {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("{id}.shape : disponible dans les kernels avec gs.index() et des dimensions nommées"),
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
                    "le paramètre '{name}' ne peut pas être modifié : utilisez une variable locale"
                ),
            ));
        }
        if self.constants.contains_key(name) || self.modules.contains_key(name) {
            return Err(self.e(
                ErrorKind::Syntax,
                format!("'{name}' masque une constante ou un module global"),
            ));
        }
        let idx = match self.local_index.get(name) {
            Some(&i) => {
                if self.locals[i].len.is_some() {
                    return Err(self.e(ErrorKind::Syntax, format!("'{name}' est un tableau local : écrivez dans ses cases ({name}[k] = ...)")));
                }
                if let Some(d) = declared {
                    if d != self.locals[i].ty {
                        return Err(self.e(
                            ErrorKind::Type,
                            format!("'{name}' redéclarée avec un autre type"),
                        ));
                    }
                }
                let cur = self.locals[i].ty;
                let vt = v.ty();
                // Python est dynamique : une locale entière qui reçoit un flottant devient
                // flottante ; une locale née d'un littéral prend le type de la valeur suivante.
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
                });
                self.weak
                    .push(declared.is_none() && v.is_lit() && !self.hints.contains_key(name));
                self.local_index.insert(name.to_string(), i);
                i
            }
        };
        let ty = self.locals[idx].ty;
        if self.restart {
            // Le type sera corrigé à la passe suivante : valeur provisoire.
            return Ok(Stmt::Assign {
                local: idx,
                value: ex(ty, ExprKind::Const(gpusync_ir::interp::zero(ty))),
            });
        }
        let value = self.to_ty(v, ty, &format!("affectation de '{name}'"))?;
        Ok(Stmt::Assign { local: idx, value })
    }

    fn for_loop(&mut self, s: &J) -> R<Stmt> {
        if s["target"]["_type"] != "Name" {
            return Err(self.e(
                ErrorKind::Syntax,
                "boucle for : une seule variable (for i in range(...))",
            ));
        }
        if s["orelse"].as_array().is_some_and(|a| !a.is_empty()) {
            return Err(self.e(ErrorKind::Syntax, "for ... else n'est pas supporté"));
        }
        let it = &s["iter"];
        let is_range =
            it["_type"] == "Call" && it["func"]["_type"] == "Name" && it["func"]["id"] == "range";
        if !is_range {
            return Err(self.e(
                ErrorKind::Syntax,
                "boucle for : seul `for i in range(...)` est supporté dans un kernel",
            ));
        }
        let args = it["args"].as_array().cloned().unwrap_or_default();
        if args.is_empty()
            || args.len() > 3
            || it["keywords"].as_array().is_some_and(|k| !k.is_empty())
        {
            return Err(self.e(
                ErrorKind::Syntax,
                "range() attend 1 à 3 arguments positionnels",
            ));
        }
        let mut vals = args.iter().map(|a| self.expr(a)).collect::<R<Vec<_>>>()?;
        for v in &vals {
            if !v.ty().is_int() {
                return Err(self.e(
                    ErrorKind::Type,
                    "range() : les bornes doivent être entières",
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
                        "variable de boucle '{name}' déjà utilisée avec le type {}",
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
                    "condition de type {} : écrivez une comparaison explicite (ex. x != 0)",
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

    fn name(&self, id: &str) -> R<Val> {
        if let Some(i) = self.params.iter().position(|p| p.name == id && !p.hidden) {
            if self.params[i].kind == ParamKind::Array && self.rank.is_some() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("'{id}' est un tableau : indexez-le ({id}[i, ...]) avec les coordonnées de gs.index()"),
                ));
            }
            return Ok(Val::E(ex(self.params[i].ty, ExprKind::Param(i))));
        }
        if let Some(&i) = self.local_index.get(id) {
            if self.locals[i].len.is_some() {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("'{id}' est un tableau local : indexez-le ({id}[k])"),
                ));
            }
            return Ok(Val::E(ex(self.locals[i].ty, ExprKind::Local(i))));
        }
        if let Some(l) = self.constants.get(id) {
            return Ok(Val::Lit(*l));
        }
        Err(self.e(
            ErrorKind::Syntax,
            format!("nom inconnu « {id} » : seuls les paramètres, les variables locales et les constantes globales int/float/bool sont utilisables"),
        ))
    }

    fn expr(&mut self, node: &J) -> R<Val> {
        self.set_line(node);
        match node["_type"].as_str().unwrap_or("") {
            "Constant" => match parse_lit(&node["value"]) {
                Some(l) => Ok(Val::Lit(l)),
                None => Err(self.e(
                    ErrorKind::Syntax,
                    "seules les constantes numériques et booléennes sont supportées",
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
                        _ => Err(self.e(ErrorKind::Type, "négation d'une valeur non numérique")),
                    },
                    "Not" => match a {
                        Val::Lit(Lit::Bool(b)) => Ok(Val::Lit(Lit::Bool(!b))),
                        Val::E(e) if e.ty == Scalar::Bool => Ok(Val::E(ex(Scalar::Bool, ExprKind::Unary(UnOp::Not, Box::new(e))))),
                        _ => Err(self.e(ErrorKind::Type, "`not` attend un booléen : écrivez une comparaison explicite (ex. x == 0)")),
                    },
                    _ => Err(self.e(ErrorKind::Syntax, "opérateur ~ non supporté")),
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
                            "`and` / `or` attendent des booléens (comparaisons)",
                        ));
                    }
                    let e = self.materialize(v)?;
                    acc = Some(match acc {
                        None => e,
                        Some(a) => ex(Scalar::Bool, ExprKind::Binary(op, Box::new(a), Box::new(e))),
                    });
                }
                Ok(Val::E(acc.ok_or_else(|| {
                    self.e(ErrorKind::Syntax, "expression booléenne vide")
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
                                format!("comparaison « {o} » non supportée (is, in…)"),
                            ));
                        }
                    };
                    let ty = if left.ty() == Scalar::Bool
                        && right.ty() == Scalar::Bool
                        && matches!(cmp, CmpOp::Eq | CmpOp::Ne)
                    {
                        Scalar::Bool
                    } else {
                        self.unify(&left, &right, "comparaison")?
                    };
                    let (a, b) = (
                        self.to_ty(left, ty, "comparaison")?,
                        self.to_ty(right.clone(), ty, "comparaison")?,
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
                    self.e(ErrorKind::Syntax, "comparaison vide")
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
                    self.to_ty(a, ty, "ternaire")?,
                    self.to_ty(b, ty, "ternaire")?,
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
                        format!("attribut « .{attr} » non supporté dans un kernel"),
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
                                            "x.shape[k] : k doit être un entier constant",
                                        ));
                                    }
                                }
                            }
                            _ => {
                                return Err(self.e(
                                    ErrorKind::Syntax,
                                    "x.shape[k] : k doit être un entier constant",
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
                "listes, tuples et dictionnaires ne sont pas supportés dans un kernel",
            )),
            "Lambda" => Err(self.e(ErrorKind::Syntax, "lambda non supportée dans un kernel")),
            other => Err(self.e(
                ErrorKind::Syntax,
                format!("expression Python non supportée dans un kernel : {other}"),
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
            o => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("opérateur « {o} » non supporté (opérations bit à bit, @…)"),
                ));
            }
        };
        if let (Val::Lit(x), Val::Lit(y)) = (&a, &b) {
            if let Some(l) = fold(bop, *x, *y) {
                return Ok(Val::Lit(l));
            }
        }
        let mut ty = self.unify(&a, &b, op_symbol(bop))?;
        if bop == BinOp::Div && ty.is_int() {
            ty = Scalar::F32; // `/` est une vraie division, comme en Python
        }
        if bop == BinOp::Pow && ty.is_int() {
            return Err(self.e(
                ErrorKind::Type,
                "** sur des entiers n'est pas supporté : utilisez des flottants (float(x) ** 2) ou une multiplication",
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

    fn call(&mut self, node: &J) -> R<Val> {
        if node["keywords"].as_array().is_some_and(|k| !k.is_empty()) {
            return Err(self.e(
                ErrorKind::Syntax,
                "arguments nommés non supportés dans les appels d'un kernel",
            ));
        }
        let func = &node["func"];
        let (module_owned, fname) = match func["_type"].as_str() {
            Some("Name") => (None, func["id"].as_str().unwrap_or("")),
            Some("Attribute") => (
                self.module_of(&func["value"]).map(str::to_string),
                func["attr"].as_str().unwrap_or(""),
            ),
            _ => return Err(self.e(ErrorKind::Syntax, "appel de fonction non supporté")),
        };
        let module = module_owned.as_deref();
        let args = node["args"].as_array().cloned().unwrap_or_default();
        // Appels dont les arguments ne sont pas des valeurs : traités avant d'évaluer les arguments.
        match (module, fname) {
            (None, "len") => {
                let a = args
                    .first()
                    .filter(|a| a["_type"] == "Name" && args.len() == 1);
                return match a {
                    Some(a) => self.shape_of(a["id"].as_str().unwrap_or(""), 0),
                    None => Err(self.e(ErrorKind::Syntax, "len(x) : x doit être un tableau")),
                };
            }
            (Some("gpusync"), "index") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "gs.index() s'utilise seul dans une affectation : i, j = gs.index()",
                ));
            }
            (Some("gpusync"), "local") => {
                return Err(self.e(ErrorKind::Syntax, "gs.local(...) s'utilise seul dans une affectation : buf = gs.local(gs.f32, 16)"));
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
                return Err(self.e(ErrorKind::Syntax, format!("{fname}() attend un argument")));
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
                        format!("fonction « {fname} » non disponible dans un kernel"),
                    ));
                }
            },
            (None, "range") => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    "range() n'est utilisable que dans `for i in range(...)`",
                ));
            }

            _ => {
                return Err(self.e(
                    ErrorKind::Syntax,
                    format!("appel de « {fname} » non supporté : seules les fonctions de gs et de math sont utilisables dans un kernel"),
                ));
            }
        };

        // min / max à plus de deux arguments : repliés deux à deux.
        if matches!(f, Intrinsic::Min | Intrinsic::Max) && vals.len() > 2 {
            let mut it = vals.into_iter();
            let mut acc = it.next().unwrap();
            for v in it {
                acc = self.intrinsic(f, vec![acc, v], fname)?;
            }
            return Ok(acc);
        }
        let v = self.intrinsic(f, vals, fname)?;
        // math.floor / math.ceil renvoient un entier en Python.
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
                    "{fname}() attend {} argument(s), {} reçu(s)",
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
            return Err(self.e(ErrorKind::Type, format!("{fname}() attend un nombre")));
        }
        if ty.is_int() && !f.accepts_int() {
            ty = Scalar::F32; // math.sqrt(4) fonctionne en Python
        }
        let args = vals
            .into_iter()
            .map(|v| self.to_ty(v, ty, fname))
            .collect::<R<Vec<_>>>()?;
        Ok(Val::E(ex(ty, ExprKind::Call(f, args))))
    }
}

/// Repli des opérations entre littéraux, avec la sémantique de Python.
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
    }
}

fn stmt_label(t: &str) -> String {
    match t {
        "FunctionDef" => "définition de fonction imbriquée".into(),
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

/// Le bloc se termine-t-il par un `return` sur tous les chemins ?
fn always_returns(block: &Block) -> bool {
    match block.last() {
        Some(Stmt::Return(_)) => true,
        Some(Stmt::If {
            then, otherwise, ..
        }) => always_returns(then) && always_returns(otherwise),
        _ => false,
    }
}
