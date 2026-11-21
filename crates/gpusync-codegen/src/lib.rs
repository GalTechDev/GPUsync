//! Génération de code : IR → OpenCL C (≥ 1.2) et WGSL (ARCHITECTURE.md §7).
//!
//! Disposition des arguments (commune à l'exécution) :
//! - **OpenCL** : paramètres dans l'ordre déclaré (tableaux = buffers, scalaires = valeurs, dont les
//!   dimensions cachées des kernels avec index, en dernier), puis un buffer par sortie, puis `n` (uint).
//! - **WGSL** : un binding par tableau d'entrée (ordre déclaré), puis un par sortie, puis un buffer
//!   `array<u32>` contenant les scalaires non f64 (un mot chacun) suivis de `n`, puis — seulement si le
//!   kernel a des scalaires f64 — un buffer `array<f64>` avec ces scalaires.

mod opencl;
mod wgsl;

use std::collections::BTreeSet;

use gpusync_ir::*;

pub use opencl::{emit_c_variant, emit_cuda, emit_opencl};
pub use wgsl::{WGSL_WORKGROUP_SIZE, emit_wgsl, emit_wgsl_variant, has_f64_params};

/// Paramètres scalaires remplacés par une constante dans le code généré (spécialisation, §5.4) :
/// (indice du paramètre, valeur). Le paramètre reste dans la signature, inutilisé.
pub type Spec = [(usize, i64)];

pub(crate) fn spec_value(spec: &Spec, i: usize) -> Option<i64> {
    spec.iter().find(|(p, _)| *p == i).map(|(_, v)| *v)
}

/// Code généré, prêt à compiler.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSource {
    pub code: String,
    /// Nom du point d'entrée.
    pub entry: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodegenError {}

/// Identifiant ASCII sûr dans toutes les cibles (les noms Python peuvent contenir « é »…).
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

pub(crate) fn local_name(k: &Kernel, i: usize) -> String {
    format!("l{i}_{}", sanitize(&k.locals[i].name))
}

pub(crate) fn param_name(k: &Kernel, i: usize) -> String {
    format!("p{i}_{}", sanitize(&k.params[i].name))
}

/// Fonctions d'aide nécessaires au kernel (émises seulement si utilisées).
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Helper {
    #[default]
    FloorDivI32,
    FloorDivU32,
    ModI32,
    ModU32,
    ModF32,
    ModF64,
    PowF32,
    FloorDivI64,
    FloorDivU64,
    ModI64,
    ModU64,
}

pub(crate) fn helpers(k: &Kernel) -> BTreeSet<Helper> {
    let mut set = BTreeSet::new();
    visit_exprs(&k.body, &mut |e| {
        if let ExprKind::Binary(op, a, _) = &e.kind {
            match (op, a.ty) {
                (BinOp::FloorDiv, Scalar::I32) => {
                    set.insert(Helper::FloorDivI32);
                }
                (BinOp::FloorDiv, Scalar::U32) => {
                    set.insert(Helper::FloorDivU32);
                }
                (BinOp::Mod, Scalar::I32) => {
                    set.insert(Helper::ModI32);
                }
                (BinOp::Mod, Scalar::U32) => {
                    set.insert(Helper::ModU32);
                }
                (BinOp::Mod, Scalar::F32) => {
                    set.insert(Helper::ModF32);
                }
                (BinOp::Mod, Scalar::F64) => {
                    set.insert(Helper::ModF64);
                }
                (BinOp::FloorDiv, Scalar::I64) => {
                    set.insert(Helper::FloorDivI64);
                }
                (BinOp::FloorDiv, Scalar::U64) => {
                    set.insert(Helper::FloorDivU64);
                }
                (BinOp::Mod, Scalar::I64) => {
                    set.insert(Helper::ModI64);
                }
                (BinOp::Mod, Scalar::U64) => {
                    set.insert(Helper::ModU64);
                }
                (BinOp::Pow, Scalar::F32) => {
                    set.insert(Helper::PowF32);
                }
                _ => {}
            }
        }
        if let ExprKind::Call(Intrinsic::Pow, args) = &e.kind {
            if args[0].ty == Scalar::F32 {
                set.insert(Helper::PowF32);
            }
        }
    });
    set
}

/// Opérations atomiques du kernel : (opération, type de la sortie), sans doublon.
pub(crate) fn atomics(k: &Kernel) -> Vec<(u8, Scalar)> {
    fn walk(b: &Block, k: &Kernel, out: &mut Vec<(u8, Scalar)>) {
        for s in b {
            match s {
                Stmt::Atomic { op, param, .. } => {
                    let a = (*op as u8, k.params[*param].ty);
                    if !out.contains(&a) {
                        out.push(a);
                    }
                }
                Stmt::If {
                    then, otherwise, ..
                } => {
                    walk(then, k, out);
                    walk(otherwise, k, out);
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => walk(body, k, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(&k.body, k, &mut out);
    out
}

/// Indentation simple pour un code lisible (`kernel.source("opencl")`).
pub(crate) struct Out {
    pub buf: String,
    pub depth: usize,
}

impl Out {
    pub fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.depth {
            self.buf.push_str("    ");
        }
        self.buf.push_str(s.as_ref());
        self.buf.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn ret(e: Expr) -> Stmt {
        Stmt::Return(vec![e])
    }

    pub(crate) fn e(ty: Scalar, kind: ExprKind) -> Expr {
        Expr { ty, kind }
    }

    /// normalize(x: f32[:], mean: f32, std: f32) avec un peu de tout : boucle, //, %, **, ternaire.
    pub(crate) fn sample() -> Kernel {
        let x = || Box::new(e(Scalar::F32, ExprKind::Param(0)));
        let c = |v: f32| Box::new(e(Scalar::F32, ExprKind::Const(Const::F32(v))));
        let ci = |v: i32| e(Scalar::I32, ExprKind::Const(Const::I32(v)));
        Kernel {
            name: "normalize".into(),
            params: vec![
                Param {
                    name: "x".into(),
                    kind: ParamKind::Array,
                    ty: Scalar::F32,
                    dims: vec![],
                    hidden: false,
                },
                Param {
                    name: "moyenne".into(),
                    kind: ParamKind::Scalar,
                    ty: Scalar::F32,
                    dims: vec![],
                    hidden: false,
                },
                Param {
                    name: "écart".into(),
                    kind: ParamKind::Scalar,
                    ty: Scalar::I32,
                    dims: vec![],
                    hidden: false,
                },
            ],
            outputs: vec![Scalar::F32],
            grid: Grid::Elementwise,
            locals: vec![
                Local {
                    name: "acc".into(),
                    ty: Scalar::F32,
                    len: None,
                },
                Local {
                    name: "i".into(),
                    ty: Scalar::I32,
                    len: None,
                },
            ],
            body: vec![
                Stmt::Assign {
                    local: 0,
                    value: *c(0.0),
                },
                Stmt::For {
                    var: 1,
                    start: ci(0),
                    stop: e(Scalar::I32, ExprKind::Param(2)),
                    step: ci(1),
                    body: vec![Stmt::Assign {
                        local: 0,
                        value: e(
                            Scalar::F32,
                            ExprKind::Binary(
                                BinOp::Add,
                                Box::new(e(Scalar::F32, ExprKind::Local(0))),
                                Box::new(e(Scalar::F32, ExprKind::Binary(BinOp::Mod, x(), c(3.0)))),
                            ),
                        ),
                    }],
                },
                ret(e(
                    Scalar::F32,
                    ExprKind::Select(
                        Box::new(e(Scalar::Bool, ExprKind::Compare(CmpOp::Gt, x(), c(0.0)))),
                        Box::new(e(Scalar::F32, ExprKind::Binary(BinOp::Pow, x(), c(2.0)))),
                        Box::new(e(
                            Scalar::F32,
                            ExprKind::Binary(
                                BinOp::Sub,
                                Box::new(e(Scalar::F32, ExprKind::Local(0))),
                                Box::new(e(Scalar::F32, ExprKind::Param(1))),
                            ),
                        )),
                    ),
                )),
            ],
        }
    }

    #[test]
    fn helpers_detectes() {
        let h = helpers(&sample());
        assert!(h.contains(&Helper::ModF32) && h.contains(&Helper::PowF32));
        assert!(!h.contains(&Helper::FloorDivI32));
    }
}
