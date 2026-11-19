//! IR typée de GPUsync (ARCHITECTURE.md §6).
//!
//! M1 : kernels **élément par élément**. Chaque tableau en paramètre désigne l'élément courant ;
//! la sortie a la forme des tableaux d'entrée. L'IR est *structurée* (arbre d'instructions), comme
//! les langages cibles (OpenCL C, WGSL).
//!
//! ## Sémantique (commune à l'interpréteur et aux émetteurs)
//!
//! | Opération | Entiers | Flottants |
//! |---|---|---|
//! | `+ - *`, négation | arithmétique modulaire (débordement = enroulement) | IEEE 754 |
//! | `/` | — (le frontend convertit en flottant, comme Python) | IEEE 754 |
//! | `//` | arrondi vers −∞ (Python) ; diviseur nul → 0 | `floor(a / b)` |
//! | `%` | signe du diviseur (Python) ; diviseur nul → 0 | `fmod` corrigé du signe du diviseur (Python) |
//! | `**` | — (flottants uniquement en M1) | `pow` |
//! | conversion flottant → entier | troncature vers 0 | |
//!
//! Les expressions n'ont pas d'effet de bord : `and`, `or` et l'opérateur ternaire peuvent évaluer
//! leurs deux côtés.

pub mod interp;

use std::fmt;

/// Types scalaires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    Bool,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    pub fn name(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }

    pub fn parse(s: &str) -> Option<Scalar> {
        Some(match s {
            "bool" => Scalar::Bool,
            "i32" => Scalar::I32,
            "u32" => Scalar::U32,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            _ => return None,
        })
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(self, Scalar::I32 | Scalar::U32)
    }

    pub fn is_numeric(self) -> bool {
        self.is_float() || self.is_int()
    }

    /// Taille en octets dans un tableau (bool n'est pas autorisé dans les tableaux).
    pub fn size(self) -> usize {
        match self {
            Scalar::Bool => 1,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Valeur constante typée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
}

impl Const {
    pub fn ty(self) -> Scalar {
        match self {
            Const::Bool(_) => Scalar::Bool,
            Const::I32(_) => Scalar::I32,
            Const::U32(_) => Scalar::U32,
            Const::F32(_) => Scalar::F32,
            Const::F64(_) => Scalar::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Tableau : dans le corps, désigne l'élément courant.
    Array,
    /// Scalaire passé tel quel à tous les threads.
    Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub ty: Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: Scalar,
}

/// Un kernel élément par élément.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Param>,
    pub output: Scalar,
    pub locals: Vec<Local>,
    pub body: Block,
}

pub type Block = Vec<Stmt>;

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign { local: usize, value: Expr },
    If { cond: Expr, then: Block, otherwise: Block },
    /// `for var in range(start, stop, step)` : bornes évaluées une fois, pas nul → aucune itération.
    For { var: usize, start: Expr, stop: Expr, step: Expr, body: Block },
    While { cond: Expr, body: Block },
    Break,
    Continue,
    /// Écrit la valeur de l'élément courant de la sortie et termine le thread.
    Return(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub ty: Scalar,
    pub kind: ExprKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Const(Const),
    Local(usize),
    /// Paramètre : valeur du scalaire, ou élément courant du tableau.
    Param(usize),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Comparaison : opérandes de même type, résultat `bool`.
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    /// `cond ? a : b`
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Intrinsic, Vec<Expr>),
    /// Conversion vers `ty`.
    Cast(Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Fonctions intrinsèques (flottantes sauf `Abs`, `Min`, `Max`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intrinsic {
    Sqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Tan,
    Tanh,
    Atan2,
    Floor,
    Ceil,
    Pow,
    Abs,
    Min,
    Max,
}

impl Intrinsic {
    /// Nombre d'arguments attendus.
    pub fn arity(self) -> usize {
        match self {
            Intrinsic::Atan2 | Intrinsic::Pow | Intrinsic::Min | Intrinsic::Max => 2,
            _ => 1,
        }
    }

    /// Accepte des entiers (sinon : flottants uniquement).
    pub fn accepts_int(self) -> bool {
        matches!(self, Intrinsic::Abs | Intrinsic::Min | Intrinsic::Max)
    }
}

/// Capacités exigées par un kernel (§6.2), comparées à celles du périphérique avant compilation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Requirements {
    pub f64: bool,
}

impl Kernel {
    pub fn requirements(&self) -> Requirements {
        let mut f64 = self.output == Scalar::F64
            || self.params.iter().any(|p| p.ty == Scalar::F64)
            || self.locals.iter().any(|l| l.ty == Scalar::F64);
        visit_exprs(&self.body, &mut |e| f64 |= e.ty == Scalar::F64);
        Requirements { f64 }
    }

    pub fn array_params(&self) -> impl Iterator<Item = (usize, &Param)> {
        self.params.iter().enumerate().filter(|(_, p)| p.kind == ParamKind::Array)
    }
}

/// Parcourt toutes les expressions (et sous-expressions) d'un bloc.
pub fn visit_exprs(block: &Block, f: &mut dyn FnMut(&Expr)) {
    fn expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
        f(e);
        match &e.kind {
            ExprKind::Const(_) | ExprKind::Local(_) | ExprKind::Param(_) => {}
            ExprKind::Unary(_, a) | ExprKind::Cast(a) => expr(a, f),
            ExprKind::Binary(_, a, b) | ExprKind::Compare(_, a, b) => {
                expr(a, f);
                expr(b, f);
            }
            ExprKind::Select(c, a, b) => {
                expr(c, f);
                expr(a, f);
                expr(b, f);
            }
            ExprKind::Call(_, args) => args.iter().for_each(|a| expr(a, f)),
        }
    }
    for s in block {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return(value) => expr(value, f),
            Stmt::If { cond, then, otherwise } => {
                expr(cond, f);
                visit_exprs(then, f);
                visit_exprs(otherwise, f);
            }
            Stmt::For { start, stop, step, body, .. } => {
                expr(start, f);
                expr(stop, f);
                expr(step, f);
                visit_exprs(body, f);
            }
            Stmt::While { cond, body } => {
                expr(cond, f);
                visit_exprs(body, f);
            }
            Stmt::Break | Stmt::Continue => {}
        }
    }
}

/// Quotient entier « à la Python » (arrondi vers −∞), diviseur nul → 0.
pub fn floordiv_i32(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let q = a.wrapping_div(b);
    if a.wrapping_rem(b) != 0 && ((a < 0) != (b < 0)) { q.wrapping_sub(1) } else { q }
}

/// Reste entier « à la Python » (signe du diviseur), diviseur nul → 0.
pub fn mod_i32(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let r = a.wrapping_rem(b);
    if r != 0 && ((r < 0) != (b < 0)) { r.wrapping_add(b) } else { r }
}

/// Reste flottant « à la Python » : `fmod` corrigé du signe du diviseur.
pub fn mod_f64(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) { r + b } else { r }
}

pub fn mod_f32(a: f32, b: f32) -> f32 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) { r + b } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_et_modulo_comme_python() {
        // Valeurs de référence : Python 3.
        assert_eq!(floordiv_i32(7, 2), 3);
        assert_eq!(floordiv_i32(-7, 2), -4);
        assert_eq!(floordiv_i32(7, -2), -4);
        assert_eq!(floordiv_i32(-7, -2), 3);
        assert_eq!(mod_i32(-7, 2), 1);
        assert_eq!(mod_i32(7, -2), -1);
        assert_eq!(mod_i32(-7, -2), -1);
        assert_eq!(floordiv_i32(5, 0), 0);
        assert_eq!(mod_i32(5, 0), 0);
        assert_eq!(floordiv_i32(i32::MIN, -1), i32::MIN); // enroulement, pas de panique
        assert_eq!(mod_f64(-7.5, 2.0), 0.5);
        assert_eq!(mod_f64(7.5, -2.0), -0.5);
        assert_eq!(mod_f32(-1.0, 3.0), 2.0);
    }
}
