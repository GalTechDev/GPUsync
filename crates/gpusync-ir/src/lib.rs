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
    I64,
    U64,
}

impl Scalar {
    pub fn name(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::I64 => "i64",
            Scalar::U64 => "u64",
        }
    }

    pub fn parse(s: &str) -> Option<Scalar> {
        Some(match s {
            "bool" => Scalar::Bool,
            "i32" => Scalar::I32,
            "u32" => Scalar::U32,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            "i64" => Scalar::I64,
            "u64" => Scalar::U64,
            _ => return None,
        })
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(self, Scalar::I32 | Scalar::U32 | Scalar::I64 | Scalar::U64)
    }

    /// Entier non signé (u32, u64).
    pub fn is_unsigned(self) -> bool {
        matches!(self, Scalar::U32 | Scalar::U64)
    }

    /// Entier 64 bits (demande la capacité `int64` du périphérique).
    pub fn is_int64(self) -> bool {
        matches!(self, Scalar::I64 | Scalar::U64)
    }

    pub fn is_numeric(self) -> bool {
        self.is_float() || self.is_int()
    }

    /// Taille en octets dans un tableau (bool n'est pas autorisé dans les tableaux).
    pub fn size(self) -> usize {
        match self {
            Scalar::Bool => 1,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 | Scalar::I64 | Scalar::U64 => 8,
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
    I64(i64),
    U64(u64),
}

impl Const {
    pub fn ty(self) -> Scalar {
        match self {
            Const::Bool(_) => Scalar::Bool,
            Const::I32(_) => Scalar::I32,
            Const::U32(_) => Scalar::U32,
            Const::F32(_) => Scalar::F32,
            Const::F64(_) => Scalar::F64,
            Const::I64(_) => Scalar::I64,
            Const::U64(_) => Scalar::U64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Tableau : élément courant (kernel élément par élément) ou lu par indices (`x[i, k]`).
    Array,
    /// Scalaire passé tel quel à tous les threads.
    Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub ty: Scalar,
    /// Tableau d'un kernel avec index : paramètres (cachés, i32) donnant la taille de chaque axe.
    pub dims: Vec<usize>,
    /// Paramètre ajouté par le frontend (taille d'une dimension nommée), fourni par le runtime.
    pub hidden: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: Scalar,
    /// `Some(n)` : tableau local de `n` éléments (`gs.local(type, n)`), propre à chaque thread.
    pub len: Option<u32>,
    /// Tableau commun aux threads d'un groupe (`gs.shared(type, n)`, mémoire partagée du GPU).
    pub shared: bool,
}

/// Forme de la grille de threads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grid {
    /// Un thread par élément des tableaux (tous de même forme) ; les tableaux désignent l'élément courant.
    Elementwise,
    /// Un thread par case de la sortie ; `axes` : paramètres cachés donnant la taille de chaque axe.
    ///
    /// Lancement par blocs de lignes (premier axe de la sortie, ARCHITECTURE.md §9.7) : le thread
    /// `gs_i` d'un lancement calcule la ligne `gs_i / (cases par ligne) + row0`, écrit la sortie à
    /// `gs_i + obase`, et les tableaux `chunked` (lus seulement à la ligne courante) sont indexés à
    /// partir de la ligne `inrow`. `row0`, `inrow` (i32) et `obase` (u32) sont des paramètres cachés
    /// fournis par le moteur, pas par l'appelant.
    Indexed {
        axes: Vec<usize>,
        chunked: Vec<usize>,
        row0: usize,
        inrow: usize,
        obase: usize,
        /// Mode explicite (`grid=`, ARCHITECTURE.md §4.5) : tableaux `gs.Out`, écrits seulement par
        /// des opérations atomiques, entiers sur le périphérique et rendus comme résultats. Le
        /// dernier est `gs_err` (u32 × 2), où une écriture hors limites est signalée.
        writable: Vec<usize>,
    },
}

/// Un kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Param>,
    /// Types des sorties (une ou plusieurs, de même forme).
    pub outputs: Vec<Scalar>,
    pub locals: Vec<Local>,
    pub body: Block,
    pub grid: Grid,
    /// Threads par groupe (`@gs.kernel(group=n)`, kernels avec index) : un groupe calcule `n` cases
    /// consécutives de la sortie, partage les tableaux `gs.shared` et se synchronise par
    /// `gs.barrier()`. `None` : taille choisie par le backend, pas de mémoire partagée.
    pub group: Option<u32>,
}

impl Kernel {
    /// Octets de mémoire partagée par groupe.
    pub fn shared_bytes(&self) -> usize {
        self.locals
            .iter()
            .filter(|l| l.shared)
            .map(|l| l.len.unwrap_or(0) as usize * l.ty.size())
            .sum()
    }

    /// Paramètres fournis par le moteur d'exécution (et non par l'appelant) : `row0`, `inrow`, `obase`.
    pub fn engine_params(&self) -> [Option<usize>; 3] {
        match &self.grid {
            Grid::Indexed {
                row0, inrow, obase, ..
            } => [Some(*row0), Some(*inrow), Some(*obase)],
            Grid::Elementwise => [None; 3],
        }
    }

    pub fn is_engine_param(&self, i: usize) -> bool {
        self.engine_params().contains(&Some(i))
    }

    /// Tableaux `gs.Out` du mode explicite (dont `gs_err`, en dernier).
    pub fn writable(&self) -> &[usize] {
        match &self.grid {
            Grid::Indexed { writable, .. } => writable,
            Grid::Elementwise => &[],
        }
    }
}

pub type Block = Vec<Stmt>;

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        local: usize,
        value: Expr,
    },
    If {
        cond: Expr,
        then: Block,
        otherwise: Block,
    },
    /// `for var in range(start, stop, step)` : bornes évaluées une fois, pas nul → aucune itération.
    For {
        var: usize,
        start: Expr,
        stop: Expr,
        step: Expr,
        body: Block,
    },
    While {
        cond: Expr,
        body: Block,
    },
    Break,
    Continue,
    /// `buf[index] = value` sur un tableau local.
    LocalStore {
        local: usize,
        index: Expr,
        value: Expr,
    },
    /// Écrit la case courante de chaque sortie et termine le thread.
    Return(Vec<Expr>),
    /// `gs.barrier()` : attend tous les threads du groupe ; leurs écritures en mémoire partagée
    /// sont ensuite visibles. Atteinte par tous les threads du groupe (vérifié par le frontend).
    Barrier,
    /// `gs.atomic_add(param, index, value)` (ou min, max) sur un tableau `gs.Out`. Si `in_bounds`
    /// est faux, rien n'est écrit : `code` est signalé dans `gs_err[0]` (maximum atomique) et
    /// l'indice dans `gs_err[1]`.
    Atomic {
        op: AtomicOp,
        param: usize,
        index: Expr,
        in_bounds: Expr,
        value: Expr,
        err: usize,
        code: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Min,
    Max,
}

impl AtomicOp {
    pub fn name(self) -> &'static str {
        match self {
            AtomicOp::Add => "add",
            AtomicOp::Min => "min",
            AtomicOp::Max => "max",
        }
    }
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
    /// Lecture `param[index]` (indice aplati, i32) d'un tableau d'entrée.
    Load(usize, Box<Expr>),
    /// Lecture `local[index]` d'un tableau local.
    LocalLoad(usize, Box<Expr>),
    /// Coordonnée du thread sur l'axe donné de la grille (i32).
    GridIndex(usize),
    /// Position du thread dans son groupe, de 0 à `group − 1` (`gs.lane()`, i32).
    Lane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    /// Complément bit à bit `~x` (entiers).
    BitNot,
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
    /// Opérations bit à bit sur des entiers (complément à deux pour les signés, comme en Python).
    BitAnd,
    BitOr,
    BitXor,
    /// Décalages : le compte (même type que la valeur) est lu comme non signé ; un compte supérieur
    /// ou égal à la largeur donne 0 (ou −1 pour `>>` d'un négatif), comme en Python borné à la largeur.
    Shl,
    Shr,
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
    /// Partie haute du produit complet de deux entiers de même type (`gs.mulhi`).
    MulHi,
}

impl Intrinsic {
    /// Nombre d'arguments attendus.
    pub fn arity(self) -> usize {
        match self {
            Intrinsic::Atan2
            | Intrinsic::Pow
            | Intrinsic::Min
            | Intrinsic::Max
            | Intrinsic::MulHi => 2,
            _ => 1,
        }
    }

    /// Accepte des entiers (sinon : flottants uniquement).
    pub fn accepts_int(self) -> bool {
        matches!(
            self,
            Intrinsic::Abs | Intrinsic::Min | Intrinsic::Max | Intrinsic::MulHi
        )
    }
}

/// Capacités exigées par un kernel (§6.2), comparées à celles du périphérique avant compilation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Requirements {
    /// Le kernel manipule des f64 (arithmétique, comparaisons, `//`, `%`, sqrt, floor, ceil, abs,
    /// min, max, conversions).
    pub f64: bool,
    /// Il appelle en plus une fonction transcendante en f64 (exp, log, sin, pow…) : indisponible en
    /// Vulkan, où elle serait calculée en précision f32 sans erreur, voire ferait planter le pilote.
    pub f64_math: bool,
    /// Première opération qui exige `f64_math`, pour les messages (ex. « gs.exp »).
    pub f64_math_op: Option<&'static str>,
    /// Le kernel manipule des entiers 64 bits (i64, u64).
    pub int64: bool,
}

impl Intrinsic {
    /// Fonction transcendante (indisponible en f64 sur Vulkan).
    pub fn is_transcendental(self) -> bool {
        !matches!(
            self,
            Intrinsic::Sqrt
                | Intrinsic::Floor
                | Intrinsic::Ceil
                | Intrinsic::Abs
                | Intrinsic::Min
                | Intrinsic::Max
                | Intrinsic::MulHi
        )
    }

    pub fn python_name(self) -> &'static str {
        match self {
            Intrinsic::Sqrt => "gs.sqrt",
            Intrinsic::Exp => "gs.exp",
            Intrinsic::Exp2 => "gs.exp2",
            Intrinsic::Log => "gs.log",
            Intrinsic::Log2 => "gs.log2",
            Intrinsic::Sin => "gs.sin",
            Intrinsic::Cos => "gs.cos",
            Intrinsic::Tan => "gs.tan",
            Intrinsic::Tanh => "gs.tanh",
            Intrinsic::Atan2 => "gs.atan2",
            Intrinsic::Floor => "gs.floor",
            Intrinsic::Ceil => "gs.ceil",
            Intrinsic::Pow => "gs.pow",
            Intrinsic::Abs => "abs",
            Intrinsic::Min => "min",
            Intrinsic::Max => "max",
            Intrinsic::MulHi => "gs.mulhi",
        }
    }
}

impl Kernel {
    pub fn requirements(&self) -> Requirements {
        let mut f64 = self.outputs.contains(&Scalar::F64)
            || self.params.iter().any(|p| p.ty == Scalar::F64)
            || self.locals.iter().any(|l| l.ty == Scalar::F64);
        let mut f64_math_op: Option<&'static str> = None;
        let mut int64 = self.outputs.iter().any(|t| t.is_int64())
            || self.params.iter().any(|p| p.ty.is_int64())
            || self.locals.iter().any(|l| l.ty.is_int64());
        visit_exprs(&self.body, &mut |e| {
            f64 |= e.ty == Scalar::F64;
            int64 |= e.ty.is_int64();
            let op = match &e.kind {
                ExprKind::Call(f, args)
                    if args.first().is_some_and(|a| a.ty == Scalar::F64)
                        && f.is_transcendental() =>
                {
                    Some(f.python_name())
                }
                ExprKind::Binary(BinOp::Pow, a, _) if a.ty == Scalar::F64 => Some("**"),
                _ => None,
            };
            if f64_math_op.is_none() {
                f64_math_op = op;
            }
        });
        Requirements {
            f64,
            f64_math: f64_math_op.is_some(),
            f64_math_op,
            int64,
        }
    }

    pub fn array_params(&self) -> impl Iterator<Item = (usize, &Param)> {
        self.params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == ParamKind::Array)
    }
}

/// Parcourt toutes les expressions (et sous-expressions) d'un bloc.
pub fn visit_exprs(block: &Block, f: &mut dyn FnMut(&Expr)) {
    fn expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
        f(e);
        match &e.kind {
            ExprKind::Const(_)
            | ExprKind::Local(_)
            | ExprKind::Param(_)
            | ExprKind::GridIndex(_)
            | ExprKind::Lane => {}
            ExprKind::Unary(_, a)
            | ExprKind::Cast(a)
            | ExprKind::Load(_, a)
            | ExprKind::LocalLoad(_, a) => expr(a, f),
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
            Stmt::Assign { value, .. } => expr(value, f),
            Stmt::Return(values) => values.iter().for_each(|v| expr(v, f)),
            Stmt::LocalStore { index, value, .. } => {
                expr(index, f);
                expr(value, f);
            }
            Stmt::Atomic {
                index,
                in_bounds,
                value,
                ..
            } => {
                expr(index, f);
                expr(in_bounds, f);
                expr(value, f);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                expr(cond, f);
                visit_exprs(then, f);
                visit_exprs(otherwise, f);
            }
            Stmt::For {
                start,
                stop,
                step,
                body,
                ..
            } => {
                expr(start, f);
                expr(stop, f);
                expr(step, f);
                visit_exprs(body, f);
            }
            Stmt::While { cond, body } => {
                expr(cond, f);
                visit_exprs(body, f);
            }
            Stmt::Break | Stmt::Continue | Stmt::Barrier => {}
        }
    }
}

/// Quotient entier « à la Python » (arrondi vers −∞), diviseur nul → 0.
pub fn floordiv_i32(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let q = a.wrapping_div(b);
    if a.wrapping_rem(b) != 0 && ((a < 0) != (b < 0)) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// Reste entier « à la Python » (signe du diviseur), diviseur nul → 0.
pub fn mod_i32(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let r = a.wrapping_rem(b);
    if r != 0 && ((r < 0) != (b < 0)) {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// Quotient entier 64 bits « à la Python » (arrondi vers −∞), diviseur nul → 0.
pub fn floordiv_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let q = a.wrapping_div(b);
    if a.wrapping_rem(b) != 0 && ((a < 0) != (b < 0)) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// Reste entier 64 bits « à la Python » (signe du diviseur), diviseur nul → 0.
pub fn mod_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let r = a.wrapping_rem(b);
    if r != 0 && ((r < 0) != (b < 0)) {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// Reste flottant « à la Python » : `fmod` corrigé du signe du diviseur.
pub fn mod_f64(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
        r + b
    } else {
        r
    }
}

pub fn mod_f32(a: f32, b: f32) -> f32 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
        r + b
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_and_modulo_like_python() {
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
