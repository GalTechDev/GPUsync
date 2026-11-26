//! Typed IR of GPUsync (ARCHITECTURE.md §6).
//!
//! M1: **element-wise** kernels. Each array parameter stands for the current element;
//! the output has the shape of the input arrays. The IR is *structured* (a tree of statements), like
//! the target languages (OpenCL C, WGSL).
//!
//! ## Semantics (shared by the interpreter and the emitters)
//!
//! | Operation | Integers | Floats |
//! |---|---|---|
//! | `+ - *`, negation | modular arithmetic (overflow = wrap-around) | IEEE 754 |
//! | `/` | — (the frontend converts to float, as Python does) | IEEE 754 |
//! | `//` | rounded toward −∞ (Python); zero divisor → 0 | `floor(a / b)` |
//! | `%` | sign of the divisor (Python); zero divisor → 0 | `fmod` corrected to the divisor's sign (Python) |
//! | `**` | — (floats only in M1) | `pow` |
//! | float → integer conversion | truncation toward 0 | |
//!
//! Expressions have no side effects: `and`, `or` and the ternary operator may evaluate
//! both sides.

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

    /// Unsigned integer (u32, u64).
    pub fn is_unsigned(self) -> bool {
        matches!(self, Scalar::U32 | Scalar::U64)
    }

    /// 64-bit integer (requires the device's `int64` capability).
    pub fn is_int64(self) -> bool {
        matches!(self, Scalar::I64 | Scalar::U64)
    }

    pub fn is_numeric(self) -> bool {
        self.is_float() || self.is_int()
    }

    /// Size in bytes in an array (bool is not allowed in arrays).
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

/// Typed constant value.
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
    /// Array: current element (element-wise kernel) or read by indices (`x[i, k]`).
    Array,
    /// Scalar passed as is to every thread.
    Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub ty: Scalar,
    /// Array of an indexed kernel: parameters (hidden, i32) giving the size of each axis.
    pub dims: Vec<usize>,
    /// Parameter added by the frontend (size of a named dimension), supplied by the runtime.
    pub hidden: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: Scalar,
    /// `Some(n)`: local array of `n` elements (`gs.local(type, n)`), private to each thread.
    pub len: Option<u32>,
    /// Array shared by the threads of a group (`gs.shared(type, n)`, GPU shared memory).
    pub shared: bool,
}

/// Shape of the thread grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grid {
    /// One thread per element of the arrays (all of the same shape); arrays stand for the current element.
    Elementwise,
    /// One thread per output cell; `axes`: hidden parameters giving the size of each axis.
    ///
    /// Launch in blocks of rows (first axis of the output, ARCHITECTURE.md §9.7): thread
    /// `gs_i` of a launch computes row `gs_i / (cells per row) + row0`, writes the output at
    /// `gs_i + obase`, and the `chunked` arrays (read only at the current row) are indexed
    /// from row `inrow`. `row0`, `inrow` (i32) and `obase` (u32) are hidden parameters
    /// supplied by the engine, not by the caller.
    Indexed {
        axes: Vec<usize>,
        chunked: Vec<usize>,
        row0: usize,
        inrow: usize,
        obase: usize,
        /// Explicit mode (`grid=`, ARCHITECTURE.md §4.5): `gs.Out` arrays, written only by
        /// atomic operations, whole on the device and returned as results. The
        /// last one is `gs_err` (u32 × 2), where an out-of-bounds write is reported.
        writable: Vec<usize>,
    },
}

/// A kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Param>,
    /// Types of the outputs (one or more, of the same shape).
    pub outputs: Vec<Scalar>,
    pub locals: Vec<Local>,
    pub body: Block,
    pub grid: Grid,
    /// Threads per group (`@gs.kernel(group=n)`, indexed kernels): a group computes `n` consecutive
    /// output cells, shares the `gs.shared` arrays and synchronizes with
    /// `gs.barrier()`. `None`: size chosen by the backend, no shared memory.
    pub group: Option<u32>,
    /// Output cells per thread (`@gs.kernel(group=g, cells=c)`, 1 by default): a group owns
    /// a tile of g·c consecutive cells; the thread of rank r of the group writes cells
    /// `start + r + k·g` (k < c) with `gs.store(k, value)`. A single output.
    pub cells: u32,
}

impl Kernel {
    /// Bytes of shared memory per group.
    pub fn shared_bytes(&self) -> usize {
        self.locals
            .iter()
            .filter(|l| l.shared)
            .map(|l| l.len.unwrap_or(0) as usize * l.ty.size())
            .sum()
    }

    /// Parameters supplied by the execution engine (not by the caller): `row0`, `inrow`, `obase`.
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

    /// `gs.Out` arrays of the explicit mode (including `gs_err`, last).
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
    /// `for var in range(start, stop, step)`: bounds evaluated once, zero step → no iteration.
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
    /// `buf[index] = value` on a local array.
    LocalStore {
        local: usize,
        index: Expr,
        value: Expr,
    },
    /// Writes the current cell of each output and ends the thread.
    Return(Vec<Expr>),
    /// `gs.store(k, value)` (kernels with `cells=`): writes the thread's k-th cell to the output.
    Store {
        k: Expr,
        value: Expr,
    },
    /// `gs.barrier()`: waits for every thread of the group; their shared-memory writes
    /// are visible afterwards. Reached by every thread of the group (checked by the frontend).
    Barrier,
    /// `gs.matmul_u8(c, a, b, m, n, k)`: product of unsigned 8-bit integer matrices, collective
    /// (the whole group, barrier before and after), on `gs.shared` u32 arrays:
    /// `c[i·n + j] += Σ_t a[i, t]·b[j, t]` (32-bit sums, modulo 2³²), `a` stored by rows and `b` by
    /// columns, 4 bytes per word (k a multiple of 4). Tensor Cores with CUDA (Ampere and later).
    MatMul {
        c: usize,
        a: usize,
        b: usize,
        m: u32,
        n: u32,
        k: u32,
    },
    /// `gs.atomic_add(param, index, value)` (or min, max) on a `gs.Out` array. If `in_bounds`
    /// is false, nothing is written: `code` is reported in `gs_err[0]` (atomic maximum) and
    /// the index in `gs_err[1]`.
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
    /// Parameter: value of the scalar, or current element of the array.
    Param(usize),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Comparison: operands of the same type, `bool` result.
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    /// `cond ? a : b`
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Intrinsic, Vec<Expr>),
    /// Conversion vers `ty`.
    Cast(Box<Expr>),
    /// Read `param[index]` (flattened index, i32) of an input array.
    Load(usize, Box<Expr>),
    /// Read `local[index]` of a local array.
    LocalLoad(usize, Box<Expr>),
    /// Coordinate of the thread along the given grid axis (i32).
    GridIndex(usize),
    /// Position of the thread in its group, from 0 to `group − 1` (`gs.lane()`, i32).
    Lane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    /// Bitwise complement `~x` (integers).
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
    /// Bitwise operations on integers (two's complement for signed ones, as in Python).
    BitAnd,
    BitOr,
    BitXor,
    /// Shifts: the count (same type as the value) is read as unsigned; a count greater
    /// than or equal to the width gives 0 (or −1 for `>>` of a negative), as Python clamped to the width.
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

/// Intrinsic functions (float except `Abs`, `Min`, `Max`).
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
    /// High half of the full product of two integers of the same type (`gs.mulhi`).
    MulHi,
}

impl Intrinsic {
    /// Number of expected arguments.
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

    /// Accepts integers (otherwise: floats only).
    pub fn accepts_int(self) -> bool {
        matches!(
            self,
            Intrinsic::Abs | Intrinsic::Min | Intrinsic::Max | Intrinsic::MulHi
        )
    }
}

/// Capabilities required by a kernel (§6.2), compared with the device's before compiling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Requirements {
    /// The kernel handles f64 (arithmetic, comparisons, `//`, `%`, sqrt, floor, ceil, abs,
    /// min, max, conversions).
    pub f64: bool,
    /// It also calls a transcendental function in f64 (exp, log, sin, pow…): unavailable with
    /// Vulkan, where it would be computed at f32 precision without error, or even crash the driver.
    pub f64_math: bool,
    /// First operation that requires `f64_math`, for messages (e.g. "gs.exp").
    pub f64_math_op: Option<&'static str>,
    /// The kernel handles 64-bit integers (i64, u64).
    pub int64: bool,
}

impl Intrinsic {
    /// Transcendental function (unavailable in f64 on Vulkan).
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

/// Walks every expression (and sub-expression) of a block.
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
            Stmt::Store { k, value } => {
                expr(k, f);
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
            Stmt::Break | Stmt::Continue | Stmt::Barrier | Stmt::MatMul { .. } => {}
        }
    }
}

/// Python-style integer quotient (rounded toward −∞), zero divisor → 0.
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

/// Python-style integer remainder (sign of the divisor), zero divisor → 0.
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

/// Python-style 64-bit integer quotient (rounded toward −∞), zero divisor → 0.
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

/// Python-style 64-bit integer remainder (sign of the divisor), zero divisor → 0.
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

/// Python-style float remainder: `fmod` corrected to the divisor's sign.
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
        // Reference values: Python 3.
        assert_eq!(floordiv_i32(7, 2), 3);
        assert_eq!(floordiv_i32(-7, 2), -4);
        assert_eq!(floordiv_i32(7, -2), -4);
        assert_eq!(floordiv_i32(-7, -2), 3);
        assert_eq!(mod_i32(-7, 2), 1);
        assert_eq!(mod_i32(7, -2), -1);
        assert_eq!(mod_i32(-7, -2), -1);
        assert_eq!(floordiv_i32(5, 0), 0);
        assert_eq!(mod_i32(5, 0), 0);
        assert_eq!(floordiv_i32(i32::MIN, -1), i32::MIN); // wrap-around, no panic
        assert_eq!(mod_f64(-7.5, 2.0), 0.5);
        assert_eq!(mod_f64(7.5, -2.0), -0.5);
        assert_eq!(mod_f32(-1.0, 3.0), 2.0);
    }
}
