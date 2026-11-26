//! IR → WGSL (wgpu: Metal, Vulkan, DX12). No f64 in WGSL: rejected before reaching this point.

use gpusync_ir::*;

use crate::opencl::{float_repr, intrinsic_name};
use crate::{CodegenError, Helper, KernelSource, Out, helpers, local_name, param_name};

/// Workgroup size. The grid is 2-D (x, y) to go beyond 65,535 groups.
pub const WGSL_WORKGROUP_SIZE: u32 = 256;

fn w_type(t: Scalar) -> &'static str {
    match t {
        Scalar::Bool => "bool",
        Scalar::I32 => "i32",
        Scalar::U32 => "u32",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
        Scalar::I64 => "i64",
        Scalar::U64 => "u64",
    }
}

fn lit(c: Const) -> String {
    match c {
        Const::Bool(b) => (if b { "true" } else { "false" }).into(),
        Const::I32(i32::MIN) => "bitcast<i32>(0x80000000u)".into(),
        Const::I32(v) if v < 0 => format!("({v}i)"),
        Const::I32(v) => format!("{v}i"),
        Const::U32(v) => format!("{v}u"),
        Const::F32(v) if v.is_nan() => "bitcast<f32>(0x7fc00000u)".into(),
        Const::F32(v) if v.is_infinite() => (if v > 0.0 {
            "bitcast<f32>(0x7f800000u)"
        } else {
            "bitcast<f32>(0xff800000u)"
        })
        .into(),
        Const::F32(v) if v < 0.0 => format!("({}f)", float_repr(format!("{v:?}"))),
        Const::F32(v) => format!("{}f", float_repr(format!("{v:?}"))),
        // No bitcast to f64 in naga: infinity and NaN go through an f32 (exact conversion).
        Const::F64(v) if v.is_nan() => "f64(bitcast<f32>(0x7fc00000u))".into(),
        Const::F64(v) if v.is_infinite() => (if v > 0.0 {
            "f64(bitcast<f32>(0x7f800000u))"
        } else {
            "f64(bitcast<f32>(0xff800000u))"
        })
        .into(),
        Const::F64(v) if v < 0.0 => format!("({}lf)", float_repr(format!("{v:?}"))),
        Const::F64(v) => format!("{}lf", float_repr(format!("{v:?}"))),
        Const::I64(i64::MIN) => "(-9223372036854775807li - 1li)".into(),
        Const::I64(v) if v < 0 => format!("({v}li)"),
        Const::I64(v) => format!("{v}li"),
        Const::U64(v) => format!("{v}lu"),
    }
}

struct Gen<'a> {
    k: &'a Kernel,
    out: Out,
    tmp: usize,
    /// Parameters replaced by a constant (specialization).
    spec: Vec<(usize, i64)>,
    /// Locals with a constant integer value (assigned once, at top level, from a
    /// constant expression: `t = len(ref)` once the size is specialized).
    const_locals: Vec<(usize, i64)>,
}

impl Gen<'_> {
    /// Write index of the outputs: offset by `obase` for an indexed kernel launched in blocks.
    fn out_index(&self) -> String {
        match &self.k.grid {
            Grid::Indexed { obase, .. } => format!("gs_i + {}", param_name(self.k, *obase)),
            Grid::Elementwise => "gs_i".into(),
        }
    }

    fn expr(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Const(c) => lit(*c),
            ExprKind::Local(i) => local_name(self.k, *i),
            ExprKind::Param(i) => match self.k.params[*i].kind {
                ParamKind::Array => format!("{}[gs_i]", param_name(self.k, *i)),
                ParamKind::Scalar => match crate::spec_value(&self.spec, *i) {
                    Some(v) => format!("{v}i"),
                    None => param_name(self.k, *i),
                },
            },
            ExprKind::Unary(UnOp::Neg, a) => format!("(-{})", self.expr(a)),
            ExprKind::Unary(UnOp::Not, a) => format!("(!{})", self.expr(a)),
            ExprKind::Unary(UnOp::BitNot, a) => format!("(~{})", self.expr(a)),
            ExprKind::Binary(op @ (BinOp::Shl | BinOp::Shr), a, b) => {
                let (x, y) = (self.expr(a), self.expr(b));
                let name = if *op == BinOp::Shl { "shl" } else { "shr" };
                match crate::const_shift(b) {
                    // Left shift of a signed value: done unsigned (avoids the overflow error of
                    // WGSL constant evaluation), the conversion keeping the bits.
                    Some(c) if *op == BinOp::Shl && matches!(a.ty, Scalar::I32 | Scalar::I64) => {
                        let (t, u) = (
                            w_type(a.ty),
                            if a.ty == Scalar::I32 { "u32" } else { "u64" },
                        );
                        format!("{t}({u}({x}) << {c}u)")
                    }
                    Some(c) => {
                        format!("({x} {} {c}u)", if *op == BinOp::Shl { "<<" } else { ">>" })
                    }
                    None => format!("gs_{name}_{}({x}, {y})", a.ty.name()),
                }
            }
            ExprKind::Binary(op, a, b) => {
                let (x, y) = (self.expr(a), self.expr(b));
                match (op, a.ty) {
                    (BinOp::FloorDiv, Scalar::I32) => format!("gs_floordiv_i32({x}, {y})"),
                    (BinOp::FloorDiv, Scalar::U32) => format!("gs_floordiv_u32({x}, {y})"),
                    (BinOp::FloorDiv, Scalar::I64) => format!("gs_floordiv_i64({x}, {y})"),
                    (BinOp::FloorDiv, Scalar::U64) => format!("gs_floordiv_u64({x}, {y})"),
                    (BinOp::Mod, Scalar::I64) => format!("gs_mod_i64({x}, {y})"),
                    (BinOp::Mod, Scalar::U64) => format!("gs_mod_u64({x}, {y})"),
                    (BinOp::FloorDiv, _) => format!("floor({x} / {y})"),
                    (BinOp::Mod, Scalar::I32) => format!("gs_mod_i32({x}, {y})"),
                    (BinOp::Mod, Scalar::U32) => format!("gs_mod_u32({x}, {y})"),
                    (BinOp::Mod, Scalar::F64) => format!("gs_mod_f64({x}, {y})"),
                    (BinOp::Mod, _) => format!("gs_mod_f32({x}, {y})"),
                    (BinOp::Pow, _) => format!("gs_pow_f32({x}, {y})"),
                    (BinOp::Add, _) => format!("({x} + {y})"),
                    (BinOp::Sub, _) => format!("({x} - {y})"),
                    (BinOp::Mul, _) => format!("({x} * {y})"),
                    (BinOp::Div, _) => format!("({x} / {y})"),
                    (BinOp::And, _) => format!("({x} && {y})"),
                    (BinOp::Or, _) => format!("({x} || {y})"),
                    (BinOp::BitAnd, _) => format!("({x} & {y})"),
                    (BinOp::BitOr, _) => format!("({x} | {y})"),
                    (BinOp::BitXor, _) => format!("({x} ^ {y})"),
                    (BinOp::Shl | BinOp::Shr, _) => unreachable!(),
                }
            }
            ExprKind::Compare(op, a, b) => {
                let s = match op {
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                };
                format!("({} {s} {})", self.expr(a), self.expr(b))
            }
            // select(false, true, condition)
            ExprKind::Select(c, a, b) => format!(
                "select({}, {}, {})",
                self.expr(b),
                self.expr(a),
                self.expr(c)
            ),
            ExprKind::Call(f, args) => {
                let a: Vec<String> = args.iter().map(|x| self.expr(x)).collect();
                match f {
                    Intrinsic::Pow => format!("gs_pow_f32({}, {})", a[0], a[1]),
                    Intrinsic::MulHi => {
                        format!("gs_mulhi_{}({}, {})", args[0].ty.name(), a[0], a[1])
                    }
                    Intrinsic::Abs if args[0].ty.is_unsigned() => a[0].clone(),
                    _ => format!("{}({})", intrinsic_name(*f), a.join(", ")),
                }
            }
            ExprKind::Load(p, i) => format!("{}[{}]", param_name(self.k, *p), self.expr(i)),
            ExprKind::LocalLoad(l, i) => format!("{}[{}]", local_name(self.k, *l), self.expr(i)),
            ExprKind::GridIndex(a) => format!("gs_g{a}"),
            ExprKind::Lane => "i32(gs_lid)".into(),
            ExprKind::Cast(a) => format!("{}({})", w_type(e.ty), self.expr(a)),
        }
    }

    fn block(&mut self, b: &Block) {
        for s in b {
            self.stmt(s);
        }
    }

    fn indent(&mut self, b: &Block) {
        self.out.depth += 1;
        self.block(b);
        self.out.depth -= 1;
    }

    /// Locals assigned once, at the kernel's top level, from a constant expression.
    fn find_const_locals(&self) -> Vec<(usize, i64)> {
        fn count(b: &Block, c: &mut Vec<usize>) {
            for s in b {
                match s {
                    Stmt::Assign { local, .. } => c[*local] += 1,
                    Stmt::For { var, body, .. } => {
                        c[*var] += 2; // loop variable: never constant
                        count(body, c);
                    }
                    Stmt::If {
                        then, otherwise, ..
                    } => {
                        count(then, c);
                        count(otherwise, c);
                    }
                    Stmt::While { body, .. } => count(body, c),
                    _ => {}
                }
            }
        }
        let mut assigns = vec![0usize; self.k.locals.len()];
        count(&self.k.body, &mut assigns);
        let mut out: Vec<(usize, i64)> = Vec::new();
        for s in &self.k.body {
            if let Stmt::Assign { local, value } = s {
                if assigns[*local] == 1 && self.k.locals[*local].len.is_none() {
                    let g = Gen {
                        k: self.k,
                        out: Out {
                            buf: String::new(),
                            depth: 0,
                        },
                        tmp: 0,
                        spec: self.spec.clone(),
                        const_locals: out.clone(),
                    };
                    if let Some(v) = g.const_int(value) {
                        out.push((*local, v));
                    }
                }
            }
        }
        out
    }

    /// Constant integer value of an expression (literal, specialized size, + − ×).
    fn const_int(&self, e: &Expr) -> Option<i64> {
        match &e.kind {
            ExprKind::Const(Const::I32(v)) => Some(*v as i64),
            ExprKind::Const(Const::U32(v)) => Some(*v as i64),
            ExprKind::Param(i) => crate::spec_value(&self.spec, *i),
            ExprKind::Local(l) => crate::spec_value(&self.const_locals, *l),
            ExprKind::Binary(op, a, b) => {
                let (x, y) = (self.const_int(a)?, self.const_int(b)?);
                match op {
                    BinOp::Add => Some(x + y),
                    BinOp::Sub => Some(x - y),
                    BinOp::Mul => Some(x * y),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Successive values of the variable if the loop can be unrolled: constant bounds, at most
    /// 64 iterations, innermost loop, no `break`, `continue` or `return`.
    fn unrollable(&self, start: &Expr, stop: &Expr, step: &Expr, body: &Block) -> Option<Vec<i64>> {
        fn plain(b: &Block) -> bool {
            b.iter().all(|s| match s {
                Stmt::If {
                    then, otherwise, ..
                } => plain(then) && plain(otherwise),
                Stmt::For { .. }
                | Stmt::While { .. }
                | Stmt::Break
                | Stmt::Continue
                | Stmt::Return(_) => false,
                _ => true,
            })
        }
        let (a, b, c) = (
            self.const_int(start)?,
            self.const_int(stop)?,
            self.const_int(step)?,
        );
        if c == 0 || !plain(body) {
            return None;
        }
        let values: Vec<i64> = if c > 0 {
            (a..b).step_by(c as usize).collect()
        } else {
            let mut v = Vec::new();
            let mut x = a;
            while x > b {
                v.push(x);
                x += c;
            }
            v
        };
        (values.len() <= 64).then_some(values)
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Atomic {
                op,
                param,
                index,
                in_bounds,
                value,
                err,
                code,
            } => {
                let ty = self.k.params[*param].ty;
                let p = param_name(self.k, *param);
                let e = param_name(self.k, *err);
                let (idx, ok, v) = (self.expr(index), self.expr(in_bounds), self.expr(value));
                self.out.line("{");
                self.out.depth += 1;
                self.out.line(format!("let gs_ai = {idx};"));
                self.out.line(format!("let gs_av = {v};"));
                self.out.line(format!("if ({ok}) {{"));
                let f = match op {
                    AtomicOp::Add => "atomicAdd",
                    AtomicOp::Min => "atomicMin",
                    AtomicOp::Max => "atomicMax",
                };
                if ty == Scalar::F32 {
                    // No float atomics in WGSL: compare-and-swap loop on the bits.
                    let combine = match op {
                        AtomicOp::Add => "bitcast<f32>(gs_old) + gs_av",
                        AtomicOp::Min => "min(bitcast<f32>(gs_old), gs_av)",
                        AtomicOp::Max => "max(bitcast<f32>(gs_old), gs_av)",
                    };
                    self.out
                        .line(format!("    var gs_old = atomicLoad(&{p}[u32(gs_ai)]);"));
                    self.out.line("    loop {");
                    self.out.line(format!(
                        "        let gs_r = atomicCompareExchangeWeak(&{p}[u32(gs_ai)], gs_old, bitcast<u32>({combine}));"
                    ));
                    self.out.line("        if (gs_r.exchanged) { break; }");
                    self.out.line("        gs_old = gs_r.old_value;");
                    self.out.line("    }");
                } else {
                    self.out
                        .line(format!("    _ = {f}(&{p}[u32(gs_ai)], gs_av);"));
                }
                self.out.line("} else {");
                self.out
                    .line(format!("    _ = atomicMax(&{e}[0], {code}u);"));
                self.out
                    .line(format!("    atomicStore(&{e}[1], bitcast<u32>(gs_ai));"));
                self.out.line("}");
                self.out.depth -= 1;
                self.out.line("}");
            }
            Stmt::Assign { local, value } => {
                let l = format!("{} = {};", local_name(self.k, *local), self.expr(value));
                self.out.line(l);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let c = format!("if ({}) {{", self.expr(cond));
                self.out.line(c);
                self.indent(then);
                if !otherwise.is_empty() {
                    self.out.line("} else {");
                    self.indent(otherwise);
                }
                self.out.line("}");
            }
            Stmt::For {
                var,
                start,
                stop,
                step,
                body,
            } => {
                let v = local_name(self.k, *var);
                if let Some(values) = self.unrollable(start, stop, step, body) {
                    // Unrolled loop: constant bounds (after specialization), few iterations.
                    // Local array indices become constants: Metal and Vulkan can
                    // keep them in registers (naga rewrites loops, which stops the driver from
                    // unrolling them itself).
                    let suffix = if start.ty == Scalar::U32 { "u" } else { "i" };
                    for val in values {
                        self.out.line(format!("{v} = {val}{suffix};"));
                        self.out.line("{");
                        self.indent(body);
                        self.out.line("}");
                    }
                    return;
                }
                let n = self.tmp;
                self.tmp += 1;
                let (s0, e0, t0) = (self.expr(start), self.expr(stop), self.expr(step));
                self.out.line("{");
                self.out.depth += 1;
                self.out.line(format!("let gs_stop{n} = {e0};"));
                self.out.line(format!("let gs_step{n} = {t0};"));
                // Hidden counter (as in Python): the variable gets its value on each iteration and
                // keeps the last one after the loop.
                let it = format!("gs_it{n}");
                self.out.line(format!("var {it} = {s0};"));
                let cond = if start.ty == Scalar::U32 {
                    format!("gs_step{n} > 0u && {it} < gs_stop{n}")
                } else {
                    format!(
                        "(gs_step{n} > 0i && {it} < gs_stop{n}) || (gs_step{n} < 0i && {it} > gs_stop{n})"
                    )
                };
                self.out.line("loop {");
                self.out.depth += 1;
                self.out.line(format!("if (!({cond})) {{ break; }}"));
                self.out.line(format!("{v} = {it};"));
                self.block(body);
                self.out
                    .line(format!("continuing {{ {it} = {it} + gs_step{n}; }}"));
                self.out.depth -= 1;
                self.out.line("}");
                self.out.depth -= 1;
                self.out.line("}");
            }
            Stmt::While { cond, body } => {
                let c = format!("while ({}) {{", self.expr(cond));
                self.out.line(c);
                self.indent(body);
                self.out.line("}");
            }
            Stmt::MatMul { c, a, b, m, n, k } => {
                let (cn, an, bn) = (
                    local_name(self.k, *c),
                    local_name(self.k, *a),
                    local_name(self.k, *b),
                );
                let g = self.k.group.unwrap_or(1);
                let kw = k / 4;
                for l in [
                    "workgroupBarrier();".to_string(),
                    "{".into(),
                    format!("    for (var gs_mt = i32(gs_lid); gs_mt < {}i; gs_mt = gs_mt + {g}i) {{", m * n),
                    format!("        let gs_mr = gs_mt / {n}i;"),
                    format!("        let gs_mc = gs_mt - gs_mr * {n}i;"),
                    format!("        var gs_acc = {cn}[gs_mt];"),
                    format!("        for (var gs_mw = 0i; gs_mw < {kw}i; gs_mw = gs_mw + 1i) {{"),
                    format!("            let gs_x = {an}[gs_mr * {kw}i + gs_mw];"),
                    format!("            let gs_y = {bn}[gs_mc * {kw}i + gs_mw];"),
                    "            gs_acc = gs_acc + (gs_x & 255u) * (gs_y & 255u) + ((gs_x >> 8u) & 255u) * ((gs_y >> 8u) & 255u)".into(),
                    "                + ((gs_x >> 16u) & 255u) * ((gs_y >> 16u) & 255u) + (gs_x >> 24u) * (gs_y >> 24u);".into(),
                    "        }".into(),
                    format!("        {cn}[gs_mt] = gs_acc;"),
                    "    }".into(),
                    "}".into(),
                    "workgroupBarrier();".into(),
                ] {
                    self.out.line(l);
                }
            }
            Stmt::Store { k, value } => {
                let g = self.k.group.unwrap_or(1);
                let l = format!(
                    "gs_out0[{} + u32({}) * {g}u] = {};",
                    self.out_index(),
                    self.expr(k),
                    self.expr(value)
                );
                self.out.line(l);
            }
            Stmt::Barrier => self.out.line("workgroupBarrier();"),
            Stmt::Break => self.out.line("break;"),
            Stmt::Continue => self.out.line("continue;"),
            Stmt::LocalStore {
                local,
                index,
                value,
            } => {
                let l = format!(
                    "{}[{}] = {};",
                    local_name(self.k, *local),
                    self.expr(index),
                    self.expr(value)
                );
                self.out.line(l);
            }
            Stmt::Return(es) => {
                for (o, e) in es.iter().enumerate() {
                    let r = format!("gs_out{o}[{}] = {};", self.out_index(), self.expr(e));
                    self.out.line(r);
                }
                self.out.line("return;");
            }
        }
    }
}

const HELPERS: &[(Helper, &str)] = &[
    (
        Helper::FloorDivI32,
        "fn gs_floordiv_i32(a: i32, b: i32) -> i32 {\n    if (b == 0i) { return 0i; }\n    var q = a / b;\n    if ((a % b != 0i) && ((a < 0i) != (b < 0i))) { q = q - 1i; }\n    return q;\n}",
    ),
    (
        Helper::FloorDivU32,
        "fn gs_floordiv_u32(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a / b; }",
    ),
    (
        Helper::ModI32,
        "fn gs_mod_i32(a: i32, b: i32) -> i32 {\n    if (b == 0i) { return 0i; }\n    var r = a % b;\n    if (r != 0i && ((r < 0i) != (b < 0i))) { r = r + b; }\n    return r;\n}",
    ),
    (
        Helper::ModU32,
        "fn gs_mod_u32(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a % b; }",
    ),
    (
        Helper::FloorDivI64,
        "fn gs_floordiv_i64(a: i64, b: i64) -> i64 {\n    if (b == 0li) { return 0li; }\n    var q = a / b;\n    if ((a % b != 0li) && ((a < 0li) != (b < 0li))) { q = q - 1li; }\n    return q;\n}",
    ),
    (
        Helper::FloorDivU64,
        "fn gs_floordiv_u64(a: u64, b: u64) -> u64 { if (b == 0lu) { return 0lu; } return a / b; }",
    ),
    (
        Helper::ModI64,
        "fn gs_mod_i64(a: i64, b: i64) -> i64 {\n    if (b == 0li) { return 0li; }\n    var r = a % b;\n    if (r != 0li && ((r < 0li) != (b < 0li))) { r = r + b; }\n    return r;\n}",
    ),
    (
        Helper::ModU64,
        "fn gs_mod_u64(a: u64, b: u64) -> u64 { if (b == 0lu) { return 0lu; } return a % b; }",
    ),
    (
        Helper::ModF32,
        "fn gs_mod_f32(a: f32, b: f32) -> f32 {\n    var r = a % b;\n    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) { r = r + b; }\n    return r;\n}",
    ),
    (
        Helper::ModF64,
        "fn gs_mod_f64(a: f64, b: f64) -> f64 {\n    var r = a % b;\n    if (r != 0.0lf && ((r < 0.0lf) != (b < 0.0lf))) { r = r + b; }\n    return r;\n}",
    ),
    // WGSL pow is only defined for x ≥ 0: restore the C / Python semantics.
    (
        Helper::PowF32,
        "fn gs_pow_f32(x: f32, y: f32) -> f32 {\n    if (y == 0.0f) { return 1.0f; }\n    if (x > 0.0f) { return pow(x, y); }\n    if (x == 0.0f) { return select(bitcast<f32>(0x7f800000u), 0.0f, y > 0.0f); }\n    if (floor(y) != y) { return bitcast<f32>(0x7fc00000u); }\n    let r = pow(-x, y);\n    if (abs(y % 2.0f) == 1.0f) { return -r; }\n    return r;\n}",
    ),
];

/// Metal: 64-bit Euclidean division by shifts and subtractions (64 rounds), without `/` or `%`.
const METAL_U64_DIVREM: &str = "fn gs_udiv_u64(a: u64, b: u64) -> u64 {\n    if (b == 0lu) { return 0lu; }\n    var q = 0lu;\n    var r = 0lu;\n    for (var i = 63i; i >= 0i; i = i - 1i) {\n        r = (r << 1u) | ((a >> u32(i)) & 1lu);\n        if (r >= b) { r = r - b; q = q | (1lu << u32(i)); }\n    }\n    return q;\n}\nfn gs_urem_u64(a: u64, b: u64) -> u64 {\n    if (b == 0lu) { return 0lu; }\n    var r = 0lu;\n    for (var i = 63i; i >= 0i; i = i - 1i) {\n        r = (r << 1u) | ((a >> u32(i)) & 1lu);\n        if (r >= b) { r = r - b; }\n    }\n    return r;\n}\nfn gs_uabs_i64(a: i64) -> u64 {\n    if (a < 0li) { return u64(0li - a); }\n    return u64(a);\n}";

/// Metal: same functions as `HELPERS` (Python semantics), on `gs_udiv_u64` / `gs_urem_u64`.
const METAL_INT64: &[(Helper, &str)] = &[
    (
        Helper::FloorDivI64,
        "fn gs_floordiv_i64(a: i64, b: i64) -> i64 {\n    if (b == 0li) { return 0li; }\n    var q = i64(gs_udiv_u64(gs_uabs_i64(a), gs_uabs_i64(b)));\n    let r = gs_urem_u64(gs_uabs_i64(a), gs_uabs_i64(b));\n    if ((a < 0li) != (b < 0li)) {\n        q = 0li - q;\n        if (r != 0lu) { q = q - 1li; }\n    }\n    return q;\n}",
    ),
    (
        Helper::FloorDivU64,
        "fn gs_floordiv_u64(a: u64, b: u64) -> u64 { return gs_udiv_u64(a, b); }",
    ),
    (
        Helper::ModI64,
        "fn gs_mod_i64(a: i64, b: i64) -> i64 {\n    if (b == 0li) { return 0li; }\n    var r = i64(gs_urem_u64(gs_uabs_i64(a), gs_uabs_i64(b)));\n    if (a < 0li) { r = 0li - r; }\n    if (r != 0li && ((r < 0li) != (b < 0li))) { r = r + b; }\n    return r;\n}",
    ),
    (
        Helper::ModU64,
        "fn gs_mod_u64(a: u64, b: u64) -> u64 { return gs_urem_u64(a, b); }",
    ),
];

/// Shifts with a variable count (semantics of `BinOp::Shl` / `Shr`) and high half of a product
/// (`gs.mulhi`, missing from WGSL: partial products on halves).
fn bits_helper(op: u8, t: Scalar) -> String {
    let ty = w_type(t);
    let name = t.name();
    let wide = t.is_int64();
    let (u, w, s) = if wide {
        ("u64", "64lu", if t == Scalar::I64 { "li" } else { "lu" })
    } else {
        ("u32", "32u", if t == Scalar::I32 { "i" } else { "u" })
    };
    let signed = matches!(t, Scalar::I32 | Scalar::I64);
    let cnt = if signed || wide { "u32(n)" } else { "n" };
    let test = if signed {
        format!("{u}(n) < {w}")
    } else {
        format!("n < {w}")
    };
    match op {
        crate::BITS_SHL if signed => format!(
            "fn gs_shl_{name}(a: {ty}, n: {ty}) -> {ty} {{ if ({test}) {{ return {ty}({u}(a) << {cnt}); }} return 0{s}; }}"
        ),
        crate::BITS_SHL => format!(
            "fn gs_shl_{name}(a: {ty}, n: {ty}) -> {ty} {{ if ({test}) {{ return a << {cnt}; }} return 0{s}; }}"
        ),
        crate::BITS_SHR if signed => format!(
            "fn gs_shr_{name}(a: {ty}, n: {ty}) -> {ty} {{ if ({test}) {{ return a >> {cnt}; }} return select(0{s}, -1{s}, a < 0{s}); }}"
        ),
        crate::BITS_SHR => format!(
            "fn gs_shr_{name}(a: {ty}, n: {ty}) -> {ty} {{ if ({test}) {{ return a >> {cnt}; }} return 0{s}; }}"
        ),
        _ if signed => format!(
            "fn gs_mulhi_{name}(a: {ty}, b: {ty}) -> {ty} {{\n    return {ty}(gs_mulhi_{u}({u}(a), {u}(b))) - select(0{s}, b, a < 0{s}) - select(0{s}, a, b < 0{s});\n}}"
        ),
        _ => {
            let (h, m) = if wide {
                ("32u", "0xfffffffflu")
            } else {
                ("16u", "0xffffu")
            };
            format!(
                "fn gs_mulhi_{name}(a: {ty}, b: {ty}) -> {ty} {{\n    let al = a & {m}; let ah = a >> {h}; let bl = b & {m}; let bh = b >> {h};\n    let lh = al * bh; let hl = ah * bl;\n    let mid = ((al * bl) >> {h}) + (lh & {m}) + (hl & {m});\n    return ah * bh + (lh >> {h}) + (hl >> {h}) + (mid >> {h});\n}}"
            )
        }
    }
}

/// Does the kernel have f64 scalar parameters (binding `gs_fparams`)?
pub fn has_f64_params(k: &Kernel) -> bool {
    k.params
        .iter()
        .any(|p| p.kind == ParamKind::Scalar && p.ty == Scalar::F64)
}

/// Generates the WGSL module. Entry point: `main`.
pub fn emit_wgsl(k: &Kernel) -> Result<KernelSource, CodegenError> {
    emit_wgsl_variant(k, &[], false)
}

/// Variant whose `spec` parameters (i32 sizes known at call time) are constants.
/// `metal`: 64-bit integer division and remainder without the `/` and `%` operators (naga 30 translates
/// them into invalid MSL: "call to 'select' is ambiguous").
pub fn emit_wgsl_variant(
    k: &Kernel,
    spec: &crate::Spec,
    metal: bool,
) -> Result<KernelSource, CodegenError> {
    let req = k.requirements();
    if let Some(op) = req.f64_math_op {
        return Err(CodegenError(format!(
            "{op} in f64 is not available through wgpu (Vulkan would compute it at f32 precision, or even crash)"
        )));
    }
    let mut g = Gen {
        k,
        out: Out {
            buf: String::new(),
            depth: 0,
        },
        tmp: 0,
        spec: spec.to_vec(),
        const_locals: Vec::new(),
    };
    g.const_locals = g.find_const_locals();
    g.out
        .line(format!("// GPUsync — kernel \"{}\" (WGSL, generated)", k.name));
    let mut binding = 0;
    for (i, p) in k.array_params() {
        if k.writable().contains(&i) {
            // gs.Out: written only by atomics (f32: bits in an atomic<u32>).
            let t = if p.ty == Scalar::I32 { "i32" } else { "u32" };
            g.out.line(format!(
                "@group(0) @binding({binding}) var<storage, read_write> {}: array<atomic<{t}>>;",
                param_name(k, i)
            ));
        } else {
            g.out.line(format!(
                "@group(0) @binding({binding}) var<storage, read> {}: array<{}>;",
                param_name(k, i),
                w_type(p.ty)
            ));
        }
        binding += 1;
    }
    for (o, t) in k.outputs.iter().enumerate() {
        g.out.line(format!(
            "@group(0) @binding({binding}) var<storage, read_write> gs_out{o}: array<{}>;",
            w_type(*t)
        ));
        binding += 1;
    }
    let binding = binding - 1; // last output binding: the parameters follow
    g.out.line(format!(
        "@group(0) @binding({}) var<storage, read> gs_params: array<u32>;",
        binding + 1
    ));
    if has_f64_params(k) {
        // No bitcast to f64 in naga: f64 scalars have their own buffer.
        g.out.line(format!(
            "@group(0) @binding({}) var<storage, read> gs_fparams: array<f64>;",
            binding + 2
        ));
    }
    let used = helpers(k);
    let int64_div = [
        Helper::FloorDivI64,
        Helper::FloorDivU64,
        Helper::ModI64,
        Helper::ModU64,
    ];
    if metal && int64_div.iter().any(|h| used.contains(h)) {
        g.out.line(METAL_U64_DIVREM);
    }
    for (h, code) in HELPERS {
        if used.contains(h) {
            match METAL_INT64.iter().find(|(m, _)| metal && m == h) {
                Some((_, alt)) => g.out.line(*alt),
                None => g.out.line(*code),
            }
        }
    }
    for h in &used {
        if let Helper::Bits(op, t) = h {
            g.out.line(bits_helper(*op, crate::INT_TYPES[*t as usize]));
        }
    }
    // u32 words of the scalars (two per 64-bit integer: low, then high), then n.
    let n_scalars: usize = k
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar && p.ty != Scalar::F64)
        .map(|p| if p.ty.is_int64() { 2 } else { 1 })
        .sum();
    // Shared memory: module variables (zero-filled by wgpu for each group).
    for (i, l) in k.locals.iter().enumerate() {
        if let (true, Some(n)) = (l.shared, l.len) {
            g.out.line(format!(
                "var<workgroup> {}: array<{}, {n}>;",
                local_name(k, i),
                w_type(l.ty)
            ));
        }
    }
    match k.group {
        None => {
            g.out
                .line(format!("@compute @workgroup_size({WGSL_WORKGROUP_SIZE})"));
            g.out.line("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {");
            g.out.depth = 1;
            g.out.line(format!(
                "let gs_i = gid.x + gid.y * (nwg.x * {WGSL_WORKGROUP_SIZE}u);"
            ));
            g.out
                .line(format!("if (gs_i >= gs_params[{n_scalars}]) {{ return; }}"));
        }
        Some(gsz) => {
            // Guard on the group number (uniform): barriers stay in uniform control flow.
            g.out.line(format!("@compute @workgroup_size({gsz})"));
            g.out.line("fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) gs_lid: u32, @builtin(num_workgroups) nwg: vec3<u32>) {");
            g.out.depth = 1;
            g.out.line("let gs_wg = wid.x + wid.y * nwg.x;");
            g.out.line(format!(
                "if (gs_wg * {gsz}u >= gs_params[{n_scalars}]) {{ return; }}"
            ));
            // cells=: first cell of the thread in its group's tile (gsz·cells cells).
            g.out.line(format!(
                "let gs_i = gs_wg * {}u + gs_lid;",
                gsz * k.cells.max(1)
            ));
        }
    }
    let (mut word, mut fword) = (0, 0);
    for (i, p) in k.params.iter().enumerate() {
        if p.kind == ParamKind::Scalar {
            let read = match p.ty {
                Scalar::F64 => {
                    fword += 1;
                    format!("gs_fparams[{}]", fword - 1)
                }
                Scalar::U32 => {
                    word += 1;
                    format!("gs_params[{}]", word - 1)
                }
                t @ (Scalar::I64 | Scalar::U64) => {
                    word += 2;
                    let w = w_type(t);
                    format!(
                        "(({w}(gs_params[{}]) << 32u) | {w}(gs_params[{}]))",
                        word - 1,
                        word - 2
                    )
                }
                t => {
                    word += 1;
                    format!("bitcast<{}>(gs_params[{}])", w_type(t), word - 1)
                }
            };
            g.out.line(format!("let {} = {read};", param_name(k, i)));
        }
    }
    if let Grid::Indexed { axes, row0, .. } = &k.grid {
        // Coordinates of the thread in the grid (output shape, C order); the launch
        // covers a block of rows starting at row row0.
        g.out.line("var gs_r = gs_i;");
        for (a, p) in axes.iter().enumerate().skip(1).rev() {
            let size = param_name(k, *p);
            g.out
                .line(format!("let gs_g{a} = i32(gs_r % u32({size}));"));
            g.out.line(format!("gs_r = gs_r / u32({size});"));
        }
        g.out
            .line(format!("let gs_g0 = i32(gs_r) + {};", param_name(k, *row0)));
    }
    for (i, l) in k.locals.iter().enumerate() {
        match l.len {
            Some(_) if l.shared => {}
            Some(n) => g.out.line(format!(
                "var {}: array<{}, {n}>;",
                local_name(k, i),
                w_type(l.ty)
            )),
            None => g
                .out
                .line(format!("var {}: {};", local_name(k, i), w_type(l.ty))),
        }
    }
    g.block(&k.body);
    g.out.depth = 0;
    g.out.line("}");
    Ok(KernelSource {
        code: g.out.buf,
        entry: "main".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_wgsl_module() {
        let src = emit_wgsl(&crate::tests::sample()).unwrap();
        let c = &src.code;
        assert!(
            c.contains("@group(0) @binding(0) var<storage, read> p0_x: array<f32>;"),
            "{c}"
        );
        assert!(
            c.contains("@binding(1) var<storage, read_write> gs_out0: array<f32>;"),
            "{c}"
        );
        assert!(
            c.contains("let p1_mean = bitcast<f32>(gs_params[0]);"),
            "{c}"
        );
        assert!(
            c.contains("let p2__cart = bitcast<i32>(gs_params[1]);"),
            "{c}"
        );
        assert!(c.contains("if (gs_i >= gs_params[2]) { return; }"), "{c}");
        assert!(c.contains("gs_pow_f32(p0_x[gs_i], 2.0f)"), "{c}");
        assert!(c.contains("select("), "{c}");
    }

    #[test]
    fn f64_arithmetic_accepted_but_not_transcendental_functions() {
        let mut k = crate::tests::sample();
        // Everything in f64: the sample uses ** (pow) → rejected.
        k.outputs = vec![Scalar::F64];
        k.params[0].ty = Scalar::F64;
        k.params[1].ty = Scalar::F64;
        fn retype(b: &mut Block) {
            for s in b {
                match s {
                    Stmt::Assign { value, .. } => retype_e(value),
                    Stmt::Return(values) => values.iter_mut().for_each(retype_e),
                    Stmt::For { body, .. } => retype(body),
                    _ => {}
                }
            }
        }
        fn retype_e(e: &mut Expr) {
            if e.ty == Scalar::F32 {
                e.ty = Scalar::F64;
            }
            match &mut e.kind {
                ExprKind::Const(Const::F32(v)) => e.kind = ExprKind::Const(Const::F64(*v as f64)),
                ExprKind::Binary(_, a, b) | ExprKind::Compare(_, a, b) => {
                    retype_e(a);
                    retype_e(b);
                }
                ExprKind::Select(c, a, b) => {
                    retype_e(c);
                    retype_e(a);
                    retype_e(b);
                }
                _ => {}
            }
        }
        k.locals[0].ty = Scalar::F64;
        retype(&mut k.body);
        let err = emit_wgsl(&k).unwrap_err().0;
        assert!(err.contains("**") && err.contains("f64"), "{err}");
        // Without the **: accepted, with the gs_fparams buffer and the f64 float %.
        if let Stmt::Return(es) = &mut k.body[2] {
            let e = &mut es[0];
            if let ExprKind::Select(_, a, _) = &mut e.kind {
                **a = Expr {
                    ty: Scalar::F64,
                    kind: ExprKind::Param(0),
                };
            }
        }
        let c = emit_wgsl(&k).unwrap().code;
        assert!(
            c.contains("var<storage, read> gs_fparams: array<f64>;"),
            "{c}"
        );
        assert!(c.contains("let p1_mean = gs_fparams[0];"), "{c}");
        assert!(c.contains("gs_mod_f64(p0_x[gs_i], 3.0lf)"), "{c}");
        assert!(c.contains("if (gs_i >= gs_params[1]) { return; }"), "{c}");
    }
}
