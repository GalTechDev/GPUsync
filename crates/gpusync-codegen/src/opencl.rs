//! IR → OpenCL C, base 1.2 (`cl_khr_fp64` activée seulement si le kernel utilise f64), et IR → CUDA C
//! (même dialecte C à quelques différences près : signature, indice du thread, fonctions f32
//! suffixées, atomiques natifs).

use gpusync_ir::*;

use crate::{CodegenError, Helper, KernelSource, Out, helpers, local_name, param_name};

fn c_type(t: Scalar) -> &'static str {
    match t {
        Scalar::Bool => "bool",
        Scalar::I32 => "int",
        Scalar::U32 => "uint",
        Scalar::F32 => "float",
        Scalar::F64 => "double",
        // Alias définis en tête du code : `long` / `ulong` en OpenCL, `long long` en CUDA.
        Scalar::I64 => "gs_i64",
        Scalar::U64 => "gs_u64",
    }
}

fn lit(c: Const) -> String {
    match c {
        Const::Bool(b) => (if b { "true" } else { "false" }).into(),
        Const::I32(i32::MIN) => "(-2147483647 - 1)".into(),
        Const::I32(v) => format!("{v}"),
        Const::U32(v) => format!("{v}u"),
        Const::F32(v) if v.is_nan() => "NAN".into(),
        Const::F32(v) if v.is_infinite() => {
            (if v > 0.0 { "INFINITY" } else { "(-INFINITY)" }).into()
        }
        Const::F32(v) => format!("{}f", float_repr(format!("{v:?}"))),
        Const::F64(v) if v.is_nan() => "((double)NAN)".into(),
        Const::F64(v) if v.is_infinite() => (if v > 0.0 {
            "((double)INFINITY)"
        } else {
            "(-(double)INFINITY)"
        })
        .into(),
        Const::F64(v) => float_repr(format!("{v:?}")),
        Const::I64(i64::MIN) => "((gs_i64)(-9223372036854775807 - 1))".into(),
        Const::I64(v) => format!("((gs_i64){v})"),
        Const::U64(v) => format!("((gs_u64){v}u)"),
    }
}

/// Garantit un littéral flottant (« 3 » → « 3.0 »).
pub(crate) fn float_repr(s: String) -> String {
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

struct Gen<'a> {
    k: &'a Kernel,
    out: Out,
    tmp: usize,
    /// CUDA C (NVRTC) plutôt qu'OpenCL C.
    cuda: bool,
    /// Indices de tableaux calculés en 64 bits (un tableau dépasse 2³¹ − 1 éléments).
    wide: bool,
    /// Paramètres remplacés par une constante (spécialisation).
    spec: Vec<(usize, i64)>,
}

impl Gen<'_> {
    /// Indice d'écriture des sorties : décalé de `obase` pour un kernel avec index lancé par blocs.
    fn out_index(&self) -> String {
        match &self.k.grid {
            Grid::Indexed { obase, .. } => format!("gs_i + {}", param_name(self.k, *obase)),
            Grid::Elementwise => "gs_i".into(),
        }
    }

    /// Nom d'une fonction mathématique : en CUDA, la version f32 est suffixée (`sqrtf`), ce qui
    /// évite toute ambiguïté de surcharge et tout passage implicite en f64.
    fn math(&self, base: &str, ty: Scalar) -> String {
        if self.cuda && ty == Scalar::F32 {
            format!("{base}f")
        } else {
            base.to_string()
        }
    }

    /// Indice aplati d'un tableau. Variante large : additions, soustractions et produits en 64 bits
    /// (`long` en OpenCL, `long long` en CUDA, où `long` fait 32 bits sous Windows).
    fn index(&self, e: &Expr) -> String {
        if !self.wide {
            return self.expr(e);
        }
        let t = if self.cuda { "long long" } else { "long" };
        match &e.kind {
            ExprKind::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), a, b) => {
                format!("({} {} {})", self.index(a), infix(*op), self.index(b))
            }
            _ => format!("(({t}){})", self.expr(e)),
        }
    }

    fn expr(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Const(c) => lit(*c),
            ExprKind::Local(i) => local_name(self.k, *i),
            ExprKind::Param(i) => match self.k.params[*i].kind {
                ParamKind::Array => format!("{}[gs_i]", param_name(self.k, *i)),
                ParamKind::Scalar => match crate::spec_value(&self.spec, *i) {
                    Some(v) => format!("{v}"),
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
                    // Décalage à gauche d'un signé : en non signé (débordement indéfini en C).
                    Some(c) if *op == BinOp::Shl && matches!(a.ty, Scalar::I32 | Scalar::I64) => {
                        format!(
                            "(({})(({}){x} << {c}))",
                            c_type(a.ty),
                            c_type(unsigned(a.ty))
                        )
                    }
                    Some(c) => format!("({x} {} {c})", if *op == BinOp::Shl { "<<" } else { ">>" }),
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
                    (BinOp::FloorDiv, t) => format!("{}({x} / {y})", self.math("floor", t)),
                    (BinOp::Mod, Scalar::I32) => format!("gs_mod_i32({x}, {y})"),
                    (BinOp::Mod, Scalar::U32) => format!("gs_mod_u32({x}, {y})"),
                    (BinOp::Mod, Scalar::F32) => format!("gs_mod_f32({x}, {y})"),
                    (BinOp::Mod, _) => format!("gs_mod_f64({x}, {y})"),
                    (BinOp::Pow, t) => format!("{}({x}, {y})", self.math("pow", t)),
                    _ => format!("({x} {} {y})", infix(*op)),
                }
            }
            ExprKind::Compare(op, a, b) => {
                format!("({} {} {})", self.expr(a), cmp(*op), self.expr(b))
            }
            ExprKind::Select(c, a, b) => {
                format!("({} ? {} : {})", self.expr(c), self.expr(a), self.expr(b))
            }
            ExprKind::Call(f, args) => {
                let a: Vec<String> = args.iter().map(|x| self.expr(x)).collect();
                let float = args[0].ty.is_float();
                match f {
                    Intrinsic::MulHi if !self.cuda => format!("mul_hi({}, {})", a[0], a[1]),
                    Intrinsic::MulHi => {
                        let n = ["__mulhi", "__umulhi", "__mul64hi", "__umul64hi"]
                            [crate::int_index(args[0].ty) as usize];
                        format!("{n}({}, {})", a[0], a[1])
                    }
                    Intrinsic::Abs if float => {
                        format!("{}({})", self.math("fabs", args[0].ty), a[0])
                    }
                    Intrinsic::Abs if args[0].ty == Scalar::I32 => format!("((int)abs({}))", a[0]),
                    Intrinsic::Abs if args[0].ty == Scalar::I64 => {
                        format!("(({0}) < 0 ? -({0}) : ({0}))", a[0])
                    }
                    // min / max 64 bits : expression ternaire (surcharges variables selon le dialecte).
                    Intrinsic::Min if args[0].ty.is_int64() => {
                        format!("(({0}) < ({1}) ? ({0}) : ({1}))", a[0], a[1])
                    }
                    Intrinsic::Max if args[0].ty.is_int64() => {
                        format!("(({0}) > ({1}) ? ({0}) : ({1}))", a[0], a[1])
                    }
                    Intrinsic::Abs => a[0].clone(),
                    Intrinsic::Min if float => {
                        format!("{}({}, {})", self.math("fmin", args[0].ty), a[0], a[1])
                    }
                    Intrinsic::Max if float => {
                        format!("{}({}, {})", self.math("fmax", args[0].ty), a[0], a[1])
                    }
                    Intrinsic::Min => format!("min({}, {})", a[0], a[1]),
                    Intrinsic::Max => format!("max({}, {})", a[0], a[1]),
                    _ => format!(
                        "{}({})",
                        self.math(intrinsic_name(*f), args[0].ty),
                        a.join(", ")
                    ),
                }
            }
            ExprKind::Load(p, i) => format!("{}[{}]", param_name(self.k, *p), self.index(i)),
            ExprKind::LocalLoad(l, i) => format!("{}[{}]", local_name(self.k, *l), self.expr(i)),
            ExprKind::GridIndex(a) => format!("gs_g{a}"),
            ExprKind::Lane => "gs_lane".into(),
            ExprKind::Cast(a) if e.ty == Scalar::Bool => format!("({} != 0)", self.expr(a)),
            ExprKind::Cast(a) => format!("(({}){})", c_type(e.ty), self.expr(a)),
        }
    }

    /// `gs.matmul_u8` : boucle portable (un thread par case de c, produits d'octets) ; en CUDA, sur
    /// Ampere et suivants (`__CUDA_ARCH__ >= 800`), tuiles 16 × 8 × 32 par les Tensor Cores
    /// (`mma.sync … u8.u8.s32`, une tuile par warp à la fois).
    #[allow(clippy::too_many_arguments)]
    fn matmul(&mut self, c: usize, a: usize, b: usize, m: u32, n: u32, k: u32) {
        let (cn, an, bn) = (
            local_name(self.k, c),
            local_name(self.k, a),
            local_name(self.k, b),
        );
        let g = self.k.group.unwrap_or(1);
        let kw = k / 4;
        let barrier = if self.cuda {
            "__syncthreads();"
        } else {
            "barrier(CLK_LOCAL_MEM_FENCE);"
        };
        let portable = [
            format!("for (int gs_mt = gs_lane; gs_mt < {}; gs_mt += {g}) {{", m * n),
            format!("    const int gs_mr = gs_mt / {n};"),
            format!("    const int gs_mc = gs_mt - gs_mr * {n};"),
            format!("    uint gs_acc = {cn}[gs_mt];"),
            format!("    for (int gs_mw = 0; gs_mw < {kw}; gs_mw++) {{"),
            format!("        const uint gs_x = {an}[gs_mr * {kw} + gs_mw];"),
            format!("        const uint gs_y = {bn}[gs_mc * {kw} + gs_mw];"),
            "        gs_acc += (gs_x & 255u) * (gs_y & 255u) + ((gs_x >> 8) & 255u) * ((gs_y >> 8) & 255u)".into(),
            "            + ((gs_x >> 16) & 255u) * ((gs_y >> 16) & 255u) + (gs_x >> 24) * (gs_y >> 24);".into(),
            "    }".into(),
            format!("    {cn}[gs_mt] = gs_acc;"),
            "}".into(),
        ];
        self.out.line(barrier);
        self.out.line("{");
        let tensor = self.cuda
            && m.is_multiple_of(16)
            && n.is_multiple_of(8)
            && k.is_multiple_of(32)
            && g.is_multiple_of(32);
        if tensor {
            let (nt, tiles, warps) = (n / 8, (m / 16) * (n / 8), g / 32);
            let at = |row: &str, off: &str| format!("{an}[({row}) * {kw} + gs_kk + {off}gs_mq]");
            let ct = |row: &str, col: &str| format!("{cn}[({row}) * {n} + gs_tn + 2 * gs_mq{col}]");
            let lines = [
                "#if __CUDA_ARCH__ >= 800".to_string(),
                "const int gs_ml = gs_lane & 31;".into(),
                "const int gs_mg = gs_ml >> 2;".into(),
                "const int gs_mq = gs_ml & 3;".into(),
                format!("for (int gs_tile = gs_lane >> 5; gs_tile < {tiles}; gs_tile += {warps}) {{"),
                format!("    const int gs_tm = (gs_tile / {nt}) * 16;"),
                format!("    const int gs_tn = (gs_tile % {nt}) * 8;"),
                format!("    uint gs_d0 = {};", ct("gs_tm + gs_mg", "")),
                format!("    uint gs_d1 = {};", ct("gs_tm + gs_mg", " + 1")),
                format!("    uint gs_d2 = {};", ct("gs_tm + gs_mg + 8", "")),
                format!("    uint gs_d3 = {};", ct("gs_tm + gs_mg + 8", " + 1")),
                format!("    for (int gs_kk = 0; gs_kk < {kw}; gs_kk += 8) {{"),
                format!("        const uint gs_a0 = {};", at("gs_tm + gs_mg", "")),
                format!("        const uint gs_a1 = {};", at("gs_tm + gs_mg + 8", "")),
                format!("        const uint gs_a2 = {};", at("gs_tm + gs_mg", "4 + ")),
                format!("        const uint gs_a3 = {};", at("gs_tm + gs_mg + 8", "4 + ")),
                format!("        const uint gs_b0 = {bn}[(gs_tn + gs_mg) * {kw} + gs_kk + gs_mq];"),
                format!("        const uint gs_b1 = {bn}[(gs_tn + gs_mg) * {kw} + gs_kk + 4 + gs_mq];"),
                "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, \"".into(),
                "                     \"{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"".into(),
                "                     : \"+r\"(gs_d0), \"+r\"(gs_d1), \"+r\"(gs_d2), \"+r\"(gs_d3)".into(),
                "                     : \"r\"(gs_a0), \"r\"(gs_a1), \"r\"(gs_a2), \"r\"(gs_a3), \"r\"(gs_b0), \"r\"(gs_b1));".into(),
                "    }".into(),
                format!("    {} = gs_d0;", ct("gs_tm + gs_mg", "")),
                format!("    {} = gs_d1;", ct("gs_tm + gs_mg", " + 1")),
                format!("    {} = gs_d2;", ct("gs_tm + gs_mg + 8", "")),
                format!("    {} = gs_d3;", ct("gs_tm + gs_mg + 8", " + 1")),
                "}".into(),
                "#else".into(),
            ];
            for l in lines {
                self.out.line(l);
            }
            for l in &portable {
                self.out.line(l.clone());
            }
            self.out.line("#endif");
        } else {
            for l in &portable {
                self.out.line(l.clone());
            }
        }
        self.out.line("}");
        self.out.line(barrier);
    }

    fn block(&mut self, b: &Block) {
        for s in b {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
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
                let n = self.tmp;
                self.tmp += 1;
                let t = c_type(start.ty);
                let v = local_name(self.k, *var);
                let (s0, e0, t0) = (self.expr(start), self.expr(stop), self.expr(step));
                self.out.line("{");
                self.out.depth += 1;
                self.out.line(format!("const {t} gs_start{n} = {s0};"));
                self.out.line(format!("const {t} gs_stop{n} = {e0};"));
                self.out.line(format!("const {t} gs_step{n} = {t0};"));
                // Compteur caché (comme en Python) : la variable reçoit sa valeur à chaque tour et
                // garde la dernière après la boucle.
                let it = format!("gs_it{n}");
                let cond = if start.ty == Scalar::U32 {
                    format!("gs_step{n} > 0u && {it} < gs_stop{n}")
                } else {
                    format!(
                        "(gs_step{n} > 0 && {it} < gs_stop{n}) || (gs_step{n} < 0 && {it} > gs_stop{n})"
                    )
                };
                self.out.line(format!(
                    "for ({t} {it} = gs_start{n}; {cond}; {it} += gs_step{n}) {{"
                ));
                self.out.line(format!("    {v} = {it};"));
                self.indent(body);
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
            Stmt::MatMul { c, a, b, m, n, k } => self.matmul(*c, *a, *b, *m, *n, *k),
            Stmt::Store { k, value } => {
                let g = self.k.group.unwrap_or(1);
                let l = format!(
                    "gs_out0[{} + (uint)({}) * {g}u] = {};",
                    self.out_index(),
                    self.expr(k),
                    self.expr(value)
                );
                self.out.line(l);
            }
            Stmt::Barrier => self.out.line(if self.cuda {
                "__syncthreads();"
            } else {
                "barrier(CLK_LOCAL_MEM_FENCE);"
            }),
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
                let (idx, ok, v) = (self.index(index), self.expr(in_bounds), self.expr(value));
                self.out.line("{");
                self.out.depth += 1;
                let it = match (self.wide, self.cuda) {
                    (false, _) => "int",
                    (true, true) => "long long",
                    (true, false) => "long",
                };
                self.out.line(format!("const {it} gs_ai = {idx};"));
                self.out.line(format!("if ({ok}) {{"));
                let cu = ["atomicAdd", "atomicMin", "atomicMax"][*op as usize];
                let call = match ty {
                    // CUDA : addition flottante atomique native ; min et max émulés.
                    Scalar::F32 if self.cuda && *op == AtomicOp::Add => {
                        format!("atomicAdd(&{p}[gs_ai], {v});")
                    }
                    Scalar::F32 => format!("gs_atomic_{}_f32(&{p}[gs_ai], {v});", op.name()),
                    _ if self.cuda => format!("{cu}(&{p}[gs_ai], {v});"),
                    _ => format!("atomic_{}(&{p}[gs_ai], {v});", op.name()),
                };
                self.out.line(format!("    {call}"));
                self.out.line("} else {");
                let amax = if self.cuda { "atomicMax" } else { "atomic_max" };
                self.out.line(format!("    {amax}(&{e}[0], {code}u);"));
                self.out.line(format!("    {e}[1] = (uint)gs_ai;"));
                self.out.line("}");
                self.out.depth -= 1;
                self.out.line("}");
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

    fn indent(&mut self, b: &Block) {
        self.out.depth += 1;
        self.block(b);
        self.out.depth -= 1;
    }
}

fn infix(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        _ => "?",
    }
}

fn cmp(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

pub(crate) fn intrinsic_name(f: Intrinsic) -> &'static str {
    match f {
        Intrinsic::Sqrt => "sqrt",
        Intrinsic::Exp => "exp",
        Intrinsic::Exp2 => "exp2",
        Intrinsic::Log => "log",
        Intrinsic::Log2 => "log2",
        Intrinsic::Sin => "sin",
        Intrinsic::Cos => "cos",
        Intrinsic::Tan => "tan",
        Intrinsic::Tanh => "tanh",
        Intrinsic::Atan2 => "atan2",
        Intrinsic::Floor => "floor",
        Intrinsic::Ceil => "ceil",
        Intrinsic::Pow => "pow",
        Intrinsic::Abs => "abs",
        Intrinsic::Min => "min",
        Intrinsic::Max => "max",
        Intrinsic::MulHi => "mul_hi",
    }
}

fn unsigned(t: Scalar) -> Scalar {
    match t {
        Scalar::I32 => Scalar::U32,
        Scalar::I64 => Scalar::U64,
        t => t,
    }
}

/// Décalage dont le compte n'est pas une constante : lu comme non signé, ≥ largeur → 0 (ou −1 pour
/// `>>` d'un négatif), comme en Python.
fn c_shift_helper(op: u8, t: Scalar) -> String {
    let (ty, u) = (c_type(t), c_type(unsigned(t)));
    let w = if t.is_int64() { 64 } else { 32 };
    let name = t.name();
    if op == crate::BITS_SHL {
        format!(
            "inline {ty} gs_shl_{name}({ty} a, {ty} n) {{ return ({u})n < {w} ? ({ty})(({u})a << ({u})n) : ({ty})0; }}"
        )
    } else if t == unsigned(t) {
        format!(
            "inline {ty} gs_shr_{name}({ty} a, {ty} n) {{ return n < {w} ? a >> n : ({ty})0; }}"
        )
    } else {
        format!(
            "inline {ty} gs_shr_{name}({ty} a, {ty} n) {{ return ({u})n < {w} ? a >> ({u})n : (a < 0 ? ({ty})-1 : ({ty})0); }}"
        )
    }
}

const HELPERS: &[(Helper, &str)] = &[
    (
        Helper::FloorDivI32,
        "inline int gs_floordiv_i32(int a, int b) {\n    if (b == 0) return 0;\n    int q = a / b;\n    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;\n    return q;\n}",
    ),
    (
        Helper::FloorDivU32,
        "inline uint gs_floordiv_u32(uint a, uint b) { return b == 0u ? 0u : a / b; }",
    ),
    (
        Helper::ModI32,
        "inline int gs_mod_i32(int a, int b) {\n    if (b == 0) return 0;\n    int r = a % b;\n    if (r != 0 && ((r < 0) != (b < 0))) r += b;\n    return r;\n}",
    ),
    (
        Helper::ModU32,
        "inline uint gs_mod_u32(uint a, uint b) { return b == 0u ? 0u : a % b; }",
    ),
    (
        Helper::ModF32,
        "inline float gs_mod_f32(float a, float b) {\n    float r = fmod(a, b);\n    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;\n    return r;\n}",
    ),
    (
        Helper::ModF64,
        "inline double gs_mod_f64(double a, double b) {\n    double r = fmod(a, b);\n    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;\n    return r;\n}",
    ),
    (
        Helper::FloorDivI64,
        "inline gs_i64 gs_floordiv_i64(gs_i64 a, gs_i64 b) {\n    if (b == 0) return 0;\n    gs_i64 q = a / b;\n    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;\n    return q;\n}",
    ),
    (
        Helper::FloorDivU64,
        "inline gs_u64 gs_floordiv_u64(gs_u64 a, gs_u64 b) { return b == 0 ? 0 : a / b; }",
    ),
    (
        Helper::ModI64,
        "inline gs_i64 gs_mod_i64(gs_i64 a, gs_i64 b) {\n    if (b == 0) return 0;\n    gs_i64 r = a % b;\n    if (r != 0 && ((r < 0) != (b < 0))) r += b;\n    return r;\n}",
    ),
    (
        Helper::ModU64,
        "inline gs_u64 gs_mod_u64(gs_u64 a, gs_u64 b) { return b == 0 ? 0 : a % b; }",
    ),
];

/// Génère le kernel OpenCL C. Point d'entrée : `gs_<nom>`.
pub fn emit_opencl(k: &Kernel) -> Result<KernelSource, CodegenError> {
    emit_c(k, false, false, &[])
}

/// Génère le kernel CUDA C (compilé par NVRTC). Point d'entrée : `gs_<nom>` (`extern "C"`).
pub fn emit_cuda(k: &Kernel) -> Result<KernelSource, CodegenError> {
    emit_c(k, true, false, &[])
}

/// Variante (OpenCL C ou CUDA C) : indices 64 bits si `wide` (un tableau dépasse 2³¹ − 1 éléments ;
/// plus lente), et paramètres de `spec` remplacés par des constantes (tailles connues à l'appel : le
/// compilateur peut dérouler les boucles et garder les tableaux locaux en registres).
pub fn emit_c_variant(
    k: &Kernel,
    cuda: bool,
    wide: bool,
    spec: &crate::Spec,
) -> Result<KernelSource, CodegenError> {
    emit_c(k, cuda, wide, spec)
}

fn emit_c(
    k: &Kernel,
    cuda: bool,
    wide: bool,
    spec: &crate::Spec,
) -> Result<KernelSource, CodegenError> {
    let entry = format!("gs_{}", crate::sanitize(&k.name));
    let mut g = Gen {
        k,
        out: Out {
            buf: String::new(),
            depth: 0,
        },
        tmp: 0,
        cuda,
        wide,
        spec: spec.to_vec(),
    };
    g.out.line(format!(
        "// GPUsync — kernel « {} » ({}, généré)",
        k.name,
        if cuda { "CUDA C" } else { "OpenCL C" }
    ));
    if cuda {
        // NVRTC sans en-têtes : types et constantes du dialecte OpenCL utilisés par l'émetteur.
        g.out.line("typedef unsigned int uint;");
        g.out.line("typedef long long gs_i64;");
        g.out.line("typedef unsigned long long gs_u64;");
        g.out.line("#define INFINITY __int_as_float(0x7f800000)");
        g.out.line("#define NAN __int_as_float(0x7fffffff)");
    } else {
        if k.requirements().f64 {
            g.out.line("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
        }
        if k.requirements().int64 {
            g.out.line("typedef long gs_i64;");
            g.out.line("typedef ulong gs_u64;");
        }
    }
    let used = helpers(k);
    for (h, code) in HELPERS {
        if used.contains(h) {
            if cuda {
                let c = code.replacen("inline ", "__device__ inline ", 1);
                let c = if *h == Helper::ModF32 {
                    c.replace("fmod(", "fmodf(")
                } else {
                    c
                };
                g.out.line(c);
            } else {
                g.out.line(*code);
            }
        }
    }
    for h in &used {
        if let Helper::Bits(op @ (crate::BITS_SHL | crate::BITS_SHR), t) = h {
            let c = c_shift_helper(*op, crate::INT_TYPES[*t as usize]);
            g.out.line(if cuda {
                c.replacen("inline ", "__device__ inline ", 1)
            } else {
                c
            });
        }
    }
    // Atomiques flottants : absents d'OpenCL 1.2 (et min / max en CUDA), émulés par une boucle
    // de comparaison-échange.
    for (op, ty) in crate::atomics(k) {
        if ty == Scalar::F32 && cuda && op == AtomicOp::Add as u8 {
            continue;
        }
        if ty == Scalar::F32 && cuda {
            let name = ["add", "min", "max"][op as usize];
            let combine = [
                "",
                "fminf(__uint_as_float(prev), v)",
                "fmaxf(__uint_as_float(prev), v)",
            ][op as usize];
            g.out.line(format!(
                "__device__ inline void gs_atomic_{name}_f32(float* p, float v) {{\n    unsigned int* u = (unsigned int*)p;\n    unsigned int prev, old = *u;\n    do {{\n        prev = old;\n        old = atomicCAS(u, prev, __float_as_uint({combine}));\n    }} while (old != prev);\n}}"
            ));
        } else if ty == Scalar::F32 {
            let name = ["add", "min", "max"][op as usize];
            let combine = [
                "as_float(prev) + v",
                "fmin(as_float(prev), v)",
                "fmax(as_float(prev), v)",
            ][op as usize];
            g.out.line(format!(
                "inline void gs_atomic_{name}_f32(__global float* p, float v) {{\n    volatile __global uint* u = (volatile __global uint*)p;\n    uint prev, old = *u;\n    do {{\n        prev = old;\n        old = atomic_cmpxchg(u, prev, as_uint({combine}));\n    }} while (old != prev);\n}}"
            ));
        }
    }
    let (gq, restrict) = if cuda {
        ("", "__restrict__")
    } else {
        ("__global ", "restrict")
    };
    let mut args: Vec<String> = k
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| match p.kind {
            ParamKind::Array if k.writable().contains(&i) => {
                format!("{}{}* {}", gq, c_type(p.ty), param_name(k, i))
            }
            ParamKind::Array => format!(
                "{gq}const {}* {restrict} {}",
                c_type(p.ty),
                param_name(k, i)
            ),
            ParamKind::Scalar => format!("const {} {}", c_type(p.ty), param_name(k, i)),
        })
        .collect();
    for (o, t) in k.outputs.iter().enumerate() {
        args.push(format!("{gq}{}* {restrict} gs_out{o}", c_type(*t)));
    }
    args.push("const uint gs_n".into());
    // Groupe (`group=`) : taille imposée au lancement, connue du compilateur.
    let head = match (cuda, k.group) {
        (true, Some(g)) => format!("extern \"C\" __global__ void __launch_bounds__({g})"),
        (true, None) => "extern \"C\" __global__ void".into(),
        (false, Some(g)) => {
            format!("__kernel __attribute__((reqd_work_group_size({g}, 1, 1))) void")
        }
        (false, None) => "__kernel void".into(),
    };
    g.out
        .line(format!("{head} {entry}(\n    {}) {{", args.join(",\n    ")));
    g.out.depth = 1;
    let thread = if cuda {
        "blockIdx.x * blockDim.x + threadIdx.x"
    } else {
        "get_global_id(0)"
    };
    match (k.group, k.cells) {
        (Some(gsz), c) if c > 1 => {
            // cells= : première case du thread dans la tuile de son groupe (gsz·c cases).
            g.out.line(format!("const uint gs_t = {thread};"));
            g.out.line(format!(
                "const uint gs_i = (gs_t / {gsz}u) * {}u + (gs_t % {gsz}u);",
                gsz * c
            ));
        }
        _ => g.out.line(format!("const uint gs_i = {thread};")),
    }
    if k.group.is_none() {
        g.out.line("if (gs_i >= gs_n) return;");
    } else {
        // Groupes complets (n multiple de la taille de groupe, vérifié au lancement) : pas de garde,
        // tous les threads atteignent les barrières.
        g.out.line(if cuda {
            "const int gs_lane = (int)threadIdx.x;"
        } else {
            "const int gs_lane = (int)get_local_id(0);"
        });
    }
    if let Grid::Indexed { axes, row0, .. } = &k.grid {
        // Coordonnées du thread dans la grille (forme de la sortie, ordre C) ; le lancement
        // couvre un bloc de lignes commençant à la ligne row0.
        g.out.line("uint gs_r = gs_i;");
        for (a, p) in axes.iter().enumerate().skip(1).rev() {
            let size = param_name(k, *p);
            g.out
                .line(format!("const int gs_g{a} = (int)(gs_r % (uint){size});"));
            g.out.line(format!("gs_r /= (uint){size};"));
        }
        g.out.line(format!(
            "const int gs_g0 = (int)gs_r + {};",
            param_name(k, *row0)
        ));
    }
    let mut shared = false;
    for (i, l) in k.locals.iter().enumerate() {
        match l.len {
            // Mémoire partagée : sans initialiseur en C ; mise à zéro par le groupe (comme gs.local).
            Some(n) if l.shared => {
                let q = if cuda { "__shared__" } else { "__local" };
                let name = local_name(k, i);
                g.out.line(format!("{q} {} {name}[{n}];", c_type(l.ty)));
                g.out.line(format!(
                    "for (int gs_z = gs_lane; gs_z < {n}; gs_z += {}) {name}[gs_z] = 0;",
                    k.group.unwrap_or(1)
                ));
                shared = true;
            }
            Some(n) => g.out.line(format!(
                "{} {}[{n}] = {{0}};",
                c_type(l.ty),
                local_name(k, i)
            )),
            None => g
                .out
                .line(format!("{} {};", c_type(l.ty), local_name(k, i))),
        }
    }
    if shared {
        g.stmt(&Stmt::Barrier);
    }
    g.block(&k.body);
    g.out.depth = 0;
    g.out.line("}");
    Ok(KernelSource {
        code: g.out.buf,
        entry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_readable_kernel() {
        let src = emit_opencl(&crate::tests::sample()).unwrap();
        assert_eq!(src.entry, "gs_normalize");
        let c = &src.code;
        assert!(c.contains("__global const float* restrict p0_x"), "{c}");
        assert!(c.contains("const int p2__cart"), "{c}"); // « écart » assaini en ASCII
        assert!(c.contains("gs_mod_f32(p0_x[gs_i], 3.0f)"), "{c}");
        assert!(c.contains("pow(p0_x[gs_i], 2.0f)"), "{c}");
        assert!(c.contains("gs_out0[gs_i] ="), "{c}");
        assert!(!c.contains("cl_khr_fp64"), "{c}");
    }

    #[test]
    fn literals() {
        assert_eq!(lit(Const::F32(3.0)), "3.0f");
        assert_eq!(lit(Const::F32(1e-7)), "1e-7f");
        assert_eq!(lit(Const::F32(f32::INFINITY)), "INFINITY");
        assert_eq!(lit(Const::F64(0.1)), "0.1");
        assert_eq!(lit(Const::I32(i32::MIN)), "(-2147483647 - 1)");
        assert_eq!(lit(Const::U32(7)), "7u");
    }
}
