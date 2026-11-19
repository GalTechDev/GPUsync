//! IR → OpenCL C, base 1.2 (`cl_khr_fp64` activée seulement si le kernel utilise f64).

use gpusync_ir::*;

use crate::{CodegenError, Helper, KernelSource, Out, helpers, local_name, param_name};

fn c_type(t: Scalar) -> &'static str {
    match t {
        Scalar::Bool => "bool",
        Scalar::I32 => "int",
        Scalar::U32 => "uint",
        Scalar::F32 => "float",
        Scalar::F64 => "double",
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
}

impl Gen<'_> {
    fn expr(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Const(c) => lit(*c),
            ExprKind::Local(i) => local_name(self.k, *i),
            ExprKind::Param(i) => match self.k.params[*i].kind {
                ParamKind::Array => format!("{}[gs_i]", param_name(self.k, *i)),
                ParamKind::Scalar => param_name(self.k, *i),
            },
            ExprKind::Unary(UnOp::Neg, a) => format!("(-{})", self.expr(a)),
            ExprKind::Unary(UnOp::Not, a) => format!("(!{})", self.expr(a)),
            ExprKind::Binary(op, a, b) => {
                let (x, y) = (self.expr(a), self.expr(b));
                match (op, a.ty) {
                    (BinOp::FloorDiv, Scalar::I32) => format!("gs_floordiv_i32({x}, {y})"),
                    (BinOp::FloorDiv, Scalar::U32) => format!("gs_floordiv_u32({x}, {y})"),
                    (BinOp::FloorDiv, _) => format!("floor({x} / {y})"),
                    (BinOp::Mod, Scalar::I32) => format!("gs_mod_i32({x}, {y})"),
                    (BinOp::Mod, Scalar::U32) => format!("gs_mod_u32({x}, {y})"),
                    (BinOp::Mod, Scalar::F32) => format!("gs_mod_f32({x}, {y})"),
                    (BinOp::Mod, _) => format!("gs_mod_f64({x}, {y})"),
                    (BinOp::Pow, _) => format!("pow({x}, {y})"),
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
                    Intrinsic::Abs if float => format!("fabs({})", a[0]),
                    Intrinsic::Abs if args[0].ty == Scalar::I32 => format!("((int)abs({}))", a[0]),
                    Intrinsic::Abs => a[0].clone(),
                    Intrinsic::Min => {
                        format!("{}({}, {})", if float { "fmin" } else { "min" }, a[0], a[1])
                    }
                    Intrinsic::Max => {
                        format!("{}({}, {})", if float { "fmax" } else { "max" }, a[0], a[1])
                    }
                    _ => format!("{}({})", intrinsic_name(*f), a.join(", ")),
                }
            }
            ExprKind::Cast(a) if e.ty == Scalar::Bool => format!("({} != 0)", self.expr(a)),
            ExprKind::Cast(a) => format!("(({}){})", c_type(e.ty), self.expr(a)),
        }
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
                let cond = if start.ty == Scalar::U32 {
                    format!("gs_step{n} > 0u && {v} < gs_stop{n}")
                } else {
                    format!(
                        "(gs_step{n} > 0 && {v} < gs_stop{n}) || (gs_step{n} < 0 && {v} > gs_stop{n})"
                    )
                };
                self.out.line(format!(
                    "for ({v} = gs_start{n}; {cond}; {v} += gs_step{n}) {{"
                ));
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
            Stmt::Break => self.out.line("break;"),
            Stmt::Continue => self.out.line("continue;"),
            Stmt::Return(e) => {
                let r = format!("gs_out[gs_i] = {};", self.expr(e));
                self.out.line(r);
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
];

/// Génère le kernel OpenCL C. Point d'entrée : `gs_<nom>`.
pub fn emit_opencl(k: &Kernel) -> Result<KernelSource, CodegenError> {
    let entry = format!("gs_{}", crate::sanitize(&k.name));
    let mut g = Gen {
        k,
        out: Out {
            buf: String::new(),
            depth: 0,
        },
        tmp: 0,
    };
    g.out.line(format!(
        "// GPUsync — kernel « {} » (OpenCL C, généré)",
        k.name
    ));
    if k.requirements().f64 {
        g.out.line("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
    }
    let used = helpers(k);
    for (h, code) in HELPERS {
        if used.contains(h) {
            g.out.line(*code);
        }
    }
    let mut args: Vec<String> = k
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| match p.kind {
            ParamKind::Array => format!(
                "__global const {}* restrict {}",
                c_type(p.ty),
                param_name(k, i)
            ),
            ParamKind::Scalar => format!("const {} {}", c_type(p.ty), param_name(k, i)),
        })
        .collect();
    args.push(format!("__global {}* restrict gs_out", c_type(k.output)));
    args.push("const uint gs_n".into());
    g.out.line(format!(
        "__kernel void {entry}(\n    {}) {{",
        args.join(",\n    ")
    ));
    g.out.depth = 1;
    g.out.line("const uint gs_i = get_global_id(0);");
    g.out.line("if (gs_i >= gs_n) return;");
    for (i, l) in k.locals.iter().enumerate() {
        g.out
            .line(format!("{} {};", c_type(l.ty), local_name(k, i)));
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
    fn genere_un_kernel_lisible() {
        let src = emit_opencl(&crate::tests::sample()).unwrap();
        assert_eq!(src.entry, "gs_normalize");
        let c = &src.code;
        assert!(c.contains("__global const float* restrict p0_x"), "{c}");
        assert!(c.contains("const int p2__cart"), "{c}"); // « écart » assaini en ASCII
        assert!(c.contains("gs_mod_f32(p0_x[gs_i], 3.0f)"), "{c}");
        assert!(c.contains("pow(p0_x[gs_i], 2.0f)"), "{c}");
        assert!(c.contains("gs_out[gs_i] ="), "{c}");
        assert!(!c.contains("cl_khr_fp64"), "{c}");
    }

    #[test]
    fn litteraux() {
        assert_eq!(lit(Const::F32(3.0)), "3.0f");
        assert_eq!(lit(Const::F32(1e-7)), "1e-7f");
        assert_eq!(lit(Const::F32(f32::INFINITY)), "INFINITY");
        assert_eq!(lit(Const::F64(0.1)), "0.1");
        assert_eq!(lit(Const::I32(i32::MIN)), "(-2147483647 - 1)");
        assert_eq!(lit(Const::U32(7)), "7u");
    }
}
