//! IR → WGSL (wgpu : Metal, Vulkan, DX12). Pas de f64 en WGSL : refusé avant d'arriver ici.

use gpusync_ir::*;

use crate::opencl::{float_repr, intrinsic_name};
use crate::{CodegenError, Helper, KernelSource, Out, helpers, local_name, param_name};

/// Taille d'un groupe de travail. La grille est en 2D (x, y) pour dépasser 65 535 groupes.
pub const WGSL_WORKGROUP_SIZE: u32 = 256;

fn w_type(t: Scalar) -> &'static str {
    match t {
        Scalar::Bool => "bool",
        Scalar::I32 => "i32",
        Scalar::U32 => "u32",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
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
        // Pas de bitcast vers f64 en naga : l'infini et NaN passent par un f32 (conversion exacte).
        Const::F64(v) if v.is_nan() => "f64(bitcast<f32>(0x7fc00000u))".into(),
        Const::F64(v) if v.is_infinite() => (if v > 0.0 {
            "f64(bitcast<f32>(0x7f800000u))"
        } else {
            "f64(bitcast<f32>(0xff800000u))"
        })
        .into(),
        Const::F64(v) if v < 0.0 => format!("({}lf)", float_repr(format!("{v:?}"))),
        Const::F64(v) => format!("{}lf", float_repr(format!("{v:?}"))),
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
                    (BinOp::Mod, Scalar::F64) => format!("gs_mod_f64({x}, {y})"),
                    (BinOp::Mod, _) => format!("gs_mod_f32({x}, {y})"),
                    (BinOp::Pow, _) => format!("gs_pow_f32({x}, {y})"),
                    (BinOp::Add, _) => format!("({x} + {y})"),
                    (BinOp::Sub, _) => format!("({x} - {y})"),
                    (BinOp::Mul, _) => format!("({x} * {y})"),
                    (BinOp::Div, _) => format!("({x} / {y})"),
                    (BinOp::And, _) => format!("({x} && {y})"),
                    (BinOp::Or, _) => format!("({x} || {y})"),
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
            // select(faux, vrai, condition)
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
                    Intrinsic::Abs if args[0].ty == Scalar::U32 => a[0].clone(),
                    _ => format!("{}({})", intrinsic_name(*f), a.join(", ")),
                }
            }
            ExprKind::Load(p, i) => format!("{}[{}]", param_name(self.k, *p), self.expr(i)),
            ExprKind::LocalLoad(l, i) => format!("{}[{}]", local_name(self.k, *l), self.expr(i)),
            ExprKind::GridIndex(a) => format!("gs_g{a}"),
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
                let v = local_name(self.k, *var);
                let (s0, e0, t0) = (self.expr(start), self.expr(stop), self.expr(step));
                self.out.line("{");
                self.out.depth += 1;
                self.out.line(format!("let gs_stop{n} = {e0};"));
                self.out.line(format!("let gs_step{n} = {t0};"));
                self.out.line(format!("{v} = {s0};"));
                let cond = if start.ty == Scalar::U32 {
                    format!("gs_step{n} > 0u && {v} < gs_stop{n}")
                } else {
                    format!(
                        "(gs_step{n} > 0i && {v} < gs_stop{n}) || (gs_step{n} < 0i && {v} > gs_stop{n})"
                    )
                };
                self.out.line("loop {");
                self.out.depth += 1;
                self.out.line(format!("if (!({cond})) {{ break; }}"));
                self.block(body);
                self.out
                    .line(format!("continuing {{ {v} = {v} + gs_step{n}; }}"));
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
                    let r = format!("gs_out{o}[gs_i] = {};", self.expr(e));
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
        Helper::ModF32,
        "fn gs_mod_f32(a: f32, b: f32) -> f32 {\n    var r = a % b;\n    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) { r = r + b; }\n    return r;\n}",
    ),
    (
        Helper::ModF64,
        "fn gs_mod_f64(a: f64, b: f64) -> f64 {\n    var r = a % b;\n    if (r != 0.0lf && ((r < 0.0lf) != (b < 0.0lf))) { r = r + b; }\n    return r;\n}",
    ),
    // pow WGSL n'est défini que pour x ≥ 0 : on rétablit la sémantique de C / Python.
    (
        Helper::PowF32,
        "fn gs_pow_f32(x: f32, y: f32) -> f32 {\n    if (y == 0.0f) { return 1.0f; }\n    if (x > 0.0f) { return pow(x, y); }\n    if (x == 0.0f) { return select(bitcast<f32>(0x7f800000u), 0.0f, y > 0.0f); }\n    if (floor(y) != y) { return bitcast<f32>(0x7fc00000u); }\n    let r = pow(-x, y);\n    if (abs(y % 2.0f) == 1.0f) { return -r; }\n    return r;\n}",
    ),
];

/// Le kernel a-t-il des paramètres scalaires f64 (binding `gs_fparams`) ?
pub fn has_f64_params(k: &Kernel) -> bool {
    k.params
        .iter()
        .any(|p| p.kind == ParamKind::Scalar && p.ty == Scalar::F64)
}

/// Génère le module WGSL. Point d'entrée : `main`.
pub fn emit_wgsl(k: &Kernel) -> Result<KernelSource, CodegenError> {
    let req = k.requirements();
    if let Some(op) = req.f64_math_op {
        return Err(CodegenError(format!(
            "{op} en f64 n'est pas disponible via wgpu (Vulkan le calculerait en précision f32, voire planterait)"
        )));
    }
    let mut g = Gen {
        k,
        out: Out {
            buf: String::new(),
            depth: 0,
        },
        tmp: 0,
    };
    g.out
        .line(format!("// GPUsync — kernel « {} » (WGSL, généré)", k.name));
    let mut binding = 0;
    for (i, p) in k.array_params() {
        g.out.line(format!(
            "@group(0) @binding({binding}) var<storage, read> {}: array<{}>;",
            param_name(k, i),
            w_type(p.ty)
        ));
        binding += 1;
    }
    for (o, t) in k.outputs.iter().enumerate() {
        g.out.line(format!(
            "@group(0) @binding({binding}) var<storage, read_write> gs_out{o}: array<{}>;",
            w_type(*t)
        ));
        binding += 1;
    }
    let binding = binding - 1; // dernier binding de sortie : les paramètres suivent
    g.out.line(format!(
        "@group(0) @binding({}) var<storage, read> gs_params: array<u32>;",
        binding + 1
    ));
    if has_f64_params(k) {
        // Pas de bitcast vers f64 en naga : les scalaires f64 ont leur propre buffer.
        g.out.line(format!(
            "@group(0) @binding({}) var<storage, read> gs_fparams: array<f64>;",
            binding + 2
        ));
    }
    let used = helpers(k);
    for (h, code) in HELPERS {
        if used.contains(h) {
            g.out.line(*code);
        }
    }
    let n_scalars = k
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar && p.ty != Scalar::F64)
        .count();
    g.out
        .line(format!("@compute @workgroup_size({WGSL_WORKGROUP_SIZE})"));
    g.out.line("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {");
    g.out.depth = 1;
    g.out.line(format!(
        "let gs_i = gid.x + gid.y * (nwg.x * {WGSL_WORKGROUP_SIZE}u);"
    ));
    g.out
        .line(format!("if (gs_i >= gs_params[{n_scalars}]) {{ return; }}"));
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
                t => {
                    word += 1;
                    format!("bitcast<{}>(gs_params[{}])", w_type(t), word - 1)
                }
            };
            g.out.line(format!("let {} = {read};", param_name(k, i)));
        }
    }
    if let Grid::Indexed { axes } = &k.grid {
        // Coordonnées du thread dans la grille (forme de la sortie, ordre C).
        g.out.line("var gs_r = gs_i;");
        for (a, p) in axes.iter().enumerate().rev() {
            let size = param_name(k, *p);
            g.out
                .line(format!("let gs_g{a} = i32(gs_r % u32({size}));"));
            if a > 0 {
                g.out.line(format!("gs_r = gs_r / u32({size});"));
            }
        }
    }
    for (i, l) in k.locals.iter().enumerate() {
        match l.len {
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
    fn genere_un_module_wgsl() {
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
            c.contains("let p1_moyenne = bitcast<f32>(gs_params[0]);"),
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
    fn f64_arithmetique_accepte_mais_pas_les_fonctions_transcendantes() {
        let mut k = crate::tests::sample();
        // Tout en f64 : le sample utilise ** (pow) → refusé.
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
        // Sans le ** : accepté, avec le buffer gs_fparams et le % flottant f64.
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
        assert!(c.contains("let p1_moyenne = gs_fparams[0];"), "{c}");
        assert!(c.contains("gs_mod_f64(p0_x[gs_i], 3.0lf)"), "{c}");
        assert!(c.contains("if (gs_i >= gs_params[1]) { return; }"), "{c}");
    }
}
