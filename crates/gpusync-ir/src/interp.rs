//! Interpréteur de référence de l'IR (§6.3) : oracle des tests, mode debug, backend « cpu ».
//!
//! Exécute le kernel séquentiellement, élément par élément, sur les octets des tableaux (même
//! représentation que sur le GPU). Lent, mais toujours disponible et sans ambiguïté.

use crate::*;

/// Argument d'un kernel : octets d'un tableau (ordre natif) ou valeur scalaire.
pub enum Arg<'a> {
    Array(&'a [u8]),
    Scalar(Const),
}

enum Flow {
    Normal,
    Break,
    Continue,
    Return(Vec<Const>),
}

/// Exécute `kernel` sur une grille de `n` threads ; `outs[k]` reçoit `n` valeurs de la sortie `k`.
pub fn run(kernel: &Kernel, n: usize, args: &[Arg], outs: &mut [&mut [u8]]) -> Result<(), String> {
    if args.len() != kernel.params.len() {
        return Err(format!(
            "{} argument(s) attendu(s), {} reçu(s)",
            kernel.params.len(),
            args.len()
        ));
    }
    if outs.len() != kernel.outputs.len() {
        return Err(format!(
            "{} sortie(s) attendue(s), {} reçue(s)",
            kernel.outputs.len(),
            outs.len()
        ));
    }
    if kernel.grid == Grid::Elementwise {
        for (p, a) in kernel.params.iter().zip(args) {
            if let (ParamKind::Array, Arg::Array(bytes)) = (p.kind, a) {
                if bytes.len() < n * p.ty.size() {
                    return Err(format!("tableau '{}' trop court", p.name));
                }
            }
        }
    }
    for (o, t) in outs.iter().zip(&kernel.outputs) {
        if o.len() < n * t.size() {
            return Err("tableau de sortie trop court".into());
        }
    }
    // Taille des axes de la grille (kernels avec index).
    let axes: Vec<usize> = match &kernel.grid {
        Grid::Elementwise => vec![],
        Grid::Indexed { axes } => axes
            .iter()
            .map(|p| match args[*p] {
                Arg::Scalar(c) => match convert(c, Scalar::I32) {
                    Const::I32(v) => Ok(v.max(0) as usize),
                    _ => Err("dimension invalide".to_string()),
                },
                _ => Err("dimension invalide".to_string()),
            })
            .collect::<Result<_, _>>()?,
    };
    let mut locals: Vec<Const> = kernel.locals.iter().map(|l| zero(l.ty)).collect();
    let mut arrays: Vec<Vec<Const>> = kernel
        .locals
        .iter()
        .map(|l| vec![zero(l.ty); l.len.unwrap_or(0) as usize])
        .collect();
    for i in 0..n {
        for (a, l) in arrays.iter_mut().zip(&kernel.locals) {
            a.fill(zero(l.ty)); // tableaux locaux remis à zéro pour chaque thread
        }
        let mut coords = vec![0usize; axes.len()];
        let mut rest = i;
        for a in (0..axes.len()).rev() {
            coords[a] = rest % axes[a].max(1);
            rest /= axes[a].max(1);
        }
        let mut st = State {
            kernel,
            args,
            i,
            coords,
            locals: &mut locals,
            arrays: &mut arrays,
        };
        let values = match st.block(&kernel.body)? {
            Flow::Return(v) => v,
            _ => {
                return Err(format!(
                    "le kernel '{}' s'est terminé sans return",
                    kernel.name
                ));
            }
        };
        for ((v, o), t) in values.into_iter().zip(outs.iter_mut()).zip(&kernel.outputs) {
            store(v, &mut o[i * t.size()..(i + 1) * t.size()]);
        }
    }
    Ok(())
}

struct State<'a, 'b> {
    kernel: &'a Kernel,
    args: &'a [Arg<'a>],
    i: usize,
    coords: Vec<usize>,
    locals: &'b mut Vec<Const>,
    arrays: &'b mut Vec<Vec<Const>>,
}

fn index_of(c: Const) -> i64 {
    match c {
        Const::I32(v) => v as i64,
        Const::U32(v) => v as i64,
        _ => -1,
    }
}

impl State<'_, '_> {
    fn block(&mut self, block: &Block) -> Result<Flow, String> {
        for s in block {
            match self.stmt(s)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn stmt(&mut self, s: &Stmt) -> Result<Flow, String> {
        match s {
            Stmt::Assign { local, value } => {
                self.locals[*local] = self.eval(value)?;
                Ok(Flow::Normal)
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                if as_bool(self.eval(cond)?) {
                    self.block(then)
                } else {
                    self.block(otherwise)
                }
            }
            Stmt::For {
                var,
                start,
                stop,
                step,
                body,
            } => {
                let (start, stop, step) = (self.eval(start)?, self.eval(stop)?, self.eval(step)?);
                self.locals[*var] = start;
                loop {
                    let v = self.locals[*var];
                    let go = (gt(step, zero(step.ty())) && lt(v, stop))
                        || (lt(step, zero(step.ty())) && gt(v, stop));
                    if !go {
                        break;
                    }
                    match self.block(body)? {
                        Flow::Break => break,
                        Flow::Return(r) => return Ok(Flow::Return(r)),
                        Flow::Normal | Flow::Continue => {}
                    }
                    let v = self.locals[*var];
                    self.locals[*var] = arith(BinOp::Add, v, step);
                }
                Ok(Flow::Normal)
            }
            Stmt::While { cond, body } => {
                while as_bool(self.eval(cond)?) {
                    match self.block(body)? {
                        Flow::Break => break,
                        Flow::Return(r) => return Ok(Flow::Return(r)),
                        Flow::Normal | Flow::Continue => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
            Stmt::LocalStore {
                local,
                index,
                value,
            } => {
                let (idx, v) = (index_of(self.eval(index)?), self.eval(value)?);
                let len = self.arrays[*local].len();
                if idx < 0 || idx as usize >= len {
                    return Err(format!(
                        "indice {idx} hors limites pour le tableau local '{}' ({len} éléments)",
                        self.kernel.locals[*local].name
                    ));
                }
                self.arrays[*local][idx as usize] = convert(v, self.kernel.locals[*local].ty);
                Ok(Flow::Normal)
            }
            Stmt::Return(es) => {
                let mut vals = Vec::with_capacity(es.len());
                for (e, t) in es.iter().zip(&self.kernel.outputs) {
                    vals.push(convert(self.eval(e)?, *t));
                }
                Ok(Flow::Return(vals))
            }
        }
    }

    fn eval(&mut self, e: &Expr) -> Result<Const, String> {
        Ok(match &e.kind {
            ExprKind::GridIndex(a) => Const::I32(self.coords[*a] as i32),
            ExprKind::Load(p, index) => {
                let idx = index_of(self.eval(index)?);
                let Arg::Array(bytes) = &self.args[*p] else {
                    return Err("lecture indexée d'un scalaire".into());
                };
                let ty = self.kernel.params[*p].ty;
                let len = bytes.len() / ty.size();
                if idx < 0 || idx as usize >= len {
                    return Err(format!(
                        "indice {idx} hors limites pour '{}' ({len} éléments) — thread {}",
                        self.kernel.params[*p].name, self.i
                    ));
                }
                let o = idx as usize * ty.size();
                load(ty, &bytes[o..o + ty.size()])
            }
            ExprKind::LocalLoad(l, index) => {
                let idx = index_of(self.eval(index)?);
                let len = self.arrays[*l].len();
                if idx < 0 || idx as usize >= len {
                    return Err(format!(
                        "indice {idx} hors limites pour le tableau local '{}' ({len} éléments)",
                        self.kernel.locals[*l].name
                    ));
                }
                self.arrays[*l][idx as usize]
            }
            ExprKind::Const(c) => *c,
            ExprKind::Local(l) => self.locals[*l],
            ExprKind::Param(p) => match &self.args[*p] {
                Arg::Scalar(c) => convert(*c, self.kernel.params[*p].ty),
                Arg::Array(bytes) => {
                    let ty = self.kernel.params[*p].ty;
                    load(ty, &bytes[self.i * ty.size()..(self.i + 1) * ty.size()])
                }
            },
            ExprKind::Unary(op, a) => {
                let a = self.eval(a)?;
                match (op, a) {
                    (UnOp::Not, c) => Const::Bool(!as_bool(c)),
                    (UnOp::Neg, Const::I32(v)) => Const::I32(v.wrapping_neg()),
                    (UnOp::Neg, Const::U32(v)) => Const::U32(v.wrapping_neg()),
                    (UnOp::Neg, Const::F32(v)) => Const::F32(-v),
                    (UnOp::Neg, Const::F64(v)) => Const::F64(-v),
                    (UnOp::Neg, Const::Bool(_)) => return Err("négation d'un booléen".into()),
                }
            }
            ExprKind::Binary(op, a, b) => {
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                match op {
                    BinOp::And => Const::Bool(as_bool(a) && as_bool(b)),
                    BinOp::Or => Const::Bool(as_bool(a) || as_bool(b)),
                    _ => arith(*op, a, b),
                }
            }
            ExprKind::Compare(op, a, b) => {
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                Const::Bool(match op {
                    CmpOp::Eq => eq(a, b),
                    CmpOp::Ne => !eq(a, b),
                    CmpOp::Lt => lt(a, b),
                    CmpOp::Le => lt(a, b) || eq(a, b),
                    CmpOp::Gt => gt(a, b),
                    CmpOp::Ge => gt(a, b) || eq(a, b),
                })
            }
            ExprKind::Select(c, a, b) => {
                let (c, a, b) = (self.eval(c)?, self.eval(a)?, self.eval(b)?);
                if as_bool(c) { a } else { b }
            }
            ExprKind::Call(f, args) => {
                let vals = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>, _>>()?;
                intrinsic(*f, &vals)
            }
            ExprKind::Cast(a) => convert(self.eval(a)?, e.ty),
        })
    }
}

pub fn zero(ty: Scalar) -> Const {
    match ty {
        Scalar::Bool => Const::Bool(false),
        Scalar::I32 => Const::I32(0),
        Scalar::U32 => Const::U32(0),
        Scalar::F32 => Const::F32(0.0),
        Scalar::F64 => Const::F64(0.0),
    }
}

fn as_bool(c: Const) -> bool {
    match c {
        Const::Bool(b) => b,
        Const::I32(v) => v != 0,
        Const::U32(v) => v != 0,
        Const::F32(v) => v != 0.0,
        Const::F64(v) => v != 0.0,
    }
}

/// Conversion de type (flottant → entier : troncature vers 0, saturée).
pub fn convert(c: Const, to: Scalar) -> Const {
    let f = match c {
        Const::Bool(b) => b as i64 as f64,
        Const::I32(v) => v as f64,
        Const::U32(v) => v as f64,
        Const::F32(v) => v as f64,
        Const::F64(v) => v,
    };
    match (to, c) {
        (t, c) if c.ty() == t => c,
        (Scalar::Bool, c) => Const::Bool(as_bool(c)),
        (Scalar::I32, Const::U32(v)) => Const::I32(v as i32),
        (Scalar::U32, Const::I32(v)) => Const::U32(v as u32),
        (Scalar::F32, Const::F64(v)) => Const::F32(v as f32),
        (Scalar::F32, Const::I32(v)) => Const::F32(v as f32),
        (Scalar::F32, Const::U32(v)) => Const::F32(v as f32),
        (Scalar::I32, _) => Const::I32(f as i32),
        (Scalar::U32, _) => Const::U32(f as u32),
        (Scalar::F32, _) => Const::F32(f as f32),
        (Scalar::F64, _) => Const::F64(f),
    }
}

fn eq(a: Const, b: Const) -> bool {
    match (a, b) {
        (Const::Bool(x), Const::Bool(y)) => x == y,
        (Const::I32(x), Const::I32(y)) => x == y,
        (Const::U32(x), Const::U32(y)) => x == y,
        (Const::F32(x), Const::F32(y)) => x == y,
        (Const::F64(x), Const::F64(y)) => x == y,
        _ => false,
    }
}

fn lt(a: Const, b: Const) -> bool {
    match (a, b) {
        (Const::I32(x), Const::I32(y)) => x < y,
        (Const::U32(x), Const::U32(y)) => x < y,
        (Const::F32(x), Const::F32(y)) => x < y,
        (Const::F64(x), Const::F64(y)) => x < y,
        (Const::Bool(x), Const::Bool(y)) => !x & y,
        _ => false,
    }
}

fn gt(a: Const, b: Const) -> bool {
    lt(b, a)
}

fn arith(op: BinOp, a: Const, b: Const) -> Const {
    match (a, b) {
        (Const::I32(x), Const::I32(y)) => Const::I32(match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::FloorDiv | BinOp::Div => floordiv_i32(x, y),
            BinOp::Mod => mod_i32(x, y),
            BinOp::Pow | BinOp::And | BinOp::Or => 0,
        }),
        (Const::U32(x), Const::U32(y)) => Const::U32(match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::FloorDiv | BinOp::Div => x.checked_div(y).unwrap_or(0),
            BinOp::Mod => x.checked_rem(y).unwrap_or(0),
            BinOp::Pow | BinOp::And | BinOp::Or => 0,
        }),
        (Const::F32(x), Const::F32(y)) => Const::F32(match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::FloorDiv => (x / y).floor(),
            BinOp::Mod => mod_f32(x, y),
            BinOp::Pow => x.powf(y),
            BinOp::And | BinOp::Or => 0.0,
        }),
        (Const::F64(x), Const::F64(y)) => Const::F64(match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::FloorDiv => (x / y).floor(),
            BinOp::Mod => mod_f64(x, y),
            BinOp::Pow => x.powf(y),
            BinOp::And | BinOp::Or => 0.0,
        }),
        _ => a, // types incohérents : refusés par le frontend
    }
}

fn intrinsic(f: Intrinsic, v: &[Const]) -> Const {
    use Intrinsic::*;
    match (f, v) {
        (Abs, [Const::I32(x)]) => Const::I32(x.wrapping_abs()),
        (Abs, [Const::U32(x)]) => Const::U32(*x),
        (Min, [Const::I32(x), Const::I32(y)]) => Const::I32(*x.min(y)),
        (Max, [Const::I32(x), Const::I32(y)]) => Const::I32(*x.max(y)),
        (Min, [Const::U32(x), Const::U32(y)]) => Const::U32(*x.min(y)),
        (Max, [Const::U32(x), Const::U32(y)]) => Const::U32(*x.max(y)),
        (_, [Const::F32(x)]) => Const::F32(f32_1(f, *x)),
        (_, [Const::F64(x)]) => Const::F64(f64_1(f, *x)),
        (_, [Const::F32(x), Const::F32(y)]) => Const::F32(match f {
            Min => x.min(*y),
            Max => x.max(*y),
            Pow => x.powf(*y),
            _ => x.atan2(*y),
        }),
        (_, [Const::F64(x), Const::F64(y)]) => Const::F64(match f {
            Min => x.min(*y),
            Max => x.max(*y),
            Pow => x.powf(*y),
            _ => x.atan2(*y),
        }),
        _ => v.first().copied().unwrap_or(Const::F32(0.0)),
    }
}

macro_rules! unary_float {
    ($name:ident, $t:ty) => {
        fn $name(f: Intrinsic, x: $t) -> $t {
            use Intrinsic::*;
            match f {
                Sqrt => x.sqrt(),
                Exp => x.exp(),
                Exp2 => x.exp2(),
                Log => x.ln(),
                Log2 => x.log2(),
                Sin => x.sin(),
                Cos => x.cos(),
                Tan => x.tan(),
                Tanh => x.tanh(),
                Floor => x.floor(),
                Ceil => x.ceil(),
                Abs => x.abs(),
                _ => x,
            }
        }
    };
}
unary_float!(f32_1, f32);
unary_float!(f64_1, f64);

pub fn load(ty: Scalar, b: &[u8]) -> Const {
    match ty {
        Scalar::Bool => Const::Bool(b[0] != 0),
        Scalar::I32 => Const::I32(i32::from_ne_bytes(b[..4].try_into().unwrap())),
        Scalar::U32 => Const::U32(u32::from_ne_bytes(b[..4].try_into().unwrap())),
        Scalar::F32 => Const::F32(f32::from_ne_bytes(b[..4].try_into().unwrap())),
        Scalar::F64 => Const::F64(f64::from_ne_bytes(b[..8].try_into().unwrap())),
    }
}

pub fn store(c: Const, out: &mut [u8]) {
    match c {
        Const::Bool(v) => out[0] = v as u8,
        Const::I32(v) => out.copy_from_slice(&v.to_ne_bytes()),
        Const::U32(v) => out.copy_from_slice(&v.to_ne_bytes()),
        Const::F32(v) => out.copy_from_slice(&v.to_ne_bytes()),
        Const::F64(v) => out.copy_from_slice(&v.to_ne_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn ret(e: Expr) -> Stmt {
        Stmt::Return(vec![e])
    }

    fn e(ty: Scalar, kind: ExprKind) -> Expr {
        Expr { ty, kind }
    }

    /// `def k(x: f32[:], s: f32) -> f32[:]: acc = 0.0; for i in range(3): acc += x * s; return acc if x > 0 else -1.0`
    fn sample() -> Kernel {
        let x = || Box::new(e(Scalar::F32, ExprKind::Param(0)));
        let acc = || Box::new(e(Scalar::F32, ExprKind::Local(0)));
        Kernel {
            name: "k".into(),
            params: vec![
                Param {
                    name: "x".into(),
                    kind: ParamKind::Array,
                    ty: Scalar::F32,
                    dims: vec![],
                    hidden: false,
                },
                Param {
                    name: "s".into(),
                    kind: ParamKind::Scalar,
                    ty: Scalar::F32,
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
                    value: e(Scalar::F32, ExprKind::Const(Const::F32(0.0))),
                },
                Stmt::For {
                    var: 1,
                    start: e(Scalar::I32, ExprKind::Const(Const::I32(0))),
                    stop: e(Scalar::I32, ExprKind::Const(Const::I32(3))),
                    step: e(Scalar::I32, ExprKind::Const(Const::I32(1))),
                    body: vec![Stmt::Assign {
                        local: 0,
                        value: e(
                            Scalar::F32,
                            ExprKind::Binary(
                                BinOp::Add,
                                acc(),
                                Box::new(e(
                                    Scalar::F32,
                                    ExprKind::Binary(
                                        BinOp::Mul,
                                        x(),
                                        Box::new(e(Scalar::F32, ExprKind::Param(1))),
                                    ),
                                )),
                            ),
                        ),
                    }],
                },
                ret(e(
                    Scalar::F32,
                    ExprKind::Select(
                        Box::new(e(
                            Scalar::Bool,
                            ExprKind::Compare(
                                CmpOp::Gt,
                                x(),
                                Box::new(e(Scalar::F32, ExprKind::Const(Const::F32(0.0)))),
                            ),
                        )),
                        acc(),
                        Box::new(e(Scalar::F32, ExprKind::Const(Const::F32(-1.0)))),
                    ),
                )),
            ],
        }
    }

    #[test]
    fn execute_boucle_et_ternaire() {
        let k = sample();
        let xs: Vec<u8> = [1.0f32, -2.0, 0.5]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let mut out = [0u8; 12];
        run(
            &k,
            3,
            &[Arg::Array(&xs), Arg::Scalar(Const::F32(2.0))],
            &mut [&mut out[..]],
        )
        .unwrap();
        let got: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![6.0, -1.0, 3.0]);
        assert!(!k.requirements().f64);
    }

    #[test]
    fn range_comme_python() {
        // range(5, 0, -2) → 5, 3, 1 ; pas nul → aucune itération (au lieu d'une boucle infinie).
        for (start, stop, step, expected) in
            [(5, 0, -2, 3), (0, 5, 1, 5), (0, 5, 0, 0), (3, 3, 1, 0)]
        {
            let c = |v| e(Scalar::I32, ExprKind::Const(Const::I32(v)));
            let k = Kernel {
                name: "r".into(),
                params: vec![],
                outputs: vec![Scalar::I32],
                grid: Grid::Elementwise,
                locals: vec![
                    Local {
                        name: "n".into(),
                        ty: Scalar::I32,
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
                        value: c(0),
                    },
                    Stmt::For {
                        var: 1,
                        start: c(start),
                        stop: c(stop),
                        step: c(step),
                        body: vec![Stmt::Assign {
                            local: 0,
                            value: e(
                                Scalar::I32,
                                ExprKind::Binary(
                                    BinOp::Add,
                                    Box::new(e(Scalar::I32, ExprKind::Local(0))),
                                    Box::new(c(1)),
                                ),
                            ),
                        }],
                    },
                    ret(e(Scalar::I32, ExprKind::Local(0))),
                ],
            };
            let mut out = [0u8; 4];
            run(&k, 1, &[], &mut [&mut out[..]]).unwrap();
            assert_eq!(
                i32::from_ne_bytes(out),
                expected,
                "range({start}, {stop}, {step})"
            );
        }
    }
}
