//! Barriers of a kernel with `group=`: every thread of a group must reach the same
//! `gs.barrier()`, otherwise the GPU hangs (CUDA, OpenCL) or rejects the kernel (WGSL, uniformity
//! analysis). A "thread-dependent" value comes from `gs.index()`, `gs.lane()`, a read of shared
//! memory, or a computation that depends on them; a barrier cannot depend on it (condition,
//! loop bounds, loop exit, `return` taken by only some of the threads).

use gpusync_ir::*;

use crate::{ErrorKind, FrontendError, err};

type R<T> = Result<T, FrontendError>;

struct Taint<'a> {
    k: &'a Kernel,
    /// Local (or local array) whose value may differ from one thread to another.
    varying: Vec<bool>,
    changed: bool,
}

impl Taint<'_> {
    fn expr(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Const(_) => false,
            ExprKind::GridIndex(_) | ExprKind::Lane => true,
            ExprKind::Local(l) => self.varying[*l],
            ExprKind::Param(p) => self.k.params[*p].kind == ParamKind::Array,
            ExprKind::LocalLoad(l, i) => {
                self.k.locals[*l].shared || self.varying[*l] || self.expr(i)
            }
            ExprKind::Load(_, i) | ExprKind::Unary(_, i) | ExprKind::Cast(i) => self.expr(i),
            ExprKind::Binary(_, a, b) | ExprKind::Compare(_, a, b) => self.expr(a) || self.expr(b),
            ExprKind::Select(c, a, b) => self.expr(c) || self.expr(a) || self.expr(b),
            ExprKind::Call(_, args) => args.iter().any(|a| self.expr(a)),
        }
    }

    fn mark(&mut self, l: usize) {
        if !self.varying[l] {
            self.varying[l] = true;
            self.changed = true;
        }
    }

    /// Can the loop body leave it (break, continue, return) on a thread-dependent
    /// condition? Later iterations then only concern part of the group.
    fn varying_exit(&self, b: &Block, cv: bool) -> bool {
        b.iter().any(|s| match s {
            Stmt::Break | Stmt::Continue | Stmt::Return(_) => cv,
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let c = cv || self.expr(cond);
                self.varying_exit(then, c) || self.varying_exit(otherwise, c)
            }
            // A break in an inner loop does not leave it; a return does.
            Stmt::For { .. } | Stmt::While { .. } => {
                returns_under(std::slice::from_ref(s), cv, self)
            }
            _ => false,
        })
    }

    fn loop_cv(&self, cv: bool, bounds: &[&Expr], body: &Block) -> bool {
        let c = cv || bounds.iter().any(|e| self.expr(e));
        c || self.varying_exit(body, c)
    }

    fn walk(&mut self, b: &Block, cv: bool) {
        for s in b {
            match s {
                Stmt::Assign { local, value } => {
                    if cv || self.expr(value) {
                        self.mark(*local);
                    }
                }
                Stmt::LocalStore {
                    local,
                    index,
                    value,
                } => {
                    if cv || self.expr(index) || self.expr(value) {
                        self.mark(*local);
                    }
                }
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let c = cv || self.expr(cond);
                    self.walk(then, c);
                    self.walk(otherwise, c);
                }
                Stmt::For {
                    var,
                    start,
                    stop,
                    step,
                    body,
                } => {
                    let c = self.loop_cv(cv, &[start, stop, step], body);
                    if c {
                        self.mark(*var);
                    }
                    self.walk(body, c);
                }
                Stmt::While { cond, body } => {
                    let c = self.loop_cv(cv, &[cond], body);
                    self.walk(body, c);
                }
                _ => {}
            }
        }
    }
}

fn returns_under(b: &[Stmt], cv: bool, t: &Taint) -> bool {
    b.iter().any(|s| match s {
        Stmt::Return(_) => cv,
        Stmt::If {
            cond,
            then,
            otherwise,
        } => {
            let c = cv || t.expr(cond);
            returns_under(then, c, t) || returns_under(otherwise, c, t)
        }
        Stmt::For {
            start,
            stop,
            step,
            body,
            ..
        } => returns_under(body, t.loop_cv(cv, &[start, stop, step], body), t),
        Stmt::While { cond, body } => returns_under(body, t.loop_cv(cv, &[cond], body), t),
        _ => false,
    })
}

struct Check<'a, 'b> {
    t: &'a Taint<'b>,
    lines: &'a [Option<u32>],
    next: usize,
    /// A thread-dependent `return` has already happened: part of the group has finished.
    diverged: bool,
}

impl Check<'_, '_> {
    fn walk(&mut self, b: &Block, cv: bool) -> R<()> {
        for s in b {
            match s {
                Stmt::Barrier | Stmt::MatMul { .. } => {
                    let line = self.lines.get(self.next).copied().flatten();
                    self.next += 1;
                    if self.diverged {
                        return Err(err(
                            ErrorKind::Syntax,
                            "gs.barrier() after a `return` that only some threads of the group take: \
                             the others would wait forever. Keep the result in a variable and end \
                             the kernel with a single return"
                                .into(),
                            line,
                        ));
                    }
                    if cv {
                        return Err(err(
                            ErrorKind::Syntax,
                            "gs.barrier() inside a condition or a loop that depends on the thread (gs.index(), \
                             gs.lane(), a read of gs.shared, or anything computed from them): some \
                             threads of the group would not reach it. Move the barrier out of that condition, \
                             or make the condition the same for the whole group"
                                .into(),
                            line,
                        ));
                    }
                }
                Stmt::Return(_) if cv => self.diverged = true,
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let c = cv || self.t.expr(cond);
                    self.walk(then, c)?;
                    self.walk(otherwise, c)?;
                }
                Stmt::For {
                    start,
                    stop,
                    step,
                    body,
                    ..
                } => {
                    let c = self.t.loop_cv(cv, &[start, stop, step], body);
                    self.walk(body, c)?;
                }
                Stmt::While { cond, body } => {
                    let c = self.t.loop_cv(cv, &[cond], body);
                    self.walk(body, c)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Checks that every `gs.barrier()` is reached by all threads of the group. `lines`: line of
/// each barrier, in code order.
pub(crate) fn check_barriers(k: &Kernel, lines: &[Option<u32>]) -> R<()> {
    let mut t = Taint {
        k,
        varying: vec![false; k.locals.len()],
        changed: true,
    };
    while t.changed {
        t.changed = false;
        t.walk(&k.body, false);
    }
    Check {
        t: &t,
        lines,
        next: 0,
        diverged: false,
    }
    .walk(&k.body, false)
}
