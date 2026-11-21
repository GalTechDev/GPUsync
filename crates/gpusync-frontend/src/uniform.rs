//! Barrières d'un kernel avec `group=` : tous les threads d'un groupe doivent atteindre les mêmes
//! `gs.barrier()`, sinon le GPU se bloque (CUDA, OpenCL) ou refuse le kernel (WGSL, analyse
//! d'uniformité). Une valeur « propre au thread » vient de `gs.index()`, `gs.lane()`, d'une lecture de
//! mémoire partagée, ou d'un calcul qui en dépend ; une barrière ne peut pas dépendre d'elle (condition,
//! bornes de boucle, sortie de boucle, `return` exécuté par une partie des threads seulement).

use gpusync_ir::*;

use crate::{ErrorKind, FrontendError, err};

type R<T> = Result<T, FrontendError>;

struct Taint<'a> {
    k: &'a Kernel,
    /// Locale (ou tableau local) dont la valeur peut différer d'un thread à l'autre.
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

    /// Le corps d'une boucle peut-il en sortir (break, continue, return) sur une condition propre
    /// au thread ? Les tours suivants ne concernent alors qu'une partie du groupe.
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
            // Un break d'une boucle intérieure n'en sort pas ; un return, si.
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
    /// Un `return` propre au thread a déjà eu lieu : une partie du groupe est terminée.
    diverged: bool,
}

impl Check<'_, '_> {
    fn walk(&mut self, b: &Block, cv: bool) -> R<()> {
        for s in b {
            match s {
                Stmt::Barrier => {
                    let line = self.lines.get(self.next).copied().flatten();
                    self.next += 1;
                    if self.diverged {
                        return Err(err(
                            ErrorKind::Syntax,
                            "gs.barrier() après un `return` que seule une partie des threads du groupe exécute : \
                             les autres attendraient sans fin. Gardez le résultat dans une variable et terminez \
                             le kernel par un seul return"
                                .into(),
                            line,
                        ));
                    }
                    if cv {
                        return Err(err(
                            ErrorKind::Syntax,
                            "gs.barrier() dans une condition ou une boucle qui dépend du thread (gs.index(), \
                             gs.lane(), une lecture de gs.shared, ou un calcul qui en dépend) : une partie des \
                             threads du groupe ne l'atteindrait pas. Placez la barrière hors de cette condition, \
                             ou rendez la condition identique pour tout le groupe"
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

/// Vérifie que chaque `gs.barrier()` est atteinte par tous les threads du groupe. `lines` : ligne de
/// chaque barrière, dans l'ordre du code.
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
