"""Large-scale atmosphere: hydrostatic primitive equations, sigma coordinate, prognostic surface pressure.

C grid: `u` on the west faces, `v` on the south faces of cell `(k, j, i)`, the rest at the centre. `k = 0` is
the top layer. x is periodic, y is bounded by two free-slip walls. `pi = p_surface - p_top` equals
`pi0 + pip`: the reference value `pi0` (terrain) stays fixed, the deviation `pip` keeps single-precision accuracy.

Columns are stored `(j, block, c, rank)` with `i = block · group + rank` (one thread per column)."""

import gpusync as gs

CP = 1004.64
RD = 287.04
LV = 2.501e6
GRAVITY = 9.81
KAPPA = RD / CP
T_REF = 288.0  # reference atmosphere of the pressure gradient: T = T_REF (p / 1000 hPa)^BETA
BETA = 0.1902
P_TOP = 5000.0
GROUP = 16  # threads per group: the domain width is a multiple of it
LEVELS_MAX = 64


# ───────────────────────── Dynamique ─────────────────────────


def _dyn_columns(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    tpl: gs.f32["ny", "nb", "cc", "g"],
) -> gs.f32["ny", "nb", "cc", "g"]:
    """Per column: geopotential deviation, vertical velocity `pi · dσ/dt` at the bottom of each layer, tendency of `pi`."""
    j, b, c0, r = gs.index()
    i = b * 16 + r
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    div = gs.local(gs.f32, 64)
    phi = gs.local(gs.f32, 64)
    ie = (i + 1) % nx
    iw = (i - 1) % nx
    jn = gs.min(j + 1, ny - 1)
    js = gs.max(j - 1, 0)
    pc = pi0[j, i] + pip[j, i]
    pe = 0.5 * (pc + pi0[j, ie] + pip[j, ie]) * inv_dx
    pw = 0.5 * (pc + pi0[j, iw] + pip[j, iw]) * inv_dx
    pn = 0.5 * (pc + pi0[jn, i] + pip[jn, i]) * inv_dy if j < ny - 1 else 0.0
    ps = 0.5 * (pc + pi0[js, i] + pip[js, i]) * inv_dy if j > 0 else 0.0
    total = 0.0
    for k in range(nz):
        d = u[k, j, ie] * pe - u[k, j, i] * pw + v[k, jn, i] * pn - v[k, j, i] * ps
        div[k] = d
        total += d * lev[k, 2]
    acc = 0.0
    for k in range(nz):
        acc += div[k] * lev[k, 2]
        gs.store(nz + k, lev[k + 1, 0] * total - acc if k < nz - 1 else 0.0)
    gs.store(2 * nz, -total)
    # Hydrostatics as a deviation from the reference atmosphere: series of ((1 + x)^β - 1), x = relative deviation of p at the ground
    p_s0 = pi0[j, i] + 5000.0
    x = pip[j, i] / p_s0
    grow = 0.1902 * x * (1.0 - 0.4049 * x * (1.0 - 0.6032667 * x * (1.0 - 0.70245 * x)))
    ph = 287.04 * 288.0 / 0.1902 * gs.pow(p_s0 * 1e-5, 0.1902) * grow
    for m in range(nz):
        k = nz - 1 - m
        p_mid = 5000.0 + lev[k, 1] * pc
        p_low = 5000.0 + lev[k + 1, 0] * pc
        p_up = 5000.0 + lev[k, 0] * pc
        tv = th[k, j, i] * gs.pow(p_mid * 1e-5, 0.285714) * (1.0 + 0.608 * qv[k, j, i] - qc[k, j, i])
        dtv = 287.04 * (tv - 288.0 * gs.pow(p_mid * 1e-5, 0.1902))
        ph += dtv * gs.log(p_low / p_mid)
        phi[k] = ph
        ph += dtv * gs.log(p_mid / p_up)
    for k in range(nz):
        gs.store(k, phi[k])


@gs.kernel
def wx_momentum(
    base: gs.f32["nz", "ny", "nx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    col: gs.f32["ny", "nb", "cc", "g"],
    pip: gs.f32["ny", "nx"],
    pip_old: gs.f32["ny", "nx"],
    pip_new: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    coriolis: gs.f32["ny"],
    comp: gs.i32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    dt: gs.f32,
    damp: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Wind at the next stage (flux form, order-5 upwind horizontally and order-3 vertically): `comp` 0 for u, 1 for v."""
    k, j, i = gs.index()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    if comp == 1 and j == 0:
        return 0.0
    # The other cell of the face
    jo = j
    io = i
    if comp == 0:
        io = (i - 1) % nx
    else:
        jo = j - 1
    q = gs.local(gs.f32, 6)
    pa = pi0[j, i] + pip[j, i]
    pb = pi0[jo, io] + pip[jo, io]
    pm = 0.5 * (pa + pb)
    pm_old = 0.5 * (pi0[j, i] + pip_old[j, i] + pi0[jo, io] + pip_old[jo, io])
    pm_new = 0.5 * (pi0[j, i] + pip_new[j, i] + pi0[jo, io] + pip_new[jo, io])
    total = 0.0
    for ax in range(2):
        for side in range(2):
            # Mass flux through the face of the staggered cell: mean of the two cells
            fi = i
            fj = j
            flow = 0.0
            for s in range(2):
                cj = j
                ci = i
                if s == 1:
                    cj = jo
                    ci = io
                if ax == 0:
                    ci = (ci + side) % nx
                    cw = (ci - 1) % nx
                    flow += 0.25 * u[k, cj, ci] * (pi0[cj, ci] + pip[cj, ci] + pi0[cj, cw] + pip[cj, cw])
                else:
                    cj = cj + side
                    if cj > 0 and cj < ny:
                        flow += 0.25 * v[k, cj, ci] * (pi0[cj, ci] + pip[cj, ci] + pi0[cj - 1, ci] + pip[cj - 1, ci])
            flow = flow * (inv_dx if ax == 0 else inv_dy)
            fi = (i + side) % nx if ax == 0 else i
            fj = j + side if ax == 1 else j
            for m in range(6):
                val = 0.0
                if comp == 0:
                    if ax == 0:
                        val = u[k, j, (fi + m - 3) % nx]
                    else:
                        val = u[k, gs.min(gs.max(fj + m - 3, 0), ny - 1), i]
                else:
                    if ax == 0:
                        val = v[k, j, (fi + m - 3) % nx]
                    else:
                        fl = gs.min(gs.max(fj + m - 3, 0), ny)
                        val = v[k, fl, i] if fl < ny else 0.0
                q[m] = val
            even = 37.0 * (q[2] + q[3]) - 8.0 * (q[1] + q[4]) + q[0] + q[5]
            odd = 10.0 * (q[3] - q[2]) - 5.0 * (q[4] - q[1]) + q[5] - q[0]
            f = (flow * even - gs.abs(flow) * odd) * 0.016666667
            if side == 0:
                total += f
            else:
                total -= f
    for side in range(2):
        kk = k - 1 + side  # bottom of layer kk
        if kk >= 0 and kk < nz - 1:
            om = 0.5 * (col[j, i // 16, nz + kk, i % 16] + col[jo, io // 16, nz + kk, io % 16])
            for m in range(4):
                kl = gs.min(gs.max(kk - 1 + m, 0), nz - 1)
                q[m] = u[kl, j, i] if comp == 0 else v[kl, j, i]
            even = 7.0 * (q[1] + q[2]) - q[0] - q[3]
            odd = 3.0 * (q[2] - q[1]) - (q[3] - q[0])
            f = (om * even - gs.abs(om) * odd) * 0.083333333 / lev[k, 2]
            if side == 0:
                total += f
            else:
                total -= f
    # Pressure gradient, as a deviation from the reference atmosphere
    inv_d = inv_dx if comp == 0 else inv_dy
    sg = lev[k, 1]
    p_a = 5000.0 + sg * pa
    p_b = 5000.0 + sg * pb
    ta = th[k, j, i] * gs.pow(p_a * 1e-5, 0.285714) * (1.0 + 0.608 * qv[k, j, i] - qc[k, j, i]) - 288.0 * gs.pow(p_a * 1e-5, 0.1902)
    tb = th[k, jo, io] * gs.pow(p_b * 1e-5, 0.285714) * (1.0 + 0.608 * qv[k, jo, io] - qc[k, jo, io]) - 288.0 * gs.pow(p_b * 1e-5, 0.1902)
    dpi = pi0[j, i] - pi0[jo, io] + (pip[j, i] - pip[jo, io])
    grad = col[j, i // 16, k, i % 16] - col[jo, io // 16, k, io % 16]
    grad += 0.5 * 287.04 * (ta / p_a + tb / p_b) * sg * dpi
    acc = -grad * inv_d
    # Coriolis (mean of the four neighbouring winds) and divergence damping
    if comp == 0:
        jn = j + 1
        vn = 0.0
        if jn < ny:
            vn = v[k, jn, i] + v[k, jn, io]
        acc += coriolis[j] * 0.25 * (v[k, j, i] + v[k, j, io] + vn)
    else:
        ie = (i + 1) % nx
        acc -= 0.5 * (coriolis[j] + coriolis[jo]) * 0.25 * (u[k, j, i] + u[k, j, ie] + u[k, jo, i] + u[k, jo, ie])
    da = (u[k, j, (i + 1) % nx] - u[k, j, i]) * inv_dx - v[k, j, i] * inv_dy
    if j < ny - 1:
        da += v[k, j + 1, i] * inv_dy
    db = (u[k, jo, (io + 1) % nx] - u[k, jo, io]) * inv_dx + (v[k, jo + 1, io] * inv_dy if jo < ny - 1 else 0.0) - v[k, jo, io] * inv_dy
    acc += damp * (da - db) * inv_d
    # Sponge layer at the top: horizontal diffusion
    sponge = lev[k, 3]
    if sponge > 0.0:
        if comp == 0:
            lap = u[k, j, (i + 1) % nx] + u[k, j, (i - 1) % nx] + u[k, gs.min(j + 1, ny - 1), i] + u[k, gs.max(j - 1, 0), i] - 4.0 * u[k, j, i]
        else:
            lap = v[k, j, (i + 1) % nx] + v[k, j, (i - 1) % nx] + (v[k, j + 1, i] if j < ny - 1 else 0.0) + v[k, j - 1, i] - 4.0 * v[k, j, i]
        acc += sponge * lap
    return (base[k, j, i] * pm_old + dt * (total + pm * acc)) / pm_new


@gs.kernel
def wx_scalar(
    base: gs.f32["nz", "ny", "nx"],
    phi: gs.f32["nz", "ny", "nx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    col: gs.f32["ny", "nb", "cc", "g"],
    pip: gs.f32["ny", "nx"],
    pip_old: gs.f32["ny", "nx"],
    pip_new: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    dt: gs.f32,
    limit: gs.i32,
) -> gs.f32["nz", "ny", "nx"]:
    """Scalar at the next stage (flux form). Horizontal: order-5 upwind, bounded (MP) if `limit`. Vertical: order-3
    upwind, or van Leer if `limit`."""
    k, j, i = gs.index()
    nz = phi.shape[0]
    ny = phi.shape[1]
    nx = phi.shape[2]
    q = gs.local(gs.f32, 5)
    total = 0.0
    for ax in range(3):
        for side in range(2):
            fi = i
            fj = j
            fk = k + side  # top of layer fk
            flow = 0.0
            scale = inv_dx
            if ax == 0:
                fi = (i + side) % nx
                w = (fi - 1) % nx
                flow = 0.5 * u[k, j, fi] * (pi0[j, fi] + pip[j, fi] + pi0[j, w] + pip[j, w])
            elif ax == 1:
                scale = inv_dy
                fj = j + side
                if fj > 0 and fj < ny:
                    flow = 0.5 * v[k, fj, i] * (pi0[fj, i] + pip[fj, i] + pi0[fj - 1, i] + pip[fj - 1, i])
            else:
                scale = 1.0 / lev[k, 2]
                if fk > 0 and fk < nz:
                    flow = col[j, i // 16, nz + fk - 1, i % 16]
            if flow != 0.0:
                for m in range(5):
                    off = m - 3 if flow > 0.0 else 2 - m
                    val = 0.0
                    if ax == 0:
                        val = phi[k, j, (fi + off) % nx]
                    elif ax == 1:
                        val = phi[k, gs.min(gs.max(fj + off, 0), ny - 1), i]
                    else:
                        val = phi[gs.min(gs.max(fk + off, 0), nz - 1), j, i]
                    q[m] = val
                up = q[2]
                m1 = q[1] - up
                p1 = q[3] - up
                f = 0.0
                if ax == 2:
                    if limit == 0:
                        f = (2.0 * p1 - m1) * 0.16666667
                    elif m1 * p1 < 0.0:
                        f = -m1 * p1 / (p1 - m1)  # van Leer: harmonic mean of the slopes / 2
                else:
                    m2 = q[0] - up
                    p2 = q[4] - up
                    f = (2.0 * m2 - 13.0 * m1 + 27.0 * p1 - 3.0 * p2) * 0.016666667
                    if limit != 0:
                        s4 = -2.0 * m1
                        f_mp = 0.0
                        if p1 * s4 > 0.0:
                            f_mp = p1 if gs.abs(p1) < gs.abs(s4) else s4
                        if f * (f - f_mp) > 0.0:
                            d_m = m2 - 2.0 * m1
                            d_0 = m1 + p1
                            d_p = p2 - 2.0 * p1
                            cur_p = 0.0
                            if d_0 > 0.0 and d_p > 0.0:
                                cur_p = gs.min(d_0, d_p)
                            elif d_0 < 0.0 and d_p < 0.0:
                                cur_p = gs.max(d_0, d_p)
                            cur_m = 0.0
                            if d_0 > 0.0 and d_m > 0.0:
                                cur_m = gs.min(d_0, d_m)
                            elif d_0 < 0.0 and d_m < 0.0:
                                cur_m = gs.max(d_0, d_m)
                            f_md = 0.5 * p1 - 0.5 * cur_p
                            f_lc = -0.5 * m1 + 1.3333333 * cur_m
                            f_lo = gs.max(gs.min(0.0, p1, f_md), gs.min(0.0, s4, f_lc))
                            f_hi = gs.min(gs.max(0.0, p1, f_md), gs.max(0.0, s4, f_lc))
                            f = gs.min(gs.max(f, f_lo), f_hi)
                flux = flow * (up + f) * scale
                if side == 0:
                    total += flux
                else:
                    total -= flux
    p_old = pi0[j, i] + pip_old[j, i]
    p_new = pi0[j, i] + pip_new[j, i]
    return (base[k, j, i] * p_old + dt * total) / p_new


@gs.kernel
def wx_pressure(
    pip: gs.f32["ny", "nx"], col: gs.f32["ny", "nb", "cc", "g"], nz: gs.i32, dt: gs.f32
) -> gs.f32["ny", "nx"]:
    j, i = gs.index()
    return pip[j, i] + dt * col[j, i // 16, 2 * nz, i % 16]


@gs.kernel
def wx_unpack(col: gs.f32["ny", "nb", "cc", "g"], tpl: gs.f32["nz", "ny", "nx"], field: gs.i32) -> gs.f32["nz", "ny", "nx"]:
    """Field number `field` of a packet of columns."""
    k, j, i = gs.index()
    return col[j, i // 16, field * tpl.shape[0] + k, i % 16]


@gs.kernel
def wx_unpack_surface(col: gs.f32["ny", "nb", "cc", "g"], tpl: gs.f32["ny", "nx"], c: gs.i32) -> gs.f32["ny", "nx"]:
    j, i = gs.index()
    return col[j, i // 16, c, i % 16]


_CACHE: dict = {}


def column_kernel(func, cells: int):
    """Kernel `func` with one thread per column and `cells` values per column (compiled once per size)."""
    key = (func.__name__, cells)
    if key not in _CACHE:
        _CACHE[key] = gs.kernel(func, group=GROUP, cells=cells)
    return _CACHE[key]
