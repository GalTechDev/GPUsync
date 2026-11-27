"""Moist atmosphere at cloud scale (anelastic LES).

C grid: `u`, `v`, `w` are on the lower faces of each cell `(k, j, i)`; x and y are periodic,
the ground and the top are rigid (`w[0] = 0`, and `w` is zero above the last cell).

Scalars: `s` = liquid water static energy / cp (K, up to a constant), `qt` = vapour + cloud water,
`qr` = rain. `s` and `qt` are conserved during condensation; temperature and cloud water are
derived from them by saturation adjustment."""

import gpusync as gs

CP = 1004.64
RD = 287.04
RV = 461.5
LV = 2.501e6
GRAVITY = 9.81

# Columns of the reference table `(nz, 8)`
REF_Z, REF_P, REF_T, REF_QV, REF_QC, REF_RHO, REF_RHO_F, REF_EXNER = range(8)


# ───────────────────────── Thermodynamique ─────────────────────────


@gs.kernel
def atm_thermo(
    s: gs.f32["nz", "ny", "nx"],
    qt: gs.f32["nz", "ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    qs: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    s_ref: gs.f32,
    ice: gs.f32,
    fusion: gs.f32,
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    """Saturation adjustment: `(temperature, cloud water, buoyancy)`.

    `s = T + g z / cp - Lv (qc_liquid + qr) / cp - Ls (qc_ice + qs) / cp`: find T such that `qc = max(qt - qs(T), 0)`
    (Newton). Bolton (1980) saturation vapour pressure; pressure of the reference state. `ice` = 1: mixed water-ice
    saturation between 0 and -23 °C, pure ice below; the liquid part of the cloud is the weight `f` of the liquid in that
    saturation. `fusion` = 1: the ice (cloud and snow) holds the latent heat of fusion; 0: everything counted as liquid."""
    k, j, i = gs.index()
    qr_c = gs.max(qr[k, j, i], 0.0)
    qs_c = gs.max(qs[k, j, i], 0.0)
    q_tot = gs.max(qt[k, j, i], 0.0)
    p = ref[k, 1]
    t_dry = s[k, j, i] + s_ref - 9.81 / 1004.64 * ref[k, 0] + 2489.4489 * qr_c + (2489.4489 + 332.1588 * fusion) * qs_c
    t = t_dry
    qc = 0.0
    es = 611.2 * gs.exp(17.67 * (t - 273.15) / (t - 29.65))
    if ice > 0.0 and t < 273.15:
        mix = gs.max((t - 250.16) / 23.0, 0.0)
        mix = gs.min(mix * mix, 1.0)
        es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (t - 273.15) / (t - 0.55))
    qsat = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
    if q_tot > qsat:
        for _ in range(6):
            es = 611.2 * gs.exp(17.67 * (t - 273.15) / (t - 29.65))
            des = es * 4302.645 / ((t - 29.65) * (t - 29.65))
            liq = 1.0
            dliq = 0.0
            if ice > 0.0 and t < 273.15:
                mix = gs.max((t - 250.16) / 23.0, 0.0)
                if mix < 1.0:
                    dliq = 2.0 * mix / 23.0
                mix = gs.min(mix * mix, 1.0)
                liq = mix
                ei = 611.2 * gs.exp(22.46 * (t - 273.15) / (t - 0.55))
                des = des * mix + (1.0 - mix) * ei * 6122.6 / ((t - 0.55) * (t - 0.55)) + dliq * (es - ei)
                es = es * mix + (1.0 - mix) * ei
            qsat = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
            dqs = qsat / es * des * p / gs.max(p - 0.37803 * es, 1.0)
            cond = q_tot - qsat
            f = t - t_dry - 2489.4489 * cond - 332.1588 * fusion * (1.0 - liq) * cond
            slope = 1.0 + 2489.4489 * dqs + 332.1588 * fusion * ((1.0 - liq) * dqs + dliq * cond)
            t = t - f / slope
        es = 611.2 * gs.exp(17.67 * (t - 273.15) / (t - 29.65))
        liq = 1.0
        if ice > 0.0 and t < 273.15:
            mix = gs.max((t - 250.16) / 23.0, 0.0)
            mix = gs.min(mix * mix, 1.0)
            liq = mix
            es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (t - 273.15) / (t - 0.55))
        qsat = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
        qc = gs.max(q_tot - qsat, 0.0)
        t = t_dry + 2489.4489 * qc + 332.1588 * fusion * (1.0 - liq) * qc
    qv = q_tot - qc
    b = 9.81 * ((t - ref[k, 2]) / ref[k, 2] + 0.60778 * (qv - ref[k, 3]) - (qc - ref[k, 4]) - qr_c - qs_c)
    return t, qc, b


# ───────────────────────── Advection ─────────────────────────


@gs.kernel
def atm_scalar_stage(
    base: gs.f32["nz", "ny", "nx"],
    phi: gs.f32["nz", "ny", "nx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    slow: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    dt: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """`base + dt (advection of phi + slow)`: flux form, limited order-9 upwind reconstruction
    (MP of Suresh & Huynh 1997: overshoots of at most 2 % at the Courant number used), computed as deviations from the upwind value to keep precision."""
    k, j, i = gs.index()
    nz = phi.shape[0]
    ny = phi.shape[1]
    nx = phi.shape[2]
    q = gs.local(gs.f32, 9)
    total = 0.0
    for ax in range(3):
        for side in range(2):
            # Lower face of the cell (side 0) or of the next one (side 1) along the axis
            fi = i
            fj = j
            fk = k
            vel = 0.0
            inv_d = inv_dx
            if ax == 0:
                fi = (i + side) % nx
                vel = u[k, j, fi] * ref[k, 5]
            elif ax == 1:
                inv_d = inv_dy
                fj = (j + side) % ny
                vel = v[k, fj, i] * ref[k, 5]
            else:
                inv_d = inv_dz
                fk = k + side
                if fk > 0 and fk < nz:
                    vel = w[fk, j, i] * ref[fk, 6]
            if vel != 0.0:
                for m in range(9):
                    off = m - 5
                    if vel < 0.0:
                        off = 4 - m
                    val = 0.0
                    if ax == 0:
                        val = phi[k, j, (fi + off) % nx]
                    elif ax == 1:
                        val = phi[k, (fj + off) % ny, i]
                    else:
                        val = phi[gs.min(gs.max(fk + off, 0), nz - 1), j, i]
                    q[m] = val
                up = q[4]
                m1 = q[3] - up
                m2 = q[2] - up
                p1 = q[5] - up
                p2 = q[6] - up
                f = (
                    4.0 * (q[0] - up) - 41.0 * (q[1] - up) + 199.0 * m2 - 641.0 * m1 + 1375.0 * p1 - 305.0 * p2 + 55.0 * (q[7] - up) - 5.0 * (q[8] - up)
                ) * 3.968254e-4
                # Borne monotone : q0 + minmod(p1, 4 (q0 - m1))
                s4 = -4.0 * m1
                f_mp = 0.0
                if p1 * s4 > 0.0:
                    f_mp = p1 if gs.abs(p1) < gs.abs(s4) else s4
                if f * (f - f_mp) > 0.0:
                    d_m = m2 - 2.0 * m1
                    d_0 = m1 + p1
                    d_p = p2 - 2.0 * p1
                    # minmod of four curvatures, on both sides
                    cur_p = 0.0
                    a1 = 4.0 * d_0 - d_p
                    a2 = 4.0 * d_p - d_0
                    if a1 > 0.0 and a2 > 0.0 and d_0 > 0.0 and d_p > 0.0:
                        cur_p = gs.min(a1, a2, d_0, d_p)
                    elif a1 < 0.0 and a2 < 0.0 and d_0 < 0.0 and d_p < 0.0:
                        cur_p = gs.max(a1, a2, d_0, d_p)
                    cur_m = 0.0
                    a1 = 4.0 * d_0 - d_m
                    a2 = 4.0 * d_m - d_0
                    if a1 > 0.0 and a2 > 0.0 and d_0 > 0.0 and d_m > 0.0:
                        cur_m = gs.min(a1, a2, d_0, d_m)
                    elif a1 < 0.0 and a2 < 0.0 and d_0 < 0.0 and d_m < 0.0:
                        cur_m = gs.max(a1, a2, d_0, d_m)
                    f_md = 0.5 * p1 - 0.5 * cur_p
                    f_lc = -0.5 * m1 + 1.3333333 * cur_m
                    f_lo = gs.max(gs.min(0.0, p1, f_md), gs.min(0.0, s4, f_lc))
                    f_hi = gs.min(gs.max(0.0, p1, f_md), gs.max(0.0, s4, f_lc))
                    f = gs.min(gs.max(f, f_lo), f_hi)
                flux = vel * (up + f) * inv_d
                if side == 0:
                    total += flux
                else:
                    total -= flux
    return base[k, j, i] + dt * (total / ref[k, 5] + slow[k, j, i])


@gs.kernel
def atm_momentum_stage(
    base: gs.f32["nz", "ny", "nx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    buoyancy: gs.f32["nz", "ny", "nx"],
    slow: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    forcing: gs.f32["nz", "c8"],
    comp: gs.i32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    coriolis: gs.f32,
    frame_u: gs.f32,
    frame_v: gs.f32,
    dt: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """`base + dt (advection + buoyancy, Coriolis, subsidence + slow)` of one wind component.

    Advection in flux form, order-9 upwind reconstruction. Mass fluxes are the means of those of the
    discrete continuity: a uniform wind stays uniform."""
    k, j, i = gs.index()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    if comp == 2 and k == 0:
        return 0.0
    coef = gs.local(gs.f32, 9)
    coef[0] = 1.5873016e-3
    coef[1] = -1.6269841e-2
    coef[2] = 7.8968254e-2
    coef[3] = -2.5436508e-1
    coef[4] = 7.4563492e-1
    coef[5] = 5.4563492e-1
    coef[6] = -1.2103175e-1
    coef[7] = 2.1825397e-2
    coef[8] = -1.9841270e-3
    total = 0.0
    for ax in range(3):
        for side in range(2):
            qi = i
            qj = j
            qk = k
            inv_d = inv_dx
            if ax == 0:
                qi = (i + side) % nx
            elif ax == 1:
                inv_d = inv_dy
                qj = (j + side) % ny
            else:
                inv_d = inv_dz
                qk = k + side
            # Mass flux: mean of two faces shifted by one cell along `comp`
            mass = 0.0
            for h in range(2):
                hi = qi
                hj = qj
                hk = qk
                if h == 1:
                    if comp == 0:
                        hi = (qi - 1 + nx) % nx
                    elif comp == 1:
                        hj = (qj - 1 + ny) % ny
                    else:
                        hk = qk - 1
                if ax == 0:
                    if hk >= 0 and hk < nz:
                        mass += 0.5 * u[hk, hj, hi] * ref[hk, 5]
                elif ax == 1:
                    if hk >= 0 and hk < nz:
                        mass += 0.5 * v[hk, hj, hi] * ref[hk, 5]
                elif hk > 0 and hk < nz:
                    mass += 0.5 * w[hk, hj, hi] * ref[hk, 6]
            if mass != 0.0:
                face = 0.0
                for m in range(9):
                    off = m - 5
                    if mass < 0.0:
                        off = 4 - m
                    ci = qi
                    cj = qj
                    ck = qk
                    if ax == 0:
                        ci = (qi + off) % nx
                    elif ax == 1:
                        cj = (qj + off) % ny
                    else:
                        ck = qk + off
                    val = 0.0
                    if comp == 2:
                        if ck > 0 and ck < nz:
                            val = w[ck, cj, ci]
                    else:
                        ck = gs.min(gs.max(ck, 0), nz - 1)
                        val = u[ck, cj, ci] if comp == 0 else v[ck, cj, ci]
                    face += coef[m] * val
                if side == 0:
                    total += mass * face * inv_d
                else:
                    total -= mass * face * inv_d
    # Forces
    tend = 0.0
    if comp == 2:
        total = total / ref[k, 6]
        damp = 0.5 * (forcing[k, 7] + forcing[k - 1, 7])
        tend = 0.5 * (buoyancy[k, j, i] + buoyancy[k - 1, j, i]) - damp * w[k, j, i]
    else:
        total = total / ref[k, 5]
        c = u[k, j, i]
        if comp == 0:
            im = (i - 1 + nx) % nx
            jp = (j + 1) % ny
            other = 0.25 * (v[k, j, i] + v[k, jp, i] + v[k, j, im] + v[k, jp, im]) + frame_v
            tend = coriolis * (other - forcing[k, 4]) + forcing[k, 5]
        else:
            c = v[k, j, i]
            jm = (j - 1 + ny) % ny
            ip = (i + 1) % nx
            other = 0.25 * (u[k, j, i] + u[k, j, ip] + u[k, jm, i] + u[k, jm, ip]) + frame_u
            tend = -coriolis * (other - forcing[k, 3]) + forcing[k, 6]
        w_ls = forcing[k, 0]
        if w_ls < 0.0 and k < nz - 1:
            above = u[k + 1, j, i] if comp == 0 else v[k + 1, j, i]
            tend -= w_ls * (above - c) * inv_dz
    return base[k, j, i] + dt * (total + tend + slow[k, j, i])


# ───────────────────────── Subgrid turbulence (Smagorinsky-Lilly) ─────────────────────────


@gs.kernel
def atm_eddy_viscosity(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    qt: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dx: gs.f32,
    dy: gs.f32,
    dz: gs.f32,
    cs: gs.f32,
    prandtl: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Eddy viscosity at cell centres, with dry or moist stability (Durran & Klemp 1982)."""
    k, j, i = gs.index()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = gs.min(k + 1, nz - 1)
    km = gs.max(k - 1, 0)
    w_top = w[k + 1, j, i] if k < nz - 1 else 0.0
    s11 = (u[k, j, ip] - u[k, j, i]) / dx
    s22 = (v[k, jp, i] - v[k, j, i]) / dy
    s33 = (w_top - w[k, j, i]) / dz
    # Centred shears
    uc_jp = 0.5 * (u[k, jp, i] + u[k, jp, ip])
    uc_jm = 0.5 * (u[k, jm, i] + u[k, jm, ip])
    vc_ip = 0.5 * (v[k, j, ip] + v[k, jp, ip])
    vc_im = 0.5 * (v[k, j, im] + v[k, jp, im])
    s12 = 0.5 * ((uc_jp - uc_jm) / (2.0 * dy) + (vc_ip - vc_im) / (2.0 * dx))
    span = float(kp - km) * dz
    uz = (u[kp, j, i] + u[kp, j, ip] - u[km, j, i] - u[km, j, ip]) * 0.5 / span
    vz = (v[kp, j, i] + v[kp, jp, i] - v[km, j, i] - v[km, jp, i]) * 0.5 / span
    wt_ip = w[k + 1, j, ip] if k < nz - 1 else 0.0
    wt_im = w[k + 1, j, im] if k < nz - 1 else 0.0
    wt_jp = w[k + 1, jp, i] if k < nz - 1 else 0.0
    wt_jm = w[k + 1, jm, i] if k < nz - 1 else 0.0
    wx = (w[k, j, ip] + wt_ip - w[k, j, im] - wt_im) * 0.25 / dx
    wy = (w[k, jp, i] + wt_jp - w[k, jm, i] - wt_jm) * 0.25 / dy
    s13 = 0.5 * (uz + wx)
    s23 = 0.5 * (vz + wy)
    strain2 = 2.0 * (s11 * s11 + s22 * s22 + s33 * s33) + 4.0 * (s12 * s12 + s13 * s13 + s23 * s23)

    t_c = t[k, j, i]
    dtdz = (t[kp, j, i] - t[km, j, i]) / span
    dqt = (qt[kp, j, i] - qt[km, j, i]) / span
    n2 = 0.0
    if qc[k, j, i] > 1e-6:
        # Saturated air: temperature gradient against the moist adiabat
        p = ref[k, 1]
        es = 611.2 * gs.exp(17.67 * (t_c - 273.15) / (t_c - 29.65))
        qs = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
        lqrt = 2.501e6 * qs / (287.04 * t_c)
        gamma_m = 9.81 / 1004.64 * (1.0 + lqrt) / (1.0 + 0.62197 * 2.501e6 * 2.501e6 * qs / (1004.64 * 287.04 * t_c * t_c))
        n2 = 9.81 * ((dtdz + gamma_m) / t_c * (1.0 + lqrt) - dqt)
    else:
        qvz = dqt
        n2 = 9.81 * ((dtdz + 9.81 / 1004.64) / t_c + 0.60778 * qvz)
    delta = gs.pow(dx * dy * dz, 0.33333333)
    # Mixing length limited near the ground
    height = ref[k, 0]
    inv_l2 = 1.0 / (cs * cs * delta * delta) + 1.0 / (0.16 * height * height)
    fb = gs.max(strain2 - n2 / prandtl, 0.0)
    return gs.sqrt(fb) / inv_l2


@gs.kernel
def atm_diffuse_scalar(
    phi: gs.f32["nz", "ny", "nx"],
    km: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    inv_prandtl: gs.f32,
    surface_flux: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Turbulent diffusion of a scalar; `surface_flux` (unit · m/s) enters through the ground."""
    k, j, i = gs.index()
    nz = phi.shape[0]
    ny = phi.shape[1]
    nx = phi.shape[2]
    c = phi[k, j, i]
    kc = km[k, j, i]
    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    total = 0.0
    total += ((kc + km[k, j, ip]) * (phi[k, j, ip] - c) - (kc + km[k, j, im]) * (c - phi[k, j, im])) * 0.5 * inv_dx * inv_dx
    total += ((kc + km[k, jp, i]) * (phi[k, jp, i] - c) - (kc + km[k, jm, i]) * (c - phi[k, jm, i])) * 0.5 * inv_dy * inv_dy
    total = total * ref[k, 5]
    if k < nz - 1:
        total += ref[k + 1, 6] * 0.5 * (kc + km[k + 1, j, i]) * (phi[k + 1, j, i] - c) * inv_dz * inv_dz
    if k > 0:
        total -= ref[k, 6] * 0.5 * (kc + km[k - 1, j, i]) * (c - phi[k - 1, j, i]) * inv_dz * inv_dz
    total = total * inv_prandtl
    if k == 0:
        total += ref[0, 6] * surface_flux * inv_dz
    return total / ref[k, 5]


@gs.kernel
def atm_diffuse_momentum(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    km: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    comp: gs.i32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    ustar: gs.f32,
    frame_u: gs.f32,
    frame_v: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Turbulent diffusion of one component (variable-viscosity Laplacian) and ground friction `u*²`."""
    k, j, i = gs.index()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    if comp == 2 and k == 0:
        return 0.0
    total = 0.0
    c = u[k, j, i]
    if comp == 1:
        c = v[k, j, i]
    elif comp == 2:
        c = w[k, j, i]
    for ax in range(3):
        for side in range(2):
            ni = i
            nj = j
            nk = k
            d = 2 * side - 1
            inv_d = inv_dx
            if ax == 0:
                ni = (i + d + nx) % nx
            elif ax == 1:
                inv_d = inv_dy
                nj = (j + d + ny) % ny
            else:
                inv_d = inv_dz
                nk = k + d
            val = c
            visc = 0.0
            if nk >= 0 and nk < nz:
                if comp == 0:
                    val = u[nk, nj, ni]
                elif comp == 1:
                    val = v[nk, nj, ni]
                else:
                    val = w[nk, nj, ni] if nk > 0 else 0.0
                visc = 0.5 * (km[k, j, i] + km[nk, nj, ni])
            elif comp == 2 and nk == nz:
                val = 0.0
                visc = km[nz - 1, j, i]
            weight = ref[k, 5]
            if ax == 2 and comp != 2:
                weight = ref[gs.min(k + side, nz - 1), 6]
            elif comp == 2:
                weight = ref[k, 6]
            total += weight * visc * (val - c) * inv_d * inv_d
    rho = ref[k, 5]
    if comp == 2:
        rho = ref[k, 6]
    total = total / rho
    if k == 0 and comp != 2 and ustar > 0.0:
        ug = u[0, j, i] + frame_u
        vg = v[0, j, i] + frame_v
        speed = gs.sqrt(ug * ug + vg * vg + 0.01)
        wind = ug
        if comp == 1:
            wind = vg
        total -= ustar * ustar * wind / speed * inv_dz
    return total


# ───────────────────────── Forcings ─────────────────────────


@gs.kernel
def atm_forcing_scalar(
    phi: gs.f32["nz", "ny", "nx"],
    forcing: gs.f32["nz", "c8"],
    column: gs.i32,
    inv_dz: gs.f32,
    sponge_ref: gs.f32["nz"],
) -> gs.f32["nz", "ny", "nx"]:
    """Large-scale subsidence (column 0), imposed tendency (`column`) and relaxation near the top (column 7)."""
    k, j, i = gs.index()
    nz = phi.shape[0]
    w_ls = forcing[k, 0]
    grad = 0.0
    if w_ls < 0.0 and k < nz - 1:
        grad = (phi[k + 1, j, i] - phi[k, j, i]) * inv_dz
    elif w_ls > 0.0 and k > 0:
        grad = (phi[k, j, i] - phi[k - 1, j, i]) * inv_dz
    return -w_ls * grad + forcing[k, column] - forcing[k, 7] * (phi[k, j, i] - sponge_ref[k])


@gs.kernel
def atm_axpy(
    base: gs.f32["nz", "ny", "nx"],
    t1: gs.f32["nz", "ny", "nx"],
    t2: gs.f32["nz", "ny", "nx"],
    dt: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """`base + dt (t1 + t2)`."""
    k, j, i = gs.index()
    return base[k, j, i] + dt * (t1[k, j, i] + t2[k, j, i])


@gs.kernel
def atm_add(a: gs.f32["nz", "ny", "nx"], b: gs.f32["nz", "ny", "nx"]) -> gs.f32["nz", "ny", "nx"]:
    k, j, i = gs.index()
    return a[k, j, i] + b[k, j, i]


# ───────────────────────── Pression (solveur direct) ─────────────────────────


@gs.kernel
def atm_divergence(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Divergence of the mass flux `ρ0 u`."""
    k, j, i = gs.index()
    nz = u.shape[0]
    ip = (i + 1) % u.shape[2]
    jp = (j + 1) % u.shape[1]
    top = 0.0
    if k < nz - 1:
        top = ref[k + 1, 6] * w[k + 1, j, i]
    bot = 0.0
    if k > 0:
        bot = ref[k, 6] * w[k, j, i]
    return ref[k, 5] * ((u[k, j, ip] - u[k, j, i]) * inv_dx + (v[k, jp, i] - v[k, j, i]) * inv_dy) + (top - bot) * inv_dz


@gs.kernel
def atm_hartley_x1(
    a: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c2"], n1: gs.i32, n2: gs.i32
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    """Hartley in x for N = n1·n2, first pass (Cooley-Tukey): real and imaginary parts."""
    k, j, i = gs.index()
    n = n1 * n2
    k1 = i // n2
    col = i - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        v = a[k, j, m * n2 + col]
        re += v * trig[idx, 0]
        im -= v * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def atm_hartley_x2(
    re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c2"], n1: gs.i32, n2: gs.i32
) -> gs.f32["nz", "ny", "nx"]:
    """Second pass: Re - Im of the Fourier transform."""
    k, j, i = gs.index()
    n = n1 * n2
    k2 = i // n1
    row = (i - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    total = 0.0
    for m in range(n2):
        c = trig[idx, 0]
        q = trig[idx, 1]
        total += re[k, j, row + m] * (c + q) + im[k, j, row + m] * (q - c)
        idx += step
        if idx >= n:
            idx -= n
    return total / gs.sqrt(float(n))


@gs.kernel
def atm_hartley_y1(
    a: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c2"], n1: gs.i32, n2: gs.i32
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    k, j, i = gs.index()
    n = n1 * n2
    k1 = j // n2
    col = j - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        v = a[k, m * n2 + col, i]
        re += v * trig[idx, 0]
        im -= v * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def atm_hartley_y2_columns(
    re: gs.f32["nz", "ny", "nx"],
    im: gs.f32["nz", "ny", "nx"],
    trig: gs.f32["ny", "c2"],
    tpl: gs.f32["ny", "nb", "nz", "g"],
    n1: gs.i32,
    n2: gs.i32,
) -> gs.f32["ny", "nb", "nz", "g"]:
    """Second pass in y, result stored by columns: `(j, block, k, rank)` with `i = block · g + rank`."""
    j, b, k, r = gs.index()
    i = b * tpl.shape[3] + r
    n = n1 * n2
    k2 = j // n1
    row = (j - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    total = 0.0
    for m in range(n2):
        c = trig[idx, 0]
        q = trig[idx, 1]
        total += re[k, row + m, i] * (c + q) + im[k, row + m, i] * (q - c)
        idx += step
        if idx >= n:
            idx -= n
    return total / gs.sqrt(float(n))


@gs.kernel
def atm_columns_hartley_y1(
    a: gs.f32["ny", "nb", "nz", "g"], tpl: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c2"], n1: gs.i32, n2: gs.i32
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    """First pass in y from the column layout."""
    k, j, i = gs.index()
    b = i // a.shape[3]
    r = i - b * a.shape[3]
    n = n1 * n2
    k1 = j // n2
    col = j - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        v = a[m * n2 + col, b, k, r]
        re += v * trig[idx, 0]
        im -= v * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def atm_hartley_y2(
    re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c2"], n1: gs.i32, n2: gs.i32
) -> gs.f32["nz", "ny", "nx"]:
    k, j, i = gs.index()
    n = n1 * n2
    k2 = j // n1
    row = (j - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    total = 0.0
    for m in range(n2):
        c = trig[idx, 0]
        q = trig[idx, 1]
        total += re[k, row + m, i] * (c + q) + im[k, row + m, i] * (q - c)
        idx += step
        if idx >= n:
            idx -= n
    return total / gs.sqrt(float(n))


def _pressure_columns(
    rhs: gs.f32["ny", "nb", "nz", "g"],
    ref: gs.f32["nz", "c8"],
    eig_x: gs.f32["nx"],
    eig_y: gs.f32["ny"],
    inv_dz: gs.f32,
) -> gs.f32["ny", "nb", "nz", "g"]:
    """Vertical tridiagonal system of each horizontal mode (Thomas), Neumann walls: one thread per column."""
    j, b, k0, r = gs.index()
    nz = rhs.shape[2]
    i = b * rhs.shape[3] + r
    cp = gs.local(gs.f32, 256)
    dp = gs.local(gs.f32, 256)
    lam = eig_x[i] + eig_y[j]
    iz2 = inv_dz * inv_dz
    mean = 0
    excess = 0.0
    if i == 0 and j == 0:
        # Open edges: the lateral mass budget is zero only up to rounding, spread the residue over the column
        mean = 1
        mass = 0.0
        for m in range(nz):
            excess += rhs[j, b, m, r]
            mass += ref[m, 5]
        excess = excess / mass
    for m in range(nz):
        lo = ref[m, 6] * iz2 if m > 0 else 0.0
        hi = ref[m + 1, 6] * iz2 if m < nz - 1 else 0.0
        diag = -(lo + hi) - ref[m, 5] * lam
        d = rhs[j, b, m, r] - ref[m, 5] * excess
        if mean != 0 and m == 0:
            # Mean mode: pressure is only defined up to a constant
            diag = -1.0
            hi = 0.0
            d = 0.0
        if mean != 0 and m == 1:
            lo = 0.0
            diag = -hi - ref[1, 6] * iz2
        if m == 0:
            cp[0] = hi / diag
            dp[0] = d / diag
        else:
            den = diag - lo * cp[m - 1]
            cp[m] = hi / den
            dp[m] = (d - lo * dp[m - 1]) / den
    x = dp[nz - 1]
    gs.store(nz - 1, x)
    for m in range(nz - 1):
        x = dp[nz - 2 - m] - cp[nz - 2 - m] * x
        gs.store(nz - 2 - m, x)


_COLUMN_KERNELS: dict[tuple[int, int], object] = {}


def pressure_columns(group: int, nz: int):
    """Column kernel for `group` threads and `nz` levels (compiled once per size)."""
    key = (group, nz)
    if key not in _COLUMN_KERNELS:
        _COLUMN_KERNELS[key] = gs.kernel(_pressure_columns, group=group, cells=nz)
    return _COLUMN_KERNELS[key]


@gs.kernel
def atm_project(
    vel: gs.f32["nz", "ny", "nx"],
    phi: gs.f32["nz", "ny", "nx"],
    comp: gs.i32,
    inv_d: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Removes the pressure gradient: the mass flux becomes divergence-free."""
    k, j, i = gs.index()
    if comp == 0:
        return vel[k, j, i] - (phi[k, j, i] - phi[k, j, (i - 1 + vel.shape[2]) % vel.shape[2]]) * inv_d
    if comp == 1:
        return vel[k, j, i] - (phi[k, j, i] - phi[k, (j - 1 + vel.shape[1]) % vel.shape[1], i]) * inv_d
    if k == 0:
        return 0.0
    return vel[k, j, i] - (phi[k, j, i] - phi[k - 1, j, i]) * inv_d


# ───────────────────────── Warm rain (Kessler) ─────────────────────────


@gs.kernel
def atm_kessler(
    qt: gs.f32["nz", "ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dt: gs.f32,
    ice: gs.f32,
    fusion: gs.f32,
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    """Conversion, accretion and evaporation of rain: `(qt, qr, transfer)`.

    `transfer` > 0: water moved from cloud to rain. `s` does not change: it already counts all the liquid water.
    `fusion` = 1 (with snow): only the liquid part of the cloud turns into rain, the ice goes to the snow."""
    k, j, i = gs.index()
    rho = ref[k, 5]
    rain = gs.max(qr[k, j, i], 0.0)
    cloud = qc[k, j, i]
    vap = qt[k, j, i] - cloud
    t_c = t[k, j, i]
    if fusion > 0.0 and t_c < 273.15:
        part = gs.max((t_c - 250.16) / 23.0, 0.0)
        cloud = cloud * gs.min(part * part, 1.0)
    auto = 1e-3 * gs.max(cloud - 1e-3, 0.0)
    accr = 2.2 * cloud * gs.pow(rain, 0.875)
    gain = gs.min((auto + accr) * dt, cloud)
    loss = 0.0
    p = ref[k, 1]
    es = 611.2 * gs.exp(17.67 * (t_c - 273.15) / (t_c - 29.65))
    if ice > 0.0 and t_c < 273.15:
        mix = gs.max((t_c - 250.16) / 23.0, 0.0)
        mix = gs.min(mix * mix, 1.0)
        es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (t_c - 273.15) / (t_c - 0.55))
    qs = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
    if qc[k, j, i] <= 0.0 and rain > 0.0 and vap < qs:
        # Evaporation with ventilation (Kessler, coefficients of Klemp & Wilhelmson 1978)
        rr = rho * 1e-3 * rain
        vent = 1.6 + 124.9 * gs.pow(rr, 0.2046)
        evap = (1.0 - vap / qs) * vent * gs.pow(rr, 0.525) / (rho * 1e-3 * (5.4e5 + 2.55e6 / (p * 0.01 * qs)))
        loss = gs.min(gs.min(evap * dt, rain), qs - vap)
    delta = gain - loss
    return qt[k, j, i] - delta, rain + delta, delta


@gs.kernel
def atm_sediment(
    qr: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dt: gs.f32,
    inv_dz: gs.f32,
    coef: gs.f32,
    expo: gs.f32,
    scale: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Fall of rain or snow (upwind flux); what leaves through the bottom reaches the ground. Mass-weighted speed
    `coef (scale rho q)^expo (rho_0 / rho)^0.5`: rain 36.34, 0.1346, 1e-3 (Kessler, as in Klemp & Wilhelmson 1978),
    snow 2.414, 0.1025, 1 (Lin et al. 1983, exponential distribution)."""
    k, j, i = gs.index()
    nz = qr.shape[0]
    out = 0.0
    inflow = 0.0
    for h in range(2):
        m = k + h
        if m < nz:
            q = gs.max(qr[m, j, i], 0.0)
            speed = coef * gs.pow(ref[m, 5] * scale * q, expo) * gs.sqrt(ref[0, 5] / ref[m, 5])
            flux = ref[m, 5] * q * gs.min(speed, 0.9 / (dt * inv_dz))
            if h == 0:
                out = flux
            else:
                inflow = flux
    return qr[k, j, i] + (inflow - out) * dt * inv_dz / ref[k, 5]


@gs.kernel
def atm_ice_fall(
    qc: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dt: gs.f32,
    inv_dz: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Change of cloud water due to falling crystals (Heymsfield & Donner 1990: 3.29 (rho q)^0.16 m/s, in
    proportion to the ice between 258 and 238 K). Nothing reaches the ground."""
    k, j, i = gs.index()
    out = 0.0
    inflow = 0.0
    for h in range(2):
        m = k + h
        if m < qc.shape[0] and m > 0:
            q = qc[m, j, i]
            part = gs.min(gs.max((258.0 - t[m, j, i]) / 20.0, 0.0), 1.0)
            if q > 0.0 and part > 0.0:
                fall = gs.min(3.29 * gs.pow(ref[m, 5] * q, 0.16) * part, 0.9 / (dt * inv_dz))
                if h == 0:
                    out = ref[m, 5] * q * fall
                else:
                    inflow = ref[m, 5] * q * fall
    return (inflow - out) * dt * inv_dz / ref[k, 5]


@gs.kernel
def atm_snow(
    qt: gs.f32["nz", "ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    qs: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dt: gs.f32,
) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    """Snow processes of Lin, Farley & Orville (1983): `(qt, qr, qs)`.

    Exponential snow distribution (N0 = 3e6 m^-4, density 100 kg/m³), fall speed 11.72 D^0.41 (Locatelli & Hobbs 1974),
    rain N0 = 8e6 m^-4. Below 0 °C: autoconversion of cloud ice (threshold 0.1 g/kg), collection of cloud ice and
    riming of cloud water, deposition or sublimation, freezing of rain by contact with snow and by immersion (Bigg).
    Above: melting from the heat and vapour budget (so it follows the wet-bulb temperature), cloud water collected
    goes to the rain, sublimation in unsaturated air. Every phase change stays inside `s` (liquid-ice static energy)."""
    k, j, i = gs.index()
    rho = ref[k, 5]
    p = ref[k, 1]
    dens = gs.sqrt(ref[0, 5] / rho)
    temp = t[k, j, i]
    tc = temp - 273.15
    cloud = qc[k, j, i]
    vap = qt[k, j, i] - cloud
    snow = gs.max(qs[k, j, i], 0.0)
    rain = gs.max(qr[k, j, i], 0.0)
    liq = 1.0
    if tc < 0.0:
        part = gs.max((temp - 250.16) / 23.0, 0.0)
        liq = gs.min(part * part, 1.0)
    water = cloud * liq
    cice = cloud - water
    es = 611.2 * gs.exp(17.67 * tc / (temp - 29.65))
    esi = 611.2 * gs.exp(22.46 * tc / (temp - 0.55))
    qsw = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
    qsi = 0.62197 * esi / gs.max(p - 0.37803 * esi, 1.0)
    e0 = 611.2
    qs0 = 0.62197 * e0 / gs.max(p - 0.37803 * e0, 1.0)
    to_snow = 0.0  # from vapour and cloud (taken from qt)
    rain_to_snow = 0.0
    snow_to_rain = 0.0
    cloud_to_rain = 0.0
    if tc < 0.0 and cice > 1e-4:
        to_snow += 1e-3 * gs.exp(0.025 * tc) * (cice - 1e-4)
    lam_r = 0.0
    if rain > 1e-9:
        lam_r = gs.pow(2.5132741e10 / (rho * rain), 0.25)
    if snow > 1e-9:
        lam = gs.pow(9.424778e8 / (rho * snow), 0.25)
        coll = 8.3211486e7 * gs.pow(lam, -3.41) * dens
        vent = 0.78 / (lam * lam) + 334.91695 * gs.sqrt(rho) * gs.sqrt(dens) * gs.pow(lam, -2.705)
        if tc < 0.0:
            to_snow += coll * (water + gs.exp(0.025 * tc) * cice)
            a2 = 8.0317e12 / (0.0243 * 461.5 * temp * temp)
            b2 = 1.0 / (rho * qsi * 2.26e-5)
            to_snow += 6.2831853 * (vap / qsi - 1.0) * 3e6 * vent / (rho * (a2 + b2))
            if rain > 1e-9:
                v_r = 36.34 * gs.pow(rho * 1e-3 * rain, 0.1346) * dens
                v_s = 2.4138622 * gs.pow(rho * snow, 0.1025) * dens
                inv_r = 1.0 / lam_r
                inv_s = 1.0 / lam
                shape = 5.0 * gs.pow(inv_r, 6.0) * inv_s + 2.0 * gs.pow(inv_r, 5.0) * inv_s * inv_s + 0.5 * gs.pow(inv_r, 4.0) * inv_s * inv_s * inv_s
                rain_to_snow += 9.8696044 * 8e6 * 3e6 * gs.abs(v_r - v_s) * 1000.0 / rho * shape
        else:
            cloud_to_rain = coll * water
            melt = 6.2831853 / (rho * 3.337e5) * (0.0243 * tc - 2.501e6 * 2.26e-5 * rho * (qs0 - vap)) * 3e6 * vent
            snow_to_rain = gs.max(melt, 0.0) + 4187.0 * tc / 3.337e5 * cloud_to_rain
            if vap < qsw:
                a3 = 6.2550e12 / (0.0243 * 461.5 * temp * temp)
                b3 = 1.0 / (rho * qsw * 2.26e-5)
                to_snow += 6.2831853 * (vap / qsw - 1.0) * 3e6 * vent / (rho * (a3 + b3))
    if tc < 0.0 and rain > 1e-9:
        rain_to_snow += 1.9739209e4 * 8e6 * 1000.0 / rho * (gs.exp(0.66 * (-tc)) - 1.0) * gs.pow(1.0 / lam_r, 7.0)
    # Limits over the step
    gain = to_snow * dt
    if gain > 0.0:
        room = cloud
        if tc < 0.0:
            room = cloud + gs.max(vap - qsi, 0.0)
        gain = gs.min(gain, room)
    else:
        gain = gs.max(gain, -snow)
        if tc < 0.0:
            gain = gs.max(gain, gs.min(vap - qsi, 0.0))
        else:
            gain = gs.max(gain, gs.min(vap - qsw, 0.0))
    left = snow + gain
    melted = gs.min(snow_to_rain * dt, left)
    frozen = gs.min(rain_to_snow * dt, rain)
    collected = gs.min(cloud_to_rain * dt, gs.max(cloud - gs.max(gain, 0.0), 0.0))
    return qt[k, j, i] - gain - collected, rain - frozen + melted + collected, left - melted + frozen


@gs.kernel
def atm_surface_rain(
    acc: gs.f32["ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    dt: gs.f32,
    inv_dz: gs.f32,
    coef: gs.f32,
    expo: gs.f32,
    scale: gs.f32,
) -> gs.f32["ny", "nx"]:
    """Accumulated rain or snow at the ground (kg/m², i.e. mm of water): flux leaving the first layer, as in `atm_sediment`."""
    j, i = gs.index()
    q = gs.max(qr[0, j, i], 0.0)
    speed = coef * gs.pow(ref[0, 5] * scale * q, expo)
    return acc[j, i] + ref[0, 5] * q * gs.min(speed, 0.9 / (dt * inv_dz)) * dt


# ───────────────────────── Coupling with the large-scale weather ─────────────────────────


@gs.kernel
def atm_level_means(
    s: gs.f32["nz", "ny", "nx"],
    qt: gs.f32["nz", "ny", "nx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    qs: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    tpl: gs.f32["nz", "c8"],
) -> gs.f32["nz", "c8"]:
    """Horizontal mean of each level (summed by rows, for precision)."""
    k, c = gs.index()
    total = 0.0
    for j in range(s.shape[1]):
        row = 0.0
        for i in range(s.shape[2]):
            if c == 0:
                row += s[k, j, i]
            elif c == 1:
                row += qt[k, j, i]
            elif c == 2:
                row += u[k, j, i]
            elif c == 3:
                row += v[k, j, i]
            elif c == 4:
                row += qc[k, j, i]
            elif c == 5:
                row += qr[k, j, i]
            elif c == 6:
                row += qs[k, j, i]
            else:
                row += t[k, j, i]
        total += row / float(s.shape[2])
    return total / float(s.shape[1])


@gs.kernel
def atm_offset(phi: gs.f32["nz", "ny", "nx"], change: gs.f32["nz", "c2"]) -> gs.f32["nz", "ny", "nx"]:
    """`phi · factor + addition`, one factor and one addition per level."""
    k, j, i = gs.index()
    return phi[k, j, i] * change[k, 1] + change[k, 0]


@gs.kernel
def atm_level_max(
    u: gs.f32["nz", "ny", "nx"], v: gs.f32["nz", "ny", "nx"], w: gs.f32["nz", "ny", "nx"], tpl: gs.f32["nz", "c3"]
) -> gs.f32["nz", "c3"]:
    """Largest absolute value of each wind component, per level."""
    k, c = gs.index()
    top = 0.0
    for j in range(u.shape[1]):
        for i in range(u.shape[2]):
            val = u[k, j, i]
            if c == 1:
                val = v[k, j, i]
            elif c == 2:
                val = w[k, j, i]
            top = gs.max(top, gs.abs(val))
    return top
