"""Physique en colonnes de la météo de grande échelle : sol et couche limite, convection en panaches, condensation.

Chaque noyau traite une colonne par thread et rend un paquet `(j, bloc, c, rang)` (voir `kweather`)."""

import gpusync as gs

# Colonnes des tables
SURF_LAND, SURF_Z0, SURF_ALBEDO, SURF_WET, SURF_LAT, SURF_HOUR, SURF_Z, SURF_DEEP = range(8)
SFC_TS, SFC_WTH, SFC_WQ, SFC_H, SFC_WSTAR, SFC_SW, SFC_CLOUD, SFC_USTAR, SFC_RN, SFC_T2, SFC_WIND, SFC_LW = range(12)
CNV_RAIN, CNV_CAPE, CNV_CIN, CNV_BASE, CNV_TOP, CNV_MASS, CNV_WMAX, CNV_DOWN, CNV_COLD, CNV_KIND, CNV_EVAP, CNV_LFC, CNV_MEMORY = range(13)
SURFACE_CELLS = 12
CONVECTION_CELLS = 13


def _surface_columns(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    ts: gs.f32["ny", "nx"],
    surf: gs.f32["ny", "nx", "c8"],
    conv_cover: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    sun: gs.f32["c8"],
    cooling: gs.f32,
    dt: gs.f32,
    tpl: gs.f32["ny", "nb", "cc", "g"],
) -> gs.f32["ny", "nb", "cc", "g"]:
    """Flux de surface (Louis 1979), bilan d'énergie du sol, mélange turbulent implicite (profil K de Troen & Mahrt
    1986 dans la couche limite, Louis au-dessus), refroidissement radiatif imposé de la troposphère."""
    j, b, c0, r = gs.index()
    i = b * 16 + r
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    z = gs.local(gs.f32, 64)
    p = gs.local(gs.f32, 64)
    uu = gs.local(gs.f32, 64)
    vv = gs.local(gs.f32, 64)
    tt = gs.local(gs.f32, 64)
    qq = gs.local(gs.f32, 64)
    cc = gs.local(gs.f32, 64)
    km = gs.local(gs.f32, 64)
    kh = gs.local(gs.f32, 64)
    cp_ = gs.local(gs.f32, 64)
    da = gs.local(gs.f32, 64)
    db = gs.local(gs.f32, 64)
    ex = gs.local(gs.f32, 64)
    ie = (i + 1) % nx
    pi = pi0[j, i] + pip[j, i]
    h_low = 0.0
    c_low = 0.0
    c_mid = 0.0
    c_high = 0.0
    # Profils (z au-dessus du sol)
    for m in range(nz):
        k = nz - 1 - m
        pm = 5000.0 + lev[k, 1] * pi
        p[k] = pm
        ex[k] = gs.pow(pm * 1e-5, 0.285714)
        uu[k] = 0.5 * (u[k, j, i] + u[k, j, ie])
        vv[k] = 0.5 * (v[k, j, i] + (v[k, j + 1, i] if j < ny - 1 else 0.0))
        tt[k] = th[k, j, i]
        qq[k] = gs.max(qv[k, j, i], 0.0)
        cc[k] = gs.max(qc[k, j, i], 0.0)
        t = tt[k] * ex[k]
        tv = t * (1.0 + 0.608 * qq[k])
        p_low = 5000.0 + lev[k + 1, 0] * pi
        p_up = 5000.0 + lev[k, 0] * pi
        z[k] = h_low + 29.2599 * tv * gs.log(p_low / pm)
        h_low += 29.2599 * tv * gs.log(p_low / p_up)
        # Nébulosité de la couche (Sundqvist) : maximum par étage, étages indépendants
        es = 611.2 * gs.exp(17.67 * (t - 273.15) / (t - 29.65))
        if t < 273.15:
            mix = gs.max((t - 250.16) / 23.0, 0.0)
            mix = gs.min(mix * mix, 1.0)
            es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (t - 273.15) / (t - 0.55))
        rh = gs.min(qq[k] / (0.62197 * es / gs.max(pm - 0.378 * es, 1.0)), 1.0)
        rc = 0.94 if pm > 70000.0 else 0.88  # bas : couverture du cas BOMEX 3D
        if rh > rc and pm > 15000.0:
            fr = 1.0 - gs.sqrt((1.0 - rh) / (1.0 - rc))
            if pm > 70000.0:
                c_low = gs.max(c_low, fr)
            elif pm > 40000.0:
                c_mid = gs.max(c_mid, fr)
            else:
                # Un cirrus ne compte qu'à proportion de sa glace (pleine à 20 mg/kg)
                c_high = gs.max(c_high, fr * gs.min(cc[k] / 2e-5, 1.0))
    conv = 1.0 - conv_cover[j, i]
    cloud = 1.0 - (1.0 - c_low) * (1.0 - c_mid) * (1.0 - c_high) * conv
    dim = 1.0 - (1.0 - c_low) * (1.0 - c_mid) * (1.0 - 0.35 * c_high) * conv  # vu du soleil
    n = nz - 1
    # Rayonnement au sol
    hour = sun[0] + surf[j, i, 5]
    cosz = gs.sin(surf[j, i, 4]) * sun[1] + gs.cos(surf[j, i, 4]) * sun[2] * gs.cos((hour - 12.0) * 0.2617994)
    cosz = gs.max(cosz, 0.0)
    sw = 1361.0 * cosz * gs.pow(0.76, 1.0 / gs.max(cosz, 0.05)) * 1.12 * (1.0 - 0.75 * gs.pow(dim, 3.4))
    t1 = tt[n] * ex[n]
    e1 = qq[n] * p[n] / (0.622 + qq[n])
    eps = gs.min(1.24 * gs.pow(e1 * 0.01 / t1, 0.142857), 1.0)
    eps = eps + (1.0 - eps) * 0.84 * dim
    lw = eps * 5.67e-8 * t1 * t1 * t1 * t1
    # Flux de surface
    tg = ts[j, i]
    land = surf[j, i, 0]
    z0 = surf[j, i, 1]
    p_s = 5000.0 + pi
    ex_s = gs.pow(p_s * 1e-5, 0.285714)
    es = 611.2 * gs.exp(17.67 * (tg - 273.15) / (tg - 29.65))
    q_sat = 0.62197 * es / gs.max(p_s - 0.378 * es, 1.0)
    q_s = qq[n] + surf[j, i, 3] * gs.max(q_sat - qq[n], 0.0) if land > 0.5 else 0.98 * q_sat
    if land > 0.5 and q_sat < qq[n]:
        q_s = q_sat
    thv1 = tt[n] * (1.0 + 0.608 * qq[n])
    thvs = tg / ex_s * (1.0 + 0.608 * q_s)
    wind = gs.sqrt(uu[n] * uu[n] + vv[n] * vv[n])
    # Rafales convectives : ajout d'une vitesse liée au déficit
    gust = gs.pow(gs.max(9.81 / thv1 * (thvs - thv1) * 1000.0 * 0.002 * (wind + 1.0), 0.0), 0.333333)
    wnd = gs.max(gs.sqrt(wind * wind + 1.44 * gust * gust), 0.5)
    ri = 9.81 * z[n] * (thv1 - thvs) / (thv1 * wnd * wnd)
    ln = gs.log(z[n] / z0)
    cn = 0.16 / (ln * ln)
    fm = 1.0
    fh = 1.0
    if ri < 0.0:
        root = gs.sqrt(-ri * z[n] / z0)
        fm = 1.0 - 9.4 * ri / (1.0 + 69.56 * cn * root)
        fh = 1.0 - 9.4 * ri / (1.0 + 49.82 * cn * root)
    else:
        fm = 1.0 / ((1.0 + 4.7 * ri) * (1.0 + 4.7 * ri))
        fh = fm
    fm = gs.max(fm, 0.02)
    fh = gs.max(fh, 0.02)
    cd = cn * fm
    ch = cn * fh / 0.74
    wth = ch * wnd * (tg / ex_s - tt[n])
    wq = ch * wnd * (q_s - qq[n])
    rho = p_s / (287.04 * t1)
    heat = rho * 1004.64 * wth * ex_s
    evap = rho * 2.501e6 * wq
    rn = (1.0 - surf[j, i, 2]) * sw + lw - 0.96 * 5.67e-8 * tg * tg * tg * tg - 0.04 * lw
    tg_new = tg
    if land > 0.5:
        # Sol de Blackadar (1979) : force-rappel vers la température profonde, implicite dans le rappel et l'émission
        cg = 1.2e5
        stiff = 4.0 * 5.67e-8 * tg * tg * tg + rho * ch * wnd * 1004.64 + rho * ch * wnd * 2.501e6 * surf[j, i, 3] * q_sat * 4302.6 / ((tg - 29.65) * (tg - 29.65))
        tg_new = tg + dt * ((rn - heat - evap) / cg - 8.58e-5 * (tg - surf[j, i, 7])) / (1.0 + dt * (stiff / cg + 8.58e-5))
        # Flux cohérents avec la nouvelle température
        dtg = tg_new - tg
        wth += ch * wnd * dtg / ex_s
        wq += ch * wnd * surf[j, i, 3] * q_sat * 4302.6 / ((tg - 29.65) * (tg - 29.65)) * dtg if q_sat > qq[n] else 0.0
    if sun[3] > 0.5:
        # Flux imposés (cas de comparaison) : chaleur (K m/s), humidité (m/s), vitesse de frottement
        tg_new = tg
        wth = sun[4]
        wq = sun[5]
        cd = sun[6] * sun[6] / (wnd * wnd)
    ustar = gs.sqrt(cd) * wnd
    # Hauteur de couche limite : nombre de Richardson global
    wthv = wth + 0.608 * tt[n] * wq
    h = z[n]
    th_par = thv1
    wstar = 0.0
    for it in range(2):
        found = 0
        ri_old = 0.0
        z_old = z[n]
        for m in range(1, nz):
            k = nz - 1 - m
            if found == 0:
                thv = tt[k] * (1.0 + 0.608 * qq[k])
                du = uu[k] - uu[n]
                dv = vv[k] - vv[n]
                s2 = du * du + dv * dv + 100.0 * ustar * ustar * 0.1 + 0.01
                rk = 9.81 / thv1 * (thv - th_par) * (z[k] - z[n]) / s2
                if rk > 0.25:
                    h = z_old + (z[k] - z_old) * gs.max(0.0, gs.min(1.0, (0.25 - ri_old) / gs.max(rk - ri_old, 1e-6)))
                    found = 1
                ri_old = rk
                z_old = z[k]
        h = gs.min(gs.max(h, 60.0), 4000.0)
        if wthv > 0.0 and it == 0:
            wstar = gs.pow(9.81 / thv1 * wthv * h, 0.333333)
            ws = gs.pow(ustar * ustar * ustar + 0.28 * wstar * wstar * wstar, 0.333333)
            th_par = thv1 + gs.min(6.5 * wthv / ws, 3.0)
    wstar = gs.pow(gs.max(9.81 / thv1 * wthv * h, 0.0), 0.333333)
    ws = gs.max(gs.pow(ustar * ustar * ustar + 0.28 * wstar * wstar * wstar, 0.333333), 0.05)
    counter = 0.0
    counter_q = 0.0
    if wthv > 0.0:
        counter = gs.min(7.2 * wstar * gs.max(wth, 0.0) / (ws * ws * h), 2e-3)
        counter_q = gs.min(7.2 * wstar * gs.max(wq, 0.0) / (ws * ws * h), 2e-6)
    # Coefficients d'échange aux interfaces k (entre k et k+1)
    for k in range(nz - 1):
        zi = 0.5 * (z[k] + z[k + 1])
        dz = z[k] - z[k + 1]
        du = uu[k] - uu[k + 1]
        dv = vv[k] - vv[k + 1]
        s2 = (du * du + dv * dv) / (dz * dz) + 1e-8
        dth = tt[k] * (1.0 + 0.608 * qq[k]) - tt[k + 1] * (1.0 + 0.608 * qq[k + 1])
        rig = 9.81 / tt[k] * dth / dz / s2
        ll = 0.4 * zi / (1.0 + 0.4 * zi / 100.0)
        base_k = ll * ll * gs.sqrt(s2)
        if rig < 0.0:
            f = gs.sqrt(1.0 - 16.0 * gs.max(rig, -100.0))
            km[k] = base_k * f
            kh[k] = base_k * f
        else:
            f = 1.0 / (1.0 + 10.0 * rig * (1.0 + 8.0 * rig))
            km[k] = base_k * f
            kh[k] = base_k * f
        if zi < h:
            shape_ = (1.0 - zi / h) * (1.0 - zi / h)
            prof = 0.4 * ws * zi * shape_
            if wthv <= 0.0:
                prof = 0.4 * ustar * zi * shape_ / (1.0 + 5.0 * gs.max(9.81 / thv1 * (tt[k] - tt[n]) * 0.4 * zi / gs.max(ustar * ustar, 1e-3), 0.0) * 0.2)
            km[k] = gs.max(km[k], prof)
            kh[k] = gs.max(kh[k], prof * (1.25 if wthv > 0.0 else 1.0))
        km[k] = gs.min(km[k], 1000.0)
        kh[k] = gs.min(kh[k], 1000.0)
    # Diffusion implicite : quatre champs, quantité de mouvement puis scalaires
    for pass_ in range(5):
        for k in range(nz):
            mass = (lev[k, 2] * pi) / 9.81  # kg/m²
            lo = 0.0
            hi = 0.0
            src = 0.0
            val = uu[k]
            if pass_ == 1:
                val = vv[k]
            elif pass_ == 2:
                val = tt[k]
            elif pass_ == 3:
                val = qq[k]
            elif pass_ == 4:
                val = cc[k]
            if k > 0:
                dens = (5000.0 + lev[k, 0] * pi) / (287.04 * 0.5 * (tt[k] * ex[k] + tt[k - 1] * ex[k - 1]))
                kk = km[k - 1] if pass_ < 2 else kh[k - 1]
                lo = dens * kk / (z[k - 1] - z[k]) * dt / mass
                if pass_ == 2 and 0.5 * (z[k] + z[k - 1]) < h:
                    src -= dens * kk * counter * dt / mass
                if pass_ == 3 and 0.5 * (z[k] + z[k - 1]) < h:
                    src -= dens * kk * counter_q * dt / mass
            if k < nz - 1:
                dens = (5000.0 + lev[k + 1, 0] * pi) / (287.04 * 0.5 * (tt[k] * ex[k] + tt[k + 1] * ex[k + 1]))
                kk = km[k] if pass_ < 2 else kh[k]
                hi = dens * kk / (z[k] - z[k + 1]) * dt / mass
                if pass_ == 2 and 0.5 * (z[k] + z[k + 1]) < h:
                    src += dens * kk * counter * dt / mass
                if pass_ == 3 and 0.5 * (z[k] + z[k + 1]) < h:
                    src += dens * kk * counter_q * dt / mass
            diag = 1.0 + lo + hi
            if k == nz - 1:
                if pass_ < 2:
                    diag += rho * cd * wnd * dt / mass
                elif pass_ == 2:
                    src += rho * wth * dt / mass
                elif pass_ == 3:
                    src += rho * wq * dt / mass
            if pass_ == 2 and p[k] > 15000.0:
                src -= cooling * gs.min((p[k] - 15000.0) / 10000.0, 1.0) * dt / ex[k]
            if k == 0:
                cp_[0] = -hi / diag
                da[0] = (val + src) / diag
            else:
                den = diag + lo * cp_[k - 1]
                cp_[k] = -hi / den
                da[k] = (val + src + lo * da[k - 1]) / den
        x = da[nz - 1]
        for m in range(nz):
            k = nz - 1 - m
            if m > 0:
                x = da[k] - cp_[k] * x
            if pass_ == 0:
                gs.store(3 * nz + k, x - uu[k])
            elif pass_ == 1:
                gs.store(4 * nz + k, x - vv[k])
            elif pass_ == 2:
                gs.store(k, x)
            elif pass_ == 3:
                gs.store(nz + k, x)
            else:
                gs.store(2 * nz + k, x)
    o = 5 * nz
    gs.store(o, tg_new)
    gs.store(o + 1, wth)
    gs.store(o + 2, wq)
    gs.store(o + 3, h)
    gs.store(o + 4, wstar)
    gs.store(o + 5, sw)
    gs.store(o + 6, cloud)
    gs.store(o + 7, ustar)
    gs.store(o + 8, rn)
    # Température à 2 m : interpolation logarithmique entre sol et premier niveau
    gs.store(o + 9, (tg / ex_s + (tt[n] - tg / ex_s) * gs.log(gs.max(2.0 / z0, 1.1)) / gs.max(ln, 0.1)) * ex_s if z0 < 1.5 else t1)
    gs.store(o + 10, wind)
    gs.store(o + 11, lw)


def _convection_columns(
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    sfc: gs.f32["ny", "nx", "c4"],
    lift: gs.f32["ny", "nx"],
    mass_old: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    par: gs.f32["c24"],
    dt: gs.f32,
    tpl: gs.f32["ny", "nb", "cc", "g"],
) -> gs.f32["ny", "nb", "cc", "g"]:
    """Un panache entraînant par colonne, bas-fond ou profond, avec courant descendant et pluie.

    `sfc` : flux de chaleur (K m/s), d'humidité (m/s), hauteur de couche limite, w*. `lift` : vitesse de soulèvement
    (rafales, reliefs, convergence). `par` : 0 entraînement profond, 1 entraînement des bas-fonds, 2 accélération,
    3 freinage, 4 conversion en pluie (1/m), 5 seuil de conversion, 6 part du courant descendant, 7 part de glace
    dans la saturation, 8 durée de dissipation, 9 inertie (m⁴/kg), 10 w minimal, 11 évaporation de la pluie, 12 humidité
    relative des descentes, 13 flux maximal, 14 part d'air du nuage dans les descentes, 15 croissance du flux (1/m),
    16 excès des thermiques, 17 vitesse des thermiques / w*, 18 soulèvement entretenu par les orages (m/s), 19 dilution en plus
    avant organisation, 20 flux de masse d'organisation, 21 évaporation de la pluie dans la couche nuageuse."""
    j, b, c0, r = gs.index()
    i = b * 16 + r
    nz = th.shape[0]
    z = gs.local(gs.f32, 64)
    p = gs.local(gs.f32, 64)
    t = gs.local(gs.f32, 64)
    q = gs.local(gs.f32, 64)
    hh = gs.local(gs.f32, 64)  # énergie statique humide de l'environnement
    dm = gs.local(gs.f32, 64)  # masse de la couche (kg/m²)
    hu = gs.local(gs.f32, 64)
    qu = gs.local(gs.f32, 64)  # eau totale du panache
    lu = gs.local(gs.f32, 64)  # eau liquide du panache
    wu = gs.local(gs.f32, 64)
    et = gs.local(gs.f32, 64)  # flux de masse relatif au bas du nuage, au-dessus de la couche k
    gr = gs.local(gs.f32, 64)  # pluie formée par unité de flux de base
    bu = gs.local(gs.f32, 64)
    md = gs.local(gs.f32, 64)
    ice = par[7]
    pi = pi0[j, i] + pip[j, i]
    n = nz - 1
    h_low = 0.0
    for m in range(nz):
        k = n - m
        pm = 5000.0 + lev[k, 1] * pi
        p[k] = pm
        q[k] = gs.max(qv[k, j, i], 0.0)
        t[k] = th[k, j, i] * gs.pow(pm * 1e-5, 0.285714)
        tv = t[k] * (1.0 + 0.608 * q[k])
        p_low = 5000.0 + lev[k + 1, 0] * pi
        p_up = 5000.0 + lev[k, 0] * pi
        z[k] = h_low + 29.2599 * tv * gs.log(p_low / pm)
        h_low += 29.2599 * tv * gs.log(p_low / p_up)
        hh[k] = 1004.64 * t[k] + 9.81 * z[k] + 2.501e6 * q[k]
        dm[k] = lev[k, 2] * pi / 9.81
        hu[k] = hh[k]
        qu[k] = q[k]
        lu[k] = 0.0
        wu[k] = 0.0
        et[k] = 0.0
        gr[k] = 0.0
        bu[k] = 0.0
        md[k] = 0.0
    # Air de départ : moyenne de la basse couche, plus l'excès des thermiques
    depth = gs.min(gs.max(0.6 * sfc[j, i, 2], 200.0), 1000.0)
    sm = 0.0
    sh = 0.0
    sq = 0.0
    k_src = n
    for m in range(nz):
        k = n - m
        if z[k] < depth:
            sm += dm[k]
            sh += dm[k] * hh[k]
            sq += dm[k] * q[k]
            k_src = k
    wstar = sfc[j, i, 3]
    ex_t = 0.0
    ex_q = 0.0
    if wstar > 0.05:
        ex_t = gs.min(par[16] * gs.max(sfc[j, i, 0], 0.0) / wstar, 1.5)
        ex_q = gs.min(par[16] * gs.max(sfc[j, i, 1], 0.0) / wstar, 1.5e-3)
    h_src = sh / sm + 1004.64 * ex_t + 2.501e6 * ex_q
    q_src = sq / sm + ex_q
    grown = par[18] * (1.0 - gs.exp(-mass_old[j, i] / 0.01))  # orages déjà en place : leur air froid soulève l'air voisin
    w_src = gs.sqrt(par[10] * par[10] + par[17] * par[17] * wstar * wstar + lift[j, i] * lift[j, i] + grown * grown)
    kind = 0
    k_base = -1
    k_top = -1
    cape = 0.0
    cin = 0.0
    lfc = 0.0
    wmax = 0.0
    # Convection qui débute : nuages étroits, très dilués ; elle s'élargit en s'organisant (Mapes & Neale 2011)
    entr = par[0] * (1.0 + par[19] * gs.exp(-mass_old[j, i] / par[20]))
    # Deux essais : panache profond, puis panache de bas-fond s'il n'est pas monté assez haut
    for trial in range(2):
        if kind == 0:
            k_base = -1
            k_top = -1
            cape = 0.0
            cin = 0.0
            lfc = 0.0
            wmax = 0.0
            hc = h_src
            qt = q_src
            w2 = w_src * w_src
            alive = 1
            qs_base = 0.0
            for m in range(nz):
                k = n - m
                if alive == 1 and k <= k_src and k > 0:
                    dz = z[k] - z[k + 1] if k < n else z[k]
                    if k_base >= 0:
                        lam = par[1] * dz
                        if trial == 0:
                            es = 611.2 * gs.exp(17.67 * (t[k] - 273.15) / (t[k] - 29.65))
                            qse = 0.62197 * es / gs.max(p[k] - 0.378 * es, 1.0)
                            rat = qse / qs_base
                            lam = entr * dz * gs.max(1.3 - gs.min(q[k] / qse, 1.0), 0.1) * rat * rat * rat
                        hc = (hc + lam * hh[k]) / (1.0 + lam)
                        qt = (qt + lam * q[k]) / (1.0 + lam)
                        w2 = w2 / (1.0 + 2.0 * par[3] * lam)
                    # Température du panache : h et eau totale connues
                    tu = (hc - 9.81 * z[k] - 2.501e6 * qt) / 1004.64
                    ql = 0.0
                    es = 611.2 * gs.exp(17.67 * (tu - 273.15) / (tu - 29.65))
                    if tu < 273.15 and ice > 0.0:
                        mix = gs.max((tu - 250.16) / 23.0, 0.0)
                        mix = 1.0 - ice * (1.0 - gs.min(mix * mix, 1.0))
                        es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (tu - 273.15) / (tu - 0.55))
                    qs = 0.62197 * es / gs.max(p[k] - 0.378 * es, 1.0)
                    if qt > qs:
                        for _ in range(6):
                            es = 611.2 * gs.exp(17.67 * (tu - 273.15) / (tu - 29.65))
                            des = es * 4302.6 / ((tu - 29.65) * (tu - 29.65))
                            if tu < 273.15 and ice > 0.0:
                                mix = gs.max((tu - 250.16) / 23.0, 0.0)
                                mix = 1.0 - ice * (1.0 - gs.min(mix * mix, 1.0))
                                ei = 611.2 * gs.exp(22.46 * (tu - 273.15) / (tu - 0.55))
                                des = des * mix + (1.0 - mix) * ei * 6122.6 / ((tu - 0.55) * (tu - 0.55))
                                es = es * mix + (1.0 - mix) * ei
                            qs = 0.62197 * es / gs.max(p[k] - 0.378 * es, 1.0)
                            f = 1004.64 * tu + 9.81 * z[k] + 2.501e6 * qs - hc
                            tu = tu - f / (1004.64 + 2.501e6 * qs / es * des)
                        ql = gs.max(qt - qs, 0.0)
                        if k_base < 0:
                            k_base = k
                            qs_base = qs
                    rain = 0.0
                    if k_base >= 0 and ql > par[5]:
                        rain = (ql - par[5]) * (1.0 - gs.exp(-par[4] * dz))
                        ql = ql - rain
                        qt = qt - rain
                    tvu = tu * (1.0 + 0.608 * (qt - ql) - ql)
                    tve = t[k] * (1.0 + 0.608 * q[k] - gs.max(qc[k, j, i], 0.0))
                    buoy = 9.81 * (tvu - tve) / tve
                    w2 = w2 + 2.0 * (par[2] if buoy > 0.0 else gs.min(2.0 * par[2], 1.0)) * buoy * dz
                    if k_base >= 0 and buoy > 0.0:
                        cape += buoy * dz
                        if lfc == 0.0:
                            lfc = z[k]
                    elif lfc == 0.0:
                        cin += gs.min(buoy, 0.0) * dz
                    if w2 <= 0.0:
                        alive = 0
                    else:
                        hu[k] = hc
                        qu[k] = qt
                        lu[k] = ql
                        gr[k] = rain
                        bu[k] = buoy
                        wu[k] = gs.sqrt(w2)
                        if k_base >= 0:
                            k_top = k
                            wmax = gs.max(wmax, wu[k])
            if k_base >= 0 and k_top >= 0 and lfc > 0.0:
                if trial == 0:
                    if p[k_base] - p[k_top] > 20000.0:
                        kind = 2
                elif k_top < k_base:
                    kind = 1
    mass = 0.0
    frac = 0.0
    rate = 0.0
    surface_rain = 0.0
    cold = 0.0
    evap_total = 0.0
    eps_s = par[1]
    if kind > 0:
        # Profil du flux de masse (relatif) au sommet de chaque couche
        total_g = 0.0
        flux = 1.0
        for m in range(nz):
            k = n - m
            if k > k_base:
                # Sous la base : croissance linéaire en pression depuis le sol
                et[k] = (5000.0 + pi - (5000.0 + lev[k, 0] * pi)) / (5000.0 + pi - (5000.0 + lev[k_base + 1, 0] * pi))
                hu[k] = h_src
                qu[k] = q_src
            elif k >= k_top:
                dz = z[k] - z[k + 1]
                if kind == 2:
                    if k < k_base and wu[k] < wu[k + 1]:
                        flux = flux * wu[k] / wu[k + 1]
                    else:
                        flux = flux * gs.exp(par[15] * dz)
                else:
                    flux = (p[k] - p[k_top]) / gs.max(p[k_base] - p[k_top], 1.0)
                et[k] = flux if k > k_top else 0.0
                gr[k] = gr[k] * (et[k + 1] if k < n else 1.0)
                total_g += gr[k]
        # Fermeture
        if kind == 2:
            # Énergie cinétique des panaches pronostique (Pan & Randall 1998) : nourrie par la flottabilité, dissipée
            mass = (mass_old[j, i] + dt * cape / (2.0 * par[9])) / (1.0 + dt / (2.0 * par[8]))
        else:
            mass = 0.03 * wstar * p[k_base] / (287.04 * t[k_base])
        # Bornes : au plus 40 % de la masse d'une couche par pas
        for k in range(nz):
            if et[k] > 0.0:
                mass = gs.min(mass, 0.4 * gs.min(dm[k], dm[gs.max(k - 1, 0)]) / (et[k] * dt))
        mass = gs.min(mass, par[13])
        rate = total_g * mass
        # Courant descendant saturé, parti du minimum d'énergie statique humide
        down = 0.0
        h_d = 0.0
        q_l = 0.0
        k_d = -1
        if kind == 2 and rate > 1e-6:
            h_min = 1e9
            for k in range(nz):
                if k <= k_base and k > k_top and hh[k] < h_min and p[k] > 0.5 * (5000.0 + pi):
                    h_min = hh[k]
                    k_d = k
            if k_d >= 0:
                h_d = hh[k_d] + par[14] * (hu[k_d] - hh[k_d])
                down = par[6] * mass
                # Eau nécessaire pour rester saturé jusqu'au sol
                td = t[n]
                qs = 0.0
                for _ in range(8):
                    es = 611.2 * gs.exp(17.67 * (td - 273.15) / (td - 29.65))
                    qs = 0.62197 * es / gs.max(p[n] - 0.378 * es, 1.0)
                    f = 1004.64 * td + 9.81 * z[n] + 2.501e6 * par[12] * qs - h_d
                    td = td - f / (1004.64 + 2.501e6 * par[12] * qs * 4302.6 / ((td - 29.65) * (td - 29.65)))
                q_l = par[12] * qs
                need = down * gs.max(q_l - (q[k_d] + par[14] * (qu[k_d] - lu[k_d] - q[k_d])), 0.0)
                if need > 0.7 * rate:
                    down = down * 0.7 * rate / need
                cold = 9.81 * ((td - t[n]) / t[n] + 0.608 * (q_l - q[n]))
                if cold > -0.005:
                    down = 0.0
                    cold = 0.0
        # Humidité du courant descendant à chaque niveau : saturée (90 %) à énergie constante, sous k_d
        if down > 0.0:
            for k in range(nz):
                if k >= k_d:
                    td = t[k]
                    qs = 0.0
                    for _ in range(6):
                        es = 611.2 * gs.exp(17.67 * (td - 273.15) / (td - 29.65))
                        qs = 0.62197 * es / gs.max(p[k] - 0.378 * es, 1.0)
                        f = 1004.64 * td + 9.81 * z[k] + 2.501e6 * par[12] * qs - h_d
                        td = td - f / (1004.64 + 2.501e6 * par[12] * qs * 4302.6 / ((td - 29.65) * (td - 29.65)))
                    md[k] = par[12] * qs
        # Tendances en forme flux, puis pluie qui tombe et s'évapore
        rain = 0.0
        for k in range(nz):
            # Flux vers le haut de h et d'eau à travers le haut et le bas de la couche
            fh_up = 0.0
            fq_up = 0.0
            fh_lo = 0.0
            fq_lo = 0.0
            if k > 0:
                mu = mass * et[k]
                fh_up = mu * (hu[k] - hh[k - 1])
                fq_up = mu * (qu[k] - q[k - 1])
                if down > 0.0 and k > k_d:
                    md_f = down
                    if k > k_base:
                        md_f = down * et[k]
                    fh_up -= md_f * (h_d - hh[k])
                    fq_up -= md_f * (md[k - 1] - q[k])
            if k < n:
                mu = mass * et[k + 1]
                fh_lo = mu * (hu[k + 1] - hh[k])
                fq_lo = mu * (qu[k + 1] - q[k])
                if down > 0.0 and k + 1 > k_d:
                    md_f = down
                    if k + 1 > k_base:
                        md_f = down * et[k + 1]
                    fh_lo -= md_f * (h_d - hh[k + 1])
                    fq_lo -= md_f * (md[k] - q[k + 1])
            dh = (fh_lo - fh_up) * dt / dm[k]
            dq = (fq_lo - fq_up) * dt / dm[k]
            made = gr[k] * mass if k >= k_top and k <= k_base else 0.0
            dq -= made * dt / dm[k]
            rain += made
            # Eau évaporée dans le courant descendant : prise à la pluie
            ev_d = 0.0
            if down > 0.0 and k > k_d and k <= n:
                md_f = down
                if k > k_base:
                    md_f = down * et[k]
                ev_d = md_f * gs.max(md[k] - md[k - 1], 0.0)
            if down > 0.0 and k == k_d:
                ev_d = down * gs.max(md[k] - (q[k] + par[14] * (qu[k] - lu[k] - q[k])), 0.0)
            ev_d = gs.min(ev_d, rain)
            # Évaporation directe de la pluie dans l'air sous la base
            ev_e = 0.0
            if k > k_top and rain > 0.0:
                es = 611.2 * gs.exp(17.67 * (t[k] - 273.15) / (t[k] - 29.65))
                qse = 0.62197 * es / gs.max(p[k] - 0.378 * es, 1.0)
                ev_e = (par[11] if k > k_base else par[21]) * gs.max(1.0 - q[k] / qse, 0.0) * gs.sqrt(rain / 0.05) * 0.05 * dm[k]
                ev_e = gs.min(ev_e, 0.3 * gs.max(rain - ev_d, 0.0))
            rain -= ev_d + ev_e
            evap_total += ev_d + ev_e
            # L'eau évaporée revient à l'air à énergie constante (h compte la vapeur)
            dq += (ev_d + ev_e) * dt / dm[k]
            # Eau nuageuse détrainée
            dl = 0.0
            if k >= k_top and k <= k_base:
                det = gs.max((et[k + 1] if k < n else 1.0) - et[k], 0.0)
                dl = det * mass * lu[k] * dt / dm[k]
            dqv = dq - dl
            # Pas de vapeur négative
            dtk = (dh - 2.501e6 * dqv) / 1004.64
            gs.store(k, (t[k] + dtk) * gs.pow(p[k] * 1e-5, -0.285714))
            gs.store(nz + k, qv[k, j, i] + dqv)
            gs.store(2 * nz + k, qc[k, j, i] + dl)
            gs.store(3 * nz + k, mass * et[k])
        surface_rain = rain
        frac = down
    else:
        for k in range(nz):
            gs.store(k, th[k, j, i])
            gs.store(nz + k, qv[k, j, i])
            gs.store(2 * nz + k, qc[k, j, i])
            gs.store(3 * nz + k, 0.0)
    o = 4 * nz
    gs.store(o, surface_rain)
    gs.store(o + 1, cape)
    gs.store(o + 2, cin)
    gs.store(o + 3, z[k_base] if k_base >= 0 else 0.0)
    gs.store(o + 4, z[k_top] if kind > 0 else 0.0)
    memory = mass
    if kind < 2:
        memory = mass_old[j, i] / (1.0 + dt / (2.0 * par[8]))
    gs.store(o + 5, mass)
    gs.store(o + 12, memory)
    gs.store(o + 6, wmax if kind > 0 else 0.0)
    gs.store(o + 7, frac)
    gs.store(o + 8, cold)
    gs.store(o + 9, float(kind))
    gs.store(o + 10, evap_total)
    gs.store(o + 11, lfc)


def _condensation_columns(
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    dt: gs.f32,
    tpl: gs.f32["ny", "nb", "cc", "g"],
) -> gs.f32["ny", "nb", "cc", "g"]:
    """Nuages résolus : ajustement à saturation, conversion en pluie, chute immédiate avec évaporation (Sundqvist)."""
    j, b, c0, r = gs.index()
    i = b * 16 + r
    nz = th.shape[0]
    pi = pi0[j, i] + pip[j, i]
    rain = 0.0
    snow = 0.0
    for k in range(nz):
        p = 5000.0 + lev[k, 1] * pi
        ex = gs.pow(p * 1e-5, 0.285714)
        dm = lev[k, 2] * pi / 9.81
        v0 = qv[k, j, i]
        c1 = qc[k, j, i] + snow * dt / dm
        snow = 0.0
        t = th[k, j, i] * ex
        total = v0 + c1
        # Tout évaporer, puis condenser ce qui dépasse la saturation
        t = t - 2489.4489 * c1
        tl = t
        ql = 0.0
        qs = 0.0
        for it in range(6):
            es = 611.2 * gs.exp(17.67 * (t - 273.15) / (t - 29.65))
            des = es * 4302.6 / ((t - 29.65) * (t - 29.65))
            if t < 273.15:
                mix = gs.max((t - 250.16) / 23.0, 0.0)
                mix = gs.min(mix * mix, 1.0)
                ei = 611.2 * gs.exp(22.46 * (t - 273.15) / (t - 0.55))
                des = des * mix + (1.0 - mix) * ei * 6122.6 / ((t - 0.55) * (t - 0.55))
                es = es * mix + (1.0 - mix) * ei
            qs = 0.62197 * es / gs.max(p - 0.378 * es, 1.0)
            if it == 0 and total <= qs:
                break
            f = t - tl - 2489.4489 * (total - qs)
            t = t - f / (1.0 + 2489.4489 * qs / es * des)
        if total > qs:
            ql = gs.max(total - qs, 0.0)
            t = tl + 2489.4489 * ql
        vap = total - ql
        # Conversion (Kessler, seuil 0,3 g/kg) et chute
        made = 0.0
        thr = 3e-4
        if t < 258.0 and ql > 0.0:
            # Cristaux : chute de Heymsfield & Donner (1990) vers la couche du dessous, fraction glace de 258 à 238 K
            dens = p / (287.04 * t)
            fall = 3.29 * gs.pow(dens * ql, 0.16) * gs.min((258.0 - t) / 20.0, 1.0)
            made = ql * (1.0 - gs.exp(-fall * dt * dens / dm))
            ql -= made
            snow = made * dm / dt
            made = 0.0
        if ql > thr and t > 238.0:
            made = (ql - thr) * (1.0 - gs.exp(-1e-3 * dt)) * gs.min((t - 238.0) / 20.0, 1.0)
            ql -= made
        rain += made * dm / dt
        if ql <= 0.0 and rain > 0.0 and vap < qs:
            ev = 1e-5 * (1.0 - vap / qs) * gs.sqrt(rain) * dt
            ev = gs.min(gs.min(ev, rain * dt / dm), 0.5 * (qs - vap))
            vap += ev
            t -= 2489.4489 * ev
            rain -= ev * dm / dt
        gs.store(k, t / ex)
        gs.store(nz + k, vap)
        gs.store(2 * nz + k, ql)
    gs.store(3 * nz, rain + snow)


@gs.kernel
def wx_wind_tendency(
    wind: gs.f32["nz", "ny", "nx"], col: gs.f32["ny", "nb", "cc", "g"], field: gs.i32, comp: gs.i32
) -> gs.f32["nz", "ny", "nx"]:
    """Ajoute au vent des faces la moyenne des variations calculées au centre des mailles."""
    k, j, i = gs.index()
    nz = wind.shape[0]
    c = field * nz + k
    if comp == 0:
        io = (i - 1) % wind.shape[2]
        return wind[k, j, i] + 0.5 * (col[j, i // 16, c, i % 16] + col[j, io // 16, c, io % 16])
    if j == 0:
        return 0.0
    return wind[k, j, i] + 0.5 * (col[j, i // 16, c, i % 16] + col[j - 1, i // 16, c, i % 16])


@gs.kernel
def wx_surface_inputs(col: gs.f32["ny", "nb", "cc", "g"], tpl: gs.f32["ny", "nx", "c4"], base: gs.i32) -> gs.f32["ny", "nx", "c4"]:
    """Flux de chaleur, d'humidité, hauteur de couche limite et w* pour la convection."""
    j, i, c = gs.index()
    return col[j, i // 16, base + 1 + c, i % 16]


@gs.kernel
def wx_accumulate(
    acc: gs.f32["ny", "nx"], a: gs.f32["ny", "nb", "ca", "g"], ka: gs.i32, b: gs.f32["ny", "nb", "cb", "g"], kb: gs.i32, dt: gs.f32
) -> gs.f32["ny", "nx"]:
    """Cumul de pluie (mm) : convective et résolue."""
    j, i = gs.index()
    return acc[j, i] + dt * (a[j, i // 16, ka, i % 16] + b[j, i // 16, kb, i % 16])


@gs.kernel
def wx_convective_cover(cnv: gs.f32["ny", "nb", "cc", "g"], tpl: gs.f32["ny", "nx"], base: gs.i32) -> gs.f32["ny", "nx"]:
    """Couverture convective vue du soleil : 5 % d'un champ de cumulus, plus l'enclume des nuages profonds."""
    j, i = gs.index()
    kind = cnv[j, i // 16, base + 9, i % 16]
    mass = cnv[j, i // 16, base + 5, i % 16]
    wmax = gs.max(cnv[j, i // 16, base + 6, i % 16], 1.0)
    if kind > 1.5:
        return gs.min(0.1 + 25.0 * mass / wmax, 0.8)
    if kind > 0.5:
        return gs.min(4.0 * mass / wmax, 0.3)
    return 0.0


# ───────────────────────── Air froid des orages (couche 2D fine) ─────────────────────────


@gs.kernel
def wx_cold_pool(
    h: gs.f32["fy", "fx"],
    hu: gs.f32["fy", "fx"],
    hv: gs.f32["fy", "fx"],
    hb: gs.f32["fy", "fx"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    cnv: gs.f32["ny", "nb", "cc", "g"],
    terrain: gs.f32["ny", "nx"],
    base: gs.i32,
    refine: gs.i32,
    inv_dx: gs.f32,
    dt: gs.f32,
    epoch: gs.i32,
) -> (gs.f32["fy", "fx"], gs.f32["fy", "fx"], gs.f32["fy", "fx"], gs.f32["fy", "fx"]):
    """Étalement de l'air froid : eau peu profonde à flottabilité variable (équations de Ripa, flux de Rusanov).

    L'air apporté par les courants descendants naît dans une maille fine tirée au hasard (renouvelée à chaque `epoch`)
    de chaque maille large, avec la vitesse du vent bas. Il se réchauffe, frotte et se dilue."""
    j, i = gs.index()
    fy = h.shape[0]
    fx = h.shape[1]
    nz = u.shape[0]
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    h_c = h[j, i]
    inv_c = 1.0 / gs.max(h_c, 1.0)
    u_c = hu[j, i] * inv_c
    v_c = hv[j, i] * inv_c
    b_c = hb[j, i] * inv_c
    for side in range(4):
        jn = j
        i_n = i
        sx = 0.0
        sy = 0.0
        if side == 0:
            i_n = (i + 1) % fx
            sx = 1.0
        elif side == 1:
            i_n = (i - 1) % fx
            sx = -1.0
        elif side == 2:
            jn = gs.min(j + 1, fy - 1)
            sy = 1.0
        else:
            jn = gs.max(j - 1, 0)
            sy = -1.0
        h_n = h[jn, i_n]
        inv_n = 1.0 / gs.max(h_n, 1.0)
        u_n = hu[jn, i_n] * inv_n
        v_n = hv[jn, i_n] * inv_n
        b_n = hb[jn, i_n] * inv_n
        if jn == j and i_n == i:
            v_n = -v_n  # paroi
        un_c = u_c * sx + v_c * sy
        un_n = u_n * sx + v_n * sy
        wave = gs.max(gs.abs(un_c) + gs.sqrt(gs.max(b_c * h_c, 0.0)), gs.abs(un_n) + gs.sqrt(gs.max(b_n * h_n, 0.0)))
        p_c = 0.5 * b_c * h_c * h_c
        p_n = 0.5 * b_n * h_n * h_n
        f0 += 0.5 * (h_c * un_c + h_n * un_n) - 0.5 * wave * (h_n - h_c)
        f1 += 0.5 * (h_c * un_c * u_c + p_c * sx + h_n * un_n * u_n + p_n * sx) - 0.5 * wave * (h_n * u_n - h_c * u_c)
        f2 += 0.5 * (h_c * un_c * v_c + p_c * sy + h_n * un_n * v_n + p_n * sy) - 0.5 * wave * (h_n * v_n - h_c * v_c)
        f3 += 0.5 * (h_c * un_c * b_c + h_n * un_n * b_n) - 0.5 * wave * (h_n * b_n - h_c * b_c)
    r = dt * inv_dx
    h_new = gs.max(h_c - r * f0, 0.0)
    mu = hu[j, i] - r * f1
    mv = hv[j, i] - r * f2
    mb = gs.max(hb[j, i] - r * f3, 0.0)
    # Frottement, réchauffement, dilution
    speed = gs.sqrt(u_c * u_c + v_c * v_c)
    drag = 1.0 / (1.0 + dt * 3e-3 * speed / gs.max(h_c, 50.0))
    mu = mu * drag
    mv = mv * drag
    mb = mb * gs.exp(-dt / 14400.0)
    keep = gs.exp(-dt / 10800.0)
    h_new = h_new * keep
    mu = mu * keep
    mv = mv * keep
    mb = mb * keep
    # Source
    cj = j // refine
    ci = i // refine
    down = cnv[cj, ci // 16, base + 7, ci % 16]
    if down > 0.0:
        mix_ = gs.u32(cj * 8192 + ci) + gs.u32(epoch) * gs.u32(2654435769)
        for _ in range(2):
            mix_ = (mix_ ^ (mix_ >> gs.u32(16))) * gs.u32(73244475)
        mix_ = mix_ ^ (mix_ >> gs.u32(16))
        pick = gs.i32(mix_ % gs.u32(refine * refine))
        if pick == (j - cj * refine) * refine + (i - ci * refine):
            add = down / 1.15 * float(refine * refine) * dt
            k = nz - 4
            h_new += add
            mu += add * u[k, cj, ci]
            mv += add * v[k, cj, ci]
            mb += add * gs.max(-cnv[cj, ci // 16, base + 8, ci % 16], 0.0)
    if h_new < 1.0 or mb < 0.003 * h_new:
        return 0.0, 0.0, 0.0, 0.0
    return h_new, mu, mv, mb


@gs.kernel
def wx_front_lift(
    h: gs.f32["fy", "fx"],
    hu: gs.f32["fy", "fx"],
    hv: gs.f32["fy", "fx"],
    hb: gs.f32["fy", "fx"],
    col: gs.f32["ny", "nb", "cd", "g"],
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    lev: gs.f32["nl", "c4"],
    terrain: gs.f32["ny", "nx"],
    refine: gs.i32,
    inv_dx: gs.f32,
    front: gs.f32,
    resolved: gs.f32,
) -> gs.f32["ny", "nx"]:
    """Vitesse de soulèvement (m/s) : bord d'une poche d'air froid, ascendance résolue vers 850 hPa, pente du relief."""
    j, i = gs.index()
    fy = h.shape[0]
    fx = h.shape[1]
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    best = 0.0
    for a in range(refine):
        for b in range(refine):
            fj = j * refine + a
            fi = i * refine + b
            if h[fj, fi] < 60.0:
                for side in range(4):
                    jn = fj
                    i_n = fi
                    if side == 0:
                        i_n = (fi + 1) % fx
                    elif side == 1:
                        i_n = (fi - 1) % fx
                    elif side == 2:
                        jn = gs.min(fj + 1, fy - 1)
                    else:
                        jn = gs.max(fj - 1, 0)
                    if h[jn, i_n] > 100.0:
                        best = gs.max(best, front * gs.sqrt(gs.max(hb[jn, i_n], 0.0)))
    pi = pi0[j, i] + pip[j, i]
    # Pression décroissante en suivant l'air = ascendance : omega ≈ pi · dσ/dt vers σ = 0,85
    k = nz - 1
    while k > 0 and lev[k, 0] > 0.85:
        k -= 1
    om = col[j, i // 16, nz + k - 1, i % 16]
    up = gs.max(-om / (9.81 * 1.05), 0.0)
    # Vent contre la pente
    km = nz - 3
    ie = (i + 1) % nx
    iw = (i - 1) % nx
    jn2 = gs.min(j + 1, ny - 1)
    js2 = gs.max(j - 1, 0)
    slope = (u[km, j, i] * (terrain[j, ie] - terrain[j, iw]) + v[km, j, i] * (terrain[jn2, i] - terrain[js2, i])) * 0.5 * inv_dx
    up += gs.max(slope, 0.0)
    return best + resolved * up



# ───────────────────────── Raccord avec la simulation 3D ─────────────────────────


@gs.kernel
def wx_sample_column(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    th: gs.f32["nz", "ny", "nx"],
    qv: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    th_a: gs.f32["nz", "ny", "nx"],
    q_a: gs.f32["nz", "ny", "nx"],
    l_a: gs.f32["nz", "ny", "nx"],
    th_b: gs.f32["nz", "ny", "nx"],
    q_b: gs.f32["nz", "ny", "nx"],
    l_b: gs.f32["nz", "ny", "nx"],
    dyn: gs.f32["ny", "nb", "cd", "g"],
    x: gs.f32,
    y: gs.f32,
    width: gs.f32,
    tpl: gs.f32["nz", "c9"],
) -> gs.f32["nz", "c9"]:
    """Colonne autour du point `(x, y)` (en mailles), pondérée par une gaussienne : θ, qv, qc, u, v, πσ̇ (milieu de
    couche), variation de θ et d'eau pendant les pas dynamiques (`a` → `b`), poids."""
    k, c = gs.index()
    ny = u.shape[1]
    nx = u.shape[2]
    i0 = int(gs.floor(x))
    j0 = int(gs.floor(y))
    total = 0.0
    weight = 0.0
    for a in range(6):
        for b in range(6):
            j = j0 + a - 2
            i = i0 + b - 2
            if j >= 0 and j < ny:
                dx = float(i) + 0.5 - x
                dy = float(j) + 0.5 - y
                w = gs.exp(-(dx * dx + dy * dy) / (2.0 * width * width))
                i = i % nx
                val = 0.0
                if c == 0:
                    val = th[k, j, i]
                elif c == 1:
                    val = qv[k, j, i]
                elif c == 2:
                    val = qc[k, j, i]
                elif c == 3:
                    val = 0.5 * (u[k, j, i] + u[k, j, (i + 1) % nx])
                elif c == 4:
                    val = 0.5 * (v[k, j, i] + (v[k, j + 1, i] if j < ny - 1 else 0.0))
                elif c == 5:
                    val = 0.5 * dyn[j, i // 16, u.shape[0] + k, i % 16]
                    if k > 0:
                        val += 0.5 * dyn[j, i // 16, u.shape[0] + k - 1, i % 16]
                elif c == 6:
                    val = th_b[k, j, i] - th_a[k, j, i]
                elif c == 7:
                    val = q_b[k, j, i] - q_a[k, j, i] + (l_b[k, j, i] - l_a[k, j, i])
                else:
                    val = 1.0
                total += w * val
                weight += w
    if c == 8:
        return weight
    return total / weight


@gs.kernel
def wx_sample_surface(
    pip: gs.f32["ny", "nx"],
    pi0: gs.f32["ny", "nx"],
    surface: gs.f32["ny", "nb", "ca", "g"],
    cnv: gs.f32["ny", "nb", "cb", "g"],
    cond: gs.f32["ny", "nb", "cl", "g"],
    surf: gs.f32["ny", "nx", "c8"],
    nz: gs.i32,
    x: gs.f32,
    y: gs.f32,
    width: gs.f32,
    tpl: gs.f32["c10"],
) -> gs.f32["c10"]:
    """Au sol, même pondération : pression, flux de chaleur et d'humidité, frottement, altitude, T à 2 m, nébulosité,
    pluie (kg/m²/s), hauteur de couche limite, part de terre."""
    c = gs.index()
    ny = pip.shape[0]
    nx = pip.shape[1]
    i0 = int(gs.floor(x))
    j0 = int(gs.floor(y))
    total = 0.0
    weight = 0.0
    for a in range(6):
        for b in range(6):
            j = j0 + a - 2
            i = i0 + b - 2
            if j >= 0 and j < ny:
                dx = float(i) + 0.5 - x
                dy = float(j) + 0.5 - y
                w = gs.exp(-(dx * dx + dy * dy) / (2.0 * width * width))
                i = i % nx
                bb = i // 16
                r = i % 16
                o = 5 * nz
                val = 0.0
                if c == 0:
                    val = pip[j, i] + (pi0[j, i] + 5000.0 - 100000.0)
                elif c == 1:
                    val = surface[j, bb, o + 1, r]
                elif c == 2:
                    val = surface[j, bb, o + 2, r]
                elif c == 3:
                    val = surface[j, bb, o + 7, r]
                elif c == 4:
                    val = surf[j, i, 6]
                elif c == 5:
                    val = surface[j, bb, o + 9, r]
                elif c == 6:
                    val = surface[j, bb, o + 6, r]
                elif c == 7:
                    val = cnv[j, bb, 4 * nz, r] + cond[j, bb, 3 * nz, r]
                elif c == 8:
                    val = surface[j, bb, o + 3, r]
                else:
                    val = surf[j, i, 0]
                total += w * val
                weight += w
    return total / weight


# ───────────────────────── Carte ─────────────────────────


@gs.kernel
def wx_qnh(pip: gs.f32["ny", "nx"], pi0: gs.f32["ny", "nx"], terrain: gs.f32["ny", "nx"]) -> gs.f32["ny", "nx"]:
    """QNH (hPa), atmosphère standard, lissé sur les mailles voisines."""
    j, i = gs.index()
    ny = pip.shape[0]
    nx = pip.shape[1]
    total = 0.0
    weight = 0.0
    for a in range(3):
        for b in range(3):
            jj = gs.min(gs.max(j + a - 1, 0), ny - 1)
            ii = (i + b - 1) % nx
            p = 5000.0 + pi0[jj, ii] + pip[jj, ii]
            w = (2.0 - gs.abs(float(a - 1))) * (2.0 - gs.abs(float(b - 1)))
            total += w * 1013.25 * gs.pow(gs.pow(p / 101325.0, 0.190263) + 0.0065 * terrain[jj, ii] / 288.15, 5.255876)
            weight += w
    return total / weight


@gs.kernel
def wx_map(
    qnh: gs.f32["ny", "nx"],
    surface: gs.f32["ny", "nb", "ca", "g"],
    cnv: gs.f32["ny", "nb", "cb", "g"],
    cond: gs.f32["ny", "nb", "cl", "g"],
    surf: gs.f32["ny", "nx", "c8"],
    pool: gs.f32["fy", "fx"],
    nz: gs.i32,
    mode: gs.i32,
    lines: gs.i32,
    tpl: gs.u32["h", "w"],
) -> gs.u32["h", "w"]:
    """Carte : relief et mer, nuages (mode 0) ou température à 2 m (mode 1), pluie, bord de l'air froid, isobares (4 hPa)."""
    y, x = gs.index()
    ny = qnh.shape[0]
    nx = qnh.shape[1]
    hh = float(tpl.shape[0])
    ww = float(tpl.shape[1])
    gx = (float(x) + 0.5) / ww * float(nx) - 0.5
    gy = (1.0 - (float(y) + 0.5) / hh) * float(ny) - 0.5
    val = gs.local(gs.f32, 3)
    land = 0.0
    elev = 0.0
    cloud = 0.0
    t2 = 0.0
    rain = 0.0
    crain = 0.0
    # Interpolation au pixel et à ses voisins de droite et du bas (pente des isobares)
    for n in range(3):
        px = gx + (float(nx) / ww if n == 1 else 0.0)
        py = gy + (float(ny) / hh if n == 2 else 0.0)
        py = gs.min(gs.max(py, 0.0), float(ny) - 1.001)
        i0 = int(gs.floor(px))
        j0 = int(gs.floor(py))
        fx = px - float(i0)
        fy = py - float(j0)
        acc = 0.0
        for a in range(2):
            for b in range(2):
                jj = j0 + a
                ii = (i0 + b) % nx
                wt = (fy if a == 1 else 1.0 - fy) * (fx if b == 1 else 1.0 - fx)
                acc += wt * qnh[jj, ii]
                if n == 0:
                    bb = ii // 16
                    rr = ii % 16
                    land += wt * surf[jj, ii, 0]
                    elev += wt * surf[jj, ii, 6]
                    cloud += wt * surface[jj, bb, 5 * nz + 6, rr]
                    t2 += wt * surface[jj, bb, 5 * nz + 9, rr]
                    rain += wt * cond[jj, bb, 3 * nz, rr] * 3600.0
                    crain += wt * cnv[jj, bb, 4 * nz, rr] * 3600.0
        val[n] = acc
    # Fond
    r = 0.05
    g = 0.16
    b_ = 0.33
    if land > 0.5:
        e = gs.min(elev / 2500.0, 1.0)
        r = 0.22 + 0.4 * e
        g = 0.33 + 0.16 * e
        b_ = 0.16 + 0.2 * e
    if mode == 1:
        # Bleu, gris neutre vers 10 °C, rouge
        s = gs.min(gs.max((t2 - 283.15) / 25.0, -1.0), 1.0)
        base = 0.0 if s < 0.0 else 1.0
        m = gs.abs(s)
        r = 0.93 + m * ((0.10 if base < 0.5 else 0.72) - 0.93)
        g = 0.93 + m * ((0.30 if base < 0.5 else 0.22) - 0.93)
        b_ = 0.92 + m * ((0.70 if base < 0.5 else 0.16) - 0.92)
        if gs.abs(land - 0.5) < 0.12:
            r = 0.25
            g = 0.25
            b_ = 0.25
    else:
        c = gs.min(cloud, 1.0) * 0.85
        r += (0.95 - r) * c
        g += (0.96 - g) * c
        b_ += (0.97 - b_) * c
    if rain > 0.1:
        ink_a = gs.min(0.35 + 0.2 * gs.log(rain * 10.0), 0.9)
        r += (0.08 - r) * ink_a
        g += (0.45 - g) * ink_a
        b_ += (0.25 - b_) * ink_a
    if crain > 0.2:
        ink_a = gs.min(0.4 + 0.15 * gs.log(crain * 5.0), 0.95)
        r += (0.92 - r) * ink_a
        g += ((0.60 if crain < 2.0 else 0.22) - g) * ink_a
        b_ += (0.12 - b_) * ink_a
    fj = gs.min(int((1.0 - (float(y) + 0.5) / hh) * float(pool.shape[0])), pool.shape[0] - 1)
    fi = gs.min(int((float(x) + 0.5) / ww * float(pool.shape[1])), pool.shape[1] - 1)
    if pool[fj, fi] > 100.0:
        edge = gs.min(pool[fj, (fi + 1) % pool.shape[1]], pool[fj, (fi - 1) % pool.shape[1]])
        edge = gs.min(edge, gs.min(pool[gs.min(fj + 1, pool.shape[0] - 1), fi], pool[gs.max(fj - 1, 0), fi]))
        if edge < 100.0:
            r = 0.42
            g = 0.11
            b_ = 0.60
    if lines != 0:
        level = val[0] / 4.0
        slope = gs.sqrt((val[1] - val[0]) * (val[1] - val[0]) + (val[2] - val[0]) * (val[2] - val[0])) / 4.0
        dist = gs.abs(level - gs.floor(level + 0.5)) / gs.max(slope, 1e-5)
        if dist < 0.8:
            ink = 1.0 - gs.max(dist - 0.3, 0.0) / 0.5
            r = r * (1.0 - ink) + 0.02 * ink
            g = g * (1.0 - ink) + 0.02 * ink
            b_ = b_ * (1.0 - ink) + 0.05 * ink
    packed = gs.u32(4278190080)
    packed = packed | gs.u32(gs.min(gs.max(r, 0.0), 1.0) * 255.0 + 0.5)
    packed = packed | (gs.u32(gs.min(gs.max(g, 0.0), 1.0) * 255.0 + 0.5) << gs.u32(8))
    packed = packed | (gs.u32(gs.min(gs.max(b_, 0.0), 1.0) * 255.0 + 0.5) << gs.u32(16))
    return packed
