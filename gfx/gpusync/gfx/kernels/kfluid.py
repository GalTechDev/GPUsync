"""Kernels `@gs.kernel` 100 % GPU pour les 3 moteurs fluides de `gpusync.gfx` :
1. Surface d'eau 3D (équation des ondes 2D + normales lisses ou flat-shaded low-poly)
2. Fluide particulaire 3D SPH (densité, pression, viscosité, cohésion → TRS GPU)
3. Fluide 2D eulérien Navier-Stokes / Stable Fluids (advection, vorticité, Jacobi, teinture, cel-shading)
"""

from __future__ import annotations

import gpusync as gs


# ───────────────────────── 1. Surface d'eau 3D (équation des ondes) ─────────────────────────


@gs.kernel
def water_wave_step(
    h_curr: gs.f32["gh", "gw"],
    h_prev: gs.f32["gh", "gw"],
    c2: gs.f32,
    damping: gs.f32,
    smooth: gs.f32,
    absorb_border: gs.i32,
) -> gs.f32["gh", "gw"]:
    """Équation des ondes 2D, un pas (`c2` = (vitesse · dt / cellule)², bords réfléchissants ou absorbants).

    L'amortissement et le lissage agissent sur la vitesse (`h_curr - h_prev`) et non sur la hauteur : en bassin
    fermé, le niveau moyen ne bouge pas."""
    y, x = gs.index()
    gh = h_curr.shape[0]
    gw = h_curr.shape[1]

    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, gw - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, gh - 1)

    hc = h_curr[y, x]
    hp = h_prev[y, x]
    lap = h_curr[y, xm] + h_curr[y, xp] + h_curr[ym, x] + h_curr[yp, x] - 4.0 * hc
    lap_p = h_prev[y, xm] + h_prev[y, xp] + h_prev[ym, x] + h_prev[yp, x] - 4.0 * hp
    # `smooth` éteint d'abord les ondes de la taille d'une cellule
    hn = hc + (hc - hp) * damping + c2 * lap + smooth * (lap - lap_p)

    if absorb_border != 0:
        bmin = gs.min(gs.min(x, gw - 1 - x), gs.min(y, gh - 1 - y))
        if bmin < 5:
            hn = hn * (0.70 + 0.06 * float(bmin))

    return gs.min(gs.max(hn, -2.5), 2.5)


@gs.kernel
def water_rescale_prev(h_curr: gs.f32["gh", "gw"], h_prev: gs.f32["gh", "gw"], ratio: gs.f32) -> gs.f32["gh", "gw"]:
    """Multiplie la vitesse (`h_curr - h_prev`) par `ratio`."""
    y, x = gs.index()
    return h_curr[y, x] - (h_curr[y, x] - h_prev[y, x]) * ratio


@gs.kernel
def water_add_splash(
    h: gs.f32["gh", "gw"],
    cx: gs.f32,
    cy: gs.f32,
    inv_r2: gs.f32,
    strength: gs.f32,
    ring: gs.f32,
) -> gs.f32["gh", "gw"]:
    """Creux (ou bosse) circulaire, entouré d'une couronne de signe opposé (`ring`) qui rend le même volume d'eau."""
    y, x = gs.index()
    dx = float(x) - cx
    dy = float(y) - cy
    d2 = (dx * dx + dy * dy) * inv_r2
    w = gs.exp(-d2) - ring * gs.exp(-d2 * 0.3)
    return gs.min(gs.max(h[y, x] + strength * w, -2.5), 2.5)


@gs.kernel
def water_build_smooth_mesh(
    base_pos: gs.f32["nv", "c3"],
    h_grid: gs.f32["gh", "gw"],
    cell_size: gs.f32,
    h_scale: gs.f32,
    amb_amp: gs.f32,
    amb_freq: gs.f32,
    amb_t: gs.f32,
) -> (gs.f32["nv", "c3"], gs.f32["nv", "c3"]):
    """Écrit directement les sommets et normales lisses (style réaliste) dans `Object3D.gpu_positions/normals`."""
    v, c = gs.index()
    gh = h_grid.shape[0]
    gw = h_grid.shape[1]

    gy = v // gw
    gx = v - gy * gw
    gy = gs.min(gs.max(gy, 0), gh - 1)
    gx = gs.min(gs.max(gx, 0), gw - 1)

    bx = base_pos[v, 0]
    bz = base_pos[v, 2]

    xm = gs.max(gx - 1, 0)
    xp = gs.min(gx + 1, gw - 1)
    ym = gs.max(gy - 1, 0)
    yp = gs.min(gy + 1, gh - 1)

    h_c = h_grid[gy, gx] * h_scale
    h_l = h_grid[gy, xm] * h_scale
    h_r = h_grid[gy, xp] * h_scale
    h_d = h_grid[ym, gx] * h_scale
    h_u = h_grid[yp, gx] * h_scale

    # Houle procédurale ambiante optionnelle (mer / océan)
    y_amb = 0.0
    dax = 0.0
    daz = 0.0
    if amb_amp > 0.0:
        p1 = bx * amb_freq + bz * (amb_freq * 0.55) + amb_t
        p2 = bx * (-amb_freq * 0.72) + bz * (amb_freq * 1.15) - amb_t * 0.82
        y_amb = amb_amp * (0.62 * gs.sin(p1) + 0.38 * gs.cos(p2))
        dax = amb_amp * (0.62 * amb_freq * gs.cos(p1) + 0.38 * (amb_freq * 0.72) * gs.sin(p2))
        daz = amb_amp * (0.62 * (amb_freq * 0.55) * gs.cos(p1) - 0.38 * (amb_freq * 1.15) * gs.sin(p2))

    y_pos = h_c + y_amb
    inv_2d = 0.5 / gs.max(cell_size, 1e-4)
    dhdx = (h_r - h_l) * inv_2d + dax
    dhdz = (h_u - h_d) * inv_2d + daz
    inv_n = 1.0 / gs.sqrt(dhdx * dhdx + 1.0 + dhdz * dhdz)

    p_out = bx
    n_out = -dhdx * inv_n
    if c == 1:
        p_out = y_pos
        n_out = inv_n
    elif c == 2:
        p_out = bz
        n_out = -dhdz * inv_n
    return p_out, n_out


@gs.kernel
def water_build_flat_mesh(
    base_pos: gs.f32["nv", "c3"],
    h_grid: gs.f32["gh", "gw"],
    half_size: gs.f32,
    inv_size: gs.f32,
    h_scale: gs.f32,
    amb_amp: gs.f32,
    amb_freq: gs.f32,
    amb_t: gs.f32,
) -> (gs.f32["nv", "c3"], gs.f32["nv", "c3"]):
    """Écrit un maillage non-indexé à facettes nettes (style low-poly / cristal) sur GPU."""
    v, c = gs.index()
    gh = h_grid.shape[0]
    gw = h_grid.shape[1]
    fgw = float(gw - 1)
    fgh = float(gh - 1)

    tri = v // 3
    v0 = tri * 3
    v1 = v0 + 1
    v2 = v0 + 2

    x0 = base_pos[v0, 0]
    z0 = base_pos[v0, 2]
    x1 = base_pos[v1, 0]
    z1 = base_pos[v1, 2]
    x2 = base_pos[v2, 0]
    z2 = base_pos[v2, 2]

    gx0 = gs.min(gs.max(int((x0 + half_size) * inv_size * fgw + 0.5), 0), gw - 1)
    gy0 = gs.min(gs.max(int((z0 + half_size) * inv_size * fgh + 0.5), 0), gh - 1)
    gx1 = gs.min(gs.max(int((x1 + half_size) * inv_size * fgw + 0.5), 0), gw - 1)
    gy1 = gs.min(gs.max(int((z1 + half_size) * inv_size * fgh + 0.5), 0), gh - 1)
    gx2 = gs.min(gs.max(int((x2 + half_size) * inv_size * fgw + 0.5), 0), gw - 1)
    gy2 = gs.min(gs.max(int((z2 + half_size) * inv_size * fgh + 0.5), 0), gh - 1)

    y0 = h_grid[gy0, gx0] * h_scale
    y1 = h_grid[gy1, gx1] * h_scale
    y2 = h_grid[gy2, gx2] * h_scale

    if amb_amp > 0.0:
        y0 += amb_amp * (0.62 * gs.sin(x0 * amb_freq + z0 * amb_freq * 0.55 + amb_t) + 0.38 * gs.cos(-x0 * amb_freq * 0.72 + z0 * amb_freq * 1.15 - amb_t * 0.82))
        y1 += amb_amp * (0.62 * gs.sin(x1 * amb_freq + z1 * amb_freq * 0.55 + amb_t) + 0.38 * gs.cos(-x1 * amb_freq * 0.72 + z1 * amb_freq * 1.15 - amb_t * 0.82))
        y2 += amb_amp * (0.62 * gs.sin(x2 * amb_freq + z2 * amb_freq * 0.55 + amb_t) + 0.38 * gs.cos(-x2 * amb_freq * 0.72 + z2 * amb_freq * 1.15 - amb_t * 0.82))

    # Normale exacte de la facette triangulaire (garantie orientée vers le haut +Y)
    e1x = x1 - x0
    e1y = y1 - y0
    e1z = z1 - z0
    e2x = x2 - x0
    e2y = y2 - y0
    e2z = z2 - z0
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    if ny < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz
    inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx = nx * inv_n
    ny = ny * inv_n
    nz = nz * inv_n

    # Position du sommet courant v
    px = x0
    py = y0
    pz = z0
    if v == v1:
        px = x1
        py = y1
        pz = z1
    elif v == v2:
        px = x2
        py = y2
        pz = z2

    p_out = px
    n_out = nx
    if c == 1:
        p_out = py
        n_out = ny
    elif c == 2:
        p_out = pz
        n_out = nz
    return p_out, n_out


# ───────────────────────── 2. Fluide particulaire 3D SPH ─────────────────────────


@gs.kernel
def sph_to_trs(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    trs_tpl: gs.f32["n", "c8"],
    base_scale: gs.f32,
    vel_stretch: gs.f32,
    spin_rate: gs.f32,
    t: gs.f32,
) -> gs.f32["n", "c8"]:
    """Convertit les particules SPH en matrices TRS `(N, 8)` directement en VRAM pour `InstancedMesh3D`."""
    i, c = gs.index()
    val = pos[i, 0] + trs_tpl[i, c] * 0.0
    if c == 1:
        val = pos[i, 1]
    elif c == 2:
        val = pos[i, 2]
    elif c == 3:
        val = spin_rate * (t + float(i) * 0.17)
    elif c == 4:
        val = spin_rate * (t * 1.3 + float(i) * 0.29)
    elif c == 5:
        val = spin_rate * 0.6 * (t + float(i) * 0.11)
    elif c == 6:
        sp = gs.sqrt(vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1] + vel[i, 2] * vel[i, 2])
        val = base_scale * (1.0 + gs.min(sp * vel_stretch, 0.45))
    elif c == 7:
        val = 1.0
    return val


# ───────────────────────── 3. Fluide 2D Eulérien (Navier-Stokes / Stable Fluids) ─────────────────────────


@gs.kernel
def fluid2d_advect_vel(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
    dr: gs.f32["h", "w"],
    dg: gs.f32["h", "w"],
    db: gs.f32["h", "w"],
    dt: gs.f32,
    dissipation: gs.f32,
    buoyancy: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Advection semi-lagrangienne bilinéaire du champ de vitesse + flottabilité thermique optionnelle."""
    y, x = gs.index()
    h = vx.shape[0]
    w = vx.shape[1]

    px = gs.min(gs.max(float(x) - vx[y, x] * dt, 0.5), float(w) - 1.5)
    py = gs.min(gs.max(float(y) - vy[y, x] * dt, 0.5), float(h) - 1.5)

    x0 = int(gs.floor(px))
    y0 = int(gs.floor(py))
    x1 = gs.min(x0 + 1, w - 1)
    y1 = gs.min(y0 + 1, h - 1)
    tx = px - float(x0)
    ty = py - float(y0)
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    nvx = (w00 * vx[y0, x0] + w10 * vx[y0, x1] + w01 * vx[y1, x0] + w11 * vx[y1, x1]) * dissipation
    nvy = (w00 * vy[y0, x0] + w10 * vy[y0, x1] + w01 * vy[y1, x0] + w11 * vy[y1, x1]) * dissipation

    if buoyancy != 0.0:
        heat = (dr[y, x] + dg[y, x] + db[y, x]) * 0.3333333
        nvy -= heat * buoyancy

    return nvx, nvy


@gs.kernel
def fluid2d_curl(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
) -> gs.f32["h", "w"]:
    """Calcule la vorticité scalaire (rotationnel 2D) pour le confinement de tourbillons."""
    y, x = gs.index()
    h = vx.shape[0]
    w = vx.shape[1]
    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, w - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, h - 1)
    return 0.5 * ((vy[y, xp] - vy[y, xm]) - (vx[yp, x] - vx[ym, x]))


@gs.kernel
def fluid2d_vorticity(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
    curl: gs.f32["h", "w"],
    strength: gs.f32,
    dt: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Injecte la force de confinement de vorticité (rend les volutes fines et vivantes)."""
    y, x = gs.index()
    h = vx.shape[0]
    w = vx.shape[1]
    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, w - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, h - 1)

    c_l = gs.abs(curl[y, xm])
    c_r = gs.abs(curl[y, xp])
    c_d = gs.abs(curl[ym, x])
    c_u = gs.abs(curl[yp, x])
    c_c = curl[y, x]

    dx = c_r - c_l
    dy = c_u - c_d
    inv_l = 1.0 / (gs.sqrt(dx * dx + dy * dy) + 1e-4)
    nx = dx * inv_l
    ny = dy * inv_l

    fx = ny * c_c * strength * dt
    fy = -nx * c_c * strength * dt
    return vx[y, x] + fx, vy[y, x] + fy


@gs.kernel
def fluid2d_divergence(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
) -> gs.f32["h", "w"]:
    """Calcule la divergence du champ de vitesse."""
    y, x = gs.index()
    h = vx.shape[0]
    w = vx.shape[1]
    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, w - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, h - 1)
    return 0.5 * ((vx[y, xp] - vx[y, xm]) + (vy[yp, x] - vy[ym, x]))


@gs.kernel
def fluid2d_pressure_jacobi(
    p: gs.f32["h", "w"],
    div: gs.f32["h", "w"],
) -> gs.f32["h", "w"]:
    """Une itération de relaxation de Jacobi pour l'équation de Poisson de pression."""
    y, x = gs.index()
    h = p.shape[0]
    w = p.shape[1]
    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, w - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, h - 1)
    return 0.25 * (p[y, xm] + p[y, xp] + p[ym, x] + p[yp, x] - div[y, x])


@gs.kernel
def fluid2d_subtract_grad(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
    p: gs.f32["h", "w"],
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Projette la vitesse sur un champ incompressible (∇·v = 0) avec bords étanches."""
    y, x = gs.index()
    h = vx.shape[0]
    w = vx.shape[1]
    xm = gs.max(x - 1, 0)
    xp = gs.min(x + 1, w - 1)
    ym = gs.max(y - 1, 0)
    yp = gs.min(y + 1, h - 1)

    nvx = gs.min(gs.max(vx[y, x] - 0.5 * (p[y, xp] - p[y, xm]), -8.0), 8.0)
    nvy = gs.min(gs.max(vy[y, x] - 0.5 * (p[yp, x] - p[ym, x]), -8.0), 8.0)
    if x == 0 or x == w - 1:
        nvx = 0.0
    if y == 0 or y == h - 1:
        nvy = 0.0
    return nvx, nvy


@gs.kernel
def fluid2d_advect_dye(
    dr: gs.f32["h", "w"],
    dg: gs.f32["h", "w"],
    db: gs.f32["h", "w"],
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
    dt: gs.f32,
    dissipation: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Advecte les 3 canaux d'encre/fumée RGB le long du champ de vitesse incompressible."""
    y, x = gs.index()
    h = dr.shape[0]
    w = dr.shape[1]

    px = gs.min(gs.max(float(x) - vx[y, x] * dt, 0.5), float(w) - 1.5)
    py = gs.min(gs.max(float(y) - vy[y, x] * dt, 0.5), float(h) - 1.5)
    x0 = int(gs.floor(px))
    y0 = int(gs.floor(py))
    x1 = gs.min(x0 + 1, w - 1)
    y1 = gs.min(y0 + 1, h - 1)
    tx = px - float(x0)
    ty = py - float(y0)
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    nr = (w00 * dr[y0, x0] + w10 * dr[y0, x1] + w01 * dr[y1, x0] + w11 * dr[y1, x1]) * dissipation
    ng = (w00 * dg[y0, x0] + w10 * dg[y0, x1] + w01 * dg[y1, x0] + w11 * dg[y1, x1]) * dissipation
    nb = (w00 * db[y0, x0] + w10 * db[y0, x1] + w01 * db[y1, x0] + w11 * db[y1, x1]) * dissipation
    return nr, ng, nb


@gs.kernel
def fluid2d_splat_vel(
    vx: gs.f32["h", "w"],
    vy: gs.f32["h", "w"],
    cx: gs.f32,
    cy: gs.f32,
    fx: gs.f32,
    fy: gs.f32,
    inv_r2: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Injecte une impulsion de vitesse gaussienne à `(cx, cy)`."""
    y, x = gs.index()
    dx = float(x) - cx
    dy = float(y) - cy
    w = gs.exp(-(dx * dx + dy * dy) * inv_r2)
    return vx[y, x] + fx * w, vy[y, x] + fy * w


@gs.kernel
def fluid2d_splat_dye(
    dr: gs.f32["h", "w"],
    dg: gs.f32["h", "w"],
    db: gs.f32["h", "w"],
    cx: gs.f32,
    cy: gs.f32,
    cr: gs.f32,
    cg: gs.f32,
    cb: gs.f32,
    inv_r2: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Injecte de l'encre / fumée colorée à `(cx, cy)`."""
    y, x = gs.index()
    dx = float(x) - cx
    dy = float(y) - cy
    w = gs.exp(-(dx * dx + dy * dy) * inv_r2)
    return gs.min(dr[y, x] + cr * w, 4.0), gs.min(dg[y, x] + cg * w, 4.0), gs.min(db[y, x] + cb * w, 4.0)


@gs.kernel
def fluid2d_render_u32(
    dr: gs.f32["h", "w"],
    dg: gs.f32["h", "w"],
    db: gs.f32["h", "w"],
    bg_r: gs.f32,
    bg_g: gs.f32,
    bg_b: gs.f32,
    exposure: gs.f32,
    posterize: gs.f32,
) -> gs.u32["h", "w"]:
    """Empaquette le champ de teinture RGB en texture `u32` RGBA8 (lisse ou cel-shaded / low-poly)."""
    y, x = gs.index()
    r = (bg_r + dr[y, x]) * exposure
    g = (bg_g + dg[y, x]) * exposure
    b = (bg_b + db[y, x]) * exposure

    # Tonemapping doux (Reinhard) pour éviter la saturation dure sur les croisements d'encre
    r = r / (1.0 + r * 0.55)
    g = g / (1.0 + g * 0.55)
    b = b / (1.0 + b * 0.55)

    r = gs.min(gs.max(r, 0.0), 1.0)
    g = gs.min(gs.max(g, 0.0), 1.0)
    b = gs.min(gs.max(b, 0.0), 1.0)

    if posterize > 1.0:
        r = gs.floor(r * posterize + 0.5) / posterize
        g = gs.floor(g * posterize + 0.5) / posterize
        b = gs.floor(b * posterize + 0.5) / posterize

    ri = int(gs.sqrt(r) * 255.0 + 0.5)
    gi = int(gs.sqrt(g) * 255.0 + 0.5)
    bi = int(gs.sqrt(b) * 255.0 + 0.5)
    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


# ───────────────────────── 4. Rendu Volumétrique Liquide Screen-Space (SSFR 100 % GPU) ─────────────────────────

EMPTY_FLUID_DEPTH = 4294967295


@gs.kernel(grid=("n",))
def fluid_splat_spheres(
    pos: gs.f32["n", "c4"],
    zbuf: gs.u32["h", "w"],
    vp: gs.f32["c16"],
    tan_hfov: gs.f32,
    near: gs.f32,
    far: gs.f32,
    surf_scale: gs.f32,
    depth_out: gs.Out[gs.u32["h", "w"], EMPTY_FLUID_DEPTH],
    thick_out: gs.Out[gs.u32["h", "w"], 0],
):
    """Projette chaque particule SPH en sphère analytique (profondeur minimale + épaisseur cumulée)."""
    i = gs.index()
    wx = pos[i, 0]
    wy = pos[i, 1]
    wz = pos[i, 2]
    rad = pos[i, 3] * surf_scale

    cx = vp[0] * wx + vp[1] * wy + vp[2] * wz + vp[3]
    cy = vp[4] * wx + vp[5] * wy + vp[6] * wz + vp[7]
    cw = vp[12] * wx + vp[13] * wy + vp[14] * wz + vp[15]
    if cw <= near + rad * 0.5:
        return

    h = zbuf.shape[0]
    w = zbuf.shape[1]
    inv_w = 1.0 / cw
    sx = (cx * inv_w * 0.5 + 0.5) * float(w)
    sy = (0.5 - cy * inv_w * 0.5) * float(h)

    px_rad = gs.min(rad * (float(h) * 0.5) / gs.max(cw * tan_hfov, 1e-4), 48.0)
    if px_rad < 0.75:
        return

    min_x = gs.max(int(gs.floor(sx - px_rad)), 0)
    max_x = gs.min(int(gs.ceil(sx + px_rad)), w - 1)
    min_y = gs.max(int(gs.floor(sy - px_rad)), 0)
    max_y = gs.min(int(gs.ceil(sy + px_rad)), h - 1)
    if min_x > max_x or min_y > max_y:
        return

    inv_pr = 1.0 / px_rad
    depth_range = gs.max(far - near, 1e-4)
    inv_dr = 1.0 / depth_range

    for py in range(min_y, max_y + 1):
        dy = (float(py) + 0.5 - sy) * inv_pr
        dy2 = dy * dy
        if dy2 < 1.0:
            for px in range(min_x, max_x + 1):
                dx = (float(px) + 0.5 - sx) * inv_pr
                r2 = dx * dx + dy2
                if r2 < 1.0:
                    dz_s = gs.sqrt(1.0 - r2)
                    z_front = cw - rad * dz_s
                    if z_front > near and z_front < far:
                        sz = zbuf[py, px]
                        scene_z = far + 10.0
                        if sz != gs.u32(4294967295):
                            scene_z = near + (float(sz) / 16777215.0) * depth_range
                        if z_front < scene_z + 0.04:
                            dn = gs.min(gs.max((z_front - near) * inv_dr, 0.0), 1.0)
                            d_u32 = gs.u32(dn * 4000000000.0)
                            gs.atomic_min(depth_out, (py, px), d_u32)
                            # Épaisseur optique cumulée (corde traversée par le rayon)
                            t_u32 = gs.u32(rad * 2.0 * dz_s * 2048.0)
                            gs.atomic_add(thick_out, (py, px), t_u32)


@gs.kernel
def fluid_unpack_dt(
    depth_u32: gs.u32["h", "w"],
    thick_u32: gs.u32["h", "w"],
    near: gs.f32,
    far: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Convertit les buffers atomiques `u32` en profondeur linéaire (`0.0` = vide) et épaisseur `f32`."""
    py, px = gs.index()
    du = depth_u32[py, px]
    d_out = 0.0
    t_out = 0.0
    if du != gs.u32(4294967295):
        d_out = near + (float(du) / 4000000000.0) * (far - near)
        t_out = float(thick_u32[py, px]) / 2048.0
    return d_out, t_out


@gs.kernel
def fluid_curvature_flow(
    depth: gs.f32["h", "w"],
    thick: gs.f32["h", "w"],
    max_dz: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Lissage par flux de courbure (van der Laan / NVIDIA FleX) : élimine 100 % des marches et bosses sphériques."""
    py, px = gs.index()
    h = depth.shape[0]
    w = depth.shape[1]
    zc = depth[py, px]
    tc = thick[py, px]
    if zc <= 0.0:
        return 0.0, 0.0

    xm = gs.max(px - 1, 0)
    xp = gs.min(px + 1, w - 1)
    ym = gs.max(py - 1, 0)
    yp = gs.min(py + 1, h - 1)

    sum_d = zc * 2.0
    sum_t = tc * 2.0
    sum_w = 2.0

    # 4 voisins cardinaux (poids 1.0) + 4 diagonaux (poids 0.5) si dans la même nappe de profondeur
    d_l = depth[py, xm]
    if d_l > 0.0 and gs.abs(d_l - zc) < max_dz:
        sum_d += d_l
        sum_t += thick[py, xm]
        sum_w += 1.0
    d_r = depth[py, xp]
    if d_r > 0.0 and gs.abs(d_r - zc) < max_dz:
        sum_d += d_r
        sum_t += thick[py, xp]
        sum_w += 1.0
    d_u = depth[ym, px]
    if d_u > 0.0 and gs.abs(d_u - zc) < max_dz:
        sum_d += d_u
        sum_t += thick[ym, px]
        sum_w += 1.0
    d_d = depth[yp, px]
    if d_d > 0.0 and gs.abs(d_d - zc) < max_dz:
        sum_d += d_d
        sum_t += thick[yp, px]
        sum_w += 1.0

    d_ul = depth[ym, xm]
    if d_ul > 0.0 and gs.abs(d_ul - zc) < max_dz:
        sum_d += d_ul * 0.5
        sum_t += thick[ym, xm] * 0.5
        sum_w += 0.5
    d_ur = depth[ym, xp]
    if d_ur > 0.0 and gs.abs(d_ur - zc) < max_dz:
        sum_d += d_ur * 0.5
        sum_t += thick[ym, xp] * 0.5
        sum_w += 0.5
    d_dl = depth[yp, xm]
    if d_dl > 0.0 and gs.abs(d_dl - zc) < max_dz:
        sum_d += d_dl * 0.5
        sum_t += thick[yp, xm] * 0.5
        sum_w += 0.5
    d_dr = depth[yp, xp]
    if d_dr > 0.0 and gs.abs(d_dr - zc) < max_dz:
        sum_d += d_dr * 0.5
        sum_t += thick[yp, xp] * 0.5
        sum_w += 0.5

    inv_w = 1.0 / sum_w
    return sum_d * inv_w, sum_t * inv_w


@gs.kernel
def fluid_bilateral_blur_h(
    depth: gs.f32["h", "w"],
    thick: gs.f32["h", "w"],
    sigma_s: gs.f32,
    sigma_d: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Lissage bilatéral horizontal : fusionne les particules en surface liquide continue sans flouter les bords."""
    py, px = gs.index()
    w = depth.shape[1]
    dc = depth[py, px]
    tc = thick[py, px]
    if dc <= 0.0:
        return 0.0, 0.0

    inv_2ss = 0.5 / gs.max(sigma_s * sigma_s, 1e-4)
    inv_2sd = 0.5 / gs.max(sigma_d * sigma_d, 1e-4)
    sum_d = 0.0
    sum_t = 0.0
    sum_w = 0.0

    for k in range(-7, 8):
        nx = gs.min(gs.max(px + k, 0), w - 1)
        dn = depth[py, nx]
        if dn > 0.0:
            dd = dn - dc
            fk = float(k)
            wt = gs.exp(-(fk * fk) * inv_2ss - (dd * dd) * inv_2sd)
            sum_d += dn * wt
            sum_t += thick[py, nx] * wt
            sum_w += wt

    inv_w = 1.0 / gs.max(sum_w, 1e-6)
    return sum_d * inv_w, sum_t * inv_w


@gs.kernel
def fluid_bilateral_blur_v(
    depth: gs.f32["h", "w"],
    thick: gs.f32["h", "w"],
    sigma_s: gs.f32,
    sigma_d: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"]):
    """Lissage bilatéral vertical : complète la fusion en surface liquide lisse et continue."""
    py, px = gs.index()
    h = depth.shape[0]
    dc = depth[py, px]
    tc = thick[py, px]
    if dc <= 0.0:
        return 0.0, 0.0

    inv_2ss = 0.5 / gs.max(sigma_s * sigma_s, 1e-4)
    inv_2sd = 0.5 / gs.max(sigma_d * sigma_d, 1e-4)
    sum_d = 0.0
    sum_t = 0.0
    sum_w = 0.0

    for k in range(-7, 8):
        ny = gs.min(gs.max(py + k, 0), h - 1)
        dn = depth[ny, px]
        if dn > 0.0:
            dd = dn - dc
            fk = float(k)
            wt = gs.exp(-(fk * fk) * inv_2ss - (dd * dd) * inv_2sd)
            sum_d += dn * wt
            sum_t += thick[ny, px] * wt
            sum_w += wt

    inv_w = 1.0 / gs.max(sum_w, 1e-6)
    return sum_d * inv_w, sum_t * inv_w


@gs.kernel
def fluid_shade_volumetric(
    hdr_r: gs.f32["h", "w"],
    hdr_g: gs.f32["h", "w"],
    hdr_b: gs.f32["h", "w"],
    depth: gs.f32["h", "w"],
    thick: gs.f32["h", "w"],
    tex_pool: gs.u32["tp"],
    tex_meta: gs.i32["ntex", "c4"],
    dyn_atlas: gs.u32["dh", "dw"],
    sky_tid: gs.i32,
    sky_int: gs.f32,
    fwd_x: gs.f32,
    fwd_y: gs.f32,
    fwd_z: gs.f32,
    right_x: gs.f32,
    right_y: gs.f32,
    right_z: gs.f32,
    up_x: gs.f32,
    up_y: gs.f32,
    up_z: gs.f32,
    tan_hfov: gs.f32,
    sun_x: gs.f32,
    sun_y: gs.f32,
    sun_z: gs.f32,
    sun_r: gs.f32,
    sun_g: gs.f32,
    sun_b: gs.f32,
    sun_int: gs.f32,
    abs_r: gs.f32,
    abs_g: gs.f32,
    abs_b: gs.f32,
    tint_r: gs.f32,
    tint_g: gs.f32,
    tint_b: gs.f32,
    em_r: gs.f32,
    em_g: gs.f32,
    em_b: gs.f32,
    refract_px: gs.f32,
    metallic: gs.f32,
    roughness: gs.f32,
    foam_strength: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Shading optique volumétrique : normales lissées, réfraction du décor, Beer-Lambert, Fresnel + HDRI."""
    py, px = gs.index()
    h = depth.shape[0]
    w = depth.shape[1]
    zc = depth[py, px]
    if zc <= 0.0:
        return hdr_r[py, px], hdr_g[py, px], hdr_b[py, px]

    aspect = float(w) / float(h)
    inv_w = 1.0 / float(w)
    inv_h = 1.0 / float(h)

    # Coordonnées vue du pixel courant
    ux_c = ((float(px) + 0.5) * inv_w * 2.0 - 1.0) * aspect * tan_hfov
    uy_c = (1.0 - (float(py) + 0.5) * inv_h * 2.0) * tan_hfov
    dux = 2.0 * inv_w * aspect * tan_hfov
    duy = -2.0 * inv_h * tan_hfov

    # Différences finies lisses (±2 pixels au cœur du liquide, ±1 aux bords) pour normales miroir sans bruit
    xl1 = gs.max(px - 1, 0)
    xr1 = gs.min(px + 1, w - 1)
    yu1 = gs.max(py - 1, 0)
    yd1 = gs.min(py + 1, h - 1)
    xl2 = gs.max(px - 2, 0)
    xr2 = gs.min(px + 2, w - 1)
    yu2 = gs.max(py - 2, 0)
    yd2 = gs.min(py + 2, h - 1)

    zl = depth[py, xl1]
    zr = depth[py, xr1]
    zu = depth[yu1, px]
    zd = depth[yd1, px]
    zl2 = depth[py, xl2]
    zr2 = depth[py, xr2]
    zu2 = depth[yu2, px]
    zd2 = depth[yd2, px]

    # Vecteur tangent dP/dx (centré large si dans la même nappe, sinon unilatéral net)
    dx_x = dux * zc
    dx_y = 0.0
    dx_z = 0.0
    if zl2 > 0.0 and zr2 > 0.0 and gs.abs(zr2 - zl2) < 0.45:
        dz_c = 0.25 * (zr2 - zl2)
        dx_x = dux * zc + ux_c * dz_c
        dx_y = uy_c * dz_c
        dx_z = dz_c
    elif zr > 0.0 and (zl <= 0.0 or gs.abs(zr - zc) <= gs.abs(zc - zl)):
        dz_r = zr - zc
        dx_x = (ux_c + dux) * zr - ux_c * zc
        dx_y = uy_c * dz_r
        dx_z = dz_r
    elif zl > 0.0:
        dz_l = zc - zl
        dx_x = ux_c * zc - (ux_c - dux) * zl
        dx_y = uy_c * dz_l
        dx_z = dz_l

    # Vecteur tangent dP/dy
    dy_x = 0.0
    dy_y = duy * zc
    dy_z = 0.0
    if zu2 > 0.0 and zd2 > 0.0 and gs.abs(zd2 - zu2) < 0.45:
        dz_cy = 0.25 * (zd2 - zu2)
        dy_x = ux_c * dz_cy
        dy_y = duy * zc + uy_c * dz_cy
        dy_z = dz_cy
    elif zd > 0.0 and (zu <= 0.0 or gs.abs(zd - zc) <= gs.abs(zc - zu)):
        dz_d = zd - zc
        dy_x = ux_c * dz_d
        dy_y = (uy_c + duy) * zd - uy_c * zc
        dy_z = dz_d
    elif zu > 0.0:
        dz_u = zc - zu
        dy_x = ux_c * dz_u
        dy_y = uy_c * zc - (uy_c - duy) * zu
        dy_z = dz_u

    # Produit vectoriel dP/dx × dP/dy (orienté vers la caméra : nv_z < 0)
    nv_x = dx_y * dy_z - dx_z * dy_y
    nv_y = dx_z * dy_x - dx_x * dy_z
    nv_z = dx_x * dy_y - dx_y * dy_x
    if nv_z > 0.0:
        nv_x = -nv_x
        nv_y = -nv_y
        nv_z = -nv_z
    inv_nv = 1.0 / gs.max(gs.sqrt(nv_x * nv_x + nv_y * nv_y + nv_z * nv_z), 1e-6)
    nv_x = nv_x * inv_nv
    nv_y = nv_y * inv_nv
    nv_z = nv_z * inv_nv

    # Normale et vecteur vue en espace monde
    nx = right_x * nv_x + up_x * nv_y + fwd_x * nv_z
    ny = right_y * nv_x + up_y * nv_y + fwd_y * nv_z
    nz = right_z * nv_x + up_z * nv_y + fwd_z * nv_z
    inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx = nx * inv_n
    ny = ny * inv_n
    nz = nz * inv_n

    vx = -(fwd_x + right_x * ux_c + up_x * uy_c)
    vy = -(fwd_y + right_y * ux_c + up_y * uy_c)
    vz = -(fwd_z + right_z * ux_c + up_z * uy_c)
    inv_v = 1.0 / gs.max(gs.sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    vx = vx * inv_v
    vy = vy * inv_v
    vz = vz * inv_v

    ndotv = gs.max(nx * vx + ny * vy + nz * vz, 0.02)
    th = gs.max(thick[py, px], 0.0)
    # Découpe douce sur l'épaisseur optique : supprime les bords bosselés des sphères périphériques
    edge_alpha = gs.min(gs.max((th - 0.14) / 0.22, 0.0), 1.0)
    if edge_alpha <= 0.001:
        return hdr_r[py, px], hdr_g[py, px], hdr_b[py, px]
    th = gs.max(th, 0.05)

    # 1) Réfraction optique réaliste du décor derrière le liquide + absorption Beer-Lambert
    refr_scale = refract_px * gs.min(th * 0.55, 1.0)
    rpx = gs.min(gs.max(int(float(px) + nv_x * refr_scale), 0), w - 1)
    rpy = gs.min(gs.max(int(float(py) - nv_y * refr_scale), 0), h - 1)
    bg_r = hdr_r[rpy, rpx]
    bg_g = hdr_g[rpy, rpx]
    bg_b = hdr_b[rpy, rpx]

    tr = gs.exp(-th * abs_r)
    tg = gs.exp(-th * abs_g)
    tb = gs.exp(-th * abs_b)

    # Caustique douce sur le fond vu à travers l'eau peu profonde + diffusion sous-surface
    ndotl_wrap = gs.max((nx * sun_x + ny * sun_y + nz * sun_z) * 0.5 + 0.5, 0.0)
    caustic = 1.0 + 0.22 * gs.max(ny, 0.0) * gs.exp(-th * 0.35)
    scat = 0.18 + 0.55 * ndotl_wrap
    body_r = bg_r * tr * caustic + tint_r * (1.0 - tr) * scat
    body_g = bg_g * tg * caustic + tint_g * (1.0 - tg) * scat
    body_b = bg_b * tb * caustic + tint_b * (1.0 - tb) * scat

    # 2) Réflexion Fresnel + skybox HDRI (ou ciel procédural)
    rx = 2.0 * ndotv * nx - vx
    ry = 2.0 * ndotv * ny - vy
    rz = 2.0 * ndotv * nz - vz
    env_r = 0.0
    env_g = 0.0
    env_b = 0.0
    mask_u = gs.u32(255)
    inv255 = 0.0039215686

    if sky_tid >= 0:
        ru = gs.atan2(rz, rx) * 0.15915494 + 0.5
        rv = gs.atan2(gs.sqrt(rx * rx + rz * rz), ry) * 0.31830989
        tw = tex_meta[sky_tid, 2]
        m3 = tex_meta[sky_tid, 3]
        th_t = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        rtx = gs.min(gs.max(int(ru * float(tw)), 0), tw - 1)
        rty = gs.min(gs.max(int(rv * float(th_t)), 0), th_t - 1)
        rcs = gs.u32(0)
        if is_dyn != 0:
            rcs = dyn_atlas[tex_meta[sky_tid, 1] + rty, tex_meta[sky_tid, 0] + rtx]
        else:
            rcs = tex_pool[tex_meta[sky_tid, 0] + rty * tw + rtx]
        er_s = float(rcs & mask_u) * inv255
        eg_s = float((rcs >> gs.u32(8)) & mask_u) * inv255
        eb_s = float((rcs >> gs.u32(16)) & mask_u) * inv255
        env_r = er_s * er_s * sky_int * 1.35
        env_g = eg_s * eg_s * sky_int * 1.35
        env_b = eb_s * eb_s * sky_int * 1.35
    else:
        r_sky = gs.min(gs.max(ry * 0.5 + 0.5, 0.0), 1.0)
        env_r = 0.08 + 0.18 * (1.0 - r_sky)
        env_g = 0.12 + 0.24 * (1.0 - r_sky)
        env_b = 0.22 + 0.36 * (1.0 - r_sky)

    # Reflet spéculaire net du soleil (aspect mouillé / brillant)
    hx = vx + sun_x
    hy = vy + sun_y
    hz = vz + sun_z
    inv_h2 = 1.0 / gs.max(gs.sqrt(hx * hx + hy * hy + hz * hz), 1e-6)
    ndoth = gs.max(nx * hx * inv_h2 + ny * hy * inv_h2 + nz * hz * inv_h2, 0.0)
    gloss = gs.max(1.0 - roughness, 0.05)
    spec_pow = 18.0 + gloss * 140.0
    spec = gs.exp(gs.log(gs.max(ndoth, 1e-4)) * spec_pow) * (0.4 + gloss * 1.4) * sun_int * 0.45

    omv = 1.0 - ndotv
    f_pow = omv * omv * omv * omv * omv
    f0r = 0.04 * (1.0 - metallic) + tint_r * metallic
    f0g = 0.04 * (1.0 - metallic) + tint_g * metallic
    f0b = 0.04 * (1.0 - metallic) + tint_b * metallic
    fr = f0r + (1.0 - f0r) * f_pow
    fg = f0g + (1.0 - f0g) * f_pow
    fb = f0b + (1.0 - f0b) * f_pow

    # Écume subtile uniquement sur les crêtes supérieures très incurvées (jamais sur la vitre/sol !)
    foam = 0.0
    if foam_strength > 0.0 and ny > 0.65 and th > 0.35 and th < 1.15:
        crest = (1.15 - th) * (ny - 0.65)
        foam = gs.min(crest * crest * foam_strength * 1.4, 0.45)

    # Émissif (lave / magma / plasma) modulant la croûte refroidie sur les zones plates
    em_mod = 0.45 + 0.55 * gs.max(1.0 - ny, 0.0) + 0.35 * gs.min(th * 0.5, 1.0)

    out_r = body_r * (1.0 - fr) + env_r * fr + sun_r * spec + foam + em_r * em_mod
    out_g = body_g * (1.0 - fg) + env_g * fg + sun_g * spec + foam + em_g * em_mod
    out_b = body_b * (1.0 - fb) + env_b * fb + sun_b * spec + foam + em_b * em_mod
    # Fondu anti-aliasé sur le contour lissé par l'épaisseur
    base_r = hdr_r[py, px]
    base_g = hdr_g[py, px]
    base_b = hdr_b[py, px]
    return (
        base_r * (1.0 - edge_alpha) + out_r * edge_alpha,
        base_g * (1.0 - edge_alpha) + out_g * edge_alpha,
        base_b * (1.0 - edge_alpha) + out_b * edge_alpha,
    )


# ───────────────────────── 5. Solveur Hybride 3D FLIP / PIC O(N) (4 000–10 000+ micro-particules) ─────────────────────────


@gs.kernel(grid=("n",))
def flip_p2g(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    grid_tpl: gs.f32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    g_mass: gs.Out[gs.u32["nc"], 0],
    g_vx: gs.Out[gs.u32["nc"], 0],
    g_vy: gs.Out[gs.u32["nc"], 0],
    g_vz: gs.Out[gs.u32["nc"], 0],
):
    """P2G (Particle-to-Grid) trilinéaire O(N) : accumule masse et vitesse de milliers de gouttes sur la grille 3D."""
    i = gs.index()
    # Utilise grid_tpl en lecture nulle uniquement pour fixer la dimension `nc`
    _unused = grid_tpl[0] * 0.0
    fx = (pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(gx_n - 1)
    fy = (pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(gy_n - 1)
    fz = (pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(gz_n - 1)

    fx = gs.min(gs.max(fx, 0.001), float(gx_n - 1) - 0.001)
    fy = gs.min(gs.max(fy, 0.001), float(gy_n - 1) - 0.001)
    fz = gs.min(gs.max(fz, 0.001), float(gz_n - 1) - 0.001)

    ix0 = int(gs.floor(fx))
    iy0 = int(gs.floor(fy))
    iz0 = int(gs.floor(fz))
    tx = fx - float(ix0)
    ty = fy - float(iy0)
    tz = fz - float(iz0)

    # Biais +20.0 pour stocker les vitesses signées sans débordement en u32 atomique
    vx_b = gs.min(gs.max(vel[i, 0], -18.0), 18.0) + 20.0 + _unused
    vy_b = gs.min(gs.max(vel[i, 1], -18.0), 18.0) + 20.0
    vz_b = gs.min(gs.max(vel[i, 2], -18.0), 18.0) + 20.0
    stride_z = gx_n * gy_n

    for dz in range(2):
        wz = tz if dz == 1 else (1.0 - tz)
        iz = iz0 + dz
        for dy in range(2):
            wy = ty if dy == 1 else (1.0 - ty)
            iy = iy0 + dy
            for dx in range(2):
                wx = tx if dx == 1 else (1.0 - tx)
                w = wx * wy * wz * 1024.0
                if w > 0.5:
                    cid = iz * stride_z + iy * gx_n + (ix0 + dx)
                    # Poids entier commun aux sommes : sinon la moyenne est biaisée
                    q = gs.u32(w + 0.5)
                    qf = float(q)
                    gs.atomic_add(g_mass, cid, q)
                    gs.atomic_add(g_vx, cid, gs.u32(qf * vx_b + 0.5))
                    gs.atomic_add(g_vy, cid, gs.u32(qf * vy_b + 0.5))
                    gs.atomic_add(g_vz, cid, gs.u32(qf * vz_b + 0.5))


@gs.kernel
def flip_grid_forces(
    g_mass_u: gs.u32["nc"],
    g_vx_u: gs.u32["nc"],
    g_vy_u: gs.u32["nc"],
    g_vz_u: gs.u32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
    dt: gs.f32,
    grav_x: gs.f32,
    grav_y: gs.f32,
    grav_z: gs.f32,
    pulse_x: gs.f32,
    pulse_z: gs.f32,
) -> (gs.f32["nc"], gs.f32["nc"], gs.f32["nc"], gs.f32["nc"]):
    """Normalise la grille 3D, applique gravité + impulsion de vague + conditions aux parois."""
    cid = gs.index()
    stride_z = gx_n * gy_n
    iz = cid // stride_z
    rem = cid - iz * stride_z
    iy = rem // gx_n
    ix = rem - iy * gx_n

    mu = g_mass_u[cid]
    mass_f = float(mu) / 1024.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    if mu > gs.u32(16):
        inv_m = 1.0 / float(mu)
        vx = float(g_vx_u[cid]) * inv_m - 20.0 + (grav_x + pulse_x) * dt
        vy = float(g_vy_u[cid]) * inv_m - 20.0 + grav_y * dt
        vz = float(g_vz_u[cid]) * inv_m - 20.0 + (grav_z + pulse_z) * dt
        if (ix == 0 and vx < 0.0) or (ix == gx_n - 1 and vx > 0.0):
            vx = 0.0
        if (iy == 0 and vy < 0.0) or (iy == gy_n - 1 and vy > 0.0):
            vy = 0.0
        if (iz == 0 and vz < 0.0) or (iz == gz_n - 1 and vz > 0.0):
            vz = 0.0
    return vx, vy, vz, mass_f


@gs.kernel
def flip_grid_divergence(
    vx: gs.f32["nc"],
    vy: gs.f32["nc"],
    vz: gs.f32["nc"],
    mass_f: gs.f32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
    rest_mass: gs.f32,
    density_push: gs.f32,
) -> gs.f32["nc"]:
    """Calcule la divergence 3D + correction de volume (empêche le tassement du liquide)."""
    cid = gs.index()
    m = mass_f[cid]
    if m <= 0.05:
        return 0.0
    stride_z = gx_n * gy_n
    iz = cid // stride_z
    rem = cid - iz * stride_z
    iy = rem // gx_n
    ix = rem - iy * gx_n

    xm = gs.max(ix - 1, 0)
    xp = gs.min(ix + 1, gx_n - 1)
    ym = gs.max(iy - 1, 0)
    yp = gs.min(iy + 1, gy_n - 1)
    zm = gs.max(iz - 1, 0)
    zp = gs.min(iz + 1, gz_n - 1)

    v_l = vx[iz * stride_z + iy * gx_n + xm] if ix > 0 else 0.0
    v_r = vx[iz * stride_z + iy * gx_n + xp] if ix < gx_n - 1 else 0.0
    v_d = vy[iz * stride_z + ym * gx_n + ix] if iy > 0 else 0.0
    v_u = vy[iz * stride_z + yp * gx_n + ix] if iy < gy_n - 1 else 0.0
    v_b = vz[zm * stride_z + iy * gx_n + ix] if iz > 0 else 0.0
    v_f = vz[zp * stride_z + iy * gx_n + ix] if iz < gz_n - 1 else 0.0

    div = 0.5 * ((v_r - v_l) + (v_u - v_d) + (v_f - v_b))
    # Correction de dérive de densité (maintient le volume exact du liquide)
    excess = gs.max(m - rest_mass, 0.0)
    return div - density_push * excess


@gs.kernel
def flip_grid_pressure_jacobi(
    p: gs.f32["nc"],
    div: gs.f32["nc"],
    mass_f: gs.f32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
) -> gs.f32["nc"]:
    """Itération de Jacobi 3D pour l'équation de Poisson de pression (air = 0, parois = Neumann)."""
    cid = gs.index()
    if mass_f[cid] <= 0.05:
        return 0.0
    stride_z = gx_n * gy_n
    iz = cid // stride_z
    rem = cid - iz * stride_z
    iy = rem // gx_n
    ix = rem - iy * gx_n

    pc = p[cid]
    p_l = p[iz * stride_z + iy * gx_n + (ix - 1)] if ix > 0 else pc
    p_r = p[iz * stride_z + iy * gx_n + (ix + 1)] if ix < gx_n - 1 else pc
    p_d = p[iz * stride_z + (iy - 1) * gx_n + ix] if iy > 0 else pc
    p_u = p[iz * stride_z + (iy + 1) * gx_n + ix] if iy < gy_n - 1 else pc
    p_b = p[(iz - 1) * stride_z + iy * gx_n + ix] if iz > 0 else pc
    p_f = p[(iz + 1) * stride_z + iy * gx_n + ix] if iz < gz_n - 1 else pc

    return (p_l + p_r + p_d + p_u + p_b + p_f - div[cid]) * 0.16666667


@gs.kernel
def flip_grid_project(
    vx: gs.f32["nc"],
    vy: gs.f32["nc"],
    vz: gs.f32["nc"],
    p: gs.f32["nc"],
    mass_f: gs.f32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
) -> (gs.f32["nc"], gs.f32["nc"], gs.f32["nc"]):
    """Projette le champ de vitesse 3D sur un écoulement incompressible (∇·v = 0)."""
    cid = gs.index()
    if mass_f[cid] <= 0.05:
        return 0.0, 0.0, 0.0
    stride_z = gx_n * gy_n
    iz = cid // stride_z
    rem = cid - iz * stride_z
    iy = rem // gx_n
    ix = rem - iy * gx_n

    pc = p[cid]
    p_l = p[iz * stride_z + iy * gx_n + (ix - 1)] if ix > 0 else pc
    p_r = p[iz * stride_z + iy * gx_n + (ix + 1)] if ix < gx_n - 1 else pc
    p_d = p[iz * stride_z + (iy - 1) * gx_n + ix] if iy > 0 else pc
    p_u = p[iz * stride_z + (iy + 1) * gx_n + ix] if iy < gy_n - 1 else pc
    p_b = p[(iz - 1) * stride_z + iy * gx_n + ix] if iz > 0 else pc
    p_f = p[(iz + 1) * stride_z + iy * gx_n + ix] if iz < gz_n - 1 else pc

    nvx = vx[cid] - 0.5 * (p_r - p_l)
    nvy = vy[cid] - 0.5 * (p_u - p_d)
    nvz = vz[cid] - 0.5 * (p_f - p_b)

    if (ix == 0 and nvx < 0.0) or (ix == gx_n - 1 and nvx > 0.0):
        nvx = 0.0
    if (iy == 0 and nvy < 0.0) or (iy == gy_n - 1 and nvy > 0.0):
        nvy = 0.0
    if (iz == 0 and nvz < 0.0) or (iz == gz_n - 1 and nvz > 0.0):
        nvz = 0.0
    return nvx, nvy, nvz


@gs.kernel
def flip_g2p(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    vx_new: gs.f32["nc"],
    vy_new: gs.f32["nc"],
    vz_new: gs.f32["nc"],
    vx_old: gs.f32["nc"],
    vy_old: gs.f32["nc"],
    vz_old: gs.f32["nc"],
    gx_n: gs.i32,
    gy_n: gs.i32,
    gz_n: gs.i32,
    dt: gs.f32,
    grav_x: gs.f32,
    grav_y: gs.f32,
    grav_z: gs.f32,
    pulse_x: gs.f32,
    pulse_z: gs.f32,
    flip_ratio: gs.f32,
    damping: gs.f32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    restitution: gs.f32,
    max_speed: gs.f32,
) -> (gs.f32["n", "c4"], gs.f32["n", "c4"]):
    """G2P (Grid-to-Particle) : mélange PIC/FLIP trilinéaire + advection et rebonds parois."""
    i, c = gs.index()
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    prad = pos[i, 3]

    fx = gs.min(gs.max((px + bx) / gs.max(2.0 * bx, 1e-4) * float(gx_n - 1), 0.001), float(gx_n - 1) - 0.001)
    fy = gs.min(gs.max((py - floor_y) / gs.max(by, 1e-4) * float(gy_n - 1), 0.001), float(gy_n - 1) - 0.001)
    fz = gs.min(gs.max((pz + bz) / gs.max(2.0 * bz, 1e-4) * float(gz_n - 1), 0.001), float(gz_n - 1) - 0.001)

    ix0 = int(gs.floor(fx))
    iy0 = int(gs.floor(fy))
    iz0 = int(gs.floor(fz))
    tx = fx - float(ix0)
    ty = fy - float(iy0)
    tz = fz - float(iz0)
    stride_z = gx_n * gy_n

    pic_x = 0.0
    pic_y = 0.0
    pic_z = 0.0
    dvx = 0.0
    dvy = 0.0
    dvz = 0.0

    for dz in range(2):
        wz = tz if dz == 1 else (1.0 - tz)
        iz = iz0 + dz
        for dy in range(2):
            wy = ty if dy == 1 else (1.0 - ty)
            iy = iy0 + dy
            for dx in range(2):
                wx = tx if dx == 1 else (1.0 - tx)
                w = wx * wy * wz
                cid = iz * stride_z + iy * gx_n + (ix0 + dx)
                nx_v = vx_new[cid]
                ny_v = vy_new[cid]
                nz_v = vz_new[cid]
                pic_x += nx_v * w
                pic_y += ny_v * w
                pic_z += nz_v * w
                dvx += (nx_v - vx_old[cid]) * w
                dvy += (ny_v - vy_old[cid]) * w
                dvz += (nz_v - vz_old[cid]) * w

    # Δv FLIP complet = gravité/impulsion + correction de pression (v_new - v_star)
    flip_vx = vel[i, 0] + (grav_x + pulse_x) * dt + dvx
    flip_vy = vel[i, 1] + grav_y * dt + dvy
    flip_vz = vel[i, 2] + (grav_z + pulse_z) * dt + dvz
    vx = ((1.0 - flip_ratio) * pic_x + flip_ratio * flip_vx) * damping
    vy = ((1.0 - flip_ratio) * pic_y + flip_ratio * flip_vy) * damping
    vz = ((1.0 - flip_ratio) * pic_z + flip_ratio * flip_vz) * damping

    sp2 = vx * vx + vy * vy + vz * vz
    if sp2 > max_speed * max_speed:
        sc = max_speed / gs.sqrt(sp2)
        vx = vx * sc
        vy = vy * sc
        vz = vz * sc

    px += vx * dt
    py += vy * dt
    pz += vz * dt

    # Parois de l'aquarium
    min_y = floor_y + prad
    max_y = floor_y + by - prad
    if py < min_y:
        py = min_y
        if vy < 0.0:
            vy = -vy * restitution
        vx = vx * 0.985
        vz = vz * 0.985
    elif py > max_y:
        py = max_y
        if vy > 0.0:
            vy = -vy * restitution

    min_x = -bx + prad
    max_x = bx - prad
    if px < min_x:
        px = min_x
        if vx < 0.0:
            vx = -vx * restitution
    elif px > max_x:
        px = max_x
        if vx > 0.0:
            vx = -vx * restitution

    min_z = -bz + prad
    max_z = bz - prad
    if pz < min_z:
        pz = min_z
        if vz < 0.0:
            vz = -vz * restitution
    elif pz > max_z:
        pz = max_z
        if vz > 0.0:
            vz = -vz * restitution

    out_p = px
    out_v = vx
    if c == 1:
        out_p = py
        out_v = vy
    elif c == 2:
        out_p = pz
        out_v = vz
    elif c == 3:
        out_p = prad
        out_v = vel[i, 3]
    return out_p, out_v


