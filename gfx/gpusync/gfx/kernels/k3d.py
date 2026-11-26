"""Kernels GPU 3D de `gpusync.gfx` (Visibility Buffer, PBR Cook-Torrance, Point/SpotLight, Bloom, FXAA, Multi-Viewport)."""

from __future__ import annotations

import gpusync as gs

EMPTY_VIS = 4294967295  # 0xFFFFFFFF : ciel / fond vide
DEPTH_MAX = 16777215  # profondeur linéaire entre near et far, sur 24 bits (exacts en f32)
BLOOM_GROUP = 64
VIS_STRIPES = 32
SHADOW_STRIPES = 16


# ───────────────────────── 0. Upload statique & utilitaires VRAM ─────────────────────────


@gs.kernel
def upload_f32(x: gs.f32[:]) -> gs.f32[:]:
    return x


@gs.kernel
def upload_i32(x: gs.i32[:]) -> gs.i32[:]:
    return x


@gs.kernel
def upload_u32(x: gs.u32[:]) -> gs.u32[:]:
    return x


@gs.kernel
def clear_u32(ref: gs.u32["h", "w"], value: gs.u32) -> gs.u32["h", "w"]:
    return value


@gs.kernel
def patch_vertices(
    dst: gs.f32["nv", "c3"],
    src: gs.f32["snv", "c3"],
    offset: gs.i32,
) -> gs.f32["nv", "c3"]:
    """Met à jour en VRAM les sommets/normales d'un `Object3D` dynamique (0 copie CPU)."""
    v, c = gs.index()
    sv = v - offset
    if sv >= 0 and sv < src.shape[0]:
        return src[sv, c]
    return dst[v, c]


@gs.kernel
def patch_trs_to_matrices(
    models: gs.f32["nm", "c16"],
    norms: gs.f32["nm", "c16"],
    trs: gs.f32["k", "c8"],
    offset: gs.i32,
) -> (gs.f32["nm", "c16"], gs.f32["nm", "c16"]):
    """Convertit des instances GPU compactes `(px, py, pz, rx, ry, rz, scale, visible)` en matrices 4×4 en VRAM."""
    m, elem = gs.index()
    idx = m - offset
    if idx < 0 or idx >= trs.shape[0]:
        return models[m, elem], norms[m, elem]

    px = trs[idx, 0]
    py = trs[idx, 1]
    pz = trs[idx, 2]
    rx = trs[idx, 3]
    ry = trs[idx, 4]
    rz = trs[idx, 5]
    sc = trs[idx, 6]
    vis = trs[idx, 7]
    if vis <= 0.0:
        sc = 0.0

    cx = gs.cos(rx)
    cy = gs.cos(ry)
    cz = gs.cos(rz)
    sx = gs.sin(rx)
    sy = gs.sin(ry)
    sz = gs.sin(rz)

    r00 = cy * cz + sy * sx * sz
    r01 = -cy * sz + sy * sx * cz
    r02 = sy * cx
    r10 = cx * sz
    r11 = cx * cz
    r12 = -sx
    r20 = -sy * cz + cy * sx * sz
    r21 = sy * sz + cy * sx * cz
    r22 = cy * cx

    inv_s = 0.0
    if gs.abs(sc) > 1e-6:
        inv_s = 1.0 / sc

    mv = 0.0
    nv = 0.0
    if elem == 0:
        mv = r00 * sc
        nv = r00 * inv_s
    elif elem == 1:
        mv = r01 * sc
        nv = r01 * inv_s
    elif elem == 2:
        mv = r02 * sc
        nv = r02 * inv_s
    elif elem == 3:
        mv = px
    elif elem == 4:
        mv = r10 * sc
        nv = r10 * inv_s
    elif elem == 5:
        mv = r11 * sc
        nv = r11 * inv_s
    elif elem == 6:
        mv = r12 * sc
        nv = r12 * inv_s
    elif elem == 7:
        mv = py
    elif elem == 8:
        mv = r20 * sc
        nv = r20 * inv_s
    elif elem == 9:
        mv = r21 * sc
        nv = r21 * inv_s
    elif elem == 10:
        mv = r22 * sc
        nv = r22 * inv_s
    elif elem == 11:
        mv = pz
    elif elem == 15:
        mv = 1.0
        nv = 1.0
    return mv, nv


@gs.kernel
def bind_trs_to_matrices(
    models: gs.f32["nm", "c16"],
    norms: gs.f32["nm", "c16"],
    trs: gs.f32["k", "cs"],
    slot_row: gs.i32["nm"],
) -> (gs.f32["nm", "c16"], gs.f32["nm", "c16"]):
    """Objets dont la position vient d'un calcul GPU : la matrice de l'emplacement `m` est faite de la ligne
    `slot_row[m]` de `trs` (`px, py, pz, rx, ry, rz, échelle, visible`, colonnes suivantes ignorées ; -1 : inchangé).

    L'échelle et la visibilité fixées côté CPU (longueur des colonnes de la matrice) sont conservées."""
    m, elem = gs.index()
    row = slot_row[m]
    if row < 0:
        return models[m, elem], norms[m, elem]
    r = elem // 4
    col = elem - r * 4
    if r == 3:
        return models[m, elem], norms[m, elem]
    if col == 3:
        return trs[row, r], 0.0
    cx = gs.cos(trs[row, 3])
    sx = gs.sin(trs[row, 3])
    cy = gs.cos(trs[row, 4])
    sy = gs.sin(trs[row, 4])
    cz = gs.cos(trs[row, 5])
    sz = gs.sin(trs[row, 5])
    v = 0.0
    if r == 0:
        if col == 0:
            v = cy * cz + sy * sx * sz
        elif col == 1:
            v = -cy * sz + sy * sx * cz
        else:
            v = sy * cx
    elif r == 1:
        if col == 0:
            v = cx * sz
        elif col == 1:
            v = cx * cz
        else:
            v = -sx
    else:
        if col == 0:
            v = -sy * cz + cy * sx * sz
        elif col == 1:
            v = sy * sz + cy * sx * cz
        else:
            v = cy * cx
    sc = gs.sqrt(models[m, col] * models[m, col] + models[m, 4 + col] * models[m, 4 + col] + models[m, 8 + col] * models[m, 8 + col])
    sc = sc * trs[row, 6]
    if trs[row, 7] <= 0.0:
        sc = 0.0
    inv_s = 0.0
    if gs.abs(sc) > 1e-6:
        inv_s = 1.0 / sc
    return v * sc, v * inv_s


@gs.kernel
def patch_matrices(
    models: gs.f32["nm", "c16"],
    src: gs.f32["k", "c16"],
    offset: gs.i32,
) -> gs.f32["nm", "c16"]:
    """Copie un bloc de matrices 4×4 GPU `(K, 16)` directement dans `models`."""
    m, elem = gs.index()
    idx = m - offset
    if idx >= 0 and idx < src.shape[0]:
        return src[idx, elem]
    return models[m, elem]


@gs.kernel
def mul_vp_matrices(models: gs.f32["nm", "c16"], vp: gs.f32["c16"]) -> gs.f32["nm", "c16"]:
    """Calcule `MVP = VP @ M` pour toutes les instances directement sur le GPU."""
    m, elem = gs.index()
    row = elem // 4
    col = elem - row * 4
    r4 = row * 4
    return (
        vp[r4] * models[m, col]
        + vp[r4 + 1] * models[m, 4 + col]
        + vp[r4 + 2] * models[m, 8 + col]
        + vp[r4 + 3] * models[m, 12 + col]
    )


# ───────────────────────── 1. Vertex Shaders ─────────────────────────


@gs.kernel
def vertex_world(
    v_pos: gs.f32["nv", "c3"],
    v_norm: gs.f32["nv", "c3"],
    v_obj: gs.i32["nv"],
    models: gs.f32["nm", "c16"],
    norms: gs.f32["nm", "c16"],
) -> (gs.f32["nv", "c3"], gs.f32["nv", "c3"]):
    """Transforme positions et normales de tous les objets en espace monde (partagé entre caméras)."""
    v, c = gs.index()
    o = v_obj[v]
    px = v_pos[v, 0]
    py = v_pos[v, 1]
    pz = v_pos[v, 2]

    row = c * 4
    wpos = (
        models[o, row] * px
        + models[o, row + 1] * py
        + models[o, row + 2] * pz
        + models[o, row + 3]
    )

    nx0 = v_norm[v, 0]
    ny0 = v_norm[v, 1]
    nz0 = v_norm[v, 2]
    wx = norms[o, 0] * nx0 + norms[o, 1] * ny0 + norms[o, 2] * nz0
    wy = norms[o, 4] * nx0 + norms[o, 5] * ny0 + norms[o, 6] * nz0
    wz = norms[o, 8] * nx0 + norms[o, 9] * ny0 + norms[o, 10] * nz0
    inv_l = 1.0 / gs.max(gs.sqrt(wx * wx + wy * wy + wz * wz), 1e-6)
    wnorm = wx * inv_l
    if c == 1:
        wnorm = wy * inv_l
    elif c == 2:
        wnorm = wz * inv_l
    return wpos, wnorm


@gs.kernel
def vertex_project_all(
    v_pos: gs.f32["nv", "c3"],
    v_obj: gs.i32["nv"],
    mvps: gs.f32["nm", "c16"],
    light_mvps: gs.f32["nm", "c16"],
    width: gs.f32,
    height: gs.f32,
    near: gs.f32,
    far: gs.f32,
    is_ortho: gs.i32,
    shadow_w: gs.f32,
    shadow_h: gs.f32,
) -> (gs.f32["nv"], gs.f32["nv"], gs.f32["nv"], gs.f32["nv"], gs.f32["nv"], gs.f32["nv"]):
    """Fusionne projection caméra (perspective ou ortho) et projection soleil en 1 dispatch."""
    v = gs.index()
    o = v_obj[v]
    x = v_pos[v, 0]
    y = v_pos[v, 1]
    z = v_pos[v, 2]

    cx = mvps[o, 0] * x + mvps[o, 1] * y + mvps[o, 2] * z + mvps[o, 3]
    cy = mvps[o, 4] * x + mvps[o, 5] * y + mvps[o, 6] * z + mvps[o, 7]
    cz = mvps[o, 8] * x + mvps[o, 9] * y + mvps[o, 10] * z + mvps[o, 11]
    cw = mvps[o, 12] * x + mvps[o, 13] * y + mvps[o, 14] * z + mvps[o, 15]

    sx = 0.0
    sy = 0.0
    inv_w = 0.0
    if is_ortho != 0:
        if cz >= 0.0 and cz <= 1.0:
            sx = (cx * 0.5 + 0.5) * width
            sy = (0.5 - cy * 0.5) * height
            inv_w = 1.0 / gs.max(near + cz * (far - near), 1e-4)
    elif cw > near:
        inv_w = 1.0 / cw
        sx = (cx * inv_w * 0.5 + 0.5) * width
        sy = (0.5 - cy * inv_w * 0.5) * height
    else:
        # Derrière le plan proche : coordonnées de découpe (x, y) brutes et w codé en négatif (≤ −1)
        sx = cx
        sy = cy
        inv_w = cw - near - 1.0

    lx = light_mvps[o, 0] * x + light_mvps[o, 1] * y + light_mvps[o, 2] * z + light_mvps[o, 3]
    ly = light_mvps[o, 4] * x + light_mvps[o, 5] * y + light_mvps[o, 6] * z + light_mvps[o, 7]
    lz = light_mvps[o, 8] * x + light_mvps[o, 9] * y + light_mvps[o, 10] * z + light_mvps[o, 11]

    sh_x = (lx * 0.5 + 0.5) * shadow_w
    sh_y = (0.5 - ly * 0.5) * shadow_h
    return sx, sy, inv_w, sh_x, sh_y, lz


@gs.kernel
def vertex_project(
    v_pos: gs.f32["nv", "c3"],
    v_obj: gs.i32["nv"],
    mvps: gs.f32["nm", "c16"],
    width: gs.f32,
    height: gs.f32,
    near: gs.f32,
    far: gs.f32,
    is_ortho: gs.i32,
) -> (gs.f32["nv"], gs.f32["nv"], gs.f32["nv"]):
    """Projette chaque sommet vers l'écran (caméra secondaire ou ombres désactivées)."""
    v = gs.index()
    o = v_obj[v]
    x = v_pos[v, 0]
    y = v_pos[v, 1]
    z = v_pos[v, 2]

    cx = mvps[o, 0] * x + mvps[o, 1] * y + mvps[o, 2] * z + mvps[o, 3]
    cy = mvps[o, 4] * x + mvps[o, 5] * y + mvps[o, 6] * z + mvps[o, 7]
    cz = mvps[o, 8] * x + mvps[o, 9] * y + mvps[o, 10] * z + mvps[o, 11]
    cw = mvps[o, 12] * x + mvps[o, 13] * y + mvps[o, 14] * z + mvps[o, 15]

    sx = 0.0
    sy = 0.0
    inv_w = 0.0
    if is_ortho != 0:
        if cz >= 0.0 and cz <= 1.0:
            sx = (cx * 0.5 + 0.5) * width
            sy = (0.5 - cy * 0.5) * height
            inv_w = 1.0 / gs.max(near + cz * (far - near), 1e-4)
    elif cw > near:
        inv_w = 1.0 / cw
        sx = (cx * inv_w * 0.5 + 0.5) * width
        sy = (0.5 - cy * inv_w * 0.5) * height
    else:
        # Derrière le plan proche : coordonnées de découpe (x, y) brutes et w codé en négatif (≤ −1)
        sx = cx
        sy = cy
        inv_w = cw - near - 1.0
    return sx, sy, inv_w


# ───────────────────────── 2. Rasterizers Parallèles par Bandes ─────────────────────────


@gs.kernel(grid=("nt", SHADOW_STRIPES))
def rasterize_shadow(
    tri_idx: gs.i32["nt", "c3"],
    sx: gs.f32["nv"],
    sy: gs.f32["nv"],
    sz: gs.f32["nv"],
    smap: gs.Out[gs.u32["sh", "sw"], EMPTY_VIS],
):
    """Rasterise la shadow map en parallèle (16 bandes/triangle, pas incrémental)."""
    t, strip = gs.index()
    i0 = tri_idx[t, 0]
    i1 = tri_idx[t, 1]
    i2 = tri_idx[t, 2]

    x0 = sx[i0]
    y0 = sy[i0]
    z0 = sz[i0]
    x1 = sx[i1]
    y1 = sy[i1]
    z1 = sz[i1]
    x2 = sx[i2]
    y2 = sy[i2]
    z2 = sz[i2]

    area = (y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0)
    if gs.abs(area) < 1e-4:
        return

    sh = smap.shape[0]
    sw = smap.shape[1]
    min_y = gs.max(int(gs.floor(gs.min(y0, y1, y2))), 0)
    max_y = gs.min(int(gs.ceil(gs.max(y0, y1, y2))), sh - 1)
    first_y = min_y + strip
    if first_y > max_y:
        return
    min_x = gs.max(int(gs.floor(gs.min(x0, x1, x2))), 0)
    max_x = gs.min(int(gs.ceil(gs.max(x0, x1, x2))), sw - 1)
    if min_x > max_x:
        return

    inv_area = 1.0 / area
    dw0_dx = (y2 - y1) * inv_area
    dw1_dx = (y0 - y2) * inv_area
    dz_dx = dw0_dx * (z0 - z2) + dw1_dx * (z1 - z2)
    fx0 = float(min_x) + 0.5

    for py in range(first_y, max_y + 1, 16):
        fy = float(py) + 0.5
        w0 = ((y2 - y1) * (fx0 - x1) - (x2 - x1) * (fy - y1)) * inv_area
        w1 = ((y0 - y2) * (fx0 - x2) - (x0 - x2) * (fy - y2)) * inv_area
        z = w0 * z0 + w1 * z1 + (1.0 - w0 - w1) * z2
        for px in range(min_x, max_x + 1):
            if w0 >= 0.0 and w1 >= 0.0 and (w0 + w1) <= 1.0:
                if z >= 0.0 and z <= 1.0:
                    d = gs.u32(z * 4294967040.0)
                    gs.atomic_min(smap, (py, px), d)
            w0 += dw0_dx
            w1 += dw1_dx
            z += dz_dx


@gs.kernel(grid=("nt", VIS_STRIPES))
def rasterize_triangles(
    tri_idx: gs.i32["nt", "c3"],
    sx: gs.f32["nv"],
    sy: gs.f32["nv"],
    inv_w: gs.f32["nv"],
    zbuf: gs.u32["zh", "zw"],
    near: gs.f32,
    inv_depth_range: gs.f32,
    select: gs.i32,
    target: gs.Out[gs.u32["h", "w"], EMPTY_VIS],
):
    """Rasterisation parallèle (32 bandes/triangle, bornes X exactes par scanline), en deux passes :
    `select=0` écrit la profondeur la plus proche (24 bits), `select=1` le numéro du triangle qui a
    exactement cette profondeur dans `zbuf` (le plus petit numéro en cas d'égalité). Un seul noyau pour les
    deux passes : les calculs flottants, donc les profondeurs, sont identiques.

    Un triangle qui traverse le plan proche est découpé ici en un ou deux triangles."""
    t, strip = gs.index()
    h = target.shape[0]
    w = target.shape[1]
    k0 = tri_idx[t, 0]
    k1 = tri_idx[t, 1]
    k2 = tri_idx[t, 2]
    if inv_w[k0] == 0.0 or inv_w[k1] == 0.0 or inv_w[k2] == 0.0:
        return  # hors de la profondeur d'une caméra orthographique
    n_in = 0
    if inv_w[k0] > 0.0:
        n_in += 1
    if inv_w[k1] > 0.0:
        n_in += 1
    if inv_w[k2] > 0.0:
        n_in += 1
    if n_in == 0:
        return

    # Rotation (le sens de parcours est conservé) : un seul sommet devant → le premier ; deux → les deux premiers
    rot = 0
    if n_in == 1:
        if inv_w[k1] > 0.0:
            rot = 1
        elif inv_w[k2] > 0.0:
            rot = 2
    elif n_in == 2:
        if inv_w[k0] <= 0.0:
            rot = 1
        elif inv_w[k1] <= 0.0:
            rot = 2
    i0 = k0 if rot == 0 else (k1 if rot == 1 else k2)
    i1 = k1 if rot == 0 else (k2 if rot == 1 else k0)
    i2 = k2 if rot == 0 else (k0 if rot == 1 else k1)

    ax = sx[i0]
    ay = sy[i0]
    aw = inv_w[i0]
    bx = sx[i1]
    by = sy[i1]
    bw = inv_w[i1]
    cx = sx[i2]
    cy = sy[i2]
    cw = inv_w[i2]
    dx = 0.0
    dy = 0.0
    dw = 0.0
    parts = 1
    if n_in < 3:
        fw = float(w)
        fh = float(h)
        # Coordonnées de découpe de A (devant), C (derrière) et B (devant si n_in = 2)
        a_w = 1.0 / aw
        a_x = (ax / fw * 2.0 - 1.0) * a_w
        a_y = (1.0 - ay / fh * 2.0) * a_w
        c_w = cw + near + 1.0
        c_x = cx
        c_y = cy
        b_w = bw + near + 1.0
        b_x = bx
        b_y = by
        if n_in == 2:
            b_w = 1.0 / bw
            b_x = (bx / fw * 2.0 - 1.0) * b_w
            b_y = (1.0 - by / fh * 2.0) * b_w
        # Q sur [C, A] ; P sur [B, A] (un sommet devant) ou sur [C, B] (deux)
        sq = (near - c_w) / (a_w - c_w)
        q_x = c_x + sq * (a_x - c_x)
        q_y = c_y + sq * (a_y - c_y)
        p_x = 0.0
        p_y = 0.0
        if n_in == 1:
            sp = (near - b_w) / (a_w - b_w)
            p_x = b_x + sp * (a_x - b_x)
            p_y = b_y + sp * (a_y - b_y)
        else:
            sp = (near - c_w) / (b_w - c_w)
            p_x = c_x + sp * (b_x - c_x)
            p_y = c_y + sp * (b_y - c_y)
        inv_near = 1.0 / near
        p_sx = (p_x * inv_near * 0.5 + 0.5) * fw
        p_sy = (0.5 - p_y * inv_near * 0.5) * fh
        q_sx = (q_x * inv_near * 0.5 + 0.5) * fw
        q_sy = (0.5 - q_y * inv_near * 0.5) * fh
        if n_in == 1:
            bx = p_sx
            by = p_sy
            bw = inv_near
            cx = q_sx
            cy = q_sy
            cw = inv_near
        else:
            cx = p_sx
            cy = p_sy
            cw = inv_near
            dx = q_sx
            dy = q_sy
            dw = inv_near
            parts = 2

    for part in range(parts):
        x0 = ax
        y0 = ay
        iw0 = aw
        x1 = bx if part == 0 else cx
        y1 = by if part == 0 else cy
        iw1 = bw if part == 0 else cw
        x2 = cx if part == 0 else dx
        y2 = cy if part == 0 else dy
        iw2 = cw if part == 0 else dw

        area = (y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0)
        # Bornes limitées avant conversion : un sommet découpé peut tomber très loin de l'écran
        lo_y = gs.min(gs.max(gs.min(y0, y1, y2), -1.0), float(h))
        hi_y = gs.min(gs.max(gs.max(y0, y1, y2), -1.0), float(h))
        lo_x = gs.min(gs.max(gs.min(x0, x1, x2), -1.0), float(w))
        hi_x = gs.min(gs.max(gs.max(x0, x1, x2), -1.0), float(w))
        min_y = gs.max(int(gs.floor(lo_y)), 0)
        max_y = gs.min(int(gs.ceil(hi_y)), h - 1)
        min_x = gs.max(int(gs.floor(lo_x)), 0)
        max_x = gs.min(int(gs.ceil(hi_x)), w - 1)
        first_y = min_y + strip
        if area > 1e-4 and first_y <= max_y and min_x <= max_x:
            inv_area = 1.0 / area
            dw0_dx = (y2 - y1) * inv_area
            dw1_dx = (y0 - y2) * inv_area
            dw2_dx = (y1 - y0) * inv_area
            diw_dx = dw0_dx * (iw0 - iw2) + dw1_dx * (iw1 - iw2)
            fmin_x = float(min_x)
            fmax_x = float(max_x)

            for py in range(first_y, max_y + 1, 32):
                fy = float(py) + 0.5
                xl = fmin_x
                xr = fmax_x
                c0 = ((y2 - y1) * (0.5 - x1) - (x2 - x1) * (fy - y1)) * inv_area + 1e-5
                if dw0_dx > 1e-8:
                    xl = gs.max(xl, -c0 / dw0_dx)
                elif dw0_dx < -1e-8:
                    xr = gs.min(xr, -c0 / dw0_dx)
                elif c0 < 0.0:
                    xr = -1.0

                c1 = ((y0 - y2) * (0.5 - x2) - (x0 - x2) * (fy - y2)) * inv_area + 1e-5
                if dw1_dx > 1e-8:
                    xl = gs.max(xl, -c1 / dw1_dx)
                elif dw1_dx < -1e-8:
                    xr = gs.min(xr, -c1 / dw1_dx)
                elif c1 < 0.0:
                    xr = -1.0

                c2 = ((y1 - y0) * (0.5 - x0) - (x1 - x0) * (fy - y0)) * inv_area + 1e-5
                if dw2_dx > 1e-8:
                    xl = gs.max(xl, -c2 / dw2_dx)
                elif dw2_dx < -1e-8:
                    xr = gs.min(xr, -c2 / dw2_dx)
                elif c2 < 0.0:
                    xr = -1.0

                xl = gs.min(xl, fmax_x + 1.0)
                xr = gs.max(xr, -1.0)
                ix0 = gs.max(int(gs.ceil(xl)), min_x)
                ix1 = gs.min(int(gs.floor(xr)), max_x)
                if ix0 <= ix1:
                    fx0 = float(ix0) + 0.5
                    w0 = ((y2 - y1) * (fx0 - x1) - (x2 - x1) * (fy - y1)) * inv_area
                    w1 = ((y0 - y2) * (fx0 - x2) - (x0 - x2) * (fy - y2)) * inv_area
                    iw = w0 * iw0 + w1 * iw1 + (1.0 - w0 - w1) * iw2
                    for px in range(ix0, ix1 + 1):
                        if iw > 0.0:
                            depth = (1.0 / iw - near) * inv_depth_range
                            if depth >= -1e-5 and depth <= 1.0:
                                d24 = gs.u32(gs.max(depth, 0.0) * 16777215.0)
                                if select == 0:
                                    gs.atomic_min(target, (py, px), d24)
                                elif zbuf[py, px] == d24:
                                    gs.atomic_min(target, (py, px), gs.u32(t))
                        iw += diw_dx


# ───────────────────────── 3. Deferred PBR + Point/SpotLight + PCF Shadows ─────────────────────────


@gs.kernel
def shade_pbr(
    visbuf: gs.u32["h", "w"],
    tri_idx: gs.i32["nt", "c3"],
    tri_mat: gs.i32["nt"],
    sx: gs.f32["nv"],
    sy: gs.f32["nv"],
    inv_w: gs.f32["nv"],
    wpos: gs.f32["nv", "c3"],
    wnorm: gs.f32["nv", "c3"],
    v_uv: gs.f32["nv", "c2"],
    materials: gs.f32["nm", "c16"],
    lights: gs.f32["nl", "c16"],
    shadow_map: gs.u32["sh", "sw"],
    light_vp: gs.f32["c16"],
    tex_pool: gs.u32["tp"],
    tex_meta: gs.i32["ntex", "c4"],
    dyn_atlas: gs.u32["dh", "dw"],
    cam_x: gs.f32,
    cam_y: gs.f32,
    cam_z: gs.f32,
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
    fog_density: gs.f32,
    fog_r: gs.f32,
    fog_g: gs.f32,
    fog_b: gs.f32,
    sky_tid: gs.i32,
    sky_int: gs.f32,
    use_shadows: gs.i32,
    wireframe: gs.i32,
    near: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Shading différé PBR : Cook-Torrance GGX, soleil + PCF 3×3, Point/SpotLight, skybox HDRI, brouillard."""
    py, px = gs.index()
    h = visbuf.shape[0]
    w = visbuf.shape[1]
    vis = visbuf[py, px]
    mask_u = gs.u32(255)
    inv255 = 0.0039215686

    if vis == gs.u32(4294967295):
        aspect = float(w) / float(h)
        ndc_x = ((float(px) + 0.5) / float(w) * 2.0 - 1.0) * aspect * tan_hfov
        ndc_y = (1.0 - (float(py) + 0.5) / float(h) * 2.0) * tan_hfov
        rdx = fwd_x + right_x * ndc_x + up_x * ndc_y
        rdy = fwd_y + right_y * ndc_x + up_y * ndc_y
        rdz = fwd_z + right_z * ndc_x + up_z * ndc_y
        inv_rd = 1.0 / gs.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
        rdx = rdx * inv_rd
        rdy = rdy * inv_rd
        rdz = rdz * inv_rd

        if sky_tid >= 0:
            sky_u = gs.atan2(rdz, rdx) * 0.15915494 + 0.5
            sky_v = gs.atan2(gs.sqrt(rdx * rdx + rdz * rdz), rdy) * 0.31830989
            tw = tex_meta[sky_tid, 2]
            m3 = tex_meta[sky_tid, 3]
            th = m3 & 1048575
            is_dyn = (m3 >> 20) & 1
            fx_t = sky_u * float(tw) - 0.5
            fy_t = sky_v * float(th) - 0.5
            x0t = gs.min(gs.max(int(gs.floor(fx_t)), 0), tw - 1)
            y0t = gs.min(gs.max(int(gs.floor(fy_t)), 0), th - 1)
            x1t = (x0t + 1) % tw
            y1t = gs.min(y0t + 1, th - 1)
            ftx = gs.min(gs.max(fx_t - float(x0t), 0.0), 1.0)
            fty = gs.min(gs.max(fy_t - float(y0t), 0.0), 1.0)
            c00 = gs.u32(0)
            c10 = gs.u32(0)
            c01 = gs.u32(0)
            c11 = gs.u32(0)
            if is_dyn != 0:
                ox_d = tex_meta[sky_tid, 0]
                oy_d = tex_meta[sky_tid, 1]
                c00 = dyn_atlas[oy_d + y0t, ox_d + x0t]
                c10 = dyn_atlas[oy_d + y0t, ox_d + x1t]
                c01 = dyn_atlas[oy_d + y1t, ox_d + x0t]
                c11 = dyn_atlas[oy_d + y1t, ox_d + x1t]
            else:
                off_s = tex_meta[sky_tid, 0]
                c00 = tex_pool[off_s + y0t * tw + x0t]
                c10 = tex_pool[off_s + y0t * tw + x1t]
                c01 = tex_pool[off_s + y1t * tw + x0t]
                c11 = tex_pool[off_s + y1t * tw + x1t]
            w00 = (1.0 - ftx) * (1.0 - fty)
            w10 = ftx * (1.0 - fty)
            w01 = (1.0 - ftx) * fty
            w11 = ftx * fty
            sr = (w00 * float(c00 & mask_u) + w10 * float(c10 & mask_u) + w01 * float(c01 & mask_u) + w11 * float(c11 & mask_u)) * inv255
            sg = (w00 * float((c00 >> gs.u32(8)) & mask_u) + w10 * float((c10 >> gs.u32(8)) & mask_u) + w01 * float((c01 >> gs.u32(8)) & mask_u) + w11 * float((c11 >> gs.u32(8)) & mask_u)) * inv255
            sb = (w00 * float((c00 >> gs.u32(16)) & mask_u) + w10 * float((c10 >> gs.u32(16)) & mask_u) + w01 * float((c01 >> gs.u32(16)) & mask_u) + w11 * float((c11 >> gs.u32(16)) & mask_u)) * inv255
            return sr * sr * sky_int, sg * sg * sky_int, sb * sb * sky_int

        t_sky = gs.min(gs.max(rdy * 0.5 + 0.5, 0.0), 1.0)
        oh = 1.0 - gs.abs(rdy)
        oh2 = oh * oh
        hz = oh2 * oh2 * oh2
        cr = 0.015 + 0.055 * (1.0 - t_sky) + 0.16 * hz
        cg = 0.022 + 0.070 * (1.0 - t_sky) + 0.11 * hz
        cb = 0.065 + 0.110 * (1.0 - t_sky) + 0.22 * hz
        sdot = gs.max(rdx * sun_x + rdy * sun_y + rdz * sun_z, 0.0)
        s2 = sdot * sdot
        s4 = s2 * s2
        s16 = s4 * s4 * s4 * s4
        s256 = s16 * s16
        halo = s16 * 0.75 + s256 * 6.0
        cr += sun_r * halo
        cg += sun_g * halo
        cb += sun_b * halo
        return cr, cg, cb

    t = gs.i32(vis)
    i0 = tri_idx[t, 0]
    i1 = tri_idx[t, 1]
    i2 = tri_idx[t, 2]

    x0 = sx[i0]
    y0 = sy[i0]
    iw0 = inv_w[i0]
    x1 = sx[i1]
    y1 = sy[i1]
    iw1 = inv_w[i1]
    x2 = sx[i2]
    y2 = sy[i2]
    iw2 = inv_w[i2]

    fx = float(px) + 0.5
    fy = float(py) + 0.5
    area = (y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0)
    inv_area = 1.0 / gs.max(gs.abs(area), 1e-6)
    sw0 = ((y2 - y1) * (fx - x1) - (x2 - x1) * (fy - y1)) * inv_area
    sw1 = ((y0 - y2) * (fx - x2) - (x0 - x2) * (fy - y2)) * inv_area
    sw2 = 1.0 - sw0 - sw1

    pw0 = sw0 * iw0
    pw1 = sw1 * iw1
    pw2 = sw2 * iw2
    inv_pw = 1.0 / gs.max(pw0 + pw1 + pw2, 1e-8)
    b0 = pw0 * inv_pw
    b1 = pw1 * inv_pw
    b2 = pw2 * inv_pw
    if gs.min(iw0, iw1, iw2) <= 0.0:
        # Triangle qui traverse le plan proche : barycentriques homogènes (valables derrière la caméra)
        fw = float(w)
        fh = float(h)
        hw0 = iw0 + near + 1.0
        hx0 = (x0 * 0.5 + 0.5 * hw0) * fw
        hy0 = (0.5 * hw0 - y0 * 0.5) * fh
        if iw0 > 0.0:
            hw0 = 1.0 / iw0
            hx0 = x0 * hw0
            hy0 = y0 * hw0
        hw1 = iw1 + near + 1.0
        hx1 = (x1 * 0.5 + 0.5 * hw1) * fw
        hy1 = (0.5 * hw1 - y1 * 0.5) * fh
        if iw1 > 0.0:
            hw1 = 1.0 / iw1
            hx1 = x1 * hw1
            hy1 = y1 * hw1
        hw2 = iw2 + near + 1.0
        hx2 = (x2 * 0.5 + 0.5 * hw2) * fw
        hy2 = (0.5 * hw2 - y2 * 0.5) * fh
        if iw2 > 0.0:
            hw2 = 1.0 / iw2
            hx2 = x2 * hw2
            hy2 = y2 * hw2
        gx0 = hx0 - fx * hw0
        gy0 = hy0 - fy * hw0
        gx1 = hx1 - fx * hw1
        gy1 = hy1 - fy * hw1
        gx2 = hx2 - fx * hw2
        gy2 = hy2 - fy * hw2
        e0 = gx1 * gy2 - gx2 * gy1
        e1 = gx2 * gy0 - gx0 * gy2
        e2 = gx0 * gy1 - gx1 * gy0
        es = e0 + e1 + e2
        inv_e = 1.0 / (es if gs.abs(es) > 1e-12 else 1e-12)
        b0 = e0 * inv_e
        b1 = e1 * inv_e
        b2 = e2 * inv_e

    pos_x = b0 * wpos[i0, 0] + b1 * wpos[i1, 0] + b2 * wpos[i2, 0]
    pos_y = b0 * wpos[i0, 1] + b1 * wpos[i1, 1] + b2 * wpos[i2, 1]
    pos_z = b0 * wpos[i0, 2] + b1 * wpos[i1, 2] + b2 * wpos[i2, 2]

    nx = b0 * wnorm[i0, 0] + b1 * wnorm[i1, 0] + b2 * wnorm[i2, 0]
    ny = b0 * wnorm[i0, 1] + b1 * wnorm[i1, 1] + b2 * wnorm[i2, 1]
    nz = b0 * wnorm[i0, 2] + b1 * wnorm[i1, 2] + b2 * wnorm[i2, 2]
    inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx = nx * inv_n
    ny = ny * inv_n
    nz = nz * inv_n

    uu = b0 * v_uv[i0, 0] + b1 * v_uv[i1, 0] + b2 * v_uv[i2, 0]
    vv = b0 * v_uv[i0, 1] + b1 * v_uv[i1, 1] + b2 * v_uv[i2, 1]

    m = tri_mat[t]
    ar = materials[m, 0]
    ag = materials[m, 1]
    ab = materials[m, 2]
    metallic = materials[m, 3]
    roughness = gs.max(materials[m, 4], 0.05)
    pat = int(materials[m, 5])
    pscale = materials[m, 6]
    er = materials[m, 7]
    eg = materials[m, 8]
    eb = materials[m, 9]
    a2r = materials[m, 10]
    a2g = materials[m, 11]
    a2b = materials[m, 12]
    alb_tid = int(materials[m, 13])
    em_tid = int(materials[m, 14])
    rough_tid = int(materials[m, 15])

    # Échantillonnage textures PBR (pool statique + atlas dynamique 0-copie pour RenderTarget/vidéo/GPU)
    u_tex = uu - gs.floor(uu)
    v_tex = 1.0 - (vv - gs.floor(vv))
    if pat == 5:
        # Miroir planaire vertical (plane_z) : projection écran symétrique horizontale
        u_tex = 1.0 - (float(px) + 0.5) / float(w)
        v_tex = (float(py) + 0.5) / float(h)
    elif pat == 6:
        # Miroir planaire horizontal (plane_y, eau/sol) : projection écran symétrique verticale
        u_tex = (float(px) + 0.5) / float(w)
        v_tex = 1.0 - (float(py) + 0.5) / float(h)

    if alb_tid >= 0:
        tw = tex_meta[alb_tid, 2]
        m3 = tex_meta[alb_tid, 3]
        th = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        is_lin = (m3 >> 21) & 1
        tx0 = gs.min(gs.max(int(u_tex * float(tw)), 0), tw - 1)
        ty0 = gs.min(gs.max(int(v_tex * float(th)), 0), th - 1)
        tr = 0.0
        tg = 0.0
        tb = 0.0
        if is_lin != 0:
            fx_t = u_tex * float(tw) - 0.5
            fy_t = v_tex * float(th) - 0.5
            x0t = gs.min(gs.max(int(gs.floor(fx_t)), 0), tw - 1)
            y0t = gs.min(gs.max(int(gs.floor(fy_t)), 0), th - 1)
            x1t = gs.min(x0t + 1, tw - 1)
            y1t = gs.min(y0t + 1, th - 1)
            ftx = gs.min(gs.max(fx_t - float(x0t), 0.0), 1.0)
            fty = gs.min(gs.max(fy_t - float(y0t), 0.0), 1.0)
            c00 = gs.u32(0)
            c10 = gs.u32(0)
            c01 = gs.u32(0)
            c11 = gs.u32(0)
            if is_dyn != 0:
                ox_d = tex_meta[alb_tid, 0]
                oy_d = tex_meta[alb_tid, 1]
                c00 = dyn_atlas[oy_d + y0t, ox_d + x0t]
                c10 = dyn_atlas[oy_d + y0t, ox_d + x1t]
                c01 = dyn_atlas[oy_d + y1t, ox_d + x0t]
                c11 = dyn_atlas[oy_d + y1t, ox_d + x1t]
            else:
                off_s = tex_meta[alb_tid, 0]
                c00 = tex_pool[off_s + y0t * tw + x0t]
                c10 = tex_pool[off_s + y0t * tw + x1t]
                c01 = tex_pool[off_s + y1t * tw + x0t]
                c11 = tex_pool[off_s + y1t * tw + x1t]
            w00 = (1.0 - ftx) * (1.0 - fty)
            w10 = ftx * (1.0 - fty)
            w01 = (1.0 - ftx) * fty
            w11 = ftx * fty
            tr = (w00 * float(c00 & mask_u) + w10 * float(c10 & mask_u) + w01 * float(c01 & mask_u) + w11 * float(c11 & mask_u)) * inv255
            tg = (w00 * float((c00 >> gs.u32(8)) & mask_u) + w10 * float((c10 >> gs.u32(8)) & mask_u) + w01 * float((c01 >> gs.u32(8)) & mask_u) + w11 * float((c11 >> gs.u32(8)) & mask_u)) * inv255
            tb = (w00 * float((c00 >> gs.u32(16)) & mask_u) + w10 * float((c10 >> gs.u32(16)) & mask_u) + w01 * float((c01 >> gs.u32(16)) & mask_u) + w11 * float((c11 >> gs.u32(16)) & mask_u)) * inv255
        else:
            cs = gs.u32(0)
            if is_dyn != 0:
                cs = dyn_atlas[tex_meta[alb_tid, 1] + ty0, tex_meta[alb_tid, 0] + tx0]
            else:
                cs = tex_pool[tex_meta[alb_tid, 0] + ty0 * tw + tx0]
            tr = float(cs & mask_u) * inv255
            tg = float((cs >> gs.u32(8)) & mask_u) * inv255
            tb = float((cs >> gs.u32(16)) & mask_u) * inv255
        ar = ar * tr * tr
        ag = ag * tg * tg
        ab = ab * tb * tb

    if em_tid >= 0:
        tw = tex_meta[em_tid, 2]
        m3 = tex_meta[em_tid, 3]
        th = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        is_lin = (m3 >> 21) & 1
        etr = 0.0
        etg = 0.0
        etb = 0.0
        if is_lin != 0:
            fx_t = u_tex * float(tw) - 0.5
            fy_t = v_tex * float(th) - 0.5
            x0t = gs.min(gs.max(int(gs.floor(fx_t)), 0), tw - 1)
            y0t = gs.min(gs.max(int(gs.floor(fy_t)), 0), th - 1)
            x1t = gs.min(x0t + 1, tw - 1)
            y1t = gs.min(y0t + 1, th - 1)
            ftx = gs.min(gs.max(fx_t - float(x0t), 0.0), 1.0)
            fty = gs.min(gs.max(fy_t - float(y0t), 0.0), 1.0)
            c00 = gs.u32(0)
            c10 = gs.u32(0)
            c01 = gs.u32(0)
            c11 = gs.u32(0)
            if is_dyn != 0:
                ox_d = tex_meta[em_tid, 0]
                oy_d = tex_meta[em_tid, 1]
                c00 = dyn_atlas[oy_d + y0t, ox_d + x0t]
                c10 = dyn_atlas[oy_d + y0t, ox_d + x1t]
                c01 = dyn_atlas[oy_d + y1t, ox_d + x0t]
                c11 = dyn_atlas[oy_d + y1t, ox_d + x1t]
            else:
                off_s = tex_meta[em_tid, 0]
                c00 = tex_pool[off_s + y0t * tw + x0t]
                c10 = tex_pool[off_s + y0t * tw + x1t]
                c01 = tex_pool[off_s + y1t * tw + x0t]
                c11 = tex_pool[off_s + y1t * tw + x1t]
            w00 = (1.0 - ftx) * (1.0 - fty)
            w10 = ftx * (1.0 - fty)
            w01 = (1.0 - ftx) * fty
            w11 = ftx * fty
            etr = (w00 * float(c00 & mask_u) + w10 * float(c10 & mask_u) + w01 * float(c01 & mask_u) + w11 * float(c11 & mask_u)) * inv255
            etg = (w00 * float((c00 >> gs.u32(8)) & mask_u) + w10 * float((c10 >> gs.u32(8)) & mask_u) + w01 * float((c01 >> gs.u32(8)) & mask_u) + w11 * float((c11 >> gs.u32(8)) & mask_u)) * inv255
            etb = (w00 * float((c00 >> gs.u32(16)) & mask_u) + w10 * float((c10 >> gs.u32(16)) & mask_u) + w01 * float((c01 >> gs.u32(16)) & mask_u) + w11 * float((c11 >> gs.u32(16)) & mask_u)) * inv255
        else:
            tx0 = gs.min(gs.max(int(u_tex * float(tw)), 0), tw - 1)
            ty0 = gs.min(gs.max(int(v_tex * float(th)), 0), th - 1)
            cs = gs.u32(0)
            if is_dyn != 0:
                cs = dyn_atlas[tex_meta[em_tid, 1] + ty0, tex_meta[em_tid, 0] + tx0]
            else:
                cs = tex_pool[tex_meta[em_tid, 0] + ty0 * tw + tx0]
            etr = float(cs & mask_u) * inv255
            etg = float((cs >> gs.u32(8)) & mask_u) * inv255
            etb = float((cs >> gs.u32(16)) & mask_u) * inv255
        if pat == 5 or pat == 6:
            # Miroir planaire optique : inverse ACES + dé-gamma calibrés 1:1 (sans double IBL laiteux)
            lr_m = etr * etr
            lg_m = etg * etg
            lb_m = etb * etb
            mr = er * (lr_m / gs.max(1.22 - 0.76 * lr_m, 0.16))
            mg = eg * (lg_m / gs.max(1.22 - 0.76 * lg_m, 0.16))
            mb = eb * (lb_m / gs.max(1.22 - 0.76 * lb_m, 0.16))
            return mr, mg, mb
        else:
            er = er * etr * etr
            eg = eg * etg * etg
            eb = eb * etb * etb

    if rough_tid >= 0:
        tw = tex_meta[rough_tid, 2]
        m3 = tex_meta[rough_tid, 3]
        th = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        tx0 = gs.min(gs.max(int(u_tex * float(tw)), 0), tw - 1)
        ty0 = gs.min(gs.max(int(v_tex * float(th)), 0), th - 1)
        cs = gs.u32(0)
        if is_dyn != 0:
            cs = dyn_atlas[tex_meta[rough_tid, 1] + ty0, tex_meta[rough_tid, 0] + tx0]
        else:
            cs = tex_pool[tex_meta[rough_tid, 0] + ty0 * tw + tx0]
        roughness = gs.max(roughness * (float(cs & mask_u) * inv255), 0.05)

    mix_k = 0.0
    if pat == 1:
        cx_i = int(gs.floor(uu * pscale))
        cy_i = int(gs.floor(vv * pscale))
        if ((cx_i + cy_i) & 1) != 0:
            mix_k = 1.0
    elif pat == 2:
        fx_u = (uu * pscale) % 1.0
        fy_v = (vv * pscale) % 1.0
        if fx_u < 0.08 or fx_u > 0.92 or fy_v < 0.08 or fy_v > 0.92:
            mix_k = 1.0
    elif pat == 3:
        mix_k = 0.25 + 0.25 * gs.sin(vv * pscale * 6.2831853)
    elif pat == 4:
        wave = gs.sin(
            (pos_x * 2.1 + pos_z * 1.7 + gs.sin(pos_y * 3.4 + pos_x * 1.9) * 0.85) * pscale
        )
        w01 = wave * 0.5 + 0.5
        mix_k = w01 * w01 * w01 * 0.75

    if mix_k > 0.0:
        ar = ar * (1.0 - mix_k) + a2r * mix_k
        ag = ag * (1.0 - mix_k) + a2g * mix_k
        ab = ab * (1.0 - mix_k) + a2b * mix_k
        if pat == 1 and mix_k > 0.5:
            roughness = gs.min(roughness + 0.22, 0.95)

    vx = cam_x - pos_x
    vy = cam_y - pos_y
    vz = cam_z - pos_z
    dist = gs.sqrt(vx * vx + vy * vy + vz * vz)
    inv_v = 1.0 / gs.max(dist, 1e-6)
    vx = vx * inv_v
    vy = vy * inv_v
    vz = vz * inv_v
    ndotv = gs.max(nx * vx + ny * vy + nz * vz, 0.001)

    f0r = 0.04 * (1.0 - metallic) + ar * metallic
    f0g = 0.04 * (1.0 - metallic) + ag * metallic
    f0b = 0.04 * (1.0 - metallic) + ab * metallic
    a_r = roughness * roughness
    a2 = a_r * a_r
    k_g = (roughness + 1.0) * (roughness + 1.0) * 0.125
    gv = ndotv / (ndotv * (1.0 - k_g) + k_g)
    inv_pi = 0.31830989

    shadow = 1.0
    if use_shadows != 0:
        bx = pos_x + nx * 0.055
        by = pos_y + ny * 0.055
        bz = pos_z + nz * 0.055
        ls_x = light_vp[0] * bx + light_vp[1] * by + light_vp[2] * bz + light_vp[3]
        ls_y = light_vp[4] * bx + light_vp[5] * by + light_vp[6] * bz + light_vp[7]
        ls_z = light_vp[8] * bx + light_vp[9] * by + light_vp[10] * bz + light_vp[11]
        sh = shadow_map.shape[0]
        sw = shadow_map.shape[1]
        su = int((ls_x * 0.5 + 0.5) * float(sw))
        sv = int((0.5 - ls_y * 0.5) * float(sh))
        if su >= 1 and su < sw - 1 and sv >= 1 and sv < sh - 1 and ls_z > 0.0 and ls_z < 1.0:
            ref_d = gs.u32(gs.max(ls_z - 0.0022, 0.0) * 4294967040.0)
            lit_sum = 0.0
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    if ref_d <= shadow_map[sv + oy, su + ox]:
                        lit_sum += 1.0
            shadow = lit_sum * 0.11111111

    lr = er
    lg = eg
    lb = eb

    # 1) Soleil directionnel
    ndotl = gs.max(nx * sun_x + ny * sun_y + nz * sun_z, 0.0)
    if ndotl > 0.0 and shadow > 0.0:
        hx = vx + sun_x
        hy = vy + sun_y
        hz = vz + sun_z
        inv_h = 1.0 / gs.max(gs.sqrt(hx * hx + hy * hy + hz * hz), 1e-6)
        hx = hx * inv_h
        hy = hy * inv_h
        hz = hz * inv_h
        ndoth = gs.max(nx * hx + ny * hy + nz * hz, 0.0)
        vdoth = gs.max(vx * hx + vy * hy + vz * hz, 0.0)

        denom = ndoth * ndoth * (a2 - 1.0) + 1.0
        ndf = a2 / gs.max(3.14159265 * denom * denom, 1e-6)
        gl = ndotl / (ndotl * (1.0 - k_g) + k_g)
        geom = gv * gl
        omv = 1.0 - vdoth
        f_pow = omv * omv * omv * omv * omv
        fr = f0r + (1.0 - f0r) * f_pow
        fg = f0g + (1.0 - f0g) * f_pow
        fb = f0b + (1.0 - f0b) * f_pow

        spec_s = ndf * geom / gs.max(4.0 * ndotv * ndotl, 1e-4)
        kdr = (1.0 - fr) * (1.0 - metallic) * ar * inv_pi
        kdg = (1.0 - fg) * (1.0 - metallic) * ag * inv_pi
        kdb = (1.0 - fb) * (1.0 - metallic) * ab * inv_pi

        rad_r = sun_r * sun_int * ndotl * shadow
        rad_g = sun_g * sun_int * ndotl * shadow
        rad_b = sun_b * sun_int * ndotl * shadow
        lr += (kdr + fr * spec_s) * rad_r
        lg += (kdg + fg * spec_s) * rad_g
        lb += (kdb + fb * spec_s) * rad_b

    # 2) Lumières locales unifiées (PointLight = 1, SpotLight = 2)
    for li in range(lights.shape[0]):
        lkind = lights[li, 7]
        if lkind > 0.5:
            dlx = lights[li, 0] - pos_x
            dly = lights[li, 1] - pos_y
            dlz = lights[li, 2] - pos_z
            d2_raw = dlx * dlx + dly * dly + dlz * dlz
            d2 = d2_raw + 0.35
            inv_dl = 1.0 / gs.sqrt(d2)
            lx = dlx * inv_dl
            ly = dly * inv_dl
            lz = dlz * inv_dl
            nl = gs.max(nx * lx + ny * ly + nz * lz, 0.0)
            if nl > 0.0:
                atten = lights[li, 3] / d2
                lrange = lights[li, 13]
                if lrange > 0.0:
                    rd_ratio = gs.sqrt(d2_raw) / lrange
                    if rd_ratio >= 1.0:
                        atten = 0.0
                    else:
                        rf = 1.0 - rd_ratio * rd_ratio
                        atten = atten * rf * rf
                if lkind > 1.5 and atten > 0.0:
                    # SpotLight : atténuation douce entre cos_outer (12) et cos_inner (11)
                    spot_cos = -(lx * lights[li, 8] + ly * lights[li, 9] + lz * lights[li, 10])
                    c_in = lights[li, 11]
                    c_out = lights[li, 12]
                    spot_t = gs.min(gs.max((spot_cos - c_out) / gs.max(c_in - c_out, 1e-4), 0.0), 1.0)
                    atten = atten * spot_t * spot_t * (3.0 - 2.0 * spot_t)
                if atten > 0.0:
                    hx = vx + lx
                    hy = vy + ly
                    hz = vz + lz
                    inv_h = 1.0 / gs.max(gs.sqrt(hx * hx + hy * hy + hz * hz), 1e-6)
                    hx = hx * inv_h
                    hy = hy * inv_h
                    hz = hz * inv_h
                    ndoth = gs.max(nx * hx + ny * hy + nz * hz, 0.0)
                    vdoth = gs.max(vx * hx + vy * hy + vz * hz, 0.0)

                    denom = ndoth * ndoth * (a2 - 1.0) + 1.0
                    ndf = a2 / gs.max(3.14159265 * denom * denom, 1e-6)
                    gl = nl / (nl * (1.0 - k_g) + k_g)
                    geom = gv * gl
                    omv = 1.0 - vdoth
                    f_pow = omv * omv * omv * omv * omv
                    fr = f0r + (1.0 - f0r) * f_pow
                    fg = f0g + (1.0 - f0g) * f_pow
                    fb = f0b + (1.0 - f0b) * f_pow
                    spec_s = ndf * geom / gs.max(4.0 * ndotv * nl, 1e-4)

                    kdr = (1.0 - fr) * (1.0 - metallic) * ar * inv_pi
                    kdg = (1.0 - fg) * (1.0 - metallic) * ag * inv_pi
                    kdb = (1.0 - fb) * (1.0 - metallic) * ab * inv_pi

                    rad_r = lights[li, 4] * atten * nl
                    rad_g = lights[li, 5] * atten * nl
                    rad_b = lights[li, 6] * atten * nl
                    lr += (kdr + fr * spec_s) * rad_r
                    lg += (kdg + fg * spec_s) * rad_g
                    lb += (kdb + fb * spec_s) * rad_b

    # 3) Ambiant hémisphérique + IBL ciel
    hemi = ny * 0.5 + 0.5
    amb_r = 0.05 + 0.14 * hemi
    amb_g = 0.06 + 0.18 * hemi
    amb_b = 0.09 + 0.28 * hemi
    lr += ar * (1.0 - metallic) * amb_r
    lg += ag * (1.0 - metallic) * amb_g
    lb += ab * (1.0 - metallic) * amb_b

    rx = 2.0 * ndotv * nx - vx
    ry = 2.0 * ndotv * ny - vy
    rz = 2.0 * ndotv * nz - vz
    env_r = 0.0
    env_g = 0.0
    env_b = 0.0
    gloss = 1.0 - roughness
    if sky_tid >= 0:
        ru = gs.atan2(rz, rx) * 0.15915494 + 0.5
        rv = gs.atan2(gs.sqrt(rx * rx + rz * rz), ry) * 0.31830989
        tw = tex_meta[sky_tid, 2]
        m3 = tex_meta[sky_tid, 3]
        th = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        rtx = gs.min(gs.max(int(ru * float(tw)), 0), tw - 1)
        rty = gs.min(gs.max(int(rv * float(th)), 0), th - 1)
        rcs = gs.u32(0)
        if is_dyn != 0:
            rcs = dyn_atlas[tex_meta[sky_tid, 1] + rty, tex_meta[sky_tid, 0] + rtx]
        else:
            rcs = tex_pool[tex_meta[sky_tid, 0] + rty * tw + rtx]
        er_s = float(rcs & mask_u) * inv255
        eg_s = float((rcs >> gs.u32(8)) & mask_u) * inv255
        eb_s = float((rcs >> gs.u32(16)) & mask_u) * inv255
        env_r = er_s * er_s * sky_int * 1.4
        env_g = eg_s * eg_s * sky_int * 1.4
        env_b = eb_s * eb_s * sky_int * 1.4
    else:
        r_sky = gs.min(gs.max(ry * 0.5 + 0.5, 0.0), 1.0)
        rh1 = 1.0 - gs.abs(ry)
        rh2 = rh1 * rh1
        r_hz = rh2 * rh2
        env_r = 0.05 + 0.14 * (1.0 - r_sky) + 0.26 * r_hz
        env_g = 0.07 + 0.18 * (1.0 - r_sky) + 0.24 * r_hz
        env_b = 0.14 + 0.26 * (1.0 - r_sky) + 0.36 * r_hz
        rsun = gs.max(rx * sun_x + ry * sun_y + rz * sun_z, 0.0)
        sun_refl = gs.exp(gs.log(gs.max(rsun, 1e-4)) * (6.0 + gloss * 48.0)) * (0.35 + gloss * 1.1)
        env_r += sun_r * sun_refl
        env_g += sun_g * sun_refl
        env_b += sun_b * sun_refl

    omv_e = 1.0 - ndotv
    f_env = omv_e * omv_e * omv_e * omv_e * omv_e
    env_w = 0.15 + 0.85 * gloss * gloss
    lr += (f0r + (gs.max(gloss, f0r) - f0r) * f_env) * env_r * env_w
    lg += (f0g + (gs.max(gloss, f0g) - f0g) * f_env) * env_g * env_w
    lb += (f0b + (gs.max(gloss, f0b) - f0b) * f_env) * env_b * env_w

    # 4) Brouillard exponentiel configurable
    if fog_density > 0.0:
        fog = 1.0 - gs.exp(-dist * dist * fog_density)
        fog = gs.min(gs.max(fog, 0.0), 0.85)
        lr = lr * (1.0 - fog) + fog_r * fog
        lg = lg * (1.0 - fog) + fog_g * fog
        lb = lb * (1.0 - fog) + fog_b * fog

    # 5) Mode filaire optionnel
    if wireframe != 0:
        edge = gs.min(b0, b1, b2)
        if edge < 0.04:
            lr = 0.20
            lg = 1.35
            lb = 1.85

    return lr, lg, lb


# ───────────────────────── 4. Post-Traitement (Bloom, ACES, FXAA, Multi-Viewport) ─────────────────────────


@gs.kernel(group=64)
def bloom_down_h(
    hr: gs.f32["h", "w"],
    hg: gs.f32["h", "w"],
    hb: gs.f32["h", "w"],
    half_ref: gs.f32["hh", "hw"],
    threshold: gs.f32,
) -> (gs.f32["hh", "hw"], gs.f32["hh", "hw"], gs.f32["hh", "hw"]):
    hy, hx = gs.index()
    w = hr.shape[1]
    h = hr.shape[0]
    hw = half_ref.shape[1]
    lane = gs.lane()

    sr = gs.shared(gs.f32, 72)
    sg = gs.shared(gs.f32, 72)
    sb = gs.shared(gs.f32, 72)

    y0 = gs.min(hy * 2, h - 1)
    y1 = gs.min(hy * 2 + 1, h - 1)
    x0 = gs.min(hx * 2, w - 1)
    x1 = gs.min(hx * 2 + 1, w - 1)

    r0 = 0.25 * (hr[y0, x0] + hr[y0, x1] + hr[y1, x0] + hr[y1, x1])
    g0 = 0.25 * (hg[y0, x0] + hg[y0, x1] + hg[y1, x0] + hg[y1, x1])
    b0 = 0.25 * (hb[y0, x0] + hb[y0, x1] + hb[y1, x0] + hb[y1, x1])
    lum0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0
    k0 = gs.max(lum0 - threshold, 0.0) / gs.max(lum0, 1e-4)
    sr[lane + 4] = r0 * k0
    sg[lane + 4] = g0 * k0
    sb[lane + 4] = b0 * k0

    if lane < 4:
        gx = gs.max(hx - 4, 0)
        lx0 = gs.min(gx * 2, w - 1)
        lx1 = gs.min(gx * 2 + 1, w - 1)
        rl = 0.25 * (hr[y0, lx0] + hr[y0, lx1] + hr[y1, lx0] + hr[y1, lx1])
        gl = 0.25 * (hg[y0, lx0] + hg[y0, lx1] + hg[y1, lx0] + hg[y1, lx1])
        bl = 0.25 * (hb[y0, lx0] + hb[y0, lx1] + hb[y1, lx0] + hb[y1, lx1])
        ll = 0.2126 * rl + 0.7152 * gl + 0.0722 * bl
        kl = gs.max(ll - threshold, 0.0) / gs.max(ll, 1e-4)
        sr[lane] = rl * kl
        sg[lane] = gl * kl
        sb[lane] = bl * kl
    if lane >= 60:
        gx = gs.min(hx + 4, hw - 1)
        rx0 = gs.min(gx * 2, w - 1)
        rx1 = gs.min(gx * 2 + 1, w - 1)
        rr = 0.25 * (hr[y0, rx0] + hr[y0, rx1] + hr[y1, rx0] + hr[y1, rx1])
        gr = 0.25 * (hg[y0, rx0] + hg[y0, rx1] + hg[y1, rx0] + hg[y1, rx1])
        br = 0.25 * (hb[y0, rx0] + hb[y0, rx1] + hb[y1, rx0] + hb[y1, rx1])
        lr_ = 0.2126 * rr + 0.7152 * gr + 0.0722 * br
        kr = gs.max(lr_ - threshold, 0.0) / gs.max(lr_, 1e-4)
        sr[lane + 8] = rr * kr
        sg[lane + 8] = gr * kr
        sb[lane + 8] = br * kr

    gs.barrier()

    c = lane + 4
    or_ = (
        0.227027 * sr[c]
        + 0.1945946 * (sr[c - 1] + sr[c + 1])
        + 0.1216216 * (sr[c - 2] + sr[c + 2])
        + 0.054054 * (sr[c - 3] + sr[c + 3])
        + 0.016216 * (sr[c - 4] + sr[c + 4])
    )
    og_ = (
        0.227027 * sg[c]
        + 0.1945946 * (sg[c - 1] + sg[c + 1])
        + 0.1216216 * (sg[c - 2] + sg[c + 2])
        + 0.054054 * (sg[c - 3] + sg[c + 3])
        + 0.016216 * (sg[c - 4] + sg[c + 4])
    )
    ob_ = (
        0.227027 * sb[c]
        + 0.1945946 * (sb[c - 1] + sb[c + 1])
        + 0.1216216 * (sb[c - 2] + sb[c + 2])
        + 0.054054 * (sb[c - 3] + sb[c + 3])
        + 0.016216 * (sb[c - 4] + sb[c + 4])
    )
    return or_, og_, ob_


@gs.kernel
def bloom_v(
    br: gs.f32["hh", "hw"],
    bg: gs.f32["hh", "hw"],
    bb: gs.f32["hh", "hw"],
) -> (gs.f32["hh", "hw"], gs.f32["hh", "hw"], gs.f32["hh", "hw"]):
    py, px = gs.index()
    h = br.shape[0]
    ym4 = gs.max(py - 4, 0)
    ym3 = gs.max(py - 3, 0)
    ym2 = gs.max(py - 2, 0)
    ym1 = gs.max(py - 1, 0)
    yp1 = gs.min(py + 1, h - 1)
    yp2 = gs.min(py + 2, h - 1)
    yp3 = gs.min(py + 3, h - 1)
    yp4 = gs.min(py + 4, h - 1)

    or_ = (
        0.227027 * br[py, px]
        + 0.1945946 * (br[ym1, px] + br[yp1, px])
        + 0.1216216 * (br[ym2, px] + br[yp2, px])
        + 0.054054 * (br[ym3, px] + br[yp3, px])
        + 0.016216 * (br[ym4, px] + br[yp4, px])
    )
    og_ = (
        0.227027 * bg[py, px]
        + 0.1945946 * (bg[ym1, px] + bg[yp1, px])
        + 0.1216216 * (bg[ym2, px] + bg[yp2, px])
        + 0.054054 * (bg[ym3, px] + bg[yp3, px])
        + 0.016216 * (bg[ym4, px] + bg[yp4, px])
    )
    ob_ = (
        0.227027 * bb[py, px]
        + 0.1945946 * (bb[ym1, px] + bb[yp1, px])
        + 0.1216216 * (bb[ym2, px] + bb[yp2, px])
        + 0.054054 * (bb[ym3, px] + bb[yp3, px])
        + 0.016216 * (bb[ym4, px] + bb[yp4, px])
    )
    return or_, og_, ob_


@gs.kernel
def tonemap_pack(
    hr: gs.f32["h", "w"],
    hg: gs.f32["h", "w"],
    hb: gs.f32["h", "w"],
    blr: gs.f32["hh", "hw"],
    blg: gs.f32["hh", "hw"],
    blb: gs.f32["hh", "hw"],
    bloom_strength: gs.f32,
    exposure: gs.f32,
) -> gs.u32["h", "w"]:
    py, px = gs.index()
    h = hr.shape[0]
    w = hr.shape[1]
    hh = blr.shape[0]
    hw = blr.shape[1]

    fy = (float(py) + 0.5) * 0.5 - 0.5
    fx = (float(px) + 0.5) * 0.5 - 0.5
    y0 = gs.max(int(gs.floor(fy)), 0)
    x0 = gs.max(int(gs.floor(fx)), 0)
    y1 = gs.min(y0 + 1, hh - 1)
    x1 = gs.min(x0 + 1, hw - 1)
    ty = gs.min(gs.max(fy - float(y0), 0.0), 1.0)
    tx = gs.min(gs.max(fx - float(x0), 0.0), 1.0)
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    br = w00 * blr[y0, x0] + w10 * blr[y0, x1] + w01 * blr[y1, x0] + w11 * blr[y1, x1]
    bg = w00 * blg[y0, x0] + w10 * blg[y0, x1] + w01 * blg[y1, x0] + w11 * blg[y1, x1]
    bb = w00 * blb[y0, x0] + w10 * blb[y0, x1] + w01 * blb[y1, x0] + w11 * blb[y1, x1]

    r = (hr[py, px] + br * bloom_strength) * exposure
    g = (hg[py, px] + bg * bloom_strength) * exposure
    b = (hb[py, px] + bb * bloom_strength) * exposure

    ux = (float(px) + 0.5) / float(w) - 0.5
    uy = (float(py) + 0.5) / float(h) - 0.5
    vig = gs.max(1.0 - (ux * ux + uy * uy) * 0.75, 0.45)
    r = r * vig
    g = g * vig
    b = b * vig

    r = gs.min(gs.max((r * (2.51 * r + 0.03)) / (r * (2.43 * r + 0.59) + 0.14), 0.0), 1.0)
    g = gs.min(gs.max((g * (2.51 * g + 0.03)) / (g * (2.43 * g + 0.59) + 0.14), 0.0), 1.0)
    b = gs.min(gs.max((b * (2.51 * b + 0.03)) / (b * (2.43 * b + 0.59) + 0.14), 0.0), 1.0)

    sr = gs.sqrt(r)
    sg = gs.sqrt(g)
    sb = gs.sqrt(b)
    ri = int(gs.min(sr * (1.138 - 0.138 * sr), 1.0) * 255.0 + 0.5)
    gi = int(gs.min(sg * (1.138 - 0.138 * sg), 1.0) * 255.0 + 0.5)
    bi = int(gs.min(sb * (1.138 - 0.138 * sb), 1.0) * 255.0 + 0.5)

    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def tonemap_pack_nobloom(
    hr: gs.f32["h", "w"],
    hg: gs.f32["h", "w"],
    hb: gs.f32["h", "w"],
    exposure: gs.f32,
) -> gs.u32["h", "w"]:
    py, px = gs.index()
    h = hr.shape[0]
    w = hr.shape[1]

    r = hr[py, px] * exposure
    g = hg[py, px] * exposure
    b = hb[py, px] * exposure

    ux = (float(px) + 0.5) / float(w) - 0.5
    uy = (float(py) + 0.5) / float(h) - 0.5
    vig = gs.max(1.0 - (ux * ux + uy * uy) * 0.75, 0.45)
    r = r * vig
    g = g * vig
    b = b * vig

    r = gs.min(gs.max((r * (2.51 * r + 0.03)) / (r * (2.43 * r + 0.59) + 0.14), 0.0), 1.0)
    g = gs.min(gs.max((g * (2.51 * g + 0.03)) / (g * (2.43 * g + 0.59) + 0.14), 0.0), 1.0)
    b = gs.min(gs.max((b * (2.51 * b + 0.03)) / (b * (2.43 * b + 0.59) + 0.14), 0.0), 1.0)

    sr = gs.sqrt(r)
    sg = gs.sqrt(g)
    sb = gs.sqrt(b)
    ri = int(gs.min(sr * (1.138 - 0.138 * sr), 1.0) * 255.0 + 0.5)
    gi = int(gs.min(sg * (1.138 - 0.138 * sg), 1.0) * 255.0 + 0.5)
    bi = int(gs.min(sb * (1.138 - 0.138 * sb), 1.0) * 255.0 + 0.5)

    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def fxaa_pass(src: gs.u32["h", "w"]) -> gs.u32["h", "w"]:
    py, px = gs.index()
    h = src.shape[0]
    w = src.shape[1]
    ym = gs.max(py - 1, 0)
    yp = gs.min(py + 1, h - 1)
    xm = gs.max(px - 1, 0)
    xp = gs.min(px + 1, w - 1)

    c_m = src[py, px]
    c_n = src[ym, px]
    c_s = src[yp, px]
    c_w = src[py, xm]
    c_e = src[py, xp]

    mask = gs.u32(255)
    mr = float(c_m & mask)
    mg = float((c_m >> gs.u32(8)) & mask)
    mb = float((c_m >> gs.u32(16)) & mask)

    nr = float(c_n & mask)
    ng = float((c_n >> gs.u32(8)) & mask)
    nb = float((c_n >> gs.u32(16)) & mask)

    sr = float(c_s & mask)
    sg = float((c_s >> gs.u32(8)) & mask)
    sb = float((c_s >> gs.u32(16)) & mask)

    wr = float(c_w & mask)
    wg = float((c_w >> gs.u32(8)) & mask)
    wb = float((c_w >> gs.u32(16)) & mask)

    er = float(c_e & mask)
    eg = float((c_e >> gs.u32(8)) & mask)
    eb = float((c_e >> gs.u32(16)) & mask)

    lm = 0.299 * mr + 0.587 * mg + 0.114 * mb
    ln = 0.299 * nr + 0.587 * ng + 0.114 * nb
    ls = 0.299 * sr + 0.587 * sg + 0.114 * sb
    lw = 0.299 * wr + 0.587 * wg + 0.114 * wb
    le = 0.299 * er + 0.587 * eg + 0.114 * eb

    l_min = gs.min(lm, ln, ls, lw, le)
    l_max = gs.max(lm, ln, ls, lw, le)
    contrast = l_max - l_min

    or_ = mr
    og_ = mg
    ob_ = mb
    if contrast >= gs.max(12.0, l_max * 0.125):
        edge_h = gs.abs(ln + ls - 2.0 * lm)
        edge_v = gs.abs(lw + le - 2.0 * lm)
        blend = gs.min(gs.abs(0.25 * (ln + ls + lw + le) - lm) / contrast, 1.0)
        blend = blend * blend * 0.55
        if edge_h >= edge_v:
            if gs.abs(ln - lm) >= gs.abs(ls - lm):
                or_ = mr * (1.0 - blend) + nr * blend
                og_ = mg * (1.0 - blend) + ng * blend
                ob_ = mb * (1.0 - blend) + nb * blend
            else:
                or_ = mr * (1.0 - blend) + sr * blend
                og_ = mg * (1.0 - blend) + sg * blend
                ob_ = mb * (1.0 - blend) + sb * blend
        else:
            if gs.abs(lw - lm) >= gs.abs(le - lm):
                or_ = mr * (1.0 - blend) + wr * blend
                og_ = mg * (1.0 - blend) + wg * blend
                ob_ = mb * (1.0 - blend) + wb * blend
            else:
                or_ = mr * (1.0 - blend) + er * blend
                og_ = mg * (1.0 - blend) + eg * blend
                ob_ = mb * (1.0 - blend) + eb * blend

    ri = int(gs.min(gs.max(or_, 0.0), 255.0) + 0.5)
    gi = int(gs.min(gs.max(og_, 0.0), 255.0) + 0.5)
    bi = int(gs.min(gs.max(ob_, 0.0), 255.0) + 0.5)
    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel
def blit_viewport(
    base: gs.u32["h", "w"],
    sub: gs.u32["sh", "sw"],
    ox: gs.i32,
    oy: gs.i32,
    dw: gs.i32,
    dh: gs.i32,
    border_rgba: gs.u32,
) -> gs.u32["h", "w"]:
    """Incruste une vue caméra (`sub`) dans le rectangle `(ox, oy, dw, dh)` du framebuffer principal."""
    py, px = gs.index()
    lx = px - ox
    ly = py - oy
    if lx < 0 or lx >= dw or ly < 0 or ly >= dh:
        return base[py, px]
    if border_rgba != gs.u32(0):
        if lx == 0 or lx == dw - 1 or ly == 0 or ly == dh - 1:
            return border_rgba
    sw = sub.shape[1]
    sh = sub.shape[0]
    sx = gs.min((lx * sw) // gs.max(dw, 1), sw - 1)
    sy = gs.min((ly * sh) // gs.max(dh, 1), sh - 1)
    return sub[sy, sx]


@gs.kernel
def upscale_sharp(ref: gs.u32["h", "w"], sub: gs.u32["sh", "sw"]) -> gs.u32["h", "w"]:
    """Agrandit une image RGBA8 à la taille de `ref` : Catmull-Rom, borné par les quatre voisins (pas de halo)."""
    py, px = gs.index()
    sw = sub.shape[1]
    sh = sub.shape[0]
    fx = (float(px) + 0.5) * float(sw) / float(ref.shape[1]) - 0.5
    fy = (float(py) + 0.5) * float(sh) / float(ref.shape[0]) - 0.5
    x0 = int(gs.floor(fx))
    y0 = int(gs.floor(fy))
    tx = fx - float(x0)
    ty = fy - float(y0)
    r = 0.0
    g = 0.0
    b = 0.0
    lo_r = 255.0
    lo_g = 255.0
    lo_b = 255.0
    hi_r = 0.0
    hi_g = 0.0
    hi_b = 0.0
    for j in range(-1, 3):
        dy = gs.abs(float(j) - ty)
        wy = 1.0 + dy * dy * (1.5 * dy - 2.5)
        if dy > 1.0:
            wy = 2.0 + dy * (-4.0 + dy * (2.5 - 0.5 * dy))
        yy = gs.min(gs.max(y0 + j, 0), sh - 1)
        for i in range(-1, 3):
            dx = gs.abs(float(i) - tx)
            wx = 1.0 + dx * dx * (1.5 * dx - 2.5)
            if dx > 1.0:
                wx = 2.0 + dx * (-4.0 + dx * (2.5 - 0.5 * dx))
            col = sub[yy, gs.min(gs.max(x0 + i, 0), sw - 1)]
            cr = float(col & gs.u32(255))
            cg = float((col >> gs.u32(8)) & gs.u32(255))
            cb = float((col >> gs.u32(16)) & gs.u32(255))
            r += cr * wx * wy
            g += cg * wx * wy
            b += cb * wx * wy
            if i >= 0 and i < 2 and j >= 0 and j < 2:
                lo_r = gs.min(lo_r, cr)
                lo_g = gs.min(lo_g, cg)
                lo_b = gs.min(lo_b, cb)
                hi_r = gs.max(hi_r, cr)
                hi_g = gs.max(hi_g, cg)
                hi_b = gs.max(hi_b, cb)
    ri = gs.u32(gs.min(gs.max(r, lo_r), hi_r) + 0.5)
    gi = gs.u32(gs.min(gs.max(g, lo_g), hi_g) + 0.5)
    bi = gs.u32(gs.min(gs.max(b, lo_b), hi_b) + 0.5)
    return ri | (gi << gs.u32(8)) | (bi << gs.u32(16)) | gs.u32(4278190080) | (ref[py, px] & gs.u32(0))


@gs.kernel
def ssr_pass(
    hr: gs.f32["h", "w"],
    hg: gs.f32["h", "w"],
    hb: gs.f32["h", "w"],
    visbuf: gs.u32["h", "w"],
    zbuf: gs.u32["h", "w"],
    tri_idx: gs.i32["nt", "c3"],
    tri_mat: gs.i32["nt"],
    sx: gs.f32["nv"],
    sy: gs.f32["nv"],
    inv_w: gs.f32["nv"],
    wpos: gs.f32["nv", "c3"],
    wnorm: gs.f32["nv", "c3"],
    materials: gs.f32["nm", "c16"],
    vp: gs.f32["c16"],
    cam_x: gs.f32,
    cam_y: gs.f32,
    cam_z: gs.f32,
    near: gs.f32,
    far: gs.f32,
    strength: gs.f32,
    max_steps: gs.i32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Reflets Screen-Space (SSR) par raymarching 3D dans la profondeur du Visibility Buffer."""
    py, px = gs.index()
    h = visbuf.shape[0]
    w = visbuf.shape[1]
    base_r = hr[py, px]
    base_g = hg[py, px]
    base_b = hb[py, px]
    vis = visbuf[py, px]
    if vis == gs.u32(4294967295):
        return base_r, base_g, base_b

    t = gs.i32(vis)
    m = tri_mat[t]
    roughness = materials[m, 4]
    pat_ssr = int(materials[m, 5])
    if roughness > 0.45 or pat_ssr == 5 or pat_ssr == 6:
        return base_r, base_g, base_b

    metallic = materials[m, 3]
    i0 = tri_idx[t, 0]
    i1 = tri_idx[t, 1]
    i2 = tri_idx[t, 2]

    x0 = sx[i0]
    y0 = sy[i0]
    iw0 = inv_w[i0]
    x1 = sx[i1]
    y1 = sy[i1]
    iw1 = inv_w[i1]
    x2 = sx[i2]
    y2 = sy[i2]
    iw2 = inv_w[i2]

    fx = float(px) + 0.5
    fy = float(py) + 0.5
    area = (y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0)
    inv_area = 1.0 / gs.max(gs.abs(area), 1e-6)
    sw0 = ((y2 - y1) * (fx - x1) - (x2 - x1) * (fy - y1)) * inv_area
    sw1 = ((y0 - y2) * (fx - x2) - (x0 - x2) * (fy - y2)) * inv_area
    sw2 = 1.0 - sw0 - sw1
    pw0 = sw0 * iw0
    pw1 = sw1 * iw1
    pw2 = sw2 * iw2
    inv_pw = 1.0 / gs.max(pw0 + pw1 + pw2, 1e-8)
    b0 = pw0 * inv_pw
    b1 = pw1 * inv_pw
    b2 = pw2 * inv_pw
    if gs.min(iw0, iw1, iw2) <= 0.0:
        # Triangle qui traverse le plan proche : barycentriques homogènes (valables derrière la caméra)
        fw = float(w)
        fh = float(h)
        hw0 = iw0 + near + 1.0
        hx0 = (x0 * 0.5 + 0.5 * hw0) * fw
        hy0 = (0.5 * hw0 - y0 * 0.5) * fh
        if iw0 > 0.0:
            hw0 = 1.0 / iw0
            hx0 = x0 * hw0
            hy0 = y0 * hw0
        hw1 = iw1 + near + 1.0
        hx1 = (x1 * 0.5 + 0.5 * hw1) * fw
        hy1 = (0.5 * hw1 - y1 * 0.5) * fh
        if iw1 > 0.0:
            hw1 = 1.0 / iw1
            hx1 = x1 * hw1
            hy1 = y1 * hw1
        hw2 = iw2 + near + 1.0
        hx2 = (x2 * 0.5 + 0.5 * hw2) * fw
        hy2 = (0.5 * hw2 - y2 * 0.5) * fh
        if iw2 > 0.0:
            hw2 = 1.0 / iw2
            hx2 = x2 * hw2
            hy2 = y2 * hw2
        gx0 = hx0 - fx * hw0
        gy0 = hy0 - fy * hw0
        gx1 = hx1 - fx * hw1
        gy1 = hy1 - fy * hw1
        gx2 = hx2 - fx * hw2
        gy2 = hy2 - fy * hw2
        e0 = gx1 * gy2 - gx2 * gy1
        e1 = gx2 * gy0 - gx0 * gy2
        e2 = gx0 * gy1 - gx1 * gy0
        es = e0 + e1 + e2
        inv_e = 1.0 / (es if gs.abs(es) > 1e-12 else 1e-12)
        b0 = e0 * inv_e
        b1 = e1 * inv_e
        b2 = e2 * inv_e

    pos_x = b0 * wpos[i0, 0] + b1 * wpos[i1, 0] + b2 * wpos[i2, 0]
    pos_y = b0 * wpos[i0, 1] + b1 * wpos[i1, 1] + b2 * wpos[i2, 1]
    pos_z = b0 * wpos[i0, 2] + b1 * wpos[i1, 2] + b2 * wpos[i2, 2]

    nx = b0 * wnorm[i0, 0] + b1 * wnorm[i1, 0] + b2 * wnorm[i2, 0]
    ny = b0 * wnorm[i0, 1] + b1 * wnorm[i1, 1] + b2 * wnorm[i2, 1]
    nz = b0 * wnorm[i0, 2] + b1 * wnorm[i1, 2] + b2 * wnorm[i2, 2]
    inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx = nx * inv_n
    ny = ny * inv_n
    nz = nz * inv_n

    vx = cam_x - pos_x
    vy = cam_y - pos_y
    vz = cam_z - pos_z
    inv_v = 1.0 / gs.max(gs.sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    vx = vx * inv_v
    vy = vy * inv_v
    vz = vz * inv_v
    ndotv = gs.max(nx * vx + ny * vy + nz * vz, 0.001)

    rx = 2.0 * ndotv * nx - vx
    ry = 2.0 * ndotv * ny - vy
    rz = 2.0 * ndotv * nz - vz

    inv_dr = 1.0 / gs.max(far - near, 1e-4)
    step_len = 0.26
    hit_r = 0.0
    hit_g = 0.0
    hit_b = 0.0
    hit_w = 0.0
    hit_dist = 0.0
    hit_st = 0

    for st in range(32):
        if st < max_steps and hit_w <= 0.0:
            dist = (float(st) + 1.0) * step_len
            qx = pos_x + nx * 0.04 + rx * dist
            qy = pos_y + ny * 0.04 + ry * dist
            qz = pos_z + nz * 0.04 + rz * dist
            cx = vp[0] * qx + vp[1] * qy + vp[2] * qz + vp[3]
            cy = vp[4] * qx + vp[5] * qy + vp[6] * qz + vp[7]
            cw = vp[12] * qx + vp[13] * qy + vp[14] * qz + vp[15]
            if cw > near:
                iw = 1.0 / cw
                ux = cx * iw * 0.5 + 0.5
                uy = 0.5 - cy * iw * 0.5
                if ux >= 0.01 and ux <= 0.99 and uy >= 0.01 and uy <= 0.99:
                    spx = int(ux * float(w))
                    spy = int(uy * float(h))
                    sz = zbuf[spy, spx]
                    if sz != gs.u32(4294967295):
                        sdepth = float(sz) / 16777215.0
                        rdepth = (cw - near) * inv_dr
                        dz = rdepth - sdepth
                        if dz > 0.0008 and dz < 0.045:
                            edge = gs.min(gs.min(ux, 1.0 - ux), gs.min(uy, 1.0 - uy)) * 10.0
                            fade = gs.min(edge, 1.0) * (1.0 - float(st) / float(gs.max(max_steps, 1)))
                            hit_r = hr[spy, spx]
                            hit_g = hg[spy, spx]
                            hit_b = hb[spy, spx]
                            hit_w = fade
                            hit_dist = dist
                            hit_st = st

    if hit_w > 0.0:
        # Raffinement dichotomique (3 étapes) pour éliminer tout effet d'escalier sur les surfaces courbes
        lo = gs.max(hit_dist - step_len, 0.05)
        hi = hit_dist
        for _ in range(3):
            mid = 0.5 * (lo + hi)
            qx = pos_x + nx * 0.04 + rx * mid
            qy = pos_y + ny * 0.04 + ry * mid
            qz = pos_z + nz * 0.04 + rz * mid
            cx = vp[0] * qx + vp[1] * qy + vp[2] * qz + vp[3]
            cy = vp[4] * qx + vp[5] * qy + vp[6] * qz + vp[7]
            cw = vp[12] * qx + vp[13] * qy + vp[14] * qz + vp[15]
            behind = 0
            if cw > near:
                iw = 1.0 / cw
                ux = cx * iw * 0.5 + 0.5
                uy = 0.5 - cy * iw * 0.5
                if ux >= 0.01 and ux <= 0.99 and uy >= 0.01 and uy <= 0.99:
                    spx = int(ux * float(w))
                    spy = int(uy * float(h))
                    sz = zbuf[spy, spx]
                    if sz != gs.u32(4294967295):
                        sdepth = float(sz) / 16777215.0
                        rdepth = (cw - near) * inv_dr
                        dz = rdepth - sdepth
                        if dz > 0.0005 and dz < 0.045:
                            behind = 1
                            hit_r = hr[spy, spx]
                            hit_g = hg[spy, spx]
                            hit_b = hb[spy, spx]
                            edge = gs.min(gs.min(ux, 1.0 - ux), gs.min(uy, 1.0 - uy)) * 10.0
                            hit_w = gs.min(edge, 1.0) * (1.0 - float(hit_st) / float(gs.max(max_steps, 1)))
            if behind != 0:
                hi = mid
            else:
                lo = mid

        omv = 1.0 - ndotv
        fresnel = 0.12 + 0.88 * omv * omv * omv * omv
        gloss = 1.0 - roughness / 0.45
        w_refl = gs.min(hit_w * gloss * (0.35 + 0.65 * metallic + 0.35 * fresnel) * strength, 0.85)
        base_r = base_r * (1.0 - w_refl) + hit_r * w_refl
        base_g = base_g * (1.0 - w_refl) + hit_g * w_refl
        base_b = base_b * (1.0 - w_refl) + hit_b * w_refl

    return base_r, base_g, base_b
