"""Optique d'une surface d'eau : Fresnel, réfraction vers le bassin, absorption, caustiques et reflets.

La surface est un maillage quelconque, tracé à part de la scène opaque. Le bassin (fond et murs) est une boîte
alignée sur les axes ; les objets immergés sont cherchés dans l'image opaque déjà rendue."""

import gpusync as gs


@gs.kernel
def water_depth_map(
    h_grid: gs.f32["gh", "gw"],
    half: gs.f32,
    h_scale: gs.f32,
    depth: gs.f32,
    amb_amp: gs.f32,
    amb_freq: gs.f32,
    amb_t: gs.f32,
) -> gs.f32["gh", "gw"]:
    """Épaisseur d'eau (m) au-dessus du fond, houle comprise : entrée du calcul des caustiques."""
    y, x = gs.index()
    h = h_grid[y, x] * h_scale
    if amb_amp > 0.0:
        wx = (float(x) / float(h_grid.shape[1] - 1) * 2.0 - 1.0) * half
        wz = (float(y) / float(h_grid.shape[0] - 1) * 2.0 - 1.0) * half
        p1 = wx * amb_freq + wz * (amb_freq * 0.55) + amb_t
        p2 = wx * (-amb_freq * 0.72) + wz * (amb_freq * 1.15) - amb_t * 0.82
        h += amb_amp * (0.62 * gs.sin(p1) + 0.38 * gs.cos(p2))
    return gs.max(depth + h, 0.05)


@gs.kernel
def water_merge(
    zbuf: gs.u32["h", "w"],
    visbuf: gs.u32["h", "w"],
    zbuf_w: gs.u32["h", "w"],
    vis_w: gs.u32["h", "w"],
    first_tri: gs.i32,
) -> (gs.u32["h", "w"], gs.u32["h", "w"]):
    """Ajoute l'eau à la profondeur et aux numéros de triangles (sélection à la souris, fluides suivants)."""
    y, x = gs.index()
    if vis_w[y, x] != gs.u32(4294967295) and zbuf_w[y, x] < zbuf[y, x]:
        return zbuf_w[y, x], vis_w[y, x] + gs.u32(first_tri)
    return zbuf[y, x], visbuf[y, x]


@gs.kernel
def water_optics(
    hdr_r: gs.f32["h", "w"],
    hdr_g: gs.f32["h", "w"],
    hdr_b: gs.f32["h", "w"],
    zbuf: gs.u32["h", "w"],
    zbuf_w: gs.u32["h", "w"],
    vis_w: gs.u32["h", "w"],
    tri_idx: gs.i32["nt", "c3"],
    sx: gs.f32["nv"],
    sy: gs.f32["nv"],
    inv_w: gs.f32["nv"],
    wnorm: gs.f32["nv", "c3"],
    caustic_map: gs.f32["mz", "mx"],
    shadow_map: gs.u32["sh", "sw"],
    light_vp: gs.f32["c16"],
    tex_pool: gs.u32["tp"],
    tex_meta: gs.i32["ntex", "c4"],
    dyn_atlas: gs.u32["dh", "dw"],
    sky_tid: gs.i32,
    sky_int: gs.f32,
    use_shadows: gs.i32,
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
    near: gs.f32,
    far: gs.f32,
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
    pool_x: gs.f32,
    pool_z: gs.f32,
    floor_y: gs.f32,
    half: gs.f32,
    walls: gs.i32,
    ior: gs.f32,
    abs_r: gs.f32,
    abs_g: gs.f32,
    abs_b: gs.f32,
    tint_r: gs.f32,
    tint_g: gs.f32,
    tint_b: gs.f32,
    c1_r: gs.f32,
    c1_g: gs.f32,
    c1_b: gs.f32,
    c2_r: gs.f32,
    c2_g: gs.f32,
    c2_b: gs.f32,
    tile: gs.f32,
    roughness: gs.f32,
    caustic_strength: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Couleur de l'eau pour chaque pixel où elle est visible ; l'image opaque ailleurs."""
    py, px = gs.index()
    h = zbuf.shape[0]
    w = zbuf.shape[1]
    vis = vis_w[py, px]
    zw = zbuf_w[py, px]
    if vis == gs.u32(4294967295) or zw >= zbuf[py, px]:
        return hdr_r[py, px], hdr_g[py, px], hdr_b[py, px]

    # 1) Rayon caméra et point de la surface
    aspect = float(w) / float(h)
    ndc_x = ((float(px) + 0.5) / float(w) * 2.0 - 1.0) * aspect * tan_hfov
    ndc_y = (1.0 - (float(py) + 0.5) / float(h) * 2.0) * tan_hfov
    vdx = fwd_x + right_x * ndc_x + up_x * ndc_y
    vdy = fwd_y + right_y * ndc_x + up_y * ndc_y
    vdz = fwd_z + right_z * ndc_x + up_z * ndc_y
    depth_range = gs.max(far - near, 1e-4)
    surf_z = near + float(zw) / 16777215.0 * depth_range
    hx = cam_x + vdx * surf_z
    hy = cam_y + vdy * surf_z
    hz = cam_z + vdz * surf_z
    inv_len = 1.0 / gs.sqrt(vdx * vdx + vdy * vdy + vdz * vdz)
    rdx = vdx * inv_len
    rdy = vdy * inv_len
    rdz = vdz * inv_len
    dist = surf_z / inv_len

    # 2) Normale interpolée (barycentriques homogènes : valables aussi quand le triangle traverse le plan proche)
    t = gs.i32(vis)
    nx = 0.0
    ny = 0.0
    nz = 0.0
    fw = float(w)
    fh = float(h)
    qx0 = 0.0
    qy0 = 0.0
    qx1 = 0.0
    qy1 = 0.0
    qx2 = 0.0
    qy2 = 0.0
    for k in range(3):
        i_v = tri_idx[t, k]
        iw_v = inv_w[i_v]
        hw = iw_v + near + 1.0
        ex = (sx[i_v] * 0.5 + 0.5 * hw) * fw
        ey = (0.5 * hw - sy[i_v] * 0.5) * fh
        if iw_v > 0.0:
            hw = 1.0 / iw_v
            ex = sx[i_v] * hw
            ey = sy[i_v] * hw
        ex -= (float(px) + 0.5) * hw
        ey -= (float(py) + 0.5) * hw
        if k == 0:
            qx0 = ex
            qy0 = ey
        elif k == 1:
            qx1 = ex
            qy1 = ey
        else:
            qx2 = ex
            qy2 = ey
    e0 = qx1 * qy2 - qx2 * qy1
    e1 = qx2 * qy0 - qx0 * qy2
    e2 = qx0 * qy1 - qx1 * qy0
    for k in range(3):
        i_n = tri_idx[t, k]
        b_k = e0
        if k == 1:
            b_k = e1
        elif k == 2:
            b_k = e2
        nx += b_k * wnorm[i_n, 0]
        ny += b_k * wnorm[i_n, 1]
        nz += b_k * wnorm[i_n, 2]
    sum_e = e0 + e1 + e2
    if sum_e < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz
    inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx = nx * inv_n
    ny = ny * inv_n
    nz = nz * inv_n
    cos_i = -(rdx * nx + rdy * ny + rdz * nz)
    cos_i = gs.max(cos_i, 0.02)

    # 3) Fresnel exact d'un diélectrique et direction réfractée
    eta = 1.0 / gs.max(ior, 1.001)
    k_s = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    fres = 1.0
    tx = rdx
    ty = rdy
    tz = rdz
    if k_s > 0.0:
        cos_t = gs.sqrt(k_s)
        r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
        r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
        fres = 0.5 * (r_s * r_s + r_p * r_p)
        bend = eta * cos_i - cos_t
        tx = eta * rdx + bend * nx
        ty = eta * rdy + bend * ny
        tz = eta * rdz + bend * nz

    # 4) Ombre portée sur la surface (soleil)
    shadow = 1.0
    if use_shadows != 0:
        ls_x = light_vp[0] * hx + light_vp[1] * hy + light_vp[2] * hz + light_vp[3]
        ls_y = light_vp[4] * hx + light_vp[5] * hy + light_vp[6] * hz + light_vp[7]
        ls_z = light_vp[8] * hx + light_vp[9] * hy + light_vp[10] * hz + light_vp[11]
        su = int((ls_x * 0.5 + 0.5) * float(shadow_map.shape[1]))
        sv = int((0.5 - ls_y * 0.5) * float(shadow_map.shape[0]))
        if su >= 1 and su < shadow_map.shape[1] - 1 and sv >= 1 and sv < shadow_map.shape[0] - 1 and ls_z > 0.0 and ls_z < 1.0:
            if gs.u32(gs.max(ls_z - 0.0022, 0.0) * 4294967040.0) > shadow_map[sv, su]:
                shadow = 0.0

    mask_u = gs.u32(255)
    refr_r = 0.0
    refr_g = 0.0
    refr_b = 0.0
    # 5) Bassin : sortie de la boîte (fond ou mur)
    inv_tx = 1.0 / (tx if gs.abs(tx) > 1e-5 else 1e-5)
    inv_tz = 1.0 / (tz if gs.abs(tz) > 1e-5 else 1e-5)
    to_x = gs.max((pool_x - half - hx) * inv_tx, (pool_x + half - hx) * inv_tx)
    to_z = gs.max((pool_z - half - hz) * inv_tz, (pool_z + half - hz) * inv_tz)
    to_f = (floor_y - hy) / gs.min(ty, -1e-3)
    path = to_f
    if walls != 0:
        path = gs.max(gs.min(to_f, to_x, to_z), 0.0)
    path = gs.min(path, 60.0)

    # 6) Objets immergés : le rayon réfracté est suivi dans l'image opaque
    hit = 0
    obj_r = 0.0
    obj_g = 0.0
    obj_b = 0.0
    s_prev = 0.0
    s_lo = 0.0
    s_hi = 0.0
    n_bis = 0
    for k in range(38):
        along = path * (float(k) + 0.7) / 30.0
        if hit != 0:
            along = 0.5 * (s_lo + s_hi)
        elif k >= 30:
            break
        ax = hx + tx * along - cam_x
        ay = hy + ty * along - cam_y
        az = hz + tz * along - cam_z
        vz_q = ax * fwd_x + ay * fwd_y + az * fwd_z
        if vz_q < near:
            break
        nqx = (ax * right_x + ay * right_y + az * right_z) / vz_q
        nqy = (ax * up_x + ay * up_y + az * up_z) / vz_q
        sxq = int(gs.floor((nqx / (aspect * tan_hfov) * 0.5 + 0.5) * fw))
        syq = int(gs.floor((0.5 - nqy / tan_hfov * 0.5) * fh))
        if sxq < 0 or sxq >= w or syq < 0 or syq >= h:
            break
        behind = 0
        zq = zbuf[syq, sxq]
        if zq != gs.u32(4294967295):
            z_s = near + float(zq) / 16777215.0 * depth_range
            o_x = cam_x + (fwd_x + right_x * nqx + up_x * nqy) * z_s
            o_y = cam_y + (fwd_y + right_y * nqx + up_y * nqy) * z_s
            o_z = cam_z + (fwd_z + right_z * nqx + up_z * nqy) * z_s
            # Un objet dans l'eau, strictement à l'intérieur du bassin (ni margelle, ni sol extérieur)
            if vz_q >= z_s and vz_q - z_s < 0.5 and o_y < hy + 0.05 and o_y > floor_y - 0.05:
                if gs.abs(o_x - pool_x) < half - 0.08 and gs.abs(o_z - pool_z) < half - 0.08:
                    behind = 1
        if hit == 0:
            if behind != 0:
                hit = 1
                s_lo = s_prev
                s_hi = along
                obj_r = hdr_r[syq, sxq]
                obj_g = hdr_g[syq, sxq]
                obj_b = hdr_b[syq, sxq]
            s_prev = along
        else:
            if behind != 0:
                s_hi = along
                obj_r = hdr_r[syq, sxq]
                obj_g = hdr_g[syq, sxq]
                obj_b = hdr_b[syq, sxq]
            else:
                s_lo = along
            n_bis += 1
            if n_bis == 5:
                break

    if hit != 0:
        path = s_hi
        refr_r = obj_r
        refr_g = obj_g
        refr_b = obj_b
    else:
        # 7) Fond ou mur carrelé, éclairé par le soleil à travers la surface
        bx = hx + tx * path
        by = hy + ty * path
        bz = hz + tz * path
        u_t = bx
        v_t = bz
        wall = 0
        if walls != 0 and to_f > path + 1e-4:
            wall = 1
            v_t = by
            if to_x < to_z:
                u_t = bz
        cell = (int(gs.floor(u_t / tile)) + int(gs.floor(v_t / tile))) & 1
        alb_r = c1_r if cell == 0 else c2_r
        alb_g = c1_g if cell == 0 else c2_g
        alb_b = c1_b if cell == 0 else c2_b
        caust = 1.0
        if caustic_strength > 0.0:
            mqx = gs.min(gs.max(((bx - pool_x) / (2.0 * half) + 0.5) * float(caustic_map.shape[1]) - 0.5, 0.0), float(caustic_map.shape[1]) - 1.001)
            mqz = gs.min(gs.max(((bz - pool_z) / (2.0 * half) + 0.5) * float(caustic_map.shape[0]) - 0.5, 0.0), float(caustic_map.shape[0]) - 1.001)
            mi = int(mqx)
            mj = int(mqz)
            fu = mqx - float(mi)
            fv = mqz - float(mj)
            c_lo = caustic_map[mj, mi] * (1.0 - fu) + caustic_map[mj, mi + 1] * fu
            c_hi = caustic_map[mj + 1, mi] * (1.0 - fu) + caustic_map[mj + 1, mi + 1] * fu
            c_val = gs.min(c_lo * (1.0 - fv) + c_hi * fv, 3.5)
            caust = 1.0 + caustic_strength * (c_val - 1.0)
        # Le soleil parcourt aussi l'eau avant d'atteindre ce point
        sin2 = (1.0 - sun_y * sun_y) / (ior * ior)
        cos_s = gs.sqrt(gs.max(1.0 - sin2, 0.05))
        sun_path = gs.max(hy - by, 0.0) / cos_s
        # Ombres au fond : objet flottant au point d'entrée du rayon de soleil, objet immergé sur le trajet
        lit_sh = 1.0
        if use_shadows != 0:
            row2 = gs.sqrt(light_vp[8] * light_vp[8] + light_vp[9] * light_vp[9] + light_vp[10] * light_vp[10])
            for k in range(2):
                qx = bx
                qy = by + 0.03
                qz = bz
                if k == 0:
                    qx = bx + sun_x / ior * sun_path
                    qy = hy
                    qz = bz + sun_z / ior * sun_path
                lq_x = light_vp[0] * qx + light_vp[1] * qy + light_vp[2] * qz + light_vp[3]
                lq_y = light_vp[4] * qx + light_vp[5] * qy + light_vp[6] * qz + light_vp[7]
                lq_z = light_vp[8] * qx + light_vp[9] * qy + light_vp[10] * qz + light_vp[11]
                qu = int((lq_x * 0.5 + 0.5) * float(shadow_map.shape[1]))
                qv = int((0.5 - lq_y * 0.5) * float(shadow_map.shape[0]))
                if qu >= 1 and qu < shadow_map.shape[1] - 1 and qv >= 1 and qv < shadow_map.shape[0] - 1 and lq_z > 0.0 and lq_z < 1.0:
                    blocked = 0.0
                    for j in range(9):
                        z_o = float(shadow_map[qv + j // 3 - 1, qu + j % 3 - 1]) / 4294967040.0
                        if z_o < lq_z - 0.0022:
                            # Hauteur de ce qui fait de l'ombre, le long du soleil
                            y_o = qy + (lq_z - z_o) / gs.max(row2, 1e-6) * sun_y
                            if k == 0 or y_o < hy:
                                blocked += 0.11111111
                    lit_sh = gs.min(lit_sh, 1.0 - blocked)
        face = cos_s
        if wall != 0:
            face = 0.35
        lit = 0.62 * sun_int * 0.48 * face * caust * lit_sh
        refr_r = alb_r * (0.38 + lit * sun_r * gs.exp(-sun_path * abs_r))
        refr_g = alb_g * (0.38 + lit * sun_g * gs.exp(-sun_path * abs_g))
        refr_b = alb_b * (0.38 + lit * sun_b * gs.exp(-sun_path * abs_b))

    # 8) Absorption de Beer-Lambert et lumière diffusée par l'eau
    tr_r = gs.exp(-path * abs_r)
    tr_g = gs.exp(-path * abs_g)
    tr_b = gs.exp(-path * abs_b)
    scat = 0.30 + 0.25 * gs.max(sun_y, 0.0) * (0.4 + 0.6 * shadow) * gs.min(sun_int, 3.0) * 0.5
    refr_r = refr_r * tr_r + tint_r * (1.0 - tr_r) * scat
    refr_g = refr_g * tr_g + tint_g * (1.0 - tr_g) * scat
    refr_b = refr_b * tr_b + tint_b * (1.0 - tr_b) * scat

    # 9) Reflets : image opaque si le rayon y touche quelque chose, ciel sinon
    rx = rdx + 2.0 * cos_i * nx
    ry = rdy + 2.0 * cos_i * ny
    rz = rdz + 2.0 * cos_i * nz
    if ry < 0.02:
        ry = 0.02
    found = 0
    refl_r = 0.0
    refl_g = 0.0
    refl_b = 0.0
    fade = 0.0
    r_prev = 0.02
    for k in range(28):
        # Pas qui s'allonge avec la distance
        r_s2 = 0.05 + 0.045 * float(k) + 0.011 * float(k) * float(k)
        ax = hx + rx * r_s2 - cam_x
        ay = hy + ry * r_s2 - cam_y
        az = hz + rz * r_s2 - cam_z
        vz_q = ax * fwd_x + ay * fwd_y + az * fwd_z
        if vz_q < near:
            break
        nqx = (ax * right_x + ay * right_y + az * right_z) / vz_q
        nqy = (ax * up_x + ay * up_y + az * up_z) / vz_q
        uf = nqx / (aspect * tan_hfov) * 0.5 + 0.5
        vf = 0.5 - nqy / tan_hfov * 0.5
        if uf < 0.0 or uf >= 1.0 or vf < 0.0 or vf >= 1.0:
            break
        sxq = int(uf * fw)
        syq = int(vf * fh)
        zq = zbuf[syq, sxq]
        if zq != gs.u32(4294967295):
            z_s = near + float(zq) / 16777215.0 * depth_range
            o_y = cam_y + (fwd_y + right_y * nqx + up_y * nqy) * z_s
            # Seulement ce qui dépasse de l'eau, à peu près à la profondeur du rayon
            if vz_q >= z_s and vz_q - z_s < 0.25 + (r_s2 - r_prev) * 1.5 and o_y > hy - 0.02:
                found = 1
                refl_r = hdr_r[syq, sxq]
                refl_g = hdr_g[syq, sxq]
                refl_b = hdr_b[syq, sxq]
                edge = gs.min(gs.min(uf, 1.0 - uf), gs.min(vf, 1.0 - vf)) * 12.0
                fade = gs.min(edge, 1.0)
                break
        r_prev = r_s2
    sky_r = 0.0
    sky_g = 0.0
    sky_b = 0.0
    if sky_tid >= 0:
        sky_u = gs.atan2(rz, rx) * 0.15915494 + 0.5
        sky_v = gs.atan2(gs.sqrt(rx * rx + rz * rz), ry) * 0.31830989
        tw = tex_meta[sky_tid, 2]
        m3 = tex_meta[sky_tid, 3]
        th = m3 & 1048575
        is_dyn = (m3 >> 20) & 1
        ftx = sky_u * float(tw) - 0.5
        fty = sky_v * float(th) - 0.5
        x0t = int(gs.floor(ftx))
        y0t = int(gs.floor(fty))
        ftx = ftx - float(x0t)
        fty = fty - float(y0t)
        for k in range(4):
            xt = (x0t + (k & 1) + tw) % tw
            yt = gs.min(gs.max(y0t + (k >> 1), 0), th - 1)
            col = gs.u32(0)
            if is_dyn != 0:
                col = dyn_atlas[tex_meta[sky_tid, 1] + yt, tex_meta[sky_tid, 0] + xt]
            else:
                col = tex_pool[tex_meta[sky_tid, 0] + yt * tw + xt]
            wk = (ftx if (k & 1) == 1 else 1.0 - ftx) * (fty if (k >> 1) == 1 else 1.0 - fty) * 0.0039215686
            sky_r += float(col & mask_u) * wk
            sky_g += float((col >> gs.u32(8)) & mask_u) * wk
            sky_b += float((col >> gs.u32(16)) & mask_u) * wk
        sky_r = sky_r * sky_r * sky_int
        sky_g = sky_g * sky_g * sky_int
        sky_b = sky_b * sky_b * sky_int
    else:
        t_sk = gs.min(gs.max(ry * 0.5 + 0.5, 0.0), 1.0)
        sky_r = 0.14 + 0.25 * (1.0 - t_sk)
        sky_g = 0.22 + 0.30 * (1.0 - t_sk)
        sky_b = 0.42 + 0.32 * (1.0 - t_sk)

    if found != 0:
        sky_r = sky_r * (1.0 - fade) + refl_r * fade
        sky_g = sky_g * (1.0 - fade) + refl_g * fade
        sky_b = sky_b * (1.0 - fade) + refl_b * fade
    # Éclat du soleil
    hsx = sun_x - rdx
    hsy = sun_y - rdy
    hsz = sun_z - rdz
    inv_hs = 1.0 / gs.max(gs.sqrt(hsx * hsx + hsy * hsy + hsz * hsz), 1e-6)
    ndoth = gs.max((nx * hsx + ny * hsy + nz * hsz) * inv_hs, 1e-4)
    gloss = gs.max(1.0 - roughness, 0.05)
    spec = gs.exp(gs.log(ndoth) * (60.0 + gloss * 340.0)) * (0.4 + gloss * 2.4) * sun_int * 0.5 * shadow
    out_r = refr_r * (1.0 - fres) + sky_r * fres + sun_r * spec
    out_g = refr_g * (1.0 - fres) + sky_g * fres + sun_g * spec
    out_b = refr_b * (1.0 - fres) + sky_b * fres + sun_b * spec

    if fog_density > 0.0:
        fog = gs.min(gs.max(1.0 - gs.exp(-dist * dist * fog_density), 0.0), 0.85)
        out_r = out_r * (1.0 - fog) + fog_r * fog
        out_g = out_g * (1.0 - fog) + fog_g * fog
        out_b = out_b * (1.0 - fog) + fog_b * fog
    return out_r, out_g, out_b


# ───────────────────────── Flotteurs : simulation entière sur GPU ─────────────────────────


@gs.kernel
def floating_step(
    state: gs.f32["m", "c16"],
    params: gs.f32["m", "c8"],
    kicks: gs.f32["m", "c8"],
    h_grid: gs.f32["gh", "gw"],
    size: gs.f32,
    h_scale: gs.f32,
    amb_amp: gs.f32,
    amb_freq: gs.f32,
    amb_t: gs.f32,
    dt: gs.f32,
) -> gs.f32["m", "c16"]:
    """Un pas des flotteurs : Archimède, pente des vagues, frottements, bords du bassin.

    `state` : `[px, py, pz, rx, ry, rz, échelle, visible, vx, vy, vz, immersion, 0, 0, 0, 0]`.
    `params` : `[rayon, flottabilité, gravité, poussée, frottement, frottement vertical, sillage, 0]`.
    `kicks` : `[ajout xyz, 0, facteur xyz, 0]`, appliqué aux vitesses avant le pas."""
    i, c = gs.index()
    if c == 6 or c == 7 or c > 11:
        return state[i, c]
    px = state[i, 0]
    py = state[i, 1]
    pz = state[i, 2]
    vx = state[i, 8] * kicks[i, 4] + kicks[i, 0]
    vy = state[i, 9] * kicks[i, 5] + kicks[i, 1]
    vz = state[i, 10] * kicks[i, 6] + kicks[i, 2]
    radius = params[i, 0]

    # Hauteur et pente de l'eau sous le flotteur
    n = h_grid.shape[1] - 1
    cell = size / float(n)
    gx = gs.min(gs.max((px / size + 0.5) * float(n), 1.0), float(n) - 1.001)
    gz = gs.min(gs.max((pz / size + 0.5) * float(n), 1.0), float(n) - 1.001)
    ix = int(gx)
    iz = int(gz)
    tx = gx - float(ix)
    tz = gz - float(iz)
    water_y = (
        (h_grid[iz, ix] * (1.0 - tx) + h_grid[iz, ix + 1] * tx) * (1.0 - tz)
        + (h_grid[iz + 1, ix] * (1.0 - tx) + h_grid[iz + 1, ix + 1] * tx) * tz
    ) * h_scale
    dhdx = (h_grid[iz, ix + 1] - h_grid[iz, ix - 1]) * 0.5 * h_scale / cell
    dhdz = (h_grid[iz + 1, ix] - h_grid[iz - 1, ix]) * 0.5 * h_scale / cell
    if amb_amp > 0.0:
        p1 = px * amb_freq + pz * (amb_freq * 0.55) + amb_t
        p2 = px * (-amb_freq * 0.72) + pz * (amb_freq * 1.15) - amb_t * 0.82
        water_y += amb_amp * (0.62 * gs.sin(p1) + 0.38 * gs.cos(p2))
        dhdx += amb_amp * (0.62 * amb_freq * gs.cos(p1) + 0.38 * (amb_freq * 0.72) * gs.sin(p2))
        dhdz += amb_amp * (0.62 * (amb_freq * 0.55) * gs.cos(p1) - 0.38 * (amb_freq * 1.15) * gs.sin(p2))

    sub = gs.min(gs.max((water_y - (py - radius)) / gs.max(2.0 * radius, 1e-4), 0.0), 1.0)
    drag = gs.pow(params[i, 4], dt * 60.0)
    v_drag = gs.pow(0.995, dt * 60.0)
    if sub > 0.0:
        v_drag = gs.pow(params[i, 5], dt * 60.0)
    vx = (vx - dhdx * params[i, 3] * sub * dt) * drag
    vy = (vy + (params[i, 1] * sub - params[i, 2]) * dt) * v_drag
    vz = (vz - dhdz * params[i, 3] * sub * dt) * drag
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    max_p = gs.max(0.5, size * 0.5 - 0.32 - radius)
    if px < -max_p:
        px = -max_p
        vx = gs.abs(vx) * 0.65
    elif px > max_p:
        px = max_p
        vx = -gs.abs(vx) * 0.65
    if pz < -max_p:
        pz = -max_p
        vz = gs.abs(vz) * 0.65
    elif pz > max_p:
        pz = max_p
        vz = -gs.abs(vz) * 0.65

    if c == 0:
        return px
    if c == 1:
        return py
    if c == 2:
        return pz
    if c == 3:
        return state[i, 3] + vz * dt / gs.max(radius, 0.1)
    if c == 5:
        return state[i, 5] - vx * dt / gs.max(radius, 0.1)
    if c == 8:
        return vx
    if c == 9:
        return vy
    if c == 10:
        return vz
    if c == 11:
        return sub
    return state[i, c]


@gs.kernel
def floating_wakes(
    state: gs.f32["m", "c16"],
    params: gs.f32["m", "c8"],
    h_grid: gs.f32["gh", "gw"],
    size: gs.f32,
    dt: gs.f32,
) -> gs.f32["m", "c8"]:
    """Sillage de chaque flotteur : `[cx, cy, 1/r², force, couronne, 0, 0, 0]` (même forme que `water_add_splash`)."""
    i, c = gs.index()
    n = float(h_grid.shape[1] - 1)
    cx = gs.min(gs.max(state[i, 0] / size + 0.5, 0.02), 0.98) * n
    cy = gs.min(gs.max(state[i, 2] / size + 0.5, 0.02), 0.98) * n
    r_c = gs.max(1.0, params[i, 0] * 1.15 / size * n)
    inv_r2 = 1.0 / (r_c * r_c)
    if c == 0:
        return cx
    if c == 1:
        return cy
    if c == 2:
        return inv_r2
    if c == 3:
        drop = 0.0
        if state[i, 9] < -0.35:
            drop = -state[i, 9]
        slide = gs.sqrt(state[i, 8] * state[i, 8] + state[i, 10] * state[i, 10])
        imp = (drop * 0.12 + slide * 0.025) * params[i, 6]
        if imp <= 0.004 or state[i, 11] <= 0.05:
            return 0.0
        return -gs.min(imp, 0.18) * dt * 60.0
    if c == 4:
        core_x = 0.0
        core_y = 0.0
        wide_x = 0.0
        wide_y = 0.0
        for k in range(h_grid.shape[1]):
            dx2 = (float(k) - cx) * (float(k) - cx) * inv_r2
            core_x += gs.exp(-dx2)
            wide_x += gs.exp(-dx2 * 0.3)
        for k in range(h_grid.shape[0]):
            dy2 = (float(k) - cy) * (float(k) - cy) * inv_r2
            core_y += gs.exp(-dy2)
            wide_y += gs.exp(-dy2 * 0.3)
        return core_x * core_y / (wide_x * wide_y)
    return 0.0


@gs.kernel
def water_add_wakes(
    h_curr: gs.f32["gh", "gw"],
    h_prev: gs.f32["gh", "gw"],
    wakes: gs.f32["m", "c8"],
) -> (gs.f32["gh", "gw"], gs.f32["gh", "gw"]):
    """Ajoute tous les sillages d'un coup aux deux instants de la grille."""
    y, x = gs.index()
    add = 0.0
    for i in range(wakes.shape[0]):
        if wakes[i, 3] != 0.0:
            dx = float(x) - wakes[i, 0]
            dy = float(y) - wakes[i, 1]
            d2 = (dx * dx + dy * dy) * wakes[i, 2]
            add += wakes[i, 3] * (gs.exp(-d2) - wakes[i, 4] * gs.exp(-d2 * 0.3))
    return gs.min(gs.max(h_curr[y, x] + add, -2.5), 2.5), gs.min(gs.max(h_prev[y, x] + add, -2.5), 2.5)


@gs.kernel
def floating_collide(
    state: gs.f32["m", "c16"],
    params: gs.f32["m", "c8"],
) -> gs.f32["m", "c16"]:
    """Collisions douces entre flotteurs : chacun recule de la moitié du recouvrement et échange sa vitesse."""
    i, c = gs.index()
    if not (c < 3 or (c >= 8 and c < 11)):
        return state[i, c]
    a = c
    if c >= 8:
        a = c - 8
    out = state[i, c]
    for j in range(state.shape[0]):
        if j != i:
            ex = state[j, 0] - state[i, 0]
            ey = state[j, 1] - state[i, 1]
            ez = state[j, 2] - state[i, 2]
            dist = gs.sqrt(ex * ex + ey * ey + ez * ez)
            overlap = params[i, 0] + params[j, 0] - dist
            if dist > 1e-5 and overlap > 0.0:
                inv_d = 1.0 / dist
                n_a = ex * inv_d
                if a == 1:
                    n_a = ey * inv_d
                elif a == 2:
                    n_a = ez * inv_d
                if c < 3:
                    out -= n_a * overlap * 0.5
                else:
                    rv = ((state[j, 8] - state[i, 8]) * ex + (state[j, 9] - state[i, 9]) * ey + (state[j, 10] - state[i, 10]) * ez) * inv_d
                    if rv < 0.0:
                        out += 0.8 * rv * n_a
    return out
