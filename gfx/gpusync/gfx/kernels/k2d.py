"""Kernels GPU de la couche 2D pure (`render_2d_shapes`, `scatter_particles_2d`, `composite_particles_2d`)."""

from __future__ import annotations

import gpusync as gs


@gs.kernel
def render_2d_shapes(
    base: gs.u32["h", "w"],
    shapes_f: gs.f32["ns", "c8"],
    shapes_i: gs.i32["ns", "c8"],
    tex_pool: gs.u32["tp"],
    tex_meta: gs.i32["ntex", "c4"],
    dyn_atlas: gs.u32["dh", "dw"],
    n_shapes: gs.i32,
    cam_x: gs.f32,
    cam_y: gs.f32,
    cam_inv_zoom: gs.f32,
    cam_cos: gs.f32,
    cam_sin: gs.f32,
    clear_rgba: gs.u32,
    do_clear: gs.i32,
) -> gs.u32["h", "w"]:
    """Rend rectangles, cercles, lignes anti-aliasés et sprites texturés 2D en 1 seul dispatch GPU.

    Si `do_clear != 0`, part de `clear_rgba` (scène 2D pure ou RenderTarget).
    Sinon, compose directement par-dessus `base` (couche 2D au-dessus de la 3D).
    """
    py, px = gs.index()
    h = base.shape[0]
    w = base.shape[1]

    c0 = clear_rgba
    if do_clear == 0:
        c0 = base[py, px]
    if n_shapes <= 0:
        return c0

    mask = gs.u32(255)
    r = float(c0 & mask)
    g = float((c0 >> gs.u32(8)) & mask)
    b = float((c0 >> gs.u32(16)) & mask)

    # Coordonnées pixel → monde 2D (origine caméra au centre de l'écran, Y vers le bas)
    sx = (float(px) + 0.5 - float(w) * 0.5) * cam_inv_zoom
    sy = (float(py) + 0.5 - float(h) * 0.5) * cam_inv_zoom
    wx = cam_x + sx * cam_cos - sy * cam_sin
    wy = cam_y + sx * cam_sin + sy * cam_cos

    for si in range(shapes_i.shape[0]):
        if si < n_shapes:
            kind = shapes_i[si, 0]
            col_u = gs.u32(shapes_i[si, 1])
            cr = float(col_u & mask)
            cg = float((col_u >> gs.u32(8)) & mask)
            cb = float((col_u >> gs.u32(16)) & mask)
            ca = float((col_u >> gs.u32(24)) & mask) / 255.0

            if kind == 0:
                # Rect2D : [cx, cy, hw, hh, cos_r, sin_r, border_w, 0]
                dx = wx - shapes_f[si, 0]
                dy = wy - shapes_f[si, 1]
                rc = shapes_f[si, 4]
                rs = shapes_f[si, 5]
                lx = dx * rc + dy * rs
                ly = -dx * rs + dy * rc
                hw = shapes_f[si, 2]
                hh = shapes_f[si, 3]
                ax = gs.abs(lx) - hw
                ay = gs.abs(ly) - hh
                dist = gs.max(ax, ay)
                if dist <= 0.5:
                    cov = gs.min(gs.max(0.5 - dist, 0.0), 1.0)
                    sr = cr
                    sg = cg
                    sb = cb
                    sa = ca * cov
                    bd_u = gs.u32(shapes_i[si, 2])
                    bda = float((bd_u >> gs.u32(24)) & mask) / 255.0
                    bw = shapes_f[si, 6]
                    if bda > 0.0 and dist > -bw:
                        sr = float(bd_u & mask)
                        sg = float((bd_u >> gs.u32(8)) & mask)
                        sb = float((bd_u >> gs.u32(16)) & mask)
                        sa = bda * cov
                    r = r * (1.0 - sa) + sr * sa
                    g = g * (1.0 - sa) + sg * sa
                    b = b * (1.0 - sa) + sb * sa

            elif kind == 1:
                # Circle2D : [cx, cy, radius, radius, 1, 0, border_w, 0]
                dx = wx - shapes_f[si, 0]
                dy = wy - shapes_f[si, 1]
                rad = shapes_f[si, 2]
                dist = gs.sqrt(dx * dx + dy * dy) - rad
                if dist <= 0.75:
                    cov = gs.min(gs.max(0.75 - dist, 0.0) / 1.5, 1.0)
                    sr = cr
                    sg = cg
                    sb = cb
                    sa = ca * cov
                    bd_u = gs.u32(shapes_i[si, 2])
                    bda = float((bd_u >> gs.u32(24)) & mask) / 255.0
                    bw = shapes_f[si, 6]
                    if bda > 0.0 and dist > -bw:
                        sr = float(bd_u & mask)
                        sg = float((bd_u >> gs.u32(8)) & mask)
                        sb = float((bd_u >> gs.u32(16)) & mask)
                        sa = bda * cov
                    r = r * (1.0 - sa) + sr * sa
                    g = g * (1.0 - sa) + sg * sa
                    b = b * (1.0 - sa) + sb * sa

            elif kind == 2:
                # Line2D : [x1, y1, x2, y2, half_thick, 0, 0, 0]
                x1 = shapes_f[si, 0]
                y1 = shapes_f[si, 1]
                x2 = shapes_f[si, 2]
                y2 = shapes_f[si, 3]
                ht = shapes_f[si, 4]
                abx = x2 - x1
                aby = y2 - y1
                apx = wx - x1
                apy = wy - y1
                ab2 = gs.max(abx * abx + aby * aby, 1e-6)
                t_seg = gs.min(gs.max((apx * abx + apy * aby) / ab2, 0.0), 1.0)
                ex = apx - abx * t_seg
                ey = apy - aby * t_seg
                dist = gs.sqrt(ex * ex + ey * ey) - ht
                if dist <= 0.75:
                    cov = gs.min(gs.max(0.75 - dist, 0.0) / 1.5, 1.0)
                    sa = ca * cov
                    r = r * (1.0 - sa) + cr * sa
                    g = g * (1.0 - sa) + cg * sa
                    b = b * (1.0 - sa) + cb * sa

            elif kind == 3:
                # Sprite2D : [cx, cy, hw, hh, cos_r, sin_r, 0, 0], tid = shapes_i[si, 3]
                dx = wx - shapes_f[si, 0]
                dy = wy - shapes_f[si, 1]
                rc = shapes_f[si, 4]
                rs = shapes_f[si, 5]
                lx = dx * rc + dy * rs
                ly = -dx * rs + dy * rc
                hw = shapes_f[si, 2]
                hh = shapes_f[si, 3]
                if gs.abs(lx) <= hw and gs.abs(ly) <= hh:
                    tid = shapes_i[si, 3]
                    if tid >= 0:
                        u = (lx + hw) / gs.max(hw * 2.0, 1e-5)
                        v = (ly + hh) / gs.max(hh * 2.0, 1e-5)
                        tw = tex_meta[tid, 2]
                        meta3 = tex_meta[tid, 3]
                        th = meta3 & 1048575
                        is_dyn = (meta3 >> 20) & 1
                        is_lin = (meta3 >> 21) & 1
                        tr = 0.0
                        tg = 0.0
                        tb = 0.0
                        ta = 0.0
                        if is_lin != 0:
                            fxt = u * float(tw) - 0.5
                            fyt = v * float(th) - 0.5
                            x0t = gs.min(gs.max(int(gs.floor(fxt)), 0), tw - 1)
                            y0t = gs.min(gs.max(int(gs.floor(fyt)), 0), th - 1)
                            x1t = gs.min(x0t + 1, tw - 1)
                            y1t = gs.min(y0t + 1, th - 1)
                            ftx = gs.min(gs.max(fxt - float(x0t), 0.0), 1.0)
                            fty = gs.min(gs.max(fyt - float(y0t), 0.0), 1.0)
                            c00 = gs.u32(0)
                            c10 = gs.u32(0)
                            c01 = gs.u32(0)
                            c11 = gs.u32(0)
                            if is_dyn != 0:
                                oxd = tex_meta[tid, 0]
                                oyd = tex_meta[tid, 1]
                                c00 = dyn_atlas[oyd + y0t, oxd + x0t]
                                c10 = dyn_atlas[oyd + y0t, oxd + x1t]
                                c01 = dyn_atlas[oyd + y1t, oxd + x0t]
                                c11 = dyn_atlas[oyd + y1t, oxd + x1t]
                            else:
                                offs = tex_meta[tid, 0]
                                c00 = tex_pool[offs + y0t * tw + x0t]
                                c10 = tex_pool[offs + y0t * tw + x1t]
                                c01 = tex_pool[offs + y1t * tw + x0t]
                                c11 = tex_pool[offs + y1t * tw + x1t]
                            w00 = (1.0 - ftx) * (1.0 - fty)
                            w10 = ftx * (1.0 - fty)
                            w01 = (1.0 - ftx) * fty
                            w11 = ftx * fty
                            tr = (w00 * float(c00 & mask) + w10 * float(c10 & mask) + w01 * float(c01 & mask) + w11 * float(c11 & mask)) * (cr / 255.0)
                            tg = (w00 * float((c00 >> gs.u32(8)) & mask) + w10 * float((c10 >> gs.u32(8)) & mask) + w01 * float((c01 >> gs.u32(8)) & mask) + w11 * float((c11 >> gs.u32(8)) & mask)) * (cg / 255.0)
                            tb = (w00 * float((c00 >> gs.u32(16)) & mask) + w10 * float((c10 >> gs.u32(16)) & mask) + w01 * float((c01 >> gs.u32(16)) & mask) + w11 * float((c11 >> gs.u32(16)) & mask)) * (cb / 255.0)
                            ta = (w00 * float((c00 >> gs.u32(24)) & mask) + w10 * float((c10 >> gs.u32(24)) & mask) + w01 * float((c01 >> gs.u32(24)) & mask) + w11 * float((c11 >> gs.u32(24)) & mask)) / 255.0 * ca
                        else:
                            tx = gs.min(gs.max(int(u * float(tw)), 0), tw - 1)
                            ty = gs.min(gs.max(int(v * float(th)), 0), th - 1)
                            tc = gs.u32(0)
                            if is_dyn != 0:
                                tc = dyn_atlas[tex_meta[tid, 1] + ty, tex_meta[tid, 0] + tx]
                            else:
                                tc = tex_pool[tex_meta[tid, 0] + ty * tw + tx]
                            tr = float(tc & mask) * (cr / 255.0)
                            tg = float((tc >> gs.u32(8)) & mask) * (cg / 255.0)
                            tb = float((tc >> gs.u32(16)) & mask) * (cb / 255.0)
                            ta = (float((tc >> gs.u32(24)) & mask) / 255.0) * ca
                        r = r * (1.0 - ta) + tr * ta
                        g = g * (1.0 - ta) + tg * ta
                        b = b * (1.0 - ta) + tb * ta
                    else:
                        r = r * (1.0 - ca) + cr * ca
                        g = g * (1.0 - ca) + cg * ca
                        b = b * (1.0 - ca) + cb * ca

    ri = int(gs.min(gs.max(r, 0.0), 255.0) + 0.5)
    gi = int(gs.min(gs.max(g, 0.0), 255.0) + 0.5)
    bi = int(gs.min(gs.max(b, 0.0), 255.0) + 0.5)
    return gs.u32(ri) | (gs.u32(gi) << gs.u32(8)) | (gs.u32(bi) << gs.u32(16)) | gs.u32(4278190080)


@gs.kernel(grid="px")
def scatter_particles_2d(
    px: gs.f32["np"],
    py: gs.f32["np"],
    colors: gs.u32["np"],
    cam_x: gs.f32,
    cam_y: gs.f32,
    cam_zoom: gs.f32,
    cam_cos: gs.f32,
    cam_sin: gs.f32,
    point_size: gs.i32,
    layer: gs.Out[gs.u32["h", "w"]],
):
    """Projette et dessine N particules 2D directement en VRAM (`gs.atomic_max`, 0 copie CPU)."""
    i = gs.index()
    h = layer.shape[0]
    w = layer.shape[1]

    dx = px[i] - cam_x
    dy = py[i] - cam_y
    rx = dx * cam_cos + dy * cam_sin
    ry = -dx * cam_sin + dy * cam_cos
    sx = int(rx * cam_zoom + float(w) * 0.5)
    sy = int(ry * cam_zoom + float(h) * 0.5)

    col = colors[i] | gs.u32(4278190080)
    rad = gs.max(point_size - 1, 0)
    if sx >= -rad and sx < w + rad and sy >= -rad and sy < h + rad:
        for oy in range(-rad, rad + 1):
            yy = sy + oy
            if yy >= 0 and yy < h:
                for ox in range(-rad, rad + 1):
                    xx = sx + ox
                    if xx >= 0 and xx < w:
                        gs.atomic_max(layer, (yy, xx), col)


@gs.kernel
def composite_particles_2d(base: gs.u32["h", "w"], layer: gs.u32["h", "w"]) -> gs.u32["h", "w"]:
    """Superpose la couche particules 2D (où `layer != 0`) sur le framebuffer."""
    py, px = gs.index()
    p = layer[py, px]
    if p != gs.u32(0):
        return p
    return base[py, px]
