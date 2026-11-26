"""Fine grid nested in a coarse grid: pressure with Neumann walls, and coupling.

Cosine transform (DCT-II and its inverse) in two Cooley-Tukey passes, after Makhoul (1980): a
complex Fourier transform of the same size on reordered values."""

import gpusync as gs


@gs.kernel
def dct_x1(a: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c4"], n1: gs.i32, n2: gs.i32) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    k, j, i = gs.index()
    n = n1 * n2
    k1 = i // n2
    col = i - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        p = m * n2 + col
        src = 2 * p
        if src >= n:
            src = 2 * (n - 1 - p) + 1
        val = a[k, j, src]
        re += val * trig[idx, 0]
        im -= val * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def dct_x2(
    re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c4"], n1: gs.i32, n2: gs.i32
) -> gs.f32["nz", "ny", "nx"]:
    k, j, i = gs.index()
    n = n1 * n2
    k2 = i // n1
    row = (i - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    fr = 0.0
    fi = 0.0
    for m in range(n2):
        c = trig[idx, 0]
        q = trig[idx, 1]
        vre = re[k, j, row + m]
        vim = im[k, j, row + m]
        fr += vre * c + vim * q
        fi += vim * c - vre * q
        idx += step
        if idx >= n:
            idx -= n
    return fr * trig[i, 2] + fi * trig[i, 3]


@gs.kernel
def dct_y1(a: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c4"], n1: gs.i32, n2: gs.i32) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    k, j, i = gs.index()
    n = n1 * n2
    k1 = j // n2
    col = j - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        p = m * n2 + col
        src = 2 * p
        if src >= n:
            src = 2 * (n - 1 - p) + 1
        val = a[k, src, i]
        re += val * trig[idx, 0]
        im -= val * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def dct_y2_columns(
    re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c4"], tpl: gs.f32["ny", "nb", "nz", "g"], n1: gs.i32, n2: gs.i32
) -> gs.f32["ny", "nb", "nz", "g"]:
    j, bl, k, r = gs.index()
    i = bl * tpl.shape[3] + r
    n = n1 * n2
    k2 = j // n1
    row = (j - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    fr = 0.0
    fi = 0.0
    for m in range(n2):
        c = trig[idx, 0]
        q = trig[idx, 1]
        vre = re[k, row + m, i]
        vim = im[k, row + m, i]
        fr += vre * c + vim * q
        fi += vim * c - vre * q
        idx += step
        if idx >= n:
            idx -= n
    return fr * trig[j, 2] + fi * trig[j, 3]


@gs.kernel
def idct_columns_y1(a: gs.f32["ny", "nb", "nz", "g"], tpl: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c4"], n1: gs.i32, n2: gs.i32) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    k, j, i = gs.index()
    bl = i // a.shape[3]
    rk = i - bl * a.shape[3]
    n = n1 * n2
    k1 = j // n2
    col = j - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        src = m * n2 + col
        x0 = a[src, bl, k, rk]
        x1 = 0.0
        ct = trig[src, 2]
        st = trig[src, 3]
        if src > 0:
            src = n - src
            x1 = a[src, bl, k, rk]
        vr = x0 * ct + x1 * st
        vi = x0 * st - x1 * ct
        c = trig[idx, 0]
        q = trig[idx, 1]
        re += vr * c - vi * q
        im += vr * q + vi * c
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def idct_y2(re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["ny", "c4"], n1: gs.i32, n2: gs.i32) -> gs.f32["nz", "ny", "nx"]:
    k, j, i = gs.index()
    n = n1 * n2
    dst = j // 2
    if (j & 1) == 1:
        dst = n - 1 - j // 2
    k2 = dst // n1
    row = (dst - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    total = 0.0
    for m in range(n2):
        total += re[k, row + m, i] * trig[idx, 0] - im[k, row + m, i] * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return total / float(n)


@gs.kernel
def idct_x1(a: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c4"], n1: gs.i32, n2: gs.i32) -> (gs.f32["nz", "ny", "nx"], gs.f32["nz", "ny", "nx"]):
    k, j, i = gs.index()
    n = n1 * n2
    k1 = i // n2
    col = i - k1 * n2
    re = 0.0
    im = 0.0
    idx = k1 * col % n
    step = k1 * n2 % n
    for m in range(n1):
        src = m * n2 + col
        x0 = a[k, j, src]
        x1 = 0.0
        ct = trig[src, 2]
        st = trig[src, 3]
        if src > 0:
            src = n - src
            x1 = a[k, j, src]
        vr = x0 * ct + x1 * st
        vi = x0 * st - x1 * ct
        c = trig[idx, 0]
        q = trig[idx, 1]
        re += vr * c - vi * q
        im += vr * q + vi * c
        idx += step
        if idx >= n:
            idx -= n
    return re, im


@gs.kernel
def idct_x2(re: gs.f32["nz", "ny", "nx"], im: gs.f32["nz", "ny", "nx"], trig: gs.f32["nx", "c4"], n1: gs.i32, n2: gs.i32) -> gs.f32["nz", "ny", "nx"]:
    k, j, i = gs.index()
    n = n1 * n2
    dst = i // 2
    if (i & 1) == 1:
        dst = n - 1 - i // 2
    k2 = dst // n1
    row = (dst - k2 * n1) * n2
    step = k2 * n1 % n
    idx = 0
    total = 0.0
    for m in range(n2):
        total += re[k, j, row + m] * trig[idx, 0] - im[k, j, row + m] * trig[idx, 1]
        idx += step
        if idx >= n:
            idx -= n
    return total / float(n)


# ───────────────────────── Coupling the grids ─────────────────────────
# Component `comp`: 0 = u, 1 = v, 2 = w, 3 = scalar. The first and last `halo` cells in x and y of the
# fine grid are imposed by the coarse grid.


@gs.kernel
def nest_fill(
    old: gs.f32["cz", "cy", "cx"],
    new: gs.f32["cz", "cy", "cx"],
    tpl: gs.f32["nz", "ny", "nx"],
    comp: gs.i32,
    blend: gs.f32,
    x0: gs.f32,
    y0: gs.f32,
    rx: gs.f32,
    ry: gs.f32,
    rz: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Coarse-grid field interpolated (trilinearly, and linearly between two instants) on the fine grid.

    `x0`, `y0`: position of the fine grid's corner, in coarse cells; `rx`…: fine size / coarse size."""
    k, j, i = gs.index()
    cz = old.shape[0]
    cy = old.shape[1]
    cx = old.shape[2]
    ox = 0.5
    oy = 0.5
    oz = 0.5
    if comp == 0:
        ox = 0.0
    elif comp == 1:
        oy = 0.0
    elif comp == 2:
        oz = 0.0
    fx = x0 + (float(i) + ox) * rx - ox
    fy = y0 + (float(j) + oy) * ry - oy
    fz = (float(k) + oz) * rz - oz
    if comp != 2:
        fz = gs.min(gs.max(fz, 0.0), float(cz - 1))
    ix = int(gs.floor(fx))
    iy = int(gs.floor(fy))
    iz = int(gs.floor(fz))
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)
    total = 0.0
    for m in range(8):
        a = m & 1
        b = (m >> 1) & 1
        c = m >> 2
        wt = (tx if a == 1 else 1.0 - tx) * (ty if b == 1 else 1.0 - ty) * (tz if c == 1 else 1.0 - tz)
        jx = (ix + a) % cx
        jy = (iy + b) % cy
        jz = iz + c
        val = 0.0
        if comp == 2:
            if jz > 0 and jz < cz:
                val = old[jz, jy, jx] * (1.0 - blend) + new[jz, jy, jx] * blend
        else:
            jz = gs.min(jz, cz - 1)
            val = old[jz, jy, jx] * (1.0 - blend) + new[jz, jy, jx] * blend
        total += wt * val
    if comp == 2 and k == 0:
        return 0.0
    return total + tpl[k, j, i] * 0.0


@gs.kernel
def nest_halo(
    field: gs.f32["nz", "ny", "nx"], target: gs.f32["nz", "ny", "nx"], comp: gs.i32, halo: gs.i32
) -> gs.f32["nz", "ny", "nx"]:
    """Imposes `target` outside the interior (boundary faces included for the normal component)."""
    k, j, i = gs.index()
    lo_i = halo
    lo_j = halo
    if comp == 0:
        lo_i = halo + 1
    elif comp == 1:
        lo_j = halo + 1
    if i < lo_i or i >= field.shape[2] - halo or j < lo_j or j >= field.shape[1] - halo:
        return target[k, j, i]
    return field[k, j, i]


@gs.kernel
def nest_relax(
    slow: gs.f32["nz", "ny", "nx"],
    field: gs.f32["nz", "ny", "nx"],
    target: gs.f32["nz", "ny", "nx"],
    halo: gs.i32,
    width: gs.f32,
    rate: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Adds a relaxation toward the coarse grid, strong at the edge and zero at `width` cells (Davies zone)."""
    k, j, i = gs.index()
    d = gs.min(i - halo, field.shape[2] - 1 - halo - i, j - halo, field.shape[1] - 1 - halo - j)
    x = (float(d) + 0.5) / width
    if x >= 1.0 or d < 0:
        return slow[k, j, i]
    c = gs.cos(x * 1.5707963)
    return slow[k, j, i] - rate * c * c * (field[k, j, i] - target[k, j, i])


@gs.kernel
def nest_divergence(
    u: gs.f32["nz", "ny", "nx"],
    v: gs.f32["nz", "ny", "nx"],
    w: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    inner: gs.f32["nz", "my", "mx"],
    halo: gs.i32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
) -> gs.f32["nz", "my", "mx"]:
    """Divergence of the mass flux over the interior of the fine grid."""
    k, j, i = gs.index()
    jj = j + halo
    ii = i + halo
    top = 0.0
    if k < u.shape[0] - 1:
        top = ref[k + 1, 6] * w[k + 1, jj, ii]
    bot = 0.0
    if k > 0:
        bot = ref[k, 6] * w[k, jj, ii]
    flux = (u[k, jj, ii + 1] - u[k, jj, ii]) * inv_dx + (v[k, jj + 1, ii] - v[k, jj, ii]) * inv_dy
    return ref[k, 5] * flux + (top - bot) * inv_dz + inner[k, j, i] * 0.0


@gs.kernel
def nest_project(
    vel: gs.f32["nz", "ny", "nx"], phi: gs.f32["nz", "my", "mx"], comp: gs.i32, halo: gs.i32, inv_d: gs.f32
) -> gs.f32["nz", "ny", "nx"]:
    """Removes the pressure gradient on the interior faces; boundary faces keep the imposed value."""
    k, j, i = gs.index()
    ii = i - halo
    jj = j - halo
    mx = phi.shape[2]
    my = phi.shape[1]
    val = vel[k, j, i]
    if ii < 0 or ii >= mx or jj < 0 or jj >= my:
        return val
    if comp == 0:
        if ii > 0:
            val -= (phi[k, jj, ii] - phi[k, jj, ii - 1]) * inv_d
    elif comp == 1:
        if jj > 0:
            val -= (phi[k, jj, ii] - phi[k, jj - 1, ii]) * inv_d
    elif k > 0:
        val -= (phi[k, jj, ii] - phi[k - 1, jj, ii]) * inv_d
    return val


@gs.kernel
def nest_shift(
    field: gs.f32["nz", "ny", "nx"],
    fill: gs.f32["nz", "ny", "nx"],
    comp: gs.i32,
    halo: gs.i32,
    shift_x: gs.i32,
    shift_y: gs.i32,
) -> gs.f32["nz", "ny", "nx"]:
    """Moves the fine grid by a whole number of cells; the new cells come from the coarse grid."""
    k, j, i = gs.index()
    si = i + shift_x
    sj = j + shift_y
    if si < halo or si >= field.shape[2] - halo or sj < halo or sj >= field.shape[1] - halo:
        return fill[k, j, i]
    return field[k, sj, si]
