"""100 % GPU kernels (`@gs.kernel`) for the next-generation volumetric 3D fluid engine:
1. **Staggered 3D MAC grid FLIP / PIC** solver (Blender Mantaflow / Bridson, O(N) up to 25,000+ drops).
2. **GPU whitewater system** (Sebastian Lague, `Rendering Fluids`): thousands of real micro-particles
   of spray (*spray* in the air), surface foam (*foam*) and underwater bubbles (*bubbles*).
3. **Continuous high-resolution 3D volume** (`96×60×52` in VRAM, separable 3D filtering).
4. **Two-bounce optical 3D ray tracing (`fluid_raymarch_3d`)**:
   - Exact 3D isosurface + perfectly flat contact against the aquarium glass.
   - Exact trilinear C¹ 3D normals (`-∇density`).
   - **3D Snell refraction on entry (air→water) + exit (water→air) + total internal reflection (TIR)**!
   - **Exact Beer-Lambert absorption**, **underwater caustics**, **translucent fluid shadow** on the bottom,
     **Fresnel reflection + HDRI skybox** and inlay of the *whitewater* micro-particles (spray / foam / bubbles).
"""

from __future__ import annotations

import gpusync as gs


# ───────────────────────── 1. Staggered MAC Grid 3D FLIP / PIC Solver ─────────────────────────


@gs.kernel(grid=("n",))
def mac_p2g(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    u_tpl: gs.f32["nu"],
    v_tpl: gs.f32["nv"],
    w_tpl: gs.f32["nw"],
    c_tpl: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    u_sum: gs.Out[gs.u32["nu"], 0],
    u_wt: gs.Out[gs.u32["nu"], 0],
    v_sum: gs.Out[gs.u32["nv"], 0],
    v_wt: gs.Out[gs.u32["nv"], 0],
    w_sum: gs.Out[gs.u32["nw"], 0],
    w_wt: gs.Out[gs.u32["nw"], 0],
    cell_cnt: gs.Out[gs.u32["nc"], 0],
):
    """P2G on a staggered MAC grid: splats u on X faces, v on Y faces, w on Z faces and density at the centre."""
    i = gs.index()
    _z = (u_tpl[0] + v_tpl[0] + w_tpl[0] + c_tpl[0]) * 0.0

    gx = (pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(nx)
    gy = (pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(ny)
    gz = (pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(nz)

    gx = gs.min(gs.max(gx, 0.01), float(nx) - 0.01)
    gy = gs.min(gs.max(gy, 0.01), float(ny) - 0.01)
    gz = gs.min(gs.max(gz, 0.01), float(nz) - 0.01)

    cx = gs.min(gs.max(int(gs.floor(gx)), 0), nx - 1)
    cy = gs.min(gs.max(int(gs.floor(gy)), 0), ny - 1)
    cz = gs.min(gs.max(int(gs.floor(gz)), 0), nz - 1)
    gs.atomic_add(cell_cnt, (cz * ny + cy) * nx + cx, gs.u32(1))

    vx_b = gs.min(gs.max(vel[i, 0], -18.0), 18.0) + 20.0 + _z
    vy_b = gs.min(gs.max(vel[i, 1], -18.0), 18.0) + 20.0
    vz_b = gs.min(gs.max(vel[i, 2], -18.0), 18.0) + 20.0

    # Faces U : (nx+1, ny, nz)
    ux_f = gs.min(gs.max(gx, 0.001), float(nx) - 0.001)
    uy_f = gs.min(gs.max(gy - 0.5, 0.001), float(ny - 1) - 0.001)
    uz_f = gs.min(gs.max(gz - 0.5, 0.001), float(nz - 1) - 0.001)
    ux0 = int(gs.floor(ux_f))
    uy0 = int(gs.floor(uy_f))
    uz0 = int(gs.floor(uz_f))
    utx = ux_f - float(ux0)
    uty = uy_f - float(uy0)
    utz = uz_f - float(uz0)
    nx1 = nx + 1
    u_sz = nx1 * ny
    for dz in range(2):
        wz = utz if dz == 1 else (1.0 - utz)
        for dy in range(2):
            wy = uty if dy == 1 else (1.0 - uty)
            for dx in range(2):
                wx = utx if dx == 1 else (1.0 - utx)
                wt = wx * wy * wz * 256.0
                if wt > 0.5:
                    idx = (uz0 + dz) * u_sz + (uy0 + dy) * nx1 + (ux0 + dx)
                    # Integer weight shared by both sums: otherwise the mean is biased
                    q = gs.u32(wt + 0.5)
                    gs.atomic_add(u_wt, idx, q)
                    gs.atomic_add(u_sum, idx, gs.u32(float(q) * vx_b + 0.5))

    # Faces V : (nx, ny+1, nz)
    vx_f = gs.min(gs.max(gx - 0.5, 0.001), float(nx - 1) - 0.001)
    vy_f = gs.min(gs.max(gy, 0.001), float(ny) - 0.001)
    vz_f = gs.min(gs.max(gz - 0.5, 0.001), float(nz - 1) - 0.001)
    vx0 = int(gs.floor(vx_f))
    vy0 = int(gs.floor(vy_f))
    vz0 = int(gs.floor(vz_f))
    vtx = vx_f - float(vx0)
    vty = vy_f - float(vy0)
    vtz = vz_f - float(vz0)
    ny1 = ny + 1
    v_sz = nx * ny1
    for dz in range(2):
        wz = vtz if dz == 1 else (1.0 - vtz)
        for dy in range(2):
            wy = vty if dy == 1 else (1.0 - vty)
            for dx in range(2):
                wx = vtx if dx == 1 else (1.0 - vtx)
                wt = wx * wy * wz * 256.0
                if wt > 0.5:
                    idx = (vz0 + dz) * v_sz + (vy0 + dy) * nx + (vx0 + dx)
                    # Integer weight shared by both sums: otherwise the mean is biased
                    q = gs.u32(wt + 0.5)
                    gs.atomic_add(v_wt, idx, q)
                    gs.atomic_add(v_sum, idx, gs.u32(float(q) * vy_b + 0.5))

    # Faces W : (nx, ny, nz+1)
    wx_f = gs.min(gs.max(gx - 0.5, 0.001), float(nx - 1) - 0.001)
    wy_f = gs.min(gs.max(gy - 0.5, 0.001), float(ny - 1) - 0.001)
    wz_f = gs.min(gs.max(gz, 0.001), float(nz) - 0.001)
    wx0 = int(gs.floor(wx_f))
    wy0 = int(gs.floor(wy_f))
    wz0 = int(gs.floor(wz_f))
    wtx = wx_f - float(wx0)
    wty = wy_f - float(wy0)
    wtz = wz_f - float(wz0)
    w_sz = nx * ny
    for dz in range(2):
        wz = wtz if dz == 1 else (1.0 - wtz)
        for dy in range(2):
            wy = wty if dy == 1 else (1.0 - wty)
            for dx in range(2):
                wx = wtx if dx == 1 else (1.0 - wtx)
                wt = wx * wy * wz * 256.0
                if wt > 0.5:
                    idx = (wz0 + dz) * w_sz + (wy0 + dy) * nx + (wx0 + dx)
                    # Integer weight shared by both sums: otherwise the mean is biased
                    q = gs.u32(wt + 0.5)
                    gs.atomic_add(w_wt, idx, q)
                    gs.atomic_add(w_sum, idx, gs.u32(float(q) * vz_b + 0.5))


@gs.kernel
def mac_face_forces(
    f_sum: gs.u32["nf"],
    f_wt: gs.u32["nf"],
    stride_x: gs.i32,
    stride_y: gs.i32,
    axis: gs.i32,
    axis_max: gs.i32,
    accel_dt: gs.f32,
) -> (gs.f32["nf"], gs.f32["nf"]):
    """Normalizes a grid of MAC faces (u, v or w) and returns `(vel_old, vel_star)` with solid walls."""
    idx = gs.index()
    plane = stride_x * stride_y
    iz = idx // plane
    rem = idx - iz * plane
    iy = rem // stride_x
    ix = rem - iy * stride_x

    coord = ix
    if axis == 1:
        coord = iy
    elif axis == 2:
        coord = iz

    wt = f_wt[idx]
    v_old = 0.0
    v_star = 0.0
    if wt > gs.u32(4):
        v_old = float(f_sum[idx]) / float(wt) - 20.0
        v_star = v_old + accel_dt
    if coord == 0 or coord == axis_max:
        v_old = 0.0
        v_star = 0.0
    return v_old, v_star


@gs.kernel(grid=("n",))
def mac_mass_splat(
    pos: gs.f32["n", "c4"],
    c_tpl: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    mass: gs.Out[gs.u32["nc"], 0],
):
    """Trilinear mass at the cell centres (256 = one particle)."""
    i = gs.index()
    _z = c_tpl[0] * 0.0
    gx = gs.min(gs.max((pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(nx) - 0.5 + _z, 0.0), float(nx) - 1.001)
    gy = gs.min(gs.max((pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(ny) - 0.5, 0.0), float(ny) - 1.001)
    gz = gs.min(gs.max((pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(nz) - 0.5, 0.0), float(nz) - 1.001)
    ix = int(gx)
    iy = int(gy)
    iz = int(gz)
    tx = gx - float(ix)
    ty = gy - float(iy)
    tz = gz - float(iz)
    for k in range(8):
        ox = k & 1
        oy = (k >> 1) & 1
        oz = k >> 2
        w = (tx if ox == 1 else 1.0 - tx) * (ty if oy == 1 else 1.0 - ty) * (tz if oz == 1 else 1.0 - tz)
        q = gs.u32(w * 256.0 + 0.5)
        if q > gs.u32(0):
            gs.atomic_add(mass, ((iz + oz) * ny + iy + oy) * nx + ix + ox, q)


@gs.kernel
def mac_fill(mass: gs.u32["nc"], rest_cnt: gs.f32) -> gs.f32["nc"]:
    """Fill ratio of each cell (1 = water at rest). Water from 0.5 on: the surface is at that level."""
    cid = gs.index()
    return float(mass[cid]) / (256.0 * rest_cnt)


@gs.kernel
def mac_divergence(
    u: gs.f32["nu"],
    v: gs.f32["nv"],
    w: gs.f32["nw"],
    fill: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
) -> gs.f32["nc"]:
    """MAC divergence over 1 cell."""
    cid = gs.index()
    f = fill[cid]
    if f < 0.5:
        return 0.0
    plane = nx * ny
    iz = cid // plane
    rem = cid - iz * plane
    iy = rem // nx
    ix = rem - iy * nx

    nx1 = nx + 1
    ny1 = ny + 1
    u_l = u[(iz * ny + iy) * nx1 + ix]
    u_r = u[(iz * ny + iy) * nx1 + (ix + 1)]
    v_d = v[(iz * ny1 + iy) * nx + ix]
    v_u = v[(iz * ny1 + (iy + 1)) * nx + ix]
    w_b = w[(iz * ny + iy) * nx + ix]
    w_f = w[((iz + 1) * ny + iy) * nx + ix]
    return (u_r - u_l) + (v_u - v_d) + (w_f - w_b)


@gs.kernel
def mac_pressure_diag(fill: gs.f32["nc"], nx: gs.i32, ny: gs.i32, nz: gs.i32) -> gs.f32["nc"]:
    """Inverse of the diagonal of the pressure system (0: air).

    Neumann walls; zero pressure at the free surface, placed inside the cell (distance `theta`
    to the 0.5 fill level) rather than at the centre of the air cell."""
    cid = gs.index()
    f = fill[cid]
    if f < 0.5:
        return 0.0
    plane = nx * ny
    iz = cid // plane
    rem = cid - iz * plane
    iy = rem // nx
    ix = rem - iy * nx
    s_cnt = 0.0
    for k in range(6):
        nb = cid - 1
        inside = ix > 0
        if k == 1:
            nb = cid + 1
            inside = ix < nx - 1
        elif k == 2:
            nb = cid - nx
            inside = iy > 0
        elif k == 3:
            nb = cid + nx
            inside = iy < ny - 1
        elif k == 4:
            nb = cid - plane
            inside = iz > 0
        elif k == 5:
            nb = cid + plane
            inside = iz < nz - 1
        if inside:
            fj = fill[nb]
            if fj >= 0.5:
                s_cnt += 1.0
            else:
                s_cnt += 1.0 / gs.min(gs.max((f - 0.5) / gs.max(f - fj, 1e-4), 0.25), 1.0)
    return 1.0 / gs.max(s_cnt, 1.0)


@gs.kernel
def mac_pressure_jacobi2(
    p: gs.f32["nc"],
    div: gs.f32["nc"],
    inv_diag: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
) -> gs.f32["nc"]:
    """Two Jacobi iterations in one launch: the second only depends on the six neighbours of the first."""
    cid = gs.index()
    inv_c = inv_diag[cid]
    if inv_c <= 0.0:
        return 0.0
    plane = nx * ny
    iz = cid // plane
    rem = cid - iz * plane
    iy = rem // nx
    ix = rem - iy * nx
    total = 0.0
    for k in range(6):
        jx = ix
        jy = iy
        jz = iz
        if k == 0:
            jx = ix - 1
        elif k == 1:
            jx = ix + 1
        elif k == 2:
            jy = iy - 1
        elif k == 3:
            jy = iy + 1
        elif k == 4:
            jz = iz - 1
        else:
            jz = iz + 1
        if jx >= 0 and jx < nx and jy >= 0 and jy < ny and jz >= 0 and jz < nz:
            c = jz * plane + jy * nx + jx
            inv_j = inv_diag[c]
            if inv_j > 0.0:
                around = 0.0
                if jx > 0 and inv_diag[c - 1] > 0.0:
                    around += p[c - 1]
                if jx < nx - 1 and inv_diag[c + 1] > 0.0:
                    around += p[c + 1]
                if jy > 0 and inv_diag[c - nx] > 0.0:
                    around += p[c - nx]
                if jy < ny - 1 and inv_diag[c + nx] > 0.0:
                    around += p[c + nx]
                if jz > 0 and inv_diag[c - plane] > 0.0:
                    around += p[c - plane]
                if jz < nz - 1 and inv_diag[c + plane] > 0.0:
                    around += p[c + plane]
                total += (around - div[c]) * inv_j
    return (total - div[cid]) * inv_c


@gs.kernel
def mac_project_face(
    vel_in: gs.f32["nf"],
    p: gs.f32["nc"],
    fill: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    axis: gs.i32,
) -> gs.f32["nf"]:
    """Subtracts the pressure gradient on each interior face (same free surface as `mac_pressure_diag`)."""
    idx = gs.index()
    sx = nx + 1 if axis == 0 else nx
    sy = ny + 1 if axis == 1 else ny
    axis_max = nx if axis == 0 else (ny if axis == 1 else nz)

    plane_f = sx * sy
    iz = idx // plane_f
    rem = idx - iz * plane_f
    iy = rem // sx
    ix = rem - iy * sx

    coord = ix
    if axis == 1:
        coord = iy
    elif axis == 2:
        coord = iz

    if coord == 0 or coord == axis_max:
        return 0.0

    plane_c = nx * ny
    c_hi = iz * plane_c + iy * nx + ix
    c_lo = c_hi - 1
    if axis == 1:
        c_lo = c_hi - nx
    elif axis == 2:
        c_lo = c_hi - plane_c

    f_hi = fill[c_hi]
    f_lo = fill[c_lo]
    grad = p[c_hi] - p[c_lo]
    if f_hi < 0.5 and f_lo < 0.5:
        # Isolated drops: free fall
        grad = 0.0
    elif f_hi < 0.5:
        grad = -p[c_lo] / gs.min(gs.max((f_lo - 0.5) / gs.max(f_lo - f_hi, 1e-4), 0.25), 1.0)
    elif f_lo < 0.5:
        grad = p[c_hi] / gs.min(gs.max((f_hi - 0.5) / gs.max(f_hi - f_lo, 1e-4), 0.25), 1.0)
    return vel_in[idx] - grad


@gs.kernel
def mac_volume_source(
    fill: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    density_push: gs.f32,
) -> gs.f32["nc"]:
    """Right-hand side of volume preservation: an overfull cell empties, a hollow cell fills."""
    cid = gs.index()
    f = fill[cid]
    if f < 0.5:
        return 0.0
    plane = nx * ny
    iz = cid // plane
    rem = cid - iz * plane
    iy = rem // nx
    ix = rem - iy * nx
    excess = f - 1.0
    if excess < 0.0:
        # A surface cell is half empty by nature: only cells surrounded by water count
        inner = 1
        if ix > 0 and fill[cid - 1] < 0.8:
            inner = 0
        if ix < nx - 1 and fill[cid + 1] < 0.8:
            inner = 0
        if iy > 0 and fill[cid - nx] < 0.8:
            inner = 0
        if iy < ny - 1 and fill[cid + nx] < 0.8:
            inner = 0
        if iz > 0 and fill[cid - plane] < 0.8:
            inner = 0
        if iz < nz - 1 and fill[cid + plane] < 0.8:
            inner = 0
        excess = gs.max(excess, -0.3) * float(inner)
    return -density_push * gs.min(excess, 1.5)


@gs.kernel
def mac_volume_move(
    pos: gs.f32["n", "c4"],
    q: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    dt: gs.f32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
) -> gs.f32["n", "c4"]:
    """Volume preservation through the positions only (after Kugelstadt et al., "Implicit Density Projection", 2019).

    Going through the velocities would give the water an energy it does not have: it would never calm down."""
    i, c = gs.index()
    if c == 3:
        return pos[i, 3]
    gx = gs.min(gs.max((pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(nx) - 0.5, 0.0), float(nx) - 1.001)
    gy = gs.min(gs.max((pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(ny) - 0.5, 0.0), float(ny) - 1.001)
    gz = gs.min(gs.max((pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(nz) - 0.5, 0.0), float(nz) - 1.001)
    ix = int(gx)
    iy = int(gy)
    iz = int(gz)
    tx = gx - float(ix)
    ty = gy - float(iy)
    tz = gz - float(iz)
    grad = 0.0
    for k in range(8):
        ox = k & 1
        oy = (k >> 1) & 1
        oz = k >> 2
        wx = tx if ox == 1 else 1.0 - tx
        wy = ty if oy == 1 else 1.0 - ty
        wz = tz if oz == 1 else 1.0 - tz
        if c == 0:
            wx = 1.0 if ox == 1 else -1.0
        elif c == 1:
            wy = 1.0 if oy == 1 else -1.0
        else:
            wz = 1.0 if oz == 1 else -1.0
        grad += q[((iz + oz) * ny + iy + oy) * nx + ix + ox] * wx * wy * wz
    cell = 2.0 * bx / float(nx)
    if c == 1:
        cell = by / float(ny)
    elif c == 2:
        cell = 2.0 * bz / float(nz)
    move = gs.min(gs.max(-grad * dt, -0.25 * cell), 0.25 * cell)
    lo = -bx + 0.02
    hi = bx - 0.02
    if c == 1:
        lo = floor_y + 0.02
        hi = floor_y + by - 0.02
    elif c == 2:
        lo = -bz + 0.02
        hi = bz - 0.02
    return gs.min(gs.max(pos[i, c] + move, lo), hi)


@gs.kernel
def mac_shift(
    pos: gs.f32["n", "c4"],
    dens: gs.f32["nv"],
    dx: gs.i32,
    dy: gs.i32,
    dz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    core: gs.f32,
    strength: gs.f32,
    spacing: gs.f32,
) -> gs.f32["n", "c4"]:
    """Redistributes the particles in the core of the water: a small move from dense areas toward hollow ones.

    Zero near the surface (density below 75 % of the core), where the gradient is that of the surface itself."""
    i, c = gs.index()
    val = pos[i, c]
    if c == 3:
        return val
    gx = gs.min(gs.max((pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(dx - 1), 0.0), float(dx) - 1.001)
    gy = gs.min(gs.max((pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(dy - 1), 0.0), float(dy) - 1.001)
    gz = gs.min(gs.max((pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(dz - 1), 0.0), float(dz) - 1.001)
    ix = int(gx)
    iy = int(gy)
    iz = int(gz)
    tx = gx - float(ix)
    ty = gy - float(iy)
    tz = gz - float(iz)
    d = 0.0
    grad_x = 0.0
    grad_y = 0.0
    grad_z = 0.0
    for k in range(8):
        ox = k & 1
        oy = (k >> 1) & 1
        oz = k >> 2
        v = dens[((iz + oz) * dy + iy + oy) * dx + ix + ox]
        wx = tx if ox == 1 else 1.0 - tx
        wy = ty if oy == 1 else 1.0 - ty
        wz = tz if oz == 1 else 1.0 - tz
        d += v * wx * wy * wz
        grad_x += v * (1.0 if ox == 1 else -1.0) * wy * wz
        grad_y += v * (1.0 if oy == 1 else -1.0) * wx * wz
        grad_z += v * (1.0 if oz == 1 else -1.0) * wx * wy
    fade = gs.min(gs.max((d / core - 0.75) / 0.15, 0.0), 1.0)
    grad = grad_x * float(dx - 1) / (2.0 * bx)
    if c == 1:
        grad = grad_y * float(dy - 1) / by
    elif c == 2:
        grad = grad_z * float(dz - 1) / (2.0 * bz)
    move = -strength * spacing * spacing * grad / core * fade
    return val + gs.min(gs.max(move, -0.1 * spacing), 0.1 * spacing)


@gs.kernel
def mac_g2p(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    u_new: gs.f32["nu"],
    v_new: gs.f32["nv"],
    w_new: gs.f32["nw"],
    u_old: gs.f32["nu"],
    v_old: gs.f32["nv"],
    w_old: gs.f32["nw"],
    cell_cnt: gs.u32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    dt: gs.f32,
    flip_ratio: gs.f32,
    damping: gs.f32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    restitution: gs.f32,
    max_speed: gs.f32,
) -> (gs.f32["n", "c4"], gs.f32["n", "c4"]):
    """Exact trilinear MAC G2P (PIC/FLIP) + turbulence energy (`vel[i, 3]`) for *whitewater* emission."""
    i, c = gs.index()
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    prad = pos[i, 3]

    gx = gs.min(gs.max((px + bx) / gs.max(2.0 * bx, 1e-4) * float(nx), 0.01), float(nx) - 0.01)
    gy = gs.min(gs.max((py - floor_y) / gs.max(by, 1e-4) * float(ny), 0.01), float(ny) - 0.01)
    gz = gs.min(gs.max((pz + bz) / gs.max(2.0 * bz, 1e-4) * float(nz), 0.01), float(nz) - 0.01)

    # 1) Trilinear interpolation of U
    ux_f = gs.min(gs.max(gx, 0.001), float(nx) - 0.001)
    uy_f = gs.min(gs.max(gy - 0.5, 0.001), float(ny - 1) - 0.001)
    uz_f = gs.min(gs.max(gz - 0.5, 0.001), float(nz - 1) - 0.001)
    ux0 = int(gs.floor(ux_f))
    uy0 = int(gs.floor(uy_f))
    uz0 = int(gs.floor(uz_f))
    utx = ux_f - float(ux0)
    uty = uy_f - float(uy0)
    utz = uz_f - float(uz0)
    nx1 = nx + 1
    u_sz = nx1 * ny
    pic_x = 0.0
    dvx = 0.0
    for dz in range(2):
        wz = utz if dz == 1 else (1.0 - utz)
        for dy in range(2):
            wy = uty if dy == 1 else (1.0 - uty)
            for dx in range(2):
                wx = utx if dx == 1 else (1.0 - utx)
                wt = wx * wy * wz
                idx = (uz0 + dz) * u_sz + (uy0 + dy) * nx1 + (ux0 + dx)
                un = u_new[idx]
                pic_x += un * wt
                dvx += (un - u_old[idx]) * wt

    # 2) Trilinear interpolation of V
    vx_f = gs.min(gs.max(gx - 0.5, 0.001), float(nx - 1) - 0.001)
    vy_f = gs.min(gs.max(gy, 0.001), float(ny) - 0.001)
    vz_f = gs.min(gs.max(gz - 0.5, 0.001), float(nz - 1) - 0.001)
    vx0 = int(gs.floor(vx_f))
    vy0 = int(gs.floor(vy_f))
    vz0 = int(gs.floor(vz_f))
    vtx = vx_f - float(vx0)
    vty = vy_f - float(vy0)
    vtz = vz_f - float(vz0)
    ny1 = ny + 1
    v_sz = nx * ny1
    pic_y = 0.0
    dvy = 0.0
    for dz in range(2):
        wz = vtz if dz == 1 else (1.0 - vtz)
        for dy in range(2):
            wy = vty if dy == 1 else (1.0 - vty)
            for dx in range(2):
                wx = vtx if dx == 1 else (1.0 - vtx)
                wt = wx * wy * wz
                idx = (vz0 + dz) * v_sz + (vy0 + dy) * nx + (vx0 + dx)
                vn = v_new[idx]
                pic_y += vn * wt
                dvy += (vn - v_old[idx]) * wt

    # 3) Trilinear interpolation of W
    wx_f = gs.min(gs.max(gx - 0.5, 0.001), float(nx - 1) - 0.001)
    wy_f = gs.min(gs.max(gy - 0.5, 0.001), float(ny - 1) - 0.001)
    wz_f = gs.min(gs.max(gz, 0.001), float(nz) - 0.001)
    wx0 = int(gs.floor(wx_f))
    wy0 = int(gs.floor(wy_f))
    wz0 = int(gs.floor(wz_f))
    wtx = wx_f - float(wx0)
    wty = wy_f - float(wy0)
    wtz = wz_f - float(wz0)
    w_sz = nx * ny
    pic_z = 0.0
    dvz = 0.0
    for dz in range(2):
        wz = wtz if dz == 1 else (1.0 - wtz)
        for dy in range(2):
            wy = wty if dy == 1 else (1.0 - wty)
            for dx in range(2):
                wx = wtx if dx == 1 else (1.0 - wtx)
                wt = wx * wy * wz
                idx = (wz0 + dz) * w_sz + (wy0 + dy) * nx + (wx0 + dx)
                wn = w_new[idx]
                pic_z += wn * wt
                dvz += (wn - w_old[idx]) * wt

    vx = ((1.0 - flip_ratio) * pic_x + flip_ratio * (vel[i, 0] + dvx)) * damping
    vy = ((1.0 - flip_ratio) * pic_y + flip_ratio * (vel[i, 1] + dvy)) * damping
    vz = ((1.0 - flip_ratio) * pic_z + flip_ratio * (vel[i, 2] + dvz)) * damping

    sp2 = vx * vx + vy * vy + vz * vz
    sp = gs.sqrt(sp2)
    if sp > max_speed:
        sc = max_speed / sp
        vx = vx * sc
        vy = vy * sc
        vz = vz * sc
        sp = max_speed

    # Turbulence energy / trapped air (feeds the spray / foam / bubble whitewater micro-particles)
    cx = gs.min(gs.max(int(gs.floor(gx)), 0), nx - 1)
    cy = gs.min(gs.max(int(gs.floor(gy)), 0), ny - 1)
    cz = gs.min(gs.max(int(gs.floor(gz)), 0), nz - 1)
    local_cnt = float(cell_cnt[(cz * ny + cy) * nx + cx])
    shear = gs.sqrt((vx - pic_x) * (vx - pic_x) + (vy - pic_y) * (vy - pic_y) + (vz - pic_z) * (vz - pic_z))
    turb = vel[i, 3] * 0.88
    if sp > 2.6:
        air_f = gs.min(gs.max(1.0 - local_cnt * 0.14, 0.15), 1.0)
        turb = gs.min(turb + (sp - 2.6) * 0.08 * air_f + shear * 0.12, 1.0)

    px += vx * dt
    py += vy * dt
    pz += vz * dt

    margin = gs.max(prad * 0.35, 0.02)
    min_y = floor_y + margin
    max_y = floor_y + by - margin
    if py < min_y:
        py = min_y
        if vy < 0.0:
            vy = -vy * restitution
        vx = vx * 0.992
        vz = vz * 0.992
    elif py > max_y:
        py = max_y
        if vy > 0.0:
            vy = -vy * restitution

    min_x = -bx + margin
    max_x = bx - margin
    if px < min_x:
        px = min_x
        if vx < 0.0:
            vx = -vx * restitution
    elif px > max_x:
        px = max_x
        if vx > 0.0:
            vx = -vx * restitution

    min_z = -bz + margin
    max_z = bz - margin
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
        out_v = turb
    return out_p, out_v


# ───────────────────────── 2. Micro-Particules Whitewater GPU (Spray / Foam / Bubbles) ─────────────────────────


@gs.kernel(grid=("n",))
def fluid_velocity_splat(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    cell_tpl: gs.f32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    mass: gs.Out[gs.u32["nc"], 0],
    mom_x: gs.Out[gs.u32["nc"], 0],
    mom_y: gs.Out[gs.u32["nc"], 0],
    mom_z: gs.Out[gs.u32["nc"], 0],
):
    """Local fluid velocity: mass and momentum accumulated at the cell centres."""
    i = gs.index()
    _z = cell_tpl[0] * 0.0
    gx = gs.min(gs.max((pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(nx) - 0.5 + _z, 0.0), float(nx) - 1.001)
    gy = gs.min(gs.max((pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(ny) - 0.5, 0.0), float(ny) - 1.001)
    gz = gs.min(gs.max((pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(nz) - 0.5, 0.0), float(nz) - 1.001)
    ix = int(gx)
    iy = int(gy)
    iz = int(gz)
    tx = gx - float(ix)
    ty = gy - float(iy)
    tz = gz - float(iz)
    vx = gs.min(gs.max(vel[i, 0], -18.0), 18.0) + 20.0
    vy = gs.min(gs.max(vel[i, 1], -18.0), 18.0) + 20.0
    vz = gs.min(gs.max(vel[i, 2], -18.0), 18.0) + 20.0
    for k in range(8):
        ox = k & 1
        oy = (k >> 1) & 1
        oz = k >> 2
        w = (tx if ox == 1 else 1.0 - tx) * (ty if oy == 1 else 1.0 - ty) * (tz if oz == 1 else 1.0 - tz)
        q = gs.u32(w * 256.0 + 0.5)
        if q > gs.u32(0):
            cid = ((iz + oz) * ny + iy + oy) * nx + ix + ox
            qf = float(q)
            gs.atomic_add(mass, cid, q)
            gs.atomic_add(mom_x, cid, gs.u32(qf * vx + 0.5))
            gs.atomic_add(mom_y, cid, gs.u32(qf * vy + 0.5))
            gs.atomic_add(mom_z, cid, gs.u32(qf * vz + 0.5))


@gs.kernel
def fluid_velocity_resolve(
    mass: gs.u32["nc"],
    mom_x: gs.u32["nc"],
    mom_y: gs.u32["nc"],
    mom_z: gs.u32["nc"],
    c4_tpl: gs.f32["c4"],
) -> gs.f32["nc", "c4"]:
    """`[vx, vy, vz, mass]` per cell (mass in number of particles)."""
    cid, c = gs.index()
    m = mass[cid]
    out = float(m) / 256.0 + c4_tpl[0] * 0.0
    if c < 3:
        out = 0.0
        if m > gs.u32(0):
            mom = mom_x[cid]
            if c == 1:
                mom = mom_y[cid]
            elif c == 2:
                mom = mom_z[cid]
            out = float(mom) / float(m) - 20.0
    return out


@gs.kernel
def mac_cell_velocity(
    u: gs.f32["nu"],
    v: gs.f32["nv"],
    w: gs.f32["nw"],
    fill: gs.f32["nc"],
    c4_tpl: gs.f32["c4"],
    rest_cnt: gs.f32,
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
) -> gs.f32["nc", "c4"]:
    """`[vx, vy, vz, mass]` per cell, read on the step's MAC faces (same content as `fluid_velocity_resolve`)."""
    cid, c = gs.index()
    if c == 3:
        return fill[cid] * rest_cnt + c4_tpl[0] * 0.0
    plane = nx * ny
    iz = cid // plane
    rem = cid - iz * plane
    iy = rem // nx
    ix = rem - iy * nx
    if c == 0:
        f0 = (iz * ny + iy) * (nx + 1) + ix
        return 0.5 * (u[f0] + u[f0 + 1])
    if c == 1:
        f1 = (iz * (ny + 1) + iy) * nx + ix
        return 0.5 * (v[f1] + v[f1 + nx])
    return 0.5 * (w[cid] + w[cid + plane])


@gs.kernel
def ww_step(
    ww_pos: gs.f32["nw", "c4"],  # [x, y, z, vie]
    ww_vel: gs.f32["nw", "c4"],  # [vx, vy, vz, state: 0 = bubble, 1 = foam, 2 = spray]
    fl_pos: gs.f32["nf", "c4"],
    fl_vel: gs.f32["nf", "c4"],
    cells: gs.f32["nc", "c4"],
    dens: gs.f32["nv"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    dx: gs.i32,
    dy: gs.i32,
    dz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    iso_level: gs.f32,
    rest_mass: gs.f32,
    gravity: gs.f32,
    dt: gs.f32,
    frame_seed: gs.i32,
    rate: gs.f32,
) -> (gs.f32["nw", "c4"], gs.f32["nw", "c4"]):
    """Spray, foam and bubbles, after Ihmsen et al., "Unified Spray, Foam and Bubbles" (CGI 2012).

    Birth according to three potentials (trapped air, wave crest, kinetic energy); spray follows a
    ballistic trajectory, foam is carried by the local water velocity, a bubble adds buoyancy
    and drag. Two departures from the paper: the sums over neighbours use the 26 neighbouring cells,
    and each free slot draws a fluid particle at random (the total number stays bounded)."""
    i, c = gs.index()
    nf = fl_pos.shape[0]

    px = ww_pos[i, 0]
    py = ww_pos[i, 1]
    pz = ww_pos[i, 2]
    life = ww_pos[i, 3]
    vx = ww_vel[i, 0]
    vy = ww_vel[i, 1]
    vz = ww_vel[i, 2]
    kind = ww_vel[i, 3]
    csx = 2.0 * bx / float(nx)
    csy = by / float(ny)
    csz = 2.0 * bz / float(nz)

    if life <= 0.0:
        # Pseudo-random sequence specific to the slot and the step
        h = gs.u32(i) * gs.u32(747796405) + gs.u32(frame_seed) * gs.u32(2891336453)
        h = (h ^ (h >> gs.u32(16))) * gs.u32(2246822519)
        h = (h ^ (h >> gs.u32(13))) * gs.u32(3266489917)
        h = h ^ (h >> gs.u32(16))
        src = int(h % gs.u32(nf))
        h2 = (h ^ (h >> gs.u32(15))) * gs.u32(668265263)
        r_accept = float((h2 >> gs.u32(8)) & gs.u32(65535)) / 65536.0
        h3 = (h2 ^ (h2 >> gs.u32(14))) * gs.u32(374761393)
        r_r = float((h3 >> gs.u32(4)) & gs.u32(1023)) / 1023.0
        r_t = float((h3 >> gs.u32(14)) & gs.u32(1023)) / 1023.0
        r_h = float((h2 >> gs.u32(2)) & gs.u32(1023)) / 1023.0

        fx = fl_pos[src, 0]
        fy = fl_pos[src, 1]
        fz = fl_pos[src, 2]
        ux = fl_vel[src, 0]
        uy = fl_vel[src, 1]
        uz = fl_vel[src, 2]
        sp2 = ux * ux + uy * uy + uz * uz
        # Kinetic energy (mass 1): Phi(0.5 v², 1.5, 10)
        i_k = gs.min(gs.max((0.5 * sp2 - 1.5) / 8.5, 0.0), 1.0)
        if i_k > 0.0:
            speed = gs.sqrt(sp2)
            ix = gs.min(gs.max(int((fx + bx) / csx), 1), nx - 2)
            iy = gs.min(gs.max(int((fy - floor_y) / csy), 1), ny - 2)
            iz = gs.min(gs.max(int((fz + bz) / csz), 1), nz - 2)
            base = (iz * ny + iy) * nx + ix
            plane_c = nx * ny
            # Normal: opposite to the gradient of the mass
            nnx = cells[base - 1, 3] - cells[base + 1, 3]
            nny = cells[base - nx, 3] - cells[base + nx, 3]
            nnz = cells[base - plane_c, 3] - cells[base + plane_c, 3]
            n_len = gs.sqrt(nnx * nnx + nny * nny + nnz * nnz)
            inv_nl = 1.0 / gs.max(n_len, 1e-4)
            nnx = nnx * inv_nl
            nny = nny * inv_nl
            nnz = nnz * inv_nl
            crest_ok = 0
            if n_len > 0.5 * rest_mass and (ux * nnx + uy * nny + uz * nnz) / gs.max(speed, 1e-4) >= 0.6:
                crest_ok = 1
            v_diff = 0.0
            kappa = 0.0
            inv_h = 1.0 / (2.0 * gs.max(csx, csy, csz))
            for k in range(27):
                ox = k % 3 - 1
                oy = (k // 3) % 3 - 1
                oz = k // 9 - 1
                nid = base + oz * plane_c + oy * nx + ox
                mj = cells[nid, 3]
                if k != 13 and mj > 0.05:
                    # Direction from the neighbour to the particle
                    ex = -float(ox) * csx
                    ey = -float(oy) * csy
                    ez = -float(oz) * csz
                    e_len = gs.sqrt(ex * ex + ey * ey + ez * ez)
                    ex = ex / e_len
                    ey = ey / e_len
                    ez = ez / e_len
                    wj = gs.max(1.0 - e_len * inv_h, 0.0) * mj
                    rx = ux - cells[nid, 0]
                    ry = uy - cells[nid, 1]
                    rz = uz - cells[nid, 2]
                    r_len = gs.sqrt(rx * rx + ry * ry + rz * rz)
                    # |v_ij| (1 - v̂·x̂): zero if they move apart, double if they come closer
                    v_diff += (r_len - (rx * ex + ry * ey + rz * ez)) * wj
                    if crest_ok != 0 and -(ex * nnx + ey * nny + ez * nnz) < 0.0:
                        jx = gs.min(gs.max(ix + ox, 1), nx - 2)
                        jy = gs.min(gs.max(iy + oy, 1), ny - 2)
                        jz = gs.min(gs.max(iz + oz, 1), nz - 2)
                        jb = (jz * ny + jy) * nx + jx
                        mx = cells[jb - 1, 3] - cells[jb + 1, 3]
                        my = cells[jb - nx, 3] - cells[jb + nx, 3]
                        mz = cells[jb - plane_c, 3] - cells[jb + plane_c, 3]
                        m_len = gs.sqrt(mx * mx + my * my + mz * mz)
                        if m_len > 0.5 * rest_mass:
                            kappa += (1.0 - (nnx * mx + nny * my + nnz * mz) / m_len) * wj
            inv_m = 1.0 / gs.max(rest_mass, 0.1)
            i_ta = gs.min(gs.max((v_diff * inv_m - 14.0) / 36.0, 0.0), 1.0)
            i_wc = gs.min(gs.max((kappa * inv_m - 0.6) / 2.4, 0.0), 1.0)
            n_d = i_k * (i_ta + i_wc) * rate * dt * float(nf) / float(ww_pos.shape[0])
            if r_accept < n_d:
                # Cylinder around the particle's trajectory during the step
                inv_s = 1.0 / gs.max(speed, 1e-4)
                dxu = ux * inv_s
                dyu = uy * inv_s
                dzu = uz * inv_s
                e1x = 0.0
                e1y = dzu
                e1z = -dyu
                if gs.abs(dxu) > 0.8:
                    e1x = -dzu
                    e1y = 0.0
                    e1z = dxu
                inv_e = 1.0 / gs.max(gs.sqrt(e1x * e1x + e1y * e1y + e1z * e1z), 1e-4)
                e1x = e1x * inv_e
                e1y = e1y * inv_e
                e1z = e1z * inv_e
                e2x = dyu * e1z - dzu * e1y
                e2y = dzu * e1x - dxu * e1z
                e2z = dxu * e1y - dyu * e1x
                rad = 0.5 * csx * gs.sqrt(r_r)
                ca = gs.cos(r_t * 6.2831853) * rad
                sa = gs.sin(r_t * 6.2831853) * rad
                along = r_h * speed * dt
                offx = ca * e1x + sa * e2x
                offy = ca * e1y + sa * e2y
                offz = ca * e1z + sa * e2z
                px = gs.min(gs.max(fx + offx + along * dxu, -bx + 0.03), bx - 0.03)
                py = gs.max(fy + offy + along * dyu, floor_y + 0.03)
                pz = gs.min(gs.max(fz + offz + along * dzu, -bz + 0.03), bz - 0.03)
                # The paper takes the deviation as a velocity; amplified, since the radius is a few centimetres here
                vx = ux + offx * 4.0
                vy = uy + offy * 4.0
                vz = uz + offz * 4.0
                # Foam clusters last longer: lifetime tied to the potentials
                life = 1.2 + 3.3 * gs.min(i_k * (i_ta + i_wc), 1.0)
                kind = 1.0
    else:
        # Local density (the paper's neighbour count: < 6 spray, > 20 bubble, for ~33 in the core)
        gx = gs.min(gs.max((px + bx) / gs.max(2.0 * bx, 1e-4) * float(dx - 1), 0.0), float(dx) - 1.001)
        gy = gs.min(gs.max((py - floor_y) / gs.max(by, 1e-4) * float(dy - 1), 0.0), float(dy) - 1.001)
        gz = gs.min(gs.max((pz + bz) / gs.max(2.0 * bz, 1e-4) * float(dz - 1), 0.0), float(dz) - 1.001)
        bxi = int(gx)
        byi = int(gy)
        bzi = int(gz)
        ax = gx - float(bxi)
        ay = gy - float(byi)
        az = gz - float(bzi)
        b0 = (bzi * dy + byi) * dx + bxi
        b1 = b0 + dx * dy
        d_lo = (dens[b0] * (1.0 - ax) + dens[b0 + 1] * ax) * (1.0 - ay) + (dens[b0 + dx] * (1.0 - ax) + dens[b0 + dx + 1] * ax) * ay
        d_hi = (dens[b1] * (1.0 - ax) + dens[b1 + 1] * ax) * (1.0 - ay) + (dens[b1 + dx] * (1.0 - ax) + dens[b1 + dx + 1] * ax) * ay
        d_loc = d_lo * (1.0 - az) + d_hi * az

        # Local water velocity, weighted by mass
        lx = gs.min(gs.max((px + bx) / csx - 0.5, 0.0), float(nx) - 1.001)
        ly = gs.min(gs.max((py - floor_y) / csy - 0.5, 0.0), float(ny) - 1.001)
        lz = gs.min(gs.max((pz + bz) / csz - 0.5, 0.0), float(nz) - 1.001)
        cx0 = int(lx)
        cy0 = int(ly)
        cz0 = int(lz)
        tx = lx - float(cx0)
        ty = ly - float(cy0)
        tz = lz - float(cz0)
        wx_f = 0.0
        wy_f = 0.0
        wz_f = 0.0
        wm = 0.0
        for k in range(8):
            ox = k & 1
            oy = (k >> 1) & 1
            oz = k >> 2
            cid = ((cz0 + oz) * ny + cy0 + oy) * nx + cx0 + ox
            wk = (tx if ox == 1 else 1.0 - tx) * (ty if oy == 1 else 1.0 - ty) * (tz if oz == 1 else 1.0 - tz) * cells[cid, 3]
            wx_f += cells[cid, 0] * wk
            wy_f += cells[cid, 1] * wk
            wz_f += cells[cid, 2] * wk
            wm += wk
        inv_wm = 1.0 / gs.max(wm, 1e-4)
        wx_f = wx_f * inv_wm
        wy_f = wy_f * inv_wm
        wz_f = wz_f * inv_wm

        if d_loc < iso_level * 0.30 or wm < 0.02:
            kind = 2.0
            vy -= gravity * dt
            px += vx * dt
            py += vy * dt
            pz += vz * dt
            # Addition to the paper: slow countdown, to free the slot some day
            life -= dt * 0.25
        elif d_loc < iso_level * 0.95:
            # Foam: position carried by the water, own velocity unchanged
            kind = 1.0
            px += wx_f * dt
            py += wy_f * dt
            pz += wz_f * dt
            life -= dt
        else:
            # Bubble: buoyancy k_b = 2, drag k_d = 0.8
            kind = 0.0
            vx += 0.8 * (wx_f - vx)
            vy += 2.0 * gravity * dt + 0.8 * (wy_f - vy)
            vz += 0.8 * (wz_f - vz)
            px += vx * dt
            py += vy * dt
            pz += vz * dt
            life -= dt * 0.25

        if py < floor_y + 0.03:
            py = floor_y + 0.03
            vy = gs.abs(vy) * 0.2
        if px < -bx + 0.03:
            px = -bx + 0.03
            vx = gs.abs(vx) * 0.25
        elif px > bx - 0.03:
            px = bx - 0.03
            vx = -gs.abs(vx) * 0.25
        if pz < -bz + 0.03:
            pz = -bz + 0.03
            vz = gs.abs(vz) * 0.25
        elif pz > bz - 0.03:
            pz = bz - 0.03
            vz = -gs.abs(vz) * 0.25

    out_p = px
    out_v = vx
    if c == 1:
        out_p = py
        out_v = vy
    elif c == 2:
        out_p = pz
        out_v = vz
    elif c == 3:
        out_p = life
        out_v = kind
    return out_p, out_v


@gs.kernel(grid=("nw",))
def ww_splat_screen(
    ww_pos: gs.f32["nw", "c4"],
    ww_vel: gs.f32["nw", "c4"],
    zbuf: gs.u32["h", "w"],
    vp: gs.f32["c16"],
    near: gs.f32,
    far: gs.f32,
    ww_alpha: gs.Out[gs.u32["h", "w"], 0],
    ww_depth: gs.Out[gs.u32["h", "w"], 4294967295],
):
    """Projects the 6,000 whitewater micro-particles (spray / foam / bubbles) as 1–2 px sparkling points."""
    i = gs.index()
    life = ww_pos[i, 3]
    if life <= 0.02:
        return
    wx = ww_pos[i, 0]
    wy = ww_pos[i, 1]
    wz = ww_pos[i, 2]
    cx = vp[0] * wx + vp[1] * wy + vp[2] * wz + vp[3]
    cy = vp[4] * wx + vp[5] * wy + vp[6] * wz + vp[7]
    cw = vp[12] * wx + vp[13] * wy + vp[14] * wz + vp[15]
    if cw <= near or cw >= far:
        return

    h = zbuf.shape[0]
    w = zbuf.shape[1]
    inv_w = 1.0 / cw
    px = int((cx * inv_w * 0.5 + 0.5) * float(w))
    py = int((0.5 - cy * inv_w * 0.5) * float(h))
    if px < 1 or px >= w - 1 or py < 1 or py >= h - 1:
        return

    depth_range = gs.max(far - near, 1e-4)
    sz = zbuf[py, px]
    scene_z = far + 10.0
    if sz != gs.u32(4294967295):
        scene_z = near + (float(sz) / 16777215.0) * depth_range
    if cw > scene_z + 0.03:
        return

    kind = ww_vel[i, 3]
    fade = gs.min(life * 1.4, 1.0)
    # Intensity: bright spray, vivid foam, soft bubbles
    base_a = 140.0 if kind > 1.5 else (175.0 if kind > 0.5 else 95.0)
    a_c = gs.u32(base_a * fade)
    a_n = gs.u32(base_a * 0.35 * fade)
    d_u32 = gs.u32(gs.min(gs.max((cw - near) / depth_range, 0.0), 1.0) * 4000000000.0)

    gs.atomic_add(ww_alpha, (py, px), a_c)
    gs.atomic_min(ww_depth, (py, px), d_u32)
    # 1 px cross micro-halo for a soft sparkle without hard pixels
    gs.atomic_add(ww_alpha, (py, px + 1), a_n)
    gs.atomic_add(ww_alpha, (py + 1, px), a_n)


# ───────────────────────── 3. Continuous high-resolution 3D density volume (96×60×52) ─────────────────────────


@gs.kernel(grid=("n",))
def vol_splat_density(
    pos: gs.f32["n", "c4"],
    vol_tpl: gs.f32["nv"],
    dx: gs.i32,
    dy: gs.i32,
    dz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    dens_u: gs.Out[gs.u32["nv"], 0],
):
    """Projects every particle onto a high-resolution 3D scalar field (3×3×3 cubic kernel)."""
    i = gs.index()
    _z = vol_tpl[0] * 0.0
    fx = (pos[i, 0] + bx) / gs.max(2.0 * bx, 1e-4) * float(dx - 1) + _z
    fy = (pos[i, 1] - floor_y) / gs.max(by, 1e-4) * float(dy - 1)
    fz = (pos[i, 2] + bz) / gs.max(2.0 * bz, 1e-4) * float(dz - 1)

    cx = gs.min(gs.max(int(gs.floor(fx + 0.5)), 0), dx - 1)
    cy = gs.min(gs.max(int(gs.floor(fy + 0.5)), 0), dy - 1)
    cz = gs.min(gs.max(int(gs.floor(fz + 0.5)), 0), dz - 1)
    plane = dx * dy

    for oz in range(-1, 2):
        iz = cz + oz
        if iz >= 0 and iz < dz:
            wz = gs.max(1.0 - gs.abs(float(iz) - fz) * 0.65, 0.0)
            # Node on a wall: half the kernel is outside the basin, counted by symmetry
            wz2 = wz * wz * (2.0 if iz == 0 or iz == dz - 1 else 1.0)
            for oy in range(-1, 2):
                iy = cy + oy
                if iy >= 0 and iy < dy:
                    wy = gs.max(1.0 - gs.abs(float(iy) - fy) * 0.65, 0.0)
                    wyz = wy * wy * wz2 * (2.0 if iy == 0 or iy == dy - 1 else 1.0)
                    for ox in range(-1, 2):
                        ix = cx + ox
                        if ix >= 0 and ix < dx:
                            wx = gs.max(1.0 - gs.abs(float(ix) - fx) * 0.65, 0.0)
                            w = wx * wx * wyz * (2.0 if ix == 0 or ix == dx - 1 else 1.0) * 512.0
                            if w > 0.5:
                                vid = iz * plane + iy * dx + ix
                                gs.atomic_add(dens_u, vid, gs.u32(w))


@gs.kernel
def vol_unpack_f32(
    dens_u: gs.u32["nv"],
    dens_scale: gs.f32,
) -> gs.f32["nv"]:
    """Converts the atomic `u32` volume into a normalized `f32` scalar field."""
    vid = gs.index()
    d = float(dens_u[vid]) / 512.0 * dens_scale
    return gs.min(d, 4.0)


@gs.kernel
def vol_blur_axis(
    dens: gs.f32["nv"],
    dx: gs.i32,
    dy: gs.i32,
    dz: gs.i32,
    axis: gs.i32,
) -> gs.f32["nv"]:
    """Separable 3D Gaussian filter `[1, 4, 6, 4, 1]/16` along X, Y or Z."""
    vid = gs.index()
    plane = dx * dy
    iz = vid // plane
    rem = vid - iz * plane
    iy = rem // dx
    ix = rem - iy * dx

    stride = 1
    coord = ix
    lim = dx - 1
    if axis == 1:
        stride = dx
        coord = iy
        lim = dy - 1
    elif axis == 2:
        stride = plane
        coord = iz
        lim = dz - 1

    m2 = gs.max(coord - 2, 0) - coord
    m1 = gs.max(coord - 1, 0) - coord
    p1 = gs.min(coord + 1, lim) - coord
    p2 = gs.min(coord + 2, lim) - coord

    i_m2 = vid + m2 * stride
    i_m1 = vid + m1 * stride
    i_p1 = vid + p1 * stride
    i_p2 = vid + p2 * stride

    nd = (dens[i_m2] + 4.0 * dens[i_m1] + 6.0 * dens[vid] + 4.0 * dens[i_p1] + dens[i_p2]) * 0.0625
    return nd


@gs.kernel
def vol_wet_walls(dens: gs.f32["nv"], dx: gs.i32, dy: gs.i32, dz: gs.i32) -> gs.f32["nv"]:
    """The water touches its walls: an edge node (glass and bottom) is at least as dense as its inner neighbour."""
    vid = gs.index()
    plane = dx * dy
    iz = vid // plane
    rem = vid - iz * plane
    iy = rem // dx
    ix = rem - iy * dx
    jx = gs.min(gs.max(ix, 1), dx - 2)
    jy = gs.max(iy, 1)
    jz = gs.min(gs.max(iz, 1), dz - 2)
    return gs.max(dens[vid], dens[jz * plane + jy * dx + jx])


# ───────────────────────── 3 bis. Caustics: sun rays refracted, then accumulated on the bottom ─────────────────────────


@gs.kernel
def water_height(
    dens: gs.f32["nv"],
    tpl: gs.f32["dz", "dx"],
    dy: gs.i32,
    iso_level: gs.f32,
) -> gs.f32["dz", "dx"]:
    """Water height of each column (in voxels, 0 = dry): first exit from the liquid going up from the bottom."""
    iz, ix = gs.index()
    dx = tpl.shape[1]
    base = iz * dx * dy + ix
    prev = dens[base + 2 * dx]
    height = 0.0
    if prev >= iso_level:
        height = float(dy - 1)
        for k in range(3, 512):
            if k >= dy:
                break
            cur = dens[base + k * dx]
            if cur < iso_level:
                height = float(k - 1) + (prev - iso_level) / gs.max(prev - cur, 1e-5)
                break
            prev = cur
    return height


@gs.kernel
def height_blur(src: gs.f32["dz", "dx"], axis: gs.i32) -> gs.f32["dz", "dx"]:
    """`[1, 4, 6, 4, 1]` smoothing of the height; a dry column does not count in the mean."""
    iz, ix = gs.index()
    dz = src.shape[0]
    dx = src.shape[1]
    center = src[iz, ix]
    total = 0.0
    weight = 0.0
    for k in range(-2, 3):
        jx = ix
        jz = iz
        if axis == 0:
            jx = gs.min(gs.max(ix + k, 0), dx - 1)
        else:
            jz = gs.min(gs.max(iz + k, 0), dz - 1)
        v = src[jz, jx]
        wk = 6.0
        if k == -2 or k == 2:
            wk = 1.0
        elif k != 0:
            wk = 4.0
        if v > 0.0:
            total += v * wk
            weight += wk
    if center <= 0.0 or weight <= 0.0:
        return 0.0
    return total / weight


@gs.kernel(grid=("pz", "px"))
def caustic_photons(
    height: gs.f32["dz", "dx"],
    grid_tpl: gs.f32["pz", "px"],
    map_tpl: gs.f32["mz", "mx"],
    dy: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    sun_x: gs.f32,
    sun_y: gs.f32,
    sun_z: gs.f32,
    ior: gs.f32,
    photons: gs.Out[gs.u32["mz", "mx"], 0],
):
    """One photon per grid point: refracted by the surface slope, deposited bilinearly on the bottom.

    The grid extends past the basin (height extended): otherwise an oblique sun would leave an unlit edge."""
    jz, jx = gs.index()
    pz = grid_tpl.shape[0]
    px = grid_tpl.shape[1]
    dz = height.shape[0]
    dx = height.shape[1]
    mz = map_tpl.shape[0]
    mx = map_tpl.shape[1]
    # Position on the basin widened by 50 % (25 % on each side), as a fraction of the width
    u = (float(jx) + 0.5) / float(px) * 1.5 - 0.25
    v = (float(jz) + 0.5) / float(pz) * 1.5 - 0.25
    # Height read at least three voxels from the walls, where the water keeps rising and falling
    gx = gs.min(gs.max(u * float(dx - 1), 3.0), float(dx - 4))
    gz = gs.min(gs.max(v * float(dz - 1), 3.0), float(dz - 4))
    ix = gs.min(int(gx), dx - 2)
    iz = gs.min(int(gz), dz - 2)
    fx = gx - float(ix)
    fz = gz - float(iz)
    h00 = height[iz, ix]
    h10 = height[iz, ix + 1]
    h01 = height[iz + 1, ix]
    h11 = height[iz + 1, ix + 1]
    cell_y = by / float(dy - 1)
    h = ((h00 * (1.0 - fx) + h10 * fx) * (1.0 - fz) + (h01 * (1.0 - fx) + h11 * fx) * fz) * cell_y
    land_x = (u * 2.0 - 1.0) * bx
    land_z = (v * 2.0 - 1.0) * bz
    wet = 1
    if h00 <= 0.0 or h10 <= 0.0 or h01 <= 0.0 or h11 <= 0.0:
        wet = 0
    if wet != 0:
        # Slopes smoothed between the differences of the four neighbours (continuous from one cell to the next)
        xm = gs.max(ix - 1, 0)
        xp = gs.min(ix + 2, dx - 1)
        zm = gs.max(iz - 1, 0)
        zp = gs.min(iz + 2, dz - 1)
        sl_x0 = (h10 - (height[iz, xm] if height[iz, xm] > 0.0 else h00)) * 0.5
        sl_x1 = ((height[iz, xp] if height[iz, xp] > 0.0 else h10) - h00) * 0.5
        sl_z0 = (h01 - (height[zm, ix] if height[zm, ix] > 0.0 else h00)) * 0.5
        sl_z1 = ((height[zp, ix] if height[zp, ix] > 0.0 else h01) - h00) * 0.5
        slope_x = (sl_x0 * (1.0 - fx) + sl_x1 * fx) * cell_y * float(dx - 1) / (2.0 * bx)
        slope_z = (sl_z0 * (1.0 - fz) + sl_z1 * fz) * cell_y * float(dz - 1) / (2.0 * bz)
        inv_n = 1.0 / gs.sqrt(slope_x * slope_x + slope_z * slope_z + 1.0)
        nx = -slope_x * inv_n
        ny = inv_n
        nz = -slope_z * inv_n
        cos_i = sun_x * nx + sun_y * ny + sun_z * nz
        if cos_i <= 0.02:
            return
        eta = 1.0 / gs.max(ior, 1.001)
        cos_t = gs.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))
        bend = eta * cos_i - cos_t
        tx = -eta * sun_x + bend * nx
        ty = -eta * sun_y + bend * ny
        tz = -eta * sun_z + bend * nz
        if ty > -0.05:
            return
        land_x += tx / (-ty) * h
        land_z += tz / (-ty) * h
    qx = (land_x / (2.0 * bx) + 0.5) * float(mx) - 0.5
    qz = (land_z / (2.0 * bz) + 0.5) * float(mz) - 0.5
    cx = int(gs.floor(qx))
    cz = int(gs.floor(qz))
    wx = qx - float(cx)
    wz = qz - float(cz)
    for k in range(4):
        tx_i = cx + (k & 1)
        tz_i = cz + (k >> 1)
        if tx_i >= 0 and tx_i < mx and tz_i >= 0 and tz_i < mz:
            ax = wx if (k & 1) == 1 else 1.0 - wx
            az = wz if (k >> 1) == 1 else 1.0 - wz
            gs.atomic_add(photons, (tz_i, tx_i), gs.u32(ax * az * 1024.0))


@gs.kernel
def caustic_resolve(photons: gs.u32["mz", "mx"], expected: gs.f32) -> gs.f32["mz", "mx"]:
    """Relative irradiance of the bottom (1 = flat water), smoothed over 3×3."""
    iz, ix = gs.index()
    mz = photons.shape[0]
    mx = photons.shape[1]
    total = 0.0
    weight = 0.0
    for oz in range(-1, 2):
        for ox in range(-1, 2):
            jx = ix + ox
            jz = iz + oz
            if jx >= 0 and jx < mx and jz >= 0 and jz < mz:
                wk = float(2 - gs.abs(ox)) * float(2 - gs.abs(oz))
                # The outermost edges only receive part of the bilinear deposits: ignore them
                if jx > 0 and jx < mx - 1 and jz > 0 and jz < mz - 1:
                    total += float(photons[jz, jx]) * wk
                    weight += wk
    return total / (gs.max(weight, 1.0) * 1024.0 * gs.max(expected, 1e-4))


# ───────────────────────── 4. Two-bounce 3D ray tracing (Snell entry + exit + TIR + whitewater) ─────────────────────────


@gs.kernel
def fluid_raymarch_3d(
    hdr_r: gs.f32["h", "w"],
    hdr_g: gs.f32["h", "w"],
    hdr_b: gs.f32["h", "w"],
    zbuf: gs.u32["h", "w"],
    dens: gs.f32["nv"],
    caustic_map: gs.f32["mz", "mx"],
    ww_alpha: gs.u32["h", "w"],
    ww_depth: gs.u32["h", "w"],
    tex_pool: gs.u32["tp"],
    tex_meta: gs.i32["ntex", "c4"],
    dyn_atlas: gs.u32["dh", "dw"],
    sky_tid: gs.i32,
    sky_int: gs.f32,
    dx: gs.i32,
    dy: gs.i32,
    dz: gs.i32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    floor_c1_r: gs.f32,
    floor_c1_g: gs.f32,
    floor_c1_b: gs.f32,
    floor_c2_r: gs.f32,
    floor_c2_g: gs.f32,
    floor_c2_b: gs.f32,
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
    iso_level: gs.f32,
    ior: gs.f32,
    abs_r: gs.f32,
    abs_g: gs.f32,
    abs_b: gs.f32,
    tint_r: gs.f32,
    tint_g: gs.f32,
    tint_b: gs.f32,
    em_r: gs.f32,
    em_g: gs.f32,
    em_b: gs.f32,
    metallic: gs.f32,
    roughness: gs.f32,
    foam_strength: gs.f32,
    caustic_strength: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Two-bounce optical 3D ray tracing: Snell entry & exit, total internal reflection (TIR),
    Beer-Lambert, underwater caustics, translucent shadow on the floor, Fresnel HDRI and whitewater micro-particles."""
    py, px = gs.index()
    h = zbuf.shape[0]
    w = zbuf.shape[1]
    bg_r0 = hdr_r[py, px]
    bg_g0 = hdr_g[py, px]
    bg_b0 = hdr_b[py, px]

    # 1) 3D camera ray
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

    cos_ray = gs.max(rdx * fwd_x + rdy * fwd_y + rdz * fwd_z, 0.1)
    sz = zbuf[py, px]
    depth_range = gs.max(far - near, 1e-4)
    scene_z = far + 10.0
    if sz != gs.u32(4294967295):
        scene_z = near + (float(sz) / 16777215.0) * depth_range
    scene_t = scene_z / cos_ray

    # Whitewater micro-particles (airborne spray / foam / bubbles) at this pixel
    ww_a_raw = float(ww_alpha[py, px]) / 255.0
    ww_d_u = ww_depth[py, px]
    ww_z = far + 20.0
    if ww_d_u != gs.u32(4294967295):
        ww_z = near + (float(ww_d_u) / 4000000000.0) * depth_range

    # 2) Ray ↔ aquarium box intersection
    inv_rx = 1.0 / (rdx if gs.abs(rdx) > 1e-6 else 1e-6)
    inv_ry = 1.0 / (rdy if gs.abs(rdy) > 1e-6 else 1e-6)
    inv_rz = 1.0 / (rdz if gs.abs(rdz) > 1e-6 else 1e-6)

    tx1 = (-bx - cam_x) * inv_rx
    tx2 = (bx - cam_x) * inv_rx
    ty1 = (floor_y - cam_y) * inv_ry
    ty2 = (floor_y + by - cam_y) * inv_ry
    tz1 = (-bz - cam_z) * inv_rz
    tz2 = (bz - cam_z) * inv_rz

    t_enter = gs.max(gs.min(tx1, tx2), gs.min(ty1, ty2), gs.min(tz1, tz2), near)
    t_exit = gs.min(gs.max(tx1, tx2), gs.max(ty1, ty2), gs.max(tz1, tz2))
    t_end = gs.min(t_exit, scene_t + 0.03)

    inv_2bx = 1.0 / gs.max(2.0 * bx, 1e-4)
    inv_by = 1.0 / gs.max(by, 1e-4)
    inv_2bz = 1.0 / gs.max(2.0 * bz, 1e-4)
    fdx = float(dx - 1)
    fdy = float(dy - 1)
    fdz = float(dz - 1)
    plane = dx * dy

    # 3) 3D ray marching up to the entry isosurface `density >= iso_level`
    hit = 0
    hit_wall = 0
    t_hit = t_enter
    t_prev = t_enter

    if t_enter < t_end:
        step_s = 0.072
        t_cur = t_enter + 0.003
        for st in range(120):
            if hit == 0 and t_cur <= t_end:
                qx = cam_x + rdx * t_cur
                qy = cam_y + rdy * t_cur
                qz = cam_z + rdz * t_cur
                gx = gs.min(gs.max((qx + bx) * inv_2bx * fdx, 0.0), fdx - 0.001)
                gy = gs.min(gs.max((qy - floor_y) * inv_by * fdy, 0.0), fdy - 0.001)
                gz = gs.min(gs.max((qz + bz) * inv_2bz * fdz, 0.0), fdz - 0.001)
                ix0 = int(gs.floor(gx))
                iy0 = int(gs.floor(gy))
                iz0 = int(gs.floor(gz))
                fx = gx - float(ix0)
                fy = gy - float(iy0)
                fz = gz - float(iz0)
                b000 = iz0 * plane + iy0 * dx + ix0
                b010 = b000 + dx
                b100 = b000 + plane
                b110 = b100 + dx
                d00 = dens[b000] * (1.0 - fx) + dens[b000 + 1] * fx
                d10 = dens[b010] * (1.0 - fx) + dens[b010 + 1] * fx
                d01 = dens[b100] * (1.0 - fx) + dens[b100 + 1] * fx
                d11 = dens[b110] * (1.0 - fx) + dens[b110 + 1] * fx
                d_val = (d00 * (1.0 - fy) + d10 * fy) * (1.0 - fz) + (d01 * (1.0 - fy) + d11 * fy) * fz
                if d_val >= iso_level:
                    hit = 1
                    t_hit = t_cur
                    if st == 0:
                        hit_wall = 1
                else:
                    t_prev = t_cur
                    t_cur += step_s

    if hit == 0:
        # Even outside the water, show: (a) translucent shadow / caustic on the floor + (b) spray in the air!
        out_nr = bg_r0
        out_ng = bg_g0
        out_nb = bg_b0
        if sz != gs.u32(4294967295) and rdy < -0.02:
            # Soft shadow + water caustics cast on the floor by the sun
            px_s = cam_x + rdx * scene_t
            py_s = cam_y + rdy * scene_t
            pz_s = cam_z + rdz * scene_t
            if gs.abs(py_s - floor_y) < 0.08:
                # March 6 probes toward the sun in the water box for the translucent shadow
                sh_dens = 0.0
                for sk in range(6):
                    st_l = 0.35 + float(sk) * 0.45
                    sx_l = px_s + sun_x * st_l
                    sy_l = py_s + sun_y * st_l
                    sz_l = pz_s + sun_z * st_l
                    if gs.abs(sx_l) < bx and sy_l > floor_y and sy_l < floor_y + by and gs.abs(sz_l) < bz:
                        igx = gs.min(gs.max(int((sx_l + bx) * inv_2bx * fdx), 0), dx - 1)
                        igy = gs.min(gs.max(int((sy_l - floor_y) * inv_by * fdy), 0), dy - 1)
                        igz = gs.min(gs.max(int((sz_l + bz) * inv_2bz * fdz), 0), dz - 1)
                        if dens[igz * plane + igy * dx + igx] >= iso_level * 0.7:
                            sh_dens += 0.45
                if sh_dens > 0.0:
                    # Coloured transmission (bluish shadow of clear water)
                    out_nr = out_nr * gs.exp(-sh_dens * abs_r * 0.85)
                    out_ng = out_ng * gs.exp(-sh_dens * abs_g * 0.85)
                    out_nb = out_nb * gs.exp(-sh_dens * abs_b * 0.85)
        if foam_strength > 0.0 and ww_a_raw > 0.01 and ww_z < scene_z + 0.05:
            wa = gs.min(ww_a_raw * foam_strength * 1.15, 0.90)
            out_nr = out_nr * (1.0 - wa) + 1.15 * wa
            out_ng = out_ng * (1.0 - wa) + 1.20 * wa
            out_nb = out_nb * (1.0 - wa) + 1.25 * wa
        return out_nr, out_ng, out_nb

    # 4) Bisection refinement (4 steps) on the free entry surface
    if hit_wall == 0:
        t_lo = t_prev
        t_hi = t_hit
        for _ in range(4):
            t_mid = 0.5 * (t_lo + t_hi)
            qx = cam_x + rdx * t_mid
            qy = cam_y + rdy * t_mid
            qz = cam_z + rdz * t_mid
            gx = gs.min(gs.max((qx + bx) * inv_2bx * fdx, 0.0), fdx - 0.001)
            gy = gs.min(gs.max((qy - floor_y) * inv_by * fdy, 0.0), fdy - 0.001)
            gz = gs.min(gs.max((qz + bz) * inv_2bz * fdz, 0.0), fdz - 0.001)
            ix0 = int(gs.floor(gx))
            iy0 = int(gs.floor(gy))
            iz0 = int(gs.floor(gz))
            fx = gx - float(ix0)
            fy = gy - float(iy0)
            fz = gz - float(iz0)
            b000 = iz0 * plane + iy0 * dx + ix0
            b010 = b000 + dx
            b100 = b000 + plane
            b110 = b100 + dx
            d00 = dens[b000] * (1.0 - fx) + dens[b000 + 1] * fx
            d10 = dens[b010] * (1.0 - fx) + dens[b010 + 1] * fx
            d01 = dens[b100] * (1.0 - fx) + dens[b100 + 1] * fx
            d11 = dens[b110] * (1.0 - fx) + dens[b110 + 1] * fx
            dm = (d00 * (1.0 - fy) + d10 * fy) * (1.0 - fz) + (d01 * (1.0 - fy) + d11 * fy) * fz
            if dm >= iso_level:
                t_hi = t_mid
            else:
                t_lo = t_mid
        t_hit = 0.5 * (t_lo + t_hi)

    hx = cam_x + rdx * t_hit
    hy = cam_y + rdy * t_hit
    hz = cam_z + rdz * t_hit

    # 5) C1 3D entry normal (flat glass if a wall, otherwise the exact trilinear 3D gradient)
    nx = 0.0
    ny = 1.0
    nz = 0.0
    if hit_wall != 0:
        ex_m = gs.abs(hx + bx)
        ex_p = gs.abs(bx - hx)
        ey_m = gs.abs(hy - floor_y)
        ey_p = gs.abs(floor_y + by - hy)
        ez_m = gs.abs(hz + bz)
        ez_p = gs.abs(bz - hz)
        emin = gs.min(ex_m, ex_p, ey_m, ey_p, ez_m, ez_p)
        if emin == ex_m:
            nx = -1.0
            ny = 0.0
            nz = 0.0
        elif emin == ex_p:
            nx = 1.0
            ny = 0.0
            nz = 0.0
        elif emin == ez_m:
            nx = 0.0
            ny = 0.0
            nz = -1.0
        elif emin == ez_p:
            nx = 0.0
            ny = 0.0
            nz = 1.0
        elif emin == ey_m:
            nx = 0.0
            ny = -1.0
            nz = 0.0
        else:
            nx = 0.0
            ny = 1.0
            nz = 0.0
    else:
        gcx = gs.min(gs.max((hx + bx) * inv_2bx * fdx, 1.0), fdx - 1.001)
        gcy = gs.min(gs.max((hy - floor_y) * inv_by * fdy, 1.0), fdy - 1.001)
        gcz = gs.min(gs.max((hz + bz) * inv_2bz * fdz, 1.0), fdz - 1.001)
        eps_v = 0.85

        iy1 = int(gs.floor(gcy))
        iz1 = int(gs.floor(gcz))
        fy1 = gcy - float(iy1)
        fz1 = gcz - float(iz1)

        px1 = gs.min(gcx + eps_v, fdx - 0.001)
        ix1 = int(gs.floor(px1))
        fx1 = px1 - float(ix1)
        q00 = iz1 * plane + iy1 * dx + ix1
        d_px = (
            ((dens[q00] * (1.0 - fx1) + dens[q00 + 1] * fx1) * (1.0 - fy1) + (dens[q00 + dx] * (1.0 - fx1) + dens[q00 + dx + 1] * fx1) * fy1) * (1.0 - fz1)
            + ((dens[q00 + plane] * (1.0 - fx1) + dens[q00 + plane + 1] * fx1) * (1.0 - fy1) + (dens[q00 + plane + dx] * (1.0 - fx1) + dens[q00 + plane + dx + 1] * fx1) * fy1) * fz1
        )
        px0 = gs.max(gcx - eps_v, 0.0)
        ix0n = int(gs.floor(px0))
        fx0n = px0 - float(ix0n)
        q00 = iz1 * plane + iy1 * dx + ix0n
        d_mx = (
            ((dens[q00] * (1.0 - fx0n) + dens[q00 + 1] * fx0n) * (1.0 - fy1) + (dens[q00 + dx] * (1.0 - fx0n) + dens[q00 + dx + 1] * fx0n) * fy1) * (1.0 - fz1)
            + ((dens[q00 + plane] * (1.0 - fx0n) + dens[q00 + plane + 1] * fx0n) * (1.0 - fy1) + (dens[q00 + plane + dx] * (1.0 - fx0n) + dens[q00 + plane + dx + 1] * fx0n) * fy1) * fz1
        )

        ixc = int(gs.floor(gcx))
        fxc = gcx - float(ixc)
        py1 = gs.min(gcy + eps_v, fdy - 0.001)
        iy1n = int(gs.floor(py1))
        fy1n = py1 - float(iy1n)
        q00 = iz1 * plane + iy1n * dx + ixc
        d_py = (
            ((dens[q00] * (1.0 - fxc) + dens[q00 + 1] * fxc) * (1.0 - fy1n) + (dens[q00 + dx] * (1.0 - fxc) + dens[q00 + dx + 1] * fxc) * fy1n) * (1.0 - fz1)
            + ((dens[q00 + plane] * (1.0 - fxc) + dens[q00 + plane + 1] * fxc) * (1.0 - fy1n) + (dens[q00 + plane + dx] * (1.0 - fxc) + dens[q00 + plane + dx + 1] * fxc) * fy1n) * fz1
        )
        py0 = gs.max(gcy - eps_v, 0.0)
        iy0n = int(gs.floor(py0))
        fy0n = py0 - float(iy0n)
        q00 = iz1 * plane + iy0n * dx + ixc
        d_my = (
            ((dens[q00] * (1.0 - fxc) + dens[q00 + 1] * fxc) * (1.0 - fy0n) + (dens[q00 + dx] * (1.0 - fxc) + dens[q00 + dx + 1] * fxc) * fy0n) * (1.0 - fz1)
            + ((dens[q00 + plane] * (1.0 - fxc) + dens[q00 + plane + 1] * fxc) * (1.0 - fy0n) + (dens[q00 + plane + dx] * (1.0 - fxc) + dens[q00 + plane + dx + 1] * fxc) * fy0n) * fz1
        )

        pz1 = gs.min(gcz + eps_v, fdz - 0.001)
        iz1n = int(gs.floor(pz1))
        fz1n = pz1 - float(iz1n)
        q00 = iz1n * plane + iy1 * dx + ixc
        d_pz = (
            ((dens[q00] * (1.0 - fxc) + dens[q00 + 1] * fxc) * (1.0 - fy1) + (dens[q00 + dx] * (1.0 - fxc) + dens[q00 + dx + 1] * fxc) * fy1) * (1.0 - fz1n)
            + ((dens[q00 + plane] * (1.0 - fxc) + dens[q00 + plane + 1] * fxc) * (1.0 - fy1) + (dens[q00 + plane + dx] * (1.0 - fxc) + dens[q00 + plane + dx + 1] * fxc) * fy1) * fz1n
        )
        pz0 = gs.max(gcz - eps_v, 0.0)
        iz0n = int(gs.floor(pz0))
        fz0n = pz0 - float(iz0n)
        q00 = iz0n * plane + iy1 * dx + ixc
        d_mz = (
            ((dens[q00] * (1.0 - fxc) + dens[q00 + 1] * fxc) * (1.0 - fy1) + (dens[q00 + dx] * (1.0 - fxc) + dens[q00 + dx + 1] * fxc) * fy1) * (1.0 - fz0n)
            + ((dens[q00 + plane] * (1.0 - fxc) + dens[q00 + plane + 1] * fxc) * (1.0 - fy1) + (dens[q00 + plane + dx] * (1.0 - fxc) + dens[q00 + plane + dx + 1] * fxc) * fy1) * fz0n
        )

        nx = -(d_px - d_mx) * inv_2bx * fdx
        ny = -(d_py - d_my) * inv_by * fdy
        nz = -(d_pz - d_mz) * inv_2bz * fdz
        inv_n = 1.0 / gs.max(gs.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
        nx = nx * inv_n
        ny = ny * inv_n
        nz = nz * inv_n

    ndotv = -(rdx * nx + rdy * ny + rdz * nz)
    if ndotv < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz
        ndotv = -ndotv
    ndotv = gs.max(ndotv, 0.01)

    # 6) Bounce 1 — Snell refraction on entry (air → water)
    eta1 = 1.0 / gs.max(ior, 1.001)
    k1 = 1.0 - eta1 * eta1 * (1.0 - ndotv * ndotv)
    r1x = rdx
    r1y = rdy
    r1z = rdz
    if k1 > 0.0:
        s1 = eta1 * ndotv - gs.sqrt(k1)
        r1x = eta1 * rdx + s1 * nx
        r1y = eta1 * rdy + s1 * ny
        r1z = eta1 * rdz + s1 * nz
        inv_r1 = 1.0 / gs.max(gs.sqrt(r1x * r1x + r1y * r1y + r1z * r1z), 1e-6)
        r1x = r1x * inv_r1
        r1y = r1y * inv_r1
        r1z = r1z * inv_r1

    # 7) Bounce 2 — march through the water volume to the EXIT (water → air) or the bottom / wall!
    inv_r1x = 1.0 / (r1x if gs.abs(r1x) > 1e-6 else 1e-6)
    inv_r1y = 1.0 / (r1y if gs.abs(r1y) > 1e-6 else 1e-6)
    inv_r1z = 1.0 / (r1z if gs.abs(r1z) > 1e-6 else 1e-6)
    rt_x = gs.max((-bx - hx) * inv_r1x, (bx - hx) * inv_r1x)
    rt_y = gs.max((floor_y - hy) * inv_r1y, (floor_y + by - hy) * inv_r1y)
    rt_z = gs.max((-bz - hz) * inv_r1z, (bz - hz) * inv_r1z)
    t_box_out = gs.max(gs.min(rt_x, rt_y, rt_z), 0.04)

    ds_in = gs.min(t_box_out / 24.0, 0.14)
    exited_water = 0
    t_in = t_box_out
    in_lo = 0.0
    in_hi = 0.0
    n_ref = 0
    # March, then bisection (5 steps) on the exit: continuous distance in the water, without plateaus
    for k in range(72):
        tk = 0.05 + float(k) * ds_in
        if exited_water != 0:
            tk = 0.5 * (in_lo + in_hi)
        elif tk >= t_box_out:
            break
        gx = gs.min(gs.max((hx + r1x * tk + bx) * inv_2bx * fdx, 0.0), fdx - 0.001)
        gy = gs.min(gs.max((hy + r1y * tk - floor_y) * inv_by * fdy, 0.0), fdy - 0.001)
        gz = gs.min(gs.max((hz + r1z * tk + bz) * inv_2bz * fdz, 0.0), fdz - 0.001)
        ix0 = int(gs.floor(gx))
        iy0 = int(gs.floor(gy))
        iz0 = int(gs.floor(gz))
        fx = gx - float(ix0)
        fy = gy - float(iy0)
        fz = gz - float(iz0)
        b000 = iz0 * plane + iy0 * dx + ix0
        b100 = b000 + plane
        d0 = (dens[b000] * (1.0 - fx) + dens[b000 + 1] * fx) * (1.0 - fy) + (dens[b000 + dx] * (1.0 - fx) + dens[b000 + dx + 1] * fx) * fy
        d1 = (dens[b100] * (1.0 - fx) + dens[b100 + 1] * fx) * (1.0 - fy) + (dens[b100 + dx] * (1.0 - fx) + dens[b100 + dx + 1] * fx) * fy
        di = d0 * (1.0 - fz) + d1 * fz
        if exited_water == 0:
            if di < iso_level * 0.72:
                exited_water = 1
                in_hi = tk
            else:
                in_lo = tk
        else:
            if di < iso_level * 0.72:
                in_hi = tk
            else:
                in_lo = tk
            n_ref += 1
            if n_ref == 5:
                break
    if exited_water != 0:
        t_in = 0.5 * (in_lo + in_hi)
    ex_x = hx + r1x * t_in
    ex_y = hy + r1y * t_in
    ex_z = hz + r1z * t_in
    water_dist = t_in

    # Exit direction (Snell water → air + total internal reflection, TIR!)
    r2x = r1x
    r2y = r1y
    r2z = r1z
    is_tir = 0
    n2x = 0.0
    n2y = 0.0
    n2z = 0.0
    leaves = exited_water
    if exited_water == 0 and rt_y > gs.min(rt_x, rt_z):
        # Exit through a vertical glass pane: the normal is the wall's, facing the water
        leaves = 1
        if rt_x < rt_z:
            n2x = -1.0 if r1x > 0.0 else 1.0
        else:
            n2z = -1.0 if r1z > 0.0 else 1.0
    if exited_water != 0:
        # Exit normal (pointing into the water, facing ray r1)
        # Interpolated gradient (centred differences over 6 points): continuous exit direction
        for q in range(6):
            e = 0.85 if (q & 1) == 0 else -0.85
            ax_q = q // 2
            gx = gs.min(gs.max((ex_x + bx) * inv_2bx * fdx + (e if ax_q == 0 else 0.0), 0.0), fdx - 0.001)
            gy = gs.min(gs.max((ex_y - floor_y) * inv_by * fdy + (e if ax_q == 1 else 0.0), 0.0), fdy - 0.001)
            gz = gs.min(gs.max((ex_z + bz) * inv_2bz * fdz + (e if ax_q == 2 else 0.0), 0.0), fdz - 0.001)
            ix0 = int(gs.floor(gx))
            iy0 = int(gs.floor(gy))
            iz0 = int(gs.floor(gz))
            fx = gx - float(ix0)
            fy = gy - float(iy0)
            fz = gz - float(iz0)
            b000 = iz0 * plane + iy0 * dx + ix0
            b100 = b000 + plane
            d0 = (dens[b000] * (1.0 - fx) + dens[b000 + 1] * fx) * (1.0 - fy) + (dens[b000 + dx] * (1.0 - fx) + dens[b000 + dx + 1] * fx) * fy
            d1 = (dens[b100] * (1.0 - fx) + dens[b100 + 1] * fx) * (1.0 - fy) + (dens[b100 + dx] * (1.0 - fx) + dens[b100 + dx + 1] * fx) * fy
            dq = (d0 * (1.0 - fz) + d1 * fz) * (1.0 if e > 0.0 else -1.0)
            if ax_q == 0:
                n2x += dq * inv_2bx * fdx
            elif ax_q == 1:
                n2y += dq * inv_by * fdy
            else:
                n2z += dq * inv_2bz * fdz
        inv_n2 = 1.0 / gs.max(gs.sqrt(n2x * n2x + n2y * n2y + n2z * n2z), 1e-6)
        n2x = n2x * inv_n2
        n2y = n2y * inv_n2
        n2z = n2z * inv_n2
    if leaves != 0:
        c2 = -(r1x * n2x + r1y * n2y + r1z * n2z)
        if c2 < 0.0:
            n2x = -n2x
            n2y = -n2y
            n2z = -n2z
            c2 = -c2
        k2 = 1.0 - ior * ior * (1.0 - c2 * c2)
        if k2 > 0.0:
            s2 = ior * c2 - gs.sqrt(k2)
            r2x = ior * r1x + s2 * n2x
            r2y = ior * r1y + s2 * n2y
            r2z = ior * r1z + s2 * n2z
            inv_r2 = 1.0 / gs.max(gs.sqrt(r2x * r2x + r2y * r2y + r2z * r2z), 1e-6)
            r2x = r2x * inv_r2
            r2y = r2y * inv_r2
            r2z = r2z * inv_r2
        else:
            # TOTAL INTERNAL REFLECTION (prism / diamond effect inside the waves!)
            is_tir = 1
            r2x = r1x + 2.0 * c2 * n2x
            r2y = r1y + 2.0 * c2 * n2y
            r2z = r1z + 2.0 * c2 * n2z
            # The ray stays in the water down to the bottom
            water_dist += gs.min((floor_y - ex_y) / r2y, 2.5) if r2y < -0.05 else 0.65

    # 7 bis) Scene objects under or behind the water: the refracted ray is followed in the already rendered image
    hit_obj = 0
    obj_r = 0.0
    obj_g = 0.0
    obj_b = 0.0
    air_len = 5.0
    if r2y < -1e-3:
        air_len = gs.min(gs.max((floor_y - ex_y) / r2y, 0.0), 5.0)
    path_in = t_in
    s_prev = 0.0
    s_lo = 0.0
    s_hi = 0.0
    n_bis = 0
    entry_z = t_hit * cos_ray
    for k in range(42):
        along = 0.0
        if hit_obj != 0:
            along = 0.5 * (s_lo + s_hi)
        elif k < 20:
            along = path_in * (float(k) + 0.7) / 20.0
        else:
            along = path_in + air_len * (float(k - 20) + 0.7) / 16.0
        if hit_obj == 0 and k >= 36:
            break
        qx = hx + r1x * gs.min(along, path_in)
        qy = hy + r1y * gs.min(along, path_in)
        qz = hz + r1z * gs.min(along, path_in)
        if along > path_in:
            qx = qx + r2x * (along - path_in)
            qy = qy + r2y * (along - path_in)
            qz = qz + r2z * (along - path_in)
        vz_q = (qx - cam_x) * fwd_x + (qy - cam_y) * fwd_y + (qz - cam_z) * fwd_z
        if vz_q < near:
            break
        inv_vz = 1.0 / vz_q
        nqx = ((qx - cam_x) * right_x + (qy - cam_y) * right_y + (qz - cam_z) * right_z) * inv_vz
        nqy = ((qx - cam_x) * up_x + (qy - cam_y) * up_y + (qz - cam_z) * up_z) * inv_vz
        sx_q = int(gs.floor((nqx / (aspect * tan_hfov) * 0.5 + 0.5) * float(w)))
        sy_q = int(gs.floor((0.5 - nqy / tan_hfov * 0.5) * float(h)))
        if sx_q < 0 or sx_q >= w or sy_q < 0 or sy_q >= h:
            break
        zq = zbuf[sy_q, sx_q]
        behind = 0
        if zq != gs.u32(4294967295):
            z_s = near + (float(zq) / 16777215.0) * depth_range
            y_s = cam_y + (fwd_y + right_y * nqx + up_y * nqy) * z_s
            # Neither the bottom (computed below with its caustics) nor an object placed in front of the water
            if vz_q >= z_s and vz_q - z_s < 0.45 and y_s > floor_y + 0.03 and z_s > entry_z + 0.09:
                behind = 1
        if hit_obj == 0:
            if behind != 0:
                hit_obj = 1
                s_lo = s_prev
                s_hi = along
                obj_r = hdr_r[sy_q, sx_q]
                obj_g = hdr_g[sy_q, sx_q]
                obj_b = hdr_b[sy_q, sx_q]
            s_prev = along
        else:
            if behind != 0:
                s_hi = along
                obj_r = hdr_r[sy_q, sx_q]
                obj_g = hdr_g[sy_q, sx_q]
                obj_b = hdr_b[sy_q, sx_q]
            else:
                s_lo = along
            n_bis += 1
            if n_bis == 5:
                break
    if hit_obj != 0:
        water_dist = gs.min(s_hi, path_in)
        if is_tir != 0:
            water_dist = s_hi

    # 8) Evaluate the environment seen at the end of the refracted ray (lit floor + caustics OR HDRI skybox!)
    env_r = 0.0
    env_g = 0.0
    env_b = 0.0
    mask_u = gs.u32(255)
    inv255 = 0.0039215686

    hit_floor = hit_obj
    if hit_obj != 0:
        env_r = obj_r
        env_g = obj_g
        env_b = obj_b
    elif r2y < -1e-4:
        t_fl = (floor_y - ex_y) / r2y
        if t_fl >= -0.05 and t_fl < 40.0:
            fl_x = ex_x + r2x * gs.max(t_fl, 0.0)
            fl_z = ex_z + r2z * gs.max(t_fl, 0.0)
            if gs.abs(fl_x) <= 8.5 and gs.abs(fl_z) <= 8.5:
                hit_floor = 1
                cx_i = int(gs.floor(fl_x))
                cz_i = int(gs.floor(fl_z))
                is_c2 = (cx_i + cz_i) & 1
                fa_r = floor_c1_r if is_c2 == 0 else floor_c2_r
                fa_g = floor_c1_g if is_c2 == 0 else floor_c2_g
                fa_b = floor_c1_b if is_c2 == 0 else floor_c2_b
                # Caustics: actual irradiance of the bottom by the refracted sun
                caust = 1.0
                if caustic_strength > 0.0 and gs.abs(fl_x) < bx and gs.abs(fl_z) < bz:
                    mz = caustic_map.shape[0]
                    mx = caustic_map.shape[1]
                    mqx = gs.min(gs.max((fl_x + bx) * inv_2bx * float(mx) - 0.5, 0.0), float(mx) - 1.001)
                    mqz = gs.min(gs.max((fl_z + bz) * inv_2bz * float(mz) - 0.5, 0.0), float(mz) - 1.001)
                    mix0 = int(mqx)
                    miz0 = int(mqz)
                    mfx = mqx - float(mix0)
                    mfz = mqz - float(miz0)
                    c_lo = caustic_map[miz0, mix0] * (1.0 - mfx) + caustic_map[miz0, mix0 + 1] * mfx
                    c_hi = caustic_map[miz0 + 1, mix0] * (1.0 - mfx) + caustic_map[miz0 + 1, mix0 + 1] * mfx
                    c_map = gs.min(c_lo * (1.0 - mfz) + c_hi * mfz, 3.5)
                    caust = 1.0 + caustic_strength * (c_map - 1.0)
                # The caustic only modulates the sun, not the ambient light
                sun_lit = 0.38 + 0.62 * gs.max(sun_y, 0.1) * sun_int * 0.48 * caust
                env_r = fa_r * sun_lit
                env_g = fa_g * sun_lit
                env_b = fa_b * sun_lit

    if hit_floor == 0:
        if sky_tid >= 0:
            su_b = gs.atan2(r2z, r2x) * 0.15915494 + 0.5
            sv_b = gs.atan2(gs.sqrt(r2x * r2x + r2z * r2z), r2y) * 0.31830989
            tw_b = tex_meta[sky_tid, 2]
            m3_b = tex_meta[sky_tid, 3]
            th_b = m3_b & 1048575
            is_dyn_b = (m3_b >> 20) & 1
            tx_b = gs.min(gs.max(int(su_b * float(tw_b)), 0), tw_b - 1)
            ty_b = gs.min(gs.max(int(sv_b * float(th_b)), 0), th_b - 1)
            cs_b = gs.u32(0)
            if is_dyn_b != 0:
                cs_b = dyn_atlas[tex_meta[sky_tid, 1] + ty_b, tex_meta[sky_tid, 0] + tx_b]
            else:
                cs_b = tex_pool[tex_meta[sky_tid, 0] + ty_b * tw_b + tx_b]
            sr_b = float(cs_b & mask_u) * inv255
            sg_b = float((cs_b >> gs.u32(8)) & mask_u) * inv255
            sb_b = float((cs_b >> gs.u32(16)) & mask_u) * inv255
            env_r = sr_b * sr_b * sky_int
            env_g = sg_b * sg_b * sky_int
            env_b = sb_b * sb_b * sky_int
        else:
            t_sk = gs.min(gs.max(r2y * 0.5 + 0.5, 0.0), 1.0)
            env_r = 0.14 + 0.25 * (1.0 - t_sk)
            env_g = 0.22 + 0.30 * (1.0 - t_sk)
            env_b = 0.42 + 0.32 * (1.0 - t_sk)

    # Beer-Lambert absorption over the exact distance travelled in the water
    wd = gs.max(water_dist, 0.05)
    tr = gs.exp(-wd * abs_r)
    tg = gs.exp(-wd * abs_g)
    tb = gs.exp(-wd * abs_b)
    ndotl_w = gs.max((nx * sun_x + ny * sun_y + nz * sun_z) * 0.5 + 0.5, 0.0)
    scat = 0.14 + 0.42 * ndotl_w
    refr_r = env_r * tr + tint_r * (1.0 - tr) * scat
    refr_g = env_g * tg + tint_g * (1.0 - tg) * scat
    refr_b = env_b * tb + tint_b * (1.0 - tb) * scat
    if is_tir != 0:
        # Slight prismatic silver sheen on total internal reflection
        refr_r = refr_r * 1.15
        refr_g = refr_g * 1.18
        refr_b = refr_b * 1.22

    # 9) 3D Fresnel entry reflection + HDRI skybox + sun specular
    rlx = rdx + 2.0 * ndotv * nx
    rly = rdy + 2.0 * ndotv * ny
    rlz = rdz + 2.0 * ndotv * nz

    refl_r = 0.0
    refl_g = 0.0
    refl_b = 0.0
    if sky_tid >= 0:
        su_r = gs.atan2(rlz, rlx) * 0.15915494 + 0.5
        sv_r = gs.atan2(gs.sqrt(rlx * rlx + rlz * rlz), rly) * 0.31830989
        tw_r = tex_meta[sky_tid, 2]
        m3_r = tex_meta[sky_tid, 3]
        th_r = m3_r & 1048575
        is_dyn_r = (m3_r >> 20) & 1
        tx_r = gs.min(gs.max(int(su_r * float(tw_r)), 0), tw_r - 1)
        ty_r = gs.min(gs.max(int(sv_r * float(th_r)), 0), th_r - 1)
        cs_r = gs.u32(0)
        if is_dyn_r != 0:
            cs_r = dyn_atlas[tex_meta[sky_tid, 1] + ty_r, tex_meta[sky_tid, 0] + tx_r]
        else:
            cs_r = tex_pool[tex_meta[sky_tid, 0] + ty_r * tw_r + tx_r]
        sr_r = float(cs_r & mask_u) * inv255
        sg_r = float((cs_r >> gs.u32(8)) & mask_u) * inv255
        sb_r = float((cs_r >> gs.u32(16)) & mask_u) * inv255
        refl_r = sr_r * sr_r * sky_int * 1.25
        refl_g = sg_r * sg_r * sky_int * 1.25
        refl_b = sb_r * sb_r * sky_int * 1.25
    else:
        r_sk = gs.min(gs.max(rly * 0.5 + 0.5, 0.0), 1.0)
        refl_r = 0.18 + 0.28 * (1.0 - r_sk)
        refl_g = 0.28 + 0.32 * (1.0 - r_sk)
        refl_b = 0.52 + 0.32 * (1.0 - r_sk)

    # Crystalline solar specular
    hx_s = -rdx + sun_x
    hy_s = -rdy + sun_y
    hz_s = -rdz + sun_z
    inv_hs = 1.0 / gs.max(gs.sqrt(hx_s * hx_s + hy_s * hy_s + hz_s * hz_s), 1e-6)
    ndoth = gs.max(nx * hx_s * inv_hs + ny * hy_s * inv_hs + nz * hz_s * inv_hs, 0.0)
    gloss = gs.max(1.0 - roughness, 0.05)
    spec_pow = 36.0 + gloss * 220.0
    spec = gs.exp(gs.log(gs.max(ndoth, 1e-4)) * spec_pow) * (0.55 + gloss * 2.2) * sun_int * 0.52

    f0_d = (ior - 1.0) / (ior + 1.0)
    f0_d = f0_d * f0_d
    omv = 1.0 - ndotv
    f5 = omv * omv * omv * omv * omv
    f0r = f0_d * (1.0 - metallic) + tint_r * metallic
    f0g = f0_d * (1.0 - metallic) + tint_g * metallic
    f0b = f0_d * (1.0 - metallic) + tint_b * metallic
    fr = f0r + (1.0 - f0r) * f5
    fg = f0g + (1.0 - f0g) * f5
    fb = f0b + (1.0 - f0b) * f5
    if k1 <= 0.0:
        fr = 1.0
        fg = 1.0
        fb = 1.0

    em_mod = 0.50 + 0.50 * gs.max(1.0 - ny, 0.0) + 0.35 * gs.min(wd * 0.4, 1.0)

    out_r = refr_r * (1.0 - fr) + refl_r * fr + sun_r * spec + em_r * em_mod
    out_g = refr_g * (1.0 - fg) + refl_g * fg + sun_g * spec + em_g * em_mod
    out_b = refr_b * (1.0 - fb) + refl_b * fb + sun_b * spec + em_b * em_mod

    # 10) Inlay of the sparkling whitewater micro-particles (spray / foam / bubbles)
    if foam_strength > 0.0 and ww_a_raw > 0.01:
        hit_z = t_hit * cos_ray
        # In front of the surface (spray / foam) → pure white; below the surface (bubbles) → tinted by the water!
        wa = gs.min(ww_a_raw * foam_strength, 0.85)
        if ww_z <= hit_z + 0.12:
            out_r = out_r * (1.0 - wa) + 1.18 * wa
            out_g = out_g * (1.0 - wa) + 1.22 * wa
            out_b = out_b * (1.0 - wa) + 1.26 * wa
        else:
            wb = wa * 0.55
            out_r = out_r * (1.0 - wb) + (0.65 * tr + 0.35) * wb
            out_g = out_g * (1.0 - wb) + (0.85 * tg + 0.25) * wb
            out_b = out_b * (1.0 - wb) + (0.98 * tb + 0.15) * wb

    return out_r, out_g, out_b
