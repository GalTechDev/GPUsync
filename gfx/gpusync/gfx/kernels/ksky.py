"""Cloud optics: extinction from the simulated liquid water and path tracing (multiple scattering).

Scattering follows the Draine lobe of the Mie approximation; the forward peak, almost undeviated, is counted
as unscattered (extinction multiplied by `1 - f`)."""

import gpusync as gs

BLOCK = 4  # side of the blocks of the majorant grid


@gs.kernel
def cloud_excess(
    qt: gs.f32["nz", "ny", "nx"],
    qc: gs.f32["nz", "ny", "nx"],
    qr: gs.f32["nz", "ny", "nx"],
    qs: gs.f32["nz", "ny", "nx"],
    t: gs.f32["nz", "ny", "nx"],
    ref: gs.f32["nz", "c8"],
    ice: gs.f32,
    unit: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Liquid water where the air is saturated, deficit (negative) elsewhere: this signed field interpolates without bleeding at the edge.

    `ice` = 1: the ice share counts as liquid water of the same extinction (crystals of 30 µm effective radius);
    `unit`: extinction of liquid water = unit (rho q)^(2/3). Rain and snow add their geometric extinction
    (pi N0 / lambda^3, twice the cross-section, over the exponential distributions of the microphysics:
    0.398 (rho qr)^0.75 and 1.752 (rho qs)^0.75, i.e. 13 km of visibility under 1 mm/h of rain, 1 km under 1 mm/h of
    snow); the result is returned as the liquid water of the same extinction, positive even in unsaturated air."""
    k, j, i = gs.index()
    t_c = t[k, j, i]
    q = qc[k, j, i]
    rho = ref[k, 5]
    precip = 0.398162 * gs.pow(rho * gs.max(qr[k, j, i], 0.0), 0.75) + 1.752136 * gs.pow(rho * gs.max(qs[k, j, i], 0.0), 0.75)
    if q > 0.0:
        part = gs.min(gs.max((258.0 - t_c) / 20.0, 0.0), 1.0) * ice
        liquid = unit * gs.pow(rho * q, 0.6666667)
        beta = liquid
        if part > 0.0:
            crystals = 1.5 * rho * q / (917.0 * 30e-6)
            beta = (1.0 - part) * liquid + part * gs.min(crystals, liquid)
        if part > 0.0 or precip > 0.0:
            q = gs.pow((beta + precip) / unit, 1.5) / rho
        return q
    if precip > 1e-6:
        return gs.pow(precip / unit, 1.5) / rho
    p = ref[k, 1]
    es = 611.2 * gs.exp(17.67 * (t_c - 273.15) / (t_c - 29.65))
    dqs = 17.67 * 243.5 / ((t_c - 29.65) * (t_c - 29.65))
    if ice > 0.0 and t_c < 273.15:
        mix = gs.max((t_c - 250.16) / 23.0, 0.0)
        mix = gs.min(mix * mix, 1.0)
        es = es * mix + (1.0 - mix) * 611.2 * gs.exp(22.46 * (t_c - 273.15) / (t_c - 0.55))
    qs_sat = 0.62197 * es / gs.max(p - 0.37803 * es, 1.0)
    dqs = dqs * qs_sat
    return gs.min((qt[k, j, i] - qs_sat) / (1.0 + 2489.4489 * dqs), 0.0)


@gs.kernel
def cloud_majorant(
    excess: gs.f32["nz", "ny", "nx"],
    tpl: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    scale: gs.f32,
) -> gs.f32["bz", "by", "bx"]:
    """Maximum extinction in each block of 4³ cells, neighbours included (interpolation)."""
    k, j, i = gs.index()
    top = 0.0
    for c in range(-1, 5):
        zk = gs.min(gs.max(k * 4 + c, 0), excess.shape[0] - 1)
        rho = ref[zk, 5]
        for b in range(-1, 5):
            yj = (j * 4 + b + excess.shape[1]) % excess.shape[1]
            for a in range(-1, 5):
                q = excess[zk, yj, (i * 4 + a + excess.shape[2]) % excess.shape[2]]
                if q > 0.0:
                    top = gs.max(top, scale * gs.pow(rho * q, 0.6666667))
    return top + tpl[k, j, i] * 0.0


@gs.kernel
def cloud_sun_depth(
    excess: gs.f32["nz", "ny", "nx"],
    majorant: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    scale: gs.f32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    sun_x: gs.f32,
    sun_y: gs.f32,
    sun_z: gs.f32,
    step: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Optical depth toward the sun from the centre of each cell."""
    k, j, i = gs.index()
    nz = excess.shape[0]
    ny = excess.shape[1]
    nx = excess.shape[2]
    total = 0.0
    vx = sun_x * inv_dx
    vy = sun_y * inv_dy
    vz = gs.max(sun_z * inv_dz, 1e-6)
    rx = 1.0 / (vx if gs.abs(vx) > 1e-9 else 1e-9)
    ry = 1.0 / (vy if gs.abs(vy) > 1e-9 else 1e-9)
    s = 0.5 * step
    for m in range(4000):
        gz = float(k) + vz * s
        if gz >= float(nz) - 1.0:
            break
        gx = float(i) + vx * s
        gy = float(j) + vy * s
        bx = int(gs.floor(gx * 0.25))
        by = int(gs.floor(gy * 0.25))
        bz = int(gz * 0.25)
        if majorant[gs.min(bz, majorant.shape[0] - 1), by % majorant.shape[1], bx % majorant.shape[2]] <= 0.0:
            # Cloudless block: jump to the first step after leaving it
            edge = (float(bz + 1) * 4.0 - gz) / vz
            edge = gs.min(edge, gs.max((float(bx) * 4.0 - gx) * rx, (float(bx + 1) * 4.0 - gx) * rx))
            edge = gs.min(edge, gs.max((float(by) * 4.0 - gy) * ry, (float(by + 1) * 4.0 - gy) * ry))
            s += step * gs.max(gs.floor(edge / step + 1.0), 1.0)
            continue
        s += step
        kx = int(gs.floor(gx))
        ky = int(gs.floor(gy))
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        x0 = ((kx % nx) + nx) % nx
        y0 = ((ky % ny) + ny) % ny
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny
        lo = (excess[kz, y0, x0] * (1.0 - tx) + excess[kz, y0, x1] * tx) * (1.0 - ty) + (excess[kz, y1, x0] * (1.0 - tx) + excess[kz, y1, x1] * tx) * ty
        hi = (excess[kz + 1, y0, x0] * (1.0 - tx) + excess[kz + 1, y0, x1] * tx) * (1.0 - ty) + (excess[kz + 1, y1, x0] * (1.0 - tx) + excess[kz + 1, y1, x1] * tx) * ty
        q = lo * (1.0 - tz) + hi * tz
        if q > 0.0:
            total += scale * gs.pow(ref[kz, 5] * q, 0.6666667) * step
    return total


@gs.kernel
def cloud_sun_depth_nested(
    excess: gs.f32["nz", "ny", "nx"],
    majorant: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    outer: gs.f32["oz", "oy", "ox"],
    box: gs.f32["c16"],
    scale: gs.f32,
    inv_dx: gs.f32,
    inv_dz: gs.f32,
    out_dx: gs.f32,
    out_dy: gs.f32,
    out_dz: gs.f32,
    sun_x: gs.f32,
    sun_y: gs.f32,
    sun_z: gs.f32,
    step: gs.f32,
) -> gs.f32["nz", "ny", "nx"]:
    """Optical depth toward the sun from the fine grid: step by step in the useful box, then that of the coarse
    grid at the exit point. `box`: corner (x, y), then useful bounds x0, x1, y0, y1 (m)."""
    k, j, i = gs.index()
    nz = excess.shape[0]
    ny = excess.shape[1]
    nx = excess.shape[2]
    total = 0.0
    px = box[0] + (float(i) + 0.5) / inv_dx
    py = box[1] + (float(j) + 0.5) / inv_dx
    pz = (float(k) + 0.5) / inv_dz
    vx = sun_x * inv_dx
    vy = sun_y * inv_dx
    vz = gs.max(sun_z * inv_dz, 1e-6)
    rx = 1.0 / (vx if gs.abs(vx) > 1e-9 else 1e-9)
    ry = 1.0 / (vy if gs.abs(vy) > 1e-9 else 1e-9)
    s = 0.5 * step
    for m in range(4000):
        x = px + sun_x * s
        y = py + sun_y * s
        z = pz + sun_z * s
        gz = z * inv_dz - 0.5
        if x < box[2] or x >= box[3] or y < box[4] or y >= box[5] or gz >= float(nz) - 1.001:
            break
        gx = (x - box[0]) * inv_dx - 0.5
        gy = (y - box[1]) * inv_dx - 0.5
        bx = int(gx * 0.25)
        by = int(gy * 0.25)
        bz = int(gs.max(gz, 0.0) * 0.25)
        if majorant[gs.min(bz, majorant.shape[0] - 1), gs.min(by, majorant.shape[1] - 1), gs.min(bx, majorant.shape[2] - 1)] <= 0.0:
            edge = (float(bz + 1) * 4.0 - gz) / vz
            edge = gs.min(edge, gs.max((float(bx) * 4.0 - gx) * rx, (float(bx + 1) * 4.0 - gx) * rx))
            edge = gs.min(edge, gs.max((float(by) * 4.0 - gy) * ry, (float(by + 1) * 4.0 - gy) * ry))
            s += step * gs.max(gs.floor(edge / step + 1.0), 1.0)
            continue
        kx = int(gx)
        ky = int(gy)
        kz = int(gs.max(gz, 0.0))
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gs.max(gz, 0.0) - float(kz)
        x1 = gs.min(kx + 1, nx - 1)
        y1 = gs.min(ky + 1, ny - 1)
        lo = (excess[kz, ky, kx] * (1.0 - tx) + excess[kz, ky, x1] * tx) * (1.0 - ty) + (excess[kz, y1, kx] * (1.0 - tx) + excess[kz, y1, x1] * tx) * ty
        hi = (excess[kz + 1, ky, kx] * (1.0 - tx) + excess[kz + 1, ky, x1] * tx) * (1.0 - ty) + (excess[kz + 1, y1, kx] * (1.0 - tx) + excess[kz + 1, y1, x1] * tx) * ty
        q = lo * (1.0 - tz) + hi * tz
        if q > 0.0:
            total += scale * gs.pow(ref[kz, 5] * q, 0.6666667) * step
        s += step
    # Rest of the path in the coarse grid (periodic)
    ox = (px + sun_x * s) * out_dx - 0.5
    oy = (py + sun_y * s) * out_dy - 0.5
    oz = (pz + sun_z * s) * out_dz - 0.5
    if oz < float(outer.shape[0]) - 1.001:
        kx = int(gs.floor(ox))
        ky = int(gs.floor(oy))
        kz = int(gs.max(oz, 0.0))
        tx = ox - float(kx)
        ty = oy - float(ky)
        tz = gs.max(oz, 0.0) - float(kz)
        x0 = kx % outer.shape[2]
        y0 = ky % outer.shape[1]
        x1 = (kx + 1) % outer.shape[2]
        y1 = (ky + 1) % outer.shape[1]
        lo = (outer[kz, y0, x0] * (1.0 - tx) + outer[kz, y0, x1] * tx) * (1.0 - ty) + (outer[kz, y1, x0] * (1.0 - tx) + outer[kz, y1, x1] * tx) * ty
        hi = (outer[kz + 1, y0, x0] * (1.0 - tx) + outer[kz + 1, y0, x1] * tx) * (1.0 - ty) + (outer[kz + 1, y1, x0] * (1.0 - tx) + outer[kz + 1, y1, x1] * tx) * ty
        total += lo * (1.0 - tz) + hi * tz
    return total


@gs.kernel
def sky_path_trace(
    excess: gs.f32["nz", "ny", "nx"],
    sun_depth: gs.f32["nz", "ny", "nx"],
    majorant: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    fine: gs.f32["fz", "fy", "fx"],
    fine_depth: gs.f32["fz", "fy", "fx"],
    fine_major: gs.f32["mz", "my", "mx"],
    fine_ref: gs.f32["fz", "c8"],
    box: gs.f32["c16"],
    sky: gs.f32["sh", "sw", "c3"],
    acc_r: gs.f32["h", "w"],
    acc_g: gs.f32["h", "w"],
    acc_b: gs.f32["h", "w"],
    cam: gs.f32["c16"],
    sun: gs.f32["c8"],
    scale: gs.f32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    sample: gs.i32,
    g_draine: gs.f32,
    alpha: gs.f32,
    albedo_r: gs.f32,
    albedo_g: gs.f32,
    albedo_b: gs.f32,
    haze: gs.f32,
    max_events: gs.i32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Adds one sample per pixel. `cam`: position, forward, right, up (3 + 3 + 3 + 3), tan(fov/2).
    `sun`: direction, 0, RGB irradiance at cloud altitude. `box`: fine grid — corner (x, y), useful bounds
    x0, x1, y0, y1 (m), 1/dx, 1/dz, present (1) or not (0)."""
    py, px = gs.index()
    h = acc_r.shape[0]
    w = acc_r.shape[1]
    nz = excess.shape[0]
    ny = excess.shape[1]
    nx = excess.shape[2]
    rng = gs.u32(py * w + px) * gs.u32(747796405) + gs.u32(sample) * gs.u32(2891336453) + gs.u32(1)
    rng = (rng ^ (rng >> gs.u32(16))) * gs.u32(2246822519)
    rng = (rng ^ (rng >> gs.u32(13))) * gs.u32(3266489917)
    rng = rng ^ (rng >> gs.u32(16))
    rng = rng * gs.u32(747796405) + gs.u32(2891336453)
    jx = float(rng >> gs.u32(8)) * 5.9604645e-8
    rng = rng * gs.u32(747796405) + gs.u32(2891336453)
    jy = float(rng >> gs.u32(8)) * 5.9604645e-8
    ax = ((float(px) + jx) / float(w) * 2.0 - 1.0) * float(w) / float(h) * cam[12]
    ay = (1.0 - (float(py) + jy) / float(h) * 2.0) * cam[12]
    dx = cam[3] + cam[6] * ax + cam[9] * ay
    dy = cam[4] + cam[7] * ax + cam[10] * ay
    dz = cam[5] + cam[8] * ax + cam[11] * ay
    inv_l = 1.0 / gs.sqrt(dx * dx + dy * dy + dz * dz)
    dx = dx * inv_l
    dy = dy * inv_l
    dz = dz * inv_l
    wx = cam[0]
    wy = cam[1]
    wz = cam[2]
    first_d = -1.0
    travelled = 0.0
    fx0 = dx
    fy0 = dy
    fz0 = dz
    tint_r = 1.0
    tint_g = 1.0
    tint_b = 1.0
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0
    events = 0
    escaped = 0
    for it in range(20000):
        if events >= max_events or tint_b < 1e-3:
            break
        # Exit through the top or the ground
        if wz >= box[10] and dz >= 0.0:
            escaped = 1
            break
        # Outside the cloud layer: go straight to its base, its top or the ground
        s = 0.0
        if wz > box[10]:
            s = (wz - box[10]) / gs.max(-dz, 1e-6) + 0.01
        elif wz < box[9] and dz > 0.0:
            s = (box[9] - wz) / gs.max(dz, 1e-6) + 0.01
        elif wz < box[9] and wz > 0.01:
            s = (wz - 0.005) / gs.max(-dz, 1e-6)
        if s > 0.0:
            wx += dx * s
            wy += dy * s
            wz += dz * s
            travelled += s
        if travelled > 60000.0 + gs.max(first_d, 0.0):
            escaped = 1
            break
        in_fine = 0
        if box[8] > 0.0 and wx >= box[2] and wx < box[3] and wy >= box[4] and wy < box[5]:
            in_fine = 1
        if wz <= 0.01 and dz <= 0.0:
            # Lambertian surface: direct sun, then diffuse bounce
            if first_d < 0.0:
                first_d = travelled
            tau_g = 0.0
            if in_fine != 0:
                tau_g = fine_depth[0, int((wy - box[1]) * box[6]), int((wx - box[0]) * box[6])]
            else:
                tau_g = sun_depth[0, int(gs.floor(wy * inv_dy)) % ny, int(gs.floor(wx * inv_dx)) % nx]
            lit = 0.31830989 * gs.max(sun[2], 0.0) * gs.exp(-tau_g)
            tint_r = tint_r * albedo_r
            tint_g = tint_g * albedo_g
            tint_b = tint_b * albedo_b
            out_r += tint_r * lit * sun[4]
            out_g += tint_g * lit * sun[5]
            out_b += tint_b * lit * sun[6]
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u1 = float(rng >> gs.u32(8)) * 5.9604645e-8
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u2 = float(rng >> gs.u32(8)) * 5.9604645e-8 * 6.2831853
            sr = gs.sqrt(u1 + 1e-8)  # never exactly vertical: atan2(0, 0) is not defined everywhere
            dx = sr * gs.cos(u2)
            dy = sr * gs.sin(u2)
            dz = gs.sqrt(gs.max(1.0 - u1, 1e-4))
            wz = 0.02
            events += 1
        # Grid we are in: coordinates in cells
        sx = inv_dx
        sy = inv_dy
        sz = inv_dz
        gx = wx * sx - 0.5
        gy = wy * sy - 0.5
        if in_fine != 0:
            sx = box[6]
            sy = box[6]
            sz = box[7]
            gx = (wx - box[0]) * sx - 0.5
            gy = (wy - box[1]) * sy - 0.5
        gz = wz * sz - 0.5
        # Current block and distance (m) to its exit
        bx = int(gs.floor(gx * 0.25))
        by = int(gs.floor(gy * 0.25))
        bz = int(gs.floor(gz * 0.25))
        vx = dx * sx
        vy = dy * sy
        vz = dz * sz
        edge = 1e9
        if vx > 1e-9:
            edge = gs.min(edge, (float(bx + 1) * 4.0 - gx) / vx)
        elif vx < -1e-9:
            edge = gs.min(edge, (float(bx) * 4.0 - gx) / vx)
        if vy > 1e-9:
            edge = gs.min(edge, (float(by + 1) * 4.0 - gy) / vy)
        elif vy < -1e-9:
            edge = gs.min(edge, (float(by) * 4.0 - gy) / vy)
        if vz > 1e-9:
            edge = gs.min(edge, (float(bz + 1) * 4.0 - gz) / vz)
        elif vz < -1e-9:
            edge = gs.min(edge, (float(bz) * 4.0 - gz) / vz)
        if box[8] > 0.0:
            # Entering or leaving the fine grid
            rx = 1.0 / (dx if gs.abs(dx) > 1e-9 else 1e-9)
            ry = 1.0 / (dy if gs.abs(dy) > 1e-9 else 1e-9)
            tx0 = (box[2] - wx) * rx
            tx1 = (box[3] - wx) * rx
            ty0 = (box[4] - wy) * ry
            ty1 = (box[5] - wy) * ry
            t_in = gs.max(gs.min(tx0, tx1), gs.min(ty0, ty1))
            t_out = gs.min(gs.max(tx0, tx1), gs.max(ty0, ty1))
            if in_fine != 0:
                edge = gs.min(edge, gs.max(t_out, 0.0))
            elif t_in > 0.0 and t_in < t_out:
                edge = gs.min(edge, t_in)
        edge = gs.max(edge, 0.0) + 0.05
        if dz < 0.0:
            edge = gs.min(edge, wz / (-dz) + 1e-3)
        mj = 0.0
        if in_fine != 0:
            if bz >= 0 and bz < fine_major.shape[0] and bx >= 0 and by >= 0 and bx < fine_major.shape[2] and by < fine_major.shape[1]:
                mj = fine_major[bz, by, bx]
        elif bz >= 0 and bz < majorant.shape[0]:
            mj = majorant[bz, by % majorant.shape[1], bx % majorant.shape[2]]
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        xi = gs.max(float(rng >> gs.u32(8)) * 5.9604645e-8, 1e-7)
        s = edge + 1.0
        if mj > 0.0:
            s = -gs.log(xi) / mj
        if s >= edge:
            wx += dx * edge
            wy += dy * edge
            wz += dz * edge
            travelled += edge
            continue
        wx += dx * s
        wy += dy * s
        wz += dz * s
        travelled += s
        gx += vx * s
        gy += vy * s
        gz += vz * s
        limit = float(nz) - 1.001
        if in_fine != 0:
            limit = float(fine.shape[0]) - 1.001
        if gz < 0.0 or gz >= limit:
            continue
        kx = int(gs.floor(gx))
        ky = int(gs.floor(gy))
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        x0 = kx
        y0 = ky
        x1 = kx + 1
        y1 = ky + 1
        rho = 0.0
        lo = 0.0
        hi = 0.0
        if in_fine != 0:
            lo = (fine[kz, y0, x0] * (1.0 - tx) + fine[kz, y0, x1] * tx) * (1.0 - ty) + (fine[kz, y1, x0] * (1.0 - tx) + fine[kz, y1, x1] * tx) * ty
            hi = (fine[kz + 1, y0, x0] * (1.0 - tx) + fine[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine[kz + 1, y1, x0] * (1.0 - tx) + fine[kz + 1, y1, x1] * tx) * ty
            rho = fine_ref[kz, 5] * (1.0 - tz) + fine_ref[kz + 1, 5] * tz
        else:
            x0 = kx % nx
            y0 = ky % ny
            x1 = (kx + 1) % nx
            y1 = (ky + 1) % ny
            lo = (excess[kz, y0, x0] * (1.0 - tx) + excess[kz, y0, x1] * tx) * (1.0 - ty) + (excess[kz, y1, x0] * (1.0 - tx) + excess[kz, y1, x1] * tx) * ty
            hi = (excess[kz + 1, y0, x0] * (1.0 - tx) + excess[kz + 1, y0, x1] * tx) * (1.0 - ty) + (excess[kz + 1, y1, x0] * (1.0 - tx) + excess[kz + 1, y1, x1] * tx) * ty
            rho = ref[kz, 5] * (1.0 - tz) + ref[kz + 1, 5] * tz
        q = lo * (1.0 - tz) + hi * tz
        if q <= 0.0:
            continue
        sigma = scale * gs.pow(rho * q, 0.6666667)
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        if float(rng >> gs.u32(8)) * 5.9604645e-8 * mj >= sigma:
            continue
        # Scattering: direct sun (precomputed optical depth), then a new direction
        if first_d < 0.0:
            first_d = travelled
        if in_fine != 0:
            lo = (fine_depth[kz, y0, x0] * (1.0 - tx) + fine_depth[kz, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz, y1, x0] * (1.0 - tx) + fine_depth[kz, y1, x1] * tx) * ty
            hi = (fine_depth[kz + 1, y0, x0] * (1.0 - tx) + fine_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz + 1, y1, x0] * (1.0 - tx) + fine_depth[kz + 1, y1, x1] * tx) * ty
        else:
            lo = (sun_depth[kz, y0, x0] * (1.0 - tx) + sun_depth[kz, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz, y1, x0] * (1.0 - tx) + sun_depth[kz, y1, x1] * tx) * ty
            hi = (sun_depth[kz + 1, y0, x0] * (1.0 - tx) + sun_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz + 1, y1, x0] * (1.0 - tx) + sun_depth[kz + 1, y1, x1] * tx) * ty
        tau = lo * (1.0 - tz) + hi * tz
        mu = dx * sun[0] + dy * sun[1] + dz * sun[2]
        norm = 1.0 + alpha * (1.0 + 2.0 * g_draine * g_draine) * 0.33333333
        den = 1.0 + g_draine * g_draine - 2.0 * g_draine * mu
        phase = 0.07957747 * (1.0 - g_draine * g_draine) / (den * gs.sqrt(den)) * (1.0 + alpha * mu * mu) / norm
        lit = phase * gs.exp(-tau)
        out_r += tint_r * lit * sun[4]
        out_g += tint_g * lit * sun[5]
        out_b += tint_b * lit * sun[6]
        events += 1
        # Draine sampling: Henyey-Greenstein with rejection
        ct = 1.0
        for _ in range(16):
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u1 = float(rng >> gs.u32(8)) * 5.9604645e-8
            sq = (1.0 - g_draine * g_draine) / (1.0 - g_draine + 2.0 * g_draine * u1)
            ct = (1.0 + g_draine * g_draine - sq * sq) / (2.0 * g_draine)
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            if float(rng >> gs.u32(8)) * 5.9604645e-8 * (1.0 + alpha) <= 1.0 + alpha * ct * ct:
                break
        ct = gs.min(gs.max(ct, -1.0), 1.0)
        st = gs.sqrt(gs.max(1.0 - ct * ct, 0.0))
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        ph = float(rng >> gs.u32(8)) * 5.9604645e-8 * 6.2831853
        # Orthonormal basis around the direction
        sgn = 1.0 if dz >= 0.0 else -1.0
        aa = -1.0 / (sgn + dz)
        bb = dx * dy * aa
        t1x = 1.0 + sgn * dx * dx * aa
        t1y = sgn * bb
        t1z = -sgn * dx
        t2x = bb
        t2y = sgn + dy * dy * aa
        t2z = -dy
        cp = gs.cos(ph) * st
        sp = gs.sin(ph) * st
        ndx = dx * ct + t1x * cp + t2x * sp
        ndy = dy * ct + t1y * cp + t2y * sp
        ndz = dz * ct + t1z * cp + t2z * sp
        inv_n = 1.0 / gs.sqrt(ndx * ndx + ndy * ndy + ndz * ndz)
        dx = ndx * inv_n
        dy = ndy * inv_n
        dz = ndz * inv_n
    if escaped != 0:
        su = (gs.atan2(dy, dx) * 0.15915494 + 0.5) * float(sky.shape[1])
        sv = gs.sqrt(gs.max(dz, 0.0)) * float(sky.shape[0] - 1)
        ix = int(su) % sky.shape[1]
        iy = gs.min(int(sv), sky.shape[0] - 2)
        fu = su - gs.floor(su)
        fv = gs.min(sv - float(iy), 1.0)
        i1 = (ix + 1) % sky.shape[1]
        w00 = (1.0 - fu) * (1.0 - fv)
        w10 = fu * (1.0 - fv)
        w01 = (1.0 - fu) * fv
        w11 = fu * fv
        out_r += tint_r * (sky[iy, ix, 0] * w00 + sky[iy, i1, 0] * w10 + sky[iy + 1, ix, 0] * w01 + sky[iy + 1, i1, 0] * w11)
        out_g += tint_g * (sky[iy, ix, 1] * w00 + sky[iy, i1, 1] * w10 + sky[iy + 1, ix, 1] * w01 + sky[iy + 1, i1, 1] * w11)
        out_b += tint_b * (sky[iy, ix, 2] * w00 + sky[iy, i1, 2] * w10 + sky[iy + 1, ix, 2] * w01 + sky[iy + 1, i1, 2] * w11)
    if first_d > 0.0:
        # Aerial perspective: the air between the eye and the first contact, lit like the horizon in that direction
        su = (gs.atan2(fy0, fx0) * 0.15915494 + 0.5) * float(sky.shape[1])
        ix = int(su) % sky.shape[1]
        iy = gs.min(int(gs.sqrt(gs.max(fz0, 0.0)) * float(sky.shape[0] - 1)), sky.shape[0] - 1)
        tr = gs.exp(-first_d * (5.8e-6 + haze))
        out_r = out_r * tr + sky[iy, ix, 0] * (1.0 - tr)
        tr = gs.exp(-first_d * (13.5e-6 + haze))
        out_g = out_g * tr + sky[iy, ix, 1] * (1.0 - tr)
        tr = gs.exp(-first_d * (33.1e-6 + haze))
        out_b = out_b * tr + sky[iy, ix, 2] * (1.0 - tr)
    return acc_r[py, px] + out_r, acc_g[py, px] + out_g, acc_b[py, px] + out_b


@gs.kernel(group=64, cells=12)
def cloud_cache_update(
    excess: gs.f32["nz", "ny", "nx"],
    sun_depth: gs.f32["nz", "ny", "nx"],
    majorant: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    fine: gs.f32["fz", "fy", "fx"],
    fine_depth: gs.f32["fz", "fy", "fx"],
    fine_major: gs.f32["mz", "my", "mx"],
    fine_ref: gs.f32["fz", "c8"],
    box: gs.f32["c16"],
    sky: gs.f32["sh", "sw", "c3"],
    old: gs.f32["told", "c12", "g64"],
    source: gs.i32["tiles"],
    tiles: gs.i32["tiles", "c4"],
    index: gs.i32["n_blocks"],
    layout: gs.f32["c32"],
    sun: gs.f32["c8"],
    scale: gs.f32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    sample: gs.i32,
    g_draine: gs.f32,
    alpha: gs.f32,
    albedo_r: gs.f32,
    albedo_g: gs.f32,
    albedo_b: gs.f32,
    haze: gs.f32,
    max_events: gs.i32,
    blend: gs.f32,
    forward: gs.f32,
    chi2_full: gs.f32,
    peak: gs.f32,
    gather: gs.i32,
    part: gs.i32,
    parts: gs.i32,
) -> gs.f32["tiles", "c12", "g64"]:
    """Diffuse light cache: one path per cache cell, blended with the previous ones.

    The cache is sparse: tiles of 4³ cells, only in the cloud blocks. `tiles`: first cell (x, y, z) and
    grid (0 coarse, 1 fine, -1 free); `source`: tile of the previous frame (-1: new). `layout`, 16 values
    per grid: cell size (m) in x and z, corner (x, y), cache cells per simulation cell, blocks in
    x, y, z, start in the index. Per cell: RGB spherical flux `[0..2]`, vector radiance flux `[3..5]`,
    number of paths `[6]`, second moment of radiance `[7..11]` (xx, yy, xy, xz, yz). Direct sun, and the
    barely deviated `forward` share, are not in it: ray marching adds them with the exact phase function."""
    tile, c_first, rank = gs.index()
    cell_id = tile * 64 + rank
    from_tile = source[tile]
    grid = tiles[tile, 3]
    if grid < 0 or tile % parts != part:
        # Free slot, or another part's turn: keep the old value
        for slot in range(12):
            if grid >= 0 and from_tile >= 0:
                gs.store(slot, old[from_tile, slot, rank])
            else:
                gs.store(slot, 0.0)
        return
    ci = tiles[tile, 0] + (rank & 3)
    cj = tiles[tile, 1] + ((rank >> 2) & 3)
    ck = tiles[tile, 2] + (rank >> 4)
    base = 16 * grid
    nz = excess.shape[0]
    ny = excess.shape[1]
    nx = excess.shape[2]
    rng = gs.u32(cell_id) * gs.u32(747796405) + gs.u32(sample) * gs.u32(2891336453) + gs.u32(1)
    rng = (rng ^ (rng >> gs.u32(16))) * gs.u32(2246822519)
    rng = (rng ^ (rng >> gs.u32(13))) * gs.u32(3266489917)
    rng = rng ^ (rng >> gs.u32(16))
    hash_cell = gs.u32(cell_id) * gs.u32(2654435769) + gs.u32(12345)
    hash_cell = (hash_cell ^ (hash_cell >> gs.u32(16))) * gs.u32(2246822519)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(13))
    # Start and direction: low-discrepancy sequence, randomly shifted within each cell
    turn = gs.u32(sample // parts)
    u_x = float((hash_cell + turn * gs.u32(3859688306)) >> gs.u32(8)) * 5.9604645e-8
    hash_cell = hash_cell * gs.u32(747796405) + gs.u32(2891336453)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(15))
    u_y = float((hash_cell + turn * gs.u32(3468523225)) >> gs.u32(8)) * 5.9604645e-8
    hash_cell = hash_cell * gs.u32(747796405) + gs.u32(2891336453)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(15))
    u_z = float((hash_cell + turn * gs.u32(3117001274)) >> gs.u32(8)) * 5.9604645e-8
    hash_cell = hash_cell * gs.u32(747796405) + gs.u32(2891336453)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(15))
    u_c = float((hash_cell + turn * gs.u32(2801104767)) >> gs.u32(8)) * 5.9604645e-8
    hash_cell = hash_cell * gs.u32(747796405) + gs.u32(2891336453)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(15))
    u_a = float((hash_cell + turn * gs.u32(2517223198)) >> gs.u32(8)) * 5.9604645e-8
    hash_cell = hash_cell * gs.u32(747796405) + gs.u32(2891336453)
    hash_cell = hash_cell ^ (hash_cell >> gs.u32(15))
    dz = 1.0 - 2.0 * u_c
    az = u_a * 6.2831853
    dr = gs.sqrt(gs.max(1.0 - dz * dz, 1e-8))
    dx = dr * gs.cos(az)
    dy = dr * gs.sin(az)
    lx = dx
    ly = dy
    lz = dz
    # Start inside the cloud: at the edge, the cell's clear air would distort the mean read by the camera
    wx = 0.0
    wy = 0.0
    wz = 0.0
    for trial in range(8):
        wx = layout[base + 2] + (float(ci) + u_x) * layout[base]
        wy = layout[base + 3] + (float(cj) + u_y) * layout[base]
        wz = (float(ck) + u_z) * layout[base + 1]
        water = 0.0
        for corner in range(8):
            ca = corner & 1
            cb = (corner >> 1) & 1
            cc = corner >> 2
            if grid == 1:
                gx = gs.max((wx - box[0]) * box[6] - 0.5, 0.0)
                gy = gs.max((wy - box[1]) * box[6] - 0.5, 0.0)
                gz = gs.max(wz * box[7] - 0.5, 0.0)
                kx = int(gx)
                ky = int(gy)
                kz = int(gz)
                tx = gx - float(kx)
                ty = gy - float(ky)
                tz = gz - float(kz)
                water += (tx if ca == 1 else 1.0 - tx) * (ty if cb == 1 else 1.0 - ty) * (tz if cc == 1 else 1.0 - tz) * fine[gs.min(kz + cc, fine.shape[0] - 1), gs.min(ky + cb, fine.shape[1] - 1), gs.min(kx + ca, fine.shape[2] - 1)]
            else:
                gx = wx * inv_dx - 0.5
                gy = wy * inv_dy - 0.5
                gz = gs.max(wz * inv_dz - 0.5, 0.0)
                kx = int(gs.floor(gx))
                ky = int(gs.floor(gy))
                kz = int(gz)
                tx = gx - float(kx)
                ty = gy - float(ky)
                tz = gz - float(kz)
                water += (tx if ca == 1 else 1.0 - tx) * (ty if cb == 1 else 1.0 - ty) * (tz if cc == 1 else 1.0 - tz) * excess[gs.min(kz + cc, nz - 1), (ky + cb) % ny, (kx + ca) % nx]
        if water > 0.0:
            break
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        u_x = float(rng >> gs.u32(8)) * 5.9604645e-8
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        u_y = float(rng >> gs.u32(8)) * 5.9604645e-8
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        u_z = float(rng >> gs.u32(8)) * 5.9604645e-8
    # Sun at the starting point: the barely deviated share (`forward`) is counted with the direct sun, not here
    lo = 0.0
    hi = 0.0
    if box[8] > 0.0 and wx >= box[2] and wx < box[3] and wy >= box[4] and wy < box[5]:
        gx = (wx - box[0]) * box[6] - 0.5
        gy = (wy - box[1]) * box[6] - 0.5
        gz = gs.min(gs.max(wz * box[7] - 0.5, 0.0), float(fine.shape[0]) - 1.001)
        kx = int(gx)
        ky = int(gy)
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        lo = (fine_depth[kz, ky, kx] * (1.0 - tx) + fine_depth[kz, ky, kx + 1] * tx) * (1.0 - ty) + (fine_depth[kz, ky + 1, kx] * (1.0 - tx) + fine_depth[kz, ky + 1, kx + 1] * tx) * ty
        hi = (fine_depth[kz + 1, ky, kx] * (1.0 - tx) + fine_depth[kz + 1, ky, kx + 1] * tx) * (1.0 - ty) + (fine_depth[kz + 1, ky + 1, kx] * (1.0 - tx) + fine_depth[kz + 1, ky + 1, kx + 1] * tx) * ty
        lo = lo * (1.0 - tz) + hi * tz
    else:
        gx = wx * inv_dx - 0.5
        gy = wy * inv_dy - 0.5
        gz = gs.min(gs.max(wz * inv_dz - 0.5, 0.0), float(nz) - 1.001)
        kx = int(gs.floor(gx))
        ky = int(gs.floor(gy))
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        x0 = kx % nx
        y0 = ky % ny
        x1 = (kx + 1) % nx
        y1 = (ky + 1) % ny
        lo = (sun_depth[kz, y0, x0] * (1.0 - tx) + sun_depth[kz, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz, y1, x0] * (1.0 - tx) + sun_depth[kz, y1, x1] * tx) * ty
        hi = (sun_depth[kz + 1, y0, x0] * (1.0 - tx) + sun_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz + 1, y1, x0] * (1.0 - tx) + sun_depth[kz + 1, y1, x1] * tx) * ty
        lo = lo * (1.0 - tz) + hi * tz
    beam = gs.exp(-(1.0 - forward) * lo) - gs.exp(-lo)
    beam_lum = beam * (0.2126 * sun[4] + 0.7152 * sun[5] + 0.0722 * sun[6])
    norm = 1.0 + alpha * (1.0 + 2.0 * g_draine * g_draine) * 0.33333333
    g_mean = g_draine * (1.0 + alpha * (3.0 + 2.0 * g_draine * g_draine) * 0.2) / norm
    g_mean = (g_mean - forward) / (1.0 - forward)
    chi2 = (chi2_full - forward) / (1.0 - forward)
    # New cell: full path; otherwise stop in the cache after `gather` scatterings
    stop_at = 1000000
    if from_tile >= 0:
        if old[from_tile, 6, rank] > 7.5:
            stop_at = gather
    thin = 1.0
    if stop_at <= 1:
        # Medium scaled from the start: the peak's light is in the beam, there is nothing to subtract
        thin = 1.0 - forward
        beam = 0.0
        beam_lum = 0.0
    first_d = -1.0
    travelled = 0.0
    tint_r = 1.0
    tint_g = 1.0
    tint_b = 1.0
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0
    events = 0
    escaped = 0
    for it in range(600):
        if events >= max_events or tint_b < 1e-3:
            break
        # Exit through the top or the ground
        if wz >= box[10] and dz >= 0.0:
            escaped = 1
            break
        # Outside the cloud layer: go straight to its base, its top or the ground
        s = 0.0
        if wz > box[10]:
            s = (wz - box[10]) / gs.max(-dz, 1e-6) + 0.01
        elif wz < box[9] and dz > 0.0:
            s = (box[9] - wz) / gs.max(dz, 1e-6) + 0.01
        elif wz < box[9] and wz > 0.01:
            s = (wz - 0.005) / gs.max(-dz, 1e-6)
        if s > 0.0:
            wx += dx * s
            wy += dy * s
            wz += dz * s
            travelled += s
        if travelled > 60000.0 + gs.max(first_d, 0.0):
            escaped = 1
            break
        in_fine = 0
        if box[8] > 0.0 and wx >= box[2] and wx < box[3] and wy >= box[4] and wy < box[5]:
            in_fine = 1
        if wz <= 0.01 and dz <= 0.0:
            # Lambertian surface: direct sun, then diffuse bounce
            if first_d < 0.0:
                first_d = travelled
            tau_g = 0.0
            if in_fine != 0:
                tau_g = fine_depth[0, int((wy - box[1]) * box[6]), int((wx - box[0]) * box[6])]
            else:
                tau_g = sun_depth[0, int(gs.floor(wy * inv_dy)) % ny, int(gs.floor(wx * inv_dx)) % nx]
            lit = 0.31830989 * gs.max(sun[2], 0.0) * gs.exp(-tau_g)
            tint_r = tint_r * albedo_r
            tint_g = tint_g * albedo_g
            tint_b = tint_b * albedo_b
            out_r += tint_r * lit * sun[4]
            out_g += tint_g * lit * sun[5]
            out_b += tint_b * lit * sun[6]
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u1 = float(rng >> gs.u32(8)) * 5.9604645e-8
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u2 = float(rng >> gs.u32(8)) * 5.9604645e-8 * 6.2831853
            sr = gs.sqrt(u1 + 1e-8)  # never exactly vertical: atan2(0, 0) is not defined everywhere
            dx = sr * gs.cos(u2)
            dy = sr * gs.sin(u2)
            dz = gs.sqrt(gs.max(1.0 - u1, 1e-4))
            wz = 0.02
            events += 1
        # Grid we are in: coordinates in cells
        sx = inv_dx
        sy = inv_dy
        sz = inv_dz
        gx = wx * sx - 0.5
        gy = wy * sy - 0.5
        if in_fine != 0:
            sx = box[6]
            sy = box[6]
            sz = box[7]
            gx = (wx - box[0]) * sx - 0.5
            gy = (wy - box[1]) * sy - 0.5
        gz = wz * sz - 0.5
        # Current block and distance (m) to its exit
        bx = int(gs.floor(gx * 0.25))
        by = int(gs.floor(gy * 0.25))
        bz = int(gs.floor(gz * 0.25))
        vx = dx * sx
        vy = dy * sy
        vz = dz * sz
        edge = 1e9
        if vx > 1e-9:
            edge = gs.min(edge, (float(bx + 1) * 4.0 - gx) / vx)
        elif vx < -1e-9:
            edge = gs.min(edge, (float(bx) * 4.0 - gx) / vx)
        if vy > 1e-9:
            edge = gs.min(edge, (float(by + 1) * 4.0 - gy) / vy)
        elif vy < -1e-9:
            edge = gs.min(edge, (float(by) * 4.0 - gy) / vy)
        if vz > 1e-9:
            edge = gs.min(edge, (float(bz + 1) * 4.0 - gz) / vz)
        elif vz < -1e-9:
            edge = gs.min(edge, (float(bz) * 4.0 - gz) / vz)
        if box[8] > 0.0:
            # Entering or leaving the fine grid
            rx = 1.0 / (dx if gs.abs(dx) > 1e-9 else 1e-9)
            ry = 1.0 / (dy if gs.abs(dy) > 1e-9 else 1e-9)
            tx0 = (box[2] - wx) * rx
            tx1 = (box[3] - wx) * rx
            ty0 = (box[4] - wy) * ry
            ty1 = (box[5] - wy) * ry
            t_in = gs.max(gs.min(tx0, tx1), gs.min(ty0, ty1))
            t_out = gs.min(gs.max(tx0, tx1), gs.max(ty0, ty1))
            if in_fine != 0:
                edge = gs.min(edge, gs.max(t_out, 0.0))
            elif t_in > 0.0 and t_in < t_out:
                edge = gs.min(edge, t_in)
        edge = gs.max(edge, 0.0) + 0.05
        if dz < 0.0:
            edge = gs.min(edge, wz / (-dz) + 1e-3)
        mj = 0.0
        if in_fine != 0:
            if bz >= 0 and bz < fine_major.shape[0] and bx >= 0 and by >= 0 and bx < fine_major.shape[2] and by < fine_major.shape[1]:
                mj = fine_major[bz, by, bx]
        elif bz >= 0 and bz < majorant.shape[0]:
            mj = majorant[bz, by % majorant.shape[1], bx % majorant.shape[2]]
        mj = mj * thin
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        xi = gs.max(float(rng >> gs.u32(8)) * 5.9604645e-8, 1e-7)
        s = edge + 1.0
        if mj > 0.0:
            s = -gs.log(xi) / mj
        if s >= edge:
            wx += dx * edge
            wy += dy * edge
            wz += dz * edge
            travelled += edge
            continue
        wx += dx * s
        wy += dy * s
        wz += dz * s
        travelled += s
        gx += vx * s
        gy += vy * s
        gz += vz * s
        limit = float(nz) - 1.001
        if in_fine != 0:
            limit = float(fine.shape[0]) - 1.001
        if gz < 0.0 or gz >= limit:
            continue
        kx = int(gs.floor(gx))
        ky = int(gs.floor(gy))
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        x0 = kx
        y0 = ky
        x1 = kx + 1
        y1 = ky + 1
        rho = 0.0
        lo = 0.0
        hi = 0.0
        if in_fine != 0:
            lo = (fine[kz, y0, x0] * (1.0 - tx) + fine[kz, y0, x1] * tx) * (1.0 - ty) + (fine[kz, y1, x0] * (1.0 - tx) + fine[kz, y1, x1] * tx) * ty
            hi = (fine[kz + 1, y0, x0] * (1.0 - tx) + fine[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine[kz + 1, y1, x0] * (1.0 - tx) + fine[kz + 1, y1, x1] * tx) * ty
            rho = fine_ref[kz, 5] * (1.0 - tz) + fine_ref[kz + 1, 5] * tz
        else:
            x0 = kx % nx
            y0 = ky % ny
            x1 = (kx + 1) % nx
            y1 = (ky + 1) % ny
            lo = (excess[kz, y0, x0] * (1.0 - tx) + excess[kz, y0, x1] * tx) * (1.0 - ty) + (excess[kz, y1, x0] * (1.0 - tx) + excess[kz, y1, x1] * tx) * ty
            hi = (excess[kz + 1, y0, x0] * (1.0 - tx) + excess[kz + 1, y0, x1] * tx) * (1.0 - ty) + (excess[kz + 1, y1, x0] * (1.0 - tx) + excess[kz + 1, y1, x1] * tx) * ty
            rho = ref[kz, 5] * (1.0 - tz) + ref[kz + 1, 5] * tz
        q = lo * (1.0 - tz) + hi * tz
        if q <= 0.0:
            continue
        sigma = scale * gs.pow(rho * q, 0.6666667) * thin
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        if float(rng >> gs.u32(8)) * 5.9604645e-8 * mj >= sigma:
            continue
        # Scattering: direct sun (precomputed optical depth), then a new direction
        if first_d < 0.0:
            first_d = travelled
        if in_fine != 0:
            lo = (fine_depth[kz, y0, x0] * (1.0 - tx) + fine_depth[kz, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz, y1, x0] * (1.0 - tx) + fine_depth[kz, y1, x1] * tx) * ty
            hi = (fine_depth[kz + 1, y0, x0] * (1.0 - tx) + fine_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz + 1, y1, x0] * (1.0 - tx) + fine_depth[kz + 1, y1, x1] * tx) * ty
        else:
            lo = (sun_depth[kz, y0, x0] * (1.0 - tx) + sun_depth[kz, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz, y1, x0] * (1.0 - tx) + sun_depth[kz, y1, x1] * tx) * ty
            hi = (sun_depth[kz + 1, y0, x0] * (1.0 - tx) + sun_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz + 1, y1, x0] * (1.0 - tx) + sun_depth[kz + 1, y1, x1] * tx) * ty
        tau = lo * (1.0 - tz) + hi * tz
        mu = dx * sun[0] + dy * sun[1] + dz * sun[2]
        den = 1.0 + g_draine * g_draine - 2.0 * g_draine * mu
        phase = 0.07957747 * (1.0 - g_draine * g_draine) / (den * gs.sqrt(den)) * (1.0 + alpha * mu * mu) / norm
        if events + 1 >= stop_at:
            lbase = 16 * in_fine
            sub = int(layout[lbase + 4])
            lside = 4 * sub
            nbx = int(layout[lbase + 5])
            nby = int(layout[lbase + 6])
            nbz = int(layout[lbase + 7])
            lfirst = int(layout[lbase + 8])
            fcx = (wx - layout[lbase + 2]) / layout[lbase] - 0.5
            fcy = (wy - layout[lbase + 3]) / layout[lbase] - 0.5
            fcz = wz / layout[lbase + 1] - 0.5
            cix = int(gs.floor(fcx))
            ciy = int(gs.floor(fcy))
            ciz = int(gs.floor(fcz))
            fcx = fcx - float(cix)
            fcy = fcy - float(ciy)
            fcz = fcz - float(ciz)
            phi_r = 0.0
            phi_g = 0.0
            phi_b = 0.0
            flux_x = 0.0
            flux_y = 0.0
            flux_z = 0.0
            q_xx = 0.0
            q_yy = 0.0
            q_xy = 0.0
            q_xz = 0.0
            q_yz = 0.0
            wsum = 0.0
            for m in range(8):
                ca = m & 1
                cb = (m >> 1) & 1
                cc = m >> 2
                jx = cix + ca
                jy = ciy + cb
                jz = ciz + cc
                ok = 1
                if in_fine == 0:
                    jx = jx % (nbx * lside)
                    jy = jy % (nby * lside)
                if jx < 0 or jx >= nbx * lside or jy < 0 or jy >= nby * lside or jz < 0 or jz >= nbz * lside:
                    ok = 0
                lslot = -1
                if ok != 0:
                    lslot = index[lfirst + ((jz // lside) * nby + jy // lside) * nbx + jx // lside]
                if lslot >= 0:
                    ux = (jx % lside) >> 2
                    uy = (jy % lside) >> 2
                    uz = (jz % lside) >> 2
                    tl = lslot + (uz * sub + uy) * sub + ux
                    rk = (jz & 3) * 16 + (jy & 3) * 4 + (jx & 3)
                    lcount = 0.0
                    prev = source[tl]
                    if prev >= 0:
                        lcount = old[prev, 6, rk]
                    if lcount > 0.5:
                        # Brand-new cells, still noisy, weigh less
                        wt = (fcx if ca == 1 else 1.0 - fcx) * (fcy if cb == 1 else 1.0 - fcy) * (fcz if cc == 1 else 1.0 - fcz) * gs.min(lcount, 64.0)
                        wsum += wt
                        phi_r += wt * old[prev, 0, rk]
                        phi_g += wt * old[prev, 1, rk]
                        phi_b += wt * old[prev, 2, rk]
                        flux_x += wt * old[prev, 3, rk]
                        flux_y += wt * old[prev, 4, rk]
                        flux_z += wt * old[prev, 5, rk]
                        q_xx += wt * old[prev, 7, rk]
                        q_yy += wt * old[prev, 8, rk]
                        q_xy += wt * old[prev, 9, rk]
                        q_xz += wt * old[prev, 10, rk]
                        q_yz += wt * old[prev, 11, rk]
            if wsum > 1e-4:
                phi_r = gs.max(phi_r, 0.0)
                phi_g = gs.max(phi_g, 0.0)
                phi_b = gs.max(phi_b, 0.0)
                flux = flux_x * dx + flux_y * dy + flux_z * dz
                llum = gs.max(0.2126 * phi_r + 0.7152 * phi_g + 0.0722 * phi_b, 1e-9)
                # The camera sees the light travelling toward it, i.e. toward -d
                q_zz = llum - q_xx - q_yy
                quad = q_xx * dx * dx + q_yy * dy * dy + q_zz * dz * dz + 2.0 * (q_xy * dx * dy + q_xz * dx * dz + q_yz * dy * dz)
                aniso = 1.0 - 3.0 * g_mean * flux / llum + 2.5 * chi2 * (3.0 * quad / llum - 1.0)
                aniso = gs.min(gs.max(aniso, 0.0), 1.0 + 3.0 * g_mean + 5.0 * chi2)
                aniso = aniso * 0.07957747 / wsum
                # The barely deviated share of the sun is counted with the direct light (TMS), the rest comes from the cache
                part_peak = gs.pow(gs.max(0.5 + 0.5 * mu, 1e-6), peak)
                lit = phase * (1.0 - part_peak) * gs.exp(-(1.0 - forward) * tau) / (1.0 - forward)
                out_r += tint_r * (lit * sun[4] + phi_r * aniso)
                out_g += tint_g * (lit * sun[5] + phi_g * aniso)
                out_b += tint_b * (lit * sun[6] + phi_b * aniso)
                break
            # Cache still empty here: continue the path
            thin = 1.0
            stop_at = 1000000
        lit = phase * gs.exp(-tau)
        out_r += tint_r * lit * sun[4]
        out_g += tint_g * lit * sun[5]
        out_b += tint_b * lit * sun[6]
        events += 1
        if events + 1 >= stop_at:
            thin = 1.0 - forward
        # Draine sampling: Henyey-Greenstein with rejection
        ct = 1.0
        for _ in range(16):
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            u1 = float(rng >> gs.u32(8)) * 5.9604645e-8
            sq = (1.0 - g_draine * g_draine) / (1.0 - g_draine + 2.0 * g_draine * u1)
            ct = (1.0 + g_draine * g_draine - sq * sq) / (2.0 * g_draine)
            rng = rng * gs.u32(747796405) + gs.u32(2891336453)
            if float(rng >> gs.u32(8)) * 5.9604645e-8 * (1.0 + alpha) <= 1.0 + alpha * ct * ct:
                break
        ct = gs.min(gs.max(ct, -1.0), 1.0)
        st = gs.sqrt(gs.max(1.0 - ct * ct, 0.0))
        rng = rng * gs.u32(747796405) + gs.u32(2891336453)
        ph = float(rng >> gs.u32(8)) * 5.9604645e-8 * 6.2831853
        # Orthonormal basis around the direction
        sgn = 1.0 if dz >= 0.0 else -1.0
        aa = -1.0 / (sgn + dz)
        bb = dx * dy * aa
        t1x = 1.0 + sgn * dx * dx * aa
        t1y = sgn * bb
        t1z = -sgn * dx
        t2x = bb
        t2y = sgn + dy * dy * aa
        t2z = -dy
        cp = gs.cos(ph) * st
        sp = gs.sin(ph) * st
        ndx = dx * ct + t1x * cp + t2x * sp
        ndy = dy * ct + t1y * cp + t2y * sp
        ndz = dz * ct + t1z * cp + t2z * sp
        inv_n = 1.0 / gs.sqrt(ndx * ndx + ndy * ndy + ndz * ndz)
        dx = ndx * inv_n
        dy = ndy * inv_n
        dz = ndz * inv_n
    if escaped != 0:
        su = (gs.atan2(dy, dx) * 0.15915494 + 0.5) * float(sky.shape[1])
        sv = gs.sqrt(gs.max(dz, 0.0)) * float(sky.shape[0] - 1)
        ix = int(su) % sky.shape[1]
        iy = gs.min(int(sv), sky.shape[0] - 2)
        fu = su - gs.floor(su)
        fv = gs.min(sv - float(iy), 1.0)
        i1 = (ix + 1) % sky.shape[1]
        w00 = (1.0 - fu) * (1.0 - fv)
        w10 = fu * (1.0 - fv)
        w01 = (1.0 - fu) * fv
        w11 = fu * fv
        out_r += tint_r * (sky[iy, ix, 0] * w00 + sky[iy, i1, 0] * w10 + sky[iy + 1, ix, 0] * w01 + sky[iy + 1, i1, 0] * w11)
        out_g += tint_g * (sky[iy, ix, 1] * w00 + sky[iy, i1, 1] * w10 + sky[iy + 1, ix, 1] * w01 + sky[iy + 1, i1, 1] * w11)
        out_b += tint_b * (sky[iy, ix, 2] * w00 + sky[iy, i1, 2] * w10 + sky[iy + 1, ix, 2] * w01 + sky[iy + 1, i1, 2] * w11)
    # Estimator: spherical flux = 4 pi <L>; light seen toward `l` travels toward `-l`
    lum = (0.2126 * out_r + 0.7152 * out_g + 0.0722 * out_b) * 12.566371
    # Mean with growing weights (the first, noisy paths fade away), then exponential forgetting
    count = 1.0
    if from_tile >= 0:
        count = gs.min(old[from_tile, 6, rank] + 1.0, 1.0 / gs.max(1.0 - blend, 1e-4))
    f = 2.0 / (count + 1.0)
    gain = out_r * 12.566371 - beam * sun[4]
    for slot in range(12):
        if slot == 1:
            gain = out_g * 12.566371 - beam * sun[5]
        elif slot == 2:
            gain = out_b * 12.566371 - beam * sun[6]
        elif slot == 3:
            gain = beam_lum * sun[0] - lum * lx
        elif slot == 4:
            gain = beam_lum * sun[1] - lum * ly
        elif slot == 5:
            gain = beam_lum * sun[2] - lum * lz
        elif slot == 7:
            gain = lum * lx * lx - beam_lum * sun[0] * sun[0]
        elif slot == 8:
            gain = lum * ly * ly - beam_lum * sun[1] * sun[1]
        elif slot == 9:
            gain = lum * lx * ly - beam_lum * sun[0] * sun[1]
        elif slot == 10:
            gain = lum * lx * lz - beam_lum * sun[0] * sun[2]
        elif slot == 11:
            gain = lum * ly * lz - beam_lum * sun[1] * sun[2]
        if slot == 6:
            gs.store(slot, count)
        elif from_tile >= 0:
            gs.store(slot, old[from_tile, slot, rank] * (1.0 - f) + gain * f)
        else:
            gs.store(slot, gain)


@gs.kernel
def sky_tonemap(
    acc_r: gs.f32["h", "w"], acc_g: gs.f32["h", "w"], acc_b: gs.f32["h", "w"], gain: gs.f32
) -> gs.u32["h", "w"]:
    """Exposure and filmic curve (Hable), approximate sRGB encoding."""
    y, x = gs.index()
    packed = gs.u32(4278190080)
    for c in range(3):
        v = acc_r[y, x]
        if c == 1:
            v = acc_g[y, x]
        elif c == 2:
            v = acc_b[y, x]
        v = gs.max(v * gain, 0.0)
        num = v * (0.22 * v + 0.03) + 0.002
        den = v * (0.22 * v + 0.3) + 0.06
        f = num / den - 0.033333333
        f = gs.min(gs.max(f * 1.379, 0.0), 1.0)
        packed = packed | (gs.u32(gs.pow(f, 0.45454545) * 255.0 + 0.5) << gs.u32(8 * c))
    return packed


@gs.kernel
def sky_decay(
    acc_r: gs.f32["h", "w"], acc_g: gs.f32["h", "w"], acc_b: gs.f32["h", "w"], factor: gs.f32
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Forgets part of the old samples (moving clouds)."""
    y, x = gs.index()
    return acc_r[y, x] * factor, acc_g[y, x] * factor, acc_b[y, x] * factor


@gs.kernel
def sky_blend(
    new_r: gs.f32["h", "w"], new_g: gs.f32["h", "w"], new_b: gs.f32["h", "w"],
    old_r: gs.f32["h", "w"], old_g: gs.f32["h", "w"], old_b: gs.f32["h", "w"], keep: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Running average of two images."""
    y, x = gs.index()
    return (
        new_r[y, x] * (1.0 - keep) + old_r[y, x] * keep,
        new_g[y, x] * (1.0 - keep) + old_g[y, x] * keep,
        new_b[y, x] * (1.0 - keep) + old_b[y, x] * keep,
    )  # fmt: skip


# ───────────────────────── Real time: ray marching over the light cache ─────────────────────────


@gs.kernel
def sky_ray_march(
    excess: gs.f32["nz", "ny", "nx"],
    sun_depth: gs.f32["nz", "ny", "nx"],
    majorant: gs.f32["bz", "by", "bx"],
    ref: gs.f32["nz", "c8"],
    fine: gs.f32["fz", "fy", "fx"],
    fine_depth: gs.f32["fz", "fy", "fx"],
    fine_major: gs.f32["mz", "my", "mx"],
    fine_ref: gs.f32["fz", "c8"],
    box: gs.f32["c16"],
    sky: gs.f32["sh", "sw", "c3"],
    cache: gs.f32["t0", "c12", "g64"],
    index: gs.i32["n_blocks"],
    layout: gs.f32["c32"],
    target: gs.u32["h", "w"],
    cam: gs.f32["c16"],
    sun: gs.f32["c8"],
    scale: gs.f32,
    inv_dx: gs.f32,
    inv_dy: gs.f32,
    inv_dz: gs.f32,
    frame: gs.i32,
    g_draine: gs.f32,
    alpha: gs.f32,
    albedo_r: gs.f32,
    albedo_g: gs.f32,
    albedo_b: gs.f32,
    haze: gs.f32,
    quality: gs.f32,
    forward: gs.f32,
    chi2_full: gs.f32,
    peak: gs.f32,
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """Image without path tracing: integration along the ray, direct sun with its exact phase, and diffuse
    light read from the cache (spherical flux + vector flux, first order of the phase function)."""
    py, px = gs.index()
    h = target.shape[0]
    w = target.shape[1]
    nz = excess.shape[0]
    ny = excess.shape[1]
    nx = excess.shape[2]
    ax = ((float(px) + 0.5) / float(w) * 2.0 - 1.0) * float(w) / float(h) * cam[12]
    ay = (1.0 - (float(py) + 0.5) / float(h) * 2.0) * cam[12]
    dx = cam[3] + cam[6] * ax + cam[9] * ay
    dy = cam[4] + cam[7] * ax + cam[10] * ay
    dz = cam[5] + cam[8] * ax + cam[11] * ay
    inv_l = 1.0 / gs.sqrt(dx * dx + dy * dy + dz * dz)
    dx = dx * inv_l
    dy = dy * inv_l
    dz = dz * inv_l
    # Per-pixel offset (interleaved gradient noise), different at each frame
    jit = 52.9829189 * ((0.06711056 * (float(px) + 5.588238 * float(frame % 64)) + 0.00583715 * (float(py) + 5.588238 * float(frame % 64))) % 1.0)
    jit = jit - gs.floor(jit)
    t_end = 60000.0
    ground = 0
    if dz < -1e-5:
        tg = cam[2] / (-dz)
        if tg < t_end:
            t_end = tg
            ground = 1
    elif dz > 1e-5:
        t_end = gs.min(t_end, gs.max(box[10] - cam[2], 0.0) / dz)
    mu = dx * sun[0] + dy * sun[1] + dz * sun[2]
    norm = 1.0 + alpha * (1.0 + 2.0 * g_draine * g_draine) * 0.33333333
    den = 1.0 + g_draine * g_draine - 2.0 * g_draine * mu
    phase = 0.07957747 * (1.0 - g_draine * g_draine) / (den * gs.sqrt(den)) * (1.0 + alpha * mu * mu) / norm
    # Mean cosine of the Draine phase
    g2 = g_draine * g_draine
    g_mean = g_draine * (1.0 + alpha * (3.0 + 2.0 * g2) * 0.2) / norm
    g_mean = (g_mean - forward) / (1.0 - forward)
    chi2 = (chi2_full - forward) / (1.0 - forward)
    phase = phase / (1.0 - forward)
    # Forward peak: counted at its first scattering only (energy 1 - exp(-f tau)), the rest in TMS
    part_peak = gs.pow(gs.max(0.5 + 0.5 * mu, 1e-6), peak)
    rx = 1.0 / (dx if gs.abs(dx) > 1e-9 else 1e-9)
    ry = 1.0 / (dy if gs.abs(dy) > 1e-9 else 1e-9)
    t_in = 1e9
    t_out = -1.0
    if box[8] > 0.0:
        tx0 = (box[2] - cam[0]) * rx
        tx1 = (box[3] - cam[0]) * rx
        ty0 = (box[4] - cam[1]) * ry
        ty1 = (box[5] - cam[1]) * ry
        t_in = gs.max(gs.min(tx0, tx1), gs.min(ty0, ty1))
        t_out = gs.min(gs.max(tx0, tx1), gs.max(ty0, ty1))

    trans = 1.0
    l_r = 0.0
    l_g = 0.0
    l_b = 0.0
    depth_sum = 0.0
    t = 0.0
    t_last = t_end
    if cam[2] > box[10]:
        t = (cam[2] - box[10]) / gs.max(-dz, 1e-9)
    elif cam[2] < box[9]:
        t = (box[9] - cam[2]) / gs.max(dz, 1e-9)
    if dz < 0.0:
        t_last = gs.min(t_last, gs.max(cam[2] - box[9], 0.0) / gs.max(-dz, 1e-9))
    for it in range(3000):
        if t >= t_last or trans < 0.004:
            break
        in_fine = 0
        if t >= t_in and t < t_out:
            in_fine = 1
        wx = cam[0] + dx * t
        wy = cam[1] + dy * t
        wz = gs.max(cam[2] + dz * t, 0.0)
        sx = inv_dx
        sy = inv_dy
        sz = inv_dz
        gx = wx * sx - 0.5
        gy = wy * sy - 0.5
        if in_fine != 0:
            sx = box[6]
            sy = box[6]
            sz = box[7]
            gx = (wx - box[0]) * sx - 0.5
            gy = (wy - box[1]) * sy - 0.5
        gz = wz * sz - 0.5
        bx = int(gs.floor(gx * 0.25))
        by = int(gs.floor(gy * 0.25))
        bz = int(gs.floor(gz * 0.25))
        vx = dx * sx
        vy = dy * sy
        vz = dz * sz
        edge = 1e9
        if vx > 1e-9:
            edge = gs.min(edge, (float(bx + 1) * 4.0 - gx) / vx)
        elif vx < -1e-9:
            edge = gs.min(edge, (float(bx) * 4.0 - gx) / vx)
        if vy > 1e-9:
            edge = gs.min(edge, (float(by + 1) * 4.0 - gy) / vy)
        elif vy < -1e-9:
            edge = gs.min(edge, (float(by) * 4.0 - gy) / vy)
        if vz > 1e-9:
            edge = gs.min(edge, (float(bz + 1) * 4.0 - gz) / vz)
        elif vz < -1e-9:
            edge = gs.min(edge, (float(bz) * 4.0 - gz) / vz)
        if in_fine != 0:
            edge = gs.min(edge, t_out - t)
        elif t < t_in and t_in < t_out:
            edge = gs.min(edge, t_in - t)
        edge = gs.min(gs.max(edge, 0.0) + 0.05, t_end - t + 0.01)
        mj = 0.0
        if in_fine != 0:
            if bz >= 0 and bz < fine_major.shape[0] and bx >= 0 and by >= 0 and bx < fine_major.shape[2] and by < fine_major.shape[1]:
                mj = fine_major[bz, by, bx]
        elif bz >= 0 and bz < majorant.shape[0]:
            mj = majorant[bz, by % majorant.shape[1], bx % majorant.shape[2]]
        if mj <= 0.0:
            t += edge
            continue
        # Step: a fraction of a cell, longer far away, never more than the block
        step = gs.max(gs.min(1.0 / sx, 1.0 / sz), 0.0015 * t) / quality
        step = gs.min(step, edge)
        tm = t + step * jit
        t += step
        wx = cam[0] + dx * tm
        wy = cam[1] + dy * tm
        wz = cam[2] + dz * tm
        gx = wx * sx - 0.5
        gy = wy * sy - 0.5
        if in_fine != 0:
            gx = (wx - box[0]) * sx - 0.5
            gy = (wy - box[1]) * sy - 0.5
        gz = wz * sz - 0.5
        limit = float(nz) - 1.001
        if in_fine != 0:
            limit = float(fine.shape[0]) - 1.001
        if gz < 0.0 or gz >= limit:
            continue
        kx = int(gs.floor(gx))
        ky = int(gs.floor(gy))
        kz = int(gz)
        tx = gx - float(kx)
        ty = gy - float(ky)
        tz = gz - float(kz)
        x0 = kx
        y0 = ky
        x1 = kx + 1
        y1 = ky + 1
        rho = 0.0
        lo = 0.0
        hi = 0.0
        if in_fine != 0:
            lo = (fine[kz, y0, x0] * (1.0 - tx) + fine[kz, y0, x1] * tx) * (1.0 - ty) + (fine[kz, y1, x0] * (1.0 - tx) + fine[kz, y1, x1] * tx) * ty
            hi = (fine[kz + 1, y0, x0] * (1.0 - tx) + fine[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine[kz + 1, y1, x0] * (1.0 - tx) + fine[kz + 1, y1, x1] * tx) * ty
            rho = fine_ref[kz, 5] * (1.0 - tz) + fine_ref[kz + 1, 5] * tz
        else:
            x0 = kx % nx
            y0 = ky % ny
            x1 = (kx + 1) % nx
            y1 = (ky + 1) % ny
            lo = (excess[kz, y0, x0] * (1.0 - tx) + excess[kz, y0, x1] * tx) * (1.0 - ty) + (excess[kz, y1, x0] * (1.0 - tx) + excess[kz, y1, x1] * tx) * ty
            hi = (excess[kz + 1, y0, x0] * (1.0 - tx) + excess[kz + 1, y0, x1] * tx) * (1.0 - ty) + (excess[kz + 1, y1, x0] * (1.0 - tx) + excess[kz + 1, y1, x1] * tx) * ty
            rho = ref[kz, 5] * (1.0 - tz) + ref[kz + 1, 5] * tz
        q = lo * (1.0 - tz) + hi * tz
        if q <= 0.0:
            continue
        sigma = scale * gs.pow(rho * q, 0.6666667) * (1.0 - forward)
        if in_fine != 0:
            lo = (fine_depth[kz, y0, x0] * (1.0 - tx) + fine_depth[kz, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz, y1, x0] * (1.0 - tx) + fine_depth[kz, y1, x1] * tx) * ty
            hi = (fine_depth[kz + 1, y0, x0] * (1.0 - tx) + fine_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (fine_depth[kz + 1, y1, x0] * (1.0 - tx) + fine_depth[kz + 1, y1, x1] * tx) * ty
        else:
            lo = (sun_depth[kz, y0, x0] * (1.0 - tx) + sun_depth[kz, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz, y1, x0] * (1.0 - tx) + sun_depth[kz, y1, x1] * tx) * ty
            hi = (sun_depth[kz + 1, y0, x0] * (1.0 - tx) + sun_depth[kz + 1, y0, x1] * tx) * (1.0 - ty) + (sun_depth[kz + 1, y1, x0] * (1.0 - tx) + sun_depth[kz + 1, y1, x1] * tx) * ty
        tau = lo * (1.0 - tz) + hi * tz
        direct = phase * ((1.0 - part_peak) * gs.exp(-(1.0 - forward) * tau) + part_peak * gs.exp(-tau))
        # Cache: interpolation weighted by the cells that received light
        base = 16 * in_fine
        sub = int(layout[base + 4])
        side = 4 * sub
        nbx = int(layout[base + 5])
        nby = int(layout[base + 6])
        nbz = int(layout[base + 7])
        first = int(layout[base + 8])
        cx = (wx - layout[base + 2]) / layout[base] - 0.5
        cy = (wy - layout[base + 3]) / layout[base] - 0.5
        cz = wz / layout[base + 1] - 0.5
        ix = int(gs.floor(cx))
        iy = int(gs.floor(cy))
        iz = int(gs.floor(cz))
        cx = cx - float(ix)
        cy = cy - float(iy)
        cz = cz - float(iz)
        phi_r = 0.0
        phi_g = 0.0
        phi_b = 0.0
        flux_x = 0.0
        flux_y = 0.0
        flux_z = 0.0
        q_xx = 0.0
        q_yy = 0.0
        q_xy = 0.0
        q_xz = 0.0
        q_yz = 0.0
        wsum = 0.0
        for m in range(8):
            a = m & 1
            b = (m >> 1) & 1
            c = m >> 2
            jx = ix + a
            jy = iy + b
            jz = iz + c
            ok = 1
            if in_fine == 0:
                jx = jx % (nbx * side)
                jy = jy % (nby * side)
            if jx < 0 or jx >= nbx * side or jy < 0 or jy >= nby * side or jz < 0 or jz >= nbz * side:
                ok = 0
            slot = -1
            if ok != 0:
                slot = index[first + ((jz // side) * nby + jy // side) * nbx + jx // side]
            if slot >= 0:
                ux = (jx % side) >> 2
                uy = (jy % side) >> 2
                uz = (jz % side) >> 2
                tl = slot + (uz * sub + uy) * sub + ux
                rk = (jz & 3) * 16 + (jy & 3) * 4 + (jx & 3)
                count = cache[tl, 6, rk]
                if count > 0.5:
                    # Brand-new cells, still noisy, weigh less
                    wt = (cx if a == 1 else 1.0 - cx) * (cy if b == 1 else 1.0 - cy) * (cz if c == 1 else 1.0 - cz) * gs.min(count, 64.0)
                    wsum += wt
                    phi_r += wt * cache[tl, 0, rk]
                    phi_g += wt * cache[tl, 1, rk]
                    phi_b += wt * cache[tl, 2, rk]
                    flux_x += wt * cache[tl, 3, rk]
                    flux_y += wt * cache[tl, 4, rk]
                    flux_z += wt * cache[tl, 5, rk]
                    q_xx += wt * cache[tl, 7, rk]
                    q_yy += wt * cache[tl, 8, rk]
                    q_xy += wt * cache[tl, 9, rk]
                    q_xz += wt * cache[tl, 10, rk]
                    q_yz += wt * cache[tl, 11, rk]
        s_r = direct * sun[4]
        s_g = direct * sun[5]
        s_b = direct * sun[6]
        if wsum > 1e-4:
            phi_r = gs.max(phi_r, 0.0)
            phi_g = gs.max(phi_g, 0.0)
            phi_b = gs.max(phi_b, 0.0)
            flux = flux_x * dx + flux_y * dy + flux_z * dz
            lum = gs.max(0.2126 * phi_r + 0.7152 * phi_g + 0.0722 * phi_b, 1e-9)
            # The camera sees the light travelling toward it, i.e. toward -d
            q_zz = lum - q_xx - q_yy
            quad = q_xx * dx * dx + q_yy * dy * dy + q_zz * dz * dz + 2.0 * (q_xy * dx * dy + q_xz * dx * dz + q_yz * dy * dz)
            aniso = 1.0 - 3.0 * g_mean * flux / lum + 2.5 * chi2 * (3.0 * quad / lum - 1.0)
            aniso = gs.min(gs.max(aniso, 0.0), 1.0 + 3.0 * g_mean + 5.0 * chi2)
            aniso = aniso * 0.07957747 / wsum
            s_r += phi_r * aniso
            s_g += phi_g * aniso
            s_b += phi_b * aniso
        part = trans * (1.0 - gs.exp(-sigma * step))
        l_r += part * s_r
        l_g += part * s_g
        l_b += part * s_b
        depth_sum += part * tm
        trans = trans * gs.exp(-sigma * step)

    cover = 1.0 - trans
    su = (gs.atan2(dy, dx + 1e-9) * 0.15915494 + 0.5) * float(sky.shape[1])
    sv = gs.sqrt(gs.max(dz, 0.0)) * float(sky.shape[0] - 1)
    ix = int(su) % sky.shape[1]
    iy = gs.min(int(sv), sky.shape[0] - 2)
    fu = su - gs.floor(su)
    fv = gs.min(sv - float(iy), 1.0)
    i1 = (ix + 1) % sky.shape[1]
    w00 = (1.0 - fu) * (1.0 - fv)
    w10 = fu * (1.0 - fv)
    w01 = (1.0 - fu) * fv
    w11 = fu * fv
    k_r = sky[iy, ix, 0] * w00 + sky[iy, i1, 0] * w10 + sky[iy + 1, ix, 0] * w01 + sky[iy + 1, i1, 0] * w11
    k_g = sky[iy, ix, 1] * w00 + sky[iy, i1, 1] * w10 + sky[iy + 1, ix, 1] * w01 + sky[iy + 1, i1, 1] * w11
    k_b = sky[iy, ix, 2] * w00 + sky[iy, i1, 2] * w10 + sky[iy + 1, ix, 2] * w01 + sky[iy + 1, i1, 2] * w11
    # Colour of the air toward the horizon in that direction (aerial perspective), as in the reference
    h_r = sky[0, ix, 0]
    h_g = sky[0, ix, 1]
    h_b = sky[0, ix, 2]
    if dz > 0.0:
        h_r = k_r
        h_g = k_g
        h_b = k_b
    b_r = k_r
    b_g = k_g
    b_b = k_b
    if ground != 0:
        gxw = cam[0] + dx * t_end
        gyw = cam[1] + dy * t_end
        tau_g = 0.0
        if box[8] > 0.0 and gxw >= box[2] and gxw < box[3] and gyw >= box[4] and gyw < box[5]:
            ax = (gxw - box[0]) * box[6] - 0.5
            ay = (gyw - box[1]) * box[6] - 0.5
            kx = int(ax)
            ky = int(ay)
            tx = ax - float(kx)
            ty = ay - float(ky)
            tau_g = (gs.exp(-fine_depth[0, ky, kx]) * (1.0 - tx) + gs.exp(-fine_depth[0, ky, kx + 1]) * tx) * (1.0 - ty) + (gs.exp(-fine_depth[0, ky + 1, kx]) * (1.0 - tx) + gs.exp(-fine_depth[0, ky + 1, kx + 1]) * tx) * ty
        else:
            ax = gxw * inv_dx - 0.5
            ay = gyw * inv_dy - 0.5
            kx = int(gs.floor(ax))
            ky = int(gs.floor(ay))
            tx = ax - float(kx)
            ty = ay - float(ky)
            x0 = kx % nx
            y0 = ky % ny
            x1 = (kx + 1) % nx
            y1 = (ky + 1) % ny
            tau_g = (gs.exp(-sun_depth[0, y0, x0]) * (1.0 - tx) + gs.exp(-sun_depth[0, y0, x1]) * tx) * (1.0 - ty) + (gs.exp(-sun_depth[0, y1, x0]) * (1.0 - tx) + gs.exp(-sun_depth[0, y1, x1]) * tx) * ty
        lit = 0.31830989 * gs.max(sun[2], 0.0) * tau_g
        # Sky: mean irradiance given with the sun (sun columns 7: factor), zenith colour
        top = sky.shape[0] - 1
        b_r = albedo_r * (lit * sun[4] + sun[7] * sky[top, 0, 0])
        b_g = albedo_g * (lit * sun[5] + sun[7] * sky[top, 0, 1])
        b_b = albedo_b * (lit * sun[6] + sun[7] * sky[top, 0, 2])
        tr = gs.exp(-t_end * (5.8e-6 + haze))
        b_r = b_r * tr + h_r * (1.0 - tr)
        tr = gs.exp(-t_end * (13.5e-6 + haze))
        b_g = b_g * tr + h_g * (1.0 - tr)
        tr = gs.exp(-t_end * (33.1e-6 + haze))
        b_b = b_b * tr + h_b * (1.0 - tr)
    if cover > 1e-5:
        dist = depth_sum / cover
        tr = gs.exp(-dist * (5.8e-6 + haze))
        l_r = l_r * tr + h_r * (1.0 - tr) * cover
        tr = gs.exp(-dist * (13.5e-6 + haze))
        l_g = l_g * tr + h_g * (1.0 - tr) * cover
        tr = gs.exp(-dist * (33.1e-6 + haze))
        l_b = l_b * tr + h_b * (1.0 - tr) * cover
    return l_r + trans * b_r, l_g + trans * b_g, l_b + trans * b_b
