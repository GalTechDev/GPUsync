"""Procedural clouds: ray marching of a noise-shaped density, with approximate multiple scattering.

The density follows Schneider (Horizon Zero Dawn, 2015): a cloud mask with a height profile, a low-frequency
Perlin-Worley noise for the mass, eroded by a high-frequency Worley noise at the edges. The light follows Hillaire
(Frostbite, 2016): energy-conserving integration over each step, sun optical depth by a short march toward the sun,
multiple scattering by the octaves of Wrenninge et al. (2013), and an ambient sky term graded with height."""

import gpusync as gs


@gs.kernel
def cloud_procedural(
    bg_r: gs.f32["h", "w"],
    bg_g: gs.f32["h", "w"],
    bg_b: gs.f32["h", "w"],
    base: gs.f32["nb", "nb2", "nb3"],
    detail: gs.f32["nd", "nd2", "nd3"],
    bub: gs.f32["nbub", "c4"],
    cam: gs.f32["c16"],
    prm: gs.f32["c48"],
) -> (gs.f32["h", "w"], gs.f32["h", "w"], gs.f32["h", "w"]):
    """RGB radiance of the background `bg` seen through the clouds.

    `prm`: 0-2 direction toward the sun, 3-5 sun irradiance, 6-8 sky radiance from above, 9-11 radiance from the
    ground, 12 cloud base (m), 13 top, 14 number of bubbles, 17 coverage, 18 maximum extinction
    (1/m), 19 period of the base noise (m), 20 period of the detail noise, 21 erosion, 22-24 noise offset (m, wind),
    25 forward and 26 backward asymmetry, 27 weight of the forward lobe, 28 frame number, 29 distance of the aerial
    perspective (m), 30 sharpness of the base (fraction of the height), 31 number of steps, 32 attenuation and 33 weight
    of each multiple-scattering octave, 34 number of octaves, 35 weight of the ambient light, 36-41 box around the
    cloud (x, y, z low then high).

    The cloud mask is the union of the bubbles `bub` (x, y, z, radius), each one sitting on a full column down to the
    flat base: a solid cloud under a lumpy top, like the towers of thermals."""
    py, px = gs.index()
    hh = bg_r.shape[0]
    ww = bg_r.shape[1]
    ax = ((float(px) + 0.5) / float(ww) * 2.0 - 1.0) * float(ww) / float(hh) * cam[12]
    ay = (1.0 - (float(py) + 0.5) / float(hh) * 2.0) * cam[12]
    rx = cam[3] + cam[6] * ax + cam[9] * ay
    ry = cam[4] + cam[7] * ax + cam[10] * ay
    rz = cam[5] + cam[8] * ax + cam[11] * ay
    norm = gs.sqrt(rx * rx + ry * ry + rz * rz)
    rx = rx / norm
    ry = ry / norm
    rz = rz / norm
    br = bg_r[py, px]
    bgc = bg_g[py, px]
    bb = bg_b[py, px]
    zb = prm[12]
    zt = prm[13]
    nbub = int(prm[14])
    # Box around the cloud
    lo = gs.local(gs.f32, 3)
    hi = gs.local(gs.f32, 3)
    for a in range(3):
        lo[a] = prm[36 + a]
        hi[a] = prm[39 + a]
    t0 = 0.0
    t1 = 1e30
    for a in range(3):
        o = cam[a]
        d = rx
        if a == 1:
            d = ry
        elif a == 2:
            d = rz
        if gs.abs(d) < 1e-9:
            if o < lo[a] or o > hi[a]:
                t1 = -1.0
        else:
            ta = (lo[a] - o) / d
            tb = (hi[a] - o) / d
            t0 = gs.max(t0, gs.min(ta, tb))
            t1 = gs.min(t1, gs.max(ta, tb))
    if rz < 0.0:
        t1 = gs.min(t1, -cam[2] / rz)
    if t1 <= t0:
        return br, bgc, bb
    nb = base.shape[0]
    nd = detail.shape[0]
    sx = prm[0]
    sy = prm[1]
    sz = prm[2]
    cos_t = rx * sx + ry * sy + rz * sz
    sig_max = prm[18]
    coverage = prm[17]
    erosion = prm[21]
    sharp = prm[30]
    # Phase function: two Henyey-Greenstein lobes
    g1 = prm[25]
    g2 = prm[26]
    wf = prm[27]
    # Jitter of the first step (hash of the pixel and of the frame)
    hsh = gs.u32(px) * gs.u32(1973) + gs.u32(py) * gs.u32(9277) + gs.u32(int(prm[28])) * gs.u32(26699)
    hsh = (hsh ^ (hsh >> gs.u32(13))) * gs.u32(1274126177)
    hsh = hsh ^ (hsh >> gs.u32(16))
    jitter = float(hsh & gs.u32(65535)) / 65536.0
    steps = int(prm[31])
    dt = (t1 - t0) / float(steps)
    tr = 1.0
    lr = 0.0
    lg = 0.0
    lb = 0.0
    depth = 0.0
    weight = 0.0
    for it in range(256):
        if it >= steps or tr < 0.003:
            break
        s = t0 + (float(it) + jitter) * dt
        x = cam[0] + rx * s
        y = cam[1] + ry * s
        z = cam[2] + rz * s
        h = (z - zb) / (zt - zb)
        # Cloud mask: union of the bubbles, flat base
        field = 0.0
        for bi in range(160):
            if bi >= nbub:
                break
            ex0 = x - bub[bi, 0]
            ey0 = y - bub[bi, 1]
            ez0 = gs.max(z - bub[bi, 2], 0.0)
            rb = bub[bi, 3]
            field = gs.max(field, 1.0 - (ex0 * ex0 + ey0 * ey0 + ez0 * ez0) / (rb * rb))
        mask = gs.min(gs.max(field * 1.25, 0.0), 1.0) * coverage
        grad = gs.min(gs.max(h / sharp, 0.0), 1.0)
        dens = 0.0
        if mask * grad > 0.0:
            # Base noise (trilinear, periodic)
            u = (x + prm[22]) / prm[19] * float(nb)
            v = (y + prm[23]) / prm[19] * float(nb)
            q = (z + prm[24]) / prm[19] * float(nb)
            i0 = int(gs.floor(u))
            j0 = int(gs.floor(v))
            k0 = int(gs.floor(q))
            fu = u - float(i0)
            fv = v - float(j0)
            fq = q - float(k0)
            i0 = ((i0 % nb) + nb) % nb
            j0 = ((j0 % nb) + nb) % nb
            k0 = ((k0 % nb) + nb) % nb
            i1 = (i0 + 1) % nb
            j1 = (j0 + 1) % nb
            k1 = (k0 + 1) % nb
            c00 = base[k0, j0, i0] * (1.0 - fu) + base[k0, j0, i1] * fu
            c01 = base[k0, j1, i0] * (1.0 - fu) + base[k0, j1, i1] * fu
            c10 = base[k1, j0, i0] * (1.0 - fu) + base[k1, j0, i1] * fu
            c11 = base[k1, j1, i0] * (1.0 - fu) + base[k1, j1, i1] * fu
            noise = (c00 * (1.0 - fv) + c01 * fv) * (1.0 - fq) + (c10 * (1.0 - fv) + c11 * fv) * fq
            shape = (noise * grad - (1.0 - mask)) / gs.max(mask, 1e-3)
            if shape > 0.0:
                # Detail noise: wispy at the bottom, billowy higher up
                u = (x + prm[22]) / prm[20] * float(nd)
                v = (y + prm[23]) / prm[20] * float(nd)
                q = (z + prm[24]) / prm[20] * float(nd)
                i0 = int(gs.floor(u))
                j0 = int(gs.floor(v))
                k0 = int(gs.floor(q))
                fu = u - float(i0)
                fv = v - float(j0)
                fq = q - float(k0)
                i0 = ((i0 % nd) + nd) % nd
                j0 = ((j0 % nd) + nd) % nd
                k0 = ((k0 % nd) + nd) % nd
                i1 = (i0 + 1) % nd
                j1 = (j0 + 1) % nd
                k1 = (k0 + 1) % nd
                c00 = detail[k0, j0, i0] * (1.0 - fu) + detail[k0, j0, i1] * fu
                c01 = detail[k0, j1, i0] * (1.0 - fu) + detail[k0, j1, i1] * fu
                c10 = detail[k1, j0, i0] * (1.0 - fu) + detail[k1, j0, i1] * fu
                c11 = detail[k1, j1, i0] * (1.0 - fu) + detail[k1, j1, i1] * fu
                det = (c00 * (1.0 - fv) + c01 * fv) * (1.0 - fq) + (c10 * (1.0 - fv) + c11 * fv) * fq
                mix = gs.min(gs.max(h * 4.0, 0.0), 1.0)
                det = det * (1.0 - mix) + (1.0 - det) * mix
                dens = gs.min(gs.max((shape - det * erosion) / gs.max(1.0 - det * erosion, 1e-3), 0.0), 1.0)
        if dens > 0.0:
            sigma = dens * sig_max
            # Optical depth toward the sun: six steps of growing length, mass noise only
            tau = 0.0
            seg = 25.0
            lx = x
            ly = y
            lz = z
            for l in range(6):
                lx += sx * seg * 0.5
                ly += sy * seg * 0.5
                lz += sz * seg * 0.5
                lh = (lz - zb) / (zt - zb)
                if lh < 1.0:
                    lfield = 0.0
                    for bj in range(160):
                        if bj >= nbub:
                            break
                        fx0 = lx - bub[bj, 0]
                        fy0 = ly - bub[bj, 1]
                        fz0 = gs.max(lz - bub[bj, 2], 0.0)
                        rl = bub[bj, 3]
                        lfield = gs.max(lfield, 1.0 - (fx0 * fx0 + fy0 * fy0 + fz0 * fz0) / (rl * rl))
                    lmask = gs.min(gs.max(lfield * 1.25, 0.0), 1.0) * coverage
                    lgrad = gs.min(gs.max(lh / sharp, 0.0), 1.0)
                    if lmask * lgrad > 0.0:
                        u = (lx + prm[22]) / prm[19] * float(nb)
                        v = (ly + prm[23]) / prm[19] * float(nb)
                        q = (lz + prm[24]) / prm[19] * float(nb)
                        i0 = int(gs.floor(u))
                        j0 = int(gs.floor(v))
                        k0 = int(gs.floor(q))
                        fu = u - float(i0)
                        fv = v - float(j0)
                        fq = q - float(k0)
                        i0 = ((i0 % nb) + nb) % nb
                        j0 = ((j0 % nb) + nb) % nb
                        k0 = ((k0 % nb) + nb) % nb
                        i1 = (i0 + 1) % nb
                        j1 = (j0 + 1) % nb
                        k1 = (k0 + 1) % nb
                        c00 = base[k0, j0, i0] * (1.0 - fu) + base[k0, j0, i1] * fu
                        c01 = base[k0, j1, i0] * (1.0 - fu) + base[k0, j1, i1] * fu
                        c10 = base[k1, j0, i0] * (1.0 - fu) + base[k1, j0, i1] * fu
                        c11 = base[k1, j1, i0] * (1.0 - fu) + base[k1, j1, i1] * fu
                        lnoise = (c00 * (1.0 - fv) + c01 * fv) * (1.0 - fq) + (c10 * (1.0 - fv) + c11 * fv) * fq
                        lshape = gs.min(gs.max((lnoise * lgrad - (1.0 - lmask)) / gs.max(lmask, 1e-3), 0.0), 1.0)
                        tau += lshape * sig_max * seg
                lx += sx * seg * 0.5
                ly += sy * seg * 0.5
                lz += sz * seg * 0.5
                seg *= 2.0
            # Multiple scattering: octaves of Wrenninge (a = b = c = 0.5)
            ms = 0.0
            att = 1.0
            con = 1.0
            ecc = 1.0
            for octave in range(8):
                if octave >= int(prm[34]):
                    break
                ga = g1 * ecc
                gb = g2 * ecc
                pa = (1.0 - ga * ga) / (12.566371 * gs.pow(gs.max(1.0 + ga * ga - 2.0 * ga * cos_t, 1e-4), 1.5))
                pb = (1.0 - gb * gb) / (12.566371 * gs.pow(gs.max(1.0 + gb * gb - 2.0 * gb * cos_t, 1e-4), 1.5))
                ms += con * (wf * pa + (1.0 - wf) * pb) * gs.exp(-tau * att)
                att *= prm[32]
                con *= prm[33]
                ecc *= 0.5
            hgt = gs.min(gs.max(h, 0.0), 1.0)
            amb_r = prm[9] + (prm[6] - prm[9]) * hgt
            amb_g = prm[10] + (prm[7] - prm[10]) * hgt
            amb_b = prm[11] + (prm[8] - prm[11]) * hgt
            sr = prm[3] * ms + amb_r * prm[35]
            sg = prm[4] * ms + amb_g * prm[35]
            sb = prm[5] * ms + amb_b * prm[35]
            ext = gs.exp(-sigma * dt)
            # Energy-conserving integration over the step (Hillaire 2016): scattering albedo 1
            part = tr * (1.0 - ext)
            lr += part * sr
            lg += part * sg
            lb += part * sb
            depth += part * s
            weight += part
            tr *= ext
    if weight <= 0.0:
        return br, bgc, bb
    # Aerial perspective: the cloud fades into the background with distance
    fade = gs.exp(-(depth / weight) / prm[29])
    alpha = 1.0 - tr
    out_r = br * tr + (lr * fade + br * alpha * (1.0 - fade))
    out_g = bgc * tr + (lg * fade + bgc * alpha * (1.0 - fade))
    out_b = bb * tr + (lb * fade + bb * alpha * (1.0 - fade))
    return out_r, out_g, out_b
