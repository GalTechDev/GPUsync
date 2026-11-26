"""SPH faiblement compressible sur grille triée.

- Grille uniforme par tri (Green, « CUDA Particles », NVIDIA) : cellule de côté `h`, 27 cellules voisines.
- Noyau spline cubique, équation d'état, pression symétrique, viscosité et pas de temps (Koschier et al., « SPH Techniques for the
  Physics Based Simulation of Fluids and Solids », 2019).
- Un seul noyau pour la densité et la force : avec poly6 et spiky (Müller et al., SCA 2003), la force ne dérive
  pas d'une énergie et l'eau s'agite sans fin (0,2 m/s mesurés au repos).
- Parois : demi-espaces d'eau au repos, intégrés exactement (ni particules de bord, ni densité manquante).

Unités : masse volumique au repos 1, donc `m = espacement³` et les densités calculées sont relatives.
"""

import gpusync as gs

SORT_TILE = 256
SORT_TILE_BITS = 8
# Clé de tri : cellule << 18 | particule, donc 262 144 particules et 16 384 cellules au plus


@gs.kernel
def sph_cell_keys(
    pos: gs.f32["n", "c4"],
    key_tpl: gs.u32["np"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    cell: gs.f32,
    bx: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
) -> gs.u32["np"]:
    """Clé de tri de chaque particule ; les cases au-delà de `n` (puissance de deux) restent en fin de liste."""
    i = gs.index()
    if i >= pos.shape[0]:
        return gs.u32(4294967295)
    inv_c = 1.0 / cell
    ix = gs.min(gs.max(int((pos[i, 0] + bx) * inv_c), 0), nx - 1)
    iy = gs.min(gs.max(int((pos[i, 1] - floor_y) * inv_c), 0), ny - 1)
    iz = gs.min(gs.max(int((pos[i, 2] + bz) * inv_c), 0), nz - 1)
    return (gs.u32((iz * ny + iy) * nx + ix) << gs.u32(18)) | gs.u32(i) | (key_tpl[i] & gs.u32(0))


@gs.kernel
def bitonic_steps(
    keys: gs.u32["np"],
    m0: gs.i32,
    m1: gs.i32,
    m2: gs.i32,
    m3: gs.i32,
) -> gs.u32["np"]:
    """Quatre étapes de tri bitonique en un lancement (masque 0 : étape vide).

    À chaque étape, la case `a` se compare à `a ^ masque` et la plus petite des deux adresses garde le minimum.
    Une case ne dépend donc que des 16 adresses `i ^ (xor d'une partie des masques)`."""
    i = gs.index()
    v = gs.local(gs.u32, 16)
    addr = gs.local(gs.i32, 16)
    for s in range(16):
        a = i
        if (s & 1) != 0:
            a = a ^ m0
        if (s & 2) != 0:
            a = a ^ m1
        if (s & 4) != 0:
            a = a ^ m2
        if (s & 8) != 0:
            a = a ^ m3
        addr[s] = a
        v[s] = keys[a]
    for step in range(4):
        # Après une étape, seules les adresses sans ce masque servent encore
        bit = 1 << step
        for s in range(16):
            if (s & (2 * bit - 1)) == 0:
                t = s | bit
                if addr[s] <= addr[t]:
                    v[s] = gs.min(v[s], v[t])
                else:
                    v[s] = gs.max(v[s], v[t])
    return v[0]


@gs.kernel(group=SORT_TILE)
def sort_tiles(keys: gs.u32["np"]) -> gs.u32["np"]:
    """Trie chaque tuile de 256 clés en mémoire partagée."""
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.u32, SORT_TILE)
    t[r] = keys[i]
    gs.barrier()
    for p in range(1, SORT_TILE_BITS + 1):
        for q in range(p):
            # Première étape d'un niveau : en miroir ; les suivantes par moitiés
            mask = (1 << p) - 1
            if q > 0:
                mask = 1 << (p - 1 - q)
            o = r ^ mask
            v = gs.min(t[r], t[o])
            if r > o:
                v = gs.max(t[r], t[o])
            gs.barrier()
            t[r] = v
            gs.barrier()
    return t[r]


@gs.kernel(group=SORT_TILE)
def merge_tiles(keys: gs.u32["np"]) -> gs.u32["np"]:
    """Fin d'un niveau de fusion : les étapes internes à chaque tuile."""
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.u32, SORT_TILE)
    t[r] = keys[i]
    gs.barrier()
    for q in range(SORT_TILE_BITS):
        o = r ^ (1 << (SORT_TILE_BITS - 1 - q))
        v = gs.min(t[r], t[o])
        if r > o:
            v = gs.max(t[r], t[o])
        gs.barrier()
        t[r] = v
        gs.barrier()
    return t[r]


@gs.kernel(grid=("np",))
def sph_cell_ranges(
    keys: gs.u32["np"],
    cell_tpl: gs.f32["nc"],
    start: gs.Out[gs.u32["nc"], 4294967295],
    count: gs.Out[gs.u32["nc"], 0],
):
    """Première particule et nombre de particules de chaque cellule, dans la liste triée."""
    i = gs.index()
    k = keys[i]
    if k != gs.u32(4294967295):
        c = int(k >> gs.u32(18))
        gs.atomic_min(start, c, gs.u32(i) + gs.u32(cell_tpl[0] * 0.0))
        gs.atomic_add(count, c, gs.u32(1))


@gs.kernel
def sph_gather(
    src: gs.f32["n", "c4"],
    keys: gs.u32["np"],
) -> gs.f32["n", "c4"]:
    """Recopie un tableau dans l'ordre des cellules (voisins proches en mémoire)."""
    i, c = gs.index()
    return src[int(keys[i] & gs.u32(262143)), c]


@gs.kernel
def sph_density(
    spos: gs.f32["n", "c4"],
    start: gs.u32["nc"],
    count: gs.u32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    cell: gs.f32,
    h: gs.f32,
    mass: gs.f32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
) -> gs.f32["n"]:
    """Densité relative (spline cubique), eau fictive derrière les parois comprise."""
    i = gs.index()
    px = spos[i, 0]
    py = spos[i, 1]
    pz = spos[i, 2]
    inv_h = 1.0 / h
    inv_c = 1.0 / cell
    h2 = h * h
    ix = gs.min(gs.max(int((px + bx) * inv_c), 0), nx - 1)
    iy = gs.min(gs.max(int((py - floor_y) * inv_c), 0), ny - 1)
    iz = gs.min(gs.max(int((pz + bz) * inv_c), 0), nz - 1)
    total = 0.0
    for oz in range(-1, 2):
        cz = iz + oz
        if cz >= 0 and cz < nz:
            for oy in range(-1, 2):
                cy = iy + oy
                if cy >= 0 and cy < ny:
                    for ox in range(-1, 2):
                        cx = ix + ox
                        if cx >= 0 and cx < nx:
                            cid = (cz * ny + cy) * nx + cx
                            cnt = int(count[cid])
                            if cnt > 0:
                                first = int(start[cid])
                                for j in range(first, first + cnt):
                                    ex = px - spos[j, 0]
                                    ey = py - spos[j, 1]
                                    ez = pz - spos[j, 2]
                                    d2 = ex * ex + ey * ey + ez * ez
                                    if d2 < h2:
                                        q = gs.sqrt(d2) * inv_h
                                        u = 1.0 - q
                                        if q < 0.5:
                                            total += 1.0 - 6.0 * q * q * u
                                        else:
                                            total += 2.0 * u * u * u
    # Spline cubique : 8 / (π h³)
    rho = total * mass * 2.5464791 * inv_h * inv_h * inv_h
    # Intégrale de la spline sur le demi-espace situé derrière chaque paroi, à la distance x·h
    for w in range(5):
        dist = py - floor_y
        if w == 0:
            dist = px + bx
        elif w == 1:
            dist = bx - px
        elif w == 2:
            dist = pz + bz
        elif w == 3:
            dist = bz - pz
        x = gs.max(dist * inv_h, 0.0)
        if x < 1.0:
            u = 1.0 - x
            if x >= 0.5:
                rho += u * u * u * u * u * (1.6 - 1.0666667 * u)
            else:
                # 1/60 : valeur en x = 0,5 ; puis primitive de la branche centrale
                x2 = x * x
                prim = x2 * x * (0.1666667 + x2 * (x * 0.2 - 0.3))
                rho += 0.0333333 + 16.0 * (0.0875 * (0.5 - x) - (0.0145833 - prim))
    return rho


@gs.kernel
def sph_integrate(
    pos: gs.f32["n", "c4"],
    vel: gs.f32["n", "c4"],
    spos: gs.f32["n", "c4"],
    svel: gs.f32["n", "c4"],
    rho: gs.f32["n"],
    keys: gs.u32["np"],
    start: gs.u32["nc"],
    count: gs.u32["nc"],
    nx: gs.i32,
    ny: gs.i32,
    nz: gs.i32,
    dt: gs.f32,
    cell: gs.f32,
    h: gs.f32,
    mass: gs.f32,
    sound: gs.f32,
    visc: gs.f32,
    cohesion: gs.f32,
    grav_x: gs.f32,
    grav_y: gs.f32,
    grav_z: gs.f32,
    bx: gs.f32,
    by: gs.f32,
    bz: gs.f32,
    floor_y: gs.f32,
    restitution: gs.f32,
    damping: gs.f32,
    max_speed: gs.f32,
) -> (gs.f32["n", "c4"], gs.f32["n", "c4"]):
    """Pression symétrique, viscosité, cohésion, parois, puis Euler symplectique (ordre d'origine conservé)."""
    i, c = gs.index()
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    prad = pos[i, 3]
    vx = vel[i, 0]
    vy = vel[i, 1]
    vz = vel[i, 2]
    inv_h = 1.0 / h
    inv_c = 1.0 / cell
    ix = gs.min(gs.max(int((px + bx) * inv_c), 0), nx - 1)
    iy = gs.min(gs.max(int((py - floor_y) * inv_c), 0), ny - 1)
    iz = gs.min(gs.max(int((pz + bz) * inv_c), 0), nz - 1)

    # Retrouver sa propre densité dans la liste triée
    own = (iz * ny + iy) * nx + ix
    me = int(start[own])
    for j in range(int(start[own]), int(start[own]) + int(count[own])):
        if int(keys[j] & gs.u32(262143)) == i:
            me = j
            break
    c2 = sound * sound
    rho_i = gs.max(rho[me], 0.1)
    # p = c² (ρ/ρ0 - 1), jamais négative
    pr_i = c2 * gs.max(rho_i - 1.0, 0.0) / (rho_i * rho_i)

    ax = grav_x
    ay = grav_y
    az = grav_z
    h2 = h * h
    inv_h3 = inv_h * inv_h * inv_h
    # Même noyau que la densité : sinon la force n'est plus le gradient de l'énergie et l'eau s'agite toute seule
    scale = 15.2788745 * inv_h3 * inv_h
    for oz in range(-1, 2):
        cz = iz + oz
        if cz >= 0 and cz < nz:
            for oy in range(-1, 2):
                cy = iy + oy
                if cy >= 0 and cy < ny:
                    for ox in range(-1, 2):
                        cx = ix + ox
                        if cx >= 0 and cx < nx:
                            cid = (cz * ny + cy) * nx + cx
                            cnt = int(count[cid])
                            if cnt > 0:
                                first = int(start[cid])
                                for j in range(first, first + cnt):
                                    ex = px - spos[j, 0]
                                    ey = py - spos[j, 1]
                                    ez = pz - spos[j, 2]
                                    d2 = ex * ex + ey * ey + ez * ez
                                    if j != me and d2 < h2:
                                        r = gs.sqrt(gs.max(d2, 1e-10))
                                        q = 1.0 - r * inv_h
                                        qr = r * inv_h
                                        grad = scale * q * q
                                        if qr < 0.5:
                                            grad = scale * qr * (2.0 - 3.0 * qr)
                                        rho_j = gs.max(rho[j], 0.1)
                                        pr_j = c2 * gs.max(rho_j - 1.0, 0.0) / (rho_j * rho_j)
                                        f = mass * (pr_i + pr_j) * grad / r
                                        # Cohésion douce au-delà de la distance de repos
                                        f -= cohesion * mass * inv_h3 * q * (1.0 - q) / r
                                        # Viscosité : 2 (d + 2) ν Σ m/ρ (v·x)/(x² + 0,01 h²) ∇W
                                        vdx = vx - svel[j, 0]
                                        vdy = vy - svel[j, 1]
                                        vdz = vz - svel[j, 2]
                                        vr = vdx * ex + vdy * ey + vdz * ez
                                        f -= 10.0 * visc * mass / rho_j * vr / (d2 + 0.01 * h2) * grad / r
                                        ax += ex * f
                                        ay += ey * f
                                        az += ez * f

    # Poussée du demi-espace d'eau fictive derrière chaque paroi : gradient exact de sa part de densité
    for w in range(5):
        dist = py - floor_y
        if w == 0:
            dist = px + bx
        elif w == 1:
            dist = bx - px
        elif w == 2:
            dist = pz + bz
        elif w == 3:
            dist = bz - pz
        x = gs.max(dist * inv_h, 0.0)
        if x < 1.0:
            u = 1.0 - x
            # Intégrale de la spline sur le plan à la distance x·h
            plane_w = 32.0 * u * u * u * u * (0.25 - 0.2 * u)
            if x < 0.5:
                x2 = x * x
                plane_w = 16.0 * (0.0875 - x2 * (0.5 + x2 * (1.2 * x - 1.5)))
            push = 1.0 * pr_i * inv_h * plane_w
            if w == 0:
                ax += push
            elif w == 1:
                ax -= push
            elif w == 2:
                az += push
            elif w == 3:
                az -= push
            else:
                ay += push

    vx = (vx + ax * dt) * damping
    vy = (vy + ay * dt) * damping
    vz = (vz + az * dt) * damping
    sp2 = vx * vx + vy * vy + vz * vz
    if sp2 > max_speed * max_speed:
        k_sp = max_speed / gs.sqrt(sp2)
        vx = vx * k_sp
        vy = vy * k_sp
        vz = vz * k_sp
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    # Une particule représente un cube d'eau : son centre reste près d'un demi-espacement (0,25 h) des parois
    margin = 0.2 * h
    if py < floor_y + margin:
        py = floor_y + margin
        vy = gs.abs(vy) * restitution
        # Frottement sur le fond : 0,9 par seconde
        rub = gs.max(1.0 - 0.9 * dt, 0.0)
        vx = vx * rub
        vz = vz * rub
    elif py > floor_y + by - margin:
        py = floor_y + by - margin
        vy = -gs.abs(vy) * restitution
    if px < -bx + margin:
        px = -bx + margin
        vx = gs.abs(vx) * restitution
    elif px > bx - margin:
        px = bx - margin
        vx = -gs.abs(vx) * restitution
    if pz < -bz + margin:
        pz = -bz + margin
        vz = gs.abs(vz) * restitution
    elif pz > bz - margin:
        pz = bz - margin
        vz = -gs.abs(vz) * restitution

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


def _global_steps(keys: gs.Array, masks: list[int], device: str | None) -> gs.Array:
    masks = masks + [0] * (-len(masks) % 4)
    for k in range(0, len(masks), 4):
        keys = bitonic_steps(keys, *masks[k : k + 4], device=device)
    return keys


def sort_keys(keys: gs.Array, size: int, *, device: str | None = None) -> gs.Array:
    """Trie `size` clés u32 (puissance de deux) en ordre croissant, sur GPU."""
    levels = size.bit_length() - 1
    tiled = levels >= SORT_TILE_BITS
    if tiled:
        keys = sort_tiles(keys, device=device)
    for p in range(SORT_TILE_BITS + 1 if tiled else 1, levels + 1):
        low = SORT_TILE_BITS if tiled else 0
        keys = _global_steps(keys, [(1 << p) - 1] + [1 << q for q in range(p - 2, low - 1, -1)], device)
        if tiled:
            keys = merge_tiles(keys, device=device)
    return keys
