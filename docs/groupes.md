# Groupes de threads et mémoire partagée

Sur un GPU, les threads sont regroupés : ceux d'un même groupe peuvent partager une petite mémoire rapide et
s'attendre les uns les autres. C'est ce qui permet les réductions, les transformées (FFT, NTT), les tuiles de
produits de matrices. Par défaut GPUsync forme les groupes lui-même ; `group=` les rend visibles.

## `group=`, `gs.shared`, `gs.lane`, `gs.barrier`

`@gs.kernel(group=g)` (kernels avec `gs.index()`, 1 ≤ g ≤ 1024) regroupe `g` cases **consécutives** de la sortie
(dans l'ordre de NumPy, un groupe peut couvrir plusieurs lignes). Dans le kernel :

| | |
|---|---|
| `t = gs.shared(type, taille)` | tableau commun au groupe, mis à zéro, de taille constante |
| `gs.lane()` | rang du thread dans son groupe, de 0 à g − 1 |
| `gs.barrier()` | attend tout le groupe ; ses écritures dans `gs.shared` sont ensuite visibles |

```python
import numpy as np
import gpusync as gs

@gs.kernel(group=256)
def group_sum(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    r = gs.lane()
    t = gs.shared(gs.f32, 256)
    t[r] = x[i]
    gs.barrier()
    s = 128
    while s > 0:                     # réduction en arbre : 8 étapes pour 256 valeurs
        if r < s:
            t[r] = t[r] + t[r + s]
        gs.barrier()
        s = s // 2
    return t[0]                      # chaque case reçoit la somme de son groupe

x = np.random.rand(4096).astype(np.float32)
sums = np.asarray(group_sum(x)).reshape(16, 256)
assert np.allclose(sums[:, 0], x.reshape(16, 256).sum(axis=1), rtol=1e-5)
```

Règles :

- **La sortie compte un multiple de g cases** ; sinon l'appel est refusé avec un message clair.
- **Une barrière est atteinte par tout le groupe.** GPUsync le vérifie à la compilation : une barrière sous une
  condition ou dans une boucle qui dépend du thread (de `gs.index()`, `gs.lane()`, d'une lecture de `gs.shared`
  ou d'un calcul qui en découle), ou après un `return` exécuté par une partie des threads, est refusée avec sa
  ligne. Sur GPU, l'erreur donnerait un blocage ou un résultat faux.
- **Pas d'accès concurrent sans barrière.** Entre deux barrières, une case de `gs.shared` écrite par un thread
  ne doit pas être lue ou écrite par un autre. Sur GPU, l'erreur passe en silence ; sur `cpu:0`, l'interpréteur
  la signale (« accès concurrent à t[12] : le thread 3 l'écrit et le thread 7 le lit… »). Exécutez un nouveau
  kernel à groupes une fois sur `cpu:0` (`device="cpu"`).
- Les limites dépendent du périphérique : `gs.Device.max_group` (threads par groupe) et `shared_bytes` (mémoire
  partagée par groupe). Exemples : RTX 4080 SUPER 1 024 threads et 48 Kio ; Apple M5 Pro 1 024 threads en Metal,
  256 en OpenCL, 32 Kio. Les dépasser lève `gs.UnsupportedFeature`.

## Plusieurs cases par thread : `cells=` et `gs.store`

`@gs.kernel(group=g, cells=c)` lance c fois moins de threads. Un groupe possède une tuile de g × c cases
consécutives ; le thread de rang r écrit ses cases r, r + g, r + 2g… par `gs.store(k, valeur)` (k de 0 à c − 1),
et ne renvoie rien. `gs.index()` donne les coordonnées de sa première case. C'est ce qui permet de traiter en
mémoire partagée plus de points qu'il n'y a de threads dans un groupe (une transformée de 8 192 points avec
1 024 threads).

```python
@gs.kernel(group=128, cells=4)
def reverse_tiles(x: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()                   # première case du thread : début de tuile + rang
    r = gs.lane()
    start = i - r
    t = gs.shared(gs.f32, 512)
    for k in range(4):
        t[r + k * 128] = x[start + r + k * 128]
    gs.barrier()
    for k in range(4):
        gs.store(k, t[511 - (r + k * 128)])   # chaque tuile de 512 valeurs, retournée

x = np.arange(1024, dtype=np.float32)
y = np.asarray(reverse_tiles(x))
assert np.array_equal(y, x.reshape(2, 512)[:, ::-1].ravel())
```

`cells=` demande une seule sortie, dont le nombre de cases est un multiple de g × c.

## Produit de matrices d'octets : `gs.matmul_u8`

`gs.matmul_u8(C, A, B, m, n, k)` est une opération **collective** (tout le groupe l'appelle, comme une
barrière) sur des tableaux `gs.shared` de `gs.u32` : `A` contient m lignes de k octets, `B` n colonnes de k
octets, rangés 4 octets par mot (octet de poids faible en premier), et
`C[i·n + j] += Σₜ A[i, t] · B[j, t]`, modulo 2³².

En CUDA, sur une carte Ampere ou plus récente, avec m, n, k multiples de 16, 8 et 32 et un groupe de 32 threads,
le calcul passe par les **Tensor Cores** (mesuré : 23,6 TOPS sur RTX 4080 SUPER, contre 1,4 à 1,7 par la boucle
portable). Ailleurs, une boucle portable donne exactement le même résultat.

```python
@gs.kernel(group=32)
def byte_matmul(a: gs.u32["ma"], b: gs.u32["nb"], ref: gs.u32["out"]) -> gs.u32["out"]:
    i = gs.index()
    r = gs.lane()
    sa = gs.shared(gs.u32, 128)      # 16 lignes × 32 octets = 128 mots
    sb = gs.shared(gs.u32, 64)       # 8 colonnes × 32 octets = 64 mots
    sc = gs.shared(gs.u32, 128)      # 16 × 8 résultats
    for q in range(4):
        sa[r + 32 * q] = a[r + 32 * q]
    for q in range(2):
        sb[r + 32 * q] = b[r + 32 * q]
    gs.matmul_u8(sc, sa, sb, 16, 8, 32)
    return sc[i % 128]

A = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
B = np.random.randint(0, 256, (8, 32), dtype=np.uint8)
C = byte_matmul(A.view(np.uint32).ravel(), B.view(np.uint32).ravel(), np.zeros(128, np.uint32))
assert np.array_equal(np.asarray(C).reshape(16, 8), (A.astype(np.uint64) @ B.T.astype(np.uint64)).astype(np.uint32))
```

Le paramètre `ref` ne sert qu'à donner la taille de la sortie (128 cases, soit 4 groupes qui refont chacun le
même produit).
