# Résultats et gros volumes

## `gs.Array` : un résultat qui reste sur le GPU

Appeler un kernel **met le calcul en file et rend la main tout de suite**. Le résultat est un `gs.Array` : une
référence vers des données encore sur le périphérique.

```python
import numpy as np
import gpusync as gs

@gs.kernel
def scale(x: gs.f32[:], k: gs.f32) -> gs.f32[:]:
    return x * k

@gs.kernel
def offset(x: gs.f32[:], c: gs.f32) -> gs.f32[:]:
    return x + c

x = np.arange(6, dtype=np.float32)
y = scale(x, 2.0)              # rend la main aussitôt
z = offset(y, 1.0)             # chaînage : y ne repasse pas par la mémoire centrale
print(z.shape, z.dtype, z.device)   # connus sans attendre le calcul
print(z[:3])                   # lecture : attend la fin et rapatrie
```

| Sur un `gs.Array` | Effet |
|---|---|
| le passer à un autre kernel | aucune attente, aucune copie si c'est le même périphérique |
| `y.shape`, `y.dtype`, `y.ndim`, `y.size`, `y.nbytes`, `y.device` | immédiat |
| `np.asarray(y)`, `y.numpy()`, `y[...]`, `len(y)`, `print(y)`, itération | attend la fin du calcul, puis rapatrie (une seule fois) |
| `y.sum()`, `y.mean()`, `y.min()`, `y.max()`, `y.std()` (avec `axis=`) | réductions calculées par NumPy après lecture |
| `y + 1`, `y == 2`, `np.sqrt(y)`… | opérateurs et fonctions de NumPy, sur l'hôte après lecture : rendent un tableau NumPy |
| `await y` | attend sans bloquer la boucle asyncio |
| `y.save("r.npy")` | écrit un fichier `.npy`, par blocs depuis le GPU |
| `np.from_dlpack(y)`, `torch.from_dlpack(y)` | export DLPack de la mémoire hôte, sans copie |
| `y.ready` | `True` si le résultat a déjà été lu |
| `y.mode` | comment l'appel a été exécuté (voir plus bas) |

**Les opérateurs lisent le résultat.** `z + 1`, `z * z`, `z == 2`, `-z`, `np.maximum(z, 0)` attendent la fin du
calcul, rapatrient, et rendent un **tableau NumPy** calculé sur l'hôte. C'est commode pour vérifier ou
post-traiter ; pour que le calcul reste sur le GPU, écrivez un kernel.

```python
r = z + 1                      # np.ndarray
assert isinstance(r, np.ndarray) and r[0] == 2.0
assert (z == 1.0).sum() == 1   # comparaison valeur par valeur
```

**Ne modifiez pas un tableau d'entrée avant d'avoir lu le résultat.** Sur un GPU à mémoire unifiée (Mac, GPU
intégrés), le GPU lit directement la mémoire du tableau NumPy, sans copie : la changer pendant le calcul change
le résultat.

## Écrire dans un tableau donné : `out=`

```python
dest = np.empty(6, np.float32)
r = scale(x, 3.0, out=dest)    # le résultat est écrit dans dest
r.numpy()                      # attend la fin
assert dest[5] == 15.0
```

Un kernel à plusieurs sorties prend un tuple : `out=(a, b)`. La destination doit avoir la forme et le type
exacts de la sortie, être contiguë et modifiable.

## Sans limite de taille

Les entrées et les sorties peuvent être plus grandes que la mémoire du GPU, ou que la mémoire centrale.
GPUsync découpe alors le calcul en blocs et recouvre les transferts par le calcul ; le kernel ne change pas.

```python
n = 1_000_000
big = np.lib.format.open_memmap("entree.npy", mode="w+", dtype=np.float32, shape=(n,))
big[:] = 1.0
out = np.lib.format.open_memmap("sortie.npy", mode="w+", dtype=np.float32, shape=(n,))
scale(big, 2.0, out=out).numpy()       # fichier → GPU → fichier, par blocs
assert out[-1] == 2.0
```

Pour un résultat plus grand que la mémoire centrale, donnez toujours une destination `out=` (un `np.memmap`) :
sans elle, la sortie doit être allouée en mémoire, et `gs.OutOfMemory` le dit si c'est impossible.

Dans un kernel avec index, la sortie est découpée par blocs de lignes (son premier axe). Une entrée lue
seulement à la ligne courante (`x[i, k]` dans un kernel où `i` est la ligne) est découpée avec elle ; les autres
entrées restent entières sur le GPU et doivent y tenir. Mesuré : 19,3 Gio de distances deux à deux calculées sur
une RTX de 16 Go en 3,8 s.

`y.save()` écrit un résultat resté sur le GPU directement dans un fichier, par blocs :

```python
scale(x, 2.0).save("double.npy")
assert np.load("double.npy")[5] == 10.0
```

### Flux de blocs : `kernel.stream()`

Pour des données qui arrivent par morceaux (lecture d'un capteur, d'un fichier découpé), `stream` prend un
itérable de blocs et rend un itérateur de résultats NumPy. Le bloc suivant est lancé avant que le résultat du
bloc courant soit rendu : le calcul et votre traitement se recouvrent.

```python
def blocs():
    for k in range(4):
        yield np.full(1000, k, np.float32)

total = 0.0
for res in scale.stream(blocs(), 0.5):     # les arguments suivants complètent chaque appel
    total += float(res.sum())
assert total == 0.5 * 1000 * (0 + 1 + 2 + 3)
```

Un élément du flux peut être un tuple : ce sont alors les premiers arguments du kernel.

## Modes d'exécution

Chaque appel choisit lui-même le mode le plus économe, visible dans `y.mode` :

| Mode | Quand | Mémoire en plus |
|---|---|---|
| `direct` | GPU à mémoire unifiée, entrées NumPy : le GPU travaille dans la mémoire des tableaux | ≈ 0 |
| `resident` | l'appel tient dans le budget du périphérique (50 % de sa mémoire par défaut) ; ou une entrée est déjà un résultat sur le GPU | bornée par le budget |
| `flux` | au-delà : par blocs, transferts recouverts par le calcul | ≈ 0,3 à 1,2 Gio |
| `cpu` | interpréteur `cpu:0` | |

Le budget se règle par `gs.set_memory_budget(octets)` (voir [Périphériques](peripheriques.md)).

## Asynchrone : `await`, threads, Jupyter

`await y` attend la fin du calcul sur un thread, sans bloquer la boucle asyncio ; un kernel à plusieurs sorties
s'attend en une fois. Le même code marche dans un script, dans Jupyter (qui a déjà une boucle) et dans un
serveur asynchrone.

```python
import asyncio

@gs.kernel
def minmax(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    lo = x[i, 0]
    hi = x[i, 0]
    for k in range(x.shape[1]):
        lo = gs.min(lo, x[i, k])
        hi = gs.max(hi, x[i, k])
    return lo, hi

async def main():
    y = await scale(x, 2.0)                     # y est prêt
    lo, hi = await minmax(np.random.rand(8, 50).astype(np.float32))
    # un gros appel découpé par blocs bloque son thread : le sortir de la boucle
    big2 = await asyncio.to_thread(scale, big, 3.0, out=out)
    return y, lo, hi

y, lo, hi = asyncio.run(main())
assert (np.asarray(lo) <= np.asarray(hi)).all()
```

Les appels de kernels sont sûrs depuis plusieurs threads.

## Interopérabilité

`gs.Array` implémente DLPack (export en mémoire hôte, sans copie) ; un kernel accepte en entrée tout objet
DLPack en mémoire hôte (tenseur PyTorch CPU, tableau JAX…).

```py
import torch
t = torch.from_dlpack(scale(x, 2.0))              # tenseur CPU, sans copie
scale(torch.ones(10), 2.0)                        # entrée PyTorch acceptée
scale(torch.ones(10, device="cuda"), 2.0)         # refusé : ramener d'abord en mémoire hôte (.cpu())
```

## Plusieurs kernels en une soumission : `gs.batch` et `gs.sync`

Sur Metal, Vulkan et DX12, chaque appel est une soumission au GPU, qui coûte quelques dizaines de
microsecondes. Pour une boucle de dizaines de petits kernels (une simulation, un rendu), `gs.batch()` les
regroupe en une seule soumission, envoyée à la sortie du bloc :

```python
a = np.zeros(100_000, np.float32)
y = scale(a, 1.0)
with gs.batch():
    for _ in range(20):
        y = offset(y, 1.0)                     # enregistré, pas encore soumis
gs.sync()                                      # attend la fin, sans rien rapatrier
assert np.asarray(y)[0] == 20.0
```

- Une lecture (`y.numpy()`…) à l'intérieur du bloc soumet ce qui précède.
- Sur OpenCL, CUDA et `cpu:0`, le bloc est sans effet (le code reste correct partout).
- `gs.sync(device)` attend la fin de tout ce qui a été soumis à un périphérique. Dans une longue boucle qui ne
  lit jamais ses résultats, appelez-le de temps en temps (toutes les quelques itérations) : sinon le travail en
  attente, et les tampons qu'il retient, s'accumulent dans la mémoire du GPU.
