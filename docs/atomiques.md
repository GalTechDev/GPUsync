# Mode explicite et écritures atomiques

## Quand en a-t-on besoin ?

Dans un kernel ordinaire, chaque thread calcule **sa** case de la sortie, et personne d'autre n'y touche. Certains
calculs ne marchent pas comme ça : la case à modifier **dépend de la donnée**.

- Un histogramme : chaque mesure ajoute 1 à la classe où elle tombe.
- Un cumul par maille : chaque goutte de pluie ajoute sa masse à la maille où elle touche le sol.
- Un extremum par zone : chaque station garde la température la plus basse de sa zone.

Des milliers de threads tournent en même temps sur un GPU, et beaucoup vont viser la même case. Or « ajouter 1 »
se fait en trois temps : lire la valeur, ajouter, réécrire. Si deux threads le font ensemble :

| Étape | Thread A | Thread B | Valeur de la case |
|---|---|---|---|
| 1 | lit 7 | | 7 |
| 2 | | lit 7 | 7 |
| 3 | écrit 8 | | 8 |
| 4 | | écrit 8 | **8** au lieu de 9 |

Une mesure est perdue, et pas toujours la même d'un lancement à l'autre. Une **écriture atomique** fait les trois
temps d'un seul bloc : le matériel garantit qu'aucun autre thread ne s'intercale. Si deux threads arrivent
ensemble, l'un attend l'autre, et la case vaut bien 9.

## Partir du Python ordinaire

Voici un histogramme de températures, par classes de 10 °C de −20 à +40 °C, écrit en Python normal :

```python
import numpy as np
import gpusync as gs

def histo_python(t):
    counts = np.zeros(6, np.int32)              # 1. on crée la sortie, à zéro
    for i in range(len(t)):                     # 2. on parcourt les mesures
        classe = int((t[i] + 20.0) / 10.0)      # 3. on calcule la classe
        classe = min(max(classe, 0), 5)
        counts[classe] += 1                     # 4. on ajoute 1
    return counts                               # 5. on rend le résultat
```

Et la même chose en kernel :

```python
@gs.kernel(grid="t")
def histo(t: gs.f32["n"], counts: gs.Out[gs.i32[6]]):
    i = gs.index()
    classe = gs.min(gs.max(int((t[i] + 20.0) / 10.0), 0), 5)
    gs.atomic_add(counts, classe, 1)

temperatures = np.random.uniform(-25.0, 45.0, 10_000).astype(np.float32)
counts = histo(temperatures)                    # un seul argument : voir plus bas
assert np.array_equal(counts, histo_python(temperatures))
```

Ligne à ligne :

| Python ordinaire | Kernel | Explication |
|---|---|---|
| `for i in range(len(t)):` | `grid="t"` et `i = gs.index()` | il n'y a pas de boucle : **chaque tour devient un thread**. `grid="t"` demande un thread par élément de `t` ; `gs.index()` donne à chaque thread son numéro `i` |
| `counts = np.zeros(6, np.int32)` | `counts: gs.Out[gs.i32[6]]` | la sortie est **déclarée dans les paramètres** : « un tableau de 6 entiers, mis à zéro ». GPUsync le crée |
| `min(...)`, `max(...)` | `gs.min(...)`, `gs.max(...)` | dans un kernel, les fonctions s'écrivent avec `gs.` |
| `counts[classe] += 1` | `gs.atomic_add(counts, classe, 1)` | même sens, mais protégé contre les écritures simultanées. `+=` est refusé sur une sortie |
| `return counts` | rien | GPUsync rend lui-même les sorties à la fin |
| `histo_python(temperatures)` | `histo(temperatures)` | à l'appel, on ne passe **que les entrées** |

### Les deux points qui surprennent

**1. La sortie est un paramètre qu'on ne passe pas.** La définition dit `histo(t, counts)`, mais on appelle
`histo(temperatures)`, avec un seul argument. Tout paramètre de type `gs.Out[...]` est une sortie : il est
absent de l'appel, et c'est ce que l'appel rend. Les autres paramètres sont les entrées, dans l'ordre.

```py
@gs.kernel(grid="t")
def k(t: gs.f32["n"], seuil: gs.f32, counts: gs.Out[gs.i32[6]]): ...

counts = k(temperatures, 0.5)       # entrées : t, seuil ; rendu : counts
```

**2. Pas de `return` de valeur.** Le kernel ne calcule pas une case : il agit, par ses écritures atomiques.
Un `return` seul arrête le thread plus tôt (« cette mesure ne compte pas »).

## Ce qu'on écrit dans le kernel

**La grille** dit combien de threads lancer :

- `grid="t"` : un thread par élément de l'entrée `t` (pour un tableau 2D, un par case, et `gs.index()` rend
  deux numéros : `i, j = gs.index()`) ;
- `grid=("n",)` : un thread par valeur de la dimension `n`, utile quand l'entrée a d'autres axes (par exemple
  `xy: gs.f32["n", 2]`, des points à deux coordonnées : un thread par point, pas un par coordonnée).

**Les trois écritures :**

| | Effet | Pour |
|---|---|---|
| `gs.atomic_add(sortie, case, valeur)` | `sortie[case] += valeur` | compter, cumuler |
| `gs.atomic_min(sortie, case, valeur)` | `sortie[case] = min(sortie[case], valeur)` | le plus petit par case |
| `gs.atomic_max(sortie, case, valeur)` | `sortie[case] = max(sortie[case], valeur)` | le plus grand par case |

- Types permis : `gs.i32`, `gs.u32`, `gs.f32`.
- `case` est un entier, ou un tuple pour une sortie à plusieurs dimensions : `(ligne, colonne)`.
- On ne **lit** pas une sortie dans le kernel : un thread ne sait pas où en sont les autres.

## Deux exemples complets

**Pluie cumulée par maille.** Chaque goutte (position dans [0, 1)², masse) ajoute sa masse à sa maille d'une
grille 4 × 4. Sortie à deux dimensions, donc une case en tuple :

```python
@gs.kernel(grid=("n",))                                  # un thread par goutte
def rain_map(xy: gs.f32["n", 2], mass: gs.f32["n"], total: gs.Out[gs.f32[4, 4]]):
    i = gs.index()
    col = gs.min(int(xy[i, 0] * 4.0), 3)
    row = gs.min(int(xy[i, 1] * 4.0), 3)
    gs.atomic_add(total, (row, col), mass[i])

xy = np.random.rand(5000, 2).astype(np.float32)
mass = np.random.rand(5000).astype(np.float32)
total = rain_map(xy, mass)

expected = np.zeros((4, 4), np.float32)
np.add.at(expected, ((xy[:, 1] * 4).astype(int), (xy[:, 0] * 4).astype(int)), mass)
assert np.allclose(total, expected, rtol=1e-4)
```

**Température minimale par zone.** Pour un minimum, la sortie ne doit pas partir de 0 (le minimum serait faux
dès que toutes les valeurs sont positives) mais de +∞ : c'est la **valeur de départ**, en second dans
`gs.Out[type, départ]`.

```python
@gs.kernel(grid="t")
def coldest(t: gs.f32["n"], zone: gs.i32["n"], low: gs.Out[gs.f32[3], gs.inf]):
    i = gs.index()
    gs.atomic_min(low, zone[i], t[i])

t = np.array([12.0, 3.5, 8.0, -1.0, 20.0], np.float32)
zone = np.array([0, 0, 1, 1, 2], np.int32)
assert np.array_equal(coldest(t, zone), [3.5, -1.0, 20.0])
```

Avec plusieurs `gs.Out`, l'appel rend un tuple, dans l'ordre des paramètres : `low, high = k(t, zone)`.

### Une sortie dont la taille dépend des données

Ci-dessus, les tailles sont écrites en dur (`gs.f32[3]`). Pour qu'elles suivent les données, on leur donne le
nom d'une dimension d'**entrée** : la taille est lue sur cette entrée à chaque appel. S'il n'existe pas d'entrée
naturelle de cette taille, on en ajoute une qui ne sert qu'à ça :

```python
@gs.kernel(grid="t")
def coldest_n(t: gs.f32["n"], zone: gs.i32["n"], zones: gs.i32["m"], low: gs.Out[gs.f32["m"], gs.inf]):
    i = gs.index()
    gs.atomic_min(low, zone[i], t[i])

low = coldest_n(t, zone, np.zeros(3, np.int32))    # `zones` n'est pas lue : sa taille donne m = 3
assert np.array_equal(low, [3.5, -1.0, 20.0])
```

## Cumuler sur plusieurs appels : `out=`

Pour des données qui arrivent par paquets, on passe un tableau existant par `out=` : la sortie part de son contenu
au lieu de zéro, et les appels successifs s'y ajoutent.

```python
acc = np.zeros(6, np.int32)
for _ in range(3):                                   # trois paquets de mesures
    paquet = np.random.uniform(-25.0, 45.0, 10_000).astype(np.float32)
    histo(paquet, out=acc).numpy()                   # .numpy() : attendre la fin avant le paquet suivant
assert acc.sum() == 30_000
```

## Une case hors de la sortie

Si un thread vise une case qui n'existe pas (la classe 10 d'un histogramme de 10 classes), l'écriture **n'est
pas faite** : rien n'est écrit ailleurs en mémoire, et les autres écritures le sont normalement. La lecture du
résultat lève ensuite `gs.OutOfBounds`, en nommant la sortie et la case fautive :

```python
@gs.kernel(grid="t")
def careless(t: gs.f32["n"], counts: gs.Out[gs.i32[6]]):
    i = gs.index()
    gs.atomic_add(counts, int((t[i] + 20.0) / 10.0), 1)     # 45 °C donne la classe 6 : hors limites

try:
    careless(np.array([5.0, 45.0], np.float32)).numpy()
    raised = False
except gs.OutOfBounds as e:
    print(e)                                                 # … indice 6 pour counts de forme (6,) …
    raised = True
assert raised
```

Bornez la case (`gs.min(classe, 5)`) ou écartez la donnée par un `if` suivi d'un `return`.

## Coût et précision

- **Plus les threads visent la même case, plus c'est lent**, puisqu'ils passent un par un : six classes se gênent
  bien plus que mille. Les écritures en entiers sont natives partout ; en `gs.f32`, OpenCL 1.2 et WGSL (Metal,
  Vulkan, DX12) les simulent par une boucle, plus lente encore quand les threads se bousculent.
- **En `gs.f32`, le dernier chiffre peut varier d'un lancement à l'autre** : l'ordre d'arrivée des threads change,
  et (a + b) + c n'est pas exactement a + (b + c) en virgule flottante. Pour un résultat identique à chaque fois,
  cumulez en entiers (par exemple des dixièmes de millimètre au lieu de millimètres en flottant).
- Pas encore permis : les sorties en 64 bits (`gs.i64`, `gs.f64`), et `grid=` avec `group=`.
