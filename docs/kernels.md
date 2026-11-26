# Écrire un kernel

Un kernel est une fonction Python décorée par `@gs.kernel`. On n'y écrit pas de boucle sur les données : on
écrit le calcul **d'une case du résultat**, et le GPU l'exécute pour toutes les cases à la fois, chacune sur
son propre fil d'exécution (thread).

Il y a trois formes de kernel :

| Forme | On écrit | Pour |
|---|---|---|
| élément par élément | une formule sur des tableaux de même forme | transformations point à point |
| avec index (`gs.index()`) | le calcul de la case `(i, j…)` de la sortie, en lisant où l'on veut dans les entrées | distances, convolutions, réductions par ligne, tout le reste |
| explicite (`grid=`) | ce que fait un thread par **entrée**, avec des écritures atomiques | histogrammes, dispersion : voir [Mode explicite](atomiques.md) |

Un kernel doit être défini dans un fichier `.py` ou une cellule de notebook : GPUsync relit son code source à la
décoration. Défini dans `python -c` ou dans la console interactive, il est refusé avec un message qui le dit.

## Élément par élément

```python
import numpy as np
import gpusync as gs

@gs.kernel
def blend(a: gs.f32[:], b: gs.f32[:], t: gs.f32) -> gs.f32[:]:
    return a * (1.0 - t) + b * t

a = np.zeros((480, 640), np.float32)
b = np.ones((480, 640), np.float32)
c = blend(a, b, 0.25)
assert c.shape == (480, 640) and np.allclose(c, 0.25)
```

- `gs.f32[:]` accepte un tableau de **n'importe quelle forme** ; tous les tableaux d'un même appel doivent avoir
  la même forme, qui est celle de la sortie. Il n'y a pas de diffusion (*broadcasting*) à la NumPy : une valeur
  commune à toutes les cases se passe en scalaire (`t: gs.f32`).
- Le type de retour est obligatoire (`-> gs.f32[:]`).
- Dans le corps, un tableau désigne **sa valeur dans la case courante** : `a * (1.0 - t)` multiplie deux nombres.

On peut y mettre des variables, des conditions et des boucles :

```python
@gs.kernel
def clamp_or_log(x: gs.f32[:], limit: gs.f32) -> gs.f32[:]:
    if x > limit:
        return limit
    y = 0.0
    for k in range(3):
        y += gs.log(1.0 + x * (k + 1))
    return y

r = clamp_or_log(np.array([0.5, 2.0, 50.0], np.float32), 10.0)
assert np.allclose(r[2], 10.0)
```

## Avec index

`gs.index()` donne les coordonnées de la case de sortie calculée. Les dimensions des tableaux portent des noms ;
la forme de la sortie est écrite avec ces noms, et GPUsync en déduit sa taille :

```python
@gs.kernel
def pairwise_dist(x: gs.f32["n", "d"]) -> gs.f32["n", "n"]:
    i, j = gs.index()                      # la case (i, j) de la sortie
    acc = 0.0
    for k in range(x.shape[1]):
        diff = x[i, k] - x[j, k]
        acc += diff * diff
    return gs.sqrt(acc)

pts = np.random.rand(500, 3).astype(np.float32)
dist = pairwise_dist(pts)                  # (500, 3) → (500, 500)
ref = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
assert np.allclose(dist, ref, atol=1e-5)
```

Règles des dimensions :

| Écriture | Signification |
|---|---|
| `gs.f32["n", "t"]` | deux dimensions nommées ; un même nom impose la même taille partout où il apparaît |
| `gs.f32[:, :]` | deux dimensions sans nom, de taille libre, lisible par `x.shape[k]` |
| `gs.f32["n", 3]` | taille fixe : l'appel est refusé si l'axe n'a pas 3 éléments |
| `-> gs.f32["n"]` | la sortie : uniquement des noms d'entrée ou des tailles entières |

Une incohérence (`'a' → 3, 'b' → 4` pour une même dimension `n`) lève `gs.ShapeMismatch` avant tout calcul.

Dans le corps, on indexe avec des entiers (`x[i, k]`), on lit `x.shape[k]` et `len(x)` (la première dimension).
Pas de tranches (`x[i, :]`) : on écrit la boucle.

```python
@gs.kernel
def row_mean(x: gs.f32["n", "t"]) -> gs.f32["n"]:
    i = gs.index()
    s = 0.0
    for k in range(x.shape[1]):
        s += x[i, k]
    return s / x.shape[1]

m = np.arange(12, dtype=np.float32).reshape(3, 4)
assert np.allclose(row_mean(m), m.mean(axis=1))
```

### Plusieurs sorties

Un tuple de types en retour, un tuple de valeurs dans le `return` ; l'appel rend un tuple de résultats :

```python
@gs.kernel
def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    t = x.shape[1]
    s = 0.0
    for k in range(t):
        s += x[i, k]
    mean = s / t
    v = 0.0
    for k in range(t):
        d = x[i, k] - mean
        v += d * d
    return mean, gs.sqrt(v / t)

series = np.random.rand(100, 365).astype(np.float32)
mean, std = stats(series)
assert np.allclose(std, series.std(axis=1), atol=1e-5)
```

Toutes les sorties ont la même forme.

### Tableaux locaux

`gs.local(type, taille)` crée un tableau propre à chaque case, mis à zéro, de taille constante :

```python
@gs.kernel
def histogram_per_row(x: gs.f32["n", "t"]) -> gs.i32["n"]:
    i = gs.index()
    bins = gs.local(gs.i32, 10)
    for k in range(x.shape[1]):
        b = gs.min(int(x[i, k] * 10.0), 9)
        bins[b] += 1
    best = 0
    for b in range(10):
        if bins[b] > bins[best]:
            best = b
    return best                            # la classe la plus peuplée de la ligne

data = np.random.rand(4, 1000).astype(np.float32)
assert histogram_per_row(data).shape == (4,)
```

Un tableau local vit dans la mémoire privée du thread : quelques dizaines à quelques centaines d'éléments au
plus, au-delà il devient lent.

## Types

| Type | NumPy | Remarques |
|---|---|---|
| `gs.f32` | `float32` | le type de calcul normal sur GPU |
| `gs.f64` | `float64` | absent de Metal et de nombreux GPU grand public : voir [Périphériques](peripheriques.md) |
| `gs.i32`, `gs.u32` | `int32`, `uint32` | |
| `gs.i64`, `gs.u64` | `int64`, `uint64` | division et reste 64 bits lents sur Metal (préférer OpenCL sur Mac pour ces calculs) |

Les tableaux passés doivent avoir **exactement** le type déclaré. Un tableau `float64` passé à un paramètre
`gs.f32[:]` est refusé (`gs.TypeMismatch`, avec la correction à faire) ; `@gs.kernel(cast="auto")` autorise la
conversion, qui coûte une copie.

```python
@gs.kernel(cast="auto")
def twice(x: gs.f32[:]) -> gs.f32[:]:
    return x * 2.0

assert twice(np.arange(3)).dtype == np.float32      # entiers 64 bits convertis en float32
```

Dans un kernel :

- **les littéraux prennent le type du contexte** : `0.0` dans un calcul en f32 est un f32 (il ne force jamais du
  f64) ;
- les mélanges suivent `entier op flottant → flottant`, jamais vers f64 sans le demander ; entre entiers,
  l'élargissement sans perte est implicite (i32 → i64, u32 → i64, u32 → u64), le reste se convertit à la main ;
- conversions : `float(v)`, `int(v)` (troncature vers zéro, comme Python), `gs.f32(v)`, `gs.i32(v)`, `gs.u32(v)`,
  `gs.i64(v)`… ;
- `//` et `%` se comportent comme en Python, y compris sur les négatifs (`-7 // 3 == -3`, `-7 % 3 == 2`) ;
- les variables sont typées à leur première affectation ; donner à un même nom des valeurs de types différents
  (un compteur entier puis un flottant) est refusé ou source d'erreur : prenez deux noms.

```python
@gs.kernel
def divmod_like_python(x: gs.i32[:]) -> gs.i32[:]:
    return x // 3 * 100 + x % 3

assert list(divmod_like_python(np.array([-7, 7], np.int32))) == [-298, 201]
```

## Ce qu'on peut écrire dans un kernel

| Permis | Refusé |
|---|---|
| arithmétique, comparaisons, `and` / `or` / `not` | listes, tuples (hors `gs.index()` et `return`), dictionnaires, chaînes |
| bit à bit `&` `\|` `^` `~` `<<` `>>` sur les entiers ; `&` `\|` `^` sur les booléens | appels de fonctions Python (y compris les vôtres) |
| `a if c else b`, `if` / `elif` / `else` | récursion, `lambda`, générateurs |
| `for k in range(...)`, `while`, `break`, `continue` | `try`, `with` |
| variables locales scalaires, `gs.local` | tranches `x[a:b]`, allocation de taille variable |
| constantes globales numériques du module | objets, attributs autres que `x.shape` |
| fonctions `gs.*` et `math.*` listées ci-dessous | |

Une construction refusée lève `gs.KernelSyntaxError` à la décoration, avec le fichier, la ligne et la cause.

Fonctions mathématiques (`gs.nom` ou `math.nom`) :

| | |
|---|---|
| racines, exponentielles | `sqrt`, `exp`, `exp2`, `log`, `log2`, `pow` |
| trigonométrie | `sin`, `cos`, `tan`, `tanh`, `atan2` |
| arrondis | `floor`, `ceil`, `abs` |
| extremums | `gs.min(a, b, …)`, `gs.max(a, b, …)` (2 valeurs ou plus) |
| entiers | `gs.mulhi(a, b)` : partie haute du produit complet (ex. `(a * b) >> 32` sans débordement, pour des u32) |
| constantes | `gs.pi`, `gs.e`, `gs.tau`, `gs.inf`, `gs.nan` (et `math.pi`…) |

```python
import math

GAIN = 3.0                                  # constante globale : lue à la décoration

@gs.kernel
def gained(x: gs.f32[:]) -> gs.f32[:]:
    return math.sqrt(x) * GAIN + gs.max(x, 0.0, -1.0)

assert np.allclose(gained(np.array([4.0], np.float32)), 10.0)
```

Une constante globale est figée quand le kernel est décoré : la modifier ensuite ne change pas le kernel.

## La fonction Python reste utilisable

`gs.sqrt`, `gs.exp`… sont, hors kernel, les fonctions NumPy correspondantes, et `kernel.py_func` est la fonction
d'origine : un kernel élément par élément écrit avec les fonctions `gs.*` s'exécute aussi en NumPy pur,
pratique pour comparer.

```python
assert np.allclose(blend.py_func(a, b, 0.25), blend(a, b, 0.25))
```

Ce n'est pas vrai avec `math.*` (qui n'accepte pas de tableaux), ni des kernels avec index : `gs.index()` n'a
pas de sens hors du GPU.

## Compilation

Un kernel est traduit une fois à la décoration (vérification des types, représentation interne), puis compilé
pour un périphérique au premier appel qui l'y exécute. Les appels suivants réutilisent le code compilé.
`kernel.source(cible)` montre le code généré (`"cuda"`, `"opencl"`, `"wgsl"` ou `"ir"`) :

```python
print(blend.source("opencl"))
```
