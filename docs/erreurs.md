# Erreurs, débogage, performances

## Exceptions

Toutes dérivent de `gs.GpusyncError`. Les messages disent ce qui ne va pas, où, et comment corriger.

| Exception | Quand |
|---|---|
| `gs.KernelSyntaxError` | à la décoration : construction refusée dans le kernel, type de retour manquant, source introuvable |
| `gs.TypeMismatch` | à l'appel : type d'un argument différent de la signature |
| `gs.ShapeMismatch` | à l'appel : formes incohérentes, dimension de sortie inconnue |
| `gs.UnsupportedFeature` | le périphérique ne sait pas faire une opération du kernel (avec `compat="strict"` ou `device=` explicite), ou dépasse ses limites de groupe |
| `gs.DeviceError` | périphérique introuvable, erreur du pilote, GPU perdu |
| `gs.OutOfMemory` | la sortie ne peut pas être allouée en mémoire centrale : donnez `out=` |
| `gs.OutOfBounds` | mode explicite : écriture atomique hors limites (aussi un `IndexError`) |
| `gs.CompatibilityWarning` | avertissement : le kernel s'exécute ailleurs que prévu (voir [Périphériques](peripheriques.md)) |

```python
import numpy as np
import gpusync as gs

@gs.kernel
def norm(x: gs.f32[:], m: gs.f32) -> gs.f32[:]:
    return x - m

try:
    norm(np.ones(3), 0.0)                    # float64 au lieu de float32
except gs.TypeMismatch as e:
    print(e)
# argument 'x' : gs.f32[:] attendu, reçu un tableau float64. Utilisez x.astype(np.float32),
# ou @gs.kernel(cast='auto') pour convertir

@gs.kernel
def add(a: gs.f32["n"], b: gs.f32["n"]) -> gs.f32["n"]:
    i = gs.index()
    return a[i] + b[i]

try:
    add(np.ones(3, np.float32), np.ones(4, np.float32))
except gs.ShapeMismatch as e:
    print(e)                                 # add() : dimension 'n' incohérente : 'a' → 3, 'b' → 4
```

Une erreur dans le corps du kernel est signalée à la décoration, avec le fichier et la ligne :

```text
gs.KernelSyntaxError: kernel 'k' (calcul.py), ligne 8 : appel de « foo » non supporté : seules les fonctions
de gs et de math sont utilisables dans un kernel
```

## Déboguer

- **Comparer à l'interpréteur.** `device="cpu"` exécute le kernel par l'interpréteur de référence : même
  sémantique que les GPU, résultat exact, lent. Un écart GPU / `cpu:0` révèle une course entre threads ou un
  problème de précision. Pour les kernels à groupes, l'interpréteur signale aussi les accès concurrents à la
  mémoire partagée.
- **Comparer à NumPy.** Un kernel élément par élément écrit avec les fonctions `gs.*` s'exécute aussi en NumPy
  (`kernel.py_func`).
- **Lire le code généré.** `kernel.source("cuda" | "opencl" | "wgsl" | "ir")` ; `GPUSYNC_DUMP_DIR=dossier` écrit
  chaque code au moment de sa compilation, y compris les variantes spécialisées.
- **Diagnostic de la machine.** `python -m gpusync`, `gs.backends()`.

```python
x = np.random.rand(1000).astype(np.float32)
assert np.allclose(norm(x, 0.5), norm(x, 0.5, device="cpu"))
print(norm.source("ir"))
```

## Pièges

| Piège | Ce qui se passe | À faire |
|---|---|---|
| `y + 1`, `y == 2` sur un `gs.Array` dans une boucle de calcul | chaque opération attend le GPU et calcule sur l'hôte | un kernel, pour rester sur le GPU |
| `if y:` sur un `gs.Array` | vrai dès qu'il a des éléments, quelles que soient les valeurs (NumPy lèverait une erreur) | `if (y > 0).all():` |
| modifier une entrée NumPy avant d'avoir lu le résultat | sur mémoire unifiée, le GPU lit la mémoire modifiée | lire ou attendre le résultat d'abord |
| kernel défini dans la console ou `python -c` | refusé : source introuvable | le mettre dans un fichier ou une cellule de notebook |
| changer une constante globale après la décoration | le kernel garde l'ancienne valeur | la passer en paramètre |
| un même nom pour un entier puis un flottant dans un kernel | refus ou valeur tronquée | deux noms |
| longue boucle de kernels sans jamais lire ni `gs.sync()` | le travail en attente s'accumule dans la mémoire du GPU | `gs.sync()` toutes les quelques itérations |
| `math.sqrt` dans un kernel appelé par `py_func` | `TypeError` (math n'accepte pas de tableaux) | `gs.sqrt` |
| f64 sur un Mac | repli sur `cpu:0`, exact mais lent | f32, ou OpenCL / CUDA sur une machine qui a le f64 |

## Performances

- **Le premier appel compile** le kernel pour le périphérique (de quelques millisecondes à quelques centaines
  pour CUDA, mis en cache sur disque). Mesurez à partir du deuxième appel.
- **Chaque appel a un coût fixe** (de l'ordre de 0,1 ms sur Vulkan, davantage en OpenCL). Un kernel qui fait
  peu de travail par case doit en traiter beaucoup ; pour une suite de petits kernels sur Metal, Vulkan ou DX12,
  `gs.batch()` les soumet en une fois.
- **Laissez les résultats sur le GPU** : enchaînez les kernels sur des `gs.Array` et ne lisez qu'à la fin.
- **f32 plutôt que f64** : les GPU grand public calculent le f64 de 16 à 64 fois plus lentement, quand ils le
  font.
- **Entiers 64 bits** : division et reste sont lents partout (émulés), très lents sur Metal ; `gs.mulhi` permet
  souvent de rester en 32 bits (multiplication modulaire de Montgomery).
- **Spécialisation des tailles.** Dans un kernel avec index, une dimension qui n'est pas un axe de la sortie (la
  longueur `t` d'une série, par exemple) et vaut au plus 1 024 devient une constante du code compilé : boucles
  déroulées, tableaux locaux en registres. Gain mesuré sur un DTW : ×1,5 à ×40 selon le GPU. Au plus 8 variantes
  par kernel et par périphérique ; changer la taille des axes de la sortie ne recompile rien.
  `GPUSYNC_SPECIALIZE=0` la désactive.
- **Groupes et mémoire partagée** (voir [Groupes](groupes.md)) pour tout ce qui réutilise des données voisines.

## Limites connues

- Pas de tranches (`x[a:b]`) ni d'appels de vos propres fonctions dans un kernel.
- Pas de diffusion (*broadcasting*) entre tableaux de formes différentes dans un kernel élément par élément.
- Tableaux non contigus (vues NumPy transposées, à pas) : copiés en contigu à l'appel.
- `gs.Out` en 64 bits non permis ; `grid=` ne se combine pas avec `group=`.
- Au-delà de 2³¹ − 1 éléments dans un seul tableau gardé entier sur le périphérique : géré en OpenCL, CUDA et sur
  `cpu:0` (variante à indices 64 bits compilée à la demande), refusé par wgpu (Metal, Vulkan, DX12).
- Un GPU perdu par le pilote (`gs.DeviceError`) est rouvert à l'appel suivant, mais le calcul en cours est perdu.
