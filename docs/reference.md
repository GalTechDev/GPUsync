# Référence

Chaque nom public de `gpusync` (importé en `gs`). Le moteur graphique `gs.gfx` n'est pas décrit ici.

## Décorateur et kernel

### `@gs.kernel` / `@gs.kernel(cast=None, compat="auto", grid=None, group=None, cells=None)`

Transforme une fonction annotée en `gs.Kernel`. Voir [Écrire un kernel](kernels.md).

| Paramètre | Valeurs | Effet |
|---|---|---|
| `cast` | `None`, `"auto"` | `"auto"` : convertit les tableaux d'un autre type numérique (une copie) au lieu de lever `gs.TypeMismatch` |
| `compat` | `"auto"`, `"strict"` | opération absente du périphérique : repli avec `gs.CompatibilityWarning`, ou `gs.UnsupportedFeature` |
| `grid` | nom d'une entrée, ou tuple de dimensions | mode explicite : un thread par case de la grille, sorties `gs.Out` ([Mode explicite](atomiques.md)) |
| `group` | entier de 1 à 1024 | groupes de threads : `gs.shared`, `gs.lane`, `gs.barrier` ([Groupes](groupes.md)) |
| `cells` | entier ≥ 1, avec `group` | plusieurs cases de sortie par thread, écrites par `gs.store` |

### `gs.Kernel`

| | |
|---|---|
| `k(*args, device=None, out=None)` | lance le calcul ; rend un `gs.Array`, ou un tuple de `gs.Array` (plusieurs sorties ou plusieurs `gs.Out`). `out=` : destination(s) NumPy, ou point de départ des `gs.Out` |
| `k.stream(source, *rest, device=None)` | itérateur de résultats NumPy, un par bloc de `source` ; le bloc suivant est lancé avant de rendre le courant |
| `k.source(target="opencl")` | code généré : `"cuda"`, `"opencl"`, `"wgsl"`, `"ir"` |
| `k.py_func` | la fonction Python d'origine |
| `repr(k)` | la signature, ex. `<gs.kernel f(x: gs.f32[:]) -> gs.f32[:]>` |

## Types

| | |
|---|---|
| `gs.f32`, `gs.f64`, `gs.i32`, `gs.u32`, `gs.i64`, `gs.u64` | types scalaires : annotation d'un paramètre, conversion dans un kernel (`gs.i32(v)`), conversion hors kernel (`gs.f32(3)` → `numpy.float32(3.0)`) ; `.np` donne le type NumPy |
| `gs.f32[:]` | tableau de toute forme (élément par élément) |
| `gs.f32["n", "t"]`, `gs.f32[:, :]`, `gs.f32["n", 3]` | tableau à dimensions nommées, sans nom, ou de taille fixe |
| `gs.Out[gs.i32[10]]`, `gs.Out[gs.f32["m"], gs.inf]` | sortie du mode explicite, avec sa valeur de départ (0 par défaut) |

## Dans un kernel

Hors d'un kernel, les fonctions mathématiques sont celles de NumPy ; `gs.index`, `gs.lane`, `gs.barrier` et
`gs.store` lèvent une erreur.

| Nom | Où | Effet |
|---|---|---|
| `gs.index()` | avec index, mode explicite | coordonnées de la case (un entier, ou un tuple à déballer : `i, j = gs.index()`) |
| `gs.local(type, n)` | partout | tableau propre au thread, mis à zéro, taille constante |
| `gs.shared(type, n)` | `group=` | tableau commun au groupe, mis à zéro, taille constante |
| `gs.lane()` | `group=` | rang du thread dans le groupe |
| `gs.barrier()` | `group=` | attend tout le groupe |
| `gs.store(k, v)` | `cells=` | écrit la k-ième case du thread |
| `gs.matmul_u8(c, a, b, m, n, k)` | `group=` | produit de matrices d'octets, collectif |
| `gs.atomic_add(out, i, v)`, `gs.atomic_min`, `gs.atomic_max` | `grid=` | écriture atomique dans un `gs.Out` ; `i` entier ou tuple |
| `gs.sqrt`, `exp`, `exp2`, `log`, `log2`, `pow`, `sin`, `cos`, `tan`, `tanh`, `atan2`, `floor`, `ceil`, `abs` | partout | fonctions natives du GPU (aussi par `math.*`) |
| `gs.min(a, b, …)`, `gs.max(a, b, …)` | partout | deux valeurs ou plus |
| `gs.mulhi(a, b)` | partout | partie haute du produit complet de deux entiers de même type |
| `gs.pi`, `gs.e`, `gs.tau`, `gs.inf`, `gs.nan` | partout | constantes |
| `x.shape[k]`, `len(x)` | avec index | tailles d'un tableau |
| `float()`, `int()`, `gs.f32()`… | partout | conversions (`int` tronque vers zéro) |

## `gs.Array`

Résultat d'un kernel. Voir [Résultats et gros volumes](resultats.md).

| | |
|---|---|
| `shape`, `dtype`, `ndim`, `size`, `nbytes`, `device` | immédiats |
| `mode` | `"direct"`, `"resident"`, `"flux"` ou `"cpu"` (`"host"` pour un résultat vide) |
| `zero_copy` | `True` en mode `direct` |
| `ready` | le résultat a déjà été lu |
| `numpy()`, `np.asarray(y)`, `y[...]`, `len(y)`, `iter(y)`, `print(y)` | attendent et rapatrient |
| `sum`, `mean`, `min`, `max`, `std` (`axis=`, options NumPy) | réductions par NumPy après lecture |
| `save(path)` | fichier `.npy`, écrit par blocs |
| `await y` | attente sans bloquer la boucle asyncio ; rend `y` |
| `__dlpack__`, `__dlpack_device__` | export DLPack (mémoire hôte) |

Opérateurs (`+`, `*`, `==`, `<`, `-y`…) et ufuncs NumPy : calculés sur l'hôte après lecture, ils rendent un
tableau NumPy. Un `gs.Array` reste hachable (par identité).

Un kernel à plusieurs sorties rend un tuple de `gs.Array` qui s'attend en une fois : `a, b = await k(x)`.

## Périphériques

| | |
|---|---|
| `gs.devices(refresh=False)` | liste des `gs.Device` |
| `gs.backends(refresh=False)` | liste des `gs.BackendStatus` (`name`, `available`, `error`, `devices`) |
| `gs.device(dev)` | contexte `with` : périphérique du bloc (par thread) ; rend le `gs.Device` |
| `gs.set_default_device(dev)` | périphérique par défaut (`None` : le classement) |
| `gs.default_device(kernel=None)` | périphérique qui serait utilisé maintenant |
| `gs.set_memory_budget(nbytes=None)` | budget du mode `resident` (`None` : 50 % de la mémoire du périphérique) |
| `gs.batch(device=None)` | contexte `with` : une seule soumission pour les kernels du bloc (wgpu) |
| `gs.sync(device=None)` | attend la fin des calculs soumis, sans rien rapatrier |
| `gs.Device` | `id`, `name`, `backend`, `kind`, `index`, `vendor`, `version`, `memory_bytes`, `max_buffer_bytes`, `unified_memory`, `f64`, `f64_math`, `int64`, `max_group`, `shared_bytes` |

`dev` : un `gs.Device`, un identifiant (`"cuda:0"`), un nom de backend (`"opencl"`) ou `"gpu"`. Voir
[Périphériques](peripheriques.md).

## Fenêtre

`gs.Window(width, height, title="GPUsync", *, vsync=False, resizable=True, device=None)`

| | |
|---|---|
| `poll()` | liste des événements (dictionnaires, clé `type`) |
| `present(image)` | affiche un `gs.Array` `uint32 (H, W)` ou un tableau NumPy `uint32 (H, W)`, `uint8 (H, W, 4)`, `uint8 (H, W, 3)` |
| `open`, `device`, `size`, `width`, `height`, `physical_size`, `zero_copy` | état |
| `vsync` | lisible et modifiable |
| `mouse_pos`, `keys`, `mouse_buttons`, `key_down(key)` | entrées courantes |
| `set_title(title)`, `close()` | |

Utilisable en `with`. Voir [Fenêtre](fenetre.md).

## Exceptions et avertissements

`gs.GpusyncError` (base), `gs.KernelSyntaxError`, `gs.TypeMismatch`, `gs.ShapeMismatch`, `gs.UnsupportedFeature`,
`gs.DeviceError`, `gs.OutOfMemory`, `gs.OutOfBounds` ; avertissement `gs.CompatibilityWarning`. Voir
[Erreurs](erreurs.md).

## Divers

| | |
|---|---|
| `gs.__version__` | version de la bibliothèque |
| `python -m gpusync` | diagnostic des backends et des périphériques |
