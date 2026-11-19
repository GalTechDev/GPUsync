# GPUsync v2 — Architecture

> Statut : **M1 terminé** (19/11/2026). Ce document fixe les décisions avant d'écrire le cœur en Rust.
> Il s'appuie sur le prototype Python (`GPUsync-python/`), testé sur Apple M5 Pro (OpenCL 1.2), le 19/11/2026.

### Avancement

| Jalon | État | Validé sur |
|---|---|---|
| M0 Squelette | ✅ workspace Cargo, PyO3 0.30 (abi3-py310) + maturin 1.16, `gs.devices()` / `gs.backends()` via wgpu 30 et opencl3 0.12 (chargement dynamique) | Mac M5 Pro (Python 3.15) ; PC Windows RTX 4080 SUPER + UHD 770 (Python 3.10) |
| M1 MVP | ✅ `@gs.kernel` élément par élément, IR + interpréteur, émetteurs OpenCL C et WGSL, `gs.Array`, zéro-copie OpenCL | Mac : Metal, OpenCL, cpu (75 tests) ; PC : Vulkan et OpenCL sur RTX 4080 SUPER et UHD 770, cpu (104 tests) |

#### Mesures M1 (`tools/bench_m1.py`)

Surcoût par appel, de bout en bout (appel + `np.asarray`), 1 000 éléments — objectif ≤ 0,3 ms :

| | Metal M5 Pro | OpenCL M5 Pro | Vulkan RTX | Vulkan UHD 770 | OpenCL RTX | OpenCL UHD 770 |
|---|---|---|---|---|---|---|
| ms / appel | 0,196 | 0,193 | 0,095 | 0,183 | 0,272 | 0,149 |

Prototype v1 sur le M5 Pro (OpenCL) : 3,07 ms ; pyopencl brut : 0,30 ms.

Pic mémoire supplémentaire pour 1 Gio d'entrée (la sortie seule compte pour 1,00) — objectif ×1 en OpenCL sur le Mac :

| | Metal M5 Pro | OpenCL M5 Pro | Vulkan RTX | OpenCL RTX | OpenCL UHD 770 |
|---|---|---|---|---|---|
| Gio | 3,01 | **1,00** (zéro-copie) | 5,23 | 3,35 | **1,00** (zéro-copie) |

Mesure : mémoire résidente (macOS), mémoire engagée (Windows). Sur 10 M éléments, OpenCL en zéro-copie
traite 83 Go/s sur le M5 Pro contre 9,5 Go/s pour Metal (copies).

#### Mesures f64 via wgpu (sonde sur le PC, wgpu 30)

| GPU / API | `SHADER_F64` | Résultat |
|---|---|---|
| RTX 4080 SUPER, Vulkan | ✅ | arithmétique, `//`, `%`, comparaisons, `sqrt`, `floor`, `ceil`, `abs`, `min`, `max`, `fma`, conversions : **précision f64** (≤ 2·10⁻¹⁶) |
| RTX 4080 SUPER, Vulkan | ✅ | `exp` : accepté mais **calculé en précision f32** (1,7·10⁻⁷), sans erreur ; `log` : **le pilote plante** (violation d'accès) |
| RTX 4080 SUPER, Vulkan | ✅ | paramètre f64 par `bitcast<f64>(vec2<u32>)` : refusé par naga ; par un buffer `array<f64>` : correct |
| RTX 4080 SUPER et UHD 770, DX12 | ❌ | — |
| UHD 770, Vulkan ; M5 Pro, Metal | ❌ | pas d'unités f64 dans le matériel |

Les fonctions transcendantes en f64 (GLSL.std.450 ne les définit qu'en 16 et 32 bits) sont donc **refusées
avant compilation** sur Vulkan : sinon, résultat faux silencieux ou plantage.

#### Décisions prises pendant M1

- **Sémantique** : `//` et `%` suivent Python et NumPy (arrondi vers −∞, signe du diviseur), y compris sur
  le GPU ; division entière par zéro → 0 (comme NumPy) ; `/` entre entiers donne un f32 ; `**` réservé aux
  flottants ; `pow` corrigé en WGSL pour une base négative.
- **Typage** : un littéral prend le type de son contexte (`x * 2` reste f32) ; une locale prend le type de ses
  affectations (`acc = 0` puis `acc += x` → f32) ; aucune promotion implicite vers f64.
- **f64 via wgpu** : à M1, `Device.f64` vaut `False` pour Metal, Vulkan et DX12. C'était trop restrictif :
  wgpu active le f64 sur Vulkan (`SHADER_F64`), à condition d'éviter les fonctions mathématiques. Corrigé
  à M1.5 (§6.2, mesures ci-dessous).
- **`cpu:0`** : l'interpréteur de référence est exposé comme périphérique, toujours disponible, en dernier.
- **Taille des buffers wgpu** : limitée à ~1 Gio sur l'UHD 770 en Vulkan. Un tableau trop grand lève
  `gs.UnsupportedFeature` (avant correction, wgpu renvoyait un résultat faux sans erreur). Le découpage
  automatique viendra avec l'out-of-core (M3).
- **Entrée D10** : wgpu copie et consomme 3 à 5 fois la taille des données ; OpenCL en zéro-copie, aucune.
  Sur Mac, Metal reste le défaut (même surcoût par appel), mais les gros volumes ont intérêt à passer par
  OpenCL : à trancher avec l'out-of-core (M3).

Écarts au plan, à M0 : seuls les crates nécessaires existent (`gpusync-runtime`, `gpusync-backend-wgpu`,
`gpusync-backend-opencl`, `gpusync-py`) ; `gpusync-ir`, `gpusync-frontend` et `gpusync-codegen` seront créés
en M1. Sous Windows, un GPU visible en Vulkan et en DX12 n'est listé qu'une fois (Vulkan). D2 tranchée :
Python ≥ 3.10, testé aux deux bouts (3.10 sur le PC, 3.15 sur le Mac).

---

## 1. Objectifs

**GPUsync permet à un dev Python d'écrire un kernel GPU en Python, de l'appeler comme une fonction, et d'obtenir un résultat correct sur n'importe quel GPU, sans rien connaître au GPU.**

Objectifs mesurables :

| # | Objectif | Critère |
|---|---|---|
| O1 | Installation triviale | `pip install gpusync` réussit sur macOS, Linux et Windows sans driver GPU particulier ; l'import ne plante jamais |
| O2 | Correction | Résultats identiques à NumPy (à la tolérance flottante près) sur tous les backends ; **jamais** de résultat faux silencieux |
| O3 | Overhead faible | ≤ 0,3 ms par appel sur un petit kernel (M5 Pro). Prototype : 3,1 ms ; pyopencl brut : 0,3 ms |
| O4 | Portabilité | Metal, Vulkan, DX12 (via wgpu) et OpenCL ≥ 1.2 dès le MVP ; CUDA ensuite |
| O5 | Out-of-core | Données plus grandes que la mémoire GPU traitées avec le même appel |
| O6 | Lisibilité | Un message d'erreur dit ce qui ne va pas, où, et comment le corriger |
| O7 | Pas de duplication sur mémoire unifiée | Sur Apple Silicon, GPU intégrés et Jetson, un tableau NumPy n'existe qu'une fois en RAM, en entrée comme en sortie, quand le backend le permet (voir §8.4) |

### Non-objectifs (pour l'instant)

- Remplacer PyTorch / JAX : pas d'autodiff, pas de bibliothèque de tenseurs complète.
- Accepter du Python arbitraire dans un kernel : on supporte un **sous-ensemble** clairement documenté.
- Fusion automatique de kernels et ordonnancement multi-GPU avancé (prévus en fin de feuille de route, pas dans le MVP).

---

## 2. Leçons du prototype

| Constat dans GPUsync v1 | Décision pour la v2 |
|---|---|
| 4 transpileurs écrits à la main, chacun depuis l'AST Python (CUDA, OpenCL, LLVM, PTX). LLVM échoue déjà sur `min`, `get_local_id` | **Une seule IR typée**, plusieurs émetteurs simples qui partent de l'IR, pas de l'AST |
| 5 façons de choisir le matériel (`Scheduler`, `SwarmScheduler`, `MultiDeviceScheduler`, `use_gpu`, scheduler par défaut) | **Un seul concept : `Device`**, choisi automatiquement par défaut |
| Boucle asyncio en arrière-plan + `auto_sync` + callback OpenCL par étape : ~1,1 ms par attente, ~3 ms par appel | **Le runtime Rust gère files et attentes.** Pas de boucle asyncio interne. On n'attend qu'à la lecture des données |
| `auto_sync` renvoie une coroutine dès qu'une boucle tourne → `y = f(x)` casse dans Jupyter ; un `await` oublié passe inaperçu | **`gs.Array`** : un seul objet à la fois attendable et lisible directement |
| Files en round-robin : upload, kernel et relecture sur des files différentes → synchro obligatoire à chaque étape | **Une file ordonnée par périphérique** et dépendances suivies par le runtime |
| Sorties toujours en float32 ; un tableau float64 donne des résultats faux sans erreur | **Types obligatoires dans la signature**, vérifiés à l'appel, capacités du périphérique vérifiées à la compilation |
| `cupy-cuda12x` en dépendance obligatoire → install impossible sur Mac | **Aucune dépendance GPU côté Python.** Les bibliothèques natives sont chargées dynamiquement au démarrage |
| `AsyncBuffer.__del__` remet le buffer au pool immédiatement, même si une opération l'utilise encore ; pool jamais borné | Un buffer ne retourne au pool **qu'après la fin des opérations en attente** ; pool avec budget mémoire |
| Callback OpenCL Apple : ~1,1 ms ; attente directe (`wait`) : ~0,25 ms | Attente **directe** sur l'événement (GIL libéré), pas de callbacks |
| Copie systématique hôte ↔ GPU, même quand la mémoire est unifiée (Apple Silicon) : données en double et copies inutiles | **Zéro-copie sur mémoire unifiée.** Mesuré sur M5 Pro, 256 Mo : 7,4 ms avec `USE_HOST_PTR` contre 33,3 ms avec copie |
| ✅ Écrire ses kernels en Python | **Conservé**, c'est la raison d'être de la lib |
| ✅ Out-of-core en une ligne, y compris depuis un générateur | **Conservé et généralisé** (voir §8) |
| ✅ Garder sur le GPU les arguments constants en out-of-core | **Généralisé** : toute entrée qui ne dépend pas de la dimension découpée reste résidente |
| ✅ Pool de buffers : plus rapide que pyopencl brut sur 10 M éléments | **Conservé**, dans le runtime Rust |

---

## 3. Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────────┐
│ Python : package `gpusync` (mince)                                    │
│   @gs.kernel · types gs.f32[...] · gs.Array · gs.devices() · erreurs │
└───────────────┬──────────────────────────────────────────────────────┘
                │ PyO3 (abi3)
┌───────────────▼──────────────────────────────────────────────────────┐
│ gpusync-py : liaison Python                                           │
│   lit l'`ast` Python → KernelAst (structure Rust neutre)              │
│   convertit NumPy / DLPack ↔ buffers · libère le GIL · __await__       │
└───────┬──────────────────────────────────────────┬───────────────────┘
        │                                          │
┌───────▼───────────────┐                ┌─────────▼──────────────────┐
│ gpusync-frontend      │                │ gpusync-runtime             │
│  KernelAst → IR        │                │  Device · Stream · Buffer   │
│  inférence de types    │                │  pool mémoire · cache       │
│  dimensions nommées    │                │  graphe différé · events    │
│  validation            │                │  out-of-core                │
└───────┬───────────────┘                └─────────┬──────────────────┘
        │                                          │ trait Backend
┌───────▼───────────────┐      ┌───────────────────┼───────────────────┐
│ gpusync-ir            │      │                   │                   │
│  IR typée structurée   │  ┌───▼──────────┐ ┌──────▼───────┐ ┌─────────▼────┐
│  capacités requises    │  │ backend-wgpu │ │backend-opencl│ │ backend-cuda │
│  interpréteur CPU      │  │ Metal/Vulkan │ │  OpenCL ≥1.2 │ │   (plus tard)│
└───────┬───────────────┘  │ /DX12        │ │              │ │              │
        │                  └──────▲───────┘ └──────▲───────┘ └─────────▲────┘
┌───────▼───────────────┐         │ WGSL           │ OpenCL C          │ CUDA C
│ gpusync-codegen       ├─────────┴────────────────┴───────────────────┘
│  IR → WGSL / OpenCL C │
│  / CUDA C             │
└───────────────────────┘
```

### Workspace Cargo

```
gpusync/
├── Cargo.toml                    # workspace
├── pyproject.toml                # maturin
├── crates/
│   ├── gpusync-ir/               # types, IR, capacités, interpréteur — aucune dépendance GPU
│   ├── gpusync-frontend/         # KernelAst → IR (aucune dépendance à PyO3)
│   ├── gpusync-codegen/          # IR → texte de kernel, un module par cible
│   ├── gpusync-runtime/          # trait Backend, Device, Stream, Buffer, pool, graphe, out-of-core
│   ├── gpusync-backend-wgpu/
│   ├── gpusync-backend-opencl/
│   ├── gpusync-backend-cuda/     # jalon M6
│   └── gpusync-py/               # cdylib PyO3 : seul crate qui dépend de Python
├── python/gpusync/               # couche Python pure : décorateur, annotations de types, __init__
└── tests/                        # tests Python de bout en bout
```

**Règle :** seul `gpusync-py` dépend de PyO3. Le cœur est testable en Rust pur, et réutilisable un jour depuis Rust directement.

---

## 4. API Python

### 4.1 Le modèle mental

> **Un kernel s'écrit du point de vue d'une case de la sortie.**
> « Je suis la case `(i, j)`. Je lis ce dont j'ai besoin dans les entrées, et je renvoie ma valeur. »

- Chaque thread **lit** où il veut dans les entrées.
- Chaque thread **écrit une seule case** : la sienne, via `return`.
- Donc : aucun conflit d'écriture, aucun verrou, résultat déterministe.
- **La grille (nombre de threads) est la forme de la sortie.**

### 4.2 Kernel élément par élément

```python
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std
```

- Sans `gs.index()` dans le corps, le kernel est appliqué à chaque élément : `x` désigne la valeur courante.
- La sortie a la forme de l'entrée. `gs.f32[:]` = une dimension sans nom.
- Les scalaires (`mean`, `std`) sont passés tels quels à tous les threads.

### 4.3 Kernel avec index et dimensions nommées

```python
@gs.kernel
def pairwise_dist(series: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
    i, j = gs.index()
    acc = 0.0
    for k in range(series.shape[1]):
        d = series[i, k] - series[j, k]
        acc += d * d
    return gs.sqrt(acc)

dist = pairwise_dist(series)      # (2000, 365) → n=2000, t=365 → sortie (2000, 2000)
```

À l'appel :
1. Les dimensions nommées sont liées depuis les formes des entrées (`n=2000`, `t=365`).
2. Une même dimension doit avoir la même taille partout, sinon `gs.ShapeMismatch` avant tout lancement.
3. La sortie est allouée à la forme déclarée ; la grille vaut cette forme.
4. Une dimension de sortie absente des entrées est une erreur de déclaration (à fournir via `dims={"k": 10}`).

### 4.4 Plusieurs sorties

```python
@gs.kernel
def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    s = 0.0; s2 = 0.0
    for k in range(x.shape[1]):
        s += x[i, k]; s2 += x[i, k] * x[i, k]
    m = s / x.shape[1]
    return m, gs.sqrt(s2 / x.shape[1] - m * m)

mean, std = stats(series)
```

Les sorties d'un même kernel doivent avoir la même forme (une grille unique). Sinon : deux kernels, ou le mode explicite (§4.5).

### 4.5 L'exception : grille explicite et écritures atomiques

Quand le calcul part naturellement de l'entrée (histogramme, scatter) :

```python
@gs.kernel(grid="data")                       # un thread par élément de `data`
def histogram(data: gs.f32["n"], counts: gs.Out[gs.i32[10]]):
    i = gs.index()
    gs.atomic_add(counts, gs.int(data[i] * 10), 1)

counts = histogram(data)                      # gs.Out : alloué et mis à zéro par la lib
```

- `grid=` désigne une entrée (ou une forme : `grid=("n", "n")`).
- Les sorties sont déclarées avec `gs.Out[...]` et écrites explicitement.
- Toute écriture dans une sortie qui n'est pas « sa » case doit passer par une opération atomique. Le frontend refuse les écritures directes non atomiques dans ce mode, sauf si `gs.Out(..., unsafe_writes=True)`.

### 4.6 `gs.Array` : synchrone, asynchrone, chaînable

Un appel de kernel **met le travail en file et rend la main immédiatement**. Il renvoie un `gs.Array` : un handle vers des données sur le GPU, avec un événement de fin en attente.

```python
y = normalize(x, 0.0, 1.0)         # rend la main tout de suite
print(y[:5])                        # lecture → attend la fin et rapatrie
np.asarray(y); y.numpy()            # idem
y = await normalize(x, 0.0, 1.0)    # attend sans bloquer la boucle asyncio
z = other(normalize(x, 0.0, 1.0))   # chaînage : rien ne repasse par la RAM
```

| Opération sur `gs.Array` | Effet |
|---|---|
| passer à un autre kernel | aucune attente, dépendance enregistrée |
| `await y` | future asyncio complété par le runtime (`call_soon_threadsafe`) |
| `np.asarray(y)`, `y.numpy()`, `y[...]`, `print(y)` | attente bloquante GIL libéré, puis relecture. Sur mémoire unifiée : **vue** sur la mémoire du buffer, sans copie |
| `y.shape`, `y.dtype`, `y.device` | immédiat (connu sans attendre) |
| `y.sum()`, `y.max(axis=…)`, `y.mean()` | réductions fournies par la lib (kernels internes) |
| `y.save(path)` | écriture bloc par bloc sur disque |
| `torch.from_dlpack(y)` | export DLPack (zéro-copie si même périphérique) |

Le même code fonctionne à l'identique dans un script, dans Jupyter et dans du code async.

**Règle d'utilisation :** un tableau NumPy passé à un kernel ne doit pas être modifié tant que le résultat n'a pas été lu (ou attendu). Sur mémoire unifiée, le GPU lit directement cette mémoire : la modifier pendant le calcul change le résultat. CUDA impose la même règle. `gs.debug()` la vérifie (somme de contrôle de l'entrée avant et après le kernel).

### 4.7 Out-of-core

```python
big = np.memmap("signal_50GB.f32", dtype=np.float32, mode="r")
out = normalize(big, 0.0, 1.0)              # même appel, découpage automatique
out.save("normalise.f32")

for bloc in normalize.stream(generateur(), mean=0.0, std=1.0):
    traiter(bloc)                           # np.ndarray par bloc
```

- Tableau ou `memmap` : même appel. Le résultat est un `gs.Array` « out-of-core » (matérialisé en RAM ou sur disque à la lecture).
- Générateur : `.stream()` renvoie un itérateur de résultats, pour ne jamais concaténer en RAM.
- Détails du découpage : §8.

### 4.8 Périphériques

```python
gs.devices()
# [Device('Apple M5 Pro', backend='metal', f64=False, unified_memory=True),
#  Device('Apple M5 Pro', backend='opencl', f64=False, unified_memory=True, memory='17.8 Go')]
# Sur le PC : Device('NVIDIA GeForce RTX 4080 SUPER', backend='vulkan', f64=True, f64_math=False, …)

with gs.device("opencl"):            # pour un bloc de code
    ...
normalize(x, 0.0, 1.0, device="opencl:0")   # pour un appel
```

#### Choix automatique (M1.5)

Sans `device=` ni `with gs.device(...)`, GPUsync choisit le périphérique **pour chaque kernel**, selon ce qu'il
exige (§6.2) et ce que le matériel sait faire. L'utilisateur n'a jamais à choisir entre Vulkan et OpenCL.

1. **Filtrer** les périphériques capables : `f64`, `f64_math`, et plus tard atomiques, mémoire locale…
2. **Classer** les restants :

| Rang | Catégorie | Backend préféré | Pourquoi |
|---|---|---|---|
| 1 | GPU dédié | CUDA (M6) → Vulkan → OpenCL → DX12 | le plus puissant ; Vulkan a l'overhead le plus bas (0,095 ms contre 0,27 ms en OpenCL sur la RTX) |
| 2 | GPU intégré | Metal (Mac) ; sinon OpenCL → Vulkan | zéro-copie : Metal `NoCopy`, OpenCL `USE_HOST_PTR` |
| 3 | CPU | OpenCL (POCL) → `cpu:0` | dernier recours |

3. Si seul `cpu:0` convient (ex. f64 avec `exp` sur un Mac), le calcul s'y fait, avec un avertissement
   `gs.PerformanceWarning` émis une fois par kernel : le résultat est correct, mais lent.

#### Opération indisponible : option `compat`

```python
@gs.kernel                     # compat="auto" (défaut)
@gs.kernel(compat="strict")
```

Quand le périphérique prévu (choisi par l'utilisateur, ou premier du classement) ne sait pas exécuter une
opération du kernel (ex. `gs.exp` en f64 sur Vulkan) :

| Mode | Comportement |
|---|---|
| `auto` | avertissement `gs.CompatibilityWarning` (une fois par kernel, il nomme l'opération et les deux périphériques), puis **tout le kernel** s'exécute en **OpenCL**, de préférence sur le même GPU physique (ex. `vulkan:0` → `opencl:0` sur la RTX). Si aucun périphérique OpenCL ne convient : `cpu:0`, avec le même avertissement |
| `strict` | `gs.UnsupportedFeature`, sans changer de périphérique ; le message liste les périphériques compatibles |

Règles :
- **Pas de simulation** : aucune opération n'est émulée en logiciel (ex. f64 en deux f32 sur Metal), ni
  découpée entre deux backends au milieu d'un kernel. C'est le kernel entier qui change de périphérique.
- Les données ne bougent qu'une fois : une entrée déjà sur le périphérique prévu (`gs.Array`) est transférée
  vers le périphérique de repli, puis le résultat y reste.
- `y.device` indique toujours où le calcul a réellement eu lieu.

Le choix est mis en cache par kernel et visible : `y.device`, `gs.default_device(kernel)`. Exemples sur le PC :

| Kernel | Choisi |
|---|---|
| f32 | `vulkan:0` (RTX) |
| f64 arithmétique | `vulkan:0` (RTX, `SHADER_F64`) |
| f64 avec `gs.exp` | `opencl:0` (RTX) |

Avec `device=` explicite, le périphérique demandé est utilisé tel quel ; s'il ne convient pas,
`gs.UnsupportedFeature` liste les périphériques compatibles.

Le backend « CPU » est l'interpréteur de l'IR (lent, toujours disponible) ; POCL via OpenCL est l'option CPU rapide.

### 4.9 Débogage et introspection

```python
pairwise_dist.source("opencl")     # code généré pour une cible
pairwise_dist.source("wgsl")
pairwise_dist.ir()                 # IR lisible

with gs.debug():                   # exécution par l'interpréteur de l'IR
    pairwise_dist(series[:10])     # exceptions Python, index hors bornes avec la valeur de (i, j)
```

### 4.10 Erreurs

| Exception | Quand | Exemple de message |
|---|---|---|
| `gs.KernelSyntaxError` | construction non supportée dans le kernel | `ligne 4 : les listes Python ne sont pas supportées dans un kernel ; utilisez gs.local(gs.f32, 16)` |
| `gs.TypeMismatch` | dtype d'un argument ≠ signature | `argument 'x' attendu f32[:], reçu float64[:]. Utilisez x.astype(np.float32) ou @gs.kernel(cast="auto")` |
| `gs.ShapeMismatch` | dimensions nommées incohérentes | `dimension 'n' incohérente : a → 10, b → 12` |
| `gs.UnsupportedFeature` | capacité absente du périphérique | `le kernel 'precise' utilise f64, non supporté par 'Apple M5 Pro' (metal)` |
| `gs.DeviceError` | erreur du driver | message du driver + code généré + périphérique |
| `gs.OutOfMemory` | même un bloc minimal ne tient pas | taille demandée, budget disponible, entrées résidentes en cause |

**Règle :** aucune conversion implicite de dtype par défaut. `cast="auto"` est explicite et documenté.

---

## 5. Frontend : du Python à l'IR

### 5.1 Récupérer l'AST

Au moment de la décoration :
1. `inspect.getsource` + `ast.parse` côté Python (comme le prototype).
2. `gpusync-py` parcourt les nœuds `ast` via PyO3 et construit un **`KernelAst`** : une structure Rust qui ne couvre que le sous-ensemble supporté.

Pourquoi ne pas parser en Rust : l'`ast` de l'interpréteur courant correspond toujours à sa grammaire, et on évite de maintenir un parseur Python.

Résolution des noms à la décoration :
- `gs.*`, `math.*` → fonctions intrinsèques.
- Constantes `int` / `float` / `bool` des globales ou de la closure → inlinées (remplace `constants=` du prototype).
- Fonctions décorées `@gs.func` → fonctions device, inlinées ou émises séparément.
- Tout autre nom → `gs.KernelSyntaxError` avec la ligne.

Annotations : évaluées dans les globales de la fonction, y compris avec `from __future__ import annotations`.

### 5.2 Sous-ensemble supporté (MVP)

| Supporté | Non supporté |
|---|---|
| arithmétique, comparaisons, `and`/`or`/`not`, opérateur ternaire | listes, dicts, sets, chaînes |
| variables locales scalaires | récursion |
| `if` / `elif` / `else`, `for … in range(…)`, `while`, `break`, `continue` | `try`, `with`, générateurs, `lambda` |
| indexation `a[i]`, `a[i, j]`, `a.shape[k]` | appels de fonctions Python ordinaires |
| tuples pour `gs.index()` et les retours multiples | slicing dans le kernel |
| `gs.local(type, taille)` : tableau local de taille constante | allocation dynamique |
| intrinsèques : `gs.sqrt`, `gs.exp`, `gs.min`, `gs.abs`, … ; `math.*` équivalents | |

### 5.3 Typage

- Les **paramètres** sont typés par la signature. Pas de signature → `gs.KernelSyntaxError`.
- Les **locales** sont inférées à la première affectation ; une réaffectation incompatible est une erreur.
- **Les littéraux s'adaptent au contexte** : `0.0` dans un contexte f32 est un f32. Indispensable car en OpenCL C, `3.0` est un double, ce qui force du fp64 ou échoue sur les GPU sans fp64.
- Promotions : `i32 op f32 → f32`, jamais vers f64 implicitement.
- Mode élément par élément : le frontend réécrit `x` en `x[idx]` pour chaque tableau de la forme de la sortie.

### 5.4 Spécialisation

Un `@gs.kernel` est un **modèle**. Une version compilée est identifiée par :

```
(hash de l'IR, dtypes, rangs, cible, capacités du périphérique)
```

Les tailles des dimensions ne font **pas** partie de la clé : elles sont passées en paramètres, pour ne pas recompiler à chaque taille.

Tableaux non contigus (vues NumPy) : copiés en contigu à l'upload dans le MVP ; strides en paramètres plus tard si le besoin est mesuré.

---

## 6. IR

### 6.1 Forme

IR **structurée** : un arbre d'instructions typées (`Let`, `Assign`, `If`, `For`, `While`, `Store`, `Return`, `AtomicRmw`, `Barrier`) et d'expressions typées. Pas de SSA ni de graphe de contrôle : toutes les cibles sont des langages structurés de la famille du C, donc l'IR garde leur forme et les émetteurs restent simples.

Types : `bool`, `i32`, `u32`, `i64`, `f16`, `f32`, `f64`, tableaux `T[dims]` (globaux, en lecture ou en écriture), tableaux locaux de taille fixe.

### 6.2 Capacités requises

Une passe calcule ce que le kernel exige :

```rust
struct Requirements { f64: bool, f64_math: bool, i64: bool, f16: bool, atomics_f32: bool, local_mem_bytes: u32, … }
```

- `f64` : le kernel manipule des f64 (arithmétique, comparaisons, `//`, `%`, `sqrt`, `floor`, `ceil`, `abs`,
  `min`, `max`, `fma`, conversions).
- `f64_math` : il appelle en plus une fonction transcendante en f64 (`exp`, `exp2`, `log`, `log2`, `sin`,
  `cos`, `tan`, `tanh`, `atan2`, `pow`, `**`).

| Périphérique | `f64` | `f64_math` |
|---|---|---|
| OpenCL avec `cl_khr_fp64`, CUDA, `cpu:0` | ✅ | ✅ |
| Vulkan avec `SHADER_F64` | ✅ | ❌ (résultat faux silencieux ou plantage du pilote, mesuré) |
| Metal, DX12, GPU sans f64 | ❌ | ❌ |

Comparées aux capacités du périphérique **avant** la génération de code → `gs.UnsupportedFeature` clair au lieu d'une erreur de driver ou d'un résultat faux. Elles servent aussi au choix automatique du périphérique (§4.8).

### 6.3 Interpréteur de référence

`gpusync-ir` contient un interpréteur de l'IR (séquentiel, en Rust). Il sert :
- d'**oracle de tests** pour tous les backends ;
- de mode `gs.debug()` (vérification des bornes, valeur de l'index fautif) ;
- de backend « CPU » de secours toujours disponible.

---

## 7. Génération de code

Un trait commun, une implémentation par cible :

```rust
trait Emitter {
    fn emit(&self, kernel: &ir::Kernel, layout: &LaunchLayout) -> Result<String>;
}
```

| Cible | Version de base | Points d'attention |
|---|---|---|
| **OpenCL C** | 1.2 | suffixe `f` sur les littéraux f32 ; `#pragma OPENCL EXTENSION cl_khr_fp64` seulement si requis ; pas d'atomiques flottants en 1.2 → émulation par boucle CAS ; garde de bornes |
| **WGSL** | wgpu | f64 seulement sur Vulkan avec `SHADER_F64` (extension de naga : type `f64`, littéraux `1.5lf`, paramètres f64 dans un buffer `array<f64>` séparé), sans fonctions transcendantes ; pas de i64 ; taille max d'un buffer et d'une liaison limitée → tailles de bloc (§9.2) |
| **CUDA C** | NVRTC (M6) | littéraux `f`, `__restrict__`, garde de bornes |

Règles communes :
- La grille est arrondie au multiple de la taille de groupe ; chaque kernel commence par une **garde de bornes** (`if (i >= n) return;`).
- La taille de groupe est choisie par le backend selon les limites du périphérique (comme l'auto-tuning du prototype), et mise en cache.
- Le code généré est déterministe (même IR → même texte), ce qui rend le cache et les tests de référence fiables.

---

## 8. Runtime

### 8.1 Concepts

| Concept | Rôle |
|---|---|
| `Device` | un périphérique d'un backend + ses capacités + son budget mémoire |
| `Stream` | une file **ordonnée** par périphérique (MVP : une seule) |
| `Buffer` | mémoire sur le périphérique, issue du pool |
| `Event` | fin d'une opération ; attendu directement, jamais par callback |
| `Op` | upload, kernel, relecture, copie : enregistré dans le stream |

### 8.2 Trait `Backend`

```rust
trait Backend {
    fn devices(&self) -> Vec<DeviceInfo>;
    fn compile(&self, dev: &Device, src: &KernelSource) -> Result<CompiledKernel>;
    fn alloc(&self, dev: &Device, bytes: usize) -> Result<RawBuffer>;
    fn upload(&self, s: &Stream, dst: &RawBuffer, src: &[u8]) -> Result<Event>;
    fn launch(&self, s: &Stream, k: &CompiledKernel, args: &[Arg], grid: Grid) -> Result<Event>;
    fn download(&self, s: &Stream, src: &RawBuffer, dst: &mut [u8]) -> Result<Event>;
    fn wait(&self, e: &Event) -> Result<()>;

    // Mémoire unifiée — optionnel : `None` = non supporté, le runtime revient à la copie
    fn wrap_host(&self, dev: &Device, ptr: *mut u8, len: usize) -> Option<RawBuffer>; // hôte → GPU sans copie
    fn host_view(&self, buf: &RawBuffer) -> Option<HostSlice>;                         // GPU → hôte sans copie
}
```

`DeviceInfo` expose `unified_memory: bool` (en OpenCL : `CL_DEVICE_HOST_UNIFIED_MEMORY`), ainsi que l'alignement requis pour le zéro-copie.

Bibliothèques Rust envisagées (versions et fonctionnalités à vérifier au démarrage du projet) :
- wgpu : `wgpu`.
- OpenCL : `opencl3` / `cl3`, avec **chargement dynamique** de la bibliothèque OpenCL : si elle est absente, le backend est juste indisponible.
- CUDA : `cudarc` (chargement dynamique de `libcuda` et NVRTC).

### 8.3 Exécution d'un appel

```
pairwise_dist(series)
  │ Python : wrapper léger
  ├─ 1. clé de spécialisation (dtypes, rangs, device) → cache ; sinon frontend + codegen + compile
  ├─ 2. lier les dimensions (n=2000, t=365), vérifier les formes
  ├─ 3. GIL libéré ─────────────────────────────── Rust
  │     ├─ si `series` est un np.ndarray : zéro-copie si possible (§8.4), sinon upload
  │     │  (rien à faire si c'est déjà un gs.Array)
  │     ├─ alloue la sortie depuis le pool (en mémoire partagée si unifiée)
  │     └─ enregistre le kernel dans le stream, après l'upload
  └─ 4. renvoie gs.Array(buffer, événement de fin)        ← aucune attente ici

np.asarray(dist)
  └─ GIL libéré : enregistre la relecture, attend son événement, renvoie le tableau NumPy
```

Un appel synchrone complet ne fait donc **qu'une seule attente**, au moment de la relecture.

### 8.4 Mémoire

- **Pool** par périphérique, par classes de taille (puissances de 2 au-delà d'un seuil), avec un budget (par défaut une fraction de la mémoire du périphérique).
- Un buffer libéré par Python (`gs.Array` détruit) retourne au pool **seulement après** l'événement de sa dernière opération.

#### Mémoire séparée (GPU dédié)

- Entrée `np.ndarray` : upload vers un buffer du pool.
- Sortie : relue dans un nouveau `np.ndarray` (ou dans `out=` si fourni).

#### Mémoire unifiée (Apple Silicon, GPU intégrés Intel / AMD, Jetson)

Le GPU accède directement à la RAM de l'hôte : copier reviendrait à dupliquer les données dans la même mémoire physique. Le runtime utilise donc le zéro-copie quand le backend le permet.

| Sens | Stratégie | Repli |
|---|---|---|
| Entrée `np.ndarray` | `wrap_host` : le buffer GPU **utilise la mémoire du tableau** | copie si mémoire non alignée, non contiguë, ou backend sans zéro-copie |
| Sortie | buffer alloué **en mémoire partagée** ; `np.asarray(y)` renvoie une **vue** sur cette mémoire (`host_view`) | relecture par copie |

Mesures sur Apple M5 Pro, OpenCL 1.2, 256 Mo :

| Méthode | Upload + kernel | Tableau NumPy modifié sans copie | Mémoire |
|---|---|---|---|
| `CL_MEM_COPY_HOST_PTR` | 33,3 ms | non | ×2 |
| `CL_MEM_USE_HOST_PTR` | 7,4 ms | oui (lecture du buffer = même adresse que le tableau) | ×1 |

Les gros tableaux NumPy (`np.ones`, `np.empty`) étaient déjà alignés sur une page (16 Ko) : NumPy les obtient directement du système. Les petits tableaux ne le sont pas toujours ; leur copie est peu coûteuse. Pour garantir le zéro-copie, `gs.empty(...)` / `gs.zeros(...)` renvoient des tableaux NumPy alloués alignés.

Durée de vie : une vue NumPy sur une sortie **garde le buffer en vie** (objet `base` du tableau). Le buffer ne retourne au pool qu'après destruction de la vue.

Support par backend :

| Backend | Zéro-copie | Mécanisme |
|---|---|---|
| OpenCL | ✅ mesuré | `CL_MEM_USE_HOST_PTR` en entrée, `CL_MEM_ALLOC_HOST_PTR` + map en sortie |
| Metal natif | ✅ | `newBufferWithBytesNoCopy` (mémoire alignée sur une page), `storageModeShared` |
| wgpu → Metal | ✅ faisable | `newBufferWithBytesNoCopy`, confié à wgpu par `wgpu-hal` (`metal::Device::buffer_from_raw`, vérifié dans wgpu 30) |
| wgpu → Vulkan / DX12 | ❌ | wgpu-hal 30 n'importe pas de mémoire hôte comme buffer (seulement `texture_from_raw`). Le matériel le permet pourtant (`VK_EXT_external_memory_host` présent sur la RTX 4080 SUPER et l'UHD 770, alignement 4 Kio) : il faudrait un backend Vulkan natif |
| CUDA (Jetson, Grace) | ✅ | `cudaHostRegister` ou mémoire managée |

Mesuré à M1 : wgpu copie et consomme 3 à 5 fois la taille des données. La réponse est le moteur de transferts (§9) : zéro-copie là où l'import est possible, transferts par blocs bornés ailleurs.

### 8.5 Threads et GIL

- Toute opération longue (compilation, attente, copie) se fait **GIL libéré**.
- `__await__` : le runtime attend l'événement sur un thread Rust dédié, puis complète un future asyncio via `loop.call_soon_threadsafe`.
- Aucune boucle asyncio interne à la lib.

### 8.6 Cache de compilation

- En mémoire, par processus.
- Sur disque (répertoire de cache utilisateur), clé = hash(code généré, cible, nom et version du driver). Les kernels OpenCL et CUDA compilés sont réutilisés entre deux lancements.

---

## 9. Moteur de transferts (out-of-core intégré)

**Principe : un seul chemin d'exécution.** Tout appel est planifié en blocs ; un tableau qui tient en mémoire
n'est qu'un plan à un bloc. Il n'y a **aucune limite de taille**, seulement des budgets mémoire et des
contraintes de taille de bloc. L'out-of-core n'est pas un mode spécial : c'est la gestion normale des transferts.

### 9.1 Trois modes, choisis automatiquement à chaque appel

| Mode | Quand | Transferts | Mémoire en plus |
|---|---|---|---|
| **Direct** (zéro-copie) | GPU à mémoire unifiée, mémoire importable (OpenCL `USE_HOST_PTR`, Metal `NoCopy`) | aucun : le GPU lit et écrit les tableaux NumPy | 0 |
| **Résident** | GPU dédié, l'appel tient dans le budget VRAM | un transfert DMA par entrée ; le résultat reste en VRAM (chaînage gratuit) | anneau de transfert (RAM, borné) |
| **Flux** | GPU dédié au-delà du budget VRAM ; ou source non importable (memmap refusé, générateur) ; ou wgpu sans import | un transfert par bloc, recouvert par le calcul | anneau de transfert (RAM, borné) + quelques blocs (VRAM) |

Les trois modes exécutent le même code généré : un bloc est vu par le kernel comme un tableau entier.

### 9.2 Blocs

- **Taille d'un bloc** = min(cible par défaut 64 Mio par tableau ; taille max d'un buffer ou d'une liaison du
  périphérique ; budget ÷ (nombre de tampons × nombre de tableaux)), arrondie aux alignements (page de
  16 Kio sur Apple Silicon, 4 Kio ailleurs ; 256 o pour les décalages de liaison wgpu).
- **Les limites matérielles deviennent des tailles de bloc** : 1 Gio par liaison sur l'UHD 770 en Vulkan,
  3,3 Gio par allocation sur le M5 Pro en OpenCL. Aucune n'est plus une erreur.
- **Élément par élément** : un bloc est une sous-plage d'indices de toutes les entrées et de la sortie.
- **Avec index (M2)** : découpage de la sortie le long de sa première dimension ; une entrée qui ne partage
  pas cette dimension est résidente (§9.7).

### 9.3 Transferts sur GPU dédié

- **Anneau de transfert** : K tampons (3 par défaut) de la taille d'un bloc, en mémoire épinglée (OpenCL
  `CL_MEM_ALLOC_HOST_PTR` mappé ; wgpu : buffers `MAP_WRITE` / `MAP_READ` réutilisés). Remplis par `memcpy`
  depuis la source sur un thread Rust (GIL libéré) ; le DMA se fait à pleine vitesse depuis la mémoire épinglée.
- **Pipeline** : pendant que le bloc *k* est calculé, le bloc *k+1* est envoyé et le bloc *k−1* revient
  **directement dans sa destination finale** (tableau NumPy, memmap). OpenCL : une file de transferts et une
  file de calcul synchronisées par événements, donc un vrai recouvrement. wgpu : une seule file ; le
  recouvrement se fait entre les `memcpy` CPU et le travail du GPU.
- **Budget VRAM** : 50 % de la mémoire du périphérique par défaut (OpenCL : `CL_DEVICE_GLOBAL_MEM_SIZE` ;
  Vulkan : tas mémoire lus via wgpu-hal), réglable par `gs.set_memory_budget()`.
- **Pool** : tampons et buffers de blocs réutilisés d'un appel à l'autre, aucune allocation par appel.
- **Coût en RAM borné** : de l'ordre de 3 tampons × 64 Mio × (entrées + sortie), soit 0,4 à 0,8 Gio quel que
  soit le volume (contre 3 à 5 fois le volume mesuré à M1).
- **Le dernier `memcpy`** (source → tampon épinglé) disparaît avec CUDA (M6 : `cudaHostRegister`, DMA direct
  depuis le tableau NumPy) ou un backend Vulkan natif (`VK_EXT_external_memory_host`).

### 9.4 Mémoire unifiée

- Mode Direct **par bloc** : chaque bloc est une sous-plage du tableau NumPy importée sans copie
  (`USE_HOST_PTR` en OpenCL, `newBufferWithBytesNoCopy` en Metal via wgpu-hal). Aucune copie, quel que soit
  le volume ; les blocs servent seulement à respecter la taille max d'une allocation.
- Vulkan via wgpu sur GPU intégré (UHD 770) : pas d'import en wgpu 30, donc mode Flux avec anneau borné.
  OpenCL reste le chemin zéro-copie de l'UHD 770.
- memmap (fichier) : import tenté ; s'il est refusé (pages d'un fichier), mode Flux.

### 9.5 Sources et destinations, sans limite

| Source | Appel | Résultat |
|---|---|---|
| `np.ndarray` en RAM | `f(x)` | `gs.Array` |
| `np.memmap` plus gros que la RAM | `f(x, out=np.memmap(...))` | écrit bloc par bloc dans le fichier de sortie |
| générateur / itérable de blocs | `f.stream(gen)` | itérateur de résultats, un par bloc |

- `out=` : tout tableau contigu de la bonne forme et du bon type (évite aussi d'allouer la sortie).
- Sans `out=`, la sortie est allouée en RAM : si elle ne peut pas l'être, `gs.OutOfMemory` suggère `out=`.

### 9.6 Chaînage

- Mode Résident : le `gs.Array` reste en VRAM, le kernel suivant le lit sans transfert.
- Mode Flux : le résultat est en RAM (ou dans `out=`), le kernel suivant le relit. La fusion de kernels (M7)
  supprimera ces allers-retours.
- Mode Direct : pas de transfert dans aucun cas.

### 9.7 Ce qui est découpable (kernels avec index, M2)

On découpe la grille le long de sa **première dimension**. Pour chaque entrée :
- si elle partage cette dimension (même nom), elle est découpée de la même façon ;
- sinon, elle est **résidente** : uploadée une fois et gardée sur le GPU pendant tout le traitement (généralisation des « arguments constants » du prototype).

Exemples :

| Kernel | Dimension découpée | Découpé | Résident |
|---|---|---|---|
| `normalize(x: ["n"]) -> ["n"]` | `n` | `x`, sortie | — |
| `pairwise_dist(series: ["n","t"]) -> ["n","n"]` | `n` (lignes de la sortie) | sortie | `series` (lu en entier par chaque thread) |

Si les entrées résidentes et un bloc minimal ne tiennent pas dans le budget → `gs.OutOfMemory` qui nomme l'entrée en cause.

### 9.8 Critères de réussite

| Cas | Machine | Pourquoi |
|---|---|---|
| 4 Gio en OpenCL | Mac | dépasse l'allocation max (3,3 Gio) |
| 2 Gio en Vulkan | PC, UHD 770 | dépasse la liaison max (1 Gio) |
| 24 Gio en Vulkan et en OpenCL | PC, RTX | dépasse la VRAM (16 Gio) |
| 40 Gio en memmap, `out=` memmap | Mac | dépasse la RAM (26 Go) |

- Mémoire en plus : 0 en Direct ; au plus la taille de l'anneau (~0,5 Gio) en Résident et en Flux,
  **indépendante du volume**.
- Débit en mode Flux sur la RTX : proche de la bande passante PCIe, transferts et calcul recouverts.
- Résultats identiques à NumPy dans tous les cas (tests différentiels avec des volumes qui forcent le découpage).

---

## 10. Packaging et interop

- **PyO3 en abi3** + **maturin** : une wheel par plateforme, valable pour toutes les versions de Python supportées.
- Wheels : macOS arm64 et x86_64, Linux manylinux x86_64 et aarch64, Windows x86_64.
- **Aucune dépendance Python obligatoire autre que NumPy.**
- wgpu est compilé dans la wheel ; OpenCL et CUDA sont chargés dynamiquement s'ils sont présents.
- Interop : `__array__`, protocole DLPack (export et import) ; `__cuda_array_interface__` avec le backend CUDA.

---

## 11. Tests et qualité

| Niveau | Contenu |
|---|---|
| Unitaires Rust | frontend (KernelAst → IR), passes de l'IR, interpréteur |
| Code généré | tests de référence (« snapshots ») du texte émis pour chaque cible |
| Différentiel | chaque kernel de test exécuté sur l'interpréteur, sur chaque backend disponible et comparé à NumPy ; génération aléatoire d'entrées (tests par propriétés) |
| Bout en bout Python | API, erreurs (messages compris), Jupyter, async |
| Mémoire unifiée | le zéro-copie est réellement utilisé (même adresse, pic mémoire ×1) ; replis corrects (tableau non aligné, non contigu, backend sans zéro-copie) ; durée de vie des vues |
| Performance | overhead par appel, débit sur gros volumes, out-of-core, pic mémoire ; comparaison au prototype et à pyopencl brut |

**CI sans GPU** : interpréteur + POCL (OpenCL sur CPU) + Vulkan logiciel (lavapipe) pour wgpu. Les GPU réels (Mac, NVIDIA) sont testés manuellement ou sur des runners dédiés.

---

## 12. Feuille de route

| Jalon | Contenu | Critère de fin |
|---|---|---|
| **M0** Squelette | workspace Cargo, maturin, `gs.devices()` liste wgpu + OpenCL | `pip install -e .` puis `gs.devices()` sur le Mac |
| **M1** MVP élément par élément | frontend minimal, IR, interpréteur, émetteurs WGSL + OpenCL C, `gs.Array` synchrone, zéro-copie OpenCL sur mémoire unifiée | `normalize` correct sur Metal et OpenCL ; overhead ≤ 0,3 ms ; tests différentiels verts ; pic mémoire ×1 sur 1 Go en OpenCL sur le Mac |
| **M1.5** Moteur de transferts et choix automatique | blocs et plan d'exécution ; modes Direct, Résident, Flux ; Metal `NoCopy` via wgpu-hal ; anneau épinglé et pipeline à deux files (OpenCL) ; `out=`, `np.memmap`, `.stream()` ; budget VRAM ; f64 via Vulkan (`SHADER_F64`) ; besoins `f64` / `f64_math` ; choix automatique du périphérique par kernel et option `compat` (`auto` : avertissement + repli OpenCL ; `strict` : erreur) (§4.8) | critères du §9.8 sur le Mac et le PC ; un kernel f64 arithmétique tourne sur Vulkan (RTX), un kernel f64 avec `exp` sur OpenCL (RTX), sans `device=` |
| **M2** Index et formes | `gs.index()`, dimensions nommées, retours multiples, boucles, `gs.local`, réductions de `gs.Array` | `pairwise_dist` et `stats` corrects ; ta DTW portée et comparée au prototype |
| **M3** Out-of-core avec index | découpage des kernels avec index (§9.7) : sortie découpée, entrées résidentes | `pairwise_dist` sur des données dont la sortie dépasse la VRAM |
| **M4** Async et interop | `await`, DLPack, cache disque | même script en synchrone, async et Jupyter |
| **M5** Mode explicite | `grid=`, `gs.Out`, atomiques (dont émulation OpenCL 1.2) | histogramme correct sur toutes les cibles |
| **M6** CUDA | émetteur CUDA C, backend `cudarc` | parité des tests avec les autres backends sur NVIDIA |
| **M7** Avancé | multi-GPU, plusieurs streams, fusion de kernels élément par élément | à définir avec des mesures |

---

## 13. Décisions ouvertes

| # | Question | Piste actuelle |
|---|---|---|
| D1 | IR maison ou bâtir sur CubeCL ? | IR maison, car OpenCL C est une cible obligatoire. Vérifier les cibles actuelles de CubeCL avant de trancher |
| D2 | Version minimale de Python | ✅ 3.10 (abi3), décidé à M0 |
| D3 | `f16` dans le MVP ? | non, à M2 si utile |
| D4 | Strides (vues non contiguës) | copie contiguë dans le MVP ; à mesurer |
| D5 | Nom du projet : garder « GPUsync » ? | à décider |
| D6 | Licence | MIT, comme le prototype |
| D7 | Syntaxe du mode explicite (`grid=` + `gs.Out`) | à valider sur 2–3 vrais cas (histogramme, scatter) |
| D8 | API des réductions : méthodes de `gs.Array` seulement, ou aussi des kernels de réduction écrits par l'utilisateur ? | méthodes d'abord |
| D9 | Backend natif (Metal, Vulkan) en plus de wgpu pour le zéro-copie ? | Metal : inutile, wgpu-hal permet `NoCopy` (M1.5). Vulkan : nécessaire pour importer la mémoire hôte (absent de wgpu 30) ; à décider après M1.5 selon le coût mesuré de l'anneau de transfert, sachant que CUDA (M6) couvre la RTX |
| D10 | Backend par défaut ? | ✅ remplacé par le choix automatique par kernel (§4.8, M1.5) : Metal sur Mac si `NoCopy` tient le critère « 0 en plus », OpenCL pour les GPU intégrés, Vulkan pour les GPU dédiés, OpenCL quand `f64_math` est requis |
