# GPUsync v2 — Architecture

> Statut : **M0 terminé** (19/11/2026). Ce document fixe les décisions avant d'écrire le cœur en Rust.
> Il s'appuie sur le prototype Python (`GPUsync-python/`), testé sur Apple M5 Pro (OpenCL 1.2), le 19/11/2026.

### Avancement

| Jalon | État | Validé sur |
|---|---|---|
| M0 Squelette | ✅ workspace Cargo, PyO3 0.30 (abi3-py310) + maturin 1.16, `gs.devices()` / `gs.backends()` via wgpu 30 et opencl3 0.12 (chargement dynamique) | Mac M5 Pro (Python 3.15) ; PC Windows RTX 4080 SUPER + UHD 770 (Python 3.10) |
| M1 MVP | à faire | |

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
#  Device('Apple M5 Pro', backend='opencl', version='1.2', f64=False, unified_memory=True)]

with gs.device("opencl"):            # pour un bloc de code
    ...
normalize(x, 0.0, 1.0, device="opencl:0")   # pour un appel
```

Sélection par défaut (premier disponible) :

| Plateforme | Ordre |
|---|---|
| macOS | Metal (wgpu) → OpenCL → CPU |
| Linux / Windows | CUDA (à partir de M6) → Vulkan / DX12 (wgpu) → OpenCL → CPU |

Le backend « CPU » est l'interpréteur de l'IR (lent, toujours disponible) ; POCL via OpenCL est l'option CPU rapide.

L'ordre sur macOS est provisoire : wgpu copie les données alors qu'OpenCL ne copie pas (§8.4). Il sera revu avec les mesures de M1 (décision D10).

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
struct Requirements { f64: bool, i64: bool, f16: bool, atomics_f32: bool, local_mem_bytes: u32, … }
```

Comparées aux capacités du périphérique **avant** la génération de code → `gs.UnsupportedFeature` clair au lieu d'une erreur de driver ou d'un résultat faux.

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
| **WGSL** | wgpu | pas de f64 ni de i64 ; nombre de buffers par kernel et taille max d'un buffer limités (limites par défaut de wgpu, relevables selon l'adaptateur) → à prendre en compte dans le découpage out-of-core |
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
| wgpu | ⚠️ partiel | pas d'API pour utiliser un pointeur existant → au moins une copie en entrée. La fonctionnalité native `MAPPABLE_PRIMARY_BUFFERS` pourrait éviter la copie en sortie (à vérifier) |
| CUDA (Jetson, Grace) | ✅ | `cudaHostRegister` ou mémoire managée |

Conséquence : sur Mac, le zéro-copie passe d'abord par **OpenCL** (dès M1). Un **backend Metal natif** n'est ajouté que si les mesures montrent que les copies de wgpu coûtent cher sur de gros volumes (décision D9).

### 8.5 Threads et GIL

- Toute opération longue (compilation, attente, copie) se fait **GIL libéré**.
- `__await__` : le runtime attend l'événement sur un thread Rust dédié, puis complète un future asyncio via `loop.call_soon_threadsafe`.
- Aucune boucle asyncio interne à la lib.

### 8.6 Cache de compilation

- En mémoire, par processus.
- Sur disque (répertoire de cache utilisateur), clé = hash(code généré, cible, nom et version du driver). Les kernels OpenCL et CUDA compilés sont réutilisés entre deux lancements.

---

## 9. Out-of-core

### 9.1 Ce qui est découpable

On découpe la grille le long de sa **première dimension**. Pour chaque entrée :
- si elle partage cette dimension (même nom), elle est découpée de la même façon ;
- sinon, elle est **résidente** : uploadée une fois et gardée sur le GPU pendant tout le traitement (généralisation des « arguments constants » du prototype).

Exemples :

| Kernel | Dimension découpée | Découpé | Résident |
|---|---|---|---|
| `normalize(x: ["n"]) -> ["n"]` | `n` | `x`, sortie | — |
| `pairwise_dist(series: ["n","t"]) -> ["n","n"]` | `n` (lignes de la sortie) | sortie | `series` (lu en entier par chaque thread) |

Si les entrées résidentes et un bloc minimal ne tiennent pas dans le budget → `gs.OutOfMemory` qui nomme l'entrée en cause.

### 9.2 Pipeline

Triple tampon par défaut : pendant que le bloc *k* est calculé, le bloc *k+1* est uploadé et le bloc *k−1* est relu. Taille des blocs déduite du budget mémoire, de la taille max d'un buffer du backend et du nombre de tampons.

### 9.3 Sources et destinations

| Source | Appel | Résultat |
|---|---|---|
| `np.ndarray`, `np.memmap` | `f(x)` | `gs.Array` out-of-core : `np.asarray` le matérialise en RAM, `.save()` l'écrit sur disque bloc par bloc |
| générateur / itérable de blocs | `f.stream(gen)` | itérateur de `np.ndarray`, un par bloc |

### 9.4 Sur mémoire unifiée

La limite n'est plus la mémoire du GPU mais **la RAM**, puisque le GPU y accède directement.

- Données qui tiennent en RAM : **pas de découpage**, un seul lancement en zéro-copie.
- `memmap` ou générateur plus gros que la RAM : découpage par blocs comme ci-dessus, chaque bloc étant lu dans un tampon aligné réutilisé, puis traité en zéro-copie.
- Pas de triple tampon : il n'y a plus de transfert à recouvrir par le calcul. On garde seulement un double tampon pour recouvrir **la lecture disque** et le calcul.
- La taille max d'un buffer du backend (notamment wgpu) peut encore imposer un découpage.

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
| **M2** Index et formes | `gs.index()`, dimensions nommées, retours multiples, boucles, `gs.local`, réductions de `gs.Array` | `pairwise_dist` et `stats` corrects ; ta DTW portée et comparée au prototype |
| **M3** Out-of-core | découpage, résidents, triple tampon, `memmap`, `.stream()`, `.save()`, variante mémoire unifiée (§9.4) | 50 Go traités avec un budget de 2 Go ; débit ≥ prototype ; mesure du coût des copies wgpu vs OpenCL zéro-copie (entrée de D9) |
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
| D9 | Backend Metal natif (en plus de wgpu) pour le zéro-copie sur Mac ? | décider après M3, selon le coût mesuré des copies wgpu. D'ici là, OpenCL assure le zéro-copie sur Mac |
| D10 | Sur Mac, Metal (wgpu) reste-t-il le backend par défaut s'il copie alors qu'OpenCL ne copie pas ? | à trancher avec les mesures de M1 : priorité possible à OpenCL pour les gros volumes, tant que D9 n'est pas résolue |
