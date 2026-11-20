# GPUsync v2

Écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU : Metal, Vulkan, DX12 (via wgpu) et
OpenCL ≥ 1.2, puis CUDA. Réécriture en Rust du prototype Python ; la conception est décrite dans
[ARCHITECTURE.md](ARCHITECTURE.md).

**État : jalon M5** — kernels élément par élément, kernels avec index (dimensions nommées, plusieurs
sorties, tableaux locaux) et mode explicite (opérations atomiques), sans limite de taille, sur le
périphérique le plus adapté.

```python
import numpy as np
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

x = np.random.rand(1_000_000).astype(np.float32)
y = normalize(x, x.mean(), x.std())          # gs.Array : calcul lancé sur le GPU par défaut
print(y[:5])                                 # attend la fin et rapatrie
z = normalize(y, 0.0, 2.0)                   # chaînage : y reste sur le GPU
np.asarray(z)                                # tableau NumPy

normalize(x, 0.0, 1.0, device="opencl")      # choisir le périphérique (sinon : choix automatique)
with gs.device("cpu"):                       # interpréteur de référence (lent, toujours disponible)
    normalize(x, 0.0, 1.0)

# Sans limite de taille : plus gros que la VRAM, que la RAM, ou en flux
big = np.memmap("entree.f32", dtype=np.float32, mode="r")
out = np.memmap("sortie.f32", dtype=np.float32, mode="w+", shape=big.shape)
normalize(big, 0.0, 1.0, out=out)            # traité par blocs, fichier → GPU → fichier
for bloc in normalize.stream(generateur(), 0.0, 1.0):
    ...

print(normalize.source("opencl"))            # code généré (aussi "wgsl", "ir")

# Mode explicite : un thread par entrée, sorties écrites par opérations atomiques
@gs.kernel(grid="data")
def histogram(data: gs.f32["n"], counts: gs.Out[gs.i32[10]]):
    i = gs.index()
    gs.atomic_add(counts, gs.min(int(data[i] * 10), 9), 1)   # aussi atomic_min / atomic_max

counts = histogram(data)                     # gs.Out alloué et mis à zéro ; out=acc pour accumuler
# indice hors limites : écriture non faite, gs.OutOfBounds à la lecture

# Async, Jupyter et interop : le même code partout
y = await normalize(x, 0.0, 1.0)             # attend sans bloquer la boucle asyncio (ou une cellule Jupyter)
y = await asyncio.to_thread(normalize, big, 0.0, 1.0, out=out)   # gros appel par blocs, hors de la boucle
t = torch.from_dlpack(y)                     # DLPack, sans copie ; les tenseurs CPU sont acceptés en entrée
y.save("resultat.npy")                       # écrit par blocs depuis le GPU
```

Avec index : on écrit le calcul d'**une case de la sortie**, dont la forme vient des dimensions nommées des
entrées.

```python
@gs.kernel
def pairwise_dist(x: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
    i, j = gs.index()                        # la case (i, j) de la sortie
    acc = 0.0
    for k in range(x.shape[1]):
        d = x[i, k] - x[j, k]
        acc += d * d
    return gs.sqrt(acc)

dist = pairwise_dist(series)                 # (2000, 365) → sortie (2000, 2000)

@gs.kernel
def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    buf = gs.local(gs.f32, 64)               # tableau propre à chaque case, taille constante
    ...
    return mean, std

mean, std = stats(series)                    # un gs.Array par sortie ; out=(a, b) possible
dist.max(), dist.mean(axis=0)                # réductions (sur l'hôte, via NumPy)
```

Une même dimension doit avoir la même taille dans toutes les entrées (sinon `gs.ShapeMismatch` avant tout
calcul). Sans limite de taille là aussi : la sortie est calculée par blocs de lignes (19,3 Gio de distances
sur une RTX de 16 Go en 3,8 s) ; les entrées lues seulement à la ligne courante (`x[i, k]`) sont découpées
avec elle, les autres restent entières sur le GPU. Exemple complet : le DTW de `tests/test_index.py`,
identique au prototype.

Dans un kernel : arithmétique, comparaisons, `and` / `or` / `not`, `a if c else b`, `if` / `elif` / `else`,
`for i in range(...)`, `while`, `break`, `continue`, variables locales, constantes globales numériques,
fonctions `gs.*` et `math.*` (`sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `atan2`, `floor`, `pow`, `abs`,
`min`, `max`…), conversions `float()`, `int()`, `gs.f32()`, `gs.i32()`, `gs.u32()`. Types : `gs.f32`,
`gs.f64` (si le périphérique le permet), `gs.i32`, `gs.u32`. `//` et `%` se comportent comme en Python.

**Périphérique et mémoire.** Sans `device=`, GPUsync prend le premier périphérique du classement (GPU dédié :
Vulkan puis OpenCL ; GPU intégré : Metal puis OpenCL ; CPU). Si le kernel utilise une opération que ce
périphérique n'a pas (ex. `gs.exp` en f64 sur Vulkan), `@gs.kernel(compat="auto")` (défaut) avertit et
exécute le kernel en OpenCL, puis sur `cpu:0` en dernier recours ; `compat="strict"` lève une erreur.
Chaque appel est exécuté dans le mode le plus économe (`y.mode`) :

| Mode | Quand | Mémoire en plus |
|---|---|---|
| `direct` | GPU à mémoire unifiée (Metal, OpenCL sur Apple Silicon et GPU intégrés) : aucune copie | ≈ 0 |
| `resident` | GPU dédié, l'appel tient dans le budget (`gs.set_memory_budget`, 50 % de la VRAM) | bornée |
| `flux` | au-delà : par blocs, transferts recouverts par le calcul | bornée (~0,3 à 1,2 Gio) |

Les erreurs disent ce qui ne va pas, où et comment corriger :

```
gs.TypeMismatch: argument 'x' : gs.f32[:] attendu, reçu un tableau float64.
Utilisez x.astype(np.float32), ou @gs.kernel(cast='auto') pour convertir
```

`gs.devices()` liste les périphériques, `gs.backends()` explique pourquoi un backend est indisponible,
`python -m gpusync` affiche un diagnostic complet.

## Développement

Prérequis : Rust (rustup), Python ≥ 3.10, et un compilateur C (Xcode Command Line Tools sur macOS,
Visual Studio Build Tools « C++ » sur Windows).

```bash
python3 tools/dev.py setup                       # .venv + maturin, pytest, numpy + compilation
python3 tools/dev.py test                        # recompile, puis cargo test + pytest + diagnostic
python3 tools/dev.py run tools/bench_m1.py tout  # surcoût par appel, débit, pic mémoire
python3 tools/dev.py run tools/bench_m15.py ram 4 opencl:0   # gros volumes (§9.8 de l'architecture)
python3 tools/dev.py run tools/bench_m2.py       # DTW sur chaque périphérique
python3 tools/dev.py run tools/bench_m3.py pairwise 72000 16 vulkan:0   # sortie plus grosse que la VRAM
```

`GS_EXPECT=metal,opencl` (ou `vulkan,opencl`) fait échouer les tests si ces backends n'ont aucun périphérique
sur la machine. Sur toutes les machines de test à la fois (voir [tools/remote](tools/remote/README.md)) :

```bash
python3 tools/remote/gs_remote.py run all -- @python tools/dev.py test
```

## Organisation

```
crates/
  gpusync-ir/               IR typée, sémantique, interpréteur de référence
  gpusync-frontend/         AST Python (JSON) → IR : typage, erreurs
  gpusync-codegen/          IR → OpenCL C, WGSL
  gpusync-runtime/          types communs (DeviceInfo, trait Backend), découverte
  gpusync-backend-wgpu/     Metal, Vulkan, DX12 : découverte et exécution
  gpusync-backend-opencl/   OpenCL (chargé dynamiquement) : découverte et exécution, zéro-copie
  gpusync-core/             compilation + lancement (sans Python) ; test différentiel multi-GPU
  gpusync-py/               liaison PyO3 → module gpusync._core
python/gpusync/             package Python : @gs.kernel, gs.Array, types, périphériques
tests/                      tests Python (différentiels sur chaque périphérique, erreurs)
tools/dev.py                tâches de développement multi-plateformes
tools/bench_m1.py           mesures du jalon M1
tools/bench_m15.py          essais à gros volume du jalon M1.5
tools/bench_m2.py           DTW sur chaque périphérique (jalon M2)
tools/bench_m3.py           kernels avec index à gros volume (jalon M3)
tools/bench_m4.py           coût du premier appel dans un processus neuf (jalon M4)
tools/bench_m5.py           atomiques (histogramme, somme par case) contre NumPy (jalon M5)
tools/remote/               exécution sur les machines de test
GPUsync-python/             prototype v1 (référence, hors dépôt)
```
