# GPUsync v2

Écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU : Metal, Vulkan, DX12 (via wgpu) et
OpenCL ≥ 1.2, puis CUDA. Réécriture en Rust du prototype Python ; la conception est décrite dans
[ARCHITECTURE.md](ARCHITECTURE.md).

**État : jalon M1** — kernels élément par élément.

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

normalize(x, 0.0, 1.0, device="opencl")      # choisir le périphérique
with gs.device("cpu"):                       # interpréteur de référence (lent, toujours disponible)
    normalize(x, 0.0, 1.0)

print(normalize.source("opencl"))            # code généré (aussi "wgsl", "ir")
```

Dans un kernel : arithmétique, comparaisons, `and` / `or` / `not`, `a if c else b`, `if` / `elif` / `else`,
`for i in range(...)`, `while`, `break`, `continue`, variables locales, constantes globales numériques,
fonctions `gs.*` et `math.*` (`sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `atan2`, `floor`, `pow`, `abs`,
`min`, `max`…), conversions `float()`, `int()`, `gs.f32()`, `gs.i32()`, `gs.u32()`. Types : `gs.f32`,
`gs.f64` (si le périphérique le permet), `gs.i32`, `gs.u32`. `//` et `%` se comportent comme en Python.

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
tools/remote/               exécution sur les machines de test
GPUsync-python/             prototype v1 (référence, hors dépôt)
```
