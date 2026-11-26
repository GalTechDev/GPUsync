# Installation

GPUsync n'est pas publié sur PyPI : on le compile depuis le dépôt. La compilation produit un module natif
(`gpusync._core`, écrit en Rust) et installe le paquet Python en mode modifiable.

## Prérequis

| | macOS | Windows | Linux |
|---|---|---|---|
| Rust | `rustup` | `rustup` | `rustup` |
| Python | ≥ 3.10 | ≥ 3.10 | ≥ 3.10 |
| Compilateur C | Xcode Command Line Tools | Visual Studio Build Tools, charge « C++ » | `gcc` ou `clang` |

Aucune bibliothèque GPU n'est nécessaire à la compilation : les pilotes (OpenCL, CUDA) sont chargés au
démarrage s'ils sont présents, et leur absence désactive seulement le backend correspondant.

## Compiler

```bash
python3 tools/dev.py setup      # crée .venv, installe maturin, pytest, NumPy, puis compile
```

Après une modification du code Rust :

```bash
python3 tools/dev.py build      # recompile l'extension dans .venv
python3 tools/dev.py test       # recompile, puis cargo test, pytest et le diagnostic
```

Les scripts se lancent avec le Python de `.venv` (`.venv/bin/python`, ou `.venv\Scripts\python.exe` sous
Windows), ou par `python3 tools/dev.py run script.py`.

Seule dépendance d'exécution : NumPy ≥ 1.22.

`setup` et `build` installent aussi le moteur graphique `gpusync.gfx`, qui est un paquet à part
([gpusync-gfx](../gfx/README.md), dossier `gfx/`, Python pur). Pour la bibliothèque seule, ignorez-le : sans lui,
tout fonctionne, et `gs.gfx` lève une `ImportError` qui donne la commande d'installation. Pour l'ajouter ou le
retirer à la main :

```bash
.venv/bin/python -m pip install --no-build-isolation --no-deps -e gfx
.venv/bin/python -m pip uninstall gpusync-gfx
```

## Vérifier ce qui est détecté

```bash
.venv/bin/python -m gpusync
```

affiche chaque backend, ses périphériques et, s'il est indisponible, pourquoi (ici sur le Mac) :

```text
gpusync 0.1.0 — Darwin arm64, Python 3.15.2

[cuda] indisponible : pas de driver CUDA sous macOS (aucun GPU NVIDIA)
[wgpu] 1 périphérique(s)
  metal:0    Apple M5 Pro
             integrated_gpu, Apple, metal | mémoire ?, buffer max 4.0 Go | mémoire unifiée
[opencl] 1 périphérique(s)
  opencl:0   Apple M5 Pro
             integrated_gpu, Apple, OpenCL 1.2 | mémoire 17.8 Go, buffer max 3.3 Go | mémoire unifiée
[cpu] 1 périphérique(s)
  cpu:0      Interpréteur GPUsync (CPU, lent)
             cpu, GPUsync, 0.1.0 | mémoire ?, buffer max ? | f64, mémoire unifiée

Périphérique par défaut : Device('Apple M5 Pro', backend='metal', f64=False, unified_memory=True)
```

Depuis Python :

```python
import gpusync as gs

for st in gs.backends():                       # un état par backend : cuda, wgpu, opencl, cpu
    print(st.name, st.available, st.error)
for d in gs.devices():                         # périphériques utilisables
    print(d.id, d.name, d.kind, "f64" if d.f64 else "sans f64")
```

`cpu:0` existe toujours : c'est l'interpréteur de référence de GPUsync, exact mais lent, utile pour vérifier
un résultat ou déboguer.

## Backends

| Backend | Périphériques | Remarques |
|---|---|---|
| CUDA | GPU NVIDIA | préféré sur un GPU NVIDIA ; demande le pilote et NVRTC (voir plus bas) |
| wgpu | Metal (macOS), Vulkan, DX12 | Metal sur Mac ; Vulkan sur les GPU dédiés sans CUDA |
| OpenCL ≥ 1.2 | tout pilote OpenCL installé | zéro-copie sur les GPU à mémoire unifiée ; repli quand une opération manque ailleurs |
| cpu | interpréteur | toujours disponible |

### CUDA et NVRTC

Le backend CUDA compile les kernels avec NVRTC, la bibliothèque de compilation de NVIDIA. Sans elle,
`gs.backends()` l'explique et Vulkan prend le relais sur le même GPU. Deux façons de l'obtenir :

```bash
pip install nvidia-cuda-nvrtc-cu12      # une bibliothèque, pas un compilateur à installer
```

ou le toolkit CUDA. GPUsync cherche NVRTC dans les roues pip `nvidia-*` visibles par Python, puis dans le
toolkit et le `PATH`. Le code compilé (PTX) est gardé en cache sur disque.

## Variables d'environnement

| Variable | Effet |
|---|---|
| `GPUSYNC_DEVICE` | périphérique par défaut, ex. `opencl`, `cuda:0`, `cpu` (voir [Périphériques](peripheriques.md)) |
| `GPUSYNC_CACHE_DIR` | dossier du cache du code CUDA compilé |
| `GPUSYNC_CACHE=0` | désactive ce cache |
| `GPUSYNC_NVRTC` | chemin exact de la bibliothèque NVRTC |
| `GPUSYNC_NVRTC_DIRS` | dossiers où chercher NVRTC (séparés par `:`, ou `;` sous Windows) ; vide : ne pas chercher dans les roues pip |
| `GPUSYNC_SPECIALIZE=0` | désactive la spécialisation des tailles (voir [Erreurs, débogage, performances](erreurs.md)) |
| `GPUSYNC_DUMP_DIR` | écrit dans ce dossier chaque code généré au moment de sa compilation |

## Tests

```bash
python3 tools/dev.py test
```

Les tests différentiels exécutent chaque kernel sur tous les périphériques de la machine et comparent les
résultats à l'interpréteur. `GS_EXPECT=metal,opencl` (ou `vulkan,opencl`…) fait échouer la suite si l'un de
ces backends n'a aucun périphérique, pour ne pas croire testé ce qui ne l'est pas. Les tests du moteur
(`gfx/tests`) sont lancés avec les autres s'il est installé, ignorés sinon.
