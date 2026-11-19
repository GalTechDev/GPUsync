# GPUsync v2

Écrire des kernels GPU en Python et les exécuter sur n'importe quel GPU : Metal, Vulkan, DX12 (via wgpu) et
OpenCL ≥ 1.2, puis CUDA. Réécriture en Rust du prototype Python ; la conception est décrite dans
[ARCHITECTURE.md](ARCHITECTURE.md).

**État : jalon M0** — squelette du projet et découverte des périphériques.

```python
>>> import gpusync as gs
>>> gs.devices()
[Device('NVIDIA GeForce RTX 4080 SUPER', backend='vulkan', f64=True, unified_memory=False),
 Device('Intel(R) UHD Graphics 770', backend='vulkan', f64=False, unified_memory=True),
 Device('NVIDIA GeForce RTX 4080 SUPER', backend='opencl', f64=True, unified_memory=False, memory='16.0 Go'),
 Device('Intel(R) UHD Graphics 770', backend='opencl', f64=False, unified_memory=True, memory='29.5 Go')]
>>> gs.backends()          # pourquoi un backend est (in)disponible
```

Diagnostic complet de la machine : `python -m gpusync`.

## Développement

Prérequis : Rust (rustup), Python ≥ 3.10, et un compilateur C (Xcode Command Line Tools sur macOS,
Visual Studio Build Tools « C++ » sur Windows).

```bash
python3 tools/dev.py setup   # crée .venv, installe maturin et pytest, compile l'extension (pip install -e .)
python3 tools/dev.py test    # recompile, puis cargo test + pytest + diagnostic
```

`GS_EXPECT=metal,opencl` (ou `vulkan,opencl`) fait échouer les tests si ces backends n'ont aucun périphérique
sur la machine.

Sur toutes les machines de test à la fois (voir [tools/remote](tools/remote/README.md)) :

```bash
python3 tools/remote/gs_remote.py run all -- @python tools/dev.py test
```

## Organisation

```
crates/
  gpusync-runtime/          types communs (DeviceInfo, trait Backend), découverte
  gpusync-backend-wgpu/     Metal, Vulkan, DX12 (un adaptateur par GPU physique)
  gpusync-backend-opencl/   OpenCL, chargé dynamiquement (absent = backend indisponible)
  gpusync-py/               liaison PyO3 → module gpusync._core
python/gpusync/             package Python
tests/                      tests Python
tools/dev.py                tâches de développement multi-plateformes
tools/remote/               exécution sur les machines de test
GPUsync-python/             prototype v1 (référence, hors dépôt)
```
