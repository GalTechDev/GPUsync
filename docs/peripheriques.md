# Périphériques

## Lister

```python
import numpy as np
import gpusync as gs

for d in gs.devices():
    print(d.id, d.name, d.backend, d.kind, d.memory_bytes, d.f64, d.unified_memory)
```

Attributs d'un `gs.Device` :

| Attribut | Sens |
|---|---|
| `id` | identifiant, ex. `"metal:0"`, `"cuda:0"`, `"opencl:1"`, `"cpu:0"` |
| `name`, `vendor`, `version` | nom du matériel, fabricant, version du pilote ou de l'API |
| `backend` | `"cuda"`, `"metal"`, `"vulkan"`, `"dx12"`, `"opencl"` ou `"cpu"` |
| `kind` | `"discrete_gpu"`, `"integrated_gpu"`, `"cpu"`… |
| `index` | numéro dans son backend |
| `memory_bytes`, `max_buffer_bytes` | mémoire du périphérique, plus gros tampon possible (`None` si inconnus) |
| `unified_memory` | le GPU partage la mémoire centrale (Apple Silicon, GPU intégrés) |
| `f64` | calcul en `gs.f64` possible |
| `f64_math` | fonctions mathématiques (`exp`, `sin`…) en f64 possibles |
| `int64` | calcul en `gs.i64` / `gs.u64` possible |
| `max_group`, `shared_bytes` | taille maximale d'un groupe de threads et de sa mémoire partagée (voir [Groupes](groupes.md)) |

La découverte est faite une fois ; `gs.devices(refresh=True)` la refait. `gs.backends()` donne l'état de chaque
backend et, s'il est indisponible, la raison (`st.error`).

## Choix automatique

Sans indication, GPUsync prend le premier périphérique de ce classement :

| Rang | Catégorie | Backends, dans l'ordre |
|---|---|---|
| 1 | GPU dédié | CUDA, Vulkan, OpenCL, DX12, Metal |
| 2 | GPU intégré | Metal, OpenCL, Vulkan, DX12 |
| 3 | OpenCL sur processeur (POCL…) | |
| 4 | interpréteur | `cpu:0` |

Sur un Mac : `metal:0`. Sur un PC à carte NVIDIA : `cuda:0` si NVRTC est présent, sinon `vulkan:0`.

## Choisir soi-même

Un périphérique se désigne par un `gs.Device`, son identifiant (`"opencl:1"`), un nom de backend (`"cuda"`,
`"opencl"`, `"metal"`, `"vulkan"`, `"dx12"`, `"cpu"` : le premier de ce backend) ou `"gpu"` (le premier GPU du
classement).

```python
@gs.kernel
def square(x: gs.f32[:]) -> gs.f32[:]:
    return x * x

x = np.arange(4, dtype=np.float32)

square(x, device="cpu")                 # pour un appel

with gs.device("cpu"):                  # pour un bloc de code (propre au thread)
    y = square(x)
assert y.device == "cpu:0"

gs.set_default_device("cpu")            # pour la suite du programme
gs.set_default_device(None)             # retour au classement
```

Ordre de priorité : `device=` de l'appel, puis `with gs.device(...)`, puis `gs.set_default_device(...)`, puis la
variable d'environnement `GPUSYNC_DEVICE`, puis le classement. Un nom introuvable lève `gs.DeviceError`, qui liste
les périphériques disponibles.

`y.device` dit toujours où le calcul a réellement eu lieu, et `gs.default_device(kernel)` où il aurait lieu
maintenant, compte tenu de ce que le kernel demande.

## Quand une opération manque : `compat`

Tous les périphériques ne savent pas tout faire : pas de f64 sur Metal ni sur la plupart des GPU intégrés, pas
de fonctions mathématiques en f64 sur Vulkan, entiers 64 bits absents par endroits. GPUsync déduit de chaque
kernel ce qu'il exige, et si le périphérique prévu ne convient pas :

| `@gs.kernel(compat=...)` | Comportement |
|---|---|
| `"auto"` (défaut) | avertissement `gs.CompatibilityWarning` (une fois par kernel), puis **tout le kernel** s'exécute ailleurs : CUDA ou OpenCL sur le même GPU de préférence, sinon un autre périphérique capable, sinon `cpu:0` (exact mais lent, et l'avertissement le dit) |
| `"strict"` | `gs.UnsupportedFeature`, avec la liste des périphériques compatibles |

```python
import warnings

@gs.kernel
def decay(x: gs.f64[:]) -> gs.f64[:]:
    return gs.exp(-x)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = decay(np.ones(3))
print(r.device, [str(w.message) for w in caught])   # sur un Mac : cpu:0, avec l'explication
assert np.allclose(r, np.exp(-1.0))
```

Rien n'est émulé : un f64 n'est jamais simulé par deux f32, et un kernel n'est jamais coupé entre deux
périphériques. Avec `device=` explicite, le périphérique demandé est utilisé tel quel, ou l'appel est refusé.

Sur le PC de test, par exemple : un kernel f32 → `cuda:0` ; f64 arithmétique sous Vulkan → `vulkan:0` ; f64 avec
`gs.exp` sous Vulkan → `opencl:0` (même carte).

## Mémoire

Le **budget** d'un périphérique est la mémoire qu'un appel peut occuper d'un coup avant de passer en mode `flux`
(par blocs) : 50 % de la mémoire du périphérique par défaut.

```python
gs.set_memory_budget(512 * 2**20)       # 512 Mio
gs.set_memory_budget(None)              # retour au défaut
```

Le baisser laisse de la place à d'autres programmes (ou à un rendu) ; le monter évite le découpage de calculs
qui tiennent. Il ne limite pas la taille des données : au-delà, le calcul est seulement découpé.

## Un GPU qui décroche

Si le pilote perd le périphérique (sous Windows, un calcul de plus de 2 s environ est interrompu par le
système), l'appel lève `gs.DeviceError` avec une explication, et le périphérique est rouvert à l'appel suivant.
Découpez alors le travail en appels plus courts.
