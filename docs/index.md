# ⚡ GPUsync

Bienvenue sur la documentation officielle de **GPUsync**, une bibliothèque de programmation GPU asynchrone, JIT et *Out-of-Core* conçue pour simplifier et optimiser les calculs massifs sur cartes graphiques avec Python.

GPUsync abstrait la complexité de l'allocation VRAM, de la synchronisation asynchrone des streams et de l'auto-tuning matériel, tout en offrant une intégration transparente avec NumPy et DLPack (PyTorch, CuPy).

---

## 🚀 Quickstart (Démarrage Rapide)

Voici un exemple minimal pour compiler et exécuter un noyau JIT OpenCL de façon totalement transparente (sans configuration manuelle) :

```python
import numpy as np
from gpusync import gpusync_kernel, use_gpu, AsyncBuffer

# 1. Définir un noyau avec le décorateur gpusync_kernel
@gpusync_kernel
def vector_add(a: list, b: list, c: list):
    # La variable globale get_global_id est injectée à l'exécution
    idx = get_global_id(0)
    c[idx] = a[idx] + b[idx]

# 2. Exécuter dans un context manager GPU
with use_gpu(backend="opencl"):
    # Données hôte
    size = 1024 * 1024
    a_host = np.random.rand(size).astype(np.float32)
    b_host = np.random.rand(size).astype(np.float32)

    # Conversion automatique des arguments NumPy
    # vector_add va allouer les buffers VRAM et téléverser les données à la volée
    c_gpu = vector_add(a_host, b_host)

    # Récupération asynchrone du résultat vers l'hôte
    c_host = c_gpu.to_host()

    # Vérification
    assert np.allclose(c_host, a_host + b_host)
    print("Calcul GPU réussi !")
```

---

## 💎 Fonctionnalités Clés

* **Zéro-Configuration (Usability)** : Aucun besoin de configurer manuellement le driver ou d'instancier des structures de plomberie complexes. Utilisez simplement le context manager `use_gpu` et passez vos tableaux NumPy directement à vos noyaux décorés.
* **Memory Pooling VRAM Natif** : GPUsync recycle automatiquement la mémoire physique inutilisée (grâce à son destructeur `__del__` intelligent sur `AsyncBuffer`). Les allocations futures de taille identique réutilisent le pool instantanément sans solliciter le driver matériel (coût de 0 microseconde).
* **Auto-Tuning Matériel Dynamique** : Si la taille locale (`local_size`) n'est pas spécifiée, la bibliothèque interroge le compilateur du pilote hôte pour calculer et équilibrer dynamiquement la géométrie de thread optimale (`WORK_GROUP_SIZE` et `PREFERRED_WORK_GROUP_SIZE_MULTIPLE`).
* **Compilations Hybrides & SPIR-V** : Prise en charge transparente de la compilation textuelle classique OpenCL C99, ainsi que de la compilation intermédiaire SPIR-V (grâce au support binaire natif et à la recherche intelligente du compilateur `clspv` embarqué ou système).
* **Diagnostics Symboliques (SymPy)** : Analyse d'AST et résolution symbolique des équations de bornes de tableaux lors de l'exécution pour valider la thread-safety et générer des rapports de débuggage structurés.
