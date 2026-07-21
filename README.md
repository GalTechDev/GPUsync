# GPUsync

**GPUsync** est un framework open-source novateur d'accélération matérielle Multi-GPU. Il offre une surcouche asynchrone intelligente capable de transpiler du code Python vers du C++ (CUDA), du C99 (OpenCL), du JIT CPU (LLVM) et de l'assembleur PTX (NVIDIA) à la volée via une analyse de l'Arbre Syntaxique Abstrait (AST).

En intégrant un orchestrateur basé sur des graphes de dépendance acycliques (DAG), du *Zero-Copy DLPack*, et du *Memory Pooling*, GPUsync permet d'exécuter des algorithmes complexes de manière asynchrone, distribuée, et en *Out-Of-Core* (Streaming). Il masque totalement la latence matérielle derrière une interface Python idiomatique.

## Fonctionnalités Principales

- **Transpilation JIT Multi-Backend (AST)** : Écrivez vos kernels en syntaxe Python pure (avec commentaires et docstrings gérés). GPUsync les transpile en CUDA C++, OpenCL C99, LLVM JIT CPU et NVIDIA PTX.
- **Traitement Out-Of-Core Monoligne** : Découpez, streamez et traitez automatiquement des jeux de données géants (dépassant la VRAM) en 1 seule ligne de code via `kernel.map_out_of_core(dataset, scheduler)`.
- **Streaming Disque Continu par Générateurs** : Passez directement une fonction génératrice Python (`yield`) lisant un fichier SSD NVMe/HDF5 dans `map_out_of_core` sans jamais charger l'intégralité du dataset en RAM.
- **Zero-Copy DLPack Interopérabilité** : Échangez des pointeurs mémoire VRAM directement avec PyTorch et CuPy via `AsyncBuffer.__dlpack__` et `from_dlpack`.
- **Moteur CPU LLVM Natif** : Exécutez vos kernels sur le processeur hôte (CPU) avec compilation JIT native (`backend="llvm"` via `llvmlite`).
- **Ordonnancement Multi-GPU (DAG & Swarm)** : Répartition automatique de la charge de travail entre plusieurs cartes graphiques hétérogènes.

## Démarrage Rapide (1 ligne Out-Of-Core)

```python
from gpusync import gpusync_kernel, Scheduler, list_gpus
import numpy as np

@gpusync_kernel
def process_signal(ctx: int, data: list) -> tuple:
    # Code exécuté en parallèle sur des millions de threads GPU
    return data[ctx] * 2.5 + 1.0

scheduler = Scheduler(gpu=list_gpus()[0])
scheduler.initialize()

# 1. Traitement automatique d'un dataset géant NumPy en Out-Of-Core (1 seule ligne !)
resultat = process_signal.map_out_of_core(large_dataset_50GB, scheduler=scheduler)

# 2. Streaming direct depuis un fichier sur disque SSD NVMe (Générateur yield)
def stream_fichier_disque(filepath):
    with open(filepath, "rb") as f:
        while True:
            chunk = np.fromfile(f, dtype=np.float32, count=1_000_000)
            if len(chunk) == 0: break
            yield chunk

resultat_disque = process_signal.map_out_of_core(stream_fichier_disque("big.bin"), scheduler=scheduler)
```

## Documentation

La documentation complète est disponible dans le dossier `docs/` :

- [Sommaire du Wiki](https://github.com/GalTechDev/GPUsync/blob/main/docs/Home.md)
- [1. Architecture du Compilateur AST](https://github.com/GalTechDev/GPUsync/blob/main/docs/1.-Architecture-du-Compilateur-AST.md)
- [2. Gestion de la Mémoire et des Buffers (VRAM)](https://github.com/GalTechDev/GPUsync/blob/main/docs/2.-Gestion-de-la-Mémoire-et-Buffers.md)
- [3. Sélection des Périphériques et Schedulers](https://github.com/GalTechDev/GPUsync/blob/main/docs/3.-Sélection-des-Périphériques-et-Schedulers.md)
- [4. Guide de Démarrage Développeur](https://github.com/GalTechDev/GPUsync/blob/main/docs/4.-Guide-de-Démarrage-Développeur.md)

## Remerciements

Une mention toute particulière est adressée à **[`ThePhoenix78`](https://github.com/ThePhoenix78)** pour son aide inestimable sur l'architecture mathématique et l'implémentation de l'ordonnanceur de Graphe Orienté Acyclique (DAG) Multi-GPU.
