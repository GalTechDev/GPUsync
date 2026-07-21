# Bienvenue sur le Wiki de GPUsync

GPUsync est un framework Python novateur conçu pour le calcul intensif, le calcul scientifique (HPC), et l'accélération matérielle multi-GPU. 

Il combine une **Compilation Juste-à-Temps (JIT)** intelligente (Python AST -> CUDA / OpenCL / LLVM CPU & NVPTX GPU) avec un orchestrateur de dépendances asynchrones (DAG), du **Zero-Copy DLPack**, et un système de streaming de données **Out-Of-Core monoligne et générateurs**, masquant ainsi la latence et la complexité des cartes graphiques derrière une interface simple et idiomatique.

---

## Sommaire du Wiki

### [1. Architecture du Compilateur AST](https://github.com/GalTechDev/GPUsync/wiki/1.-Architecture-du-Compilateur-AST)
Découvrez comment vos fonctions Python sont décodées, inspectées, et transpilées à la volée en code machine CUDA C++, OpenCL C99, LLVM JIT CPU, et assembleur GPU NVIDIA PTX (`llvm_cuda.py`).

### [2. Gestion de la Mémoire et des Buffers (VRAM)](https://github.com/GalTechDev/GPUsync/wiki/2.-Gestion-de-la-M%C3%A9moire-et-Buffers)
Comprenez le fonctionnement de la classe `AsyncBuffer`, l'API Out-Of-Core monoligne `kernel.map_out_of_core()`, le streaming direct par générateurs (`yield` depuis NVMe SSD), et le partage de mémoire **Zero-Copy DLPack** (`__dlpack__`, `from_dlpack`) avec PyTorch et CuPy.

### [3. Sélection des Périphériques et Schedulers](https://github.com/GalTechDev/GPUsync/wiki/3.-S%C3%A9lection-des-P%C3%A9riph%C3%A9riques-et-Schedulers)
Apprenez à lister vos cartes graphiques avec `list_gpus()`, à utiliser le moteur JIT CPU `LLVMDeviceManager` (`backend="llvm"`), et découvrez les différences et l'usage coopératif entre `SwarmScheduler` et `MultiDeviceScheduler`.

### [4. Guide de Démarrage Développeur](https://github.com/GalTechDev/GPUsync/wiki/4.-Guide-de-D%C3%A9marrage-D%C3%A9veloppeur)
Un guide étape par étape complet pour écrire vos premiers kernels avec GPUsync, utiliser l'API `map_out_of_core` en 1 ligne, intégrer des générateurs disque, et exécuter des benchmarks.
