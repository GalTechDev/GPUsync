# Bienvenue sur le Wiki de GPUsync

GPUsync est un framework Python novateur conçu pour le calcul intensif, le calcul scientifique (HPC), et l'accélération matérielle multi-GPU. 

Il combine une **Compilation Juste-à-Temps (JIT)** intelligente (Python -> CUDA/OpenCL) avec un orchestrateur de dépendances asynchrones (DAG) et un système de streaming de données **Out-Of-Core**, masquant ainsi la latence et la complexité des cartes graphiques derrière une interface simple et idiomatique.

---

## Sommaire du Wiki

### [1. Architecture du Compilateur AST](https://github.com/GalTechDev/GPUsync/wiki/1.-Architecture-du-Compilateur-AST)
Découvrez comment vos fonctions Python sont décodées, inspectées, et transpilées à la volée en code machine CUDA C++ et OpenCL C99, ainsi que les mécanismes de routage dynamique de dimensions et les Tensor Cores.

### [2. Gestion de la Mémoire et des Buffers (VRAM)](https://github.com/GalTechDev/GPUsync/wiki/2.-Gestion-de-la-M%C3%A9moire-et-Buffers)
Comprenez le fonctionnement de la classe `AsyncBuffer`, les transferts asynchrones hôte-périphérique, l'affinité de mémoire (*Memory Pinning*), et les transferts automatiques inter-GPU au sein des pipelines.

### [3. Sélection des Périphériques et Schedulers](https://github.com/GalTechDev/GPUsync/wiki/3.-S%C3%A9lection-des-P%C3%A9riph%C3%A9riques-et-Schedulers)
Apprenez à lister vos cartes graphiques avec `list_gpus()`, à cibler un GPU ou un backend précis, et découvrez les différences et l'usage coopératif entre `SwarmScheduler` et `MultiDeviceScheduler`.

### [4. Guide de Démarrage Développeur](https://github.com/GalTechDev/GPUsync/wiki/4.-Guide-de-D%C3%A9marrage-D%C3%A9veloppeur)
Un guide étape par étape complet pour écrire vos premiers kernels avec GPUsync, comprendre le concept de `ctx` (contexte de grille), utiliser la mémoire SRAM locale, et orchestrer des calculs lourds.
