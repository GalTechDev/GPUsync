# GPUsync

**GPUsync** est un framework open-source novateur d'accélération matérielle Multi-GPU. Il offre une surcouche asynchrone intelligente capable de transpiler du code Python vers du C++ (CUDA) et du C99 (OpenCL) à la volée via une analyse de l'Arbre Syntaxique Abstrait (AST).

En intégrant un orchestrateur basé sur des graphes de dépendance acycliques (DAG) et du *Memory Pooling*, GPUsync permet d'exécuter des algorithmes complexes de manière asynchrone, distribuée, et en *Out-Of-Core* (Streaming). Il masque totalement la latence matérielle derrière une interface Python idiomatique.

## Fonctionnalités Principales

- **Transpilation JIT via AST** : Écrivez des kernels GPU directement en syntaxe Python pure. Le framework se charge de les transpiler et de les exécuter en CUDA C++ ou OpenCL C99.
- **Exécution Asynchrone (Background Loop)** : L'utilisation du PCIe est maximisée grâce à une boucle d'exécution asynchrone tournant en tâche de fond.
- **Ordonnancement Multi-GPU (DAG & Swarm)** : Répartition automatique de la charge de travail entre plusieurs cartes graphiques hétérogènes (NVIDIA CUDA, Intel OpenCL, AMD OpenCL).
- **Traitement Out-Of-Core (Streaming)** : Dépassez les limites physiques de la VRAM (*Memory Wall*) grâce au *True Streaming* et au *Memory Pooling*.

## Documentation

La documentation complète est disponible sur le [Wiki GitHub](https://github.com/GalTechDev/GPUsync/wiki) ainsi que dans le dossier `docs/` :

- [Sommaire du Wiki](https://github.com/GalTechDev/GPUsync/wiki/Home) ([docs/Home.md](docs/Home.md))
- [1. Architecture du Compilateur AST](https://github.com/GalTechDev/GPUsync/wiki/1.-Architecture-du-Compilateur-AST) ([docs/1.-Architecture-du-Compilateur-AST.md](docs/1.-Architecture-du-Compilateur-AST.md))
- [2. Gestion de la Mémoire et des Buffers (VRAM)](https://github.com/GalTechDev/GPUsync/wiki/2.-Gestion-de-la-M%C3%A9moire-et-Buffers) ([docs/2.-Gestion-de-la-Mémoire-et-Buffers.md](docs/2.-Gestion-de-la-Mémoire-et-Buffers.md))
- [3. Sélection des Périphériques et Schedulers](https://github.com/GalTechDev/GPUsync/wiki/3.-S%C3%A9lection-des-P%C3%A9riph%C3%A9riques-et-Schedulers) ([docs/3.-Sélection-des-Périphériques-et-Schedulers.md](docs/3.-Sélection-des-Périphériques-et-Schedulers.md))
- [4. Guide de Démarrage Développeur](https://github.com/GalTechDev/GPUsync/wiki/4.-Guide-de-D%C3%A9marrage-D%C3%A9veloppeur) ([docs/4.-Guide-de-Démarrage-Développeur.md](docs/4.-Guide-de-Démarrage-Développeur.md))

## Exemples

Le dossier `examples/` contient des implémentations de référence prêtes à l'emploi :
- **DTW** (`examples/dtw/`) : Dynamic Time Warping.
- **K-Means** (`examples/kmeans/`) : K-Means standard, Spherical K-Means et Mini-Batch K-Means.
- **Multiplication Matricielle** (`examples/matrix/`) : Tiling 2D, Tensor Cores (WMMA), Double Buffering.
- **KNN-DTW** (`examples/knn_dtw/`) : Classification K-Nearest Neighbors basée sur DTW.
- **K-Shape** (`examples/kshape/`) : Clustering de séries temporelles par cross-corrélation (SBD).
- **LOESS** (`examples/local_regression/`) : Régression locale pondérée.
- **STOMP / Matrix Profile** (`examples/matrix_profile/`) : Calcul du Matrix Profile pour la détection de motifs et d'anomalies.
- **Soft-DTW** (`examples/gr_soft_dtw/`) : Implémentation différentiable de Soft-DTW.

## Remerciements

Une mention toute particulière est adressée à **ThePhoenix78** pour son aide inestimable sur l'architecture mathématique et l'implémentation de l'ordonnanceur de Graphe Orienté Acyclique (DAG) Multi-GPU.
