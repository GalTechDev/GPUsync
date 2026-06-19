# GPUsync

**GPUsync** est un framework open-source novateur d'accélération matérielle Multi-GPU. Il offre une surcouche asynchrone intelligente capable de transpiler du code Python vers du C++ (CUDA/OpenCL) à la volée via une analyse de l'Arbre Syntaxique Abstrait (AST).

En intégrant un orchestrateur basé sur des graphes de dépendance acycliques (DAG) et du *Memory Pooling*, GPUsync permet d'exécuter des algorithmes complexes de manière asynchrone, distribuée, et en *Out-Of-Core* (Streaming). Il masque totalement la latence matérielle derrière une interface Python idiomatique et purement synchrone.

## Fonctionnalités Principales

- **Transpilation JIT via AST** : Écrivez des kernels GPU directement en syntaxe Python pure. Le framework se charge de les compiler en CUDA C++ ou OpenCL.
- **Exécution Asynchrone (Background Loop)** : L'utilisation du PCIe est maximisée grâce à une boucle d'exécution asynchrone tournant en tâche de fond.
- **Ordonnancement Multi-GPU (DAG)** : Répartition automatique de la charge de travail entre plusieurs cartes graphiques via un algorithme de Tri Topologique.
- **Traitement Out-Of-Core (Streaming)** : Dépassez les limites physiques de la VRAM (*Memory Wall*) grâce au *True Streaming* et au *Memory Pooling*.

## Documentation

Pour plus d'informations sur l'architecture du projet et la manière de l'utiliser ou d'y contribuer, veuillez consulter la documentation :

- [Guide Développeur](docs/GUIDE_DEVELOPPEUR.md) : Instructions pour contribuer au projet, conventions de code et détails d'implémentation.
- [Rapport de Recherche GPUsync](docs/RAPPORT_GPUSYNC.md) : Document complet détaillant l'architecture logicielle, les benchmarks (DTW, K-Means, Produit Matriciel) et la théorie sous-jacente.

## Exemples

Le dossier `examples/` contient plusieurs implémentations de référence :
- **DTW** (Dynamic Time Warping)
- **K-Means** Clustering
- **Multiplication Matricielle** (Tiling 2D, Tensor Cores WMMA)

## Remerciements
Une mention toute particulière est adressée à **ThePhoenix78** pour son aide inestimable sur l'architecture mathématique et l'implémentation de l'ordonnanceur de Graphe Orienté Acyclique (DAG) Multi-GPU.
