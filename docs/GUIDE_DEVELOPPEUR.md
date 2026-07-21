# Guide du Développeur GPUsync

Bienvenue dans l'équipe GPUsync ! 
Ce document est un guide pas-à-pas détaillé conçu pour vous accompagner dans l'écriture de vos premiers scripts avec la bibliothèque GPUsync. GPUsync vous permet de coder en Python tout en exécutant vos calculs à la vitesse de la lumière sur les GPUs de votre machine (NVIDIA CUDA, AMD OpenCL, Intel OpenCL), sans vous soucier des limites de mémoire (VRAM).

Ce guide couvre la structure des instructions, le concept de contexte (`ctx`), l'utilisation de la mémoire partagée SRAM, les Tensor Cores, et la mise en place de pipelines asynchrones ou Out-Of-Core.

---

## 1. Philosophie : Que se passe-t-il sous le capot ?

Contrairement à des bibliothèques comme NumPy où Python appelle des fonctions C++ pré-compilées, GPUsync procède à une **Compilation Juste-à-Temps (JIT)**.
Lorsque vous écrivez une fonction avec `@gpusync_kernel`, le code ne s'exécute jamais en Python. Le framework lit votre code source, extrait son Arbre Syntaxique Abstrait (AST), et le **transpile** instantanément en code natif C++ (CUDA) ou C99 (OpenCL) avant de l'injecter dans la carte graphique.

Cela implique deux règles fondamentales :
1. Vous ne pouvez utiliser dans un kernel que des opérations mathématiques de base, des boucles `for/while`, et des conditions `if/else`. Les appels à des bibliothèques externes (comme `import math` ou `numpy` dans le kernel) ne sont pas autorisés.
2. Les arguments passés à vos kernels doivent être explicitement typés (`int`, `float`, `list`).

---

## 2. Structure d'un Projet Standard

Un projet GPUsync suit toujours le même cycle de vie : **Initialisation -> Transfert RAM/VRAM -> Calcul GPU -> Retour RAM**.

### A. Initialisation du Scheduler

Le `Scheduler` gère la détection de vos processeurs de calcul et la sélection du matériel :

```python
import numpy as np
from gpusync import list_gpus, Scheduler, AsyncBuffer

def main():
    # Découverte automatique des GPUs disponibles
    gpus = list_gpus()
    if not gpus:
        raise RuntimeError("Aucun GPU disponible.")
        
    # Initialisation sur le premier GPU détecté
    scheduler = Scheduler(gpu=gpus[0])
    scheduler.initialize()
```

### B. Allocation des Données (Hôte vers Périphérique)

Vous devez transférer vos tableaux NumPy (RAM CPU) vers la mémoire de la carte graphique (VRAM) avec `AsyncBuffer.to_device` :

```python
    # Données CPU (RAM)
    a_cpu = np.ones(1_000_000, dtype=np.float32)
    b_cpu = np.ones(1_000_000, dtype=np.float32)
    
    # Transfert vers VRAM
    a_gpu = AsyncBuffer.to_device(scheduler, a_cpu)
    b_gpu = AsyncBuffer.to_device(scheduler, b_cpu)
```

---

## 3. Écrire son Premier Kernel (La Magie du `ctx`)

Sur un processeur classique (CPU), on utilise souvent une boucle `for i in range(1_000_000)` pour traiter les éléments un par un.
Sur un GPU, l'approche est différente : on lance des milliers ou des millions de petits processeurs (threads) virtuels en parallèle, et chacun traite une case précise.

L'argument **`ctx` (Context)** représente l'identifiant unique (index absolu) du thread GPU qui exécute la fonction :

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def addition_kernel(ctx: int, a: list, b: list) -> tuple:
    # ctx varie de 0 à (global_size - 1)
    # Ce thread spécifique lit la case a[ctx], la case b[ctx], et retourne la somme.
    return a[ctx] + b[ctx]
```

### Règles d'écriture des Kernels :
- **`ctx`** : Premier paramètre obligatoire, typé `int`.
- **Tableaux** : Les arguments représentant des `AsyncBuffer` doivent être typés `list`.
- **Scalaires** : Les constantes doivent être typées `int` ou `float`.
- **Return** : Utilisez `-> tuple` si vous retournez une ou plusieurs valeurs de même dimension que la grille d'exécution. Les tableaux de sortie sont pré-alloués automatiquement par GPUsync.

### Appel du Kernel :

```python
    # global_size=(1_000_000,) indique au GPU de lancer 1 million de threads parallèles
    # Le paramètre scheduler est optionnel si a_gpu possède déjà son affinité matérielle
    resultat_cpu = addition_kernel(a_gpu, b_gpu, global_size=(1_000_000,), scheduler=scheduler)
    
    print(resultat_cpu) # Tableau NumPy contenant les résultats rapatriés
```

---

## 4. L'Exécution Multi-GPU avec `@gpusync_pipeline`

Si vous chaînez plusieurs calculs, vous ne voulez pas rapatrier la donnée sur la RAM CPU entre chaque étape. De plus, si deux étapes sont indépendantes (ex: `A + B` et `C * D`), vous souhaitez qu'elles s'exécutent en parallèle sur des GPUs différents.

Utilisez le **Planificateur DAG (Graphe Orienté Acyclique)** intégré :

```python
from gpusync import MultiDeviceScheduler, gpusync_pipeline

# 1. On donne au planificateur accès à nos cartes graphiques
multi_sched = MultiDeviceScheduler([scheduler])

# 2. Définition du pipeline
@gpusync_pipeline(multi_sched)
def mon_gros_algorithme(a, b, c):
    # Les étapes sans dépendance mutuelle s'exécutent en parallèle sur le cluster de GPUs
    res1 = addition_kernel(a, b, global_size=(1_000_000,))
    res2 = addition_kernel(b, c, global_size=(1_000_000,))
    
    # "final" attendra automatiquement la fin des calculs de res1 et res2
    final = multiplication_kernel(res1, res2, global_size=(1_000_000,))
    
    return final # Seul le résultat final est téléchargé sur le CPU
```

---

## 5. Mémoire Partagée (SRAM) et Opérations Atomiques

Pour optimiser les réductions ou histogrammes, GPUsync donne accès à la **Shared Memory** (SRAM), un cache ultra-rapide partagé entre tous les threads d'un même bloc :

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def histogram_kernel(ctx: int, X: list, hist_out: list):
    local_idx = get_local_id(0) # ID du thread dans son bloc (0 à 1023)
    
    # 1. Déclaration de la mémoire partagée locale au bloc de threads
    local_hist = local_memory((256,), "int")
    
    # 2. Initialisation
    if local_idx < 256:
        local_hist[local_idx] = 0
        
    # 3. Barrière de synchronisation
    barrier(CLK_LOCAL_MEM_FENCE)
    
    # 4. Accumulation avec verrouillage atomique en SRAM
    val = X[ctx]
    atomic_add(local_hist[val], 1)
    
    barrier(CLK_LOCAL_MEM_FENCE)
    
    # 5. Report en VRAM globale par bloc
    if local_idx < 256:
        atomic_add(hist_out[local_idx], local_hist[local_idx])
```

---

## 6. Memory Coalescing (SoA vs AoS)

Le matériel GPU lit la mémoire par blocs de 32 ou 64 octets consécutifs. Pour des performances optimales, préférez toujours les structures contiguës en mémoire (*Structure of Arrays* - **SoA**) aux structures imbriquées (*Array of Structures* - **AoS**) :

- **AoS (Mauvais)** : Accéder à un tableau 3D `(N, 3)` via `X[ctx * 3 + d]`.
- **SoA (Excellent)** : Transposer le tableau en Python `(3, N)` via `np.ascontiguousarray(X_cpu.T)` et y accéder via `X[d * N + ctx]`. Les threads d'un même Warp lisent des adresses adjacentes, permettant au GPU de fusionner les requêtes mémoire.

---

## 7. Les Tensor Cores et le WMMA (Multiplication Matricielle)

GPUsync expose les API bas niveau pour activer les **Tensor Cores** via les instructions WMMA (Warp Matrix Multiply and Accumulate) sur les architectures NVIDIA compatibles.

Ici, un groupe de 32 threads (Warp) coopère pour charger et multiplier une sous-matrice (Tile) de $16 \times 16$ en un nombre réduit de cycles d'horloge :

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def wmma_matmul(ctx: int, A: list, B: list, C: list, M: int, N: int, K: int):
    # 1. Déclaration des fragments matériels
    frag_a = wmma_matrix_a()
    frag_b = wmma_matrix_b()
    frag_c = wmma_accumulator()
    
    # 2. Initialisation de l'accumulateur (C) à zéro
    wmma_fill(frag_c, 0.0)
    
    # 3. Boucle sur la profondeur (K) de la matrice par pas de 16
    for k in range(0, K, 16):
        wmma_load_matrix_sync(frag_a, A, k)
        wmma_load_matrix_sync(frag_b, B, k)
        wmma_mma_sync(frag_c, frag_a, frag_b, frag_c)
        
    # 4. Écriture du résultat final vers la VRAM
    wmma_store_matrix_sync(C, frag_c)
```

Pour consulter l'implémentation complète avec Tiling 2D, référez-vous au fichier [examples/matrix/matrix_cuda_wmma_2D_tiling.py](https://github.com/GalTechDev/GPUsync/blob/main/examples/matrix/matrix_cuda_wmma_2D_tiling.py).

---

## 8. Le Streaming Asynchrone Out-Of-Core (`asyncio`)

Pour traiter de grands volumes de données sans dépasser la capacité VRAM, GPUsync s'interface avec la boucle d'événements `asyncio` de Python.

En associant un **Pool de Buffers** (`asyncio.Queue`) et du Double Buffering, le transfert PCIe et le calcul GPU s'exécutent en parallèle :

```python
import asyncio
import numpy as np
from gpusync import AsyncBuffer

async def streaming_pipeline(X_disk, scheduler):
    # 1. Pré-allocation de 2 buffers VRAM vides pour le Double Buffering PCIe
    pool = asyncio.Queue(maxsize=2)
    pool.put_nowait(AsyncBuffer.empty_like(scheduler, np.zeros(CHUNK_SIZE, dtype=np.float32)))
    pool.put_nowait(AsyncBuffer.empty_like(scheduler, np.zeros(CHUNK_SIZE, dtype=np.float32)))
    
    active_tasks = set()
    
    for i in range(num_chunks):
        sub = X_disk[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
        
        # Récupération d'un buffer VRAM disponible
        gpu_X = await pool.get()
        
        # Réécriture asynchrone sans réallocation
        await gpu_X.overwrite(sub)
        task = asyncio.create_task(mon_kernel(gpu_X, scheduler=scheduler))
        
        # Restitution du buffer au pool après exécution
        task.add_done_callback(lambda t: pool.put_nowait(gpu_X))
        active_tasks.add(task)
        
    await asyncio.gather(*active_tasks)
```

---

## 9. Support de l'Architecture ABA (Asynchronous Blackboard Architecture)

GPUsync supporte l'architecture ABA pour les traitements asynchrones multi-agents, où des kernels modifient un espace mémoire partagé pendant qu'un superviseur CPU observe l'état du système.

### Le Swarm Mode (Multi-Streams)
Vous pouvez injecter des Streams indépendants pour exécuter des calculs en parallèle sur le GPU :
```python
import cupy as cp
stream_worker1 = cp.cuda.Stream(non_blocking=True)
await mon_kernel(..., stream=stream_worker1)
```

### Le Lancement Non-Bloquant (Fire-and-Forget)
L'argument `fire_and_forget=True` demande à GPUsync d'envoyer l'ordre d'exécution au GPU et de retourner immédiatement le buffer `AsyncBuffer` VRAM sans attendre la fin du calcul ni rapatrier les données sur le CPU :
```python
buf_res = await mon_kernel(..., fire_and_forget=True)
```

### Le Polling Asynchrone (Dirty Read)
La méthode `dirty_read()` effectue un transfert DMA direct de la VRAM vers la RAM hôte sans geler les Streams de travail GPU en cours :
```python
canvas_snapshot = await mon_async_buffer.dirty_read()
```

---

## 10. Résolution des Problèmes Communs

1. **Memory Error / CUDA Out Of Memory** :
   Réduisez la taille des blocs ou utilisez l'approche Out-Of-Core (Section 8) avec `asyncio.Queue`.
   
2. **Erreur de Transpilation (IndentationError, AST Parse)** :
   Le code dans un `@gpusync_kernel` doit utiliser des opérations scalaires ou indexées (`A[ctx] + B[ctx]`). Les opérations vectorielles de haut niveau NumPy ne sont pas utilisables au sein d'un kernel JIT.

3. **Différences entre Backends CUDA et OpenCL** :
   Les buffers créés par un scheduler CUDA ne doivent pas être passés directement à un kernel OpenCL en dehors d'un `@gpusync_pipeline` (qui gère automatiquement la ré-allocation et la copie inter-GPU).

---

Consultez le dossier [examples/](https://github.com/GalTechDev/GPUsync/tree/main/examples) pour découvrir toutes nos implémentations de référence (DTW, K-Means, STOMP, K-Shape, LOESS, Soft-DTW).
