# Guide du Développeur GPUsync

Bienvenue dans l'équipe GPUsync ! 
Ce document est un guide pas-à-pas très détaillé conçu pour vous accompagner dans l'écriture de vos premiers scripts avec la bibliothèque. GPUsync vous permet de coder en Python tout en exécutant vos calculs à la vitesse de la lumière sur les GPU de votre machine (NVIDIA, AMD, Intel), sans vous soucier des limites de mémoire (VRAM).

Ce guide couvre la structure des instructions, le concept de contexte (`ctx`), et la mise en place d'un pipeline complet.

---

## 1. Philosophie : Que se passe-t-il sous le capot ?
Contrairement à des bibliothèques comme NumPy où Python appelle des fonctions C++ pré-compilées, GPUsync procède à une **Compilation Juste-à-Temps (JIT)**.
Lorsque vous écrivez une fonction avec `@gpusync_kernel`, le code ne s'exécute jamais en Python. Le framework "lit" votre code source, extrait son Arbre Syntaxique Abstrait (AST), et le **traduit** (transpile) instantanément en code natif C++ (CUDA ou OpenCL) avant de l'injecter dans la carte graphique.

Cela signifie deux choses cruciales :
1. Vous ne pouvez utiliser dans un kernel que des opérations mathématiques de base, des boucles `for/while`, et des conditions `if`. Pas d'appels à des bibliothèques externes (comme `import math` ou `numpy`).
2. Les arguments que vous passez à vos kernels doivent être explicitement typés (`int`, `float`, `list`).

---

## 2. Structure d'un Projet Standard

Un projet GPUsync suit toujours le même cycle de vie : **Initialisation -> Transfert RAM/VRAM -> Calcul GPU -> Retour RAM**.

### A. Initialisation du Scheduler
Le `Scheduler` est le chef d'orchestre. Il détecte vos cartes graphiques.

```python
import numpy as np
from gpusync import Scheduler, AsyncBuffer

def main():
    # Vous pouvez forcer backend="cuda" pour NVIDIA, ou "opencl" pour être universel
    scheduler = Scheduler(backend="cuda")
    scheduler.initialize() # Cette opération est automatique et synchrone
```

### B. Allocation des Données
Le GPU a sa propre mémoire (VRAM). Vous devez donc prendre vos tableaux NumPy (RAM) et les transférer sur la carte graphique avec `AsyncBuffer.to_device`.

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
Sur un GPU, l'approche est radicalement différente : on lance **1 million de petits processeurs (threads)** virtuels en même temps, et chacun ne traite qu'une seule case !

C'est là qu'intervient l'argument magique **`ctx` (Context)**. 
`ctx` représente "l'identifiant unique" du thread GPU qui est en train d'exécuter la fonction.

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def addition_kernel(ctx: int, a: list, b: list) -> tuple:
    # ctx varie de 0 à 999 999 (si on a lancé 1 million de threads)
    # Ce thread spécifique va lire la case a[ctx], la case b[ctx], et retourner la somme.
    return a[ctx] + b[ctx]
```

### Règles d'écriture des Kernels :
- **Toujours typer `ctx` en `int`** en premier paramètre.
- **Tableaux** : Les arguments représentant des `AsyncBuffer` doivent être typés `list`.
- **Scalaires** : Les constantes doivent être typées `int` ou `float`.
- **Return** : Utilisez `-> tuple` si vous retournez une ou plusieurs valeurs de même dimension que la grille d'exécution. Les tableaux de sortie sont pré-alloués dynamiquement par GPUsync !

### Appel du Kernel

```python
    # L'appel est 100% synchrone en apparence grâce à l'orchestrateur de GPUsync
    # global_size=(1_000_000,) indique au GPU de lancer 1 million de threads parallèles
    resultat_cpu = addition_kernel(a_gpu, b_gpu, global_size=(1_000_000,), scheduler=scheduler)
    
    print(resultat_cpu) # C'est un tableau NumPy contenant les résultats !
```

---

## 4. Le Graal : L'Exécution Multi-GPU avec `@gpusync_pipeline`

Si vous avez plusieurs opérations qui se suivent, vous ne voulez pas rapatrier la donnée sur la RAM entre chaque étape. Pire, si vous avez deux opérations indépendantes (ex: `A + B` et `C * D`), vous voudriez qu'elles s'exécutent en même temps, idéalement sur des GPUs différents !

Utilisez le **Planificateur DAG (Graphe Orienté Acyclique)** intégré.

```python
from gpusync import MultiDeviceScheduler, gpusync_pipeline

# 1. On donne au planificateur accès à nos cartes graphiques
multi_sched = MultiDeviceScheduler([scheduler]) # Ajoutez d'autres schedulers si vous avez plusieurs GPUs

# 2. On définit notre algorithme complet
@gpusync_pipeline(multi_sched)
def mon_gros_algorithme(a, b, c):
    # GPUsync lit ce code, voit que res1 et res2 ne dépendent pas l'un de l'autre
    # Il va donc les lancer en parallèle !
    res1 = addition_kernel(a, b, global_size=(1_000_000,))
    res2 = addition_kernel(b, c, global_size=(1_000_000,))
    
    # Il voit que "final" dépend de res1 et res2. Il attendra que les deux terminent.
    # Les variables "res1" et "res2" ne quittent JAMAIS la VRAM, c'est ultra rapide !
    final = multiplication_kernel(res1, res2, global_size=(1_000_000,))
    
    return final # Seul le résultat final est téléchargé sur le CPU
```

---

---

## 5. Mémoire Partagée (SRAM) et Opérations Atomiques

Pour atteindre des performances extrêmes (comme dans notre implémentation du K-Means), la mémoire globale (VRAM) devient un goulot d'étranglement. GPUsync vous donne accès à la **Shared Memory** (SRAM), un cache ultra-rapide partagé entre tous les threads d'un même bloc.

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def histogram_kernel(ctx: int, X: list, hist_out: list):
    local_idx = get_local_id(0) # ID du thread dans son bloc (0 à 1023)
    
    # 1. Déclaration de la mémoire partagée (très petite mais très rapide)
    local_hist = local_memory((256,), "int")
    
    # 2. Initialisation par les threads du bloc
    if local_idx < 256:
        local_hist[local_idx] = 0
        
    # 3. Barrière de synchronisation : on attend que l'initialisation soit finie
    barrier(CLK_LOCAL_MEM_FENCE)
    
    # 4. Accumulation avec verrouillage atomique en SRAM (ultra rapide)
    val = X[ctx]
    atomic_add(local_hist[val], 1)
    
    barrier(CLK_LOCAL_MEM_FENCE)
    
    # 5. Report en VRAM globale par un seul thread pour éviter la contention
    if local_idx < 256:
        atomic_add(hist_out[local_idx], local_hist[local_idx])
```
*Règle d'or : Ne lancez jamais des milliers de threads en écriture sur la même case de la VRAM globale. Utilisez la SRAM locale pour aggréger les résultats par bloc !*

---

## 6. Memory Coalescing (SoA vs AoS)

Le matériel GPU lit la mémoire par blocs de 32 ou 64 octets consécutifs. Si vos threads lisent des cases mémoire éparpillées (*Strided Access*), la bande passante s'effondre.

**Mauvaise Pratique (Array of Structures - AoS)** :
Supposons un tableau Numpy de points 3D de forme `(N, 3)`.
Si le thread lit `X[ctx * 3 + d]`, les 32 threads de son *Warp* liront des cases distantes. Le GPU fera 32 requêtes mémoire distinctes.

**Bonne Pratique (Structure of Arrays - SoA)** :
Avant d'envoyer les données au GPU, transposez-les en Python :
```python
X_soa = np.ascontiguousarray(X_cpu.T).ravel()
```
Le tableau a maintenant la forme `(3, N)`.
Dans le kernel, lisez `X[d * N + ctx]`.
Les 32 threads du *Warp* (ctx de 0 à 31) liront `X[0]`, `X[1]`, ..., `X[31]`. Le GPU chargera ces données en **1 seule requête mémoire**. *Gain de performance : jusqu'à 32x !*

---

## 7. Les Tensor Cores et le WMMA (Multiplication Matricielle)

Pour l'IA et le calcul matriciel, les coeurs ALU standards ne suffisent plus. GPUsync expose dynamiquement les API de très bas niveau de NVIDIA pour activer les **Tensor Cores** via les instructions WMMA (Warp Matrix Multiply and Accumulate).

Contrairement à l'exécution scalaire où un thread calcule une valeur, ici le **Warp** (groupe de 32 threads synchronisés matériellement) coopère pour charger et multiplier une sous-matrice (Tile) de $16 \times 16$ en **un seul cycle d'horloge** (en demi-précision FP16, avec accumulation en FP32 pour la stabilité).

```python
from gpusync import gpusync_kernel

@gpusync_kernel
def wmma_matmul(ctx: int, A: list, B: list, C: list, M: int, N: int, K: int):
    # Les pointeurs mémoires doivent être alignés. Le Warp entier appelle ces fonctions :
    
    # 1. Déclaration des fragments matériels (Registres Tensor Cores internes)
    frag_a = wmma_matrix_a()
    frag_b = wmma_matrix_b()
    frag_c = wmma_accumulator()
    
    # 2. Initialisation de l'accumulateur (C) à zéro
    wmma_fill(frag_c, 0.0)
    
    # 3. Boucle sur la profondeur (K) de la matrice
    for k in range(0, K, 16):
        # Chargement coopératif d'une tuile de 16x16 depuis A et B vers les registres Tensor
        wmma_load_matrix_sync(frag_a, A, k)
        wmma_load_matrix_sync(frag_b, B, k)
        
        # 4. MULTIPLICATION MATÉRIELLE ULTRA-RAPIDE (1 cycle d'horloge !)
        wmma_mma_sync(frag_c, frag_a, frag_b, frag_c)
        
    # 5. Écriture du résultat final vers la VRAM
    wmma_store_matrix_sync(C, frag_c)
```
Pour atteindre le pic de PétaFLOPS théorique de votre puce, vous devez associer cette technique au **Double Buffering asynchrone** en SRAM pour masquer le temps de chargement `wmma_load`. Pour voir cette implémentation ultime, consultez le fichier `examples/benchmark/matrix/matrix_gpu.py`.

---

## 8. Le Streaming Asynchrone Out-Of-Core (`asyncio`)

Pour traiter des téraoctets de données (ex: 50 Millions de points) sans faire exploser vos 64 Go de RAM, GPUsync couple le GPU à la boucle d'événements asynchrone de Python.

Au lieu de charger tout le dataset, vous créez un **Démon d'arrière-plan** (`BgLoop`) et un **Pool de Buffers** (`asyncio.Queue`) :

```python
import asyncio
from gpusync import AsyncBuffer

async def streaming_pipeline(X_disk, scheduler):
    # 1. On réserve 2 buffers VRAM qui tourneront en boucle (Double Buffering PCIe)
    pool = asyncio.Queue(maxsize=2)
    pool.put_nowait(AsyncBuffer(scheduler, shape=(CHUNK_SIZE,), dtype=np.float32))
    pool.put_nowait(AsyncBuffer(scheduler, shape=(CHUNK_SIZE,), dtype=np.float32))
    
    # 2. On lance une tâche en fond pour ne pas bloquer Python
    active_tasks = set()
    
    for i in range(num_chunks):
        # On lit un petit morceau depuis le disque (Zero-Copy si np.memmap)
        sub = X_disk[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
        
        # On récupère un buffer VRAM libre du pool
        gpu_X = await pool.get()
        
        # On écrit par-dessus et on lance le kernel en asynchrone
        await gpu_X.overwrite(sub)
        task = asyncio.create_task(mon_kernel(gpu_X, scheduler=scheduler))
        
        # Quand le kernel termine, on remet le buffer dans le pool
        task.add_done_callback(lambda t: pool.put_nowait(gpu_X))
        active_tasks.add(task)
        
    await asyncio.gather(*active_tasks)
```
Cette boucle asynchrone garantit que le bus PCIe est toujours saturé de données, et que le GPU n'attend jamais (100% d'Overlap Calcul/Transfert).

---

## 9. Résolution des Problèmes Communs

1. **Memory Error / CUDA Out Of Memory** :
   Le GPU a une mémoire limitée. Si vous lancez une matrice géante d'un coup, il plantera.
   *Solution* : Regardez la Section 8 sur le Streaming Out-Of-Core. Utilisez un `asyncio.Queue` pour faire transiter les données par "Chunks".
   
2. **Erreur de Transpilation (IndentationError, Ast Parse)** :
   Le code à l'intérieur de `@gpusync_kernel` doit être valide en Python pur. Les opérations Numpy comme `A + B` sur des tableaux n'existent pas nativement en C. Écrivez explicitement la boucle ou l'accès indexé `A[ctx] + B[ctx]`.

3. **Incompatibilité OpenCL/CUDA** :
   Si vous passez des buffers instanciés par un `Scheduler` OpenCL vers un kernel géré par un `Scheduler` CUDA, GPUsync le détectera dans un Pipeline et insèrera un transfert automatique par la RAM. Mais en appel manuel, cela produira un crash. Surveillez le `backend` !

---

**Bon code, et n'hésitez pas à explorer le dossier `examples/` pour comprendre comment implémenter un Memory Pool asynchrone robuste !**
