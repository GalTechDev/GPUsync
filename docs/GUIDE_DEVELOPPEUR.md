# Guide du Développeur GPUsync

Bienvenue dans l'équipe GPUsync ! 
Ce document est un guide pas-à-pas détaillé conçu pour vous accompagner dans l'écriture de vos premiers scripts avec la bibliothèque GPUsync. GPUsync vous permet de coder en Python tout en exécutant vos calculs à la vitesse de la lumière sur les GPUs de votre machine (NVIDIA CUDA, AMD OpenCL, Intel OpenCL), sur le processeur natif (LLVM CPU), ou via l'assembleur NVIDIA PTX, sans vous soucier des limites de mémoire (VRAM).

Ce guide couvre la structure des instructions, le concept de contexte (`ctx`), l'API monoligne `map_out_of_core`, le streaming par générateurs disque (`yield`), DLPack Zero-Copy, et la mise en place de pipelines asynchrones.

---

## 1. Philosophie : Que se passe-t-il sous le capot ?

GPUsync procède à une **Compilation Juste-à-Temps (JIT)**.
Lorsque vous écrivez une fonction avec `@gpusync_kernel`, le code ne s'exécute jamais en Python. Le framework lit votre code source, extrait son Arbre Syntaxique Abstrait (AST), et le **transpile** instantanément en code natif C++ (CUDA), C99 (OpenCL), JIT CPU (LLVM) ou assembleur PTX (NVIDIA) avant de l'injecter dans le matériel.

### Règles d'écriture fondamentales :
1. Vous pouvez utiliser des opérations mathématiques de base, des boucles `for/while`, des conditions `if/else`, ainsi que des commentaires `#` ou docstrings `"""..."""`.
2. Les arguments passés à vos kernels doivent être explicitement typés (`int`, `float`, `list`).

---

## 2. Démarrage Rapide : Traitement Out-Of-Core Monoligne (`map_out_of_core`)

GPUsync permet de traiter des jeux de données géants dépassants la VRAM en 1 seule ligne de code :

```python
from gpusync import gpusync_kernel, Scheduler, list_gpus
import numpy as np

@gpusync_kernel
def process_signal(ctx: int, data: list) -> tuple:
    return sqrt(data[ctx]) * 2.5 + 1.0

scheduler = Scheduler(gpu=list_gpus()[0])
scheduler.initialize()

# 1. Traitement monoligne d'un dataset géant (Tableau NumPy)
resultat = process_signal.map_out_of_core(large_dataset_50GB, scheduler=scheduler)

# 2. Traitement monoligne en STREAMING DISQUE CONTINU (Générateur yield)
def stream_fichier_disque(filepath):
    with open(filepath, "rb") as f:
        while True:
            chunk = np.fromfile(f, dtype=np.float32, count=1_000_000)
            if len(chunk) == 0: break
            yield chunk

resultat_disque = process_signal.map_out_of_core(stream_fichier_disque("big.bin"), scheduler=scheduler)
```

---

## 3. Structure d'un Projet Standard

Un script GPUsync avec gestion manuelle de buffers suit le cycle de vie : **Initialisation -> Transfert RAM/VRAM -> Calcul GPU -> Retour RAM**.

### A. Initialisation du Scheduler
```python
import numpy as np
from gpusync import list_gpus, Scheduler, AsyncBuffer

def main():
    gpus = list_gpus()
    if not gpus:
        raise RuntimeError("Aucun GPU disponible.")
        
    scheduler = Scheduler(gpu=gpus[0])
    scheduler.initialize()
```

### B. Allocation des Données
```python
a_cpu = np.ones(1_000_000, dtype=np.float32)
b_cpu = np.ones(1_000_000, dtype=np.float32)

a_gpu = AsyncBuffer.to_device(scheduler, a_cpu)
b_gpu = AsyncBuffer.to_device(scheduler, b_cpu)
```

### C. Exécution du Kernel
```python
@gpusync_kernel
def addition_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx]

resultat_cpu = addition_kernel(a_gpu, b_gpu, global_size=(1_000_000,), scheduler=scheduler)
```

---

## 4. Partage de Mémoire Zero-Copy DLPack (PyTorch / CuPy)

`AsyncBuffer` implémente le protocole standard **DLPack** (`__dlpack__`, `from_dlpack`) pour échanger des buffers VRAM avec PyTorch et CuPy sans aucune copie mémoire :

```python
import torch
from gpusync import AsyncBuffer

# Import Zero-Copy d'un Tenseur PyTorch GPU
tensor_gpu = torch.randn(1_000_000, device="cuda")
async_buf = AsyncBuffer.from_dlpack(tensor_gpu, scheduler=scheduler)

# Export Zero-Copy d'un AsyncBuffer vers PyTorch
torch_tensor = torch.from_dlpack(async_buf)
```
