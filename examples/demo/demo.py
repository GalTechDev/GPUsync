import sys
import os
import numpy as np
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel
from gpusync import list_gpus

@gpusync_kernel
def math_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx], a[ctx] - b[ctx]

def main():
    # Découverte et sélection dynamique du périphérique disponible
    gpus = list_gpus()
    if not gpus:
        print("Aucun périphérique GPU/OpenCL détecté sur le système.")
        return
        
    print("Périphériques détectés :")
    for idx, gpu in enumerate(gpus):
        print(f" - [{idx}] {gpu.name} (backend: {gpu.backend})")
        
    # Utilisation du premier périphérique disponible
    selected_gpu = gpus[0]
    print(f"\nInitialisation du scheduler sur : {selected_gpu.name} ({selected_gpu.backend})")
    
    scheduler = Scheduler(gpu=selected_gpu)
    scheduler.initialize()
    
    # 1. Allocation des données sur le CPU
    a_cpu = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    b_cpu = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float32)
    
    # 2. Transfert vers le GPU (SYNCHRONE)
    a_gpu = AsyncBuffer.to_device(scheduler, a_cpu)
    b_gpu = AsyncBuffer.to_device(scheduler, b_cpu)
    
    # 3. Exécution du Kernel et rapatriement des données (SYNCHRONE)
    resultplus_cpu, resultmoins_cpu = math_kernel(a_gpu, b_gpu, global_size=(8,), scheduler=scheduler)
    
    print("Résultat plus :", resultplus_cpu)
    print("Résultat moins :", resultmoins_cpu)

if __name__ == "__main__":
    main()