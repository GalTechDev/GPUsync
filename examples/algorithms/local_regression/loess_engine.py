import time
import asyncio
import numpy as np
import sys
import os
import math

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel


def get_loess_kernel(N, W):
    @gpusync_kernel(constants={"N_CONST": N, "W_CONST": W})
    def loess_linear_kernel(ctx: int, Y: list, out_Y: list):
        if ctx >= N_CONST:
            return
            
        j_start = ctx - W_CONST
        if j_start < 0:
            j_start = 0
            
        j_end = ctx + W_CONST + 1
        if j_end > N_CONST:
            j_end = N_CONST
            
        # Accumulateurs pour le système linéaire 2x2
        S_00 = 0.0
        S_10 = 0.0
        S_11 = 0.0
        B_0 = 0.0
        B_1 = 0.0
        
        # Le point central est ctx
        for j in range(j_start, j_end):
            # Calcul de la distance
            dist = j - ctx
            if dist < 0:
                dist = -dist
                
            # Noyau Tri-Cube : w = (1 - (dist/W)^3)^3
            u = float(dist) / float(W_CONST)
            if u >= 1.0:
                continue
                
            u3 = u * u * u
            w_base = 1.0 - u3
            w = w_base * w_base * w_base
            
            x_diff = float(j - ctx)
            y_j = Y[j]
            
            S_00 = S_00 + w
            S_10 = S_10 + w * x_diff
            S_11 = S_11 + w * x_diff * x_diff
            B_0 = B_0 + w * y_j
            B_1 = B_1 + w * x_diff * y_j
            
        # Résolution par la règle de Cramer
        # det = S_00 * S_11 - S_10^2
        det = S_00 * S_11 - S_10 * S_10
        
        eps = 1e-8
        if det > eps:
            beta_0 = (B_0 * S_11 - B_1 * S_10) / det
            out_Y[ctx] = beta_0
        else:
            # Fallback à la moyenne pondérée (Degré 0)
            if S_00 > eps:
                out_Y[ctx] = B_0 / S_00
            else:
                out_Y[ctx] = Y[ctx]

    return loess_linear_kernel


async def loess_gpu_fit(Y: np.ndarray, W: int, scheduler: Scheduler):
    """
    Exécute LOESS sur GPU pour la série temporelle Y avec une fenêtre W.
    """
    N = len(Y)
    
    Y_gpu = await AsyncBuffer.to_device(scheduler, Y.astype(np.float32))
    out_Y_gpu = AsyncBuffer.empty_like(scheduler, np.zeros(N, dtype=np.float32))
    
    kernel = get_loess_kernel(N, W)
    
    print(f"[LOESS] Lancement du Kernel GPU (N={N}, W={W})...")
    t0 = time.perf_counter()
    
    await kernel(
        Y_gpu,
        out_Y_gpu,
        global_size=(N,),
        scheduler=scheduler
    )
    
    out_Y = await out_Y_gpu.to_host()
    t1 = time.perf_counter()
    
    print(f"[LOESS] Lissage calculé en {t1-t0:.3f}s")
    
    return out_Y
