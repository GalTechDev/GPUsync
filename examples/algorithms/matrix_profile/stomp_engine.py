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

def atomic_min_float_global(arr, val):
    pass

def get_stomp_kernel(M, N):
    @gpusync_kernel(constants={"M_CONST": M, "N_CONST": N})
    def stomp_diagonal_kernel(ctx: int, T: list, MP: list, mu: list, sigma: list):
        # Exclusion zone is M/4. 
        # So thread 0 processes diagonal k = M/4 + 1
        exclusion_zone = M_CONST // 4
        k: int = ctx + exclusion_zone + 1
        
        # We only process if k < N_CONST
        if k >= N_CONST:
            return
            
        iters: int = N_CONST - k
        if iters <= 0: return
        
        # Warp-uniform stagger to preserve memory coalescing
        wg_id: int = k // 64
        base_iters: int = N_CONST - (wg_id * 64)
        stagger: int = 0
        if base_iters > 64:
            stagger = (wg_id * 137) % (base_iters - 64)
        
        j: int = 0
        mean_i: float = 0.0
        mean_j: float = 0.0
        std_i: float = 0.0
        std_j: float = 0.0
        denom: float = 0.0
        rho: float = 0.0
        dist: float = 0.0
        
        # --- PREMIÈRE BOUCLE (de stagger à iters - 1) ---
        # Initial Dot Product for stagger
        dp = 0.0
        for m in range(M_CONST):
            dp = dp + T[stagger + m] * T[k + stagger + m]
            
        eps = 1e-8
        
        for i in range(stagger, iters):
            j = i + k
            
            mean_i = mu[i]
            mean_j = mu[j]
            std_i = sigma[i]
            std_j = sigma[j]
            
            denom = M_CONST * std_i * std_j
            rho = (dp - M_CONST * mean_i * mean_j) / denom if denom > eps else 1.0
            
            if rho > 1.0: rho = 1.0
            if rho < -1.0: rho = -1.0
            
            dist = math.sqrt(2.0 * M_CONST * (1.0 - rho))
            
            atomic_min_float_global(MP[i], dist)
            atomic_min_float_global(MP[j], dist)
            
            if i + 1 < iters:
                dp = dp - T[i] * T[j] + T[i + M_CONST] * T[j + M_CONST]
                
        # --- DEUXIÈME BOUCLE (de 0 à stagger - 1) ---
        if stagger > 0:
            dp = 0.0
            for m in range(M_CONST):
                dp = dp + T[m] * T[k + m]
                
            for i in range(0, stagger):
                j = i + k
                
                mean_i = mu[i]
                mean_j = mu[j]
                std_i = sigma[i]
                std_j = sigma[j]
                
                denom = M_CONST * std_i * std_j
                rho = (dp - M_CONST * mean_i * mean_j) / denom if denom > eps else 1.0
                
                if rho > 1.0: rho = 1.0
                if rho < -1.0: rho = -1.0
                
                dist = math.sqrt(2.0 * M_CONST * (1.0 - rho))
                
                atomic_min_float_global(MP[i], dist)
                atomic_min_float_global(MP[j], dist)
                
                if i + 1 < stagger:
                    dp = dp - T[i] * T[j] + T[i + M_CONST] * T[j + M_CONST]
                    
    return stomp_diagonal_kernel

def compute_rolling_stats(T, M):
    """
    Calcule les moyennes et écarts-types glissants de fenêtres de taille M en O(L).
    """
    L = len(T)
    N = L - M + 1
    
    # Prefix sums
    cumsum = np.zeros(L + 1, dtype=np.float64)
    cumsum_sq = np.zeros(L + 1, dtype=np.float64)
    
    cumsum[1:] = np.cumsum(T)
    cumsum_sq[1:] = np.cumsum(T**2)
    
    # Rolling sums
    sum_M = cumsum[M:] - cumsum[:-M]
    sum_sq_M = cumsum_sq[M:] - cumsum_sq[:-M]
    
    mu = sum_M / M
    var = (sum_sq_M / M) - (mu**2)
    # Fix negative variances due to float precision
    var[var < 0] = 0
    sigma = np.sqrt(var)
    
    return mu.astype(np.float32), sigma.astype(np.float32)

async def stomp_matrix_profile(time_series: np.ndarray, M: int, scheduler: Scheduler):
    """
    Calcule le Matrix Profile complet de `time_series` avec fenêtre de taille `M`.
    Retourne le tableau Matrix Profile (MP).
    """
    L = len(time_series)
    N = L - M + 1
    
    if N <= 0:
        raise ValueError("La taille de la série temporelle doit être supérieure à la fenêtre M.")
        
    print(f"[STOMP] Pré-calcul des statistiques glissantes (L={L}, M={M})...")
    t0 = time.perf_counter()
    mu, sigma = compute_rolling_stats(time_series, M)
    print(f"[STOMP] Stats calculées en {time.perf_counter() - t0:.3f}s")
    
    T_gpu = await AsyncBuffer.to_device(scheduler, time_series.astype(np.float32))
    mu_gpu = await AsyncBuffer.to_device(scheduler, mu)
    sigma_gpu = await AsyncBuffer.to_device(scheduler, sigma)
    
    # Initialisation du Matrix Profile à +infini
    mp_init = np.full(N, np.inf, dtype=np.float32)
    mp_gpu = await AsyncBuffer.to_device(scheduler, mp_init)
    
    exclusion_zone = M // 4
    num_diagonals = N - exclusion_zone - 1
    
    if num_diagonals <= 0:
        return mp_init # Série trop courte
        
    kernel = get_stomp_kernel(M, N)
    
    print(f"[STOMP] Lancement du Kernel GPU (Diagonales={num_diagonals}, N={N})...")
    t1 = time.perf_counter()
    
    # Lancement d'un thread par diagonale
    await kernel(
        T_gpu,
        mp_gpu,
        mu_gpu,
        sigma_gpu,
        global_size=(num_diagonals,),
        scheduler=scheduler
    )
    
    mp_result = await mp_gpu.to_host()
    t2 = time.perf_counter()
    
    print(f"[STOMP] Profil calculé en {t2 - t1:.3f}s sur GPU")
    
    return mp_result
