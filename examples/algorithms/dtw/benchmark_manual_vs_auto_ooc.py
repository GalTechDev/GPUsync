# -*- coding: utf-8 -*-
"""
Benchmark comparatif entre le Out-Of-Core Manuel (plomberie asynchrone, pool de buffers)
et l'auto Out-Of-Core monoligne (map_out_of_core) de GPUsync.
Évalue le temps d'exécution, le débit et la complexité du code (lignes de code).
"""

import os
import sys
import time
import asyncio
import numpy as np

# Ajout des chemins de recherche
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gpusync import gpusync_kernel, Scheduler, list_gpus, AsyncBuffer
from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel

# Constantes pour le déroulage
class constant_list(list): pass

def unrolled_range(*args):
    return range(*args)


# =========================================================================
# 1. LOGIQUE OUT-OF-CORE MANUELLE (dtw_gpu_kf.py original de l'utilisateur)
# =========================================================================

async def process_chunk_manual(chunk_id, offset, chunk_size, gpu_refs_all, gpu_targets_chunk, 
                               scheduler, pool, num_refs, out_matrix, batch_kernel):
    # Récupérer un buffer du pool
    gpu_outputs_batch = await pool.get()
    
    # Exécuter le noyau GPU
    await batch_kernel(
        gpu_refs_all, gpu_targets_chunk, 0, 32, gpu_outputs_batch,
        num_refs, chunk_size, global_size=(chunk_size,), scheduler=scheduler
    )
        
    # Rapatrier et reformater
    res_flat = await gpu_outputs_batch.to_host()
    res_matrix = np.sqrt(np.maximum(res_flat[:num_refs * chunk_size].reshape(num_refs, chunk_size), 0.0))
    
    # Recopier dans la matrice globale
    np.copyto(out_matrix[:, offset : offset + chunk_size], res_matrix)
    
    # Remettre le buffer dans le pool
    pool.put_nowait(gpu_outputs_batch)
    return chunk_size


async def run_manual_ooc(sched, targets_np, refs_np, chunk_size, in_flight_limit):
    t_start = time.perf_counter()
    
    N = len(targets_np)
    K = len(refs_np)
    M = targets_np.shape[1]
    
    num_chunks = (N + chunk_size - 1) // chunk_size
    
    # 1. Compilation du noyau
    batch_kernel = get_batch_dtw_kernel(M, K)
    
    # 2. Upload des références en VRAM
    gpu_refs_all = await AsyncBuffer.to_device(sched, refs_np.flatten())
    
    # 3. Initialisation du pool d'outputs
    pool = asyncio.Queue(maxsize=in_flight_limit)
    dummy_out = np.zeros(K * chunk_size, dtype=np.float32)
    for _ in range(in_flight_limit):
        pool.put_nowait(AsyncBuffer.empty_like(sched, dummy_out))
        
    # 4. Pipeline asynchrone de streaming
    matrice_distances = np.zeros((K, N), dtype=np.float32)
    active_tasks = set()
    
    targets_flat = targets_np.flatten()
    
    for chunk_id in range(num_chunks):
        offset = chunk_id * chunk_size
        curr_size = min(chunk_size, N - offset)
        
        chunk_data = targets_flat[offset * M : (offset + curr_size) * M]
        gpu_targets_chunk = await AsyncBuffer.to_device(sched, chunk_data)
        
        task = asyncio.create_task(process_chunk_manual(
            chunk_id, offset, curr_size,
            gpu_refs_all, gpu_targets_chunk,
            sched, pool, K, matrice_distances, batch_kernel
        ))
        active_tasks.add(task)
        
        while len(active_tasks) >= in_flight_limit:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                coro.result()
                
    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            coro.result()
            
    t_total = time.perf_counter() - t_start
    return matrice_distances, t_total


# =========================================================================
# 2. LOGIQUE OUT-OF-CORE AUTOMATIQUE MONOLIGNE (GPUsync map_out_of_core)
# =========================================================================

def create_dtw_auto_kernel(M, K):
    """
    Génère un noyau JIT GPUsync pour la distance DTW unitaire entre N cibles et K=3 références.
    Le noyau renvoie un tuple de K distances float.
    """
    @gpusync_kernel(constants={"M_CONST": M})
    def dtw_auto_k3_kernel(ctx: int, targets: list, ref0: list, ref1: list, ref2: list) -> tuple:
        target_start: int = ctx * M_CONST
        
        # Charger la cible en registres locaux
        tgt_local = [0.0] * M_CONST
        for j in unrolled_range(0, M_CONST):
            tgt_local[j] = targets[target_start + j]
            
        # --- CALCUL POUR RÉFÉRENCE 0 ---
        row0 = [0.0] * M_CONST
        ref_val = ref0[0]
        d = ref_val - tgt_local[0]
        row0[0] = d * d
        for j in unrolled_range(1, M_CONST):
            d = ref_val - tgt_local[j]
            row0[j] = row0[j-1] + d * d
        for i in unrolled_range(1, M_CONST):
            ref_val = ref0[i]
            d = ref_val - tgt_local[0]
            m3_0 = row0[0]
            row0[0] = row0[0] + d * d
            m2_0 = row0[0]
            for j in unrolled_range(1, M_CONST):
                d = ref_val - tgt_local[j]
                m1_0 = row0[j]
                min_val_0 = m1_0
                if m2_0 < min_val_0: min_val_0 = m2_0
                if m3_0 < min_val_0: min_val_0 = m3_0
                m3_0 = row0[j]
                row0[j] = min_val_0 + d * d
                m2_0 = row0[j]
                
        # --- CALCUL POUR RÉFÉRENCE 1 ---
        row1 = [0.0] * M_CONST
        ref_val = ref1[0]
        d = ref_val - tgt_local[0]
        row1[0] = d * d
        for j in unrolled_range(1, M_CONST):
            d = ref_val - tgt_local[j]
            row1[j] = row1[j-1] + d * d
        for i in unrolled_range(1, M_CONST):
            ref_val = ref1[i]
            d = ref_val - tgt_local[0]
            m3_1 = row1[0]
            row1[0] = row1[0] + d * d
            m2_1 = row1[0]
            for j in unrolled_range(1, M_CONST):
                d = ref_val - tgt_local[j]
                m1_1 = row1[j]
                min_val_1 = m1_1
                if m2_1 < min_val_1: min_val_1 = m2_1
                if m3_1 < min_val_1: min_val_1 = m3_1
                m3_1 = row1[j]
                row1[j] = min_val_1 + d * d
                m2_1 = row1[j]
                
        # --- CALCUL POUR RÉFÉRENCE 2 ---
        row2 = [0.0] * M_CONST
        ref_val = ref2[0]
        d = ref_val - tgt_local[0]
        row2[0] = d * d
        for j in unrolled_range(1, M_CONST):
            d = ref_val - tgt_local[j]
            row2[j] = row2[j-1] + d * d
        for i in unrolled_range(1, M_CONST):
            ref_val = ref2[i]
            d = ref_val - tgt_local[0]
            m3_2 = row2[0]
            row2[0] = row2[0] + d * d
            m2_2 = row2[0]
            for j in unrolled_range(1, M_CONST):
                d = ref_val - tgt_local[j]
                m1_2 = row2[j]
                min_val_2 = m1_2
                if m2_2 < min_val_2: min_val_2 = m2_2
                if m3_2 < min_val_2: min_val_2 = m3_2
                m3_2 = row2[j]
                row2[j] = min_val_2 + d * d
                m2_2 = row2[j]

        # pyrefly: ignore [unknown-name]
        return sqrt(row0[M_CONST - 1]), sqrt(row1[M_CONST - 1]), sqrt(row2[M_CONST - 1])

    return dtw_auto_k3_kernel


async def run_auto_ooc(sched, targets_np, refs_np, chunk_size, concurrency):
    t_start = time.perf_counter()
    
    M = targets_np.shape[1]
    K = len(refs_np)
    
    # 1. Compilation
    dtw_kernel = create_dtw_auto_kernel(M, K)
    
    # 2. Séparation des références pour le passage en argument constant
    ref0 = refs_np[0]
    ref1 = refs_np[1]
    ref2 = refs_np[2]
    
    # 3. Appel monoligne asynchrone (l'API de compilation de GPUsync se charge de tout)
    # targets_np est découpé, ref0, ref1, ref2 sont passés comme arguments constants globaux
    res_tuple = await dtw_kernel.map_out_of_core(targets_np, sched, ref0, ref1, ref2, 
                                                 chunk_size=chunk_size, concurrency=concurrency)
    
    t_total = time.perf_counter() - t_start
    
    # Re-former la matrice de distances (K, N)
    # res_tuple est un tuple de K tableaux de longueur N
    matrice_distances = np.vstack(res_tuple)
    return matrice_distances, t_total


# =========================================================================
# 3. EXÉCUTION DU BENCHMARK COMPARATIF
# =========================================================================

async def main():
    print("=" * 85)
    print(" BENCHMARK COMPARATIF : OUT-OF-CORE MANUEL VS AUTO MONOLIGNE")
    print("=" * 85)

    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU OpenCL disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    await sched.initialize()
    
    # Paramètres de dimensionnement (K=3, N=300,000 séries réelles, M=32)
    K = 3
    N = 300_000
    M = 32
    CHUNK_SIZE = 50_000
    CONCURRENCY = 4
    
    print(f"\n[+] Génération de {N:,} séries de longueur {M} (Total = {N*M*4 / 1e6:.1f} Mo cibles)...")
    np.random.seed(42)
    refs_np = np.random.random((K, M)).astype(np.float32)
    targets_np = np.random.random((N, M)).astype(np.float32)

    # 1. Run Manuel OOC (Pool de buffers et plomberie asyncio)
    print(f"\n[1/2] Lancement du Out-Of-Core MANUEL (plomberie asynchrone)...")
    mat_manual, t_manual = await run_manual_ooc(sched, targets_np, refs_np, CHUNK_SIZE, CONCURRENCY)
    print(f"    -> Terminé en : {t_manual*1000:.2f} ms")
    print(f"    -> Débit      : {(N*K) / t_manual:.1f} distances/sec")

    # 2. Run Auto OOC (Monoligne map_out_of_core)
    print(f"\n[2/2] Lancement du Out-Of-Core AUTOMATIQUE (monoligne)...")
    mat_auto, t_auto = await run_auto_ooc(sched, targets_np, refs_np, CHUNK_SIZE, CONCURRENCY)
    print(f"    -> Terminé en : {t_auto*1000:.2f} ms")
    print(f"    -> Débit      : {(N*K) / t_auto:.1f} distances/sec")

    # 3. Vérification de l'équivalence numérique
    mae = np.mean(np.abs(mat_manual - mat_auto))
    accord = "PARFAIT" if mae < 1e-5 else "DIVERGENT"
    print(f"\n[+] Vérification des matrices : {accord} (MAE = {mae:.6f})")

    # 4. Comparaison des lignes de code
    # Code manuel requis : ~45 lignes (plomberie loop, asyncio.Queue, active_tasks wait, to_host, copyto, etc.)
    # Code auto requis   : 1 ligne (map_out_of_core)
    
    print("\n" + "=" * 95)
    print(" BILAN COMPARATIF")
    print("=" * 95)
    print(f" {'Indicateur':<30} | {'OOC Manuel (Plomberie)':<28} | {'OOC Auto (Monoligne)':<28}")
    print("-" * 95)
    print(f" {'Lignes de code (plomberie)':<30} | {'~45 lignes':<28} | {'1 ligne (map_out_of_core)':<28}")
    temps_str = "Temps d'exécution"
    complexite_str = "Complexité d'écriture"
    print(f" {temps_str:<30} | {t_manual*1000:>22.1f} ms | {t_auto*1000:>22.1f} ms")
    print(f" {'Débit (distances / sec)':<30} | {(N*K)/t_manual:>25,.0f} | {(N*K)/t_auto:>25,.0f}")
    print(f" {complexite_str:<30} | {'Élevée (Asyncio Queue/Pool)':<28} | {'Nulle (Gérée par JIT)':<28}")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(main())
