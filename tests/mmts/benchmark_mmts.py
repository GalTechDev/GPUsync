# -*- coding: utf-8 -*-
"""
Script de Benchmark pour comparer les performances de la version originale (Pure Python)
avec les versions optimisées (CPU JIT Numba et GPU GPUsync) et générer un rapport HTML stylisé.
"""

import math
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from tslearn.barycenters import euclidean_barycenter

# Importation du module xmeants et des fonctions de configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmts_module as xm
from numba import njit, prange

from config import *

try:
    from numba import cuda as numba_cuda

    NUMBA_CUDA_OK = numba_cuda.is_available()
except Exception:
    NUMBA_CUDA_OK = False

BENCHMARK_GPU = True  # GPUsync OpenCL
BENCHMARK_NUMBA_CUDA = NUMBA_CUDA_OK  # Numba CUDA (nécessite pilote CUDA)

if not NUMBA_CUDA_OK:
    print("[!] Numba CUDA non disponible sur ce système (pas de pilote CUDA). La variante 'Numba GPU' sera sautée.")

# =====================================================================
# NOYAUX NUMBA (CPU parallèle + GPU CUDA)
# =====================================================================


@njit(fastmath=True, cache=True)
def _dtw_single_numba(ref, tgt, w):
    M = len(ref)
    n_w = max(w, 0)
    dtw = np.full((M + 1, M + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0
    for i in range(1, M + 1):
        start = max(1, i - n_w)
        end = min(M, i + n_w) + 1
        for j in range(start, end):
            cost = (ref[i - 1] - tgt[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return np.sqrt(dtw[M, M])


@njit(parallel=True, fastmath=True, cache=True)
def batch_dtw_numba_parallel(refs, targets, w, out):
    """Batch DTW K références x N cibles, parallélisé sur CPU avec Numba prange."""
    K = refs.shape[0]
    N = targets.shape[0]
    for i in prange(N):
        for k in range(K):
            out[k, i] = _dtw_single_numba(refs[k], targets[i], w)


def batch_dtw_numba_cpu(refs: np.ndarray, targets: np.ndarray, w: int) -> np.ndarray:
    """Wrapper pour batch_dtw_numba_parallel retournant un tableau (K, N)."""
    K, N = refs.shape[0], targets.shape[0]
    out = np.zeros((K, N), dtype=np.float32)
    batch_dtw_numba_parallel(refs, targets, w, out)
    return out


if NUMBA_CUDA_OK:

    @numba_cuda.jit
    def _cuda_batch_dtw_kernel(refs, targets, out, w, M):
        """Noyau CUDA Numba : chaque thread traite une paire (k, i)."""
        idx = numba_cuda.grid(1)
        K = refs.shape[0]
        N = targets.shape[0]
        if idx >= K * N:
            return
        k = idx // N
        i = idx % N
        n_w = max(w, 0)
        # DTW inline (tableaux locaux)
        dtw = numba_cuda.local.array((33, 33), dtype=np.float32)
        for r in range(33):
            for c in range(33):
                dtw[r][c] = 1e30
        dtw[0][0] = 0.0
        for ri in range(1, M + 1):
            j_start = max(1, ri - n_w)
            j_end = min(M, ri + n_w) + 1
            for ci in range(j_start, j_end):
                cost = (refs[k, ri - 1] - targets[i, ci - 1]) ** 2
                v = dtw[ri - 1][ci]
                if dtw[ri][ci - 1] < v:
                    v = dtw[ri][ci - 1]
                if dtw[ri - 1][ci - 1] < v:
                    v = dtw[ri - 1][ci - 1]
                dtw[ri][ci] = cost + v
        out[k, i] = dtw[M][M] ** 0.5

    def batch_dtw_numba_cuda(refs: np.ndarray, targets: np.ndarray, w: int) -> np.ndarray:
        """Lance le noyau CUDA Numba DTW, retourne (K, N)."""
        K, M = refs.shape
        N = targets.shape[0]
        refs_d = numba_cuda.to_device(refs)
        targets_d = numba_cuda.to_device(targets)
        out_d = numba_cuda.device_array((K, N), dtype=np.float32)
        threads = 256
        blocks = (K * N + threads - 1) // threads
        _cuda_batch_dtw_kernel[blocks, threads](refs_d, targets_d, out_d, w, M)
        return out_d.copy_to_host()
else:

    def batch_dtw_numba_cuda(refs, targets, w):
        raise RuntimeError("Numba CUDA non disponible.")

# =====================================================================
# 1. IMPLÉMENTATION DES VERSIONS PURES PYTHON (ORIGINALES)
# =====================================================================


def py_dtw_original(s1, s2, w):
    """Calcule la distance DTW en Python pur (implémentation originale)"""
    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float("inf")
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w + 1)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def py_compute_dist_original(data, centroids, assignments, w):
    """Calcule les distances et fait les assignations en Python pur"""
    for c in assignments:
        assignments[c] = []

    for i in range(len(data)):
        min_dist = float("inf")
        closest_clust = None
        for c in centroids:
            if len(data[i]) > 0:
                if w > 0:
                    dist = py_dtw_original(data[i], centroids[c], w)
                else:
                    dist = np.linalg.norm(data[i] - centroids[c])

                if dist < min_dist:
                    min_dist = dist
                    closest_clust = c
        if closest_clust is not None:
            assignments[closest_clust].append(i)
    return assignments


def py_get_last_mat_original(data, centroids, df_ids, tabEnviIndice, w):
    """Récupère la matrice des distances finale en Python pur"""
    MatriceDist = pd.DataFrame()
    for uit in centroids.keys():
        listTem = []
        for ine in range(len(data)):
            listTem.append(py_dtw_original(centroids[uit], data[ine], w))
        MatriceDist[str(uit) + "_" + tabEnviIndice] = listTem

    MatriceDist[ID_COLUMN] = df_ids[ID_COLUMN]
    return MatriceDist


def py_define_coeff_original(
    data,
    num_clust,
    num_iter,
    listDataFrame,
    LastnbClust,
    tabEnviIndice,
    assignmentsResult,
    centroidsCompare,
    distCompare,
    data22,
    listeIDOk1,
    nbInstanceMinim,
    w=1,
):
    """Calcule les coefficients en Python pur (version globale originale)"""
    coeffTemp = []
    centroids = {}
    for i in range(num_clust):
        idx = random.randint(0, len(data) - 1)
        centroids[i] = data[idx]

    assignments = {}
    for i in range(num_clust):
        assignments[i] = []

    # Calcul des assignations initiales
    assignments = py_compute_dist_original(data, centroids, assignments, w)

    # Calcul des coefficients
    for i in assignments:
        if len(assignments[i]) >= nbInstanceMinim:
            values = []
            for item in assignments[i]:
                values.append(data[item])

            if values:
                # Version originale avec la dimension corrompue [values]
                bar = euclidean_barycenter([values])
                flat_list = [item for sublist in bar for item in sublist]

                distances = []
                for item in assignments[i]:
                    distances.append(py_dtw_original(data[item], flat_list, w))

                stdTemp = np.std(distances)
                entropy = stats.entropy(distances) if any(d > 0 for d in distances) else 1.0

                if entropy > 0:
                    coeffTemp.append(stdTemp / entropy)

    return coeffTemp if coeffTemp else [1.0]


class PurePythonXmeansTS:
    """Classe reproduisant le pipeline X-means original 100% Pure Python"""

    def __init__(self, data, numclust, file, listID, coeffRap):
        self.data = data
        self.dfListID = listID
        self.assignments = {}
        self.centroids = {}

    def updateAssignment(self, assignmentsTemp, listeIDOk, nbClustTempp, w):
        for key in assignmentsTemp:
            self.assignments[nbClustTempp] = assignmentsTemp[key]
            flat_list = []
            for elem in self.assignments[nbClustTempp]:
                listeIDOk.append(elem)

            if len(self.assignments[nbClustTempp]) > 0:
                values = []
                for elId in self.assignments[nbClustTempp]:
                    idTS = np.asarray(self.data)[(self.dfListID[ID_COLUMN] == elId).values]
                    if len(idTS) > 0:
                        values.append(idTS[0])

                if values:
                    # Version originale avec la dimension corrompue [values]
                    bar = euclidean_barycenter([values])
                    flat_list = [item for sublist in bar for item in sublist]

                self.centroids[nbClustTempp] = flat_list
                MatDist = pd.DataFrame(columns=[ID_COLUMN, "MEAS"])

                for elId in self.assignments[nbClustTempp]:
                    idTS = np.asarray(self.data)[(self.dfListID[ID_COLUMN] == elId).values]
                    if len(idTS) > 0:
                        LBVal = py_dtw_original(idTS[0], flat_list, w)

                        new_row = pd.DataFrame({ID_COLUMN: [int(elId)], "MEAS": [LBVal]})
                        MatDist = pd.concat([MatDist, new_row], ignore_index=True)

                nbClustTempp += 1
        return listeIDOk

    def k_means_clust(
        self,
        data,
        num_clust,
        num_iter,
        listDataFrame,
        LastnbClust,
        tabEnviIndice,
        distCompare,
        listeIDOk1,
        nbInstanceMinim,
        w,
    ):
        if num_clust == 0:
            return pd.DataFrame(listDataFrame)

        if len(data) > num_clust * 2:
            centroids = {}
            for i in range(num_clust):
                idx = random.randint(0, len(data) - 1)
                centroids[i] = data[idx]

            assignments = {i: [] for i in range(num_clust)}
            assignments = py_compute_dist_original(data, centroids, assignments, w)

            # Mise à jour centroides
            for c in assignments:
                if len(assignments[c]) > 0:
                    new_centroid = np.zeros(len(data[0]))
                    for i in assignments[c]:
                        if len(data[i]) > 0:
                            new_centroid += data[i]
                    centroids[c] = new_centroid / len(assignments[c])

            assignmentsTemp = {}
            for i in assignments:
                if len(assignments[i]) >= nbInstanceMinim:
                    ids = [int(listDataFrame.iloc[item][ID_COLUMN]) for item in assignments[i]]
                    assignmentsTemp[i] = ids

            nbClustLow = LastnbClust
            listeIDOk = []

            listeIDOk = self.updateAssignment(assignmentsTemp, listeIDOk, LastnbClust, w)
            listeIDOk1 = listeIDOk1 + listeIDOk

            # updateVar
            dfListIDTemp = listDataFrame[~listDataFrame[ID_COLUMN].isin(listeIDOk1)]
            updateVar_data = []
            for i in range(len(dfListIDTemp)):
                updateVar_data.append(self.data[i])

            # Récursion pure Python
            if len(updateVar_data) > 0 and len(updateVar_data) < len(data):
                self.k_means_clust(
                    updateVar_data,
                    num_clust,
                    num_iter,
                    dfListIDTemp,
                    LastnbClust + len(assignmentsTemp),
                    tabEnviIndice,
                    distCompare,
                    listeIDOk1,
                    nbInstanceMinim,
                    w,
                )

            if nbClustLow == 0:
                return py_get_last_mat_original(self.data, self.centroids, listDataFrame, tabEnviIndice, w)

        elif LastnbClust == 0:
            return py_get_last_mat_original(self.data, self.centroids, listDataFrame, tabEnviIndice, w)

        dfListIDTemp = pd.DataFrame(listDataFrame)
        dfListIDTemp["cluster"] = -1
        for ctr in range(len(self.assignments)):
            for elemid in self.assignments[ctr]:
                dfListIDTemp.loc[dfListIDTemp[ID_COLUMN] == elemid, "cluster"] = ctr
        return dfListIDTemp


# =====================================================================
# 2. RUN BENCHMARKS
# =====================================================================


def run_benchmarks():
    print("=" * 80)
    print("             BENCHMARK DES PERFORMANCES DU PIPELINE DTW & X-MEANS")
    print("=" * 80)

    w = 5
    nb_instance_min = 5

    # Chargement du vrai jeu de données ou fallback sur les données générées
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "RomaneDataSet_TEMP.csv")
    use_real_data = False

    if os.path.exists(csv_path):
        try:
            df_real = pd.read_csv(csv_path)
            val_col = "value" if "value" in df_real.columns else df_real.columns[1]
            data_values = df_real[val_col].values.astype(np.float32)
            M = 32
            N_all = len(data_values) - M + 1
            if N_all > 10:
                data = []
                for i in range(N_all):
                    data.append(data_values[i : i + M])
                df_ids = pd.DataFrame({ID_COLUMN: [i for i in range(N_all)]})
                use_real_data = True
                print(f"[+] Jeu de données réel chargé : {N_all} séries de longueur {M}.")
        except Exception as e:
            print(f"[!] Échec du chargement du jeu de données réel ({e}). Repli sur simulation.")

    if not use_real_data:
        N = 100
        M = 32
        data = []
        for i in range(N):
            phase = random.choice([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
            t_seq = np.arange(M)
            sig = 20.0 + 5.0 * np.sin(t_seq * (2 * np.pi / 16) + phase) + np.random.randn(M) * 0.3
            data.append(sig.astype(np.float32))

        df_ids = pd.DataFrame({ID_COLUMN: [i for i in range(N)]})
        print(f"[+] Données simulées générées : {len(data)} séries de longueur {M}.")

    # Sous-ensemble Python pur (Benchmarks 2 & 3)
    data_py = data[:150]
    df_ids_py = df_ids.iloc[:150]

    # -----------------------------------------------------------------
    # BENCHMARK 1 : BATCH DTW SUR LE DATASET COMPLET (K=3 centroides x N séries)
    # -----------------------------------------------------------------
    N_full = len(data)
    K_bench = 3
    M_len = len(data[0])

    # Préparer les tableaux batch : K centroïdes aléatoires + toutes les N séries cibles
    np.random.seed(42)
    bench_centroid_idx = np.random.choice(N_full, size=K_bench, replace=False)
    bench_centroids = np.array([data[i] for i in bench_centroid_idx], dtype=np.float32)  # (K, M)
    bench_targets = np.array(data, dtype=np.float32)  # (N, M)
    N_pairs_full = K_bench * N_full  # nombre total de paires calculées

    print(f"\n[+] BENCHMARK 1: Batch DTW réel — K={K_bench} centroides × N={N_full} séries = {N_pairs_full:,} paires")
    print("    (Résultats normalisés en équivalent-1000-paires pour comparaison)")

    # 1.1 Python Pur (sur sous-ensemble limité, extrapolé)
    PY_SUBSET = min(150, N_full)
    bench_targets_py = bench_targets[:PY_SUBSET]
    t0 = time.perf_counter()
    for k_idx in range(K_bench):
        for i in range(PY_SUBSET):
            py_dtw_original(bench_centroids[k_idx], bench_targets_py[i], w)
    t_py_raw = time.perf_counter() - t0
    pairs_py = K_bench * PY_SUBSET
    t_dtw_py = t_py_raw / pairs_py * 1000  # coût pour 1000 paires (extrapolé)

    # 1.2 CPU JIT Numba séquentiel (warmup + mesure sur dataset complet)
    _ = xm.batch_dtw_cpu(bench_centroids[:1], bench_targets[:2], w)  # warmup JIT
    t0 = time.perf_counter()
    _ = xm.batch_dtw_cpu(bench_centroids, bench_targets, w)
    t_jit_raw = time.perf_counter() - t0
    t_dtw_cpu = t_jit_raw / N_pairs_full * 1000

    # 1.3 Numba CPU Parallèle (prange) — warmup + mesure
    _out_warm = np.zeros((K_bench, 2), dtype=np.float32)
    batch_dtw_numba_parallel(bench_centroids, bench_targets[:2], w, _out_warm)  # warmup
    _out_nb = np.zeros((K_bench, N_full), dtype=np.float32)
    t0 = time.perf_counter()
    batch_dtw_numba_parallel(bench_centroids, bench_targets, w, _out_nb)
    t_par_raw = time.perf_counter() - t0
    t_dtw_numba_par = t_par_raw / N_pairs_full * 1000

    # 1.4 Numba GPU CUDA
    t_dtw_numba_cuda = -1.0
    if BENCHMARK_NUMBA_CUDA:
        try:
            batch_dtw_numba_cuda(bench_centroids, bench_targets[:2], w)  # warmup
            t0 = time.perf_counter()
            batch_dtw_numba_cuda(bench_centroids, bench_targets, w)
            t_cuda_raw = time.perf_counter() - t0
            t_dtw_numba_cuda = t_cuda_raw / N_pairs_full * 1000
        except Exception as e:
            print(f"    - Version Numba CUDA   : Échec ({e})")

    # 1.5 GPU GPUsync (OpenCL) — batch complet
    t_dtw_gpu = -1.0
    if BENCHMARK_GPU:
        try:
            from gpusync import AsyncBuffer

            M_pad = int(2 ** math.ceil(math.log2(M_len)))
            M_pad = max(32, M_pad)
            sched, kernel_batch = xm.get_gpusync_resources(M_pad, K_bench)

            refs_pad = np.zeros((K_bench, M_pad), dtype=np.float32)
            refs_pad[:, :M_len] = bench_centroids
            tgts_pad = np.zeros((N_full, M_pad), dtype=np.float32)
            tgts_pad[:, :M_len] = bench_targets

            buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
            buf_tgt = AsyncBuffer.to_device(sched, tgts_pad.flatten())
            buf_out = AsyncBuffer.empty_like(sched, np.zeros(K_bench * N_full, dtype=np.float32))

            # Warmup
            kernel_batch(buf_ref, buf_tgt, 0, M_pad, buf_out, K_bench, N_full, global_size=(N_full,), scheduler=sched)
            _ = buf_out.to_host()

            t0 = time.perf_counter()
            kernel_batch(buf_ref, buf_tgt, 0, M_pad, buf_out, K_bench, N_full, global_size=(N_full,), scheduler=sched)
            _ = buf_out.to_host()
            t_gpu_raw = time.perf_counter() - t0
            t_dtw_gpu = t_gpu_raw / N_pairs_full * 1000
        except Exception as e:
            print(f"    - GPU GPUsync Error: {e}")

    # Affichage — temps réel + coût normalisé pour 1000 paires
    def fmt(t_raw_s, t_norm_ms, ref_py):
        spd = f"{ref_py/t_norm_ms:.0f}x" if t_norm_ms > 0 else "N/A"
        return f"{t_raw_s*1000:.1f} ms total | ~{t_norm_ms:.4f} ms/1000 paires | {spd} vs Python"

    print(f"    - Python Pur (x{PY_SUBSET} séries, extrap.)  : {fmt(t_py_raw, t_dtw_py, t_dtw_py)}")
    print(f"    - CPU JIT (batch {N_full})            : {fmt(t_jit_raw, t_dtw_cpu, t_dtw_py)}")
    print(f"    - CPU Numba || (batch {N_full})       : {fmt(t_par_raw, t_dtw_numba_par, t_dtw_py)}")
    if t_dtw_numba_cuda > 0:
        print(f"    - Numba CUDA GPU (batch {N_full})     : {fmt(t_cuda_raw, t_dtw_numba_cuda, t_dtw_py)}")
    else:
        print("    - Numba CUDA GPU                : Non disponible (pas de driver CUDA).")
    if t_dtw_gpu > 0:
        print(f"    - GPU GPUsync (batch {N_full})        : {fmt(t_gpu_raw, t_dtw_gpu, t_dtw_py)}")
    else:
        print("    - GPU GPUsync                   : Non disponible.")

    # -----------------------------------------------------------------
    # BENCHMARK 2 : CALCUL DES COEFFICIENTS (defineCoeff)
    # -----------------------------------------------------------------
    print("\n[+] BENCHMARK 2: Calcul des coefficients (defineCoeff)")

    # 2.1 Python Pur Original (sur sous-ensemble de 150 séries)
    assignmentsResult_py = {}
    centroidsCompare_py = {}
    distCompare_py = {}
    listeIDOk1_py = []

    t0 = time.perf_counter()
    _ = py_define_coeff_original(
        data_py,
        3,
        1,
        df_ids_py,
        0,
        "Set_TEMP",
        assignmentsResult_py,
        centroidsCompare_py,
        distCompare_py,
        data_py,
        listeIDOk1_py,
        nb_instance_min,
        w,
    )
    t_coeff_py = time.perf_counter() - t0

    # 2.2 CPU JIT (Numba)
    assignmentsResult_cpu = {}
    centroidsCompare_cpu = {}
    distCompare_cpu = {}
    listeIDOk1_cpu = []

    t0 = time.perf_counter()
    _ = xm.defineCoeff(
        data,
        3,
        1,
        df_ids,
        0,
        "Set_TEMP",
        assignmentsResult_cpu,
        centroidsCompare_cpu,
        distCompare_cpu,
        data,
        listeIDOk1_cpu,
        nb_instance_min,
        w,
    )
    t_coeff_cpu = time.perf_counter() - t0

    # 2.3 GPU GPUsync (si activé)
    t_coeff_gpu = -1.0
    if BENCHMARK_GPU:
        xm.USE_GPU = True
        try:
            assignmentsResult_gpu = {}
            centroidsCompare_gpu = {}
            distCompare_gpu = {}
            listeIDOk1_gpu = []
            t0 = time.perf_counter()
            _ = xm.defineCoeff(
                data,
                3,
                1,
                df_ids,
                0,
                "Set_TEMP",
                assignmentsResult_gpu,
                centroidsCompare_gpu,
                distCompare_gpu,
                data,
                listeIDOk1_gpu,
                nb_instance_min,
                w,
            )
            t_coeff_gpu = time.perf_counter() - t0
        except Exception:
            pass
        xm.USE_GPU = False

    print(f"    - Version Python Pur   : {t_coeff_py * 1000:.2f} ms")
    print(f"    - Version CPU JIT      : {t_coeff_cpu * 1000:.2f} ms (Accélération: {t_coeff_py / t_coeff_cpu:.1f}x)")
    if t_coeff_gpu > 0:
        print(
            f"    - Version GPU GPUsync  : {t_coeff_gpu * 1000:.2f} ms (Accélération: {t_coeff_py / t_coeff_gpu:.1f}x)"
        )

    # -----------------------------------------------------------------
    # BENCHMARK 3 : CLUSTERING X-MEANS COMPLET & PIPELINE
    # -----------------------------------------------------------------
    print("\n[+] BENCHMARK 3: Pipeline complet du clustering X-means (1 iteration)")

    # 3.1 Python Pur Original (sur sous-ensemble de 150 séries)
    py_xmeans = PurePythonXmeansTS(data_py, 3, "dummy.xlsx", df_ids_py, 1.0)
    t0 = time.perf_counter()
    _ = py_xmeans.k_means_clust(data_py, 3, 1, df_ids_py, 0, "Set_TEMP", {}, [], nb_instance_min, w)
    t_pipe_py = time.perf_counter() - t0

    # 3.2 CPU JIT (Numba)
    # On force l'initialisation du cache JIT Numba sur une première exécution
    opt_xmeans_warm = xm.XmeansTS(data, 3, "dummy.xlsx", df_ids, 1.0)
    _ = opt_xmeans_warm.k_means_clust(data, 3, 1, df_ids, 0, "Set_TEMP", {}, [], nb_instance_min, w)

    opt_xmeans = xm.XmeansTS(data, 3, "dummy.xlsx", df_ids, 1.0)
    t0 = time.perf_counter()
    _ = opt_xmeans.k_means_clust(data, 3, 1, df_ids, 0, "Set_TEMP", {}, [], nb_instance_min, w)
    t_pipe_cpu = time.perf_counter() - t0

    # 3.3 GPU GPUsync (si activé)
    t_pipe_gpu = -1.0
    if BENCHMARK_GPU:
        xm.USE_GPU = True
        try:
            opt_xmeans_gpu = xm.XmeansTS(data, 3, "dummy.xlsx", df_ids, 1.0)
            t0 = time.perf_counter()
            _ = opt_xmeans_gpu.k_means_clust(data, 3, 1, df_ids, 0, "Set_TEMP", {}, [], nb_instance_min, w)
            t_pipe_gpu = time.perf_counter() - t0
        except Exception:
            pass
        xm.USE_GPU = False

    print(f"    - Version Python Pur (150 s.): {t_pipe_py * 1000:.2f} ms")
    print(f"    - Version CPU JIT (Complète) : {t_pipe_cpu * 1000:.2f} ms")
    if t_pipe_gpu > 0:
        print(f"    - Version GPU GPUsync (Compl.) : {t_pipe_gpu * 1000:.2f} ms")

    # -----------------------------------------------------------------
    # BENCHMARK 4 : VRAI X-MEANS (TrueXmeansTS) BIC + DTW + Génétique
    # -----------------------------------------------------------------
    print("\n[+] BENCHMARK 4: Vrai X-Means BIC + DTW + Génétique Multi-Pop (TrueXmeansTS)")

    # Warm-up pour la JIT Numba (via _dtw_distances -> batch_dtw_cpu)
    true_warm = xm.TrueXmeansTS(data, df_ids, k_min=2, k_max=6, P=4, w=w)
    true_warm.fit()

    t_true_cpu = -1.0
    t_true_gpu = -1.0

    # CPU uniquement (GPU désactivé volontairement pour avoir une baseline propre)
    true_xmeans = xm.TrueXmeansTS(data, df_ids, k_min=2, k_max=6, P=4, w=w)
    t0 = time.perf_counter()
    true_xmeans.fit()
    t_true_cpu = time.perf_counter() - t0
    K_found = len(true_xmeans.centroids_)
    print(f"    - TrueXmeansTS CPU JIT    : {t_true_cpu * 1000:.2f} ms  (K trouvé = {K_found})")

    # GPU (si disponible)
    if BENCHMARK_GPU:
        true_xmeans_gpu = xm.TrueXmeansTS(data, df_ids, k_min=2, k_max=6, P=4, w=w)
        true_xmeans_gpu._init_gpu()
        if true_xmeans_gpu._gpu_ok:
            t0 = time.perf_counter()
            true_xmeans_gpu.fit()
            t_true_gpu = time.perf_counter() - t0
            K_found_gpu = len(true_xmeans_gpu.centroids_)
            print(f"    - TrueXmeansTS GPU GPUsync: {t_true_gpu * 1000:.2f} ms  (K trouvé = {K_found_gpu})")

    # -----------------------------------------------------------------
    # TABLEAU COMPARATIF FINAL DES PERFORMANCES (Console Dashboard)
    # -----------------------------------------------------------------
    print_dashboard(
        t_dtw_py,
        t_dtw_cpu,
        t_dtw_numba_par,
        t_dtw_numba_cuda,
        t_dtw_gpu,
        t_coeff_py,
        t_coeff_cpu,
        t_coeff_gpu,
        t_pipe_py,
        t_pipe_cpu,
        t_pipe_gpu,
        t_true_cpu,
        t_true_gpu,
        len(data),
    )


def print_dashboard(
    t_dtw_py,
    t_dtw_cpu,
    t_dtw_numba_par,
    t_dtw_numba_cuda,
    t_dtw_gpu,
    t_coeff_py,
    t_coeff_cpu,
    t_coeff_gpu,
    t_pipe_py,
    t_pipe_cpu,
    t_pipe_gpu,
    t_true_cpu,
    t_true_gpu,
    n_dataset,
):
    # ANSI color codes
    GRAY = "\033[90m"
    WHITE = "\033[1;37m"
    CYAN = "\033[36m"
    ORANGE = "\033[1;33m"
    GREEN = "\033[32m"
    YELLOW = "\033[93m"
    PURPLE = "\033[35m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    has_gpu = t_dtw_gpu > 0
    has_numba_cuda = t_dtw_numba_cuda > 0
    has_true_gpu = t_true_gpu > 0
    gpu_val_dtw = f"{t_dtw_gpu:.4f} s" if has_gpu else "N/A"
    cuda_val_dtw = f"{t_dtw_numba_cuda:.4f} s" if has_numba_cuda else "N/A"
    gpu_val_coeff = f"{t_coeff_gpu:.4f} s" if has_gpu else "N/A"
    gpu_val_pipe = f"{t_pipe_gpu:.4f} s" if has_gpu else "N/A"
    gpu_val_true = f"{t_true_gpu:.4f} s" if has_true_gpu else "N/A"

    t_tot_py = t_dtw_py + t_pipe_py + t_coeff_py
    t_tot_cpu = t_dtw_cpu + t_pipe_cpu + t_coeff_cpu
    t_tot_gpu = t_dtw_gpu + t_pipe_gpu + t_coeff_gpu if has_gpu else -1.0
    gpu_val_tot = f"{t_tot_gpu:.4f} s" if has_gpu else "N/A"

    # Largeurs de colonnes fixes
    W = [30, 14, 12, 14, 14, 14, 14]  # Composant | Py | JIT | Numba|| | CUDA | GPUsync | Speedup

    def cc(val, col_idx, color=""):
        """Pad val to column width, then wrap with color (outside padding)."""
        padded = f"{str(val):<{W[col_idx]}}"
        return f"{color}{padded}{RESET}" if color else padded

    def row(
        label,
        py,
        jit,
        npar,
        cuda,
        gpu,
        speedup,
        c_py=CYAN,
        c_jit=GREEN,
        c_npar=GREEN,
        c_cuda=BLUE,
        c_gpu=YELLOW,
        c_spd=ORANGE,
    ):
        return (
            f" {cc(label,0)} | {cc(py,1,c_py)} | {cc(jit,2,c_jit)} | "
            f"{cc(npar,3,c_npar)} | {cc(cuda,4,c_cuda)} | {cc(gpu,5,c_gpu)} | {cc(speedup,6,c_spd)}"
        )

    def sp(ref, t):
        return f"{ref/t:.1f}x" if t > 0 else "N/A"

    SEP = "=" * (sum(W) + len(W) * 3 + 2)
    HDR = (
        f" {cc('Composant',0,WHITE)} | {cc('Python Pur',1,WHITE)} | {cc('CPU JIT',2,WHITE)} | "
        f"{cc('Numba || CPU',3,WHITE)} | {cc('Numba CUDA',4,WHITE)} | {cc('GPU GPUsync',5,WHITE)} | "
        f"{cc('Best Speedup',6,WHITE)}"
    )

    print(f"\n{GRAY}{SEP}{RESET}")
    print(f"{'DASHBOARD DE PERFORMANCE COMPARATIVE MULTI-BACKEND':^{len(SEP)}}")
    print(f"{GRAY}{SEP}{RESET}")
    print(HDR)
    print(f"{GRAY}{'-' * len(SEP)}{RESET}")

    # Ligne 1 : DTW batch réel
    best_dtw = min(t for t in [t_dtw_cpu, t_dtw_numba_par, t_dtw_numba_cuda, t_dtw_gpu] if t > 0)
    print(
        row(
            "1k DTW — batch réel",
            f"{t_dtw_py:.4f} s",
            f"{t_dtw_cpu:.4f} s",
            f"{t_dtw_numba_par:.4f} s",
            cuda_val_dtw,
            gpu_val_dtw,
            sp(t_dtw_py, best_dtw),
        )
    )

    # Ligne 2 : XmeansTS Glouton
    best_pipe = min(t for t in [t_pipe_cpu, t_pipe_gpu] if t > 0)
    print(
        row(
            "XmeansTS Glouton (k=3)",
            f"{t_pipe_py:.4f} s",
            f"{t_pipe_cpu:.4f} s",
            "N/A",
            "N/A",
            gpu_val_pipe,
            sp(t_pipe_py, best_pipe),
        )
    )

    # Ligne 3 : defineCoeff
    best_coeff = min(t for t in [t_coeff_cpu, t_coeff_gpu] if t > 0)
    print(
        row(
            "Calcul Coefficients",
            f"{t_coeff_py:.4f} s",
            f"{t_coeff_cpu:.4f} s",
            "N/A",
            "N/A",
            gpu_val_coeff,
            sp(t_coeff_py, best_coeff),
        )
    )

    # Ligne 4 : TrueXmeansTS
    best_true = min(t for t in [t_true_cpu, t_true_gpu] if t > 0) if (t_true_cpu > 0 or t_true_gpu > 0) else -1.0
    print(
        row(
            "TrueXmeansTS BIC (k2-6, P=4)",
            "N/A",
            f"{t_true_cpu:.4f} s",
            "N/A",
            "N/A",
            gpu_val_true,
            sp(t_true_cpu, best_true) if best_true > 0 else "N/A",
            c_py=GRAY,
            c_jit=PURPLE,
        )
    )

    # Ligne 5 : Pipeline complet
    best_tot = min(t for t in [t_tot_cpu, t_tot_gpu] if t > 0)
    print(
        row(
            "Pipeline MMTS Complet",
            f"{t_tot_py:.4f} s",
            f"{t_tot_cpu:.4f} s",
            "N/A",
            "N/A",
            gpu_val_tot,
            sp(t_tot_py, best_tot),
        )
    )

    print(f"{GRAY}{SEP}{RESET}")
    print(f"{GRAY}Note 1 : Python Pur = 150 séries (extrap.). CPU JIT / GPU = {n_dataset} séries complètes.{RESET}")
    print(
        f"{GRAY}Note 2 : 'Numba || CPU' = Numba prange multi-cœur. 'Numba CUDA' = noyau CUDA (requiert driver NVIDIA).{RESET}"
    )
    print(f"{GRAY}Note 3 : TrueXmeansTS = vrai X-Means BIC + DTW + Génétique Multi-Pop P=4, k_max=6.{RESET}")
    print(f"{GRAY}{SEP}{RESET}")


if __name__ == "__main__":
    run_benchmarks()
