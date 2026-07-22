import os
import sys
import math
import numpy as np
from typing import Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.compiler import gpusync_kernel
from gpusync.api.opencl import get_global_id, get_local_id, local_memory, barrier, CLK_LOCAL_MEM_FENCE
from gpusync.memory import AsyncBuffer
from gpusync.scheduler import Scheduler
from gpusync.utils.bg_loop import auto_sync

# Dummy helpers for AST compilation compatibility
def atomic_add_float(target, val): pass
class constant_list(list): pass

def compute_bic(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calcule le score BIC (Bayesian Information Criterion) pour un ensemble de points X,
    des étiquettes de cluster labels et des centroïdes.
    """
    N, D = X.shape
    K = len(centroids)
    if N <= K or K == 0:
        return -1e30

    # Valence (variance globale estimée sous hypothèse sphérique)
    counts = np.bincount(labels, minlength=K)
    if np.any(counts == 0):
        return -1e30

    variance = 0.0
    for k in range(K):
        cluster_points = X[labels == k]
        variance += np.sum((cluster_points - centroids[k]) ** 2)
    
    variance = variance / max(1, N - K)
    if variance <= 1e-12:
        return -1e30

    # Log-likelihood estimée (Kassian & Dan, X-means BIC)
    log_likelihood = 0.0
    for k in range(K):
        Nk = counts[k]
        if Nk > 0:
            log_likelihood += (
                Nk * np.log(Nk)
                - Nk * np.log(N)
                - (Nk * D / 2.0) * np.log(2.0 * np.pi * variance)
                - (Nk - K) / 2.0
            )

    # Nombre de paramètres libres : K * (D + 1) [centroïdes + variance]
    p = K * (D + 1)
    bic = log_likelihood - (p / 2.0) * np.log(N)
    return float(bic)


def kmeans_step_cpu(X: np.ndarray, centroids: np.ndarray, max_iter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exécution CPU (NumPy) d'une itération de K-means.
    """
    N, D = X.shape
    K = len(centroids)
    curr_centroids = centroids.copy()

    for _ in range(max_iter):
        # Distances euclidiennes (N, K)
        dists = np.linalg.norm(X[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(curr_centroids)
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                new_centroids[k] = pts.mean(axis=0)
            else:
                new_centroids[k] = curr_centroids[k]

        if np.allclose(curr_centroids, new_centroids, atol=1e-4):
            break
        curr_centroids = new_centroids

    return labels, curr_centroids


def xmeans_cpu(X: np.ndarray, k_min: int = 2, k_max: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithme X-means standard sur CPU.
    Démarre avec k_min clusters et divise récursivement les clusters si le BIC s'améliore.
    """
    N, D = X.shape
    # Initialisation K-means initiale avec k_min
    initial_idx = np.random.choice(N, size=k_min, replace=False)
    centroids = X[initial_idx].copy()
    labels, centroids = kmeans_step_cpu(X, centroids)

    while len(centroids) < k_max:
        new_centroids_list = []
        new_labels = labels.copy()
        split_occurred = False

        for k in range(len(centroids)):
            cluster_mask = (labels == k)
            cluster_points = X[cluster_mask]
            Nk = len(cluster_points)

            if Nk < 4 or len(centroids) + len(new_centroids_list) >= k_max:
                new_centroids_list.append(centroids[k])
                continue

            # BIC du cluster actuel (parent)
            parent_bic = compute_bic(cluster_points, np.zeros(Nk, dtype=int), centroids[k:k+1])

            # Division du cluster en 2 sous-clusters (enfants)
            child_init_idx = np.random.choice(Nk, size=2, replace=False)
            child_centroids_init = cluster_points[child_init_idx]
            child_labels, child_centroids = kmeans_step_cpu(cluster_points, child_centroids_init)
            child_bic = compute_bic(cluster_points, child_labels, child_centroids)

            if child_bic > parent_bic:
                new_centroids_list.append(child_centroids[0])
                new_centroids_list.append(child_centroids[1])
                split_occurred = True
            else:
                new_centroids_list.append(centroids[k])

        centroids = np.array(new_centroids_list)
        # Recalcul des labels globaux
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        if not split_occurred:
            break

    return labels, centroids


def get_xmeans_assignment_kernel(D: int, K_MAX: int):
    """
    Génère un noyau GPU compilé (gpusync_kernel) pour l'assignation de clusters
    et l'accumulation des variances dans le cadre du X-means multidimensionnel.
    """
    LOCAL_SZ = 256
    COARSE_FACTOR = 4

    @gpusync_kernel(constants={"D_CONST": D, "K_MAX_CONST": K_MAX, "LOCAL_SZ": LOCAL_SZ, "COARSE_FACTOR": COARSE_FACTOR})
    def xmeans_assign_kernel(ctx: int, X: list, centroids: constant_list, labels_out: list, N_arg: int, K_arg: int, stride: int):
        global_idx: int = get_global_id(0)
        ctx_base: int = global_idx * COARSE_FACTOR

        for c in range(0, COARSE_FACTOR):
            ctx_point: int = ctx_base + c
            if ctx_point < N_arg:
                min_dist: float = 1e30
                best_k: int = 0

                for k in range(0, K_arg):
                    dist: float = 0.0
                    for d in range(0, D_CONST):
                        diff: float = X[d * stride + ctx_point] - centroids[k * D_CONST + d]
                        dist += diff * diff

                    if dist < min_dist:
                        min_dist = dist
                        best_k = k

                labels_out[ctx_point] = best_k

    return xmeans_assign_kernel


def get_xmeans_multi_population_kernel(P: int, D: int, K_MAX: int):
    """
    Génère un noyau GPU compilé (gpusync_kernel) évaluant P populations candidates simultanément
    sur l'ensemble des N points en une seule passe GPU.
    """
    LOCAL_SZ = 256
    COARSE_FACTOR = 4

    @gpusync_kernel(constants={"P_CONST": P, "D_CONST": D, "K_MAX_CONST": K_MAX, "LOCAL_SZ": LOCAL_SZ, "COARSE_FACTOR": COARSE_FACTOR})
    def xmeans_multi_pop_kernel(ctx: int, X: list, centroids_p: constant_list, labels_out_p: list, N_arg: int, K_arg: int, stride: int):
        global_idx: int = get_global_id(0)
        ctx_base: int = global_idx * COARSE_FACTOR

        for c in range(0, COARSE_FACTOR):
            ctx_point: int = ctx_base + c
            if ctx_point < N_arg:
                for p in range(0, P_CONST):
                    min_dist: float = 1e30
                    best_k: int = 0

                    for k in range(0, K_arg):
                        dist: float = 0.0
                        for d in range(0, D_CONST):
                            diff: float = X[d * stride + ctx_point] - centroids_p[p * K_MAX_CONST * D_CONST + k * D_CONST + d]
                            dist += diff * diff

                        if dist < min_dist:
                            min_dist = dist
                            best_k = k

                    labels_out_p[p * N_arg + ctx_point] = best_k

    return xmeans_multi_pop_kernel


class XMeansEngine:
    """
    Moteur d'exécution GPU accéléré pour l'algorithme X-Means standard via GPUsync.
    """
    def __init__(self, dim: int, max_k: int = 16, P: int = 8):
        self.dim = dim
        self.max_k = max_k
        self.P = P
        self.kernel = get_xmeans_assignment_kernel(dim, max_k)
        self.multi_pop_kernel = get_xmeans_multi_population_kernel(P, dim, max_k)

    def fit(self, scheduler: Scheduler, X: np.ndarray, k_min: int = 2, k_max: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        N, D = X.shape
        X_soa = np.ascontiguousarray(X.T).ravel().astype(np.float32)
        buf_x = AsyncBuffer.to_device(scheduler, X_soa)

        # Initialisation aléatoire
        initial_idx = np.random.choice(N, size=k_min, replace=False)
        centroids = X[initial_idx].copy().astype(np.float32)

        buf_labels = AsyncBuffer.empty_like(scheduler, np.zeros(N, dtype=np.float32))

        local_sz_val = 256
        num_threads_needed = math.ceil(N / 4)
        global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val

        # Boucle X-means GPU / CPU hybride accélérée
        while len(centroids) < k_max:
            # Transfert des centroïdes vers la VRAM
            buf_c = AsyncBuffer.to_device(scheduler, centroids.flatten())

            # Assignation GPU accélérée
            self.kernel(
                buf_x, buf_c, buf_labels,
                N, len(centroids), N,
                global_size=(global_aligned,), local_size=(local_sz_val,),
                scheduler=scheduler
            )

            labels = buf_labels.to_host().astype(np.int32)

            # Mise à jour des centroïdes et décisions de division (BIC)
            new_centroids_list = []
            split_occurred = False

            for k in range(len(centroids)):
                cluster_mask = (labels == k)
                cluster_points = X[cluster_mask]
                Nk = len(cluster_points)

                if Nk < 4 or len(centroids) + len(new_centroids_list) >= k_max:
                    new_centroids_list.append(centroids[k])
                    continue

                parent_bic = compute_bic(cluster_points, np.zeros(Nk, dtype=int), centroids[k:k+1])

                child_init_idx = np.random.choice(Nk, size=2, replace=False)
                child_centroids_init = cluster_points[child_init_idx]
                child_labels, child_centroids = kmeans_step_cpu(cluster_points, child_centroids_init)
                child_bic = compute_bic(cluster_points, child_labels, child_centroids)

                if child_bic > parent_bic:
                    new_centroids_list.append(child_centroids[0])
                    new_centroids_list.append(child_centroids[1])
                    split_occurred = True
                else:
                    new_centroids_list.append(centroids[k])

            centroids = np.array(new_centroids_list, dtype=np.float32)

            if not split_occurred:
                break

        # Assignation finale GPU
        buf_c = AsyncBuffer.to_device(scheduler, centroids.flatten())
        self.kernel(
            buf_x, buf_c, buf_labels,
            N, len(centroids), N,
            global_size=(global_aligned,), local_size=(local_sz_val,),
            scheduler=scheduler
        )
        final_labels = buf_labels.to_host()

        return final_labels, centroids

    def fit_genetic(self, scheduler: Scheduler, X: np.ndarray, k_min: int = 2, k_max: int = 16, P: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithme X-Means Génétique Multi-Population.
        Évalue P initialisations de sous-clusters candidates en parallèle pour chaque division de cluster
        afin de sélectionner instantanément la division maximisant le BIC et accélérer la convergence.
        """
        N, D = X.shape
        X_soa = np.ascontiguousarray(X.T).ravel().astype(np.float32)
        buf_x = AsyncBuffer.to_device(scheduler, X_soa)

        initial_idx = np.random.choice(N, size=k_min, replace=False)
        centroids = X[initial_idx].copy().astype(np.float32)

        buf_labels = AsyncBuffer.empty_like(scheduler, np.zeros(N, dtype=np.float32))

        local_sz_val = 256
        num_threads_needed = math.ceil(N / 4)
        global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val

        # Boucle X-means GPU / CPU hybride accélérée avec Génétique multi-populations
        while len(centroids) < k_max:
            buf_c = AsyncBuffer.to_device(scheduler, centroids.flatten())

            self.kernel(
                buf_x, buf_c, buf_labels,
                N, len(centroids), N,
                global_size=(global_aligned,), local_size=(local_sz_val,),
                scheduler=scheduler
            )

            labels = buf_labels.to_host().astype(np.int32)

            new_centroids_list = []
            split_occurred = False

            for k in range(len(centroids)):
                cluster_mask = (labels == k)
                cluster_points = X[cluster_mask]
                Nk = len(cluster_points)

                if Nk < 4 or len(centroids) + len(new_centroids_list) >= k_max:
                    new_centroids_list.append(centroids[k])
                    continue

                parent_bic = compute_bic(cluster_points, np.zeros(Nk, dtype=int), centroids[k:k+1])

                # Routage Hybride Dynamique : CPU si la taille du cluster est petite pour éviter la latence PCIe
                if Nk >= 500:
                    P_child_centroids = np.zeros((P, self.max_k, D), dtype=np.float32)
                    for p_idx in range(P):
                        child_init_idx = np.random.choice(Nk, size=2, replace=False)
                        P_child_centroids[p_idx, :2] = cluster_points[child_init_idx]

                    pts_soa = np.ascontiguousarray(cluster_points.T).ravel().astype(np.float32)
                    buf_pts = AsyncBuffer.to_device(scheduler, pts_soa)
                    buf_p_centroids = AsyncBuffer.to_device(scheduler, P_child_centroids.flatten())
                    buf_p_labels = AsyncBuffer.empty_like(scheduler, np.zeros(P * Nk, dtype=np.float32))

                    pts_aligned = math.ceil(math.ceil(Nk / 4) / local_sz_val) * local_sz_val
                    self.multi_pop_kernel(
                        buf_pts, buf_p_centroids, buf_p_labels,
                        Nk, 2, Nk,
                        global_size=(max(local_sz_val, pts_aligned),), local_size=(local_sz_val,),
                        scheduler=scheduler
                    )
                    p_labels_all = buf_p_labels.to_host().astype(np.int32).reshape(P, Nk)

                    best_child_bic = -1e30
                    best_child_centroids = None

                    for p_idx in range(P):
                        p_labs = p_labels_all[p_idx]
                        p_c0 = cluster_points[p_labs == 0].mean(axis=0) if np.sum(p_labs == 0) > 0 else P_child_centroids[p_idx, 0]
                        p_c1 = cluster_points[p_labs == 1].mean(axis=0) if np.sum(p_labs == 1) > 0 else P_child_centroids[p_idx, 1]
                        p_c = np.array([p_c0, p_c1])

                        child_bic = compute_bic(cluster_points, p_labs, p_c)
                        if child_bic > best_child_bic:
                            best_child_bic = child_bic
                            best_child_centroids = p_c
                else:
                    # Évaluation rapide sur CPU (2 itérations max)
                    best_child_bic = -1e30
                    best_child_centroids = None
                    for _ in range(P):
                        child_init_idx = np.random.choice(Nk, size=2, replace=False)
                        child_centroids_init = cluster_points[child_init_idx]
                        child_labels, child_centroids = kmeans_step_cpu(cluster_points, child_centroids_init, max_iter=2)
                        child_bic = compute_bic(cluster_points, child_labels, child_centroids)

                        if child_bic > best_child_bic:
                            best_child_bic = child_bic
                            best_child_centroids = child_centroids

                if best_child_centroids is not None and best_child_bic > parent_bic:
                    new_centroids_list.append(best_child_centroids[0])
                    new_centroids_list.append(best_child_centroids[1])
                    split_occurred = True
                else:
                    new_centroids_list.append(centroids[k])

            centroids = np.array(new_centroids_list, dtype=np.float32)

            if not split_occurred:
                break

        buf_c = AsyncBuffer.to_device(scheduler, centroids.flatten())
        self.kernel(
            buf_x, buf_c, buf_labels,
            N, len(centroids), N,
            global_size=(global_aligned,), local_size=(local_sz_val,),
            scheduler=scheduler
        )
        final_labels = buf_labels.to_host()

        return final_labels, centroids

