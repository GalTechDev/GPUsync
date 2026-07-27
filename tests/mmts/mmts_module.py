# -*- coding: utf-8 -*-
"""
Module d'analyse pour le clustering de séries temporelles avec X-means et DTW
"""

import warnings

warnings.filterwarnings("ignore")
import os
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from tslearn.barycenters import euclidean_barycenter

# Configuration pour OpenCL si disponible (pour la performance)
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_CTX"] = "0"

# Import d'opencldtw si disponible (sinon on utilisera la version pure Python)
try:
    import opencldtw

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

# Importation de la configuration
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import sys
import time

from numba import njit

from config import *
from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

_gpusync_sched = None
_gpusync_failed = False
_gpusync_kernels_cache = {}  # Cache par (M, num_refs)


def get_gpusync_resources(M, num_refs):
    global _gpusync_sched, _gpusync_failed, _gpusync_kernels_cache
    if _gpusync_failed:
        raise RuntimeError("GPUsync non disponible (précédent échec).")

    if _gpusync_sched is None:
        try:
            gpus = list_gpus()
            if gpus:
                _gpusync_sched = Scheduler(gpu=gpus[0])
                _gpusync_sched.initialize()
            else:
                _gpusync_failed = True
                raise RuntimeError("Aucun GPU compatible GPUsync n'est disponible.")
        except Exception:
            _gpusync_failed = True
            raise

    key = (M, num_refs)
    if key not in _gpusync_kernels_cache:
        _gpusync_kernels_cache[key] = get_batch_dtw_kernel(M, num_refs)

    return _gpusync_sched, _gpusync_kernels_cache[key]


def centeroidnp(arr):
    """
    Calcule le centroide d'un tableau numpy

    Args:
        arr: Tableau numpy de forme (length, dim)

    Returns:
        Numpy array représentant le centroide
    """
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i]) / length for i in range(dim)])


@njit(fastmath=True, cache=True)
def DTWDistance_cpu(s1, s2, w):
    n1 = len(s1)
    n2 = len(s2)
    w = max(w, abs(n1 - n2))

    dtw = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n1 + 1):
        start = max(1, i - w)
        end = min(n2, i + w) + 1
        for j in range(start, end):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return np.sqrt(dtw[n1, n2])


USE_GPU = False  # Mettre à True pour forcer l'utilisation du GPU OpenCL via GPUsync


def DTWDistance(s1, s2, w):
    """
    Calcule la distance DTW accélérée sur GPU via GPUsync (si USE_GPU=True) avec repli CPU Numba JIT.
    """
    if USE_GPU:
        try:
            M = max(len(s1), len(s2))
            M_pad = int(2 ** math.ceil(math.log2(M)))
            M_pad = max(32, M_pad)

            sched, kernel = get_gpusync_resources(M_pad, 1)

            s1_pad = np.zeros(M_pad, dtype=np.float32)
            s1_pad[: len(s1)] = s1

            s2_pad = np.zeros(M_pad, dtype=np.float32)
            s2_pad[: len(s2)] = s2

            buf_ref = AsyncBuffer.to_device(sched, s1_pad)
            buf_tgt = AsyncBuffer.to_device(sched, s2_pad)
            buf_out = AsyncBuffer.empty_like(sched, np.zeros(1, dtype=np.float32))

            kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, 1, 1, global_size=(1,), scheduler=sched)
            return float(buf_out.to_host()[0])
        except Exception:
            pass
    return DTWDistance_cpu(s1, s2, w)


@njit(fastmath=True, cache=True)
def batch_dtw_cpu(refs, targets, w):
    K = refs.shape[0]
    N = targets.shape[0]
    M = refs.shape[1]

    distances = np.zeros((K, N), dtype=np.float32)

    for i in range(N):
        tgt = targets[i]
        for c in range(K):
            ref = refs[c]
            n_w = max(w, 0)
            dtw = np.full((M + 1, M + 1), np.inf, dtype=np.float32)
            dtw[0, 0] = 0.0
            for r_idx in range(1, M + 1):
                start = max(1, r_idx - n_w)
                end = min(M, r_idx + n_w) + 1
                for t_idx in range(start, end):
                    cost = (ref[r_idx - 1] - tgt[t_idx - 1]) ** 2
                    dtw[r_idx, t_idx] = cost + min(
                        dtw[r_idx - 1, t_idx], dtw[r_idx, t_idx - 1], dtw[r_idx - 1, t_idx - 1]
                    )
            distances[c, i] = np.sqrt(dtw[M, M])

    return distances


class XmeansTS:
    """
    Classe pour l'algorithme X-means avec DTW pour le clustering de séries temporelles
    """

    def __init__(self, data, numclust, file, listID, coeffRap):
        """
        Initialisation de la classe XmeansTS

        Args:
            data: Données d'entrée
            numclust: Nombre initial de clusters
            file: Chemin du fichier
            listID: DataFrame contenant les identifiants
            coeffRap: Coefficient de rapport
        """
        self.assignments = {}
        self.dfListID = listID
        self.data = data
        self.numclust = numclust
        self.centroids = {}
        self.file = file
        self.coeffRapport = coeffRap

    def reductMemory(self):
        """Réduit l'empreinte mémoire"""
        # À implémenter si nécessaire
        pass

    def kmeansConverg(self, data, centroids, listID, w):
        """
        Vérifie la convergence de l'algorithme k-means

        Args:
            data: Données
            centroids: Centroides actuels
            listID: Liste des IDs
            w: Paramètre de fenêtre pour DTW

        Returns:
            True si la convergence est atteinte, False sinon
        """
        sum_dist = 0
        for i in range(len(data)):
            min_dist = float("inf")
            for c in centroids:
                if len(data[i]) > 0:
                    if w > 0:
                        dist = DTWDistance(data[i], centroids[c], w)
                    else:
                        dist = np.linalg.norm(data[i] - centroids[c])

                    if dist < min_dist:
                        min_dist = dist
            sum_dist += min_dist
        return sum_dist < 0.001

    def updateCentroide(self, data, centroids, assignments):
        """
        Met à jour les centroides basés sur les assignations

        Args:
            data: Données
            centroids: Centroides actuels
            assignments: Assignations actuelles des points aux clusters

        Returns:
            Centroides mis à jour
        """
        for c in assignments:
            if len(assignments[c]) > 0:
                new_centroid = np.zeros(len(data[0]))
                for i in assignments[c]:
                    if len(data[i]) > 0:
                        new_centroid += data[i]
                centroids[c] = new_centroid / len(assignments[c])
        return centroids

    def computeDist(self, data, listID, centroids, assignments, w):
        """
        Calcule les distances entre points et centroides (accélération GPU batch via GPUsync)
        """
        for c in assignments:
            assignments[c] = []

        try:
            if not USE_GPU:
                raise RuntimeError("GPU désactivé")
            N = len(data)
            K = len(centroids)
            if N > 0 and K > 0:
                M = len(data[0])
                M_pad = int(2 ** math.ceil(math.log2(M)))
                M_pad = max(32, M_pad)

                sched, kernel = get_gpusync_resources(M_pad, K)

                # Préparation du tableau des cibles (targets)
                targets_pad = np.zeros((N, M_pad), dtype=np.float32)
                for i in range(N):
                    targets_pad[i, : len(data[i])] = data[i]

                # Préparation du tableau des références (centroïdes)
                refs_pad = np.zeros((K, M_pad), dtype=np.float32)
                keys_list = list(centroids.keys())
                for c_idx, c_key in enumerate(keys_list):
                    c_val = centroids[c_key]
                    refs_pad[c_idx, : len(c_val)] = c_val

                # Upload en VRAM
                buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
                buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
                buf_out = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))

                # Calcul de toutes les distances en un seul appel GPU
                kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, K, N, global_size=(N,), scheduler=sched)

                # Rapatriement de la matrice de distances (K, N)
                dist_matrix = buf_out.to_host().reshape(K, N)

                # Assignations
                for i in range(N):
                    min_idx = np.argmin(dist_matrix[:, i])
                    closest_clust = keys_list[min_idx]
                    assignments[closest_clust].append(i)
                return assignments
        except Exception:
            pass

        # Fallback CPU JIT Numba parallélisé
        try:
            N = len(data)
            K = len(centroids)
            if N > 0 and K > 0:
                M = len(data[0])
                targets_pad = np.zeros((N, M), dtype=np.float32)
                for i in range(N):
                    targets_pad[i, : len(data[i])] = data[i]

                refs_pad = np.zeros((K, M), dtype=np.float32)
                keys_list = list(centroids.keys())
                for c_idx, c_key in enumerate(keys_list):
                    c_val = centroids[c_key]
                    refs_pad[c_idx, : len(c_val)] = c_val

                dist_matrix = batch_dtw_cpu(refs_pad, targets_pad, w)

                for i in range(N):
                    min_idx = np.argmin(dist_matrix[:, i])
                    closest_clust = keys_list[min_idx]
                    assignments[closest_clust].append(i)
                return assignments
        except Exception:
            pass

        return assignments

    def initAtt(self, data, centroids, listID):
        """
        Initialise les assignations

        Args:
            data: Données
            centroids: Centroides initiaux
            listID: Liste des IDs

        Returns:
            Assignations initiales
        """
        assignments = {}
        for i in range(len(centroids)):
            assignments[i] = []

        return assignments

    def kmeans(self, data, listDataFrame, w):
        """
        Exécute l'algorithme k-means avec distance DTW

        Args:
            data: Données
            listDataFrame: DataFrame des IDs
            w: Paramètre de fenêtre pour DTW

        Returns:
            Centroides et assignations finaux
        """
        centroids = {}
        for i in range(self.numclust):
            centroids[i] = data[i]

        assignments = self.initAtt(data, centroids, listDataFrame)

        testStop = False
        counter = 0
        while not testStop and counter < MAX_ITERATIONS:
            assignments = self.computeDist(data, listDataFrame, centroids, assignments, w)
            centroids = self.updateCentroide(data, centroids, assignments)
            testStop = self.kmeansConverg(data, centroids, listDataFrame, w)
            counter += 1

        return centroids, assignments

    def selectSerie(
        self, data, listID, assignments, assignmentsTemp, assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim
    ):
        """
        Sélectionne les séries pour les clusters
        """
        for i in assignments:
            if len(assignments[i]) >= nbInstanceMinim:
                ids = []
                for item in assignments[i]:
                    ids.append(int(listID.iloc[item][ID_COLUMN]))
                assignmentsTemp[i] = ids
                assignmentsTempBary[i] = []
                assignmentsTempBaryDTW[i] = []

                for item in assignments[i]:
                    assignmentsTempBary[i].append(data[item])

    def updateAssignment(self, assignmentsTemp, listeIDOk, nbClustTempp):
        """
        Met à jour les assignations
        """
        id_array = self.dfListID[ID_COLUMN].values

        for key in assignmentsTemp:
            self.assignments[nbClustTempp] = assignmentsTemp[key]

            if len(assignmentsTemp[key]) > 0:
                listeIDOk.extend(assignmentsTemp[key])

                mask = np.isin(id_array, assignmentsTemp[key])
                indices = np.where(mask)[0]
                values = [self.data[i] for i in indices]

                if values:
                    np_values = np.array(values)
                    if np_values.ndim == 2:
                        np_values = np.expand_dims(np_values, axis=-1)
                    bar = euclidean_barycenter(np_values)
                    flat_list = [item for sublist in bar for item in sublist]
                else:
                    flat_list = []

                self.centroids[nbClustTempp] = flat_list
                nbClustTempp += 1
        return listeIDOk

    def updateVar(self, listFilterID, colID=ID_COLUMN, ranges=None):
        """
        Met à jour les variables
        """
        mask = ~self.dfListID[colID].isin(listFilterID)
        dfListIDTemp = self.dfListID[mask]

        mask_vals = mask.values
        data = [self.data[i] for i in range(len(self.data)) if mask_vals[i]]

        return data, dfListIDTemp

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
        w=DEFAULT_W,
    ):
        """
        Algorithme principal de clustering X-means avec DTW

        Args:
            data: Données
            num_clust: Nombre de clusters
            num_iter: Nombre d'itérations
            listDataFrame: DataFrame des IDs
            LastnbClust: Dernier nombre de clusters
            tabEnviIndice: Index de l'environnement
            distCompare: Dictionnaire pour comparaison de distances
            listeIDOk1: Liste des IDs valides
            nbInstanceMinim: Nombre minimum d'instances par cluster
            w: Paramètre de fenêtre pour DTW

        Returns:
            Résultat du clustering
        """
        # Si le nombre de clusters est 0, retourner les données
        if num_clust == 0:
            return pd.DataFrame(self.dfListID)

        # Si on a assez de données
        # NOTE: Correction par rapport à l'original qui n'avait pas LastnbClust != 0 ici
        if len(data) > num_clust * 2:
            centroids = {}
            distCompare[tabEnviIndice] = {}

            # Initialiser les centroides aléatoirement
            for i in range(num_clust):
                idx = random.randint(0, len(data) - 1)
                centroids[i] = data[idx]

            assignments = {}
            for i in range(num_clust):
                assignments[i] = []

            testStop = False
            counter = 0

            # K-means principal
            while not testStop and counter < 1:  # Limité à 1 itération pour performance
                assignments = {}
                assignmentsTemp = {}
                assignmentsTempBary = {}
                assignmentsTempBaryDTW = {}

                for i in range(num_clust):
                    assignments[i] = []
                    assignmentsTemp[i] = []
                    assignmentsTempBary[i] = []
                    assignmentsTempBaryDTW[i] = []

                # Calculer les distances et assigner les points via computeDist (GPU/JIT)
                assignments = self.computeDist(data, listDataFrame, centroids, assignments, w)

                # Mettre à jour les centroides
                for c in assignments:
                    if len(assignments[c]) > 0:
                        new_centroid = np.zeros(len(data[0]))
                        for i in assignments[c]:
                            if len(data[i]) > 0:
                                new_centroid += data[i]
                        centroids[c] = new_centroid / len(assignments[c])

                testStop = True  # On sort après une itération
                counter += 1

            # Sélectionner les séries pour chaque cluster
            self.selectSerie(
                data,
                listDataFrame,
                assignments,
                assignmentsTemp,
                assignmentsTempBary,
                assignmentsTempBaryDTW,
                nbInstanceMinim,
            )

            nbClustLow = LastnbClust
            listeIDOk = []

            # Mettre à jour les assignations
            listeIDOk = self.updateAssignment(assignmentsTemp, listeIDOk, LastnbClust)

            listeIDOk1 = listeIDOk1 + listeIDOk
            updateVar = self.updateVar(listeIDOk1)

            # Appel récursif avec les données mises à jour si la taille du dataset restant a strictement diminué
            if len(updateVar[0]) > 0 and len(updateVar[0]) < len(data):
                self.k_means_clust(
                    updateVar[0],
                    num_clust,
                    num_iter,
                    updateVar[1],
                    LastnbClust + len(assignmentsTemp),
                    tabEnviIndice,
                    distCompare,
                    listeIDOk1,
                    nbInstanceMinim,
                    w,
                )

            if nbClustLow == 0:
                return self.getLastMat(tabEnviIndice, w)

        elif LastnbClust == 0:
            return self.getLastMat(tabEnviIndice, w)

        # Si on n'a pas assez de données, retourner les données actuelles
        if len(data) < num_clust * 2 or LastnbClust == 0:
            dfListIDTemp = pd.DataFrame(listDataFrame)
            dfListIDTemp["cluster"] = -1

            for ctr in range(len(self.assignments)):
                for elemid in self.assignments[ctr]:
                    dfListIDTemp.loc[dfListIDTemp[ID_COLUMN] == elemid, "cluster"] = ctr

            return dfListIDTemp

    def getLastMat(self, tabEnviIndice, w):
        """
        Récupère la matrice finale des résultats - Conforme à l'original (accélérée sur GPU via GPUsync)
        """
        MatriceDist = pd.DataFrame()

        try:
            if not USE_GPU:
                raise RuntimeError("GPU désactivé")
            N = len(self.data)
            K = len(self.centroids)
            if N > 0 and K > 0:
                M = len(self.data[0])
                M_pad = int(2 ** math.ceil(math.log2(M)))
                M_pad = max(32, M_pad)

                sched, kernel = get_gpusync_resources(M_pad, K)

                targets_pad = np.zeros((N, M_pad), dtype=np.float32)
                for i in range(N):
                    targets_pad[i, : len(self.data[i])] = self.data[i]

                refs_pad = np.zeros((K, M_pad), dtype=np.float32)
                keys_list = list(self.centroids.keys())
                for c_idx, c_key in enumerate(keys_list):
                    c_val = self.centroids[c_key]
                    refs_pad[c_idx, : len(c_val)] = c_val

                buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
                buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
                buf_out = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))

                kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, K, N, global_size=(N,), scheduler=sched)

                dist_matrix = buf_out.to_host().reshape(K, N)

                for c_idx, c_key in enumerate(keys_list):
                    MatriceDist[str(c_key) + "_" + tabEnviIndice] = dist_matrix[c_idx, :]

                MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
                return MatriceDist
        except Exception:
            pass

        # Fallback CPU JIT Numba parallélisé
        try:
            N = len(self.data)
            K = len(self.centroids)
            if N > 0 and K > 0:
                M = len(self.data[0])
                targets_pad = np.zeros((N, M), dtype=np.float32)
                for i in range(N):
                    targets_pad[i, : len(self.data[i])] = self.data[i]

                refs_pad = np.zeros((K, M), dtype=np.float32)
                keys_list = list(self.centroids.keys())
                for c_idx, c_key in enumerate(keys_list):
                    c_val = self.centroids[c_key]
                    refs_pad[c_idx, : len(c_val)] = c_val

                dist_matrix = batch_dtw_cpu(refs_pad, targets_pad, w)

                for c_idx, c_key in enumerate(keys_list):
                    MatriceDist[str(c_key) + "_" + tabEnviIndice] = dist_matrix[c_idx, :]

                MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
                return MatriceDist
        except Exception:
            import traceback

            traceback.print_exc()
            # Fallback boucle classique si erreur
            for uit in self.centroids.keys():
                listTem = []
                for ine in range(len(self.data)):
                    listTem.append(DTWDistance_cpu(self.centroids[uit], self.data[ine], w))
                MatriceDist[str(uit) + "_" + tabEnviIndice] = listTem

        MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
        return MatriceDist

    def defineCoeff(self, listID, data, num_clust, nbInstanceMinim, w=1):
        """
        Définit les coefficients pour l'analyse

        Args:
            listID: Liste des IDs
            data: Données
            num_clust: Nombre de clusters
            nbInstanceMinim: Nombre minimum d'instances par cluster
            w: Paramètre de fenêtre pour DTW

        Returns:
            Coefficients calculés
        """
        # À implémenter si nécessaire
        return [1.0]  # Valeur par défaut


# ==============================================================================
# TrueXmeansTS — Vrai algorithme X-Means avec BIC + DTW + Génétique Multi-Pop
# ==============================================================================


def _compute_bic_dtw(dist_matrix_row: np.ndarray, labels: np.ndarray, K: int) -> float:
    """
    Calcule un score BIC adapté pour le clustering DTW.
    dist_matrix_row : tableau (K, N) de distances DTW^2.
    labels : tableau (N,) d'assignations.
    K : nombre de clusters.
    """
    N = len(labels)
    if N <= K or K == 0:
        return -1e30

    counts = np.bincount(labels, minlength=K)
    if np.any(counts == 0):
        return -1e30

    # Variance intra-cluster basée sur les distances DTW
    inertia = 0.0
    for k in range(K):
        mask = labels == k
        if counts[k] > 0:
            inertia += np.sum(dist_matrix_row[k, mask] ** 2)

    variance = inertia / max(1, N - K)
    if variance <= 1e-12:
        return -1e30

    log_likelihood = 0.0
    for k in range(K):
        Nk = counts[k]
        if Nk > 0:
            log_likelihood += (
                Nk * np.log(Nk) - Nk * np.log(N) - (Nk / 2.0) * np.log(2.0 * np.pi * variance) - (Nk - K) / 2.0
            )

    # p = K * 2 (centroïde 1D + variance par cluster)
    p = K * 2
    bic = log_likelihood - (p / 2.0) * np.log(N)
    return float(bic)


def _batch_dtw_distances(
    cluster_data: np.ndarray, centroids: np.ndarray, w: int, M_pad: int, sched=None, dtw_kernel=None
) -> np.ndarray:
    """
    Retourne une matrice de distances DTW de forme (K, N) entre K centroïdes et N séries.
    Utilise le GPU si sched et dtw_kernel sont fournis, sinon repli sur CPU JIT.
    """
    N = len(cluster_data)
    K = len(centroids)

    if sched is not None and dtw_kernel is not None:
        try:
            targets_pad = np.zeros((N, M_pad), dtype=np.float32)
            for i in range(N):
                s = cluster_data[i]
                targets_pad[i, : len(s)] = s

            refs_pad = np.zeros((K, M_pad), dtype=np.float32)
            for c_idx, c_val in enumerate(centroids):
                refs_pad[c_idx, : len(c_val)] = c_val

            buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
            buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
            buf_out = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))

            dtw_kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, K, N, global_size=(N,), scheduler=sched)
            return buf_out.to_host().reshape(K, N)
        except Exception:
            pass

    # Repli CPU Numba JIT
    M = len(centroids[0])
    refs_pad = np.zeros((K, M), dtype=np.float32)
    for c_idx, c_val in enumerate(centroids):
        refs_pad[c_idx, : len(c_val)] = c_val
    targets_pad = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        s = cluster_data[i]
        targets_pad[i, : len(s)] = s
    return batch_dtw_cpu(refs_pad, targets_pad, w)


class TrueXmeansTS:
    """
    Vrai algorithme X-Means pour le clustering de séries temporelles avec DTW.

    Différence clé avec XmeansTS (glouton) :
      - Démarre avec k_min clusters et divise chaque cluster en 2 sous-clusters
        uniquement si le score BIC (basé sur DTW) s'améliore.
      - Évalue P populations candidates en parallèle (stratégie génétique) pour
        trouver instantanément la meilleure division sans itérations excessives.

    Args:
        data     : liste de séries temporelles (numpy arrays).
        listID   : DataFrame contenant la colonne ID_COLUMN.
        k_min    : nombre initial de clusters.
        k_max    : nombre maximum de clusters autorisés.
        P        : nombre de populations candidates (génétique multi-pop).
        w        : fenêtre de contrainte DTW (Sakoe-Chiba).
    """

    def __init__(self, data, listID, k_min: int = 2, k_max: int = 16, P: int = 8, w: int = DEFAULT_W):
        self.data = data
        self.dfListID = listID
        self.k_min = k_min
        self.k_max = k_max
        self.P = P
        self.w = w

        self.labels_ = None
        self.centroids_ = None

        # Pré-calcul de la longueur de série et de M_pad pour le GPU
        self._M = len(data[0]) if data else 32
        self._M_pad = int(2 ** math.ceil(math.log2(max(self._M, 1))))
        self._M_pad = max(32, self._M_pad)

        # Ressources GPU (lazy init)
        self._sched = None
        self._dtw_kernel = None
        self._gpu_ok = False

    def _init_gpu(self):
        """Initialisation paresseuse des ressources GPU."""
        if not self._gpu_ok:
            try:
                self._sched, self._dtw_kernel = get_gpusync_resources(self._M_pad, self.k_max)
                self._gpu_ok = True
            except Exception:
                self._gpu_ok = False

    def _dtw_distances(self, cluster_data, centroids) -> np.ndarray:
        """Calcule la matrice (K, Nc) de distances DTW."""
        return _batch_dtw_distances(
            cluster_data,
            centroids,
            self.w,
            self._M_pad,
            self._sched if self._gpu_ok else None,
            self._dtw_kernel if self._gpu_ok else None,
        )

    def _assign(self, cluster_data, centroids) -> np.ndarray:
        """Assigne chaque point au centroïde DTW le plus proche."""
        dist_mat = self._dtw_distances(cluster_data, centroids)  # (K, N)
        return np.argmin(dist_mat, axis=0)  # (N,)

    def _update_centroids(self, cluster_data, labels, K) -> np.ndarray:
        """Recalcule les centroïdes comme moyenne Euclidienne des membres de chaque cluster."""
        M = self._M
        centroids = np.zeros((K, M), dtype=np.float32)
        for k in range(K):
            pts = [cluster_data[i] for i in range(len(cluster_data)) if labels[i] == k]
            if pts:
                centroids[k] = np.mean(np.array(pts, dtype=np.float32), axis=0)
        return centroids

    def _kmeans_dtw(self, cluster_data, init_centroids, max_iter: int = 3):
        """
        K-Means avec distance DTW (itérations limitées pour la vitesse).
        Retourne (labels, centroids).
        """
        centroids = np.array([np.array(c, dtype=np.float32) for c in init_centroids])
        K = len(centroids)
        for _ in range(max_iter):
            labels = self._assign(cluster_data, centroids)
            new_centroids = self._update_centroids(cluster_data, labels, K)
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        return labels, centroids

    def _bic_for_split(self, cluster_data, labels, centroids) -> float:
        """Calcule le BIC DTW pour une partition donnée."""
        dist_mat = self._dtw_distances(cluster_data, centroids)
        return _compute_bic_dtw(dist_mat, labels, len(centroids))

    def _evaluate_split_genetic(self, cluster_data):
        """
        Évalue P divisions candidates (génétique multi-pop) et retourne
        (best_labels, best_centroids, best_bic).
        """
        Nk = len(cluster_data)
        best_bic = -1e30
        best_labels = None
        best_centroids = None

        for _ in range(self.P):
            init_idx = np.random.choice(Nk, size=2, replace=False)
            init_c = [cluster_data[i] for i in init_idx]
            labels, centroids = self._kmeans_dtw(cluster_data, init_c, max_iter=2)
            bic = self._bic_for_split(cluster_data, labels, centroids)
            if bic > best_bic:
                best_bic = bic
                best_labels = labels
                best_centroids = centroids

        return best_labels, best_centroids, best_bic

    def fit(self):
        """
        Lance le vrai algorithme X-Means BIC avec DTW et sélection génétique.
        Remplit self.labels_ et self.centroids_.

        Returns:
            self (pour chaînage).
        """
        self._init_gpu()

        data = self.data
        N = len(data)

        # Initialisation K-Means avec k_min centroïdes
        init_idx = np.random.choice(N, size=self.k_min, replace=False)
        init_c = [data[i] for i in init_idx]
        labels, centroids = self._kmeans_dtw(data, init_c, max_iter=5)

        while len(centroids) < self.k_max:
            new_centroids_list = list(centroids)
            pending_replacements = {}  # index_in_new -> (c0, c1)
            split_occurred = False

            for k in range(len(centroids)):
                cluster_idx = [i for i in range(N) if labels[i] == k]
                Nk = len(cluster_idx)
                if (
                    Nk < 4
                    or len(new_centroids_list) - len(pending_replacements) + len(pending_replacements) * 2 >= self.k_max
                ):
                    continue

                cluster_data = [data[i] for i in cluster_idx]

                # BIC du parent (cluster entier avec 1 centroïde)
                parent_bic = self._bic_for_split(cluster_data, np.zeros(Nk, dtype=np.int32), centroids[k : k + 1])

                # Évaluation des P divisions candidates (génétique)
                child_labels, child_centroids, child_bic = self._evaluate_split_genetic(cluster_data)

                if child_bic > parent_bic and child_centroids is not None:
                    pending_replacements[k] = (child_centroids[0], child_centroids[1])
                    split_occurred = True

            if not split_occurred:
                break

            # Reconstruction de la liste des centroïdes en intégrant les divisions
            rebuilt = []
            for k, c in enumerate(centroids):
                if k in pending_replacements:
                    rebuilt.append(pending_replacements[k][0])
                    rebuilt.append(pending_replacements[k][1])
                else:
                    rebuilt.append(c)
                if len(rebuilt) >= self.k_max:
                    rebuilt = rebuilt[: self.k_max]
                    break

            centroids = np.array(rebuilt, dtype=np.float32)

            # Réassignation globale
            labels = self._assign(data, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        return self

    def get_result_dataframe(self, tabEnviIndice: str = "0") -> pd.DataFrame:
        """
        Retourne un DataFrame similaire à XmeansTS.getLastMat :
        une colonne de distances DTW par cluster + la colonne ID.
        """
        if self.labels_ is None:
            raise RuntimeError("Appelez fit() avant get_result_dataframe().")

        K = len(self.centroids_)
        dist_mat = self._dtw_distances(self.data, list(self.centroids_))  # (K, N)

        result = pd.DataFrame()
        for k in range(K):
            result[f"{k}_{tabEnviIndice}"] = dist_mat[k]

        result[ID_COLUMN] = self.dfListID[ID_COLUMN].values
        result["cluster"] = self.labels_
        return result


class MultiXmeansTS:
    """
    Classe pour l'analyse multi-variable avec X-means
    """

    def __init__(self):
        """Initialisation de la classe MultiXmeansTS"""
        self.XmeansTSVars = [XmeansTS]
        self.centroids = {}
        self.assignmentsMulti = {}
        self.files = []

    def KmeanMutiShape(self, nbClust, columns, dft, dicoCoff, assignmentsOk, indiceSd, w=DEFAULT_W):
        """
        Analyse multi-variable avec K-means

        Args:
            nbClust: Nombre de clusters
            columns: Colonnes à utiliser
            dft: DataFrame d'entrée
            dicoCoff: Dictionnaire des coefficients
            assignmentsOk: Dictionnaire des assignations
            indiceSd: Index de suivi
            w: Paramètre de fenêtre pour DTW

        Returns:
            Résultat de l'analyse
        """
        if len(dft) > nbClust * 2:
            dftmp = dft.copy()
            kmeans = KMeans(n_clusters=nbClust)
            y = kmeans.fit(dftmp[list(columns)].values)
            dftmp["Cluster"] = y.labels_

            for clust in dftmp["Cluster"].unique():
                retireId = True
                dftTemp = dftmp[dftmp["Cluster"] == clust]
                IdTemp = list(dftTemp[ID_COLUMN])

                for tabEnviIndice in self.files:
                    # Construire le chemin du fichier
                    if hasattr(self, "data_directory") and hasattr(self, "file_prefix"):
                        # Utiliser les chemins fournis par l'app Flask
                        file_path = os.path.join(self.data_directory, f"{self.file_prefix}{tabEnviIndice}.xlsx")
                        possible_paths = [file_path]
                    else:
                        # Fallback pour compatibilité avec l'original
                        file_name = tabEnviIndice
                        if file_name.startswith("Set_"):
                            file_name = file_name[4:]

                        possible_paths = [
                            os.path.join(os.path.dirname(os.getcwd()), f"RomaneDataSet_{file_name}.xlsx"),
                            os.path.join(os.getcwd(), f"RomaneDataSet_{file_name}.xlsx"),
                            ROMANE_DATA_PATH + tabEnviIndice + ".xlsx",
                            ROMANE_DATA_PATH + file_name + ".xlsx",
                        ]

                    # Chercher un fichier qui existe
                    file_path = None
                    for path in possible_paths:
                        print(f"Tentative avec le chemin: {path}")
                        if os.path.exists(path):
                            file_path = path
                            break

                    if file_path is None:
                        # Si aucun chemin ne fonctionne, afficher les chemins essayés et lever une exception
                        paths_str = "\n".join(possible_paths)
                        raise FileNotFoundError(
                            f"Impossible de trouver le fichier pour {tabEnviIndice}. Chemins essayés:\n{paths_str}"
                        )

                    print(f"Chargement du fichier: {file_path}")
                    xlsQualEnvi = pd.ExcelFile(file_path)

                    dataTemp = xlsQualEnvi.parse(0)
                    dataTemp = dataTemp[dataTemp[ID_COLUMN].isin(IdTemp)]
                    dataTemp1 = dataTemp.drop([ID_COLUMN], axis=1)

                    listpointCentr = []
                    listpoint = []
                    valuesMat = dataTemp1.values

                    for k in valuesMat:
                        clust_sum = 0
                        for h in range(len(k)):
                            clust_sum = clust_sum + k[h]
                        listpointCentr.append(clust_sum / len(dataTemp1))

                    try:
                        if not USE_GPU:
                            raise RuntimeError("GPU désactivé")
                        N_mat = len(valuesMat)
                        if N_mat > 0:
                            M = len(valuesMat[0])
                            M_pad = int(2 ** math.ceil(math.log2(M)))
                            M_pad = max(32, M_pad)

                            sched, kernel = get_gpusync_resources(M_pad, 1)

                            targets_pad = np.zeros((N_mat, M_pad), dtype=np.float32)
                            for idx_val in range(N_mat):
                                targets_pad[idx_val, : len(valuesMat[idx_val])] = valuesMat[idx_val]

                            ref_pad = np.zeros(M_pad, dtype=np.float32)
                            ref_pad[: len(listpointCentr)] = listpointCentr

                            buf_ref = AsyncBuffer.to_device(sched, ref_pad)
                            buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
                            buf_out = AsyncBuffer.empty_like(sched, np.zeros(N_mat, dtype=np.float32))

                            kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, 1, N_mat, global_size=(N_mat,), scheduler=sched)
                            listpoint = list(buf_out.to_host().astype(float))
                        else:
                            listpoint = []
                    except Exception:
                        listpoint = []
                        for k in valuesMat:
                            listpoint.append(DTWDistance_cpu(k, listpointCentr, w))

                    stdTemp = np.std(listpoint)
                    entropy = stats.entropy(listpoint)
                    print("------------------------------------------------------")
                    print("listpoint :", listpoint)
                    print("stdTemp :", stdTemp)
                    print("entropy :", entropy)
                    print("stdTemp / entropy :", stdTemp / entropy)
                    print("dicoCoff[tabEnviIndice] :", dicoCoff.get(tabEnviIndice, "NOT FOUND"))
                    if tabEnviIndice in dicoCoff:
                        print("dicoCoff[tabEnviIndice][0] :", dicoCoff[tabEnviIndice][0])
                        print("Type de dicoCoff[tabEnviIndice][0] :", type(dicoCoff[tabEnviIndice][0]))
                    print("------------------------------------------------------")
                    if tabEnviIndice in dicoCoff:
                        if (stdTemp / entropy) > dicoCoff[tabEnviIndice][0]:
                            retireId = False
                            print(f"Cluster doit être subdivisé: {stdTemp / entropy} > {dicoCoff[tabEnviIndice][0]}")
                        else:
                            print(f"Cluster OK: {stdTemp / entropy} <= {dicoCoff[tabEnviIndice][0]}")
                print(dicoCoff)
                if not retireId:
                    dftTemp2 = dftTemp.copy()
                    indiceSd = indiceSd + 1
                    # Assurez-vous de passer indiceSd comme un entier
                    print("Subdivision du cluster en cours...")
                    self.KmeanMutiShape(nbClust, columns, dftTemp2.copy(), dicoCoff, assignmentsOk, int(indiceSd), w=w)
                    assignmentsOk[indiceSd] = IdTemp
                    indiceSd = indiceSd + 1
        else:
            assignmentsOk[indiceSd] = list(dft[ID_COLUMN])
            indiceSd = indiceSd + 1


def defineCoeff(
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
    """
    Fonction externe pour définir les coefficients de l'analyse

    Args:
        data: Données d'entrée
        num_clust: Nombre de clusters
        num_iter: Nombre d'itérations
        listDataFrame: DataFrame avec IDs
        LastnbClust: Dernier nombre de clusters
        tabEnviIndice: Index de l'environnement
        assignmentsResult: Résultat des assignations
        centroidsCompare: Centroides à comparer
        distCompare: Distances à comparer
        data22: Données additionnelles
        listeIDOk1: Liste des IDs valides
        nbInstanceMinim: Nombre minimum d'instances par cluster
        w: Paramètre de fenêtre pour DTW

    Returns:
        Coefficients calculés
    """
    Nbtarget = 5
    coeffTemp = []
    centroids = {}

    # Initialiser les centroides aléatoirement
    for i in range(num_clust):
        idx = random.randint(0, len(data) - 1)
        centroids[i] = data[idx]

    assignments = {}
    for i in range(num_clust):
        assignments[i] = []

    # Calculer les assignations initiales (GPU/CPU JIT batch)
    try:
        if not USE_GPU:
            raise RuntimeError("GPU désactivé")
        N = len(data)
        K = len(centroids)
        if N > 0 and K > 0:
            M = len(data[0])
            M_pad = int(2 ** math.ceil(math.log2(M)))
            M_pad = max(32, M_pad)

            sched, kernel = get_gpusync_resources(M_pad, K)

            targets_pad = np.zeros((N, M_pad), dtype=np.float32)
            for i in range(N):
                targets_pad[i, : len(data[i])] = data[i]

            refs_pad = np.zeros((K, M_pad), dtype=np.float32)
            keys_list = list(centroids.keys())
            for c_idx, c_key in enumerate(keys_list):
                c_val = centroids[c_key]
                refs_pad[c_idx, : len(c_val)] = c_val

            buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
            buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
            buf_out = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))

            kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, K, N, global_size=(N,), scheduler=sched)

            dist_matrix = buf_out.to_host().reshape(K, N)

            for i in range(N):
                min_idx = np.argmin(dist_matrix[:, i])
                closest_clust = keys_list[min_idx]
                assignments[closest_clust].append(i)
    except Exception:
        # Fallback CPU JIT
        for i in range(len(data)):
            min_dist = float("inf")
            closest_clust = None
            for c in centroids:
                if len(data[i]) > 0:
                    dist = (
                        DTWDistance_cpu(data[i], centroids[c], w) if w > 0 else np.linalg.norm(data[i] - centroids[c])
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_clust = c
            if closest_clust is not None:
                assignments[closest_clust].append(i)

    # Calculer le coefficient basé sur les assignations
    for i in assignments:
        if len(assignments[i]) >= nbInstanceMinim:
            values = []
            for item in assignments[i]:
                values.append(data[item])

            if values:
                np_values = np.array(values)
                if np_values.ndim == 2:
                    np_values = np.expand_dims(np_values, axis=-1)
                bar = euclidean_barycenter(np_values)
                flat_list = [item for sublist in bar for item in sublist]

                try:
                    N_items = len(assignments[i])
                    if N_items > 0:
                        M = len(data[0])
                        M_pad = int(2 ** math.ceil(math.log2(M)))
                        M_pad = max(32, M_pad)

                        sched, kernel = get_gpusync_resources(M_pad, 1)

                        targets_pad = np.zeros((N_items, M_pad), dtype=np.float32)
                        for idx_val, item in enumerate(assignments[i]):
                            targets_pad[idx_val, : len(data[item])] = data[item]

                        ref_pad = np.zeros(M_pad, dtype=np.float32)
                        ref_pad[: len(flat_list)] = flat_list

                        buf_ref = AsyncBuffer.to_device(sched, ref_pad)
                        buf_tgt = AsyncBuffer.to_device(sched, targets_pad.flatten())
                        buf_out = AsyncBuffer.empty_like(sched, np.zeros(N_items, dtype=np.float32))

                        kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, 1, N_items, global_size=(N_items,), scheduler=sched)
                        distances = list(buf_out.to_host().astype(float))
                    else:
                        distances = []
                except Exception:
                    distances = []
                    for item in assignments[i]:
                        distances.append(DTWDistance_cpu(data[item], flat_list, w))

                stdTemp = np.std(distances)
                entropy = stats.entropy(distances) if any(d > 0 for d in distances) else 1.0

                if entropy > 0:
                    coeffTemp.append(stdTemp / entropy)

    return coeffTemp if coeffTemp else [1.0]  # Valeur par défaut si aucun coefficient n'est calculé


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("--- BENCHMARK & TEST DEMO : X-MEANS TIME SERIES CLUSTERING ---")
    print("=" * 60)

    # 1. Génération de données factices pour le test
    # 100 individus, chacun ayant une série temporelle de longueur 32
    np.random.seed(42)
    N = 100
    M = 32

    # Génération de quelques profils types sinusoïdaux avec du bruit
    data = []
    for i in range(N):
        phase = random.choice([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        t_seq = np.arange(M)
        # Signal de base oscillant + bruit
        sig = 20.0 + 5.0 * np.sin(t_seq * (2 * np.pi / 16) + phase) + np.random.randn(M) * 0.3
        data.append(sig.astype(np.float32))

    df_ids = pd.DataFrame({ID_COLUMN: [i for i in range(N)]})

    print(f"[+] Données générées : {N} séries de longueur {M}.")

    # 2. Benchmark unitaire de calcul de distance DTW (CPU vs GPU)
    print("\n[+] Comparaison de vitesse unitaire (DTWDistance) :")
    s1, s2 = data[0], data[1]

    # Version Python pure (recréée à la volée pour comparaison)
    def py_dtw(s1, s2, w):
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

    t_py = 0.0
    for _ in range(10):
        t0 = time.perf_counter()
        _ = py_dtw(s1, s2, 5)
        t_py += time.perf_counter() - t0
    t_py /= 10
    print(f"    - Temps Python Pur  : {t_py * 1000:.4f} ms")

    t_cpu = 0.0
    # Warmup Numba
    _ = DTWDistance_cpu(s1, s2, 5)
    for _ in range(100):
        t0 = time.perf_counter()
        _ = DTWDistance_cpu(s1, s2, 5)
        t_cpu += time.perf_counter() - t0
    t_cpu /= 100
    print(f"    - Temps CPU JIT     : {t_cpu * 1000:.4f} ms (Speedup vs Python: {t_py / t_cpu:.1f}x)")

    # 3. Lancement de l'algorithme complet X-means
    print("\n[+] Exécution du clustering X-means (Accélération GPUsync / JIT) :")
    xmeans = XmeansTS(data=data, numclust=3, file="dummy_path.xlsx", listID=df_ids, coeffRap=1.0)

    # Dictionnaires nécessaires pour l'algorithme original
    distCompare = {}
    listeIDOk1 = []

    t_start = time.perf_counter()
    res_df = xmeans.k_means_clust(
        data=data,
        num_clust=3,
        num_iter=1,
        listDataFrame=df_ids,
        LastnbClust=0,
        tabEnviIndice="Set_TEMP",
        distCompare=distCompare,
        listeIDOk1=listeIDOk1,
        nbInstanceMinim=NB_INSTANCE_MINIMUM,
        w=DEFAULT_W,
    )
    t_total = time.perf_counter() - t_start

    print(f"[OK] Clustering terminé avec succès en {t_total*1000:.2f} ms !")

    # 4. Affichage des résultats
    print("\n--- MATRICE DE DISTANCES FINALE (Centroïdes détectés) ---")
    print(res_df.head(10))
    print("=" * 60)
