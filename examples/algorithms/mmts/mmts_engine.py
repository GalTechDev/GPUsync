# -*- coding: utf-8 -*-
"""
Module d'analyse pour le clustering de séries temporelles avec X-means et DTW
"""

import warnings
warnings.filterwarnings("ignore")
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import os
import sys
import time
import math
import random
# pyrefly: ignore [missing-import]
from scipy import stats
# pyrefly: ignore [missing-import]
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.cluster import KMeans

# Ajout des chemins de recherche système pour importer gpusync
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importation de la configuration
from config import *

# Variables globales pour GPUsync
_gpusync_sched = None
_gpusync_kernels_cache = {}
_gpusync_buffers_cache = {}

import concurrent.futures
_gpusync_io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _async_write_file(path, data, is_json=False, is_df=False):
    import os
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if is_json:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif is_df:
            data.to_csv(path, index=False)
    except Exception as e:
        print(f"Warning: Echec de l'ecriture asynchrone de {path}: {e}")

def get_gpusync_resources(M, num_refs, scheduler=None):
    """
    Récupère ou initialise le scheduler GPUsync et le noyau DTW GPU avec fenêtre.
    Lève une RuntimeError si aucun GPU OpenCL n'est disponible.
    """
    global _gpusync_sched, _gpusync_kernels_cache
    
    if scheduler is not None:
        _gpusync_sched = scheduler
        
    if _gpusync_sched is None:
        from gpusync import Scheduler, list_gpus
        gpus = list_gpus()
        if gpus:
            _gpusync_sched = Scheduler(gpu=gpus[0])
            _gpusync_sched.initialize()
        else:
            raise RuntimeError("GPUsync requiert un GPU compatible OpenCL. Aucun GPU disponible.")

    key = (M, num_refs)
    if key not in _gpusync_kernels_cache:
        from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_window_kernel
        _gpusync_kernels_cache[key] = get_batch_dtw_window_kernel(M, num_refs)
            
    return _gpusync_sched, _gpusync_kernels_cache[key]


def _batch_dtw_gpu(refs_np, targets_np, w, scheduler=None):
    """
    Calcule la matrice de distances DTW sur GPU avec contrainte de fenêtre w et recyclage des buffers.
    """
    from gpusync import AsyncBuffer
    global _gpusync_buffers_cache
    
    K, M_len = refs_np.shape
    N = targets_np.shape[0]
    M_pad = int(2 ** math.ceil(math.log2(M_len)))
    M_pad = max(32, M_pad)
    sched, kernel = get_gpusync_resources(M_pad, K, scheduler)
    
    # Identifiants de cache uniques pour recycler les buffers
    ref_key = f"ref_{K}_{M_pad}"
    tgt_key = f"tgt_{N}_{M_pad}"
    out_key = f"out_{K}_{N}"
    
    refs_pad = np.zeros((K, M_pad), dtype=np.float32)
    refs_pad[:, :M_len] = refs_np
    tgts_pad = np.zeros((N, M_pad), dtype=np.float32)
    tgts_pad[:, :M_len] = targets_np
    
    # 1. Gérer le buffer de référence
    if ref_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[ref_key] = AsyncBuffer.to_device(sched, refs_pad.flatten())
    else:
        _gpusync_buffers_cache[ref_key].overwrite(refs_pad.flatten())
        
    # 2. Gérer le buffer des cibles
    if tgt_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[tgt_key] = AsyncBuffer.to_device(sched, tgts_pad.flatten())
    else:
        _gpusync_buffers_cache[tgt_key].overwrite(tgts_pad.flatten())
        
    # 3. Gérer le buffer de sortie
    if out_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[out_key] = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))
        
    buf_ref = _gpusync_buffers_cache[ref_key]
    buf_tgt = _gpusync_buffers_cache[tgt_key]
    buf_out = _gpusync_buffers_cache[out_key]
    
    # Résolution de la fenêtre d'alignement w
    w_val = int(w) if (w is not None and w > 0) else M_pad
    
    # kernel signature: (references, targets, targets_offset, w, outputs, num_refs_arg, chunk_size)
    kernel(buf_ref, buf_tgt, 0, w_val, buf_out, K, N, global_size=(N,), scheduler=sched)
    return np.sqrt(np.maximum(buf_out.to_host().reshape(K, N), 0.0))


def _compute_dist_matrix(centroids_np, targets_np, w, scheduler=None):
    from examples.algorithms.dtw.dtw_gpu_engine import compute_dtw_matrix
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    coro = compute_dtw_matrix(targets_np, centroids_np, scheduler=scheduler, w=w)
    
    if loop.is_running():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            def _run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            return executor.submit(_run).result()
    else:
        return loop.run_until_complete(coro)




def DTWDistance(s1, s2, w=DEFAULT_W, scheduler=None):
    """
    Calcule la distance DTW unitaire sur GPU avec contrainte de fenêtre w et cache de buffer.
    """
    from gpusync import AsyncBuffer
    global _gpusync_buffers_cache
    
    M = max(len(s1), len(s2))
    M_pad = int(2 ** math.ceil(math.log2(M)))
    M_pad = max(32, M_pad)
    sched, kernel = get_gpusync_resources(M_pad, 1, scheduler)
    
    s1_pad = np.zeros(M_pad, dtype=np.float32)
    s1_pad[:len(s1)] = s1
    s2_pad = np.zeros(M_pad, dtype=np.float32)
    s2_pad[:len(s2)] = s2
    
    ref_key = f"ref_1_{M_pad}"
    tgt_key = f"tgt_1_{M_pad}"
    out_key = f"out_1_1"
    
    if ref_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[ref_key] = AsyncBuffer.to_device(sched, s1_pad)
    else:
        _gpusync_buffers_cache[ref_key].overwrite(s1_pad)
        
    if tgt_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[tgt_key] = AsyncBuffer.to_device(sched, s2_pad)
    else:
        _gpusync_buffers_cache[tgt_key].overwrite(s2_pad)
        
    if out_key not in _gpusync_buffers_cache:
        _gpusync_buffers_cache[out_key] = AsyncBuffer.empty_like(sched, np.zeros(1, dtype=np.float32))
        
    buf_ref = _gpusync_buffers_cache[ref_key]
    buf_tgt = _gpusync_buffers_cache[tgt_key]
    buf_out = _gpusync_buffers_cache[out_key]
    
    w_val = int(w) if (w is not None and w > 0) else M_pad
    
    kernel(buf_ref, buf_tgt, 0, w_val, buf_out, 1, 1, global_size=(1,), scheduler=sched)
    return float(buf_out.to_host()[0])

def centeroidnp(arr):
    """
    Calcule le centroide d'un tableau numpy
    
    Args:
        arr: Tableau numpy de forme (length, dim)
        
    Returns:
        Numpy array représentant le centroide
    """
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])


class XmeansTS:
    """
    Classe pour l'algorithme X-means avec DTW pour le clustering de séries temporelles
    """
    
    def __init__(self, data, numclust, file, listID, coeffRap, representative_mode="euclidean", scheduler=None):
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
        self.representative_mode = representative_mode or "euclidean"
        
        if scheduler is None:
            global _gpusync_sched
            if _gpusync_sched is None:
                from gpusync import Scheduler, list_gpus
                gpus = list_gpus()
                if not gpus:
                    raise RuntimeError("GPUsync requiert un GPU compatible OpenCL.")
                _gpusync_sched = Scheduler(gpu=gpus[0])
                _gpusync_sched.initialize()
            self.scheduler = _gpusync_sched
        else:
            self.scheduler = scheduler
            
        # Préparation du buffer de données persistant sur GPU
        if data is not None and len(data) > 0:
            data_np = np.array(data, dtype=np.float32)
            N, M_len = data_np.shape
            M_pad = int(2 ** math.ceil(math.log2(M_len)))
            M_pad = max(32, M_pad)
            
            self.data_pad = np.zeros((N, M_pad), dtype=np.float32)
            self.data_pad[:, :M_len] = data_np
            
            from gpusync import AsyncBuffer
            self.gpu_data_buf = AsyncBuffer.to_device(self.scheduler, self.data_pad.flatten())
        else:
            self.data_pad = None
            self.gpu_data_buf = None
            
        # Capture de convergence pour LLM
        self.convergence_data = {
            "variable": file,
            "iterations": [],
            "final_metrics": {}
        }
        self.save_debug_csv = False  # Mode debug CSV optionnel

    def _cluster_representative(self, values, w=DEFAULT_W):
        """Return the representative according to the selected strategy."""
        clean_values = [np.asarray(v, dtype=float) for v in values if len(v) > 0]
        if not clean_values:
            return np.array([])
        if len(clean_values) == 1:
            return clean_values[0]

        mode = (self.representative_mode or "euclidean").lower()
        if mode in ("medoid", "medoid_hybrid"):
            try:
                # Calcul de lot de toutes les distances mutuelles sur GPU !
                candidates_np = np.array(clean_values, dtype=np.float32)
                dist_mat = _compute_dist_matrix(candidates_np, candidates_np, w, self.scheduler)
                best_idx = int(np.argmin(dist_mat.sum(axis=1)))
                return clean_values[best_idx]
            except Exception as e:
                best_idx = 0
                best_score = float("inf")
                for idx, candidate in enumerate(clean_values):
                    score = 0.0
                    for other in clean_values:
                        score += DTWDistance(candidate, other, w, self.scheduler)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                return clean_values[best_idx]

        if mode in ("dba", "dtw_barycenter"):
            try:
                bary = dtw_barycenter_averaging(np.asarray(clean_values, dtype=float), max_iter=10)
                return np.asarray(bary, dtype=float).reshape(-1)
            except Exception as exc:
                print(f"DBA indisponible, fallback euclidean: {exc}")

        return np.mean(np.asarray(clean_values, dtype=float), axis=0)
    
    def reductMemory(self):
        """Réduit l'empreinte mémoire"""
        # À implémenter si nécessaire
        pass
    
    def enable_debug_csv(self):
        """Active le mode debug CSV"""
        self.save_debug_csv = True
        print("Mode debug CSV activé")
    
    def capture_convergence_iteration(self, iteration, centroids, assignments, matrix_data=None, precomputed_energy=None):
        """
        Capture une itération de convergence pour analyse LLM
        
        Args:
            iteration: Numéro de l'itération
            centroids: Dictionnaire des centroïdes actuels
            assignments: Dictionnaire des assignations
            matrix_data: Matrice des distances (optionnel)
            precomputed_energy: Énergie déjà calculée (optionnel)
        """
        import time
        import json
        import os
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        variable_name = self.file.replace(" ", "_")
        
        # Calculer les métriques de convergence
        if precomputed_energy is not None:
            energy = precomputed_energy
        else:
            energy = self.kmeansConverg(self.data, centroids, self.dfListID, w=2)
        stability = self.calculate_stability(centroids, assignments)
        
        # Détecter les changements de clusters
        cluster_changes = self.detect_cluster_changes(iteration, assignments)
        
        # Créer les données de l'itération
        iteration_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "energy": float(energy),
            "stability": float(stability),
            "centroids": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in centroids.items()},
            "assignments": {k: list(v) for k, v in assignments.items()},
            "cluster_changes": cluster_changes,
            "matrix_shape": matrix_data.shape if matrix_data is not None else None,
            "num_clusters": len(centroids),
            "num_assigned": sum(len(v) for v in assignments.values())
        }
        
        # Ajouter les données de matrice si disponibles
        if matrix_data is not None:
            iteration_data["matrix_stats"] = {
                "min_distance": float(np.min(matrix_data[matrix_data > 0])),
                "max_distance": float(np.max(matrix_data)),
                "mean_distance": float(np.mean(matrix_data)),
                "symmetry_score": float(self.calculate_symmetry(matrix_data))
            }
        
        # Stocker dans les données de convergence
        self.convergence_data["iterations"].append(iteration_data)
        
        # Créer le dossier de sauvegarde
        base_path = f"data/analyses/{timestamp}_{variable_name}"
        
        # Sauvegarder en JSON (principal) de manière asynchrone
        json_path = f"{base_path}/matrices/iteration_{iteration}_matrix.json"
        _gpusync_io_executor.submit(_async_write_file, json_path, iteration_data, is_json=True)
        
        # Sauvegarder la matrice complète en CSV (debug optionnel)
        if self.save_debug_csv and matrix_data is not None:
            csv_path = f"{base_path}/matrices/iteration_{iteration}_debug.csv"
            df_matrix = pd.DataFrame(matrix_data, 
                                   columns=[f"Cluster_{i}" for i in range(matrix_data.shape[1])],
                                   index=[f"Series_{i}" for i in range(matrix_data.shape[0])])
            _gpusync_io_executor.submit(_async_write_file, csv_path, df_matrix, is_df=True)
        
        # Sauvegarder les métriques en CSV
        metrics_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "energy": energy,
            "stability": stability,
            "num_clusters": len(centroids),
            "min_distance": iteration_data.get("matrix_stats", {}).get("min_distance", 0),
            "max_distance": iteration_data.get("matrix_stats", {}).get("max_distance", 0),
            "symmetry_score": iteration_data.get("matrix_stats", {}).get("symmetry_score", 0)
        }
        
        metrics_df = pd.DataFrame([metrics_data])
        metrics_path = f"{base_path}/metrics/iteration_{iteration}_metrics.csv"
        _gpusync_io_executor.submit(_async_write_file, metrics_path, metrics_df, is_df=True)
        
        print(f"[{variable_name}] Itération {iteration} capturée - Energy: {energy:.3f}, Stability: {stability:.3f}")
    
    def calculate_stability(self, centroids, assignments):
        """Calcule un score de stabilité des clusters"""
        if len(self.convergence_data["iterations"]) == 0:
            return 0.0
        
        # Comparer avec l'itération précédente
        prev_iteration = self.convergence_data["iterations"][-1]
        prev_assignments = prev_iteration["assignments"]
        
        # Calculer le taux de changement
        total_changes = 0
        total_points = 0
        
        for cluster_id in assignments:
            current_set = set(assignments[cluster_id])
            prev_set = set(prev_assignments.get(str(cluster_id), []))
            
            # Points qui ont changé de cluster
            changes = len(current_set.symmetric_difference(prev_set))
            total_changes += changes
            total_points += len(current_set)
        
        if total_points == 0:
            return 1.0
        
        # Stabilité = 1 - taux de changement
        stability = 1.0 - (total_changes / total_points)
        return max(0.0, stability)
    
    def detect_cluster_changes(self, iteration, assignments):
        """Détecte les changements de clusters (fusion, séparation, mouvement)"""
        if len(self.convergence_data["iterations"]) == 0:
            return {"merged": [], "split": [], "moved": []}
        
        prev_iteration = self.convergence_data["iterations"][-1]
        prev_assignments = prev_iteration["assignments"]
        
        changes = {"merged": [], "split": [], "moved": []}
        
        # Logique simple de détection (à améliorer)
        for cluster_id in assignments:
            current_size = len(assignments[cluster_id])
            prev_size = len(prev_assignments.get(str(cluster_id), []))
            
            if current_size > prev_size * 1.5:
                changes["merged"].append(cluster_id)
            elif current_size < prev_size * 0.5:
                changes["split"].append(cluster_id)
            elif abs(current_size - prev_size) > 0:
                changes["moved"].append(cluster_id)
        
        return changes
    
    def calculate_symmetry(self, matrix):
        """Calcule un score de symétrie de la matrice de distances"""
        if matrix.shape[0] != matrix.shape[1]:
            return 0.0
        
        # Score de symétrie = 1 - erreur de symétrie normalisée
        symmetry_error = np.mean(np.abs(matrix - matrix.T))
        max_val = np.max(np.abs(matrix))
        
        if max_val == 0:
            return 1.0
        
        return 1.0 - (symmetry_error / max_val)
    
    def save_convergence_summary(self):
        """Sauvegarde le résumé complet de la convergence"""
        import json
        import os
        from datetime import datetime
        
        # Utiliser le même timestamp que les itérations
        if len(self.convergence_data["iterations"]) > 0:
            # Récupérer le timestamp de la première itération
            first_iteration = self.convergence_data["iterations"][0]
            timestamp = first_iteration["timestamp"]
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        variable_name = self.file.replace(" ", "_")
        base_path = f"data/analyses/{timestamp}_{variable_name}"
        
        # S'assurer que le dossier existe
        os.makedirs(base_path, exist_ok=True)
        
        # Calculer les métriques finales
        if len(self.convergence_data["iterations"]) > 0:
            first_energy = self.convergence_data["iterations"][0]["energy"]
            last_energy = self.convergence_data["iterations"][-1]["energy"]
            energy_reduction = (first_energy - last_energy) / first_energy if first_energy > 0 else 0
            
            avg_stability = np.mean([it["stability"] for it in self.convergence_data["iterations"]])
            
            self.convergence_data["final_metrics"] = {
                "total_iterations": len(self.convergence_data["iterations"]),
                "convergence_speed": "fast" if len(self.convergence_data["iterations"]) < 5 else "medium" if len(self.convergence_data["iterations"]) < 10 else "slow",
                "energy_reduction": float(energy_reduction),
                "final_energy": float(last_energy),
                "average_stability": float(avg_stability),
                "quality_score": float(avg_stability * (1 - last_energy / first_energy) if first_energy > 0 else 0)
            }
        
        # Sauvegarder le JSON principal
        json_path = f"{base_path}/convergence_data.json"
        _gpusync_io_executor.submit(_async_write_file, json_path, self.convergence_data, is_json=True)
        
        # Créer un CSV résumé
        summary_data = []
        for iteration in self.convergence_data["iterations"]:
            summary_data.append({
                "iteration": iteration["iteration"],
                "timestamp": iteration["timestamp"],
                "energy": iteration["energy"],
                "stability": iteration["stability"],
                "num_clusters": iteration["num_clusters"],
                "num_assigned": iteration["num_assigned"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{base_path}/summary.csv"
        _gpusync_io_executor.submit(_async_write_file, summary_path, summary_df, is_df=True)
        
        print(f"[{variable_name}] Résumé de convergence sauvegardé: {len(self.convergence_data['iterations'])} itérations")
        
        return json_path
    
    def compute_distances_to_centroids_parallel(self, data, centroids, w, n_processes=None, squared=False):
        """
        Calcule les distances de chaque série à chaque centroïde sur GPU avec GPUsync
        
        Args:
            data: Données (liste de séries)
            centroids: Dictionnaire des centroïdes
            w: Paramètre de fenêtre pour DTW
            n_processes: Nombre de processus (ignoré, calculé sur GPU)
            squared: Si True, retourne la distance au carré (évite np.sqrt)
            
        Returns:
            Matrice des distances (n_series x n_centroids)
        """
        if not data or not centroids:
            return np.array([[]])
            
        keys_list = list(centroids.keys())
        K = len(keys_list)
        N = len(data)
        
        if len(data[0]) == 0:
            return np.full((len(data), K), np.inf)
            
        M_len = len(data[0])
        
        # Si le buffer persistant est disponible et qu'on traite les mêmes données
        if self.gpu_data_buf is not None and N == self.data_pad.shape[0] and M_len <= self.data_pad.shape[1]:
            M_pad = self.data_pad.shape[1]
            sched, kernel = get_gpusync_resources(M_pad, K, self.scheduler)
            
            # refs_pad: K x M_pad
            refs_pad = np.zeros((K, M_pad), dtype=np.float32)
            try:
                refs_pad[:, :M_len] = np.array(list(centroids.values()), dtype=np.float32)
            except Exception:
                for c_idx, c_key in enumerate(keys_list):
                    refs_pad[c_idx, :len(centroids[c_key])] = np.array(centroids[c_key], dtype=np.float32)
                
            from gpusync import AsyncBuffer
            
            # Réutiliser un buffer de référence en cache si possible pour éviter clCreateBuffer
            ref_key = f"ref_{K}_{M_pad}"
            if ref_key not in _gpusync_buffers_cache:
                _gpusync_buffers_cache[ref_key] = AsyncBuffer.to_device(sched, refs_pad.flatten())
            else:
                _gpusync_buffers_cache[ref_key].overwrite(refs_pad.flatten())
                
            # Réutiliser un buffer de sortie en cache
            out_key = f"out_{K}_{N}"
            if out_key not in _gpusync_buffers_cache:
                _gpusync_buffers_cache[out_key] = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))
                
            buf_ref = _gpusync_buffers_cache[ref_key]
            buf_out = _gpusync_buffers_cache[out_key]
            
            w_val = int(w) if (w is not None and w > 0) else M_pad
            
            # Lancement du noyau avec le buffer persistant self.gpu_data_buf !
            kernel(buf_ref, self.gpu_data_buf, 0, w_val, buf_out, K, N, global_size=(N,), scheduler=sched)
            
            # Rapatrier et transposer
            res_flat = buf_out.to_host().reshape(K, N)
            if squared:
                return np.maximum(res_flat, 0.0).T
            else:
                return np.sqrt(np.maximum(res_flat, 0.0)).T
        else:
            # Repli classique s'il s'agit d'autres données
            # refs_np: K x M_len
            refs_np = np.zeros((K, M_len), dtype=np.float32)
            for c_idx, c_key in enumerate(keys_list):
                refs_np[c_idx] = np.array(centroids[c_key], dtype=np.float32)
                
            # targets_np: N x M_len
            targets_np = np.array(data, dtype=np.float32)
            
            dist_mat = _compute_dist_matrix(refs_np, targets_np, w, self.scheduler)
            if squared:
                return (dist_mat ** 2).T
            else:
                return dist_mat.T

    def kmeansConverg(self, data, centroids, listID, w):
        """
        Vérifie la convergence de l'algorithme k-means (version optimisée)
        
        Args:
            data: Données
            centroids: Centroides actuels
            listID: Liste des IDs
            w: Paramètre de fenêtre pour DTW
            
        Returns:
            True si la convergence est atteinte, False sinon
        """
        # Utilisation du calcul parallèle si possible
        try:
            distance_matrix = self.compute_distances_to_centroids_parallel(data, centroids, w)
            # Calcul de la somme des distances minimales
            min_distances = np.min(distance_matrix, axis=1)
            sum_dist = np.sum(min_distances)
            return sum_dist
        except:
            # Fallback vers l'implémentation originale
            sum_dist = 0
            for i in range(len(data)):
                min_dist = float('inf')
                for c in centroids:
                    if len(data[i]) > 0:
                        if w > 0:
                            dist = DTWDistance(data[i], centroids[c], w)
                        else:
                            dist = np.linalg.norm(data[i] - centroids[c])
                        
                        if dist < min_dist:
                            min_dist = dist
                sum_dist += min_dist
            return sum_dist
    
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
                centroids[c] = self._cluster_representative(
                    [data[i] for i in assignments[c] if len(data[i]) > 0],
                    DEFAULT_W
                )
        return centroids
    
    def computeDist(self, data, listID, centroids, assignments, w):
        """
        Calcule les distances entre points et centroides sur GPU avec GPUsync
        
        Args:
            data: Données
            listID: Liste des IDs
            centroids: Centroides actuels
            assignments: Assignations actuelles
            w: Paramètre de fenêtre pour DTW
            
        Returns:
            Nouvelles assignations
        """
        for c in assignments:
            assignments[c] = []
            
        if not data or not centroids:
            return assignments
            
        # Calculer toutes les distances d'un coup sur GPU !
        distance_matrix = self.compute_distances_to_centroids_parallel(data, centroids, w)
        
        # Assigner chaque point au centroïde le plus proche (argmin)
        labels = np.argmin(distance_matrix, axis=1)
        for i, c in enumerate(labels):
            assignments[int(c)].append(i)
            
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
    
    def selectSerie(self, listID, assignments, assignmentsTemp, assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim):
        """
        Sélectionne les séries pour les clusters
        
        Args:
            listID: Liste des IDs
            assignments: Assignations actuelles
            assignmentsTemp: Assignations temporaires
            assignmentsTempBary: Assignations temporaires pour barycentre
            assignmentsTempBaryDTW: Assignations temporaires pour barycentre avec DTW
            nbInstanceMinim: Nombre minimum d'instances par cluster
        """
        # Récupérer l'array des IDs pour éviter l'indexation lente de pandas .iloc dans la boucle
        id_array = listID[ID_COLUMN].values
        for i in assignments:
            if len(assignments[i]) >= nbInstanceMinim:
                ids = []
                for item in assignments[i]:
                    val = id_array[item]
                    try:
                        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
                            val = int(val)
                    except Exception:
                        pass
                    ids.append(val)
                assignmentsTemp[i] = ids
                assignmentsTempBary[i] = []
                assignmentsTempBaryDTW[i] = []
                
                for item in assignments[i]:
                    assignmentsTempBary[i].append(self.data[item])
    
    def updateAssignment(self, assignmentsTemp, listeIDOk, nbClustTempp):
        """
        Met à jour les assignations
        
        Args:
            assignmentsTemp: Assignations temporaires
            listeIDOk: Liste des IDs valides
            nbClustTempp: Nombre temporaire de clusters
            
        Returns:
            Liste mise à jour des IDs valides
        """
        # Dictionnaire pour mapper instantanément un ID vers son index de ligne (évite np.where dans la boucle)
        id_to_idx = {val: idx for idx, val in enumerate(self.dfListID[ID_COLUMN].values)}
        
        for key in assignmentsTemp:
            self.assignments[nbClustTempp] = assignmentsTemp[key]
            for elem in self.assignments[nbClustTempp]:
                listeIDOk.append(elem)
                
            if len(self.assignments[nbClustTempp]) > 0:
                values = []
                for elId in self.assignments[nbClustTempp]:
                    idx_val = id_to_idx.get(elId)
                    if idx_val is not None:
                        idTS = self.data[idx_val]
                        values.append(idTS)
                
                flat_list = []
                if values:
                    flat_list = list(self._cluster_representative(values, DEFAULT_W))
                
                self.centroids[nbClustTempp] = flat_list
                
                # Accumuler et calculer toutes les distances de ce cluster au centroïde d'un coup sur GPU !
                rows = []
                cluster_series = []
                valid_el_ids = []
                for elId in self.assignments[nbClustTempp]:
                    idx_val = id_to_idx.get(elId)
                    if idx_val is not None:
                        cluster_series.append(self.data[idx_val])
                        valid_el_ids.append(elId)
                        
                if cluster_series:
                    cluster_series_np = np.array(cluster_series, dtype=np.float32)
                    centroid_np = np.array([flat_list], dtype=np.float32)  # forme (1, M)
                    dist_mat = _compute_dist_matrix(centroid_np, cluster_series_np, DEFAULT_W, self.scheduler)
                    dists_row = dist_mat[0]
                    for elId, LBVal in zip(valid_el_ids, dists_row):
                        val_id = elId
                        try:
                            if isinstance(val_id, (int, float)) or (isinstance(val_id, str) and val_id.isdigit()):
                                val_id = int(val_id)
                        except Exception:
                            pass
                        rows.append({ID_COLUMN: val_id, 'MEAS': float(LBVal)})
                
                MatDist = pd.DataFrame(rows, columns=[ID_COLUMN, 'MEAS'])
                
                nbClustTempp += 1
        return listeIDOk
    
    def updateVar(self, listFilterID, colID=ID_COLUMN, ranges=None):
        """
        Met à jour les variables
        
        Args:
            listFilterID: Liste des IDs filtrés
            colID: Nom de la colonne d'ID
            ranges: Plages de valeurs
            
        Returns:
            Données et DataFrame mis à jour
        """
        dfListIDTemp = self.dfListID[~self.dfListID[colID].isin(listFilterID)]
        data = []
        for i in range(len(dfListIDTemp)):
            data.append(self.data[i])
        
        return data, dfListIDTemp
    
    def k_means_clust(self, data, num_clust, num_iter, listDataFrame, LastnbClust, tabEnviIndice, distCompare, listeIDOk1, nbInstanceMinim, w=DEFAULT_W):
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
            data_np = np.asarray(data, dtype=np.float32)
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
            
            # Capturer l'état initial (centroïdes aléatoires)
            self.capture_convergence_iteration(counter, centroids, assignments)
            
            # K-means principal
            while not testStop and counter < num_iter:
                assignments = {}
                assignmentsTemp = {}
                assignmentsTempBary = {}
                assignmentsTempBaryDTW = {}
                
                for i in range(num_clust):
                    assignments[i] = []
                    assignmentsTemp[i] = []
                    assignmentsTempBary[i] = []
                    assignmentsTempBaryDTW[i] = []
                
                # Calculer les distances et assigner les points de manière parallélisée
                try:
                    # distance_matrix a la forme (N, K)
                    distance_matrix = self.compute_distances_to_centroids_parallel(data, centroids, w, squared=True)
                    closest_clusters_idx = np.argmin(distance_matrix, axis=1)
                    min_distances = np.min(distance_matrix, axis=1)
                    precomputed_energy = float(np.sum(np.sqrt(min_distances)))
                    
                    keys_list = list(centroids.keys())
                    for c_idx, c_key in enumerate(keys_list):
                        pt_indices = np.where(closest_clusters_idx == c_idx)[0]
                        assignments[c_key] = pt_indices.tolist()
                except Exception as e:
                    # Fallback séquentiel
                    precomputed_energy = 0.0
                    for i in range(len(data)):
                        min_dist = float('inf')
                        closest_clust = None
                        for c in centroids:
                            if len(data[i]) > 0:
                                if w > 0:
                                    dist = DTWDistance(data[i], centroids[c], w)
                                else:
                                    dist = np.linalg.norm(data[i] - centroids[c])
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_clust = c
                        if closest_clust is not None:
                            assignments[closest_clust].append(i)
                            precomputed_energy += min_dist
                
                # Mettre à jour les centroides
                for c in assignments:
                    if len(assignments[c]) > 0:
                        if (self.representative_mode or "euclidean").lower() == "euclidean":
                            centroids[c] = np.mean(data_np[assignments[c]], axis=0)
                        else:
                            centroids[c] = self._cluster_representative(
                                [data[i] for i in assignments[c] if len(data[i]) > 0],
                                w
                            )
                
                # Capturer l'itération de convergence avec l'énergie pré-calculée
                self.capture_convergence_iteration(counter, centroids, assignments, precomputed_energy=precomputed_energy)
                
                testStop = precomputed_energy
                counter += 1
            
            # Sélectionner les séries pour chaque cluster
            self.selectSerie(listDataFrame, assignments, assignmentsTemp, assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim)
            
            nbClustLow = LastnbClust
            listeIDOk = []
            
            # Mettre à jour les assignations
            listeIDOk = self.updateAssignment(assignmentsTemp, listeIDOk, LastnbClust)
            
            listeIDOk1 = listeIDOk1 + listeIDOk
            updateVar = self.updateVar(listeIDOk1)
            
            # Appel récursif avec les données mises à jour
            if len(updateVar[0]) > 0:
                self.k_means_clust(
                    updateVar[0], num_clust, num_iter, updateVar[1], 
                    LastnbClust + len(assignmentsTemp), tabEnviIndice, 
                    distCompare, listeIDOk1, nbInstanceMinim, w
                )
            
            # Sauvegarder le résumé de convergence
            self.save_convergence_summary()
            
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
        Récupère la matrice finale des résultats - Version optimisée GPU
        Crée une colonne par centroïde avec les distances DTW
        
        Args:
            tabEnviIndice: Index de l'environnement
            w: Paramètre de fenêtre pour DTW
            
        Returns:
            DataFrame avec colonnes: elev_id, 0_Set_TEMP, 1_Set_TEMP, etc.
        """
        print(f"DEBUG getLastMat - self.centroids.keys(): {list(self.centroids.keys()) if self.centroids else 'VIDE'}")
        print(f"DEBUG getLastMat - len(self.data): {len(self.data)}")
        print(f"DEBUG getLastMat - tabEnviIndice: {tabEnviIndice}")
        
        MatriceDist = pd.DataFrame()
        if not self.centroids or self.data is None or len(self.data) == 0:
            MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
            return MatriceDist
            
        keys_list = list(self.centroids.keys())
        # Lancer le calcul de lot directement sur GPU via le buffer persistant (N x K)
        dist_mat = self.compute_distances_to_centroids_parallel(self.data, self.centroids, w)
        
        # dist_mat a la forme (N, K). Remplir chaque colonne en vectorisé sans .tolist()
        for c_idx, c_key in enumerate(keys_list):
            MatriceDist[f"{c_key}_{tabEnviIndice}"] = dist_mat[:, c_idx]
            
        MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
        print(f"DEBUG getLastMat - Colonnes créées: {list(MatriceDist.columns)}")
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
        self.split_decisions = []
        self.max_split_depth = 5

    def _assign_multi_cluster(self, dft, assignmentsOk, indiceSd, reason, depth=0):
        ids = list(dft[ID_COLUMN])
        assignmentsOk[indiceSd] = ids
        self.split_decisions.append({
            "cluster_id": int(indiceSd),
            "depth": int(depth),
            "size": int(len(ids)),
            "action": "keep",
            "reason": reason,
        })
        return indiceSd + 1

    @staticmethod
    def _extract_threshold(value):
        try:
            if isinstance(value, (list, tuple)):
                return MultiXmeansTS._extract_threshold(value[0])
            return float(value)
        except Exception:
            return 1.0

    @staticmethod
    def _variable_feature_columns(variable, columns, dicoCoff):
        available = [str(col) for col in columns]
        declared = []
        try:
            declared = [str(col) for col in (dicoCoff.get(variable, [[], []])[1][0])]
        except Exception:
            declared = []
        matched = [col for col in available if col in declared]
        if matched:
            return matched
        suffix = f"_{variable}"
        return [col for col in available if col.endswith(suffix) or variable in col]

    @staticmethod
    def _dispersion_ratio(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) <= 1:
            return 0.0, 0.0, 0.0
        std_temp = float(np.std(arr))
        entropy_input = np.abs(arr) + 1e-12
        entropy = float(stats.entropy(entropy_input))
        ratio = std_temp if not math.isfinite(entropy) or entropy <= 1e-12 else std_temp / entropy
        return float(ratio), std_temp, entropy

    def _multi_split_decision(self, dft_cluster, columns, dicoCoff):
        decisions = []
        should_split = False
        for variable in dicoCoff.keys():
            feature_cols = self._variable_feature_columns(variable, columns, dicoCoff)
            if not feature_cols:
                continue
            matrix = dft_cluster[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            matrix = matrix.fillna(matrix.mean()).fillna(0.0).values
            if len(matrix) <= 1:
                continue
            center = np.mean(matrix, axis=0)
            distances = np.linalg.norm(matrix - center, axis=1)
            ratio, std_temp, entropy = self._dispersion_ratio(distances)
            threshold = self._extract_threshold(dicoCoff.get(variable, [[1.0]])[0])
            variable_split = ratio > threshold
            should_split = should_split or variable_split
            decisions.append({
                "variable": variable,
                "ratio": ratio,
                "threshold": threshold,
                "std": std_temp,
                "entropy": entropy,
                "feature_count": len(feature_cols),
                "split": bool(variable_split),
            })
        return should_split, decisions

    def KmeanMutiShapeAdaptive(self, nbClust, columns, dft, dicoCoff, assignmentsOk, indiceSd, w=DEFAULT_W, depth=0):
        """K-means multi with controlled recursive subdivision on the final DTW matrix."""
        columns = [str(col) for col in columns]
        print(f"DEBUG KmeanMutiShapeAdaptive - len(dft): {len(dft)}, nbClust: {nbClust}, depth: {depth}")
        if ID_COLUMN not in dft.columns:
            raise KeyError(ID_COLUMN)
        if not columns:
            return self._assign_multi_cluster(dft, assignmentsOk, indiceSd, "no_feature_columns", depth)
        if depth >= self.max_split_depth:
            return self._assign_multi_cluster(dft, assignmentsOk, indiceSd, "max_depth_reached", depth)
        if len(dft) <= max(2, nbClust * 2):
            return self._assign_multi_cluster(dft, assignmentsOk, indiceSd, "not_enough_objects", depth)

        dftmp = dft.copy()
        feature_df = dftmp[columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.mean()).fillna(0.0)
        local_k = max(1, min(int(nbClust), len(dftmp) - 1))
        kmeans = KMeans(n_clusters=local_k, n_init=10, random_state=42)
        y = kmeans.fit(feature_df.values)
        dftmp['Cluster'] = y.labels_

        for clust in sorted(dftmp['Cluster'].unique()):
            dftTemp = dftmp[dftmp['Cluster'] == clust].drop(columns=['Cluster'])
            should_split, decisions = self._multi_split_decision(dftTemp, columns, dicoCoff)
            self.split_decisions.append({
                "depth": int(depth),
                "local_cluster": int(clust),
                "size": int(len(dftTemp)),
                "action": "split" if should_split else "keep_candidate",
                "decisions": decisions,
            })
            if should_split and len(dftTemp) > max(2, nbClust * 2):
                indiceSd = self.KmeanMutiShapeAdaptive(
                    nbClust, columns, dftTemp.copy(), dicoCoff, assignmentsOk, int(indiceSd), w=w, depth=depth + 1
                )
            else:
                reason = "split_requested_but_not_enough_objects" if should_split else "dispersion_ok"
                indiceSd = self._assign_multi_cluster(dftTemp, assignmentsOk, indiceSd, reason, depth)
        return indiceSd
    
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
        print(f"DEBUG KmeanMutiShape - len(dft): {len(dft)}, nbClust: {nbClust}, nbClust*2: {nbClust*2}")
        print(f"DEBUG - Colonnes utilisées pour K-means: {list(columns)}")
        print(f"DEBUG - Nombre de colonnes: {len(list(columns))}")
        if len(dft) > nbClust * 2:
            dftmp = dft.copy()
            print(f"DEBUG - Shape des données pour K-means: {dftmp[list(columns)].shape}")
            print(f"DEBUG - Premières valeurs: {dftmp[list(columns)].head()}")
            kmeans = KMeans(n_clusters=nbClust)
            y = kmeans.fit(dftmp[list(columns)].values)
            dftmp['Cluster'] = y.labels_
            print(f"DEBUG - Clusters uniques créés par K-means: {dftmp['Cluster'].unique()}")
            print(f"DEBUG - Nombre de clusters: {len(dftmp['Cluster'].unique())}")
            
            for clust in dftmp['Cluster'].unique():
                retireId = True
                dftTemp = dftmp[dftmp['Cluster'] == clust]
                IdTemp = list(dftTemp[ID_COLUMN])
                
                for tabEnviIndice in self.files:
                    # Construire le chemin du fichier
                    if hasattr(self, 'data_directory') and hasattr(self, 'file_prefix') and self.data_directory is not None:
                        file_name = tabEnviIndice
                        if file_name.startswith("Set_"):
                            file_name = file_name[4:]
                        possible_paths = [
                            os.path.join(self.data_directory, f"{self.file_prefix}{tabEnviIndice}.csv"),
                            os.path.join(self.data_directory, f"{self.file_prefix}{file_name}.csv"),
                            os.path.join(self.data_directory, f"{tabEnviIndice}.csv"),
                            os.path.join(self.data_directory, f"{file_name}.csv"),
                            os.path.join(self.data_directory, f"{self.file_prefix}{tabEnviIndice}.xlsx"),
                            os.path.join(self.data_directory, f"{self.file_prefix}{file_name}.xlsx"),
                            os.path.join(self.data_directory, f"{tabEnviIndice}.xlsx"),
                            os.path.join(self.data_directory, f"{file_name}.xlsx")
                        ]
                    else:
                        # Fallback pour compatibilité avec l'original
                        file_name = tabEnviIndice
                        if file_name.startswith("Set_"):
                            file_name = file_name[4:]
                        
                        possible_paths = [
                            os.path.join(os.path.dirname(os.getcwd()), f"RomaneDataSet_{file_name}.csv"),
                            os.path.join(os.getcwd(), f"RomaneDataSet_{file_name}.csv"),
                            os.path.join(os.path.dirname(os.getcwd()), f"RomaneDataSet_{file_name}.xlsx"),
                            os.path.join(os.getcwd(), f"RomaneDataSet_{file_name}.xlsx"),
                            ROMANE_DATA_PATH + tabEnviIndice + '.xlsx',
                            ROMANE_DATA_PATH + file_name + '.xlsx'
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
                        raise FileNotFoundError(f"Impossible de trouver le fichier pour {tabEnviIndice}. Chemins essayés:\n{paths_str}")
                    
                    print(f"Chargement du fichier: {file_path}")
                    if file_path.endswith('.csv'):
                        dataTemp = pd.read_csv(file_path)
                    else:
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
                    
                    for k in valuesMat:
                        adist = DTWDistance(k, listpointCentr, w)
                        listpoint.append(adist)
                    
                    stdTemp = np.std(listpoint)
                    entropy = stats.entropy(listpoint)
                    print( "------------------------------------------------------")
                    print('listpoint :', listpoint)
                    print('stdTemp :', stdTemp)
                    print('entropy :', entropy)
                    print('stdTemp / entropy :', stdTemp / entropy)
                    print('dicoCoff[tabEnviIndice] :', dicoCoff.get(tabEnviIndice, 'NOT FOUND'))
                    if tabEnviIndice in dicoCoff:
                        print('dicoCoff[tabEnviIndice][0] :', dicoCoff[tabEnviIndice][0])
                        print('Type de dicoCoff[tabEnviIndice][0] :', type(dicoCoff[tabEnviIndice][0]))
                    print( "------------------------------------------------------")
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
                    print(f"Subdivision du cluster en cours...")
                    self.KmeanMutiShape(nbClust, columns, dftTemp2.copy(), dicoCoff, assignmentsOk, int(indiceSd), w=w)
                else:
                    print("assignmentsOk : ", assignmentsOk)
                    print("indiceSd : ", indiceSd)
                    print("IdTemp : ", IdTemp)
                    print(f"DEBUG - Assignation du cluster {indiceSd} avec {len(IdTemp)} individus")
                    assignmentsOk[indiceSd] = IdTemp
                    indiceSd = indiceSd + 1
        else:
            print(f"DEBUG - Pas assez d'individus ({len(dft)} <= {nbClust*2}), création d'un seul cluster")
            assignmentsOk[indiceSd] = list(dft[ID_COLUMN])
            indiceSd = indiceSd + 1


def defineCoeff(data, num_clust, num_iter, listDataFrame, LastnbClust, tabEnviIndice, assignmentsResult, centroidsCompare, distCompare, data22, listeIDOk1, nbInstanceMinim, w=1, scheduler=None):
    """
    Fonction externe optimisée GPU pour définir les coefficients de l'analyse.
    """
    if not data:
        return [1.0]
    
    coeffTemp = []
    data_np = np.array(data, dtype=np.float32)
    N = len(data)
    
    # 1. Initialiser les centroides aléatoirement
    centroids = {}
    for i in range(num_clust):
        idx = random.randint(0, N - 1)
        centroids[i] = data_np[idx]
        
    # 2. Calculer les assignations initiales sur GPU
    refs = np.array(list(centroids.values()), dtype=np.float32)
    dist_mat = _compute_dist_matrix(refs, data_np, w, scheduler)
    
    labels = np.argmin(dist_mat, axis=0)
    assignments = {i: [] for i in range(num_clust)}
    for i, c in enumerate(labels):
        assignments[int(c)].append(i)
        
    # 3. Calculer le coefficient basé sur les assignations
    for i in assignments:
        if len(assignments[i]) >= nbInstanceMinim:
            idx_arr = np.array(assignments[i], dtype=np.int32)
            member_data = data_np[idx_arr]
            if len(member_data) > 0:
                # Barycentre euclidien vectorisé
                centroid = member_data.mean(axis=0).reshape(1, -1)
                
                # Calculer les distances de toutes les séries de ce cluster à leur barycentre sur GPU
                dists_row = _compute_dist_matrix(centroid, member_data, w, scheduler)[0]
                
                stdTemp = float(np.std(dists_row))
                entropy = float(stats.entropy(dists_row)) if np.any(dists_row > 0) else 1.0
                if entropy > 0:
                    coeffTemp.append(stdTemp / entropy)
    
    return coeffTemp if coeffTemp else [1.0]  # Valeur par défaut si aucun coefficient n'est calculé


class XmeansTS_MultiPopulation(XmeansTS):
    """
    Extension de XmeansTS avec initialisation multi-population (V2).
    Evalue P populations de centroïdes en parallèle sur le GPU à chaque étape
    pour éviter les minima locaux et garantir la meilleure qualité de partitionnement.
    """
    def __init__(self, data, numclust, file, listID, coeffRap, representative_mode="euclidean", scheduler=None, P=5):
        super().__init__(data, numclust, file, listID, coeffRap, representative_mode, scheduler)
        self.P = P  # Nombre de populations à évaluer en parallèle

    def k_means_clust(self, data, num_clust, num_iter, listDataFrame, LastnbClust, tabEnviIndice, distCompare, listeIDOk1, nbInstanceMinim, w):
        """
        Version Multi-Population de k_means_clust.
        Fait tourner P K-means en parallèle sur le GPU en les regroupant dans un unique calcul matriciel
        pour ne conserver que le meilleur partitionnement global (inertie minimale).
        """
        if len(data) >= num_clust * 2:
            data_np = np.array(data, dtype=np.float32)
            N, M_len = data_np.shape
            
            # 1. Initialiser P populations de centroïdes différents
            pop_centroids = []
            for p in range(self.P):
                centroids = {}
                for i in range(num_clust):
                    idx = random.randint(0, len(data) - 1)
                    centroids[i] = data[idx]
                pop_centroids.append(centroids)
                
            pop_assignments = [{} for _ in range(self.P)]
            pop_energies = [0.0 for _ in range(self.P)]
            
            # K-means principal sur toutes les populations
            counter = 0
            while counter < num_iter:
                # Ré-initialiser les assignations temporaires pour cette itération
                for p in range(self.P):
                    pop_assignments[p] = {i: [] for i in range(num_clust)}
                    pop_energies[p] = 0.0
                
                # Regrouper TOUS les centroïdes de toutes les populations dans un seul lot GPU
                total_cents_keys = []
                total_cents_dict = {}
                global_idx = 0
                for p in range(self.P):
                    for c_key in sorted(pop_centroids[p].keys()):
                        total_cents_dict[global_idx] = pop_centroids[p][c_key]
                        total_cents_keys.append((p, c_key))
                        global_idx += 1
                        
                # Calculer toutes les distances de toutes les séries vers tous les centroïdes d'un seul coup
                distance_matrix = self.compute_distances_to_centroids_parallel(data, total_cents_dict, w, squared=True)
                
                # Affecter les points aux clusters pour chaque population
                for p in range(self.P):
                    p_col_indices = [idx for idx, (pop_id, _) in enumerate(total_cents_keys) if pop_id == p]
                    p_keys = [c_key for pop_id, c_key in total_cents_keys if pop_id == p]
                    
                    sub_matrix = distance_matrix[:, p_col_indices]
                    closest_clusters_idx = np.argmin(sub_matrix, axis=1)
                    min_distances = np.min(sub_matrix, axis=1)
                    pop_energies[p] = float(np.sum(np.sqrt(min_distances)))
                    
                    for col_sub_idx, c_key in enumerate(p_keys):
                        pt_indices = np.where(closest_clusters_idx == col_sub_idx)[0]
                        pop_assignments[p][c_key] = pt_indices.tolist()
                            
                # Mettre à jour les centroïdes pour chaque population
                for p in range(self.P):
                    for c in pop_assignments[p]:
                        if len(pop_assignments[p][c]) > 0:
                            if (self.representative_mode or "euclidean").lower() == "euclidean":
                                pop_centroids[p][c] = np.mean(data_np[pop_assignments[p][c]], axis=0)
                            else:
                                pop_centroids[p][c] = self._cluster_representative(
                                    [data[i] for i in pop_assignments[p][c] if len(data[i]) > 0],
                                    w
                                )
                counter += 1
                
            # 2. Sélectionner la MEILLEURE population (celle qui a l'énergie/inertie la plus faible)
            best_pop_idx = int(np.argmin(pop_energies))
            best_centroids = pop_centroids[best_pop_idx]
            best_assignments = pop_assignments[best_pop_idx]
            best_energy = pop_energies[best_pop_idx]
            
            # Assigner les variables de la classe avec cette meilleure population
            self.centroids = best_centroids
            self.assignments = best_assignments
            
            # Log de l'itération finale de la meilleure population
            self.capture_convergence_iteration(counter, self.centroids, self.assignments, precomputed_energy=best_energy)
            
            # Sélectionner les séries pour chaque cluster
            assignmentsTemp = {i: [] for i in range(num_clust)}
            assignmentsTempBary = {i: [] for i in range(num_clust)}
            assignmentsTempBaryDTW = {i: [] for i in range(num_clust)}
            self.selectSerie(listDataFrame, self.assignments, assignmentsTemp, assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim)
            
            nbClustLow = LastnbClust
            listeIDOk = []
            
            # Mettre à jour les assignations
            listeIDOk = self.updateAssignment(assignmentsTemp, listeIDOk, LastnbClust)
            listeIDOk1 = listeIDOk1 + listeIDOk
            updateVar = self.updateVar(listeIDOk1)
            
            # Appel récursif avec les données mises à jour
            if len(updateVar[0]) > 0:
                self.k_means_clust(
                    updateVar[0], num_clust, num_iter, updateVar[1], 
                    LastnbClust + len(assignmentsTemp), tabEnviIndice, 
                    distCompare, listeIDOk1, nbInstanceMinim, w
                )
                
            # Sauvegarder le résumé de convergence
            self.save_convergence_summary()
            
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


if __name__ == "__main__":
    print("=" * 80)
    print(" EXECUTION COMPLETE DU PIPELINE MMTS SUR LE DOSSIER DATASET (GPU ONLY)")
    print("=" * 80)

    # Résolution dynamique des chemins
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(HERE, "dataset")
    
    if not os.path.exists(DATA_DIR):
        print(f"[!] Dossier dataset introuvable : {DATA_DIR}. Fin.")
        sys.exit(1)
        
    # 0. NETTOYAGE ET CONVERSION AUTOMATIQUE DES FICHIERS EXCEL EN CSV PROPRES
    CLEAN_DIR = os.path.join(HERE, "dataset_clean")
    os.makedirs(CLEAN_DIR, exist_ok=True)
    
    # Lister uniquement les variables de base principales (OXYGENE, PH, RENVLT, SALI, TEMP)
    # pour garantir une intersection d'IDs de capteurs riche et cohérente
    base_vars = ["OXYGENE", "PH", "RENVLT", "SALI", "TEMP"]
    all_xlsx_files = sorted([
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".xlsx") and any(f"RomaneDataSet_{bv}.xlsx" == f for bv in base_vars)
    ])
    
    files_to_process = []
    for filename in all_xlsx_files:
        xlsx_path = os.path.join(DATA_DIR, filename)
        csv_filename = filename.replace(".xlsx", ".csv")
        csv_path = os.path.join(CLEAN_DIR, csv_filename)
        
        # Régénérer le CSV s'il est manquant ou s'il a été mis à jour
        need_regen = not os.path.exists(csv_path) or os.path.getmtime(xlsx_path) > os.path.getmtime(csv_path)
        
        if need_regen:
            try:
                if os.path.getsize(xlsx_path) > 1000:
                    df_var = pd.read_excel(xlsx_path)
                    if ID_COLUMN in df_var.columns:
                        cols_data = [c for c in df_var.columns if c != ID_COLUMN and not str(c).startswith('Unnamed')]
                        if cols_data:
                            # Reconstruire un DataFrame propre avec elev_id en toute première colonne (A1)
                            df_clean = df_var[[ID_COLUMN] + cols_data].copy()
                            df_clean.columns = [ID_COLUMN] + [str(i) for i in range(len(cols_data))]
                            # Enregistrer au format CSV
                            df_clean.to_csv(csv_path, index=False)
                            print(f"    - [REGEN] {filename} -> {csv_filename} (Nettoyé)")
            except Exception as e:
                print(f"    [!] Erreur sur {filename} : {e}")
                
        if os.path.exists(csv_path):
            files_to_process.append(csv_filename)
            
    if not files_to_process:
        print(f"[!] Aucun fichier CSV valide n'a pu être généré dans {CLEAN_DIR}.")
        sys.exit(1)
        
    print(f"[+] {len(files_to_process)} fichiers CSV nettoyés prêts pour l'analyse.")
    
    # 1. INITIALISATION DU SCHEDULER GPU
    from gpusync import Scheduler, list_gpus
    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU compatible OpenCL détecté. MMTS requiert un GPU pour fonctionner.")
        sys.exit(1)
        
    scheduler = Scheduler(gpu=gpus[0])
    scheduler.initialize()
    print(f"[+] GPU initialisé : {gpus[0].name}")
    
    # 2. RUN PIPELINE MONO-VARIABLE ET COEFFICIENTS POUR CHAQUE FICHIER CSV
    results_dft = {}
    dicoCoff = {}
    
    t_start = time.perf_counter()
    
    for filename in files_to_process:
        var_name = filename.replace("RomaneDataSet_", "").replace(".csv", "")
        file_path = os.path.join(CLEAN_DIR, filename)
        
        print(f"\n[+] Traitement de la variable {var_name} (Fichier CSV propre)...")
        df_clean = pd.read_csv(file_path)
        
        # Limiter à 200 séries pour les besoins de la démo rapide
        limit_n = 200
        if len(df_clean) > limit_n:
            df_clean = df_clean.head(limit_n)
            
        data_matrix = df_clean.drop([ID_COLUMN], axis=1).values.astype(np.float32)
        listID_var = pd.DataFrame({ID_COLUMN: df_clean[ID_COLUMN]})
        
        # Calculer le coefficient d'entropie de cette variable
        print(f"    - Calcul du coefficient d'entropie...")
        coeffs = defineCoeff(
            data=list(data_matrix),
            num_clust=3,
            num_iter=1,
            listDataFrame=listID_var,
            LastnbClust=0,
            tabEnviIndice="Set_" + var_name,
            assignmentsResult={},
            centroidsCompare={},
            distCompare={},
            data22=None,
            listeIDOk1=[],
            nbInstanceMinim=2,
            w=5,
            scheduler=scheduler
        )
        coeffRap = np.max(coeffs) if coeffs else 1.0
        dicoCoff["Set_" + var_name] = coeffs
        print(f"      -> Coefficient d'entropie retenu : {coeffRap:.4f}")
        
        # Exécuter le clustering mono-variable
        print(f"    - Exécution du clustering X-means GPU...")
        engine = XmeansTS(data_matrix, numclust=3, file=var_name, listID=listID_var, coeffRap=coeffRap, scheduler=scheduler)
        
        dist_compare = {}
        liste_ok = []
        mat_dist = engine.k_means_clust(
            data=list(data_matrix),
            num_clust=3,
            num_iter=1,
            listDataFrame=listID_var.copy(),
            LastnbClust=0,
            tabEnviIndice="Set_" + var_name,
            distCompare=dist_compare,
            listeIDOk1=liste_ok,
            nbInstanceMinim=2,
            w=5
        )
        results_dft[var_name] = mat_dist
        print(f"      -> Clusters identifiés : {len(engine.centroids)} | Distances : {mat_dist.shape}")
        
    # 3. FUSION DES RÉSULTATS PAR ID
    print(f"\n[+] Fusion des distances de toutes les variables...")
    merged_df = None
    for var_name, mat_dist in results_dft.items():
        if merged_df is None:
            merged_df = mat_dist
        else:
            merged_df = pd.merge(merged_df, mat_dist, on=ID_COLUMN, how="inner")
            
    print(f"    -> Dimension du dataset fusionné : {merged_df.shape}")
    
    # 4. RUN PIPELINE MULTI-VARIABLES (MultiXmeansTS)
    print(f"\n[+] Exécution du pipeline multi-variables (KmeanMutiShape)...")
    multi = MultiXmeansTS()
    multi.scheduler = scheduler
    multi.files = ["Set_" + v for v in results_dft.keys()]
    multi.data_directory = CLEAN_DIR
    multi.file_prefix = "RomaneDataSet_"
    
    assignmentsOk = {}
    cols_to_cluster = [c for c in merged_df.columns if c != ID_COLUMN]
    
    multi.KmeanMutiShape(
        nbClust=3,
        columns=cols_to_cluster,
        dft=merged_df,
        dicoCoff=dicoCoff,
        assignmentsOk=assignmentsOk,
        indiceSd=0,
        w=5
    )
    
    t_total = time.perf_counter() - t_start
    print(f"\n" + "=" * 80)
    print(f" RÉPARTITION FINALE DE L'ESSAIM (Pipeline complété en {t_total:.2f} s)")
    print("=" * 80)
    for idx_sd, member_ids in assignmentsOk.items():
        print(f"   Essaim Cluster {idx_sd} : {len(member_ids):>3d} séries (ex: {member_ids[:4]})")
    print("=" * 80)
    print("\n[SUCCESS] Analyse complète exécutée avec succès sur tout le dossier dataset !")
    print("=" * 80)
