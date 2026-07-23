# -*- coding: utf-8 -*-
"""
Module d'analyse pour le clustering de séries temporelles avec X-means et DTW
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import math
import random
from scipy import stats
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from sklearn.cluster import KMeans

# Configuration pour OpenCL si disponible (pour la performance)
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

# Import d'opencldtw si disponible (sinon on utilisera la version pure Python)
try:
    import opencldtw
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

# Importation de la configuration
from config import *


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


# Import de la version optimisée DTW
try:
    from dtw_optimized import dtw_distance_vectorized as DTWDistance
    print("DTW optimisé chargé avec succès")
except ImportError:
    # Fallback vers l'implémentation originale si le module optimisé n'est pas disponible
    def DTWDistance(s1, s2, w):
        """
        Calcule la distance DTW entre deux séries temporelles (version originale)
        
        Args:
            s1: Première série temporelle
            s2: Deuxième série temporelle
            w: Fenêtre de contrainte pour l'alignement temporel
            
        Returns:
            Distance DTW entre les deux séries
        """
        DTW={}
        
        w = max(w, abs(len(s1)-len(s2)))
        
        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = float('inf')
        DTW[(-1, -1)] = 0
      
        for i in range(len(s1)):
            for j in range(max(0, i-w), min(len(s2), i+w+1)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
        
        return np.sqrt(DTW[len(s1)-1, len(s2)-1])
    
    print("DTW original utilisé (fallback)")


class XmeansTS:
    """
    Classe pour l'algorithme X-means avec DTW pour le clustering de séries temporelles
    """
    
    def __init__(self, data, numclust, file, listID, coeffRap, representative_mode="euclidean"):
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
            best_idx = 0
            best_score = float("inf")
            for idx, candidate in enumerate(clean_values):
                score = 0.0
                for other in clean_values:
                    score += DTWDistance(candidate, other, w)
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
    
    def capture_convergence_iteration(self, iteration, centroids, assignments, matrix_data=None):
        """
        Capture une itération de convergence pour analyse LLM
        
        Args:
            iteration: Numéro de l'itération
            centroids: Dictionnaire des centroïdes actuels
            assignments: Dictionnaire des assignations
            matrix_data: Matrice des distances (optionnel)
        """
        import time
        import json
        import os
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        variable_name = self.file.replace(" ", "_")
        
        # Calculer les métriques de convergence
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
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/matrices", exist_ok=True)
        os.makedirs(f"{base_path}/metrics", exist_ok=True)
        
        # Sauvegarder en JSON (principal)
        json_path = f"{base_path}/matrices/iteration_{iteration}_matrix.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(iteration_data, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder la matrice complète en CSV (debug optionnel)
        if self.save_debug_csv and matrix_data is not None:
            csv_path = f"{base_path}/matrices/iteration_{iteration}_debug.csv"
            df_matrix = pd.DataFrame(matrix_data, 
                                   columns=[f"Cluster_{i}" for i in range(matrix_data.shape[1])],
                                   index=[f"Series_{i}" for i in range(matrix_data.shape[0])])
            df_matrix.to_csv(csv_path)
        
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
        metrics_df.to_csv(metrics_path, index=False)
        
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
        import pandas as pd
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
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.convergence_data, f, indent=2, ensure_ascii=False)
        
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
        summary_df.to_csv(summary_path, index=False)
        
        print(f"[{variable_name}] Résumé de convergence sauvegardé: {len(self.convergence_data['iterations'])} itérations")
        
        return json_path
    
    def compute_distances_to_centroids_parallel(self, data, centroids, w, n_processes=None):
        """
        Calcule les distances de chaque série à chaque centroïde en parallèle
        
        Args:
            data: Données (liste de séries)
            centroids: Dictionnaire des centroïdes
            w: Paramètre de fenêtre pour DTW
            n_processes: Nombre de processus (auto si None)
            
        Returns:
            Matrice des distances (n_series x n_centroids)
        """
        try:
            from dtw_optimized import compute_dtw_to_centroids_parallel
            
            # Préparation des données
            series_list = [series[0] if len(series) > 0 else np.array([]) for series in data]
            centroid_list = list(centroids.values())
            
            if n_processes is None:
                n_processes = min(mp.cpu_count(), len(series_list))
            
            print(f"Calcul parallèle distances: {len(series_list)} séries x {len(centroid_list)} centroïdes")
            
            return compute_dtw_to_centroids_parallel(series_list, centroid_list, w, n_processes)
            
        except ImportError:
            # Fallback vers la version séquentielle originale
            print("Utilisation du calcul séquentiel (fallback)")
            n_series = len(data)
            n_centroids = len(centroids)
            distance_matrix = np.full((n_series, n_centroids), np.inf)
            
            centroid_list = list(centroids.values())
            
            for i in range(n_series):
                if len(data[i]) > 0:
                    for j, centroid in enumerate(centroid_list):
                        if w > 0:
                            distance_matrix[i, j] = DTWDistance(data[i], centroid, w)
                        else:
                            distance_matrix[i, j] = np.linalg.norm(data[i] - centroid)
            
            return distance_matrix

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
        Calcule les distances entre points et centroides
        
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
        for i in assignments:
            if len(assignments[i]) >= nbInstanceMinim:
                ids = []
                for item in assignments[i]:
                    ids.append(int(listID.iloc[item][ID_COLUMN]))
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
        for key in assignmentsTemp:
            self.assignments[nbClustTempp] = assignmentsTemp[key]
            flat_list = []
            for elem in self.assignments[nbClustTempp]:
                listeIDOk.append(elem)
                
            if len(self.assignments[nbClustTempp]) > 0:
                values = []
                for elId in self.assignments[nbClustTempp]:
                    idTS = self.data[(self.dfListID[ID_COLUMN] == elId)]
                    if len(idTS) > 0:
                        values.append(idTS[0])
                
                if values:
                    flat_list = list(self._cluster_representative(values, DEFAULT_W))
                
                self.centroids[nbClustTempp] = flat_list
                MatDist = pd.DataFrame(columns=[ID_COLUMN, 'MEAS'])
       
                for elId in self.assignments[nbClustTempp]:
                    idTS = self.data[(self.dfListID[ID_COLUMN] == elId)]
                    if len(idTS) > 0:
                        LBVal = DTWDistance(idTS[0], flat_list, DEFAULT_W)
                        
                        new_row = pd.DataFrame({ID_COLUMN: [int(elId)], 'MEAS': [LBVal]})
                        MatDist = pd.concat([MatDist, new_row], ignore_index=True)
                
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
                
                # Calculer les distances et assigner les points
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
                
                # Mettre à jour les centroides
                for c in assignments:
                    if len(assignments[c]) > 0:
                        centroids[c] = self._cluster_representative(
                            [data[i] for i in assignments[c] if len(data[i]) > 0],
                            w
                        )
                
                # Capturer l'itération de convergence
                self.capture_convergence_iteration(counter, centroids, assignments)
                
                testStop = self.kmeansConverg(data, centroids, listDataFrame, w)
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
        Récupère la matrice finale des résultats - Version optimisée parallèle
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
        
        # Utilisation du calcul parallèle si disponible
        try:
            # Préparation des données pour le calcul parallèle
            series_list = [series[0] if len(series) > 0 else np.array([]) for series in self.data]
            
            print(f"Calcul parallèle getLastMat: {len(series_list)} séries x {len(self.centroids)} centroïdes")
            
            # Calcul parallèle pour chaque centroïde
            for uit in self.centroids.keys():
                centroid = self.centroids[uit]
                
                # Calcul parallèle des distances de ce centroïde à toutes les séries
                try:
                    from dtw_optimized import compute_dtw_to_centroids_parallel
                    distances = compute_dtw_to_centroids_parallel(
                        series_list, [centroid], w, n_processes=min(mp.cpu_count(), len(series_list))
                    )
                    listTem = distances.flatten().tolist()
                except ImportError:
                    # Fallback séquentiel
                    listTem = [DTWDistance(centroid, series, w) for series in self.data]
                
                MatriceDist[str(uit) + '_' + tabEnviIndice] = listTem
            
        except Exception as e:
            print(f"Erreur calcul parallèle, fallback séquentiel: {e}")
            # Version séquentielle originale
            for uit in self.centroids.keys():
                listTem = []
                for ine in range(len(self.data)):
                    listTem.append(DTWDistance(self.centroids[uit], self.data[ine], w))
                MatriceDist[str(uit) + '_' + tabEnviIndice] = listTem
        
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
                    if hasattr(self, 'data_directory') and hasattr(self, 'file_prefix'):
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


def defineCoeff(data, num_clust, num_iter, listDataFrame, LastnbClust, tabEnviIndice, assignmentsResult, centroidsCompare, distCompare, data22, listeIDOk1, nbInstanceMinim, w=1):
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
    
    # Calculer les assignations initiales
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
    
    # Calculer le coefficient basé sur les assignations
    for i in assignments:
        if len(assignments[i]) >= nbInstanceMinim:
            values = []
            for item in assignments[i]:
                values.append(data[item])
            
            if values:
                bar = euclidean_barycenter([values])
                flat_list = [item for sublist in bar for item in sublist]
                
                distances = []
                for item in assignments[i]:
                    dist = DTWDistance(data[item], flat_list, w)
                    distances.append(dist)
                
                stdTemp = np.std(distances)
                entropy = stats.entropy(distances) if any(d > 0 for d in distances) else 1.0
                
                if entropy > 0:
                    coeffTemp.append(stdTemp / entropy)
    
    return coeffTemp if coeffTemp else [1.0]  # Valeur par défaut si aucun coefficient n'est calculé
