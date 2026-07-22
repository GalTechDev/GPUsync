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


def DTWDistance(s1, s2, w):
    """
    Calcule la distance DTW entre deux séries temporelles
    
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
                    bar = euclidean_barycenter([values])
                    flat_list = [item for sublist in bar for item in sublist]
                
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
                        new_centroid = np.zeros(len(data[0]))
                        for i in assignments[c]:
                            if len(data[i]) > 0:
                                new_centroid += data[i]
                        centroids[c] = new_centroid / len(assignments[c])
                
                testStop = True  # On sort après une itération
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
        Récupère la matrice finale des résultats - Conforme à l'original
        Crée une colonne par centroïde avec les distances DTW
        
        Args:
            tabEnviIndice: Index de l'environnement
            w: Paramètre de fenêtre pour DTW
            
        Returns:
            DataFrame avec colonnes: elev_id, 0_Set_TEMP, 1_Set_TEMP, etc.
        """
        
        MatriceDist = pd.DataFrame()
        
        # Pour chaque centroïde, calculer la distance DTW avec toutes les séries
        for uit in self.centroids.keys():
            listTem = []
            for ine in range(len(self.data)):
                listTem.append(DTWDistance(self.centroids[uit], self.data[ine], w))
            MatriceDist[str(uit) + '_' + tabEnviIndice] = listTem
        
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
