# -*- coding: utf-8 -*-
"""
Module d'analyse pour le clustering de séries temporelles avec X-means et DTW.
Version optimisée 100% basée sur GPUsync OpenCL (sans aucun fallback CPU ni Numba).
Garantit la compatibilité transparente pour l'intégration dans l'autre projet.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import sys
import time
import math
import random
from scipy import stats
from sklearn.cluster import KMeans

# Ajout des chemins de recherche système pour importer gpusync
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fallback des constantes de configuration pour assurer l'indépendance
try:
    from config import DEFAULT_W, MAX_ITERATIONS, ID_COLUMN, ROMANE_DATA_PATH
except ImportError:
    DEFAULT_W = 5
    MAX_ITERATIONS = 5
    ID_COLUMN = 'elev_id'
    ROMANE_DATA_PATH = 'RomaneData'

# Variables globales pour GPUsync
_gpusync_sched = None
_gpusync_kernels_cache = {}

def get_gpusync_resources(M, num_refs, scheduler=None):
    """
    Récupère ou initialise le scheduler GPUsync et le noyau DTW GPU.
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
        from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel
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


def DTWDistance(s1, s2, w=DEFAULT_W, scheduler=None):
    """
    Calcule la distance DTW unitaire sur GPU avec GPUsync.
    """
    from gpusync import AsyncBuffer
    M = max(len(s1), len(s2))
    M_pad = int(2 ** math.ceil(math.log2(M)))
    M_pad = max(32, M_pad)
    sched, kernel = get_gpusync_resources(M_pad, 1, scheduler)
    s1_pad = np.zeros(M_pad, dtype=np.float32)
    s1_pad[:len(s1)] = s1
    s2_pad = np.zeros(M_pad, dtype=np.float32)
    s2_pad[:len(s2)] = s2
    buf_ref = AsyncBuffer.to_device(sched, s1_pad)
    buf_tgt = AsyncBuffer.to_device(sched, s2_pad)
    buf_out = AsyncBuffer.empty_like(sched, np.zeros(1, dtype=np.float32))
    kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, 1, 1, global_size=(1,), scheduler=sched)
    return float(buf_out.to_host()[0])


def _batch_dtw_gpu(refs_np, targets_np, w, scheduler=None):
    """
    Calcule la matrice de distances DTW sur GPU avec GPUsync.
    """
    from gpusync import AsyncBuffer
    K, M_len = refs_np.shape
    N = targets_np.shape[0]
    M_pad = int(2 ** math.ceil(math.log2(M_len)))
    M_pad = max(32, M_pad)
    sched, kernel = get_gpusync_resources(M_pad, K, scheduler)
    refs_pad = np.zeros((K, M_pad), dtype=np.float32)
    refs_pad[:, :M_len] = refs_np
    tgts_pad = np.zeros((N, M_pad), dtype=np.float32)
    tgts_pad[:, :M_len] = targets_np
    buf_ref = AsyncBuffer.to_device(sched, refs_pad.flatten())
    buf_tgt = AsyncBuffer.to_device(sched, tgts_pad.flatten())
    buf_out = AsyncBuffer.empty_like(sched, np.zeros(K * N, dtype=np.float32))
    kernel(buf_ref, buf_tgt, 0, M_pad, buf_out, K, N, global_size=(N,), scheduler=sched)
    return np.sqrt(np.maximum(buf_out.to_host().reshape(K, N), 0.0))


def _compute_dist_matrix(centroids_np, targets_np, w, scheduler=None):
    return _batch_dtw_gpu(centroids_np, targets_np, w, scheduler)


class XmeansTS:
    """
    Algorithme X-means avec DTW optimisé pour le clustering de séries temporelles (GPU exclusif).
    """
    def __init__(self, data, numclust, file, listID, coeffRap, scheduler=None):
        self.assignments = {}
        self.dfListID = listID.reset_index(drop=True)
        self.data = data
        self.numclust = numclust
        self.centroids = {}
        self.file = file
        self.coeffRapport = coeffRap
        
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

    def reductMemory(self):
        pass

    def kmeansConverg(self, data, centroids, listID, w):
        if not data or not centroids:
            return True
        refs = np.array(list(centroids.values()), dtype=np.float32)
        targets = np.array(data, dtype=np.float32)
        dist_mat = _compute_dist_matrix(refs, targets, w, self.scheduler)
        return float(dist_mat.min(axis=0).sum()) < 0.001

    def updateCentroide(self, data, centroids, assignments):
        data_np = np.array(data, dtype=np.float32)
        for c in assignments:
            idx = assignments[c]
            if idx:
                centroids[c] = data_np[idx].mean(axis=0)
        return centroids

    def computeDist(self, data, listID, centroids, assignments, w):
        for c in assignments:
            assignments[c] = []
        if not data or not centroids:
            return assignments
        refs = np.array(list(centroids.values()), dtype=np.float32)
        targets = np.array(data, dtype=np.float32)
        dist_mat = _compute_dist_matrix(refs, targets, w, self.scheduler)
        labels = np.argmin(dist_mat, axis=0)
        for i, c in enumerate(labels):
            assignments[int(c)].append(i)
        return assignments

    def initAtt(self, data, centroids, listID):
        return {i: [] for i in range(len(centroids))}

    def kmeans(self, data, listDataFrame, w):
        centroids = {i: data[i] for i in range(self.numclust)}
        assignments = self.initAtt(data, centroids, listDataFrame)
        for _ in range(MAX_ITERATIONS):
            assignments = self.computeDist(data, listDataFrame, centroids, assignments, w)
            centroids = self.updateCentroide(data, centroids, assignments)
            if self.kmeansConverg(data, centroids, listDataFrame, w):
                break
        return centroids, assignments

    def selectSerie(self, data, listID, assignments, assignmentsTemp, assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim):
        id_arr = listID[ID_COLUMN].values
        data_np = np.array(data, dtype=np.float32)
        for i in assignments:
            if len(assignments[i]) >= nbInstanceMinim:
                idx = np.array(assignments[i], dtype=np.int32)
                assignmentsTemp[i] = id_arr[idx].tolist()
                assignmentsTempBary[i] = data_np[idx].tolist()
                assignmentsTempBaryDTW[i] = []

    def updateAssignment(self, assignmentsTemp, listeIDOk, nbClustTempp, w=DEFAULT_W):
        id_arr = self.dfListID[ID_COLUMN].values
        data_np = np.array(self.data, dtype=np.float32)
        for key in assignmentsTemp:
            member_ids = assignmentsTemp[key]
            self.assignments[nbClustTempp] = member_ids
            listeIDOk.extend(member_ids)
            if member_ids:
                mask = np.isin(id_arr, member_ids)
                member_data = data_np[mask]
                if len(member_data) > 0:
                    self.centroids[nbClustTempp] = member_data.mean(axis=0)
                nbClustTempp += 1
        return listeIDOk

    def updateVar(self, listFilterID, colID=ID_COLUMN, ranges=None):
        id_arr = self.dfListID[ID_COLUMN].values
        mask = ~np.isin(id_arr, listFilterID)
        dfListIDTemp = self.dfListID[mask].reset_index(drop=True)
        data_out = [self.data[i] for i in np.where(mask)[0]]
        return data_out, dfListIDTemp

    def k_means_clust(self, data, num_clust, num_iter, listDataFrame,
                       LastnbClust, tabEnviIndice, distCompare, listeIDOk1,
                       nbInstanceMinim, w=DEFAULT_W):
        if num_clust == 0:
            return pd.DataFrame(self.dfListID)

        if len(data) > num_clust * 2:
            centroids = {}
            distCompare[tabEnviIndice] = {}

            for i in range(num_clust):
                idx = random.randint(0, len(data) - 1)
                centroids[i] = data[idx]

            assignments = {i: [] for i in range(num_clust)}
            assignmentsTemp = {i: [] for i in range(num_clust)}
            assignmentsTempBary = {i: [] for i in range(num_clust)}
            assignmentsTempBaryDTW = {i: [] for i in range(num_clust)}

            assignments = self.computeDist(data, listDataFrame, centroids, assignments, w)
            self.updateCentroide(data, centroids, assignments)
            self.selectSerie(data, listDataFrame, assignments, assignmentsTemp,
                             assignmentsTempBary, assignmentsTempBaryDTW, nbInstanceMinim)

            nbClustLow = LastnbClust
            listeIDOk = []
            listeIDOk = self.updateAssignment(assignmentsTemp, listeIDOk, LastnbClust, w)
            listeIDOk1 = listeIDOk1 + listeIDOk
            updateVar = self.updateVar(listeIDOk1)

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

        if len(data) < num_clust * 2 or LastnbClust == 0:
            dfListIDTemp = pd.DataFrame(listDataFrame)
            dfListIDTemp["cluster"] = -1
            for ctr in range(len(self.assignments)):
                for elemid in self.assignments[ctr]:
                    dfListIDTemp.loc[dfListIDTemp[ID_COLUMN] == elemid, "cluster"] = ctr
            return dfListIDTemp

    def getLastMat(self, tabEnviIndice, w):
        MatriceDist = pd.DataFrame()
        if not self.centroids or self.data is None or len(self.data) == 0:
            MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN]
            return MatriceDist
        keys_list = list(self.centroids.keys())
        K = len(keys_list)
        M = len(self.data[0])
        refs_np = np.zeros((K, M), dtype=np.float32)
        for c_idx, c_key in enumerate(keys_list):
            v = self.centroids[c_key]
            refs_np[c_idx, :len(v)] = v
        targets_np = np.array(self.data, dtype=np.float32)
        
        dist_matrix = _compute_dist_matrix(refs_np, targets_np, w, self.scheduler)
        for c_idx, c_key in enumerate(keys_list):
            MatriceDist[str(c_key) + '_' + tabEnviIndice] = dist_matrix[c_idx, :]
        MatriceDist[ID_COLUMN] = self.dfListID[ID_COLUMN].values
        return MatriceDist

    def defineCoeff(self, listID, data, num_clust, nbInstanceMinim, w=1):
        return [1.0]


class MultiXmeansTS:
    """
    Analyse multi-variable avec X-means et distances DTW (GPU exclusif).
    """
    def __init__(self, scheduler=None):
        self.XmeansTSVars = [XmeansTS]
        self.centroids = {}
        self.assignmentsMulti = {}
        self.files = []
        
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

    def KmeanMutiShape(self, nbClust, columns, dft, dicoCoff, assignmentsOk, indiceSd, w=DEFAULT_W):
        if len(dft) > nbClust * 2:
            dftmp = dft.copy()
            kmeans = KMeans(n_clusters=nbClust)
            y = kmeans.fit(dftmp[list(columns)].values)
            dftmp['Cluster'] = y.labels_

            for clust in dftmp['Cluster'].unique():
                retireId = True
                dftTemp = dftmp[dftmp['Cluster'] == clust]
                IdTemp = list(dftTemp[ID_COLUMN])

                for tabEnviIndice in self.files:
                    if hasattr(self, 'data_directory') and hasattr(self, 'file_prefix'):
                        file_path = os.path.join(self.data_directory, f"{self.file_prefix}{tabEnviIndice}.xlsx")
                        possible_paths = [file_path]
                    else:
                        file_name = tabEnviIndice
                        if file_name.startswith("Set_"):
                            file_name = file_name[4:]
                        possible_paths = [
                            os.path.join(os.path.dirname(os.getcwd()), f"RomaneDataSet_{file_name}.xlsx"),
                            os.path.join(os.getcwd(), f"RomaneDataSet_{file_name}.xlsx"),
                            ROMANE_DATA_PATH + tabEnviIndice + '.xlsx',
                            ROMANE_DATA_PATH + file_name + '.xlsx'
                        ]

                    file_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            file_path = path
                            break
                    if file_path is None:
                        raise FileNotFoundError(f"Fichier introuvable pour {tabEnviIndice}.")

                    xlsQualEnvi = pd.ExcelFile(file_path)
                    dataTemp = xlsQualEnvi.parse(0)
                    dataTemp = dataTemp[dataTemp[ID_COLUMN].isin(IdTemp)]
                    dataTemp1 = dataTemp.drop([ID_COLUMN], axis=1)
                    valuesMat = dataTemp1.values

                    listpointCentr = [sum(k) / len(dataTemp1) for k in valuesMat]
                    listpoint = [DTWDistance(k, listpointCentr, w, self.scheduler) for k in valuesMat]

                    stdTemp = np.std(listpoint)
                    entropy = stats.entropy(listpoint)

                    if tabEnviIndice in dicoCoff:
                        if (stdTemp / entropy) > dicoCoff[tabEnviIndice][0]:
                            retireId = False

                if not retireId:
                    indiceSd += 1
                    self.KmeanMutiShape(nbClust, columns, dftTemp.copy(), dicoCoff, assignmentsOk, int(indiceSd), w=w)
                else:
                    assignmentsOk[indiceSd] = IdTemp
                    indiceSd += 1
        else:
            assignmentsOk[indiceSd] = list(dft[ID_COLUMN])
            indiceSd += 1


def defineCoeff(data, num_clust, num_iter, listDataFrame, LastnbClust, tabEnviIndice, assignmentsResult, 
                centroidsCompare, distCompare, data22, listeIDOk1, nbInstanceMinim, w=1, scheduler=None):
    """
    Fonction externe pour définir les coefficients de l'analyse avec support GPU exclusif.
    """
    coeffTemp = []
    centroids = {}
    for i in range(num_clust):
        idx = random.randint(0, len(data) - 1)
        centroids[i] = np.asarray(data[idx], dtype=np.float32)

    refs_np = np.array(list(centroids.values()), dtype=np.float32)
    targets_np = np.array(data, dtype=np.float32)

    dist_mat = _compute_dist_matrix(refs_np, targets_np, w, scheduler)

    labels = np.argmin(dist_mat, axis=0)
    assignments = {i: [] for i in range(num_clust)}
    for i, c in enumerate(labels):
        assignments[int(c)].append(i)

    for i in assignments:
        if len(assignments[i]) >= nbInstanceMinim:
            idx_arr = np.array(assignments[i], dtype=np.int32)
            member_data = targets_np[idx_arr]
            if len(member_data) > 0:
                centroid = member_data.mean(axis=0).reshape(1, -1)
                dists_row = _compute_dist_matrix(centroid, member_data, w, scheduler)[0]
                stdTemp = float(np.std(dists_row))
                entropy = float(stats.entropy(dists_row)) if np.any(dists_row > 0) else 1.0
                if entropy > 0:
                    coeffTemp.append(stdTemp / entropy)

    return coeffTemp if coeffTemp else [1.0]


if __name__ == "__main__":
    print("=" * 80)
    print(" EXECUTION COMPLET DU PIPELINE MMTS SUR DONNEES REELLES (GPU ONLY)")
    print("=" * 80)

    # Résolution dynamique des chemins
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(HERE, "data")
    
    ROMANE_DATA_PATH = os.path.join(DATA_DIR, "RomaneDataSet_")
    
    file_csv = os.path.join(DATA_DIR, "RomaneDataSet_TEMP.csv")
    file_temp = os.path.join(DATA_DIR, "RomaneDataSet_TEMP.xlsx")
    
    if not os.path.exists(file_csv):
        print(f"[!] Fichier CSV source introuvable : {file_csv}. Fin.")
        sys.exit(1)
        
    print(f"[+] Conversion de {file_csv} en fichier Excel de production structuré par fenêtres glissantes...")
    df_csv = pd.read_csv(file_csv)
    raw = df_csv["value"].values.astype(np.float32)
    WINDOW = 32
    N_all = len(raw) - WINDOW + 1
    
    # Limiter le nombre de séries pour accélérer le test final
    limit_n = 200
    if N_all > limit_n:
        N_all = limit_n
        
    data_list = [raw[i : i + WINDOW] for i in range(N_all)]
    
    # Création du DataFrame structuré
    df_temp = pd.DataFrame(data_list)
    df_temp.columns = [f"val_{i}" for i in range(WINDOW)]
    df_temp.insert(0, ID_COLUMN, [f"TS_{i:04d}" for i in range(N_all)])
    
    # Enregistrer au format Excel
    df_temp.to_excel(file_temp, index=False)
    print(f"    - TEMP : {N_all} séries converties et écrites dans {file_temp}")
    
    # Extraction des données et préparation des listes d'IDs
    data_temp = df_temp.drop([ID_COLUMN], axis=1).values.astype(np.float32)
    listID_temp = pd.DataFrame({ID_COLUMN: df_temp[ID_COLUMN]})
    
    # 1. INITIALISATION DU SCHEDULER ET DU MOTEUR X-MEANS GPU
    from gpusync import Scheduler, list_gpus
    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU compatible OpenCL détecté. MMTS requiert un GPU pour fonctionner.")
        sys.exit(1)
        
    scheduler = Scheduler(gpu=gpus[0])
    scheduler.initialize()
    print(f"[+] GPU initialisé : {gpus[0].name}")
    
    # 2. RUN PIPELINE MONO-VARIABLE SUR TEMP
    print(f"\n[1/3] Démarrage du clustering X-means (mono-variable) sur TEMP (K=3, W=5)...")
    t0 = time.perf_counter()
    engine = XmeansTS(data_temp, numclust=3, file="TEMP", listID=listID_temp, coeffRap=1.0, scheduler=scheduler)
    
    dist_compare = {}
    liste_ok = []
    
    # Exécuter le clustering récursif
    mat_dist = engine.k_means_clust(
        data=list(data_temp),
        num_clust=3,
        num_iter=1,
        listDataFrame=listID_temp.copy(),
        LastnbClust=0,
        tabEnviIndice="Set_TEMP",
        distCompare=dist_compare,
        listeIDOk1=liste_ok,
        nbInstanceMinim=2,
        w=5
    )
    t_mono = time.perf_counter() - t0
    print(f"    -> Clustering mono-variable terminé en {t_mono*1000:.2f} ms")
    print(f"    -> Nombre de clusters identifiés : {len(engine.centroids)}")
    print(f"    -> Dimensions de la matrice de distances : {mat_dist.shape}")
    
    # 3. CALCUL DES COEFFICIENTS D'ENTROPIE (defineCoeff)
    print(f"\n[2/3] Calcul des coefficients d'entropie (defineCoeff) sur TEMP...")
    t0 = time.perf_counter()
    coeffs = defineCoeff(
        data=list(data_temp),
        num_clust=3,
        num_iter=1,
        listDataFrame=listID_temp,
        LastnbClust=0,
        tabEnviIndice="Set_TEMP",
        assignmentsResult={},
        centroidsCompare={},
        distCompare={},
        data22=None,
        listeIDOk1=[],
        nbInstanceMinim=2,
        w=5,
        scheduler=scheduler
    )
    t_coeff = time.perf_counter() - t0
    print(f"    -> Coefficients calculés en {t_coeff*1000:.2f} ms : {coeffs}")

    # 4. RUN PIPELINE MULTI-VARIABLES (MultiXmeansTS)
    print(f"\n[3/3] Exécution du pipeline multi-variables (KmeanMutiShape) avec TEMP...")
    t0 = time.perf_counter()
    
    multi = MultiXmeansTS(scheduler=scheduler)
    multi.files = ["Set_TEMP"]
    
    dicoCoff = {"Set_TEMP": coeffs}
    assignmentsOk = {}
    cols_to_cluster = [c for c in mat_dist.columns if c != ID_COLUMN]
    
    multi.KmeanMutiShape(
        nbClust=3,
        columns=cols_to_cluster,
        dft=mat_dist,
        dicoCoff=dicoCoff,
        assignmentsOk=assignmentsOk,
        indiceSd=0,
        w=5
    )
    t_multi = time.perf_counter() - t0
    print(f"    -> Clustering multi-variable terminé en {t_multi*1000:.2f} ms")
    print("    -> Répartition finale des séries dans les clusters de l'essaim :")
    print("-" * 80)
    for idx_sd, member_ids in assignmentsOk.items():
        print(f"       Essaim Cluster {idx_sd} : {len(member_ids):>3d} séries (ex: {member_ids[:4]})")
    print("-" * 80)
    
    print("\n[SUCCESS] Pipeline MMTS complet exécuté avec succès sur GPU via GPUsync !")
    print("=" * 80)
