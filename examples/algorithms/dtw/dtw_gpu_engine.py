import sys
import os

# Permet d'importer le dossier gpusync parent si besoin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.compiler import gpusync_kernel

class constant_list(list): pass

def mad(a, b, c):
    return a * b + c

def unrolled_range(*args):
    return range(*args)

def get_sliding_dtw_kernel(M):
    """
    Génère un noyau GPU compilé (gpusync_kernel) pour le calcul DTW avec fenêtre glissante.
    Ce noyau est optimisé (unrolled) pour rechercher un motif de taille M 
    dans une série temporelle continue (1D).
    """
    @gpusync_kernel(constants={"M_CONST": M})
    def dtw_sliding_kernel(ctx: int, query: constant_list, T: list, distances: list, chunk_offset: int):
        # L'index de départ dans la base temporelle T pour ce thread
        start_idx: int = ctx
        
        # Création d'une ligne de matrice DTW dans les registres privés
        row = [0.0] * M_CONST
        
        # Remplissage de la première colonne (j=0)
        t_val_0 = T[start_idx]
        
        d = query[0] - t_val_0
        d = d * d if d > 0 else d * d
        row[0] = d
        
        for j in unrolled_range(1, M_CONST):
            d = query[j] - t_val_0
            row[j] = row[j - 1] + d * d
            
        for i in unrolled_range(1, M_CONST):
            t_val = T[start_idx + i]
            d = query[0] - t_val
            
            m3 = row[0]
            row[0] = row[0] + d * d
            m2 = row[0]
            
            for j in unrolled_range(1, M_CONST):
                d = query[j] - t_val
                m1 = row[j]
                
                min_val = m1
                if m2 < min_val: min_val = m2
                if m3 < min_val: min_val = m3
                
                m3 = row[j]
                row[j] = min_val + d * d
                m2 = row[j]
                
        distances[ctx] = row[M_CONST - 1]

    return dtw_sliding_kernel


def get_batch_dtw_kernel(M, num_refs):
    """
    Génère un noyau GPU pour le calcul DTW en batch (comparaison de N séries cibles contre K références).
    Utilise le Constant Cache pour les références et un déroulage (unroll) complet de la matrice DTW.
    """
    @gpusync_kernel(constants={"M_CONST": M, "NUM_REFS_CONST": num_refs})
    def batch_dtw_fused_constant(ctx: int, references: constant_list, targets: list, targets_offset: int, seq_length: int, outputs: list, num_refs_arg: int, chunk_size: int):
        if ctx >= chunk_size:
            return
            
        target_idx: int = targets_offset + ctx
        target_start: int = target_idx * M_CONST
        
        tgt_local = [0.0] * M_CONST
        for j in unrolled_range(0, M_CONST):
            tgt_local[j] = targets[target_start + j]
            
        # Boucle sur les références
        for r in range(NUM_REFS_CONST):
            ref_start: int = r * M_CONST
            
            ref_local = [0.0] * M_CONST
            for i in unrolled_range(0, M_CONST):
                ref_local[i] = references[ref_start + i]
                
            row = [0.0] * M_CONST
            
            ref_val = ref_local[0]
            d = ref_val - tgt_local[0]
            row[0] = d * d
            
            for j in unrolled_range(1, M_CONST):
                d = ref_val - tgt_local[j]
                row[j] = mad(d, d, row[j - 1])
                
            for i in unrolled_range(1, M_CONST):
                ref_val = ref_local[i]
                d = ref_val - tgt_local[0]
                
                m3 = row[0] 
                row[0] = mad(d, d, row[0]) 
                m2 = row[0] 
                
                for j in unrolled_range(1, M_CONST):
                    d = ref_val - tgt_local[j]
                    m1 = row[j]
                    
                    min_val = m1
                    if m2 < min_val: min_val = m2
                    if m3 < min_val: min_val = m3
                    
                    m3 = row[j] 
                    row[j] = mad(d, d, min_val)
                    m2 = row[j] 
                    
            outputs[r * chunk_size + ctx] = row[M_CONST - 1]

    return batch_dtw_fused_constant


def get_sliding_dtw_multi_kernel(M, num_queries):
    """
    Génère un noyau GPU pour le calcul DTW multi-requêtes en fenêtre glissante (Shape Dictionary).
    - M: Taille du motif
    - num_queries: Nombre de motifs dans le dictionnaire
    
    Chaque thread gère une fenêtre de T et calcule la distance avec *toutes* les requêtes.
    """
    @gpusync_kernel(constants={"M_CONST": M, "NUM_QUERIES_CONST": num_queries})
    def dtw_sliding_multi_kernel(ctx: int, queries: constant_list, T: list, distances: list, chunk_offset: int):
        start_idx: int = ctx
        
        # On lit la séquence temporelle cible (T) depuis la mémoire globale une seule fois
        # vers des registres (tableau local)
        tgt_local = [0.0] * M_CONST
        for i in unrolled_range(0, M_CONST):
            tgt_local[i] = T[start_idx + i]
            
        # Boucle sur les requêtes
        for q in range(NUM_QUERIES_CONST):
            q_start: int = q * M_CONST
            
            # Matrice DTW virtuelle (1 ligne)
            row = [0.0] * M_CONST
            
            # Remplissage colonne j=0
            t_val_0 = tgt_local[0]
            
            d = queries[q_start] - t_val_0
            d = d * d if d > 0 else d * d
            row[0] = d
            
            for j in unrolled_range(1, M_CONST):
                d = queries[q_start + j] - t_val_0
                row[j] = row[j - 1] + d * d
                
            for i in unrolled_range(1, M_CONST):
                t_val = tgt_local[i]
                d = queries[q_start] - t_val
                
                m3 = row[0]
                row[0] = row[0] + d * d
                m2 = row[0]
                
                for j in unrolled_range(1, M_CONST):
                    d = queries[q_start + j] - t_val
                    m1 = row[j]
                    
                    min_val = m1
                    if m2 < min_val: min_val = m2
                    if m3 < min_val: min_val = m3
                    
                    m3 = row[j]
                    row[j] = min_val + d * d
                    m2 = row[j]
                    
            # L'output est un tableau (Q * N_windows) ou Q est le num de query et N_windows la taille du chunk
            distances[q * chunk_offset + ctx] = row[M_CONST - 1]
            
    return dtw_sliding_multi_kernel


def get_batch_dtw_window_kernel(M, num_refs):
    """
    Génère un noyau GPU pour le calcul DTW en batch avec contrainte de fenêtre (Sakoe-Chiba band).
    Optimisé via double-buffering de lignes pour éviter les copies et la réinitialisation complète des tableaux.
    """
    @gpusync_kernel(constants={"M_CONST": M, "M_CONST_PAD": M + 1, "M_CONST_PAD_2": 2 * (M + 1), "NUM_REFS_CONST": num_refs})
    def batch_dtw_fused_window(ctx: int, references: constant_list, targets: list, targets_offset: int, w: int, outputs: list, num_refs_arg: int, chunk_size: int):
        if ctx >= chunk_size:
            return
            
        target_idx: int = targets_offset + ctx
        target_start: int = target_idx * M_CONST
        
        tgt_local = [0.0] * M_CONST
        for j in unrolled_range(0, M_CONST):
            tgt_local[j] = targets[target_start + j]
            
        # Boucle sur les références
        for r in range(NUM_REFS_CONST):
            ref_start: int = r * M_CONST
            
            ref_local = [0.0] * M_CONST
            for i in unrolled_range(0, M_CONST):
                ref_local[i] = references[ref_start + i]
                
            # dp de taille 2 * M_CONST_PAD
            dp = [0.0] * M_CONST_PAD_2
            for j in unrolled_range(0, M_CONST_PAD_2):
                dp[j] = 1e20
                
            dp[0] = 0.0 # dp[0][0] = 0.0
            
            for i in range(1, M_CONST_PAD):
                curr_idx: int = int(i % 2)
                prev_idx: int = int(1 - curr_idx)
                
                curr_offset: int = int(curr_idx * M_CONST_PAD)
                prev_offset: int = int(prev_idx * M_CONST_PAD)
                
                # Réinitialiser la colonne de garde à l'infini
                dp[curr_offset] = 1e20
                
                # Déterminer les bornes
                j_start: int = int(i - w)
                if j_start < 1:
                    j_start = 1
                j_end: int = int(i + w + 1)
                if j_end > M_CONST_PAD:
                    j_end = M_CONST_PAD
                    
                # Réinitialiser uniquement les cellules frontières à l'infini
                if j_start > 1:
                    dp[int(curr_offset + j_start - 1)] = 1e20
                if j_end < M_CONST_PAD:
                    dp[int(curr_offset + j_end)] = 1e20
                    
                for j in range(j_start, j_end):
                    d = ref_local[i - 1] - tgt_local[j - 1]
                    cost = d * d
                    
                    min_val = dp[int(prev_offset + j)]          # Insertion
                    if dp[int(curr_offset + j - 1)] < min_val:  # Deletion
                        min_val = dp[int(curr_offset + j - 1)]
                    if dp[int(prev_offset + j - 1)] < min_val:  # Match
                        min_val = dp[int(prev_offset + j - 1)]
                        
                    dp[int(curr_offset + j)] = cost + min_val
                    
            final_idx: int = int(M_CONST % 2)
            outputs[int(r * chunk_size + ctx)] = dp[int(final_idx * M_CONST_PAD + M_CONST)]

    return batch_dtw_fused_window
