import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gpusync import gpusync_kernel

@gpusync_kernel
def batch_dtw_kernel(ctx: int, reference: list, targets: list, targets_offset: int, seq_length: int) -> tuple:
    """
    Kernel DTW batch pour le calcul de distance entre une série de référence et N séries cibles.
    """
    target_idx: int = targets_offset + ctx
    target_start: int = target_idx * seq_length
    
    tgt_local = [0.0] * 30
    for j in range(0, seq_length):
        tgt_local[j] = targets[target_start + j]
        
    row = [0.0] * 30
    ref_val = reference[0]
    d = ref_val - tgt_local[0]
    row[0] = d * d
    
    for j in range(1, seq_length):
        d = ref_val - tgt_local[j]
        row[j] = row[j - 1] + d * d
        
    for i in range(1, seq_length):
        ref_val = reference[i]
        d = ref_val - tgt_local[0]
        m3 = row[0]
        row[0] = row[0] + d * d
        m2 = row[0]
        
        for j in range(1, seq_length):
            d = ref_val - tgt_local[j]
            cost = d * d
            m1 = row[j]
            min_val = m1
            if m2 < min_val:
                min_val = m2
            if m3 < min_val:
                min_val = m3
            
            m3 = m1
            row[j] = min_val + cost
            m2 = row[j]
            
    return sqrt(row[seq_length - 1])

def cpu_dtw_single(ref, tgt):
    n = len(ref)
    dtw = np.zeros((n, n), dtype=np.float32)
    dtw[0, 0] = (ref[0] - tgt[0]) ** 2
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + (ref[i] - tgt[0]) ** 2
    for j in range(1, n):
        dtw[0, j] = dtw[0, j - 1] + (ref[0] - tgt[j]) ** 2
    for i in range(1, n):
        for j in range(1, n):
            cost = (ref[i] - tgt[j]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return np.sqrt(dtw[n - 1, n - 1])
