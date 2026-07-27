import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))
from gpusync.compiler import gpusync_kernel
from gpusync.api.opencl import get_global_id

# Dummy for transpiler
def exp(x: float) -> float: return 0.0
def log(x: float) -> float: return 0.0

def get_sdtw_fwd_bwd_kernel(N: int, M: int):
    @gpusync_kernel(constants={"N": N, "M": M})
    def sdtw_fwd_bwd(ctx: int, M_mat: list, Z: list, Q1: list, Q2: list, Q3: list, A: list, gamma: float):
        d: int = get_global_id(0)
        stride: int = (N + 2) * (M + 2)
        base_idx: int = d * stride
        
        # Init padding to large value
        for i in range(0, N + 2):
            for j in range(0, M + 2):
                Z[base_idx + i * (M + 2) + j] = 1e10
                
        Z[base_idx + 0] = 0.0
        
        # Forward Pass
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                a_val: float = Z[base_idx + (i - 1) * (M + 2) + j]
                b_val: float = Z[base_idx + i * (M + 2) + (j - 1)]
                c_val: float = Z[base_idx + (i - 1) * (M + 2) + (j - 1)]
                
                # softmin
                min_val: float = a_val
                if b_val < min_val: min_val = b_val
                if c_val < min_val: min_val = c_val
                
                sum_exp: float = exp((min_val - a_val) / gamma) + exp((min_val - b_val) / gamma) + exp((min_val - c_val) / gamma)
                softmin_val: float = min_val - gamma * log(sum_exp)
                
                m_val: float = M_mat[d * N * M + (i - 1) * M + (j - 1)]
                z_val: float = m_val + softmin_val
                Z[base_idx + i * (M + 2) + j] = z_val
                
                Q1[base_idx + i * (M + 2) + j] = exp((z_val - m_val - a_val) / gamma)
                Q2[base_idx + i * (M + 2) + j] = exp((z_val - m_val - b_val) / gamma)
                Q3[base_idx + i * (M + 2) + j] = exp((z_val - m_val - c_val) / gamma)
                
        # Init A
        for i in range(0, N + 2):
            for j in range(0, M + 2):
                A[base_idx + i * (M + 2) + j] = 0.0
        A[base_idx + N * (M + 2) + M] = 1.0
        
        # Backward Pass (Computing A)
        for step_i in range(0, N):
            i: int = N - step_i
            for step_j in range(0, M):
                j: int = M - step_j
                if i == N and j == M:
                    continue
                
                a_calc: float = A[base_idx + (i + 1) * (M + 2) + j] * Q1[base_idx + (i + 1) * (M + 2) + j] + \
                                A[base_idx + i * (M + 2) + (j + 1)] * Q2[base_idx + i * (M + 2) + (j + 1)] + \
                                A[base_idx + (i + 1) * (M + 2) + (j + 1)] * Q3[base_idx + (i + 1) * (M + 2) + (j + 1)]
                A[base_idx + i * (M + 2) + j] = a_calc
                
    return sdtw_fwd_bwd


def get_sdtw_double_bwd_kernel(N: int, M: int):
    @gpusync_kernel(constants={"N": N, "M": M})
    def sdtw_double_bwd(ctx: int, A: list, Q1: list, Q2: list, Q3: list, V: list, H: list, B: list, E: list, gamma: float):
        d: int = get_global_id(0)
        stride: int = (N + 2) * (M + 2)
        base_idx: int = d * stride
        
        # Init H and B
        for i in range(0, N + 2):
            for j in range(0, M + 2):
                H[base_idx + i * (M + 2) + j] = 0.0
                B[base_idx + i * (M + 2) + j] = 0.0
                
        H[base_idx + 1 * (M + 2) + 1] = V[d * N * M + 0]
        
        # 3.1 Forward H
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if i == 1 and j == 1:
                    continue
                
                v_val: float = V[d * N * M + (i - 1) * M + (j - 1)]
                h_val: float = v_val + \
                               H[base_idx + (i - 1) * (M + 2) + j] * Q1[base_idx + i * (M + 2) + j] + \
                               H[base_idx + i * (M + 2) + (j - 1)] * Q2[base_idx + i * (M + 2) + j] + \
                               H[base_idx + (i - 1) * (M + 2) + (j - 1)] * Q3[base_idx + i * (M + 2) + j]
                H[base_idx + i * (M + 2) + j] = h_val
                
        # 3.2 & 3.3 Backward B
        for step_i in range(0, N):
            i: int = N - step_i
            for step_j in range(0, M):
                j: int = M - step_j
                
                # g1
                alpha_1: float = A[base_idx + (i + 1) * (M + 2) + j] * ( \
                                 Q1[base_idx + (i + 1) * (M + 2) + j] * H[base_idx + i * (M + 2) + j] + \
                                 Q2[base_idx + (i + 1) * (M + 2) + j] * H[base_idx + (i + 1) * (M + 2) + (j - 1)] + \
                                 Q3[base_idx + (i + 1) * (M + 2) + j] * H[base_idx + i * (M + 2) + (j - 1)])
                g1: float = (Q1[base_idx + (i + 1) * (M + 2) + j] / gamma) * (alpha_1 - H[base_idx + i * (M + 2) + j] * A[base_idx + (i + 1) * (M + 2) + j])
                
                # g2
                alpha_2: float = A[base_idx + i * (M + 2) + (j + 1)] * ( \
                                 Q1[base_idx + i * (M + 2) + (j + 1)] * H[base_idx + (i - 1) * (M + 2) + (j + 1)] + \
                                 Q2[base_idx + i * (M + 2) + (j + 1)] * H[base_idx + i * (M + 2) + j] + \
                                 Q3[base_idx + i * (M + 2) + (j + 1)] * H[base_idx + (i - 1) * (M + 2) + j])
                g2: float = (Q2[base_idx + i * (M + 2) + (j + 1)] / gamma) * (alpha_2 - H[base_idx + i * (M + 2) + j] * A[base_idx + i * (M + 2) + (j + 1)])
                
                # g3
                alpha_3: float = A[base_idx + (i + 1) * (M + 2) + (j + 1)] * ( \
                                 Q1[base_idx + (i + 1) * (M + 2) + (j + 1)] * H[base_idx + i * (M + 2) + (j + 1)] + \
                                 Q2[base_idx + (i + 1) * (M + 2) + (j + 1)] * H[base_idx + (i + 1) * (M + 2) + j] + \
                                 Q3[base_idx + (i + 1) * (M + 2) + (j + 1)] * H[base_idx + i * (M + 2) + j])
                g3: float = (Q3[base_idx + (i + 1) * (M + 2) + (j + 1)] / gamma) * (alpha_3 - H[base_idx + i * (M + 2) + j] * A[base_idx + (i + 1) * (M + 2) + (j + 1)])
                
                b_val: float = (B[base_idx + (i + 1) * (M + 2) + j] * Q1[base_idx + (i + 1) * (M + 2) + j] + g1) + \
                               (B[base_idx + i * (M + 2) + (j + 1)] * Q2[base_idx + i * (M + 2) + (j + 1)] + g2) + \
                               (B[base_idx + (i + 1) * (M + 2) + (j + 1)] * Q3[base_idx + (i + 1) * (M + 2) + (j + 1)] + g3)
                               
                B[base_idx + i * (M + 2) + j] = b_val
                E[d * N * M + (i - 1) * M + (j - 1)] = b_val
                
    return sdtw_double_bwd
