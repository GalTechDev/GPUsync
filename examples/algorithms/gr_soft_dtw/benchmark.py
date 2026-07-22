import sys
import os
import time
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.gr_soft_dtw.gr_soft_dtw_engine import get_sdtw_fwd_bwd_kernel, get_sdtw_double_bwd_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

try:
    import torch
    from examples.algorithms.gr_soft_dtw.gr_core import gr_soft_dtw_loss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class GRSoftDTWBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="GR-Soft-DTW",
            description="Graph-Regularized Soft-DTW Forward/Backward/HVP",
            num_runs=num_runs,
            variants=["GPUsync FWD+BWD+HVP"]
        )

    def get_plot_metadata(self):
        return {
            "N (longueur)": 16,
            "M (longueur)": 16,
            "D (sensors)": 5
        }

    def setup_data(self, size: int):
        # Limité à 16 max pour éviter l'explosion du JIT OpenCL compilation (boucles déroulées)
        N = min(size, 16)
        M = N
        D = 5
        gamma = 0.1
        lambd = 1.0
        
        edges = []
        for u in range(D):
            for v in range(u+1, D):
                edges.append((u, v, 0))
                
        X = np.random.randn(D, N).astype(np.float32)
        Y = np.random.randn(D, M).astype(np.float32)
        
        return {
            "N": N,
            "M": M,
            "D": D,
            "gamma": gamma,
            "lambd": lambd,
            "edges": edges,
            "X": X,
            "Y": Y
        }

    def run_cpu(self, data):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        X = torch.tensor(data["X"], requires_grad=True, dtype=torch.float32)
        Y = torch.tensor(data["Y"], dtype=torch.float32)
        gamma = data["gamma"]
        edges = data["edges"]
        lambd = data["lambd"]
        
        loss = gr_soft_dtw_loss(X, Y, gamma, edges, lambd)
        
        # Double backward check to measure full forward/backward pipeline
        grads = torch.autograd.grad(loss, X, create_graph=True)[0]
        loss_grad = grads.sum()
        torch.autograd.grad(loss_grad, X)[0]
        
        return data["D"]

    def run_gpusync(self, data, scheduler, variant: str):
        N = data["N"]
        M = data["M"]
        D = data["D"]
        gamma = data["gamma"]
        
        stride = (N + 2) * (M + 2)
        total_size = D * stride
        
        M_mat_cpu = np.random.randn(D * N * M).astype(np.float32)
        V_cpu = np.random.randn(D * N * M).astype(np.float32)
        
        # Les méthodes d'AsyncBuffer et les kernels JIT ont déjà le décorateur @auto_sync.
        # Ils s'exécutent de façon synchrone dans un contexte synchrone.
        gpu_M_mat = AsyncBuffer.to_device(scheduler, M_mat_cpu)
        gpu_V = AsyncBuffer.to_device(scheduler, V_cpu)
        
        gpu_Z = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q1 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q2 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q3 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_A = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        
        gpu_H = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_B = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_E = AsyncBuffer.empty_like(scheduler, np.zeros(D * N * M, dtype=np.float32))
        
        kernel_fwd = get_sdtw_fwd_bwd_kernel(N, M)
        kernel_bwd = get_sdtw_double_bwd_kernel(N, M)
        
        kernel_fwd(gpu_M_mat, gpu_Z, gpu_Q1, gpu_Q2, gpu_Q3, gpu_A, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
        kernel_bwd(gpu_A, gpu_Q1, gpu_Q2, gpu_Q3, gpu_V, gpu_H, gpu_B, gpu_E, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
        
        _ = gpu_E.to_host()
        return D

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = GRSoftDTWBenchmark(num_runs=2)
    bm.execute(sizes=[8, 12], scheduler=sched)

if __name__ == "__main__":
    main()
