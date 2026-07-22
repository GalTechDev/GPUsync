import sys
import os
import time
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from gpusync import AsyncBuffer, Scheduler, list_gpus

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    return loop.run_until_complete(coro)

class MatrixBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="MatrixMult",
            description="2D Dense Matrix Multiplication (N x N) - Comparison of 6 variants",
            num_runs=num_runs,
            variants=[
                "OpenCL Tiled",
                "OpenCL 2D Tiling",
                "CUDA Double Buffering",
                "CUDA WMMA",
                "CUDA WMMA 2D",
                "CUDA Direct cuBLAS"
            ]
        )

    def get_plot_metadata(self):
        return {
            "dimensions": "N x N"
        }

    def setup_data(self, size: int):
        # La taille N doit être un multiple de 32 pour les différents alignements de tuiles
        N = (size // 32) * 32
        if N == 0:
            N = 32
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)
        return {
            "N": N,
            "A_2d": A,
            "B_2d": B
        }

    def run_cpu(self, data):
        _ = np.dot(data["A_2d"], data["B_2d"])
        return data["N"]

    def run_gpusync(self, data, scheduler, variant: str):
        A = data["A_2d"]
        B = data["B_2d"]
        N = data["N"]
        
        is_cuda = (scheduler.gpu.backend == 'cuda')
        if variant.startswith("CUDA") and not is_cuda:
            raise RuntimeError("CUDA is not supported on this device/backend.")

        if variant == "OpenCL Tiled":
            from examples.algorithms.matrix.matrix_gpu import matmul_opencl
            _ = run_async(matmul_opencl(A, B, scheduler))
            return N
            
        elif variant == "OpenCL 2D Tiling":
            from examples.algorithms.matrix.matrix_gpu_2D_tiling import matmul_opencl_2d
            _ = run_async(matmul_opencl_2d(A, B, scheduler))
            return N
            
        elif variant == "CUDA Double Buffering":
            from examples.algorithms.matrix.matrix_cuda_double_buffering import matmul_cuda_cublas
            _ = run_async(matmul_cuda_cublas(A, B, scheduler))
            return N
            
        elif variant == "CUDA WMMA":
            from examples.algorithms.matrix.matrix_cuda_wmma import matmul_cuda_wmma
            _ = run_async(matmul_cuda_wmma(A, B, scheduler))
            return N
            
        elif variant == "CUDA WMMA 2D":
            from examples.algorithms.matrix.matrix_cuda_wmma_2D_tiling import matmul_cuda_wmma_2d
            _ = run_async(matmul_cuda_wmma_2d(A, B, scheduler))
            return N
            
        elif variant == "CUDA Direct cuBLAS":
            import cupy as cp
            with cp.cuda.Device(0):
                A_gpu = cp.asarray(A, dtype=cp.float16)
                B_gpu = cp.asarray(B, dtype=cp.float16)
                C_gpu = cp.matmul(A_gpu, B_gpu)
                cp.cuda.Stream.null.synchronize()
                _ = C_gpu.get()
            return N
        else:
            raise ValueError(f"Unknown variant: {variant}")

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = MatrixBenchmark(num_runs=3)
    bm.execute(sizes=[128, 256, 512], scheduler=sched)

if __name__ == "__main__":
    main()
