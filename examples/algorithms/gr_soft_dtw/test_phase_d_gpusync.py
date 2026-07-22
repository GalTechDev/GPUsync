import sys
import os
import time
# pyrefly: ignore [missing-import]
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from examples.algorithms.gr_soft_dtw.gr_soft_dtw_engine import get_sdtw_fwd_bwd_kernel, get_sdtw_double_bwd_kernel
from gpusync.utils.bg_loop import auto_sync
@auto_sync
async def run_test_8_gpusync_benchmark():
    print("=== TEST 8: Benchmark GPUsync OpenCL JIT Kernel ===")
    
    # We will test N=100 and N=500
    N_values = [100, 500]
    D = 5
    gamma = 0.1
    
    # Initialize Scheduler
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    print("Scheduler Initialized")
    
    for N in N_values:
        M = N
        
        # We need matrices: Z, Q1, Q2, Q3, A, V, H, B, E
        stride = (N + 2) * (M + 2)
        total_size = D * stride
        
        # Generate dummy distance matrix M_mat of shape (D, N, M)
        M_mat_cpu = np.random.randn(D * N * M).astype(np.float32)
        V_cpu = np.random.randn(D * N * M).astype(np.float32)
        
        # Upload to GPU
        gpu_M_mat = await AsyncBuffer.to_device(scheduler, M_mat_cpu)
        gpu_V = await AsyncBuffer.to_device(scheduler, V_cpu)
        
        gpu_Z = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q1 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q2 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_Q3 = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_A = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        
        gpu_H = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_B = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
        gpu_E = AsyncBuffer.empty_like(scheduler, np.zeros(D * N * M, dtype=np.float32))
        
        # Compile kernels
        kernel_fwd = get_sdtw_fwd_bwd_kernel(N, M)
        kernel_bwd = get_sdtw_double_bwd_kernel(N, M)
        
        # Warmup (compilation overhead)
        await kernel_fwd(gpu_M_mat, gpu_Z, gpu_Q1, gpu_Q2, gpu_Q3, gpu_A, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
        await kernel_bwd(gpu_A, gpu_Q1, gpu_Q2, gpu_Q3, gpu_V, gpu_H, gpu_B, gpu_E, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
        
        # Benchmark
        iterations = 5
        start_time = time.time()
        
        for _ in range(iterations):
            # Forward + Backward A
            await kernel_fwd(gpu_M_mat, gpu_Z, gpu_Q1, gpu_Q2, gpu_Q3, gpu_A, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
            # Double Backward
            await kernel_bwd(gpu_A, gpu_Q1, gpu_Q2, gpu_Q3, gpu_V, gpu_H, gpu_B, gpu_E, gamma, global_size=(D,), local_size=(1,), scheduler=scheduler)
            
        # Ensure completion by reading a small buffer back
        res = await gpu_E.to_host()
        
        avg_time = ((time.time() - start_time) / iterations) * 1000
        print(f"N={N} | GPUsync OpenCL JIT Time (FWD+BWD+HVP) : {avg_time:.2f} ms")

if __name__ == "__main__":
    run_test_8_gpusync_benchmark()
