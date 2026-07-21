import time
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync import gpusync_kernel, AsyncBuffer
from gpusync.devices.llvm import LLVMDeviceManager, LLVMDeviceDummy

@gpusync_kernel
def compute_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] * 2.5 + b[ctx] - 1.0

def run_benchmark_for_size(N: int, num_runs: int = 10):
    a_cpu = np.random.randn(N).astype(np.float32)
    b_cpu = np.random.randn(N).astype(np.float32)

    llvm_dev = LLVMDeviceManager(LLVMDeviceDummy(), num_queues=8)
    a_gpu = AsyncBuffer(llvm_dev, a_cpu.copy(), a_cpu.shape, a_cpu.dtype, backend="llvm")
    b_gpu = AsyncBuffer(llvm_dev, b_cpu.copy(), b_cpu.shape, b_cpu.dtype, backend="llvm")

    # Warmup
    _ = compute_kernel(a_gpu, b_gpu, global_size=(N,), scheduler=llvm_dev)

    # 1. Python Pur (10 runs sur échantillon calibré)
    n_sample = min(N, 200_000)
    py_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = [a_cpu[i] * 2.5 + b_cpu[i] - 1.0 for i in range(n_sample)]
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * (N / n_sample))
    mean_py_time = np.mean(py_times)

    # 2. GPUsync LLVM JIT (10 runs)
    llvm_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = compute_kernel(a_gpu, b_gpu, global_size=(N,), scheduler=llvm_dev)
        t1 = time.perf_counter()
        llvm_times.append(t1 - t0)
    mean_llvm_time = np.mean(llvm_times)

    speedup = mean_py_time / max(mean_llvm_time, 1e-6)
    return mean_py_time, mean_llvm_time, speedup

def main():
    print("==================================================================================")
    print("--- BENCHMARK GPUsync LLVM JIT vs PYTHON PUR : ETUDE DE SCALABILITE (10 RUNS) ---")
    print("==================================================================================")

    sizes = [100_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000]
    num_runs = 10

    results = []
    print(f"\nExécution de {num_runs} itérations par taille d'éléments...\n")
    print(f"{'Taille N':<14} | {'Moyenne Python (s)':<20} | {'Moyenne LLVM JIT (s)':<20} | {'Speedup':<10}")
    print("-" * 75)

    for N in sizes:
        py_t, llvm_t, sp = run_benchmark_for_size(N, num_runs=num_runs)
        results.append((N, py_t, llvm_t, sp))
        print(f"{N:<14,} | {py_t:<20.4f} | {llvm_t:<20.4f} | {sp:<10.2f}x")

    print("-" * 75)

    # Analyse de la tendance de l'accélération
    speedups = [r[3] for r in results]
    min_sp, max_sp = min(speedups), max(speedups)
    ratio = max_sp / max(min_sp, 1e-6)

    print("\n==================================================================================")
    print("--- ANALYSE TECHNIQUE DE LA SCALABILITE DU SPEEDUP ---")
    print("==================================================================================")
    print(f"Speedup minimum (sur N = {results[0][0]:,}): {results[0][3]:.2f}x")
    print(f"Speedup maximum (sur N = {results[-1][0]:,}): {results[-1][3]:.2f}x")

    if ratio < 1.25:
        print("-> Conclusion : Le gain de performance est STATIQUE (constante quasi-fixe par rapport à N).")
    elif ratio < 3.0:
        print("-> Conclusion : Le gain de performance a une croissance LINEAIRE (augmente progressivement avec la taille du problème N).")
    else:
        print("-> Conclusion : Le gain de performance a une croissance EXPONENTIELLE/SUPER-LINEAIRE (amplifié massivement sur de grands volumes N).")
    print("==================================================================================")

if __name__ == "__main__":
    main()
