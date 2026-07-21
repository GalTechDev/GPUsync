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

def main():
    print("=========================================================================")
    print("--- DEMO 04 : MOTEUR CPU NATIF LLVM JIT (SANS GPU) ---")
    print("=========================================================================")

    N = 5_000_000
    print(f"\n[1] Initialisation du moteur LLVM CPU et préparation de {N:,} éléments...")
    a_cpu = np.random.randn(N).astype(np.float32)
    b_cpu = np.random.randn(N).astype(np.float32)

    llvm_dev = LLVMDeviceManager(LLVMDeviceDummy(), num_queues=8)
    a_gpu = AsyncBuffer(llvm_dev, a_cpu.copy(), a_cpu.shape, a_cpu.dtype, backend="llvm")
    b_gpu = AsyncBuffer(llvm_dev, b_cpu.copy(), b_cpu.shape, b_cpu.dtype, backend="llvm")

    print("\n[2] Lancement du kernel en JIT CPU LLVM Multi-cœurs...")
    t0 = time.perf_counter()
    res = compute_kernel(a_gpu, b_gpu, global_size=(N,), scheduler=llvm_dev)
    t1 = time.perf_counter()

    print(f" -> Temps d'exécution LLVM JIT CPU : {t1 - t0:.4f} s")

if __name__ == "__main__":
    main()
