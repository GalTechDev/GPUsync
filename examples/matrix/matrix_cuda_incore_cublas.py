import cupy as cp
import numpy as np
import time

def main():
    M, K, N = 32768, 32768, 32768
    
    print("==================================================")
    print(f"ULTIMATE BENCHMARK: IN-CORE cuBLAS (Tensor Cores)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM (CPU)...")
    A_cpu = np.random.random((M, K)).astype(np.float32)
    B_cpu = np.random.random((K, N)).astype(np.float32)
    
    print(f"[2] Transfert VRAM en cours... (Les matrices font {A_cpu.nbytes / 1e9:.2f} Go chacune)")
    # En float16 pour être sûr de ne pas OOM (Windows prend ~2Go) et forcer les Tensor Cores
    A_gpu = cp.asarray(A_cpu, dtype=cp.float16)
    B_gpu = cp.asarray(B_cpu, dtype=cp.float16)
    
    print("[3] Warmup cuBLAS...")
    _ = cp.matmul(A_gpu[:1024, :1024], B_gpu[:1024, :1024])
    cp.cuda.Stream.null.synchronize()
    
    print("[4] Démarrage du calcul massif IN-CORE...")
    start_time = time.perf_counter()
    
    # Exécution native sans aucun overhead de transfert PCIe
    C_gpu = cp.matmul(A_gpu, B_gpu)
    cp.cuda.Stream.null.synchronize()
    
    total_time = time.perf_counter() - start_time
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n[SUCCESS] Matrix Multiplication IN-CORE Terminée !")
    print(f"-> Temps de Calcul Pur       : {total_time:.4f} secondes")
    print(f"-> Débit de Calcul Effectif  : {giga_flops:.2f} TeraFLOPS")

if __name__ == "__main__":
    main()
