import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync import Scheduler, AsyncBuffer, gpusync_kernel, MultiDeviceScheduler, gpusync_pipeline

@gpusync_kernel
def add_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx]

@gpusync_kernel
def mult_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] * b[ctx]

def main():
    scheduler_nvidia = Scheduler(backend="cuda")
    scheduler_nvidia.initialize()

    # On utilise deux fois le même scheduler pour simuler 2 GPUs sur un ordi qui n'en a qu'un
    multi = MultiDeviceScheduler([scheduler_nvidia, scheduler_nvidia])

    @gpusync_pipeline(multi)
    def run_my_algo(a, b, c):
        # res1 = a + b  (Sera exécuté sur le "premier" GPU)
        res1 = add_kernel(a, b, global_size=(4,))
        # res2 = b + c  (N'a aucune dépendance avec res1 -> Exécuté en PARALLÈLE sur le "deuxième" GPU !)
        res2 = add_kernel(b, c, global_size=(4,))
        # final = res1 * res2 (Dépend de res1 et res2 -> Attendra la fin des deux, rapatriera la mémoire si besoin, puis s'exécutera)
        final = mult_kernel(res1, res2, global_size=(4,))
        return final

    a_cpu = np.array([1, 2, 3, 4], dtype=np.float32)
    b_cpu = np.array([10, 10, 10, 10], dtype=np.float32)
    c_cpu = np.array([100, 100, 100, 100], dtype=np.float32)

    a_gpu = AsyncBuffer.to_device(scheduler_nvidia, a_cpu)
    b_gpu = AsyncBuffer.to_device(scheduler_nvidia, b_cpu)
    c_gpu = AsyncBuffer.to_device(scheduler_nvidia, c_cpu)

    print("Execution du graphe de dépendance via AST...")
    final_res = run_my_algo(a_gpu, b_gpu, c_gpu)
    print("Résultat final :", final_res)

if __name__ == "__main__":
    main()
