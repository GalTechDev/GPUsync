import time
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync import gpusync_kernel, Scheduler, list_gpus

@gpusync_kernel
def scale_kernel(ctx: int, data: list) -> tuple:
    return data[ctx] * 3.0 + 5.0

def main():
    print("=========================================================================")
    print("--- DEMO 01 : GUIDE DE DÉMARRAGE RAPIDE GPUSYNC ---")
    print("=========================================================================")

    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible sur le système.")
        return

    print(f"[1] Sélection du GPU : {gpus[0].name} ({gpus[0].backend})")
    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    N = 1_000_000
    data_cpu = np.linspace(1.0, 100.0, N, dtype=np.float32)

    print(f"\n[2] Lancement du kernel sur {N:,} éléments...")
    t0 = time.perf_counter()
    resultat = scale_kernel(data_cpu, global_size=(N,), scheduler=sched)
    t1 = time.perf_counter()

    print(f" -> Temps d'exécution : {t1 - t0:.4f} s")
    print(" -> Échantillon du résultat :", resultat[:5])

if __name__ == "__main__":
    main()
