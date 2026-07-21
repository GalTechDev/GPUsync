import time
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync import gpusync_kernel, Scheduler, list_gpus

@gpusync_kernel
def process_signal(ctx: int, data: list) -> tuple:
    return data[ctx] * 2.5 + 1.0

def main():
    print("=========================================================================")
    print("--- DEMO 02 : TRAITEMENT OUT-OF-CORE EN 1 SEULE LIGNE DE CODE ---")
    print("=========================================================================")

    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible sur le système.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    N = 10_000_000
    print(f"\n[1] Génération d'un dataset géant ({N:,} éléments)...")
    large_dataset = np.random.randn(N).astype(np.float32)

    print("\n[2] Lancement du traitement Out-Of-Core monoligne...")
    t0 = time.perf_counter()
    
    # UNE SEULE LIGNE DE CODE POUR TRAITER TOUT LE DATASET EN STREAMING OUT-OF-CORE !
    resultat = process_signal.map_out_of_core(large_dataset, scheduler=sched)
    
    t1 = time.perf_counter()
    
    print(f" -> Temps d'exécution total : {t1 - t0:.4f} s")
    print(f" -> Taille du résultat : {len(resultat):,} éléments")
    print(" -> Premières valeurs :", resultat[:5])

if __name__ == "__main__":
    main()
