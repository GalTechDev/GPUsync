import time
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync import gpusync_kernel, Scheduler, list_gpus

@gpusync_kernel
def process_chunk_kernel(ctx: int, data: list) -> tuple:
    # pyrefly: ignore [unknown-name]
    return sqrt(data[ctx]) * 1.5

def simulated_nvme_file_stream(num_chunks=10, elements_per_chunk=1_000_000):
    """
    Générateur simulant la lecture continue d'un fichier géant sur SSD NVMe
    sans jamais charger l'ensemble du fichier en RAM Hôte d'un coup.
    """
    print("   [Disque NVMe] Début de la lecture du fichier par paquets...")
    for i in range(num_chunks):
        chunk = np.full(elements_per_chunk, (i + 1) * 100.0, dtype=np.float32)
        yield chunk

def main():
    print("=========================================================================")
    print("--- DEMO 03 : STREAMING CONTINU DISQUE SSD -> VRAM GPU (YIELD) ---")
    print("=========================================================================")

    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible sur le système.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    print("\n[1] Lancement du streaming direct depuis le générateur disque...")
    t0 = time.perf_counter()
    
    # PASSAGE DIRECT DU GÉNÉRATEUR DISQUE À MAP_OUT_OF_CORE !
    # Disque NVMe -> RAM Hôte -> PCIe DMA -> VRAM -> GPU Compute
    resultat = process_chunk_kernel.map_out_of_core(simulated_nvme_file_stream, scheduler=sched)
    
    t1 = time.perf_counter()
    
    print(f"\n -> Temps d'exécution total : {t1 - t0:.4f} s")
    print(f" -> Nombre total d'éléments traités depuis le disque : {len(resultat):,}")
    print(" -> Échantillon des résultats :", resultat[:5])

if __name__ == "__main__":
    main()
