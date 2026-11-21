"""Sonde : coût d'un appel sur GPU, chaîne y = f(y) contre appels indépendants sur la même entrée."""
import sys, time
import numpy as np
import gpusync as gs


@gs.kernel
def plus1(x: gs.u32[:]) -> gs.u32[:]:
    return x + 1


N = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 22
K = 200
for dev in sys.argv[1].split(","):
    x = np.zeros(N, np.uint32)
    x0 = plus1(x, device=dev)  # non lu : reste sur le périphérique
    plus1(x0, device=dev).numpy()
    # 1. appels indépendants (entrée fixe sur le GPU), lecture du dernier
    t = time.perf_counter()
    for _ in range(K):
        z = plus1(x0, device=dev)
    t1 = time.perf_counter(); z.numpy(); t2 = time.perf_counter()
    indep = ((t1 - t) / K, (t2 - t) / K)
    # 2. chaîne
    t = time.perf_counter(); z = x0
    for _ in range(K):
        z = plus1(z, device=dev)
    t1 = time.perf_counter(); v = int(z.numpy()[0]); t2 = time.perf_counter()
    chain = ((t1 - t) / K, (t2 - t) / K)
    # 3. lecture seule
    t = time.perf_counter(); z.numpy(); read_only = time.perf_counter() - t
    t = time.perf_counter(); w = plus1(x0, device=dev); w.numpy(); single = time.perf_counter() - t
    gb = 2 * 4 * N / 1e9
    print(f"{dev:9} N={N:>9} | indépendants : soumission {indep[0]*1e3:.3f} ms, total {indep[1]*1e3:.3f} ms/appel ({gb/indep[1]:.0f} Go/s)"
          f" | chaîne : soumission {chain[0]*1e3:.3f}, total {chain[1]*1e3:.3f} ms/appel ({gb/chain[1]:.0f} Go/s) [{v}]"
          f" | 1 appel+lecture {single*1e3:.2f} ms, lecture seule (prête) {read_only*1e3:.2f} ms", flush=True)
