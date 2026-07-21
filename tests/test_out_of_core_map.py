import pytest
import numpy as np
from gpusync import gpusync_kernel, Scheduler, list_gpus

@gpusync_kernel
def scale_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] * 3.0 + b[ctx]

def test_map_out_of_core_single_return():
    gpus = list_gpus()
    if not gpus:
        pytest.skip("Aucun GPU disponible sur le système.")

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    N = 500_000
    a_cpu = np.linspace(1.0, 100.0, N, dtype=np.float32)
    b_cpu = np.ones(N, dtype=np.float32) * 5.0

    # Test de l'API monoligne map_out_of_core
    res = scale_kernel.map_out_of_core(a_cpu, sched, b_cpu, chunk_size=50_000, concurrency=2)

    assert len(res) == N
    expected = a_cpu * 3.0 + b_cpu
    np.testing.assert_allclose(res, expected, rtol=1e-5)
