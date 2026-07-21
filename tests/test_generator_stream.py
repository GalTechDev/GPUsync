import pytest
import numpy as np
from gpusync import gpusync_kernel, Scheduler, list_gpus

@gpusync_kernel
def add_five_kernel(ctx: int, data: list) -> tuple:
    return data[ctx] + 5.0

def simulated_disk_generator(total_elements=100_000, chunk_size=20_000):
    """
    Simule la lecture par paquets depuis un disque dur SSD.
    """
    for i in range(0, total_elements, chunk_size):
        chunk = np.arange(i, min(total_elements, i + chunk_size), dtype=np.float32)
        yield chunk

def test_generator_stream_out_of_core():
    gpus = list_gpus()
    if not gpus:
        pytest.skip("Aucun GPU disponible sur le système.")

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    # Passage direct d'une fonction génératrice (yield) à map_out_of_core !
    res = add_five_kernel.map_out_of_core(simulated_disk_generator, sched)

    assert len(res) == 100_000
    expected = np.arange(0, 100_000, dtype=np.float32) + 5.0
    np.testing.assert_allclose(res, expected, rtol=1e-5)
