import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gpusync import Scheduler, SwarmScheduler, list_gpus


def test_list_gpus():
    gpus = list_gpus()
    print("\nDiscovered GPUs:")
    for g in gpus:
        print(f" - {g.name} (backend: {g.backend}, index: {g.index})")

    # We should have at least one GPU/device available on the system
    assert isinstance(gpus, list)


def test_scheduler_specific_gpu():
    gpus = list_gpus()
    if not gpus:
        return

    target_gpu = gpus[0]
    # Initialize scheduler explicitly pointing to this GPU
    s = Scheduler(gpu=target_gpu)
    s.initialize()

    assert len(s.devices_managers) == 1
    assert s.devices_managers[0].backend == target_gpu.backend
    assert s.backend == target_gpu.backend


def test_swarm_scheduler_specific_gpus():
    gpus = list_gpus()
    if not gpus:
        return

    # Restrict swarm to only the first discovered GPU
    target_gpus = [gpus[0]]
    swarm = SwarmScheduler(gpus=target_gpus)
    swarm.initialize()

    assert len(swarm.devices_managers) == 1
    assert swarm.devices_managers[0].backend == target_gpus[0].backend
