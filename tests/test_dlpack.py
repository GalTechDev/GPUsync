# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
import numpy as np
import pytest

from gpusync.memory import AsyncBuffer
from gpusync.scheduler import Scheduler
from gpusync.utils.devices import list_gpus


def test_dlpack_device_and_protocol():
    gpus = list_gpus()
    if not gpus:
        pytest.skip("Aucun GPU disponible sur le système.")

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = AsyncBuffer.to_device(sched, data)

    # Verification __dlpack_device__
    dev_type, dev_id = buf.__dlpack_device__()
    assert isinstance(dev_type, int)
    assert isinstance(dev_id, int)

    # Verification __dlpack__ (doit exporter sans lever d'erreur si supporté par le backend)
    try:
        capsule = buf.__dlpack__()
        assert capsule is not None
    except NotImplementedError:
        pass


def test_dlpack_import():
    gpus = list_gpus()
    if not gpus:
        pytest.skip("Aucun GPU disponible sur le système.")

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    buf = AsyncBuffer.to_device(sched, data)

    # Re-création via from_dlpack
    if hasattr(buf.cl_buffer, "__dlpack__"):
        buf_from_dl = AsyncBuffer.from_dlpack(buf, scheduler=sched)
        assert buf_from_dl.shape == (3,)
        assert buf_from_dl.dtype == np.float32
        res = buf_from_dl.to_host()
        np.testing.assert_array_equal(res, data)
