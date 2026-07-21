import logging

logger = logging.getLogger(__name__)

class GPUDeviceInfo:
    """
    Représente des métadonnées sur un GPU découvert sur le système.
    """
    def __init__(self, name: str, backend: str, raw_device, index: int = 0):
        self.name = name
        self.backend = backend  # "cuda" ou "opencl"
        self.raw_device = raw_device  # cp.cuda.Device ou cl.Device
        self.index = index  # index de périphérique au sein de ce backend

    def __repr__(self):
        return f"<GPUDeviceInfo: {self.name} ({self.backend}:{self.index})>"


def list_gpus():
    """
    Découvre et renvoie la liste de tous les périphériques graphiques (GPUs)
    disponibles sur la machine en interrogeant dynamiquement chaque manager
    enregistré dans ALL_DEVICE_MANAGERS.
    """
    from gpusync.devices import ALL_DEVICE_MANAGERS
    gpus = []

    for manager_cls in ALL_DEVICE_MANAGERS:
        devs = manager_cls.discover_devices()
        for dev in devs:
            # Obtention dynamique du nom du périphérique
            if hasattr(dev, 'name'):
                name = dev.name
            else:
                try:
                    import cupy as cp
                    props = cp.cuda.runtime.getDeviceProperties(dev.id)
                    name = props['name'].decode('utf-8')
                except Exception:
                    name = f"{manager_cls.backend.upper()} Device"

            backend_idx = len([g for g in gpus if g.backend == manager_cls.backend])
            gpus.append(GPUDeviceInfo(name=name, backend=manager_cls.backend, raw_device=dev, index=backend_idx))

    return gpus
