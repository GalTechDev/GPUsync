# pyrefly: ignore [missing-import]
import numpy as np
from .utils.bg_loop import auto_sync

class AsyncBuffer:
    """
    Représente un buffer de données stocké en mémoire physique GPU (VRAM).
    Cette classe est générique et délègue toutes les opérations d'allocation,
    de lecture, et d'écriture au manager de périphérique associé.
    """
    def __init__(self, device_manager, buffer_obj, shape, dtype, backend="opencl"):
        self.device_manager = device_manager
        self.cl_buffer = buffer_obj  # cupy.ndarray (CUDA) ou pyopencl.Buffer (OpenCL)
        self.shape = shape
        self.dtype = dtype
        self.backend = backend

    @classmethod
    @auto_sync
    async def to_device(cls, scheduler, host_data: np.ndarray):
        """
        Copie asynchrone et non-bloquante de la RAM de l'hôte vers la VRAM du GPU.
        """
        if hasattr(scheduler, 'get_queue'):
            device_manager = scheduler
        else:
            device_manager = scheduler.get_best_device()
            
        buffer_obj = await device_manager.allocate_buffer_from_host(host_data)
        return cls(device_manager, buffer_obj, host_data.shape, host_data.dtype, backend=device_manager.backend)

    @auto_sync
    async def overwrite(self, host_data: np.ndarray):
        """
        Écrit les données de l'hôte dans un buffer GPU existant sans réallouer.
        """
        await self.device_manager.write_buffer(self.cl_buffer, host_data)

    @classmethod
    def empty_like(cls, scheduler, host_data: np.ndarray):
        """
        Alloue un buffer vide en VRAM de même taille et type que host_data.
        """
        if hasattr(scheduler, 'get_queue'):
            device_manager = scheduler
        else:
            device_manager = scheduler.get_best_device()
            
        buffer_obj = device_manager.allocate_empty_buffer(host_data.shape, host_data.dtype)
        return cls(device_manager, buffer_obj, host_data.shape, host_data.dtype, backend=device_manager.backend)

    @auto_sync
    async def to_host(self, out: np.ndarray = None) -> np.ndarray:
        """
        Copie non-bloquante de la VRAM du GPU vers la RAM de l'hôte.
        """
        return await self.device_manager.read_buffer(self.cl_buffer, self.shape, self.dtype, out=out)

    @auto_sync
    async def dirty_read(self, stream=None) -> np.ndarray:
        """
        Lecture rapide du buffer (DMA page-locked/command queue séparée) sans bloquer
        le stream de travail principal.
        """
        return await self.device_manager.dirty_read_buffer(self.cl_buffer, self.shape, self.dtype, stream=stream)
