import numpy as np
from typing import Any

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

    def __del__(self):
        """
        Retourne automatiquement le buffer physique au pool du manager de périphérique.
        """
        if hasattr(self, "device_manager") and self.device_manager is not None:
            if hasattr(self.device_manager, "free_buffer"):
                try:
                    self.device_manager.free_buffer(self.cl_buffer, self.shape, self.dtype)
                except Exception:
                    pass

    @classmethod
    @auto_sync
    async def to_device(cls, scheduler=None, host_data: np.ndarray | None = None):
        """
        Copie asynchrone et non-bloquante de la RAM de l'hôte vers la VRAM du GPU.
        """
        if host_data is None and isinstance(scheduler, np.ndarray):
            # Signature backward-compatibility: si l'appel était to_device(host_data)
            host_data = scheduler
            scheduler = None

        if scheduler is None:
            from .scheduler import get_default_scheduler

            scheduler = get_default_scheduler()

        if hasattr(scheduler, "get_queue"):
            device_manager = scheduler
        else:
            device_manager = scheduler.get_best_device()

        if host_data is None:
            raise ValueError("host_data ne peut pas être None dans to_device")

        buffer_obj = await device_manager.allocate_buffer_from_host(host_data)
        return cls(device_manager, buffer_obj, host_data.shape, host_data.dtype, backend=device_manager.backend)

    @auto_sync
    async def overwrite(self, host_data: np.ndarray):
        """
        Écrit les données de l'hôte dans un buffer GPU existant sans réallouer.
        """
        await self.device_manager.write_buffer(self.cl_buffer, host_data)

    @classmethod
    def empty_like(cls, scheduler=None, host_data: np.ndarray | None = None):
        """
        Alloue un buffer vide en VRAM de même taille et type que host_data.
        """
        if host_data is None and isinstance(scheduler, np.ndarray):
            host_data = scheduler
            scheduler = None

        if scheduler is None:
            from .scheduler import get_default_scheduler

            scheduler = get_default_scheduler()

        if hasattr(scheduler, "get_queue"):
            device_manager = scheduler
        else:
            device_manager = scheduler.get_best_device()

        if host_data is None:
            raise ValueError("host_data ne peut pas être None dans empty_like")

        buffer_obj = device_manager.allocate_empty_buffer(host_data.shape, host_data.dtype)
        return cls(device_manager, buffer_obj, host_data.shape, host_data.dtype, backend=device_manager.backend)

    @auto_sync
    async def to_host(self, out: np.ndarray | None = None) -> Any:
        """
        Copie non-bloquante de la VRAM du GPU vers la RAM de l'hôte.
        """
        return await self.device_manager.read_buffer(self.cl_buffer, self.shape, self.dtype, out=out)

    @auto_sync
    async def dirty_read(self, stream=None) -> Any:
        """
        Lecture rapide du buffer (DMA page-locked/command queue séparée) sans bloquer
        le stream de travail principal.
        """
        return await self.device_manager.dirty_read_buffer(self.cl_buffer, self.shape, self.dtype, stream=stream)

    def __dlpack__(self, stream=None):
        """
        Exporte le buffer sous forme de PyCapsule DLPack (Zero-Copy VRAM).
        Compatible avec PyTorch (torch.from_dlpack), CuPy (cupy.from_dlpack), etc.
        """
        if hasattr(self.cl_buffer, "__dlpack__"):
            try:
                return self.cl_buffer.__dlpack__(stream=stream)
            except TypeError:
                return self.cl_buffer.__dlpack__()
        elif hasattr(self.cl_buffer, "toDlpack"):
            return self.cl_buffer.toDlpack()
        else:
            raise NotImplementedError(
                f"L'objet buffer sous-jacent ({type(self.cl_buffer).__name__}) sous le backend '{self.backend}' "
                "ne prend pas en charge l'exportation DLPack directe."
            )

    def __dlpack_device__(self):
        """
        Renvoie la description du périphérique au format DLPack : (device_type, device_id).
        Ex: (1, 0) pour kDLCUDA (device 0), (13, 0) pour kDLCPU, (3, 0) pour kDLOpenCL.
        """
        if hasattr(self.cl_buffer, "__dlpack_device__"):
            return self.cl_buffer.__dlpack_device__()

        if self.backend == "cuda":
            dev_id = getattr(getattr(self.device_manager, "device", None), "id", 0)
            return (1, dev_id)  # 1 = kDLCUDA
        elif self.backend == "opencl":
            return (3, 0)  # 3 = kDLOpenCL
        else:
            return (13, 0)  # 13 = kDLCPU

    @classmethod
    def from_dlpack(cls, dlpack_obj, scheduler=None):
        """
        Crée un AsyncBuffer à partir d'un objet externe compatible DLPack (ex: PyTorch Tensor, CuPy array)
        sans aucune copie mémoire (Zero-Copy).
        """
        if scheduler is not None:
            if hasattr(scheduler, "get_queue"):
                device_manager = scheduler
            else:
                device_manager = scheduler.get_best_device()
        else:
            device_manager = None

        buffer_obj = None
        backend = "cuda"

        try:
            # pyrefly: ignore [missing-import]
            import cupy as cp

            if isinstance(dlpack_obj, cp.ndarray):
                buffer_obj = dlpack_obj
            else:
                buffer_obj = cp.from_dlpack(dlpack_obj)
            backend = "cuda"
        except Exception:
            buffer_obj = dlpack_obj
            backend = getattr(dlpack_obj, "backend", "cuda")

        shape = getattr(buffer_obj, "shape", ())
        dtype = getattr(buffer_obj, "dtype", np.float32)

        return cls(device_manager, buffer_obj, shape, dtype, backend=backend)
