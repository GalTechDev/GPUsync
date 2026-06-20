import numpy as np
from .scheduler import Scheduler
from .executor import await_event
from .bg_loop import auto_sync

class AsyncBuffer:
    def __init__(self, device_manager, buffer_obj, shape, dtype, backend="opencl"):
        self.device_manager = device_manager
        self.cl_buffer = buffer_obj  # Peut être un pyopencl.Buffer ou cupy.ndarray
        self.shape = shape
        self.dtype = dtype
        self.backend = backend

    @classmethod
    @auto_sync
    async def to_device(cls, scheduler: Scheduler, host_data: np.ndarray):
        """
        Copie non-bloquante de la RAM vers la VRAM du GPU.
        """
        device_manager = scheduler.get_best_device()
        backend = scheduler.backend
        queue = device_manager.get_queue()

        if backend in ["cuda", "cublas"]:
            import cupy as cp
            with device_manager.device:
                gpu_arr = cp.empty(host_data.shape, dtype=host_data.dtype)
                gpu_arr.set(host_data, stream=queue)
            await await_event(queue)
            return cls(device_manager, gpu_arr, host_data.shape, host_data.dtype, backend="cuda")
        else:
            import pyopencl as cl
            flags = cl.mem_flags.READ_WRITE
            cl_buffer = cl.Buffer(device_manager.context, flags, size=host_data.nbytes)
            event = cl.enqueue_copy(queue, cl_buffer, host_data, is_blocking=False)
            await await_event(event) # Ne bloque pas le CPU pendant le transfert
            return cls(device_manager, cl_buffer, host_data.shape, host_data.dtype, backend="opencl")

    @auto_sync
    async def overwrite(self, host_data: np.ndarray):
        """
        Copie non-bloquante de la RAM vers un buffer VRAM existant.
        Permet le Memory Pooling (0 allocation de nouvelle mémoire).
        """
        queue = self.device_manager.get_queue()
        if self.backend in ["cuda", "cublas"]:
            with self.device_manager.device:
                self.cl_buffer.set(host_data, stream=queue)
            await await_event(queue)
        else:
            import pyopencl as cl
            event = cl.enqueue_copy(queue, self.cl_buffer, host_data, is_blocking=False)
            await await_event(event)

    @classmethod
    def empty_like(cls, scheduler: Scheduler, host_data: np.ndarray):
        """
        Alloue un buffer vide sur le GPU ayant la même taille que host_data.
        """
        device_manager = scheduler.get_best_device()
        backend = scheduler.backend

        if scheduler.backend in ["cuda", "cublas"]:
            import cupy as cp
            with device_manager.device:
                gpu_arr = cp.empty(host_data.shape, dtype=host_data.dtype)
            return cls(device_manager, gpu_arr, host_data.shape, host_data.dtype, backend="cuda")
        else:
            import pyopencl as cl
            flags = cl.mem_flags.READ_WRITE
            cl_buffer = cl.Buffer(device_manager.context, flags, size=host_data.nbytes)
            return cls(device_manager, cl_buffer, host_data.shape, host_data.dtype, backend="opencl")

    @auto_sync
    async def to_host(self) -> np.ndarray:
        """
        Copie non-bloquante de la VRAM du GPU vers la RAM.
        """
        queue = self.device_manager.get_queue()
        
        if self.backend in ["cuda", "cublas"]:
            with self.device_manager.device:
                # get() avec un stream alloue asynchronement en RAM pinée
                host_data = self.cl_buffer.get(stream=queue)
            await await_event(queue)
            return host_data
        else:
            import pyopencl as cl
            host_data = np.empty(self.shape, dtype=self.dtype)
            event = cl.enqueue_copy(queue, host_data, self.cl_buffer, is_blocking=False)
            await await_event(event)
            return host_data

    @auto_sync
    async def dirty_read(self, stream=None) -> np.ndarray:
        """
        Lecture 'Dirty' de la mémoire depuis le GPU sans bloquer ou synchroniser 
        le stream de travail principal. Utilise un stream dédié (CUDA) ou 
        une command queue séparée (OpenCL).
        """
        if self.backend in ["cuda", "cublas"]:
            import cupy as cp
            with self.device_manager.device:
                read_stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
                # Allocation de mémoire Pinée (Page-Locked) pour un transfert DMA pur
                host_pinned = cp.cuda.alloc_pinned_memory(self.cl_buffer.nbytes)
                count = int(np.prod(self.shape))
                host_data = np.frombuffer(host_pinned, self.dtype, count=count).reshape(self.shape)
                # Copie asynchrone vers l'hôte sur le read_stream
                self.cl_buffer.data.copy_to_host_async(host_data.ctypes.data, self.cl_buffer.nbytes, stream=read_stream)
            await await_event(read_stream)
            return host_data
        else:
            import pyopencl as cl
            queue = stream if stream is not None else cl.CommandQueue(self.device_manager.context)
            host_data = np.empty(self.shape, dtype=self.dtype)
            event = cl.enqueue_copy(queue, host_data, self.cl_buffer, is_blocking=False)
            await await_event(event)
            return host_data
