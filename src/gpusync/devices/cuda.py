# pyrefly: ignore [missing-import]
import logging

# pyrefly: ignore [missing-import]
import numpy as np

from ..executor import await_event
from gpusync.utils.imports import cp

logger = logging.getLogger(__name__)


class CUDADeviceManager:
    """
    Gère l'activation du périphérique CUDA et le pool de Streams CUDA
    associés à un GPU physique Nvidia.
    """

    backend = "cuda"

    @classmethod
    def discover_devices(cls, name_filter: str | None = None):
        """
        Découvre et renvoie tous les périphériques CUDA disponibles sur la machine.
        """
        devices = []
        try:

            num_devices = cp.cuda.runtime.getDeviceCount()
            for i in range(num_devices):
                device = cp.cuda.Device(i)
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode("utf-8")
                if name_filter and name_filter.lower() not in name.lower():
                    continue
                devices.append(device)
        except Exception:
            pass
        return devices

    def __init__(self, device, num_queues: int = 16):

        self.device = device
        self.backend = "cuda"

        # Rend le contexte du périphérique actif pour CuPy
        self.device.use()

        # Pool de Streams pour forcer la concurrence d'exécution CUDA
        self.queues = [cp.cuda.Stream(non_blocking=True) for _ in range(num_queues)]

        self.next_queue_idx = 0
        self._buffer_pool: dict = {}

        # Récupération du nom de la carte pour les logs
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        self.name = props["name"].decode("utf-8")
        logger.info(f"CUDADeviceManager initialisé : {self.name} avec {num_queues} Streams.")

    def get_queue(self):
        """
        Retourne un Stream CUDA selon une stratégie Round-Robin.
        """
        queue = self.queues[self.next_queue_idx]
        self.next_queue_idx = (self.next_queue_idx + 1) % len(self.queues)
        return queue

    async def execute_kernel(self, task, global_size, local_size, *args, stream=None, fire_and_forget=False):
        """
        Exécute le kernel CUDA de la tâche sur ce périphérique.
        """
        queue = stream if stream is not None else self.get_queue()
        device_id = self.device.id

        if device_id not in task._compiled_programs:
            code = task.kernel_code_cuda
            name = task.kernel_name_cuda
            try:
                module = cp.RawKernel(code, name)
            except Exception as e:
                raise RuntimeError(
                    f"Erreur fatale lors de la compilation CUDA du kernel '{name}' !\n"
                    f"Message d'erreur NVCC :\n{e}\n\n"
                    f"--- CODE CUDA GENERE ---\n{code}\n--------------------------"
                ) from e
            task._compiled_programs[device_id] = module

        kernel = task._compiled_programs[device_id]

        cuda_args = []
        for arg in args:
            if hasattr(arg, "cl_buffer"):
                cuda_args.append(arg.cl_buffer)
            elif isinstance(arg, int):
                cuda_args.append(np.int32(arg))
            elif isinstance(arg, float):
                cuda_args.append(np.float32(arg))
            else:
                cuda_args.append(arg)

        if local_size is not None:
            block = local_size
            if isinstance(global_size, int):
                grid = ((global_size + block[0] - 1) // block[0],)
            elif len(global_size) == 1:
                grid = ((global_size[0] + block[0] - 1) // block[0],)
            elif len(global_size) == 2:
                grid = ((global_size[0] + block[0] - 1) // block[0], (global_size[1] + block[1] - 1) // block[1])
            elif len(global_size) == 3:
                grid = (
                    (global_size[0] + block[0] - 1) // block[0],
                    (global_size[1] + block[1] - 1) // block[1],
                    (global_size[2] + block[2] - 1) // block[2],
                )
        else:
            if isinstance(global_size, int):
                threads_per_block = 256
                blocks_per_grid = (global_size + threads_per_block - 1) // threads_per_block
                grid = (blocks_per_grid,)
                block = (threads_per_block,)
            else:
                if len(global_size) == 1:
                    block = (256,)
                    grid = ((global_size[0] + 255) // 256,)
                elif len(global_size) == 2:
                    block = (16, 16)
                    grid = ((global_size[0] + 15) // 16, (global_size[1] + 15) // 16)
                elif len(global_size) == 3:
                    block = (8, 8, 8)
                    grid = ((global_size[0] + 7) // 8, (global_size[1] + 7) // 8, (global_size[2] + 7) // 8)
                else:
                    raise ValueError("global_size ne peut pas dépasser 3 dimensions en CUDA.")

        with self.device:
            kernel(grid, block, tuple(cuda_args), stream=queue)

        if not fire_and_forget:
            return await await_event(queue)
        else:
            return None

    async def allocate_buffer_from_host(self, host_data, queue=None):
        """
        Alloue un buffer CUDA (cupy.ndarray) en VRAM et y copie les données de l'hôte.
        """
        with self.device:
            gpu_arr = cp.asarray(host_data)
        return gpu_arr

    def allocate_empty_buffer(self, shape, dtype):
        """
        Alloue un buffer CUDA (cupy.ndarray) vide en VRAM.
        """
        with self.device:
            gpu_arr = cp.empty(shape, dtype=dtype)
        return gpu_arr

    async def write_buffer(self, dest_buffer, host_data, queue=None):
        """
        Écrit des données de l'hôte dans un buffer CUDA existant.
        """
        q = queue if queue is not None else self.get_queue()
        with self.device:
            if dest_buffer.shape != host_data.shape:
                slices = tuple(slice(0, dim) for dim in host_data.shape)
                dest_buffer[slices].set(host_data, stream=q)
            else:
                dest_buffer.set(host_data, stream=q)
        await await_event(q)

    async def read_buffer(self, src_buffer, shape, dtype, out=None, queue=None):
        """
        Lit un buffer CUDA vers la mémoire de l'hôte.
        """
        q = queue if queue is not None else self.get_queue()
        with self.device:
            if out is None:
                host_data = src_buffer.get(stream=q)
            else:
                if src_buffer.shape != out.shape:
                    slices = tuple(slice(0, dim) for dim in out.shape)
                    src_buffer[slices].get(out=out, stream=q)
                else:
                    src_buffer.get(out=out, stream=q)
                host_data = out
        await await_event(q)
        return host_data

    async def dirty_read_buffer(self, src_buffer, shape, dtype, stream=None):
        """
        Lecture rapide du buffer (Page-Locked DMA) sans affecter la queue principale.
        """
        with self.device:
            read_stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
            host_pinned = cp.cuda.alloc_pinned_memory(src_buffer.nbytes)
            count = int(np.prod(shape))
            host_data = np.frombuffer(host_pinned, dtype, count=count).reshape(shape)
            src_buffer.data.copy_to_host_async(host_data.ctypes.data, src_buffer.nbytes, stream=read_stream)
        await await_event(read_stream)
        return host_data

    def free_buffer(self, buffer_obj, shape, dtype):
        """
        Libère un buffer CUDA (CuPy gère son propre MemoryPool en interne).
        """
        pass
