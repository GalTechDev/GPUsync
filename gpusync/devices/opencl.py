# pyrefly: ignore [missing-import]
import pyopencl as cl
import logging
# pyrefly: ignore [missing-import]
import numpy as np
from ..executor import await_event

logger = logging.getLogger(__name__)

class OpenCLDeviceManager:
    """
    Gère le contexte de calcul OpenCL et le pool de CommandQueues
    associées à un GPU physique OpenCL.
    """
    backend = "opencl"

    @classmethod
    def discover_devices(cls, name_filter: str = None):
        """
        Découvre et renvoie tous les périphériques OpenCL disponibles sur la machine.
        """
        devices = []
        try:
            # pyrefly: ignore [missing-import]
            import pyopencl as cl
            platforms = cl.get_platforms()
            all_devs = []
            for platform in platforms:
                try:
                    all_devs.extend(platform.get_devices())
                except Exception:
                    pass
            
            # Filtrer pour privilégier uniquement les périphériques GPU
            gpu_devs = [d for d in all_devs if d.type & cl.device_type.GPU]
            devs_to_use = gpu_devs if gpu_devs else all_devs
            
            for dev in devs_to_use:
                if name_filter and name_filter.lower() not in dev.name.lower():
                    continue
                devices.append(dev)
        except Exception:
            pass
        return devices

    def __init__(self, device: cl.Device, num_queues: int = 16):
        self.device = device
        self.backend = "opencl"
        self.name = device.name
        
        # Création d'un unique contexte pour ce périphérique
        self.context = cl.Context([device])
        
        # Pool de CommandQueues pour forcer la concurrence d'exécution
        properties = cl.command_queue_properties.PROFILING_ENABLE
        self.queues = [
            cl.CommandQueue(self.context, self.device, properties=properties) 
            for _ in range(num_queues)
        ]
        
        self.next_queue_idx = 0
        self._buffer_pool = {}
        logger.info(f"OpenCLDeviceManager initialisé : {self.name} avec {num_queues} Command Queues.")

    def get_queue(self) -> cl.CommandQueue:
        """
        Retourne une file de commande (CommandQueue) selon une stratégie Round-Robin.
        """
        queue = self.queues[self.next_queue_idx]
        self.next_queue_idx = (self.next_queue_idx + 1) % len(self.queues)
        return queue

    async def execute_kernel(self, task, global_size, local_size, *args, stream=None, fire_and_forget=False):
        """
        Exécute le kernel OpenCL de la tâche sur ce périphérique.
        """
        queue = stream if stream is not None else self.get_queue()
        
        if self.context not in task._compiled_programs:
            build_opts = ["-cl-fast-relaxed-math", "-cl-mad-enable"]
            prg = cl.Program(self.context, task.kernel_code_opencl).build(options=build_opts)
            task._compiled_programs[self.context] = getattr(prg, task.kernel_name_opencl)
            
        kernel = task._compiled_programs[self.context]
        
        cl_args = []
        for arg in args:
            if hasattr(arg, 'cl_buffer'):
                cl_args.append(arg.cl_buffer)
            elif isinstance(arg, int):
                cl_args.append(np.int32(arg))
            elif isinstance(arg, float):
                cl_args.append(np.float32(arg))
            else:
                cl_args.append(arg)
                
        if local_size is None:
            if isinstance(global_size, int):
                global_size = (global_size,)
                local_size = (256,)
            elif len(global_size) == 1:
                local_size = (256,)
            elif len(global_size) == 2:
                local_size = (16, 16)
            elif len(global_size) == 3:
                local_size = (8, 8, 8)
                
        if local_size:
            global_size = tuple(((g + l - 1) // l) * l for g, l in zip(global_size, local_size))
            
        event = kernel(queue, global_size, local_size, *cl_args)
        
        if not fire_and_forget:
            return await await_event(event)
        else:
            return None

    async def allocate_buffer_from_host(self, host_data, queue=None):
        """
        Alloue un buffer OpenCL en VRAM (ou réutilise un buffer disponible du pool)
        et y copie les données de l'hôte.
        """
        q = queue if queue is not None else self.get_queue()
        cl_buffer = self.allocate_empty_buffer(host_data.shape, host_data.dtype)
        event = cl.enqueue_copy(q, cl_buffer, host_data, is_blocking=False)
        await await_event(event)
        return cl_buffer

    def allocate_empty_buffer(self, shape, dtype):
        """
        Alloue un buffer OpenCL vide en VRAM (ou le récupère du pool).
        """
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        if nbytes in self._buffer_pool and self._buffer_pool[nbytes]:
            return self._buffer_pool[nbytes].pop()
            
        flags = cl.mem_flags.READ_WRITE
        cl_buffer = cl.Buffer(self.context, flags, size=nbytes)
        return cl_buffer

    def free_buffer(self, buffer_obj, shape, dtype):
        """
        Remet le buffer dans le pool pour une réutilisation ultérieure.
        """
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        if nbytes not in self._buffer_pool:
            self._buffer_pool[nbytes] = []
        self._buffer_pool[nbytes].append(buffer_obj)

    async def write_buffer(self, dest_buffer, host_data, queue=None):
        """
        Écrit des données de l'hôte dans un buffer OpenCL existant.
        """
        q = queue if queue is not None else self.get_queue()
        event = cl.enqueue_copy(q, dest_buffer, host_data, is_blocking=False)
        await await_event(event)

    async def read_buffer(self, src_buffer, shape, dtype, out=None, queue=None):
        """
        Lit un buffer OpenCL vers la mémoire de l'hôte.
        """
        q = queue if queue is not None else self.get_queue()
        if out is None:
            host_data = np.empty(shape, dtype=dtype)
        else:
            host_data = out
        event = cl.enqueue_copy(q, host_data, src_buffer, is_blocking=False)
        await await_event(event)
        return host_data

    async def dirty_read_buffer(self, src_buffer, shape, dtype, stream=None):
        """
        Lecture rapide du buffer sans affecter la queue principale.
        """
        q = stream if stream is not None else cl.CommandQueue(self.context, self.device)
        host_data = np.empty(shape, dtype=dtype)
        event = cl.enqueue_copy(q, host_data, src_buffer, is_blocking=False)
        await await_event(event)
        return host_data
