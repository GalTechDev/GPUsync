# pyrefly: ignore [missing-import]
import logging

# pyrefly: ignore [missing-import]
import numpy as np
import pyopencl as cl

from ..executor import await_event

logger = logging.getLogger(__name__)


class OpenCLDeviceManager:
    """
    Gère le contexte de calcul OpenCL et le pool de CommandQueues
    associées à un GPU physique OpenCL.
    """

    backend = "opencl"

    @classmethod
    def discover_devices(cls, name_filter: str | None = None):
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
        self.queues = [cl.CommandQueue(self.context, self.device, properties=properties) for _ in range(num_queues)]

        self.next_queue_idx = 0
        self._buffer_pool: dict = {}
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
            if isinstance(task.kernel_code_opencl, bytes):
                # Chargement transparent du bytecode SPIR-V
                prg = cl.Program(self.context, task.kernel_code_opencl).build()
            else:
                # Compilation du code source OpenCL C textuel standard
                build_opts = ["-cl-fast-relaxed-math", "-cl-mad-enable"]
                prg = cl.Program(self.context, task.kernel_code_opencl).build(options=build_opts)
            task._compiled_programs[self.context] = getattr(prg, task.kernel_name_opencl)

        kernel = task._compiled_programs[self.context]

        cl_args = []
        for arg in args:
            if hasattr(arg, "cl_buffer"):
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

            # Auto-tuning dynamique basé sur les caractéristiques réelles du GPU et du Kernel
            try:
                max_group_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
                pref_multiple = kernel.get_work_group_info(
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device
                )
            except Exception:
                max_group_size = 256
                pref_multiple = 32

            dims = len(global_size)
            if dims == 1:
                g_size = global_size[0]
                if g_size < pref_multiple:
                    local_size = (max(1, g_size),)
                else:
                    # Cible par défaut à 256 si possible, ajustée selon la limite matérielle
                    candidate = 256
                    while candidate > max_group_size:
                        candidate //= 2
                    local_size = (candidate,)
            elif dims == 2:
                candidate_dim = 16
                while (candidate_dim * candidate_dim) > max_group_size:
                    candidate_dim //= 2
                local_size = (candidate_dim, candidate_dim)
            elif dims == 3:
                candidate_dim = 8
                while (candidate_dim * candidate_dim * candidate_dim) > max_group_size:
                    candidate_dim //= 2
                local_size = (candidate_dim, candidate_dim, candidate_dim)

            logger.debug(
                f"[Auto-Tuning] Kernel '{task.kernel_name_opencl}' -> local_size: {local_size} (Max: {max_group_size}, Multiple préféré: {pref_multiple})"
            )

        else:
            if isinstance(global_size, int):
                global_size = (global_size,)

        if local_size:
            global_size = tuple(((g + loc_sz - 1) // loc_sz) * loc_sz for g, loc_sz in zip(global_size, local_size, strict=False))

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
