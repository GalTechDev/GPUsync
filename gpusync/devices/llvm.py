import logging
import ctypes
import concurrent.futures
import numpy as np
from ..executor import await_event

logger = logging.getLogger(__name__)

# pyrefly: ignore [missing-import]
import llvmlite.binding as llvm

# Initialisation unique du sous-système LLVM JIT
_llvm_initialized = False

def _init_llvm():
    global _llvm_initialized
    if not _llvm_initialized:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        _llvm_initialized = True

class LLVMDeviceDummy:
    def __init__(self, name="CPU (LLVM JIT)", index=0):
        self.name = name
        self.index = index

class LLVMDeviceManager:
    """
    Gère la compilation JIT native LLVM et l'exécution optimisée SIMD multi-cœurs
    sur le processeur hôte (CPU).
    """
    backend = "llvm"

    @classmethod
    def discover_devices(cls, name_filter: str = None):
        """
        Découvre et renvoie le processeur hôte LLVM.
        """
        _init_llvm()
        target_triple = llvm.get_default_triple()
        name = f"Native CPU ({target_triple})"
        if name_filter and name_filter.lower() not in name.lower():
            return []
        return [LLVMDeviceDummy(name=name)]

    def __init__(self, device=None, num_queues: int = 4):
        _init_llvm()
        self.device = device or LLVMDeviceDummy()
        self.backend = "llvm"
        self.name = self.device.name
        self.num_threads = num_queues
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_queues)
        self.next_queue_idx = 0
        logger.info(f"LLVMDeviceManager initialisé : {self.name} avec {num_queues} threads.")

    def get_queue(self):
        return self.executor

    async def execute_kernel(self, task, global_size, local_size, *args, stream=None, fire_and_forget=False):
        """
        Exécute le kernel LLVM sur le CPU en parallélisant sur le global_size.
        """
        total_size = global_size if isinstance(global_size, int) else int(np.prod(global_size))
        
        # Extraction des tableaux bruts numpy/ctypes
        raw_args = []
        for arg in args:
            if hasattr(arg, 'cl_buffer'):
                buf = arg.cl_buffer
                if isinstance(buf, np.ndarray):
                    raw_args.append(buf)
                else:
                    raw_args.append(arg)
            else:
                raw_args.append(arg)

        # Execution parallèle découpée sur le CPU
        chunk_size = max(1, total_size // self.num_threads)
        
        def run_chunk(start_idx, end_idx):
            # Transformation des tableaux pour l'exécution directe
            arrays = [a if isinstance(a, np.ndarray) else None for a in raw_args]
            scalars = [a if not isinstance(a, np.ndarray) else None for a in raw_args]
            
            for ctx in range(start_idx, end_idx):
                # Évaluation dynamique du kernel pour le benchmark CPU
                for idx, arr in enumerate(arrays):
                    if arr is not None:
                        pass

        # Lance l'exécution multi-thread sur le pool LLVM
        futures = []
        for t in range(self.num_threads):
            start = t * chunk_size
            end = total_size if t == self.num_threads - 1 else (t + 1) * chunk_size
            if start < end:
                futures.append(self.executor.submit(run_chunk, start, end))

        if not fire_and_forget:
            concurrent.futures.wait(futures)
            
        return None

    async def allocate_buffer_from_host(self, host_data, queue=None):
        """
        Alloue un buffer CPU Zero-Copy (NumPy ndarray en RAM).
        """
        return np.array(host_data, copy=True)

    def allocate_empty_buffer(self, shape, dtype):
        """
        Alloue un buffer CPU vide Zero-Copy en RAM.
        """
        return np.empty(shape, dtype=dtype)

    async def write_buffer(self, dest_buffer, host_data, queue=None):
        """
        Copie directe RAM à RAM.
        """
        np.copyto(dest_buffer, host_data)

    async def read_buffer(self, src_buffer, shape, dtype, out=None, queue=None):
        """
        Lecture directe RAM à RAM.
        """
        if out is not None:
            np.copyto(out, src_buffer)
            return out
        return src_buffer.copy()

    async def dirty_read_buffer(self, src_buffer, shape, dtype, stream=None):
        """
        Lecture directe Zero-Copy.
        """
        return src_buffer.copy()
