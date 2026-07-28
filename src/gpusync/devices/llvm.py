import concurrent.futures
import logging
import ctypes
import math
import numpy as np

# pyrefly: ignore [missing-import]
import llvmlite.binding as llvm

logger = logging.getLogger(__name__)

# Initialisation unique du sous-système LLVM JIT
_llvm_initialized = False

def _init_llvm():
    global _llvm_initialized
    if not _llvm_initialized:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        _llvm_initialized = True


class LLVMDeviceDummy:
    def __init__(self, name="CPU (LLVM MCJIT)", index=0):
        self.name = name
        self.index = index


class LLVMDeviceManager:
    """
    Gère la compilation JIT native LLVM et l'exécution optimisée SIMD multi-cœurs
    sur le processeur hôte (CPU) via llvmlite.binding.
    """

    backend = "llvm"

    @classmethod
    def discover_devices(cls, name_filter: str | None = None):
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
        self._buffer_pool: dict = {}
        logger.info(f"LLVMDeviceManager initialisé : {self.name} avec {num_queues} threads (MCJIT).")

    def get_queue(self):
        return self.executor

    async def execute_kernel(self, task, global_size, local_size, *args, stream=None, fire_and_forget=False):
        """
        Exécute le kernel LLVM sur le CPU en parallélisant sur le global_size
        via ThreadPoolExecutor et la boucle interne du wrapper JIT.
        """
        total_size = global_size if isinstance(global_size, int) else int(np.prod(global_size))

        # Extraction des tableaux bruts numpy/ctypes
        raw_args = []
        for arg in args:
            if hasattr(arg, "cl_buffer"):
                buf = arg.cl_buffer
                if isinstance(buf, np.ndarray):
                    raw_args.append(buf)
                else:
                    raw_args.append(arg)
            else:
                raw_args.append(arg)

        # Calculer le nombre de retours et séparer les arguments d'entrée et de sortie
        num_returns = getattr(task, "num_returns", 0)
        if num_returns > 0:
            inputs = raw_args[:-num_returns]
            outputs = raw_args[-num_returns:]
        else:
            inputs = raw_args
            outputs = []

        name = getattr(task, "kernel_name_llvm", getattr(task, "name", "unknown_kernel"))
        
        # 1. Compilation JIT LLVM
        if name not in getattr(task, "_compiled_programs", {}):
            code_ir = getattr(task, "kernel_code_llvm", None)
            if not code_ir:
                logger.error("LLVM execute_kernel: kernel_code_llvm (LLVM IR) est manquant.")
                return None
                
            try:
                mod = llvm.parse_assembly(code_ir)
                mod.verify()
                
                target = llvm.Target.from_default_triple()
                target_machine = target.create_target_machine(opt=3)
                
                # Optimisation native par le target machine
                engine = llvm.create_mcjit_compiler(mod, target_machine)
                engine.finalize_object()
                engine.run_static_constructors()
                
                func_ptr = engine.get_function_address(f"launch_{name}")
                
                # Signature CTypes dynamique
                # args: start(int), end(int), *inputs(float/int/ptr), *outputs(ptr)
                argtypes = [ctypes.c_int32, ctypes.c_int32]
                for arg in inputs:
                    if isinstance(arg, int): argtypes.append(ctypes.c_int32)
                    elif isinstance(arg, float): argtypes.append(ctypes.c_float)
                    else: argtypes.append(ctypes.POINTER(ctypes.c_float))
                for _ in outputs:
                    argtypes.append(ctypes.POINTER(ctypes.c_float))
                    
                cfunc = ctypes.CFUNCTYPE(None, *argtypes)(func_ptr)
                
                if not hasattr(task, "_compiled_programs"):
                    task._compiled_programs = {}
                task._compiled_programs[name] = (engine, cfunc)
                
            except Exception as e:
                logger.error(f"Erreur de compilation JIT LLVM : {e}")
                return None
                
        engine, cfunc = task._compiled_programs[name]
        
        # 2. Préparation des arguments CTypes (hors start/end)
        ctypes_args = []
        for arg in inputs:
            if isinstance(arg, int):
                ctypes_args.append(ctypes.c_int32(arg))
            elif isinstance(arg, float):
                ctypes_args.append(ctypes.c_float(arg))
            elif isinstance(arg, np.ndarray):
                ctypes_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                
        for out_buf in outputs:
            ctypes_args.append(out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            
        # 3. Exécution native (zéro-copy) en multi-threading
        chunk_size = math.ceil(total_size / self.num_threads)
        
        def run_chunk(start):
            end = min(start + chunk_size, total_size)
            if start < end:
                cfunc(start, end, *ctypes_args)
                
        futures = []
        for i in range(self.num_threads):
            start = i * chunk_size
            futures.append(self.executor.submit(run_chunk, start))
            
        concurrent.futures.wait(futures)
        return None

    async def allocate_buffer_from_host(self, host_data, queue=None):
        return np.array(host_data, copy=True)

    def allocate_empty_buffer(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    async def write_buffer(self, dest_buffer, host_data, queue=None):
        if dest_buffer.shape != host_data.shape:
            slices = tuple(slice(0, dim) for dim in host_data.shape)
            np.copyto(dest_buffer[slices], host_data)
        else:
            np.copyto(dest_buffer, host_data)

    async def read_buffer(self, src_buffer, shape, dtype, out=None, queue=None):
        if out is not None:
            if src_buffer.shape != out.shape:
                slices = tuple(slice(0, dim) for dim in out.shape)
                np.copyto(out, src_buffer[slices])
            else:
                np.copyto(out, src_buffer)
            return out
        return src_buffer.copy()

    async def dirty_read_buffer(self, src_buffer, shape, dtype, stream=None):
        return src_buffer.copy()

    def free_buffer(self, buffer_obj, shape, dtype):
        pass
