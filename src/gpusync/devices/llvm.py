import concurrent.futures
import logging

# pyrefly: ignore [missing-import]
import llvmlite.binding as llvm
import numpy as np

import tempfile
import os
import ctypes
import sys
import distutils.ccompiler

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
        
        # 1. Compilation JIT du wrapper C
        if name not in getattr(task, "_compiled_programs", {}):
            code = getattr(task, "kernel_code_llvm", None)
            if not code:
                logger.error("LLVM execute_kernel: kernel_code_llvm est manquant.")
                return None
                
            cache_dir = os.path.join(tempfile.gettempdir(), "gpusync_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            c_path = os.path.join(cache_dir, f"{name}.c")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            compiler = distutils.ccompiler.new_compiler()
            try:
                # Ajout du flag OpenMP si MSVC, ou -fopenmp si GCC
                # distutils gère les flags additionnels via extra_postargs
                is_msvc = compiler.compiler_type == "msvc"
                cflags = ["/openmp"] if is_msvc else ["-fopenmp"]
                ldflags = ["/openmp"] if is_msvc else ["-fopenmp"]
                
                try:
                    objs = compiler.compile([c_path], output_dir=cache_dir, extra_postargs=cflags)
                    out_path = os.path.join(cache_dir, f"{name}_lib")
                    compiler.link_shared_object(objs, out_path, extra_postargs=ldflags)
                except Exception as comp_e:
                    logger.warning(f"Compilation avec OpenMP a échoué ({comp_e}), tentative sans OpenMP...")
                    # Fallback sans multi-threading matériel
                    objs = compiler.compile([c_path], output_dir=cache_dir)
                    out_path = os.path.join(cache_dir, f"{name}_lib")
                    compiler.link_shared_object(objs, out_path)
                
                # Charger la DLL
                ext = ".dll" if sys.platform == "win32" else ".so"
                dll_path = out_path + ext
                if not os.path.exists(dll_path):
                     for f in os.listdir(cache_dir):
                         if f.startswith(f"{name}_lib") and (f.endswith(".dll") or f.endswith(".so") or f.endswith(".pyd")):
                             dll_path = os.path.join(cache_dir, f)
                             break
                             
                lib = ctypes.CDLL(dll_path)
                c_func = getattr(lib, f"launch_{name}")
                
                if not hasattr(task, "_compiled_programs"):
                    task._compiled_programs = {}
                task._compiled_programs[name] = c_func
                
            except Exception as e:
                logger.error(f"Erreur de compilation JIT LLVM/C++ : {e}")
                return None
                
        c_func = task._compiled_programs[name]
        
        # 2. Préparation des arguments CTypes
        ctypes_args = [ctypes.c_int(total_size)]
        
        for arg in inputs:
            if isinstance(arg, int):
                ctypes_args.append(ctypes.c_int(arg))
            elif isinstance(arg, float):
                ctypes_args.append(ctypes.c_float(arg))
            elif isinstance(arg, np.ndarray):
                ctypes_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                
        for out_buf in outputs:
            ctypes_args.append(out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            
        # 3. Exécution native (zéro-copy)
        c_func(*ctypes_args)

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
        if dest_buffer.shape != host_data.shape:
            slices = tuple(slice(0, dim) for dim in host_data.shape)
            np.copyto(dest_buffer[slices], host_data)
        else:
            np.copyto(dest_buffer, host_data)

    async def read_buffer(self, src_buffer, shape, dtype, out=None, queue=None):
        """
        Lecture directe RAM à RAM.
        """
        if out is not None:
            if src_buffer.shape != out.shape:
                slices = tuple(slice(0, dim) for dim in out.shape)
                np.copyto(out, src_buffer[slices])
            else:
                np.copyto(out, src_buffer)
            return out
        return src_buffer.copy()

    async def dirty_read_buffer(self, src_buffer, shape, dtype, stream=None):
        """
        Lecture directe Zero-Copy.
        """
        return src_buffer.copy()

    def free_buffer(self, buffer_obj, shape, dtype):
        """
        Libère un buffer CPU (LLVM CPU JIT).
        """
        pass
