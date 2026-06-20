import pyopencl as cl
import numpy as np
from .scheduler import Scheduler
from .executor import await_event

class AsyncTask:
    def __init__(self, kernel_code_opencl: str, kernel_name_opencl: str, kernel_code_cuda: str, kernel_name_cuda: str):
        self.kernel_code_opencl = kernel_code_opencl
        self.kernel_name_opencl = kernel_name_opencl
        self.kernel_code_cuda = kernel_code_cuda
        self.kernel_name_cuda = kernel_name_cuda
        self._compiled_programs = {}

    async def execute(self, scheduler: Scheduler, global_size, local_size, *args):
        if not args:
            device_manager = scheduler.get_best_device()
        else:
            device_manager = args[0].device_manager

        backend = scheduler.backend
        queue = device_manager.get_queue()

        if backend == "cuda":
            import cupy as cp
            device = device_manager.device
            
            device_id = device.id
            
            if device_id not in self._compiled_programs:
                # CuPy ajoute automatiquement -arch en fonction du GPU actif
                module = cp.RawKernel(self.kernel_code_cuda, self.kernel_name_cuda)
                self._compiled_programs[device_id] = module
            
            kernel = self._compiled_programs[device_id]
            
            cuda_args = []
            for arg in args:
                if hasattr(arg, 'cl_buffer'):
                    cuda_args.append(arg.cl_buffer) # cupy.ndarray
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
                    grid = ((global_size[0] + block[0] - 1) // block[0], (global_size[1] + block[1] - 1) // block[1], (global_size[2] + block[2] - 1) // block[2])
            else:
                # Simple 1D layout for CUDA blocks
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
                
            with device:
                kernel(grid, block, tuple(cuda_args), stream=queue)
                
            return await await_event(queue)

        else:
            context = device_manager.context
            
            if context not in self._compiled_programs:
                prg = cl.Program(context, self.kernel_code_opencl).build()
                self._compiled_programs[context] = getattr(prg, self.kernel_name_opencl)
            
            kernel = self._compiled_programs[context]
            
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
            else:
                local_size = None
                
            if local_size:
                # OpenCL exige que global_size soit un multiple strict de local_size
                global_size = tuple(((g + l - 1) // l) * l for g, l in zip(global_size, local_size))
                
            event = kernel(queue, global_size, local_size, *cl_args)
            return await await_event(event)
