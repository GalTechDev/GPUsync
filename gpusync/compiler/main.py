import ast
import inspect
import textwrap
# pyrefly: ignore [missing-import]
import numpy as np

from .opencl import compile_to_opencl
from .cuda import compile_to_cuda
from .llvm import compile_to_llvm
from .llvm_cuda import compile_to_llvm_cuda

def _compile_and_wrap_kernel(tree, kernel_name, raw_cuda=None, constants=None):
    """
    Fonction helper interne partagée (DRY).
    1. Analyse l'AST pour calculer le nombre de valeurs de retour (returns).
    2. Lance la transpilation vers OpenCL, CUDA, LLVM CPU et LLVM NVPTX GPU.
    3. Compile et met en cache l'AsyncTask correspondante.
    4. Construit l'enveloppe de fonction synchrone/asynchrone `@auto_sync` pour l'utilisateur final.
    """
    # Calcul du nombre de retours
    num_returns = 0
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if returns:
        ret_node = returns[0]
        if ret_node.value is None:
            num_returns = 0
        elif isinstance(ret_node.value, ast.Tuple):
            num_returns = len(ret_node.value.elts)
        else:
            num_returns = 1
            
    # Compilation
    kernel_code_opencl = compile_to_opencl(tree, kernel_name, num_returns, constants)
    kernel_code_llvm = compile_to_llvm(tree, kernel_name, num_returns, constants)
    kernel_code_ptx = compile_to_llvm_cuda(tree, kernel_name, num_returns, constants)
    
    if raw_cuda:
        kernel_code_cuda = raw_cuda
        kernel_name_cuda = kernel_name
    else:
        kernel_code_cuda = compile_to_cuda(tree, kernel_name, num_returns, constants)
        kernel_name_cuda = kernel_name
        
    # Imports dynamiques locaux pour éviter les cycles de dépendances de packages
    from ..task import AsyncTask
    from ..utils.bg_loop import auto_sync
    from ..memory import AsyncBuffer
    
    task = AsyncTask(
        kernel_code_opencl=kernel_code_opencl, kernel_name_opencl=kernel_name,
        kernel_code_cuda=kernel_code_cuda, kernel_name_cuda=kernel_name_cuda,
        kernel_code_llvm=kernel_code_llvm, kernel_name_llvm=kernel_name,
        kernel_code_ptx=kernel_code_ptx, kernel_name_ptx=kernel_name
    )
    
    @auto_sync
    async def wrapper(*args, global_size, scheduler, _pipeline_mode=False, local_size=None, stream=None, fire_and_forget=False):
        """
        Enveloppe de kernel générée dynamiquement au moment de l'appel.
        Alloue les buffers de sortie VRAM, appelle AsyncTask pour exécuter le kernel
        sur la queue, le stream GPU ou le moteur JIT CPU LLVM.
        """
        out_buffers = []
        if num_returns > 0:
            total_size = global_size if isinstance(global_size, int) else np.prod(global_size)
            for _ in range(num_returns):
                buf = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
                out_buffers.append(buf)
                
        new_args = list(args) + out_buffers
        await task.execute(scheduler, global_size, local_size, *new_args, stream=stream, fire_and_forget=fire_and_forget)
        
        # Gestion intelligente des retours (Auto-rapatriement sur le CPU)
        if num_returns == 0:
            return None
        elif num_returns == 1:
            if _pipeline_mode or fire_and_forget:
                return out_buffers[0]
            return await out_buffers[0].to_host()
        else:
            if _pipeline_mode or fire_and_forget:
                return tuple(out_buffers)
            return tuple([await buf.to_host() for buf in out_buffers])
            
    @auto_sync
    async def map_out_of_core(host_data, scheduler, *extra_args, chunk_size=None, concurrency=2):
        """
        Traitement Out-of-Core 100% automatique et monoligne.
        Accepte aussi bien un tableau NumPy qu'un Générateur / Itérable (ex: flux de lecture disque NVMe).
        """
        import asyncio
        import inspect

        is_gen = inspect.isgenerator(host_data) or inspect.isgeneratorfunction(host_data) or not hasattr(host_data, '__len__')

        if is_gen:
            gen = host_data() if inspect.isgeneratorfunction(host_data) else host_data
            sem = asyncio.Semaphore(concurrency)
            pending_tasks = []

            async def process_gen_chunk(slice_data, idx):
                async with sem:
                    buf_in = await AsyncBuffer.to_device(scheduler, slice_data)
                    gpu_extra = []
                    for ex in extra_args:
                        if isinstance(ex, np.ndarray):
                            gpu_extra.append(await AsyncBuffer.to_device(scheduler, ex))
                        else:
                            gpu_extra.append(ex)

                    res = await wrapper(buf_in, *gpu_extra, global_size=len(slice_data), scheduler=scheduler, _pipeline_mode=True)
                    if isinstance(res, tuple):
                        res_host = tuple([await b.to_host() for b in res])
                    else:
                        res_host = await res.to_host()
                    return (idx, res_host)

            for idx, slice_data in enumerate(gen):
                task = asyncio.create_task(process_gen_chunk(slice_data, idx))
                pending_tasks.append(task)

            chunk_results = await asyncio.gather(*pending_tasks)
            chunk_results.sort(key=lambda x: x[0])
            results_host = [r[1] for r in chunk_results]
        else:
            N = len(host_data)
            if chunk_size is None:
                chunk_size = max(50_000, min(N, 500_000))

            num_chunks = (N + chunk_size - 1) // chunk_size
            results_host = [None] * num_chunks
            sem = asyncio.Semaphore(concurrency)

            async def process_chunk(idx):
                async with sem:
                    start_i = idx * chunk_size
                    end_i = min(N, (idx + 1) * chunk_size)
                    slice_data = host_data[start_i:end_i]

                    chunk_extra = []
                    for extra in extra_args:
                        if isinstance(extra, np.ndarray) and len(extra) == N:
                            chunk_extra.append(extra[start_i:end_i])
                        else:
                            chunk_extra.append(extra)

                    buf_in = await AsyncBuffer.to_device(scheduler, slice_data)
                    gpu_extra = []
                    for ex in chunk_extra:
                        if isinstance(ex, np.ndarray):
                            gpu_extra.append(await AsyncBuffer.to_device(scheduler, ex))
                        else:
                            gpu_extra.append(ex)

                    res = await wrapper(buf_in, *gpu_extra, global_size=len(slice_data), scheduler=scheduler, _pipeline_mode=True)

                    if isinstance(res, tuple):
                        res_host = tuple([await b.to_host() for b in res])
                    else:
                        res_host = await res.to_host()
                    results_host[idx] = res_host

            tasks = [asyncio.create_task(process_chunk(i)) for i in range(num_chunks)]
            await asyncio.gather(*tasks)

        if num_returns == 0:
            return None
        elif num_returns == 1:
            return np.concatenate(results_host)
        else:
            return tuple(
                np.concatenate([r[ret_idx] for r in results_host])
                for ret_idx in range(num_returns)
            )

    wrapper.opencl_code = kernel_code_opencl 
    wrapper.cuda_code = kernel_code_cuda
    wrapper.llvm_code = kernel_code_llvm
    wrapper.ptx_code = kernel_code_ptx
    wrapper.task = task
    wrapper.map_out_of_core = map_out_of_core
    return wrapper

def gpusync_kernel(func=None, raw_cuda=None, constants=None):
    """
    Décorateur JIT multi-backend principal.
    Traduit la fonction Python en kernels OpenCL et CUDA C++.
    
    Exemple :
        @gpusync_kernel
        def addition(ctx: int, a: list, b: list) -> tuple:
            return a[ctx] + b[ctx]
            
    Args:
        func (function, optional): La fonction Python à décorer.
        raw_cuda (str, optional): Code CUDA brut pour bypasser la transpilation automatique.
        constants (dict, optional): Constantes de configuration pour l'AST (ex: tailles de blocs).
    """
    if func is None:
        def wrapper(f):
            return gpusync_kernel(f, raw_cuda=raw_cuda, constants=constants)
        return wrapper

    # Extraction et nettoyage du code source Python
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    lines = source.split('\n')
    lines = [l for l in lines if not l.strip().startswith('@')]
    source = '\n'.join(lines)
    
    tree = ast.parse(source)
    func_def = tree.body[0]
    kernel_name = func_def.name
    
    return _compile_and_wrap_kernel(tree, kernel_name, raw_cuda, constants)

def gpusync_kernel_from_source(source: str, raw_cuda=None, constants=None):
    """
    Variante JIT de `gpusync_kernel` pour les codes sources générés dynamiquement (factories).
    Évite l'utilisation d'inspect.getsource() sur les objets de fonction temporaires.
    
    Args:
        source (str): Chaîne de caractères contenant le code de la fonction Python.
        raw_cuda (str, optional): Code CUDA brut pour bypasser la transpilation.
        constants (dict, optional): Constantes de configuration AST.
    """
    lines = source.split('\n')
    lines = [l for l in lines if not l.strip().startswith('@')]
    source = '\n'.join(lines)

    tree = ast.parse(source)
    func_def = tree.body[0]
    kernel_name = func_def.name

    return _compile_and_wrap_kernel(tree, kernel_name, raw_cuda, constants)
