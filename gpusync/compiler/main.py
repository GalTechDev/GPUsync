import ast
import inspect
import textwrap
# pyrefly: ignore [missing-import]
import numpy as np

from .opencl import compile_to_opencl
from .cuda import compile_to_cuda

def _compile_and_wrap_kernel(tree, kernel_name, raw_cuda=None, constants=None):
    """
    Fonction helper interne partagée (DRY).
    1. Analyse l'AST pour calculer le nombre de valeurs de retour (returns).
    2. Lance la transpilation vers OpenCL et CUDA.
    3. Compile et met en cache l'AsyncTask correspondante.
    4. Construit l'enveloppe de fonction synchrone/asynchrone `@auto_sync` pour l'utilisateur final,
       qui alloue automatiquement les buffers de sortie VRAM si nécessaire.
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
        kernel_code_cuda=kernel_code_cuda, kernel_name_cuda=kernel_name_cuda
    )
    
    @auto_sync
    async def wrapper(*args, global_size, scheduler, _pipeline_mode=False, local_size=None, stream=None, fire_and_forget=False):
        """
        Enveloppe de kernel générée dynamiquement au moment de l'appel.
        Alloue les buffers de sortie VRAM, appelle AsyncTask pour exécuter le kernel
        sur la queue ou le stream GPU cible, et renvoie les résultats vers le CPU ou
        sous forme d'AsyncBuffer (selon le pipeline_mode).
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
            
    wrapper.opencl_code = kernel_code_opencl 
    wrapper.cuda_code = kernel_code_cuda
    wrapper.task = task
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
