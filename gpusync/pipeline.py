import ast
import inspect
import asyncio
from .memory import AsyncBuffer

class PipelineNode:
    def __init__(self, node_id, func_name, target_names, arg_names, kwarg_names):
        self.id = node_id
        self.func_name = func_name
        self.target_names = target_names
        self.arg_names = arg_names
        self.kwarg_names = kwarg_names
        self.dependencies = []

class PipelineVisitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            target_names = []
            if isinstance(node.targets[0], ast.Name):
                target_names = [node.targets[0].id]
            elif isinstance(node.targets[0], ast.Tuple):
                target_names = [elt.id for elt in node.targets[0].elts if isinstance(elt, ast.Name)]

            func_name = ""
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                func_name = node.value.func.attr

            arg_names = [a.id for a in node.value.args if isinstance(a, ast.Name)]
            kwarg_names = {}
            for kw in node.value.keywords:
                if isinstance(kw.value, ast.Name):
                    kwarg_names[kw.arg] = kw.value.id
                else:
                    try:
                        # Pour les constantes (comme global_size=(4,))
                        kwarg_names[kw.arg] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass

            pnode = PipelineNode(self.node_counter, func_name, target_names, arg_names, kwarg_names)
            self.nodes.append(pnode)
            self.node_counter += 1

        self.generic_visit(node)

class MultiDeviceScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers
        self._next_idx = 0

    def get_next_scheduler(self):
        if not self.schedulers:
            return None
        s = self.schedulers[self._next_idx]
        self._next_idx = (self._next_idx + 1) % len(self.schedulers)
        return s

def gpusync_pipeline(multi_scheduler: MultiDeviceScheduler):
    def decorator(func):
        import textwrap
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        
        visitor = PipelineVisitor()
        visitor.visit(tree)
        nodes = visitor.nodes

        for i in range(len(nodes)):
            node_A = nodes[i]
            for j in range(i):
                node_B = nodes[j]
                deps = set(node_A.arg_names + list(node_A.kwarg_names.values()))
                if any(t in deps for t in node_B.target_names):
                    node_A.dependencies.append(node_B)

        from .bg_loop import auto_sync

        @auto_sync
        async def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Initialiser l'environnement local avec les arguments
            local_env = dict(bound_args.arguments)
            # Ajouter les fonctions globales disponibles au moment de l'appel
            local_env.update(func.__globals__)

            in_degree = {n.id: len(n.dependencies) for n in nodes}
            graph = {n.id: [] for n in nodes}
            for n in nodes:
                for d in n.dependencies:
                    graph[d.id].append(n.id)

            queue = [n for n in nodes if in_degree[n.id] == 0]
            executed_nodes = 0
            all_done = asyncio.Event()

            async def run_node(node):
                nonlocal executed_nodes
                
                func_obj = local_env[node.func_name]
                real_args = [local_env[arg_name] for arg_name in node.arg_names]
                real_kwargs = {}
                for k, v_name in node.kwarg_names.items():
                    if isinstance(v_name, str) and v_name in local_env:
                        real_kwargs[k] = local_env[v_name]
                    else:
                        real_kwargs[k] = v_name

                scheduler = multi_scheduler.get_next_scheduler()
                
                # Auto-Transfert Inter-GPU
                for i, val in enumerate(real_args):
                    if isinstance(val, AsyncBuffer) and val.device_manager not in scheduler.devices_managers:
                        host_val = await val.to_host()
                        real_args[i] = await AsyncBuffer.to_device(scheduler, host_val)
                        
                # Injection du scheduler et du pipeline mode
                if hasattr(func_obj, '__wrapped__') or hasattr(func_obj, 'opencl_code'):
                    real_kwargs["scheduler"] = scheduler
                    real_kwargs["_pipeline_mode"] = True

                result = await func_obj(*real_args, **real_kwargs)
                
                if len(node.target_names) == 1:
                    local_env[node.target_names[0]] = result
                else:
                    for i, target_name in enumerate(node.target_names):
                        local_env[target_name] = result[i]
                        
                executed_nodes += 1
                if executed_nodes == len(nodes):
                    all_done.set()
                    
                for neighbor_id in graph[node.id]:
                    in_degree[neighbor_id] -= 1
                    if in_degree[neighbor_id] == 0:
                        next_node = next(n for n in nodes if n.id == neighbor_id)
                        asyncio.create_task(run_node(next_node))

            for n in queue:
                asyncio.create_task(run_node(n))
                
            if nodes:
                await all_done.wait()
                
            # Extraire le retour en lisant le vrai AST (Return node) et rapatrier les données
            return_node = next((n for n in ast.walk(tree) if isinstance(n, ast.Return)), None)
            if return_node:
                if isinstance(return_node.value, ast.Name):
                    val = local_env[return_node.value.id]
                    if isinstance(val, AsyncBuffer):
                        return await val.to_host()
                    return val
                elif isinstance(return_node.value, ast.Tuple):
                    val = tuple(local_env[elt.id] for elt in return_node.value.elts)
                    res = []
                    for v in val:
                        if isinstance(v, AsyncBuffer):
                            res.append(await v.to_host())
                        else:
                            res.append(v)
                    return tuple(res)
            return None

        return wrapper
    return decorator
