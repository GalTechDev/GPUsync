import ast
import inspect
import asyncio
from .memory import AsyncBuffer

class PipelineNode:
    def __init__(self, node_id, func_name, target_names, arg_names, kwarg_names, device_affinity=None, stream_affinity=None):
        self.id = node_id
        self.func_name = func_name
        self.target_names = target_names
        self.arg_names = arg_names
        self.kwarg_names = kwarg_names
        self.device_affinity = device_affinity
        self.stream_affinity = stream_affinity
        self.dependencies = []

class PipelineVisitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def _process_call(self, call_node, target_names):
        func_name = ""
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            func_name = call_node.func.attr

        arg_names = []
        for a in call_node.args:
            if isinstance(a, ast.Name):
                arg_names.append(a.id)
            elif isinstance(a, ast.Constant):
                arg_names.append(a.value)
            else:
                arg_names.append(None)
        kwarg_names = {}
        device_affinity = None
        stream_affinity = None
        
        for kw in call_node.keywords:
            if kw.arg == "_device":
                try:
                    device_affinity = ast.literal_eval(kw.value)
                except ValueError:
                    pass
            elif kw.arg == "_stream":
                try:
                    stream_affinity = ast.literal_eval(kw.value)
                except ValueError:
                    pass
            else:
                if isinstance(kw.value, ast.Name):
                    kwarg_names[kw.arg] = kw.value.id
                elif isinstance(kw.value, ast.Attribute):
                    if isinstance(kw.value.value, ast.Name):
                        kwarg_names[kw.arg] = f"{kw.value.value.id}.{kw.value.attr}"
                else:
                    try:
                        kwarg_names[kw.arg] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass

        pnode = PipelineNode(self.node_counter, func_name, target_names, arg_names, kwarg_names, device_affinity, stream_affinity)
        self.nodes.append(pnode)
        self.node_counter += 1

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            target_names = []
            if isinstance(node.targets[0], ast.Name):
                target_names = [node.targets[0].id]
            elif isinstance(node.targets[0], ast.Tuple):
                target_names = [elt.id for elt in node.targets[0].elts if isinstance(elt, ast.Name)]
            self._process_call(node.value, target_names)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            # Pas de cibles pour un appel in-place
            self._process_call(node.value, [])

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

        # Calcul des dépendances par analyse d'accès (inclut les mutations in-place)
        for i in range(len(nodes)):
            node_A = nodes[i]
            for j in range(i):
                node_B = nodes[j]
                
                # Récupère tous les symboles utilisés (args, kwargs, cibles)
                deps_A = set(node_A.arg_names + [v for v in node_A.kwarg_names.values() if isinstance(v, str)] + node_A.target_names)
                deps_B = set(node_B.arg_names + [v for v in node_B.kwarg_names.values() if isinstance(v, str)] + node_B.target_names)
                
                # Si deux nœuds partagent un même objet en argument, ils doivent être séquentiels 
                # (car en GPU, un pointeur argument est potentiellement modifié in-place)
                if deps_A.intersection(deps_B):
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
                real_args = []
                for arg_name in node.arg_names:
                    if isinstance(arg_name, str) and arg_name in local_env:
                        real_args.append(local_env[arg_name])
                    else:
                        real_args.append(arg_name)
                real_kwargs = {}
                for k, v_name in node.kwarg_names.items():
                    if isinstance(v_name, str):
                        if v_name in local_env:
                            real_kwargs[k] = local_env[v_name]
                        elif '.' in v_name:
                            parts = v_name.split('.')
                            if parts[0] in local_env:
                                obj = local_env[parts[0]]
                                try:
                                    real_kwargs[k] = getattr(obj, parts[1])
                                except AttributeError:
                                    real_kwargs[k] = v_name
                            else:
                                real_kwargs[k] = v_name
                        else:
                            real_kwargs[k] = v_name
                    else:
                        real_kwargs[k] = v_name

                # --- ROUTAGE DU DEVICE ---
                if node.device_affinity and hasattr(multi_scheduler, 'get_device'):
                    device_manager = multi_scheduler.get_device(node.device_affinity)
                else:
                    # Round-Robin classique
                    sched_or_dm = multi_scheduler.get_next_scheduler()
                    if hasattr(sched_or_dm, 'get_best_device'):
                        device_manager = sched_or_dm.get_best_device()
                    else:
                        device_manager = sched_or_dm

                # --- ROUTAGE DU STREAM ---
                stream = None
                if node.stream_affinity is not None:
                    if hasattr(device_manager, 'queues') and node.stream_affinity < len(device_manager.queues):
                        stream = device_manager.queues[node.stream_affinity]
                
                # Auto-Transfert Inter-GPU
                for i, val in enumerate(real_args):
                    if isinstance(val, AsyncBuffer) and getattr(val, 'device_manager', None) != device_manager:
                        host_val = await val.to_host()
                        # to_device accepte maintenant directement un device_manager
                        real_args[i] = await AsyncBuffer.to_device(device_manager, host_val)
                        
                # Injection du scheduler et du pipeline mode
                if hasattr(func_obj, '__wrapped__') or hasattr(func_obj, 'opencl_code'):
                    # On injecte directement le device_manager, qui agira comme un Scheduler pour le task
                    real_kwargs["scheduler"] = device_manager
                    real_kwargs["_pipeline_mode"] = True
                    if stream is not None:
                        real_kwargs["stream"] = stream

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
