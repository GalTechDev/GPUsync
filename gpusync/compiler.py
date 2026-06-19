import inspect
import ast
import textwrap

class BaseTranspiler(ast.NodeVisitor):
    def __init__(self, kernel_name, local_mem_qualifier="__local", num_returns=0, backend="opencl"):
        self.kernel_name = kernel_name
        self.local_mem_qualifier = local_mem_qualifier
        self.num_returns = num_returns
        self.backend = backend
        self.c_code = []
        self.indent_level = 1
        self.declared_vars = set()
        
    def add_line(self, line):
        self.c_code.append("    " * self.indent_level + line)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            if isinstance(node.value.left, ast.List) and len(node.value.left.elts) == 1:
                if isinstance(node.value.right, ast.Constant):
                    size = node.value.right.value
                    target_name = node.targets[0].id
                    if target_name not in self.declared_vars:
                        self.add_line(f"float {target_name}[{size}];")
                        self.declared_vars.add(target_name)
                        return
                        
        # Gestion de l'allocation OpenCL/CUDA local_memory
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'local_memory':
            target_name = node.targets[0].id
            shape_tuple = node.value.args[0]
            dims = [self.visit(e) for e in shape_tuple.elts]
            dtype = node.value.args[1].value
            dims_str = "".join([f"[{d}]" for d in dims])
            self.add_line(f"{self.local_mem_qualifier} {dtype} {target_name}{dims_str};")
            self.declared_vars.add(target_name)
            return

        targets = [self.visit(t) for t in node.targets]
        
        # Tensor Core Detection (A @ B)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            self._handle_matmult(targets[0], node.value.left, node.value.right)
            return

        value = self.visit(node.value)
        
        for t in targets:
            if '[' in t:
                self.add_line(f"{t} = {value};")
                continue
                
            if t not in self.declared_vars:
                if 'get_global_id' in value or 'blockIdx.x' in value:
                    self.add_line(f"int {t} = {value};")
                else:
                    self.add_line(f"float {t} = {value};")
                self.declared_vars.add(t)
            else:
                self.add_line(f"{t} = {value};")

    def visit_AnnAssign(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value) if node.value else "0"
        
        if target not in self.declared_vars:
            if isinstance(node.annotation, ast.Name):
                type_name = node.annotation.id
                if type_name == 'int':
                    self.add_line(f"int {target} = {value};")
                elif type_name == 'float4':
                    self.add_line(f"float4 {target} = {value};")
                else:
                    self.add_line(f"float {target} = {value};")
            else:
                self.add_line(f"float {target} = {value};")
            self.declared_vars.add(target)
        else:
            self.add_line(f"{target} = {value};")

    def visit_Tuple(self, node):
        elts = [self.visit(e) for e in node.elts]
        return "][".join(elts)

    def visit_Name(self, node):
        return node.id
        
    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return f"{value}.{node.attr}"
        
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for line in node.value.value.split('\n'):
                if line.strip():
                    self.add_line(f"// {line.strip()}")
            return
            
        value = self.visit(node.value)
        if value:
            self.add_line(f"{value};")
    def visit_Constant(self, node):
        return str(node.value)
        
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
            
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.FloorDiv: '/',
            ast.Mod: '%', ast.BitAnd: '&', ast.BitOr: '|', ast.BitXor: '^', ast.LShift: '<<', ast.RShift: '>>'
        }
        op = op_map[type(node.op)]
        return f"({left} {op} {right})"
        
    def visit_BoolOp(self, node):
        op_map = {ast.And: '&&', ast.Or: '||'}
        op = op_map[type(node.op)]
        values = [self.visit(v) for v in node.values]
        return f"({f' {op} '.join(values)})"
        
    def visit_UnaryOp(self, node):
        op_map = {ast.Not: '!', ast.USub: '-', ast.UAdd: '+', ast.Invert: '~'}
        op = op_map[type(node.op)]
        operand = self.visit(node.operand)
        return f"({op}{operand})"
        
    def visit_For(self, node):
        target = node.target.id
        if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == 'range':
            args = [self.visit(a) for a in node.iter.args]
            start = "0"
            step = "1"
            if len(args) == 1:
                end = args[0]
            elif len(args) == 2:
                start = args[0]
                end = args[1]
            elif len(args) == 3:
                start = args[0]
                end = args[1]
                step = args[2]
            
            self.add_line(f"for (int {target} = {start}; {target} < {end}; {target} += {step}) {{")
            self.indent_level += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent_level -= 1
            self.add_line("}")

    def visit_Return(self, node):
        if node.value is None:
            self.add_line("return;")
            return
            
        idx_expr = "get_global_id(0)" if self.backend == "opencl" else "(blockIdx.x * blockDim.x + threadIdx.x)"
        
        if isinstance(node.value, ast.Tuple):
            for i, elt in enumerate(node.value.elts):
                val = self.visit(elt)
                self.add_line(f"__out_{i}[{idx_expr}] = {val};")
        else:
            val = self.visit(node.value)
            self.add_line(f"__out_{0}[{idx_expr}] = {val};")
        self.add_line("return;")

    def visit_While(self, node):
        test = self.visit(node.test)
        self.add_line(f"while ({test}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.add_line("}")

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        op_map = {
            ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/=', ast.Mod: '%=',
            ast.BitAnd: '&=', ast.BitOr: '|=', ast.BitXor: '^=', ast.LShift: '<<=', ast.RShift: '>>='
        }
        op = op_map[type(node.op)]
        self.add_line(f"{target} {op} {value};")

    def visit_If(self, node):
        test = self.visit(node.test)
        self.add_line(f"if ({test}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.add_line("}")
        if node.orelse:
            self.add_line("else {")
            self.indent_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent_level -= 1
            self.add_line("}")
            
    def visit_Compare(self, node):
        left = self.visit(node.left)
        op_map = {ast.Lt: '<', ast.Gt: '>', ast.LtE: '<=', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
        op = op_map[type(node.ops[0])]
        right = self.visit(node.comparators[0])
        return f"({left} {op} {right})"

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice_val = self.visit(node.slice)
        return f"{value}[{slice_val}]"

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"({test} ? {body} : {orelse})"


class OpenCLTranspiler(BaseTranspiler):
    def visit_FunctionDef(self, node):
        args_c = []
        has_ctx = False
        for arg in node.args.args:
            if arg.arg == 'ctx':
                has_ctx = True
                continue
                
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                type_name = arg.annotation.id
                if type_name == 'int':
                    args_c.append(f"int {arg.arg}")
                    self.declared_vars.add(arg.arg)
                elif type_name == 'float':
                    args_c.append(f"float {arg.arg}")
                    self.declared_vars.add(arg.arg)
                else:
                    args_c.append(f"__global float* {arg.arg}")
            else:
                args_c.append(f"__global float* {arg.arg}")
            
        for i in range(self.num_returns):
            args_c.append(f"__global float* __out_{i}")
            
        sig = f"__kernel void {self.kernel_name}({', '.join(args_c)}) {{"
        self.c_code.append(sig)
        
        if has_ctx:
            self.c_code.append("    int ctx = get_global_id(0);")
            self.declared_vars.add("ctx")
            
        for stmt in node.body:
            self.visit(stmt)
        self.c_code.append("}")

    def visit_Call(self, node):
        func_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr 
        
        # Interception des fonctions vectorielles OpenCL
        if func_name == 'vload4':
            array_name = self.visit(node.args[0])
            index = self.visit(node.args[1])
            return f"vload4(0, &{array_name}[{index}])"
        elif func_name == 'vstore4':
            value = self.visit(node.args[0])
            array_name = self.visit(node.args[1])
            index = self.visit(node.args[2])
            return f"vstore4({value}, 0, &{array_name}[{index}])"
        elif func_name == 'dot':
            arg1 = self.visit(node.args[0])
            arg2 = self.visit(node.args[1])
            return f"dot({arg1}, {arg2})"
        elif func_name == 'atomic_add':
            array_access = self.visit(node.args[0])
            value = self.visit(node.args[1])
            # OpenCL 1.2 ne supporte pas atomic_add float nativement, mais on génère la syntaxe générique
            return f"atomic_add(&{array_access}, {value})"
            
        args = [self.visit(a) for a in node.args]
        return f"{func_name}({', '.join(args)})"

    def _handle_matmult(self, target, left, right):
        self.add_line(f"// ERREUR: Tensor Cores (@) non supportés en OpenCL.")
        self.add_line(f"{target} = 0;")


class CUDATranspiler(BaseTranspiler):
    def visit_FunctionDef(self, node):
        args_c = []
        has_ctx = False
        for arg in node.args.args:
            if arg.arg == 'ctx':
                has_ctx = True
                continue
                
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                type_name = arg.annotation.id
                if type_name == 'int':
                    args_c.append(f"int {arg.arg}")
                    self.declared_vars.add(arg.arg)
                elif type_name == 'float':
                    args_c.append(f"float {arg.arg}")
                    self.declared_vars.add(arg.arg)
                else:
                    args_c.append(f"float* {arg.arg}")
            else:
                args_c.append(f"float* {arg.arg}")
            
        for i in range(self.num_returns):
            args_c.append(f"float* __out_{i}")
            
        sig = f"extern \"C\" __global__ void {self.kernel_name}({', '.join(args_c)}) {{"
        self.c_code.append("#include <mma.h>")
        self.c_code.append("using namespace nvcuda;")
        
        helpers = """
__device__ __forceinline__ void cp_async_16(float* shared_dst, const float* global_src) {
    unsigned int smem_ptr = __cvta_generic_to_shared(shared_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(global_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;");
}
"""
        self.c_code.append(helpers)
        self.c_code.append(sig)
        
        if has_ctx:
            self.c_code.append("    int ctx = blockIdx.x * blockDim.x + threadIdx.x;")
            self.declared_vars.add("ctx")
            
        for stmt in node.body:
            self.visit(stmt)
        self.c_code.append("}")

    def visit_Assign(self, node):
        # Gestion des fragments WMMA matériels
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'wmma' and node.value.func.attr == 'fragment':
            target = self.visit(node.targets[0])
            matrix_type = node.value.args[0].value
            m = node.value.args[1].value
            n = node.value.args[2].value
            k = node.value.args[3].value
            data_type = node.value.args[4].value
            
            layout_str = ""
            if len(node.value.args) > 5:
                layout = node.value.args[5].value
                layout_str = f", wmma::{layout}"
            
            m_type = f"wmma::{matrix_type}"
            if data_type == "tf32":
                d_type = "wmma::precision::tf32"
            else:
                d_type = data_type
            
            self.add_line(f"wmma::fragment<{m_type}, {m}, {n}, {k}, {d_type}{layout_str}> {target};")
            self.declared_vars.add(target)
            return
        super().visit_Assign(node)

    def visit_Call(self, node):
        # API Matérielle WMMA
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'wmma':
                func_name = f"wmma::{node.func.attr}"
                if node.func.attr == 'store_matrix_sync' and len(node.args) == 4:
                    args = [self.visit(a) for a in node.args[:3]]
                    layout = node.args[3].value
                    args.append(f"wmma::{layout}")
                    return f"{func_name}({', '.join(args)})"
                else:
                    args = [self.visit(a) for a in node.args]
                    return f"{func_name}({', '.join(args)})"
            elif node.func.value.id == 'cp_async':
                if node.func.attr == 'copy_16':
                    dst = self.visit(node.args[0])
                    src = self.visit(node.args[1])
                    return f"cp_async_16(&{dst}, &{src})"
                elif node.func.attr == 'commit':
                    return "cp_async_commit()"
                elif node.func.attr == 'wait_0':
                    return "cp_async_wait_0()"
                
        func_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr 
        if func_name == 'get_global_id':
            # Equivalent CUDA: Block ID * Block Dimension + Thread ID
            if getattr(self, 'backend', 'cuda') == 'cuda':
                # We need to map argument 0 -> x, 1 -> y, 2 -> z
                # But since this AST visitor doesn't have backend info easily, we know CuPy compiler throws if it's get_global_id.
                # Actually, wait. It's currently hardcoded to 'x'.
                # Let's fix this properly for CUDA.
                arg_val = node.args[0].value if hasattr(node.args[0], 'value') else 0
                dim = 'x' if arg_val == 0 else ('y' if arg_val == 1 else 'z')
                return f"(blockIdx.{dim} * blockDim.{dim} + threadIdx.{dim})"
            else:
                args = [self.visit(a) for a in node.args]
                return f"get_global_id({args[0]})"
                
        if func_name == 'get_local_id':
            arg_val = node.args[0].value if hasattr(node.args[0], 'value') else 0
            dim = 'x' if arg_val == 0 else ('y' if arg_val == 1 else 'z')
            return f"threadIdx.{dim}"
            
        if func_name == 'get_group_id':
            arg_val = node.args[0].value if hasattr(node.args[0], 'value') else 0
            dim = 'x' if arg_val == 0 else ('y' if arg_val == 1 else 'z')
            return f"blockIdx.{dim}"

        if func_name == 'barrier':
            return "__syncthreads()"
            
        if func_name == 'atomic_add':
            array_access = self.visit(node.args[0])
            value = self.visit(node.args[1])
            return f"atomicAdd(&{array_access}, {value})"
            
        args = [self.visit(a) for a in node.args]
        return f"{func_name}({', '.join(args)})"

    def _handle_matmult(self, target, left, right):
        left_str = self.visit(left)
        right_str = self.visit(right)
        self.add_line(f"// --- AUTO-DETECTION TENSOR CORE (@) ---")
        self.add_line(f"// L'AST intercepte l'opérateur `A @ B` et lance un Kernel Parallèle 2D ! ")
        self.add_line(f"// (Note : Les variables M, K, N doivent être définies dans la signature du kernel)")
        self.add_line(f"int row = blockIdx.y * blockDim.y + threadIdx.y;")
        self.add_line(f"int col = blockIdx.x * blockDim.x + threadIdx.x;")
        self.add_line(f"if (row < M && col < N) {{")
        self.add_line(f"    float sum = 0.0f;")
        self.add_line(f"    for(int k = 0; k < K; k++) {{")
        self.add_line(f"        sum += {left_str}[row * K + k] * {right_str}[k * N + col];")
        self.add_line(f"    }}")
        self.add_line(f"    {target}[row * N + col] = sum;")
        self.add_line(f"}}")


def compile_to_opencl(tree, kernel_name, num_returns=0):
    transpiler = OpenCLTranspiler(kernel_name, num_returns=num_returns, backend="opencl")
    transpiler.visit(tree)
    return '\n'.join(transpiler.c_code)

def compile_to_cuda(tree, kernel_name, num_returns=0):
    transpiler = CUDATranspiler(kernel_name, local_mem_qualifier="__shared__", num_returns=num_returns, backend="cuda")
    transpiler.visit(tree)
    return '\n'.join(transpiler.c_code)


def gpusync_kernel(func=None, raw_cuda=None):
    """
    Décorateur multi-backend.
    Traduit la fonction Python en kernels OpenCL et CUDA C++.
    Permet l'injection de raw_cuda pour bypasser la transpilation CUDA.
    """
    if func is None:
        def wrapper(f):
            return gpusync_kernel(f, raw_cuda=raw_cuda)
        return wrapper

    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    lines = source.split('\n')
    lines = [l for l in lines if not l.strip().startswith('@')]
    source = '\n'.join(lines)
    
    tree = ast.parse(source)
    func_def = tree.body[0]
    kernel_name = func_def.name
    
    num_returns = 0
    returns = [node for node in ast.walk(func_def) if isinstance(node, ast.Return)]
    if returns:
        ret_node = returns[0]
        if ret_node.value is None:
            num_returns = 0
        elif isinstance(ret_node.value, ast.Tuple):
            num_returns = len(ret_node.value.elts)
        else:
            num_returns = 1
            
    kernel_code_opencl = compile_to_opencl(tree, kernel_name, num_returns)
    
    if raw_cuda:
        kernel_code_cuda = raw_cuda
        kernel_name_cuda = kernel_name
    else:
        kernel_code_cuda = compile_to_cuda(tree, kernel_name, num_returns)
        kernel_name_cuda = kernel_name
    
    from .task import AsyncTask
    from .bg_loop import auto_sync
    
    task = AsyncTask(
        kernel_code_opencl=kernel_code_opencl, kernel_name_opencl=kernel_name,
        kernel_code_cuda=kernel_code_cuda, kernel_name_cuda=kernel_name_cuda
    )
    
    @auto_sync
    async def wrapper(*args, global_size, scheduler, _pipeline_mode=False, local_size=None):
        out_buffers = []
        if num_returns > 0:
            import numpy as np
            from .memory import AsyncBuffer
            
            if isinstance(global_size, int):
                total_size = global_size
            else:
                total_size = np.prod(global_size)
                
            for _ in range(num_returns):
                buf = AsyncBuffer.empty_like(scheduler, np.zeros(total_size, dtype=np.float32))
                out_buffers.append(buf)
                
        new_args = list(args) + out_buffers
        await task.execute(scheduler, global_size, local_size, *new_args)
        
        if num_returns == 0:
            return None
        elif num_returns == 1:
            if _pipeline_mode:
                return out_buffers[0]
            return await out_buffers[0].to_host()
        else:
            if _pipeline_mode:
                return tuple(out_buffers)
            return tuple([await buf.to_host() for buf in out_buffers])
        
    wrapper.opencl_code = kernel_code_opencl 
    wrapper.cuda_code = kernel_code_cuda
    return wrapper
