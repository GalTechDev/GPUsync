import ast
from .base import BaseTranspiler

class CUDATranspiler(BaseTranspiler):
    """
    Transpilateur d'AST Python spécifique pour CUDA C++.
    Hérite de BaseTranspiler et gère les signatures de fonctions, les fragments WMMA Tensor Cores,
    la copie asynchrone (cp_async), et le mappage dynamique des dimensions (get_global_id, get_local_id, etc.).
    """

    def visit_FunctionDef(self, node):
        """
        Transpile la définition de fonction Python en signature de kernel CUDA.
        Gère le typage des paramètres (ex: int, float, list) et injecte les en-têtes et helpers CUDA.
        """
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
            
        # Déclaration des variables de retour en paramètres de sortie (pointeurs float*)
        for i in range(self.num_returns):
            args_c.append(f"float* __out_{i}")
            
        sig = f"extern \"C\" __global__ void {self.kernel_name}({', '.join(args_c)}) {{"
        self.c_code.append("#include <mma.h>")
        self.c_code.append("using namespace nvcuda;")
        
        # Définition des helpers matériels cp.async pour charger les données asynchrones vers la shared memory (SRAM)
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
        
        # Initialisation implicite du thread index absolu 1D en CUDA
        if has_ctx:
            self.c_code.append("    int ctx = blockIdx.x * blockDim.x + threadIdx.x;")
            self.declared_vars.add("ctx")
            
        for stmt in node.body:
            self.visit(stmt)
        self.c_code.append("}")

    def visit_Assign(self, node):
        """
        Détecte les affectations de fragments WMMA pour les Tensor Cores.
        Exemple : frag = wmma.fragment("matrix_a", 16, 16, 16, "half", "row_major")
        """
        # Détection de l'appel wmma.fragment(...)
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

    # Dictionnaire de traduction pour les fonctions mathématiques de base en CUDA C++
    MATH_FUNCTIONS = {
        'min': 'fminf',
        'max': 'fmaxf',
        'abs': 'fabsf',
    }

    def resolve_special_call(self, func_name, node, args):
        """
        Intercepte et résout les appels spécifiques à CUDA (WMMA, cp_async, mathématiques, threads, barrières, atomiques).
        """
        # 1. API Matérielle WMMA
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'wmma':
            func_name_cuda = f"wmma::{node.func.attr}"
            if node.func.attr == 'store_matrix_sync' and len(node.args) == 4:
                # store_matrix_sync nécessite le layout (ex: wmma::mem_row_major)
                layout = node.args[3].value
                return f"{func_name_cuda}({args[0]}, {args[1]}, {args[2]}, wmma::{layout})"
            return f"{func_name_cuda}({', '.join(args)})"

        # 2. Primitives cp_async (chargement matériel asynchrone global->shared)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'cp_async':
            if node.func.attr == 'copy_16':
                return f"cp_async_16(&{args[0]}, &{args[1]})"
            elif node.func.attr == 'commit':
                return "cp_async_commit()"
            elif node.func.attr == 'wait_0':
                return "cp_async_wait_0()"

        # 3. Fonctions mathématiques de base
        if func_name in self.MATH_FUNCTIONS:
            return f"{self.MATH_FUNCTIONS[func_name]}({', '.join(args)})"

        # 4. Mappage dynamique des dimensions (get_global_id, get_local_id, get_group_id)
        if func_name in ('get_global_id', 'get_local_id', 'get_group_id'):
            arg_val = 0
            if node.args:
                if isinstance(node.args[0], ast.Constant):
                    arg_val = node.args[0].value
                elif isinstance(node.args[0], ast.Name) and node.args[0].id in self.constants:
                    arg_val = self.constants[node.args[0].id]
            dim = 'x' if arg_val == 0 else ('y' if arg_val == 1 else 'z')
            
            if func_name == 'get_global_id':
                return f"(blockIdx.{dim} * blockDim.{dim} + threadIdx.{dim})"
            elif func_name == 'get_local_id':
                return f"threadIdx.{dim}"
            elif func_name == 'get_group_id':
                return f"blockIdx.{dim}"

        # 5. Barrière de synchronisation
        if func_name == 'barrier':
            return "__syncthreads()"
            
        # 6. Opérations atomiques CUDA
        if func_name == 'atomic_add':
            return f"atomicAdd(&{args[0]}, {args[1]})"

        return None

    def _handle_matmult(self, target, left, right):
        """
        Intercepte l'opérateur `@` (produit matriciel) pour générer automatiquement 
        le code d'un produit matriciel par pavage 2D classique.
        """
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


def compile_to_cuda(tree, kernel_name, num_returns=0, constants=None):
    """
    Fonction utilitaire principale pour compiler un AST Python en code source CUDA C++.
    """
    transpiler = CUDATranspiler(kernel_name, local_mem_qualifier="__shared__", num_returns=num_returns, backend="cuda", constants=constants)
    transpiler.visit(tree)
    return '\n'.join(transpiler.c_code)
