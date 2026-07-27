import ast

from .base import BaseTranspiler


class OpenCLTranspiler(BaseTranspiler):
    """
    Transpilateur d'AST Python spécifique pour OpenCL C99.
    Hérite de BaseTranspiler et gère les signatures de fonctions, les fonctions vectorielles,
    les barrières de synchronisation et les opérations atomiques en OpenCL.
    """

    def visit_FunctionDef(self, node):
        """
        Transpile la définition de fonction Python en signature de kernel OpenCL.
        Gère le typage des paramètres (ex: int, float, list) et injecte les helpers d'atomic_add_float.
        """
        args_c = []
        has_ctx = False
        for arg in node.args.args:
            if arg.arg == "ctx":
                has_ctx = True
                continue

            if arg.annotation and isinstance(arg.annotation, ast.Name):
                type_name = arg.annotation.id
                if type_name == "int":
                    args_c.append(f"int {arg.arg}")
                    self.declared_vars.add(arg.arg)
                elif type_name == "float":
                    args_c.append(f"float {arg.arg}")
                    self.declared_vars.add(arg.arg)
                elif type_name == "constant_list":
                    # Support de la mémoire constante globale en OpenCL
                    args_c.append(f"__constant float* {arg.arg}")
                    self.declared_vars.add(arg.arg)
                else:
                    args_c.append(f"__global float* {arg.arg}")
            else:
                args_c.append(f"__global float* {arg.arg}")

        # Déclaration des variables de retour en paramètres de sortie __global
        for i in range(self.num_returns):
            args_c.append(f"__global float* __out_{i}")

        # Code d'émulation pour les atomiques float sur OpenCL (qui ne sont pas natifs en OpenCL 1.2)
        helpers = """
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
/* Addition atomique flottante en mémoire locale via boucle de comparaison atomique (atom_cmpxchg) */
inline void atomic_add_float(volatile __local float* p, float operand) {
    volatile __local unsigned int* pi = (volatile __local unsigned int*)p;
    unsigned int old_val, new_val;
    do {
        old_val = *pi;
        new_val = as_uint(as_float(old_val) + operand);
    } while (atom_cmpxchg(pi, old_val, new_val) != old_val);
}
/* Addition atomique flottante en mémoire globale via boucle de comparaison atomique (atom_cmpxchg) */
inline void atomic_add_float_global(volatile __global float* p, float operand) {
    volatile __global unsigned int* pi = (volatile __global unsigned int*)p;
    unsigned int old_val, new_val;
    do {
        old_val = *pi;
        new_val = as_uint(as_float(old_val) + operand);
    } while (atom_cmpxchg(pi, old_val, new_val) != old_val);
}
/* Minimum atomique flottant en mémoire globale */
inline void atomic_min_float_global(volatile __global float* p, float val) {
    volatile __global unsigned int* pi = (volatile __global unsigned int*)p;
    unsigned int old_val, new_val;
    do {
        old_val = *pi;
        float old_float = as_float(old_val);
        float new_float = min(old_float, val);
        new_val = as_uint(new_float);
    } while (atom_cmpxchg(pi, old_val, new_val) != old_val);
}
"""
        self.c_code.append(helpers)
        sig = f"__kernel void {self.kernel_name}({', '.join(args_c)}) {{"
        self.c_code.append(sig)

        # Initialisation implicite du contexte de thread get_global_id(0)
        if has_ctx:
            self.c_code.append("    int ctx = get_global_id(0);")
            self.declared_vars.add("ctx")

        for stmt in node.body:
            self.visit(stmt)
        self.c_code.append("}")

    # Dictionnaire de traduction pour les fonctions mathématiques de base
    MATH_FUNCTIONS = {
        "min": "fmin",
        "max": "fmax",
        "abs": "fabs",
    }

    def resolve_special_call(self, func_name, node, args):
        """
        Intercepte et résout les appels spécifiques à OpenCL (mathématiques, threads, barrières, atomiques).
        """
        # 1. Fonctions mathématiques de base
        if func_name in self.MATH_FUNCTIONS:
            return f"{self.MATH_FUNCTIONS[func_name]}({', '.join(args)})"

        # 2. Fonctions d'accès grille et blocs de threads
        if func_name in ("get_local_id", "get_local_size", "get_global_id", "get_group_id"):
            return f"{func_name}({args[0]})"

        # 3. Barrières
        if func_name == "barrier":
            return "barrier(CLK_LOCAL_MEM_FENCE)"

        # 4. Chargements/Écritures vectoriels et fonctions intrinsèques
        if func_name == "vload4":
            return f"vload4(0, &{args[0]}[{args[1]}])"
        elif func_name == "vstore4":
            return f"vstore4({args[0]}, 0, &{args[1]}[{args[2]}])"
        elif func_name == "dot":
            return f"dot({args[0]}, {args[1]})"

        # 5. Opérations atomiques émulées ou natives
        elif func_name == "atomic_add":
            return f"atomic_add(&{args[0]}, {args[1]})"
        elif func_name == "atomic_add_float":
            return f"atomic_add_float(&{args[0]}, {args[1]})"
        elif func_name == "atomic_add_float_global":
            return f"atomic_add_float_global(&{args[0]}, {args[1]})"
        elif func_name == "atomic_min_float_global":
            return f"atomic_min_float_global(&{args[0]}, {args[1]})"

        return None

    def _handle_matmult(self, target, left, right):
        """Les Tensor Cores ne sont pas supportés nativement de manière universelle en OpenCL."""
        # TODO Ajouter les Matrix Cores en OpenCL pour les cartes AMD et XMX pour Intel
        self.add_line("// ERREUR: Opération(@) non supportée en OpenCL.")
        self.add_line(f"{target} = 0;")


def compile_to_opencl(tree, kernel_name, num_returns=0, constants=None):
    """
    Fonction utilitaire principale pour compiler un AST Python en code source OpenCL C.
    """
    transpiler = OpenCLTranspiler(kernel_name, num_returns=num_returns, backend="opencl", constants=constants)
    transpiler.visit(tree)
    return "\n".join(transpiler.c_code)
