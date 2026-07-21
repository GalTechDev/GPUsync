import ast
from .base import BaseTranspiler

class LLVMTranspiler(BaseTranspiler):
    """
    Transpilateur d'AST Python spécifique pour le backend LLVM CPU.
    Hérite de BaseTranspiler et génère la signature C/LLVM IR pour l'exécution JIT CPU.
    """

    def resolve_special_call(self, func_name, node, args):
        """
        Résout les fonctions intrinsèques mathématiques et de synchronisation pour le backend LLVM CPU.
        """
        if func_name == 'min':
            return f"fminf({args[0]}, {args[1]})"
        elif func_name == 'max':
            return f"fmaxf({args[0]}, {args[1]})"
        elif func_name == 'abs' or func_name == 'fabs':
            return f"fabsf({args[0]})"
        elif func_name == 'sqrt':
            return f"sqrtf({args[0]})"
        elif func_name == 'get_global_id':
            # Sur CPU JIT LLVM, ctx est transmis directement dans la boucle d'exécution par thread
            return "ctx"
        elif func_name == 'get_local_id':
            return "ctx"
        elif func_name == 'barrier':
            return "/* CPU Barrier */"
        elif func_name == 'atomic_add':
            return f"{args[0]} += {args[1]}"
        return None

    def visit_FunctionDef(self, node):
        """
        Transpile la définition de fonction Python en signature de fonction C/LLVM pour le CPU.
        """
        args_c = ["int ctx"]
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
                    self.declared_vars.add(arg.arg)
            else:
                args_c.append(f"float* {arg.arg}")
                self.declared_vars.add(arg.arg)
            
        for i in range(self.num_returns):
            args_c.append(f"float* __out_{i}")
            self.declared_vars.add(f"__out_{i}")

        args_str = ", ".join(args_c)
        self.c_code.append(f"void {self.kernel_name}({args_str}) {{")
        
        # Visite des instructions à l'intérieur du corps du kernel
        for stmt in node.body:
            self.visit(stmt)
            
        self.c_code.append("}")

def compile_to_llvm(tree, kernel_name, num_returns=0, constants=None):
    """
    Point d'entrée de transpilation pour le backend LLVM CPU.
    """
    transpiler = LLVMTranspiler(
        kernel_name=kernel_name,
        local_mem_qualifier="",
        num_returns=num_returns,
        backend="llvm",
        constants=constants
    )
    transpiler.visit(tree)
    return "\n".join(transpiler.c_code)
