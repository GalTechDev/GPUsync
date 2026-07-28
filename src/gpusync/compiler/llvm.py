import ast
import logging
# pyrefly: ignore [missing-import]
import llvmlite.ir as ir

logger = logging.getLogger(__name__)

class LLVMIRTranspiler(ast.NodeVisitor):
    """
    Transpilateur d'AST Python vers LLVM IR (Représentation Intermédiaire)
    en utilisant llvmlite.ir pour le backend JIT natif.
    """
    def __init__(self, kernel_name, num_returns=0, constants=None):
        self.kernel_name = kernel_name
        self.num_returns = num_returns
        self.constants = constants or {}
        self.module = ir.Module(name=kernel_name)
        self.func = None
        self.builder = None
        
        self.symbols = {} # name -> (ir_value, is_pointer)
        self.args_map = {} # name -> ir_argument
        
        self.f32 = ir.FloatType()
        self.i32 = ir.IntType(32)
        self.f32_ptr = self.f32.as_pointer()
        self.loop_counter = 0

    def visit_FunctionDef(self, node):
        arg_types = [self.i32] # ctx
        arg_names = ["ctx"]
        
        # Parse args
        for arg in node.args.args:
            if arg.arg == "ctx": continue
            arg_names.append(arg.arg)
            ann = getattr(arg.annotation, 'id', '') if hasattr(arg, 'annotation') else ''
            if ann == 'int':
                arg_types.append(self.i32)
            elif ann == 'float':
                arg_types.append(self.f32)
            else:
                arg_types.append(self.f32_ptr)
                
        # Add output pointers
        for i in range(self.num_returns):
            arg_names.append(f"__out_{i}")
            arg_types.append(self.f32_ptr)
            
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        self.func = ir.Function(self.module, func_type, name=self.kernel_name)
        
        for arg_val, name in zip(self.func.args, arg_names):
            arg_val.name = name
            self.args_map[name] = arg_val
            
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        
        # Allocate local variables for arguments to allow reassignment
        for name, arg_val in self.args_map.items():
            ptr = self.builder.alloca(arg_val.type, name=name + "_ptr")
            self.builder.store(arg_val, ptr)
            self.symbols[name] = (ptr, arg_val.type == self.f32_ptr)
            
        for stmt in node.body:
            self.visit(stmt)
            
        if not self.builder.block.is_terminated:
            self.builder.ret_void()
            
        self.generate_launch_wrapper(self.func)

    def generate_launch_wrapper(self, kernel_func):
        arg_types = [self.i32, self.i32]
        arg_names = ["start", "end"]
        
        for i, arg in enumerate(kernel_func.args):
            if i == 0: continue
            arg_types.append(arg.type)
            arg_names.append(arg.name)
            
        wrapper_type = ir.FunctionType(ir.VoidType(), arg_types)
        wrapper = ir.Function(self.module, wrapper_type, name=f"launch_{self.kernel_name}")
        
        for arg, name in zip(wrapper.args, arg_names):
            arg.name = name
            
        block = wrapper.append_basic_block("entry")
        builder = ir.IRBuilder(block)
        
        start_val = wrapper.args[0]
        end_val = wrapper.args[1]
        
        ctx_ptr = builder.alloca(self.i32, name="ctx_ptr")
        builder.store(start_val, ctx_ptr)
        
        loop_cond = wrapper.append_basic_block("loop_cond")
        loop_body = wrapper.append_basic_block("loop_body")
        loop_end = wrapper.append_basic_block("loop_end")
        
        builder.branch(loop_cond)
        
        builder.position_at_end(loop_cond)
        curr_ctx = builder.load(ctx_ptr)
        cmp_res = builder.icmp_signed("<", curr_ctx, end_val)
        builder.cbranch(cmp_res, loop_body, loop_end)
        
        builder.position_at_end(loop_body)
        call_args = [curr_ctx] + list(wrapper.args)[2:]
        builder.call(kernel_func, call_args)
        
        next_ctx = builder.add(curr_ctx, ir.Constant(self.i32, 1))
        builder.store(next_ctx, ctx_ptr)
        builder.branch(loop_cond)
        
        builder.position_at_end(loop_end)
        builder.ret_void()

    def get_symbol_ptr(self, name, default_type=None):
        if name in self.constants:
            val = self.constants[name]
            # Constants should be evaluated at point of use, but if assigned, we allocate
        if name not in self.symbols:
            typ = default_type or self.f32
            ptr = self.builder.alloca(typ, name=name)
            self.symbols[name] = (ptr, False)
        return self.symbols[name][0]

    def visit_Name(self, node):
        if node.id in self.constants:
            val = self.constants[node.id]
            return ir.Constant(self.i32 if isinstance(val, int) else self.f32, val)
        if node.id == "get_global_id" or node.id == "get_local_id":
            # For CPU backend, just return ctx (assumes it's called as get_global_id(0))
            return self.builder.load(self.symbols["ctx"][0])
            
        ptr, is_array = self.symbols[node.id]
        if is_array:
            return self.builder.load(ptr)
        else:
            return self.builder.load(ptr)

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return ir.Constant(self.i32, node.value)
        return ir.Constant(self.f32, float(node.value))

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Basic auto-casting
        if left.type == self.i32 and right.type == self.f32:
            left = self.builder.sitofp(left, self.f32)
        elif right.type == self.i32 and left.type == self.f32:
            right = self.builder.sitofp(right, self.f32)
            
        is_f32 = left.type == self.f32
        
        if isinstance(node.op, ast.Add):
            return self.builder.fadd(left, right) if is_f32 else self.builder.add(left, right)
        elif isinstance(node.op, ast.Sub):
            return self.builder.fsub(left, right) if is_f32 else self.builder.sub(left, right)
        elif isinstance(node.op, ast.Mult):
            return self.builder.fmul(left, right) if is_f32 else self.builder.mul(left, right)
        elif isinstance(node.op, ast.Div) or isinstance(node.op, ast.FloorDiv):
            return self.builder.fdiv(left, right) if is_f32 else self.builder.sdiv(left, right)
        elif isinstance(node.op, ast.Mod):
            return self.builder.frem(left, right) if is_f32 else self.builder.srem(left, right)
        raise NotImplementedError(f"BinOp {type(node.op)} non supporté en LLVM IR")

    def visit_Subscript(self, node):
        array_ptr = self.visit(node.value)
        idx = self.visit(node.slice)
        if idx.type == self.f32:
            idx = self.builder.fptosi(idx, self.i32)
        gep = self.builder.gep(array_ptr, [idx], inbounds=True)
        if isinstance(node.ctx, ast.Load):
            return self.builder.load(gep)
        return gep

    def visit_Assign(self, node):
        val = self.visit(node.value)
        target = node.targets[0]
        
        if isinstance(target, ast.Name):
            ptr = self.get_symbol_ptr(target.id, val.type)
            if ptr.type.pointee == self.f32 and val.type == self.i32:
                val = self.builder.sitofp(val, self.f32)
            self.builder.store(val, ptr)
        elif isinstance(target, ast.Subscript):
            target.ctx = ast.Store()
            gep = self.visit(target)
            if val.type == self.i32:
                val = self.builder.sitofp(val, self.f32)
            self.builder.store(val, gep)

    def visit_AugAssign(self, node):
        target_load = ast.Name(id=node.target.id, ctx=ast.Load()) if isinstance(node.target, ast.Name) else node.target
        binop = ast.BinOp(left=target_load, op=node.op, right=node.value)
        assign = ast.Assign(targets=[node.target], value=binop)
        self.visit(assign)

    def visit_Return(self, node):
        if node.value is None:
            self.builder.ret_void()
            return
            
        ctx_val = self.builder.load(self.symbols['ctx'][0])
        
        if isinstance(node.value, ast.Tuple):
            for i, elt in enumerate(node.value.elts):
                val = self.visit(elt)
                out_ptr = self.builder.load(self.symbols[f'__out_{i}'][0])
                gep = self.builder.gep(out_ptr, [ctx_val], inbounds=True)
                if val.type == self.i32: val = self.builder.sitofp(val, self.f32)
                self.builder.store(val, gep)
        else:
            val = self.visit(node.value)
            out_ptr = self.builder.load(self.symbols['__out_0'][0])
            gep = self.builder.gep(out_ptr, [ctx_val], inbounds=True)
            if val.type == self.i32: val = self.builder.sitofp(val, self.f32)
            self.builder.store(val, gep)
            
        self.builder.ret_void()

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        
        if left.type == self.i32 and right.type == self.f32:
            left = self.builder.sitofp(left, self.f32)
        elif right.type == self.i32 and left.type == self.f32:
            right = self.builder.sitofp(right, self.f32)
            
        is_f32 = left.type == self.f32
        
        cmp_op = ""
        if isinstance(op, ast.Lt): cmp_op = "<"
        elif isinstance(op, ast.Gt): cmp_op = ">"
        elif isinstance(op, ast.LtE): cmp_op = "<="
        elif isinstance(op, ast.GtE): cmp_op = ">="
        elif isinstance(op, ast.Eq): cmp_op = "=="
        elif isinstance(op, ast.NotEq): cmp_op = "!="
        
        if is_f32:
            return self.builder.fcmp_ordered(cmp_op, left, right)
        else:
            return self.builder.icmp_signed(cmp_op, left, right)

    def visit_If(self, node):
        test = self.visit(node.test)
        with self.builder.if_else(test) as (then, otherwise):
            with then:
                for stmt in node.body:
                    self.visit(stmt)
            with otherwise:
                for stmt in node.orelse:
                    self.visit(stmt)

    def visit_For(self, node):
        # Support uniquement pour `for i in range(x)`
        self.loop_counter += 1
        target_name = node.target.id
        
        start_val = ir.Constant(self.i32, 0)
        end_val = None
        step_val = ir.Constant(self.i32, 1)
        
        if isinstance(node.iter, ast.Call):
            args = [self.visit(a) for a in node.iter.args]
            if len(args) == 1:
                end_val = args[0]
            elif len(args) == 2:
                start_val = args[0]
                end_val = args[1]
            elif len(args) == 3:
                start_val = args[0]
                end_val = args[1]
                step_val = args[2]
                
        if end_val.type == self.f32:
            end_val = self.builder.fptosi(end_val, self.i32)
            
        target_ptr = self.get_symbol_ptr(target_name, self.i32)
        self.builder.store(start_val, target_ptr)
        
        loop_cond = self.func.append_basic_block(name=f"loop_cond_{self.loop_counter}")
        loop_body = self.func.append_basic_block(name=f"loop_body_{self.loop_counter}")
        loop_end = self.func.append_basic_block(name=f"loop_end_{self.loop_counter}")
        
        self.builder.branch(loop_cond)
        
        # Condition block
        self.builder.position_at_end(loop_cond)
        curr_val = self.builder.load(target_ptr)
        cmp_res = self.builder.icmp_signed("<", curr_val, end_val)
        self.builder.cbranch(cmp_res, loop_body, loop_end)
        
        # Body block
        self.builder.position_at_end(loop_body)
        for stmt in node.body:
            self.visit(stmt)
        # Increment
        curr_val = self.builder.load(target_ptr)
        next_val = self.builder.add(curr_val, step_val)
        self.builder.store(next_val, target_ptr)
        self.builder.branch(loop_cond)
        
        # End block
        self.builder.position_at_end(loop_end)

    def visit_Call(self, node):
        func_name = getattr(node.func, "id", "")
        if not func_name and isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            
        if func_name == "get_global_id":
            return self.builder.load(self.symbols["ctx"][0])
            
        args = [self.visit(a) for a in node.args]
        
        if func_name == "float":
            if args[0].type == self.i32:
                return self.builder.sitofp(args[0], self.f32)
            return args[0]
        if func_name == "int":
            if args[0].type == self.f32:
                return self.builder.fptosi(args[0], self.i32)
            return args[0]
            
        # VERY basic intrinsic support for math
        if func_name in ("sin", "cos", "sqrt", "exp", "log"):
            # Declare function if not exists
            f_type = ir.FunctionType(self.f32, [self.f32])
            intr_name = f"llvm.{func_name}.f32"
            if intr_name not in self.module.globals:
                ir.Function(self.module, f_type, name=intr_name)
            intr_func = self.module.globals[intr_name]
            return self.builder.call(intr_func, args)
            
        raise NotImplementedError(f"Call {func_name} non supporté en LLVM IR")
        
    def visit_Expr(self, node):
        self.visit(node.value)


def compile_to_llvm(tree, kernel_name, num_returns=0, constants=None):
    """
    Point d'entrée de transpilation pour le backend LLVM CPU (IR).
    Retourne le texte LLVM IR généré.
    """
    transpiler = LLVMIRTranspiler(kernel_name, num_returns, constants)
    transpiler.visit(tree)
    return str(transpiler.module)
