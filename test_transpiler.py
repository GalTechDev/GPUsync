import ast
import llvmlite.ir as ir
import llvmlite.binding as llvm
import ctypes
import numpy as np
import concurrent.futures

class LLVMIRTranspiler(ast.NodeVisitor):
    def __init__(self, kernel_name, num_returns=0, constants=None):
        self.kernel_name = kernel_name
        self.num_returns = num_returns
        self.constants = constants or {}
        self.module = ir.Module(name=__name__)
        self.func = None
        self.builder = None
        
        self.symbols = {} # name -> (ir_value, is_pointer)
        self.args_map = {} # name -> ir_argument
        
        self.f32 = ir.FloatType()
        self.i32 = ir.IntType(32)
        self.f32_ptr = self.f32.as_pointer()
        
    def visit_FunctionDef(self, node):
        arg_types = [self.i32] # ctx
        arg_names = ["ctx"]
        
        # Parse args
        for arg in node.args.args:
            if arg.arg == "ctx": continue
            arg_names.append(arg.arg)
            # Default to float* for lists/arrays, float for scalars
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

    def get_symbol_ptr(self, name):
        if name in self.constants:
            return ir.Constant(self.f32, self.constants[name]) # Treat as literal
        if name not in self.symbols:
            # Implicit float variable allocation
            ptr = self.builder.alloca(self.f32, name=name)
            self.symbols[name] = (ptr, False)
        return self.symbols[name][0]

    def visit_Name(self, node):
        if node.id in self.constants:
            val = self.constants[node.id]
            return ir.Constant(self.i32 if isinstance(val, int) else self.f32, val)
        ptr, is_array = self.symbols[node.id]
        if is_array:
            return self.builder.load(ptr) # Load the pointer itself
        else:
            return self.builder.load(ptr) # Load the scalar value

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return ir.Constant(self.i32, node.value)
        return ir.Constant(self.f32, float(node.value))

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Auto-cast i32 to f32 if mixed (very basic)
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
        elif isinstance(node.op, ast.Div):
            return self.builder.fdiv(left, right) if is_f32 else self.builder.sdiv(left, right)
        raise NotImplementedError(f"BinOp {type(node.op)} not implemented")

    def visit_Subscript(self, node):
        array_ptr = self.visit(node.value) # Gets the float*
        idx = self.visit(node.slice)
        gep = self.builder.gep(array_ptr, [idx], inbounds=True)
        # If we are loading (not part of assign target), we load it
        if isinstance(node.ctx, ast.Load):
            return self.builder.load(gep)
        else:
            return gep # Return pointer for Store

    def visit_Assign(self, node):
        val = self.visit(node.value)
        target = node.targets[0]
        
        if isinstance(target, ast.Name):
            ptr = self.get_symbol_ptr(target.id)
            # Cast if needed
            if ptr.type.pointee == self.f32 and val.type == self.i32:
                val = self.builder.sitofp(val, self.f32)
            self.builder.store(val, ptr)
        elif isinstance(target, ast.Subscript):
            target.ctx = ast.Store()
            gep = self.visit(target)
            if val.type == self.i32:
                val = self.builder.sitofp(val, self.f32)
            self.builder.store(val, gep)

    def visit_Return(self, node):
        if node.value is None:
            self.builder.ret_void()
            return
            
        ctx_val = self.visit(ast.Name(id='ctx', ctx=ast.Load()))
        
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

def compile_to_llvm(tree, kernel_name, num_returns=0, constants=None):
    transpiler = LLVMIRTranspiler(kernel_name, num_returns, constants)
    transpiler.visit(tree)
    return str(transpiler.module)

source = """
def add_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx]
"""
tree = ast.parse(source)
llvm_ir = compile_to_llvm(tree.body[0], 'add_kernel', 1)

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

mod = llvm.parse_assembly(llvm_ir)
mod.verify()

# Optional: Run optimization passes

engine.add_module(mod)
engine.finalize_object()
engine.run_static_constructors()

func_ptr = engine.get_function_address("add_kernel")
cfunc = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))(func_ptr)

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
out = np.zeros(4, dtype=np.float32)

a_p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
b_p = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def run_ctx(ctx):
    cfunc(ctx, a_p, b_p, out_p)

with concurrent.futures.ThreadPoolExecutor() as ex:
    ex.map(run_ctx, range(4))

print("LLVM JIT Result:", out)
