import ast


class BaseTranspiler(ast.NodeVisitor):
    """
    Classe de base pour la transpilation de l'AST Python en code C99/C++ (OpenCL/CUDA).
    Parcourt l'arbre syntaxique abstrait (AST) de Python pour générer le code équivalent.
    """

    def __init__(self, kernel_name, local_mem_qualifier="__local", num_returns=0, backend="opencl", constants=None):
        """
        Initialise le transpilateur de base.

        Args:
            kernel_name (str): Nom du kernel à générer.
            local_mem_qualifier (str): Qualificatif pour la mémoire partagée (ex: __local, __shared__).
            num_returns (int): Nombre de valeurs retournées par le kernel.
            backend (str): Nom du backend ciblé ('opencl' ou 'cuda').
            constants (dict, optional): Constantes de configuration à injecter à la volée.
        """
        self.kernel_name = kernel_name
        self.local_mem_qualifier = local_mem_qualifier
        self.num_returns = num_returns
        self.backend = backend
        self.c_code = []
        self.indent_level = 1
        self.declared_vars = set()
        self.constants = constants or {}

    def add_line(self, line):
        """Ajoute une ligne de code C avec le niveau d'indentation courant."""
        self.c_code.append("    " * self.indent_level + line)

    def visit_Assign(self, node):
        """
        Gère les assignations (ex: x = 1.0 ou local_buf = local_memory(...)).
        Détecte également les allocations de listes statiques et l'opérateur de multiplication matricielle '@'.
        """
        # Détection d'allocations de type: array = [0.0] * size (tableau statique dans les registres)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            if isinstance(node.value.left, ast.List) and len(node.value.left.elts) == 1:
                size = None
                if isinstance(node.value.right, ast.Constant):
                    size = node.value.right.value
                elif isinstance(node.value.right, ast.Name) and node.value.right.id in self.constants:
                    size = self.constants[node.value.right.id]

                if size is not None:
                    target_name = node.targets[0].id
                    if target_name not in self.declared_vars:
                        elt = node.value.left.elts[0]
                        dtype = "float"
                        if isinstance(elt, ast.Call) and getattr(elt.func, "id", "") == "float4":
                            dtype = "float4"
                        self.add_line(f"{dtype} {target_name}[{size}];")
                        self.declared_vars.add(target_name)
                        return

        # Détection et allocation de la mémoire partagée (Shared Memory SRAM)
        if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "local_memory":
            target_name = node.targets[0].id
            shape_tuple = node.value.args[0]
            dims = [self.visit(e) for e in shape_tuple.elts]
            dtype = node.value.args[1].value
            dims_str = "".join([f"[{d}]" for d in dims])
            self.add_line(f"{self.local_mem_qualifier} {dtype} {target_name}{dims_str};")
            self.declared_vars.add(target_name)
            return

        targets = [self.visit(t) for t in node.targets]

        # Détection automatique du produit matriciel (A @ B) pour activation des Tensor Cores
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            self._handle_matmult(targets[0], node.value.left, node.value.right)
            return

        value = self.visit(node.value)

        # Génération du code d'affectation avec déclaration implicite des variables locales
        for t in targets:
            if "[" in t:
                # Écriture dans un tableau existant (déjà déclaré)
                self.add_line(f"{t} = {value};")
                continue

            if t not in self.declared_vars:
                # Déclaration au premier usage : int pour les indices de grille, float par défaut
                if "get_global_id" in value or "blockIdx.x" in value:
                    self.add_line(f"int {t} = {value};")
                else:
                    self.add_line(f"float {t} = {value};")
                self.declared_vars.add(t)
            else:
                self.add_line(f"{t} = {value};")

    def visit_AnnAssign(self, node):
        """
        Gère les assignations typées (ex: x: int = 0).
        Permet de typer explicitement les variables locales pour éviter la déclaration en float par défaut.
        """
        target = self.visit(node.target)
        value = self.visit(node.value) if node.value else "0"

        if target not in self.declared_vars:
            if isinstance(node.annotation, ast.Name):
                type_name = node.annotation.id
                if type_name == "int":
                    self.add_line(f"int {target} = {value};")
                elif type_name == "float4":
                    self.add_line(f"float4 {target} = {value};")
                else:
                    self.add_line(f"float {target} = {value};")
            else:
                self.add_line(f"float {target} = {value};")
            self.declared_vars.add(target)
        else:
            self.add_line(f"{target} = {value};")

    def visit_Tuple(self, node):
        """Traduit un tuple en indices de tableau contigus (ex: [i, j] -> [i][j])."""
        elts = [self.visit(e) for e in node.elts]
        return "][".join(elts)

    def visit_Name(self, node):
        """Traduit un nom de variable ou remplace par sa valeur constante si configuré."""
        if self.constants and node.id in self.constants:
            return str(self.constants[node.id])
        return node.id

    def visit_Attribute(self, node):
        """Traduit l'accès aux attributs d'objets (ex: self.x -> self.x)."""
        value = self.visit(node.value)
        return f"{value}.{node.attr}"

    def visit_Expr(self, node):
        """Gère les expressions isolées (ex: appels de fonctions in-place ou docstrings traduites en commentaires)."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for line in node.value.value.split("\n"):
                if line.strip():
                    self.add_line(f"// {line.strip()}")
            return

        value = self.visit(node.value)
        if value:
            self.add_line(f"{value};")

    def visit_Constant(self, node):
        """Traduit les constantes litérales (nombres, chaînes, booléens)."""
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_BinOp(self, node):
        """Traduit les opérations binaires mathématiques et logiques."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"

        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",
            ast.Mod: "%",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }
        op = op_map[type(node.op)]
        return f"({left} {op} {right})"

    def visit_BoolOp(self, node):
        """Traduit les opérations logiques (and, or) en opérateurs C (&&, ||)."""
        op_map = {ast.And: "&&", ast.Or: "||"}
        op = op_map[type(node.op)]
        values = [self.visit(v) for v in node.values]
        return f"({f' {op} '.join(values)})"

    def visit_UnaryOp(self, node):
        """Traduit les opérateurs unaires (ex: not, -)."""
        op_map = {ast.Not: "!", ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}
        op = op_map[type(node.op)]
        operand = self.visit(node.operand)
        return f"({op}{operand})"

    def visit_Call(self, node):
        """
        Traduit les appels de fonction.
        Délègue les traductions spécifiques au backend via le crochet `resolve_special_call`.
        """
        func_name = getattr(node.func, "id", "")
        if not func_name and isinstance(node.func, ast.Attribute):
            # Support des appels de méthodes ou d'attributs (ex: wmma.fragment)
            func_name = node.func.attr

        args = [self.visit(a) for a in node.args]

        # 1. Tentative de résolution par le backend (OpenCL/CUDA)
        special_translation = self.resolve_special_call(func_name, node, args)
        if special_translation is not None:
            return special_translation

        # 2. Traductions génériques communes
        if func_name in ("float", "int"):
            return f"({func_name})({args[0]})"
        elif func_name == "float4":
            return f"(float4)({', '.join(args)})"

        return f"{func_name}({', '.join(args)})"

    def resolve_special_call(self, func_name, node, args):
        """
        Méthode crochet (hook) à surcharger par les classes filles pour intercepter
        et traduire des appels système ou de fonctions intrinsèques spécifiques au matériel.
        """
        return None

    def visit_For(self, node):
        """
        Traduit les boucles `for i in range(...)`.
        Gère `unrolled_range` pour forcer le déroulement de la boucle par le compilateur GPU.
        """
        target = node.target.id
        func_name = getattr(node.iter.func, "id", "") if isinstance(node.iter, ast.Call) else ""

        if func_name in ("range", "unrolled_range"):
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

            if func_name == "unrolled_range":
                self.add_line("#pragma unroll")

            self.add_line(f"for (int {target} = {start}; {target} < {end}; {target} += {step}) {{")
            self.indent_level += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent_level -= 1
            self.add_line("}")

    def visit_Return(self, node):
        """
        Gère le retour d'un kernel.
        Traduit les valeurs renvoyées en écritures dans les buffers de sortie prédéfinis (`__out_i`).
        """
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
        """Traduit les boucles `while`."""
        test = self.visit(node.test)
        self.add_line(f"while ({test}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.add_line("}")

    def visit_AugAssign(self, node):
        """Traduit les assignations augmentées (ex: x += 1)."""
        target = self.visit(node.target)
        value = self.visit(node.value)
        op_map = {
            ast.Add: "+=",
            ast.Sub: "-=",
            ast.Mult: "*=",
            ast.Div: "/=",
            ast.Mod: "%=",
            ast.BitAnd: "&=",
            ast.BitOr: "|=",
            ast.BitXor: "^=",
            ast.LShift: "<<=",
            ast.RShift: ">>=",
        }
        op = op_map[type(node.op)]
        self.add_line(f"{target} {op} {value};")

    def visit_If(self, node):
        """Traduit les blocs conditionnels `if / elif / else`."""
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
        """Traduit les comparaisons binaires (ex: x < y)."""
        left = self.visit(node.left)
        op_map = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
        op = op_map[type(node.ops[0])]
        right = self.visit(node.comparators[0])
        return f"({left} {op} {right})"

    def visit_Subscript(self, node):
        """Traduit les accès indexés aux listes/tableaux (ex: a[i])."""
        value = self.visit(node.value)
        slice_val = self.visit(node.slice)
        return f"{value}[{slice_val}]"

    def visit_IfExp(self, node):
        """Traduit l'expression conditionnelle ternaire de Python (a if c else b) en C (c ? a : b)."""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"({test} ? {body} : {orelse})"

    def _handle_matmult(self, target, left, right):
        """Méthode abstraite interne devant être définie par les classes dérivées pour le produit matriciel."""
        raise NotImplementedError("Les sous-classes doivent implémenter _handle_matmult")
