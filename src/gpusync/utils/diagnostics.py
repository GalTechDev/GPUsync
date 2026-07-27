import ast
import logging
import os

# pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)


def analyze_symbolic_bounds(tree, kernel_name):
    """
    Analyse l'AST d'un noyau GPUsync pour extraire symboliquement les contraintes de dimensions
    des tableaux d'entrée à l'aide de SymPy.

    Retourne une chaîne de caractères documentant l'analyse symbolique.
    """
    try:
        # pyrefly: ignore [missing-import]
        import sympy as sp
    except ImportError:
        return "SymPy non installé. Analyse symbolique des dimensions ignorée."

    func_def = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    if not func_def:
        return "Aucune définition de fonction trouvée dans l'AST."

    # Identifier le paramètre de contexte (typiquement le premier argument)
    ctx_name = None
    if func_def.args.args:
        ctx_name = func_def.args.args[0].arg

    # Identifier les listes / tableaux d'entrée
    array_params = []
    for arg in func_def.args.args[1:]:
        # Par défaut, on suppose que les listes ou arguments non typés ou annotés comme list/ndarray sont des tableaux
        annotation = getattr(arg.annotation, "id", "") if hasattr(arg, "annotation") else ""
        if annotation in ("list", "ndarray", "np.ndarray") or not annotation:
            array_params.append(arg.arg)

    if not ctx_name:
        return "Aucun paramètre de contexte (ctx) identifié."

    # Création du symbole SymPy pour le contexte d'itération (souvent lié à global_size / thread ID)
    ctx_sym = sp.Symbol(ctx_name, integer=True, nonnegative=True)

    symbolic_report = []
    symbolic_report.append("================================================================================")
    symbolic_report.append(f" ANALYSE SYMBOLIQUE DES DIMENSIONS (SymPy) - Noyau: {kernel_name}")
    symbolic_report.append("================================================================================")
    symbolic_report.append(f"Symbole de contexte d'exécution (Thread ID) : {ctx_sym}")

    constraints_found = False

    # Parcourir l'AST à la recherche des accès par indexation (Subscript) sur les tableaux
    for node in ast.walk(func_def):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            array_name = node.value.id
            if array_name in array_params:
                # Analyse de l'index de subscript
                index_ast = node.slice
                # Convertir l'AST de l'index en expression symbolique SymPy si possible
                expr = _ast_to_sympy_expr(index_ast, ctx_name)
                if expr is not None:
                    constraints_found = True
                    # Taille symbolique du tableau
                    size_sym = sp.Symbol(f"N_{array_name}", integer=True, positive=True)
                    # Contrainte d'accès sécurisé : index < taille_du_tableau
                    inequality = expr < size_sym
                    symbolic_report.append(
                        f"\n* Accès détecté sur '{array_name}' : indexation '{node.value.id}[{sp.pretty(expr)}]'"
                    )
                    symbolic_report.append(f"  -> Contrainte de sécurité VRAM : {sp.pretty(inequality)}")

                    # Tenter de résoudre l'inéquation par rapport au symbole de contexte (ctx_sym)
                    try:
                        solutions = sp.solve_univariate_inequality(inequality, ctx_sym, relational=False)
                        symbolic_report.append(
                            f"  -> Plage d'indices de thread valides (Thread Safety Zone) : {ctx_sym} dans {solutions}"
                        )
                    except Exception:
                        # Si l'équation est non-linéaire ou trop complexe, on affiche la forme simplifiée
                        simplified = sp.simplify(inequality)
                        symbolic_report.append(f"  -> Condition simplifiée : {sp.pretty(simplified)}")

    if not constraints_found:
        symbolic_report.append("\nAucune contrainte symbolique d'accès mémoire n'a été identifiée dans le noyau.")

    symbolic_report.append("================================================================================")
    return "\n".join(symbolic_report)


def _ast_to_sympy_expr(node, ctx_name):
    """
    Traduit récursivement un nœud AST numérique ou arithmétique simple en expression SymPy.
    """
    # pyrefly: ignore [missing-import]
    import sympy as sp

    if isinstance(node, ast.Name):
        return sp.Symbol(node.id, integer=True)
    elif isinstance(node, ast.Constant):
        return sp.Integer(node.value)
    elif isinstance(node, ast.BinOp):
        left = _ast_to_sympy_expr(node.left, ctx_name)
        right = _ast_to_sympy_expr(node.right, ctx_name)
        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.FloorDiv):
            return left // right
        elif isinstance(node.op, ast.Mod):
            return left % right
    return None


def write_structured_debug_code(kernel_name, opencl_code, cuda_code, llvm_code, ptx_code, sympy_report):
    """
    Écrit le code source généré pour chaque backend dans un dossier .gpusync_debug/
    à des fins d'analyse, d'inspection, de profilage et de débuggage.
    """
    debug_dir = os.path.join(os.getcwd(), ".gpusync_debug")
    try:
        os.makedirs(debug_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Impossible de créer le dossier de débuggage {debug_dir}: {e}")
        return

    # Formater l'analyse symbolique sous forme de commentaire pour chaque langage
    c_comment = f"/*\n{sympy_report}\n*/\n\n"
    asm_comment = f";\n; {sympy_report.replace(chr(10), chr(10) + '; ')}\n;\n\n"

    # 1. Écriture OpenCL
    try:
        opencl_file = os.path.join(debug_dir, f"{kernel_name}.cl")
        with open(opencl_file, "w", encoding="utf-8") as f:
            f.write(c_comment)
            f.write(opencl_code)
    except Exception as e:
        logger.error(f"Erreur d'écriture OpenCL debug : {e}")

    # 2. Écriture CUDA
    try:
        cuda_file = os.path.join(debug_dir, f"{kernel_name}.cu")
        with open(cuda_file, "w", encoding="utf-8") as f:
            f.write(c_comment)
            f.write(cuda_code)
    except Exception as e:
        logger.error(f"Erreur d'écriture CUDA debug : {e}")

    # 3. Écriture LLVM
    if llvm_code:
        try:
            llvm_file = os.path.join(debug_dir, f"{kernel_name}.ll")
            with open(llvm_file, "w", encoding="utf-8") as f:
                f.write(c_comment)
                f.write(llvm_code)
        except Exception as e:
            logger.error(f"Erreur d'écriture LLVM debug : {e}")

    # 4. Écriture PTX
    if ptx_code:
        try:
            ptx_file = os.path.join(debug_dir, f"{kernel_name}.ptx")
            with open(ptx_file, "w", encoding="utf-8") as f:
                f.write(asm_comment)
                f.write(ptx_code)
        except Exception as e:
            logger.error(f"Erreur d'écriture PTX debug : {e}")

    logger.info(f"Fichiers de débuggage structurés générés avec succès dans {debug_dir}")
