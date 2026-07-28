class AbsentModule:
    """
    Classe de remplacement pour les modules optionnels non installés.
    Ne plante pas à l'import, mais lève une ImportError explicite 
    uniquement lorsqu'on tente d'accéder à l'un de ses attributs.
    """
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, item):
        raise ImportError(f"Le module '{self.name}' n'est pas installé mais est requis pour cette fonctionnalité.")


# --- Importation de CuPy ---
try:
    import cupy as cp
except ImportError:
    cp = AbsentModule("cupy")

# --- Importation de PyOpenCL ---
try:
    import pyopencl as cl
except ImportError:
    cl = AbsentModule("pyopencl")

# --- Importation de SymPy ---
try:
    import sympy as sp
except ImportError:
    sp = AbsentModule("sympy")
