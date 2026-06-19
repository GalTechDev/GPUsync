"""
Module factice pour l'auto-complétion Python.
Les fonctions de ce module ne sont jamais exécutées par l'interpréteur Python,
elles sont interceptées par le compilateur AST et traduites en C++ CUDA (nvcuda::wmma).
"""

def fragment(matrix_type: str, m: int, n: int, k: int, precision: str, layout: str = ""):
    """
    Déclare un fragment Tensor Core (wmma::fragment)
    - matrix_type: "matrix_a", "matrix_b", ou "accumulator"
    - m, n, k: Taille de la tuile matérielle (ex: 16, 16, 8 pour TF32)
    - precision: "tf32", "half", ou "float"
    - layout: "row_major" ou "col_major"
    """
    pass

def fill_fragment(frag, value: float):
    """
    Remplit un fragment avec une valeur (wmma::fill_fragment)
    """
    pass

def load_matrix_sync(frag, ptr, leading_dimension: int):
    """
    Charge une tuile depuis la mémoire globale vers le fragment (wmma::load_matrix_sync)
    - ptr: Pointeur ou tranche (ex: A + offset)
    - leading_dimension: Nombre d'éléments séparant deux lignes consécutives
    """
    pass

def mma_sync(d_frag, a_frag, b_frag, c_frag):
    """
    Exécute la multiplication de matrices matérielle (wmma::mma_sync)
    D = A * B + C
    """
    pass

def store_matrix_sync(ptr, frag, leading_dimension: int, layout: str):
    """
    Sauvegarde un fragment dans la mémoire globale (wmma::store_matrix_sync)
    - layout: "mem_row_major" ou "mem_col_major"
    """
    pass
