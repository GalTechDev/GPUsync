"""
Module factice pour l'auto-complétion Python.
Les fonctions de ce module ne sont jamais exécutées directement par l'interpréteur Python.
Elles sont interceptées par le compilateur AST et transpilées en instructions matérielles CUDA C++ de Tensor Cores (nvcuda::wmma).
"""


def fragment(matrix_type: str, m: int, n: int, k: int, precision: str, layout: str = ""):
    """
    Déclare un fragment Tensor Core (nvcuda::wmma::fragment) stocké dans les registres du Warp.
    Un fragment est une sous-matrice coopérative partagée entre les 32 threads du Warp.

    Args:
        matrix_type (str): "matrix_a" (première entrée), "matrix_b" (deuxième entrée), ou "accumulator" (accumulateur C/D).
        m (int): Dimension M de la forme de la tuile (ex: 16).
        n (int): Dimension N de la forme de la tuile (ex: 16).
        k (int): Dimension K de la forme de la tuile (ex: 16 ou 8).
        precision (str): Précision matérielle, ex: "half" (FP16), "tf32" (TF32), "float" (FP32).
        layout (str, optional): Obligatoire pour matrix_a et matrix_b : "row_major" (lignes contiguës) ou "col_major" (colonnes contiguës).
                                Inutile pour "accumulator".

    Exemple d'utilisation :
        a_frag = wmma.fragment("matrix_a", 16, 16, 16, "half", "row_major")
        c_frag = wmma.fragment("accumulator", 16, 16, 16, "float")
    """
    pass


def fill_fragment(frag, value: float):
    """
    Remplit tous les éléments d'un fragment avec une valeur scalaire constante (nvcuda::wmma::fill_fragment).
    Généralement utilisé pour initialiser l'accumulateur à 0.0.

    Args:
        frag: L'instance de fragment à initialiser (généralement de type "accumulator").
        value (float): Valeur d'initialisation.

    Exemple d'utilisation :
        wmma.fill_fragment(c_frag, 0.0)
    """
    pass


def load_matrix_sync(frag, ptr, leading_dimension: int):
    """
    Charge de manière coopérative et synchrone une tuile mémoire depuis la mémoire globale (VRAM)
    ou la mémoire partagée (SRAM) vers un fragment (nvcuda::wmma::load_matrix_sync).

    Args:
        frag: Le fragment cible (matrix_a, matrix_b ou accumulator).
        ptr (list/pointer): Pointeur ou index de départ en mémoire (ex: A[row * K + col]).
        leading_dimension (int): La dimension principale en mémoire (nombre d'éléments séparant deux lignes/colonnes consécutives).

    Exemple d'utilisation :
        wmma.load_matrix_sync(a_frag, A[offset_A], K)
    """
    pass


def mma_sync(d_frag, a_frag, b_frag, c_frag):
    """
    Exécute l'opération de multiplication-accumulation matricielle matérielle (nvcuda::wmma::mma_sync)
    sur les Tensor Cores.
    Calcule : D = A * B + C

    Args:
        d_frag: Le fragment accumulateur cible (output).
        a_frag: Le fragment matrix_a contenant la sous-matrice A.
        b_frag: Le fragment matrix_b contenant la sous-matrice B.
        c_frag: Le fragment accumulateur source contenant la sous-matrice C.

    Exemple d'utilisation :
        wmma.mma_sync(c_frag, a_frag, b_frag, c_frag)
    """
    pass


def store_matrix_sync(ptr, frag, leading_dimension: int, layout: str):
    """
    Sauvegarde coopérative et synchrone du contenu d'un fragment accumulateur vers la mémoire globale (VRAM)
    ou la mémoire partagée (SRAM) (nvcuda::wmma::store_matrix_sync).

    Args:
        ptr (list/pointer): Pointeur ou index de départ en mémoire (ex: C[row * N + col]).
        frag: Le fragment accumulateur à sauvegarder.
        leading_dimension (int): La dimension principale en mémoire (nombre d'éléments séparant deux lignes/colonnes consécutives).
        layout (str): Disposition en mémoire cible : "mem_row_major" ou "mem_col_major".

    Exemple d'utilisation :
        wmma.store_matrix_sync(C[offset_C], c_frag, N, "mem_row_major")
    """
    pass
