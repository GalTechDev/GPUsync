"""
Module factice pour l'auto-complétion Python et le typage IDE pour OpenCL.
Les fonctions de ce module ne sont jamais exécutées directement par l'interpréteur Python.
Elles sont interceptées par le compilateur AST et transpilées en expressions et déclarations OpenCL C99.
"""

CLK_LOCAL_MEM_FENCE = "CLK_LOCAL_MEM_FENCE"
"""Drapeau de barrière de synchronisation : applique la barrière sur la mémoire locale (SRAM)."""

CLK_GLOBAL_MEM_FENCE = "CLK_GLOBAL_MEM_FENCE"
"""Drapeau de barrière de synchronisation : applique la barrière sur la mémoire globale (VRAM)."""


def get_local_id(dim: int) -> int:
    """
    Retourne l'identifiant (index unique) du thread au sein de son bloc de threads (Work-Group)
    pour la dimension spécifiée.

    Args:
        dim (int): La dimension (axe) à interroger. 0 pour l'axe X, 1 pour l'axe Y, 2 pour l'axe Z.

    Exemple d'utilisation dans un kernel :
        local_idx = get_local_id(0)
    """
    return 0


def get_group_id(dim: int) -> int:
    """
    Retourne l'identifiant (index unique) du bloc de threads (Work-Group) au sein de la grille d'exécution
    pour la dimension spécifiée.

    Args:
        dim (int): La dimension (axe) à interroger. 0 pour l'axe X, 1 pour l'axe Y, 2 pour l'axe Z.

    Exemple d'utilisation dans un kernel :
        group_idx = get_group_id(0)
    """
    return 0


def get_global_id(dim: int) -> int:
    """
    Retourne l'identifiant (index absolu) global du thread au sein de la grille d'exécution complète
    pour la dimension spécifiée.

    Args:
        dim (int): La dimension (axe) à interroger. 0 pour l'axe X, 1 pour l'axe Y, 2 pour l'axe Z.

    Exemple d'utilisation dans un kernel :
        global_idx = get_global_id(0)
    """
    return 0


def local_memory(shape: tuple, dtype: str):
    """
    Alloue un espace en mémoire partagée locale au bloc de threads (SRAM) partagé entre tous
    les threads d'un même Work-Group.
    Cette allocation est transpilée en déclaration `__local dtype nom[shape[0]][shape[1]]...`.

    Args:
        shape (tuple): Les dimensions du tableau de mémoire locale (ex: (256,)).
        dtype (str): Le type d'élément sous forme de chaîne de caractères (ex: "int" ou "float").

    Exemple d'utilisation dans un kernel :
        local_cache = local_memory((256,), "float")
    """
    pass


def barrier(flags):
    """
    Force tous les threads du même Work-Group à s'attendre au point d'exécution avant de continuer
    (barrière de synchronisation). Garantit la visibilité de la mémoire selon les drapeaux spécifiés.

    Args:
        flags: Le drapeau spécifiant les types de mémoire à synchroniser (ex: CLK_LOCAL_MEM_FENCE).

    Exemple d'utilisation dans un kernel :
        barrier(CLK_LOCAL_MEM_FENCE)
    """
    pass
