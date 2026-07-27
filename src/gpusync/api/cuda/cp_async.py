"""
Module factice pour l'auto-complétion Python.
Les fonctions de ce module ne sont jamais exécutées directement par l'interpréteur Python.
Elles sont interceptées par le compilateur AST et transpilées en instructions matérielles CUDA C++ de copie asynchrone (cp.async).
"""


def copy_16(shared_dst, global_src):
    """
    Copie asynchrone matérielle de 16 octets (128 bits) depuis la mémoire globale
    directement vers la mémoire partagée (SRAM) sans passer par les registres CPU/GPU.
    Tire parti des instructions matérielles d'accélération cp.async (architectures Ampere+).

    Args:
        shared_dst (list/pointer): Adresse cible en mémoire partagée (SRAM) locale au bloc.
        global_src (list/pointer): Adresse source en mémoire globale (VRAM).

    Exemple d'utilisation dans un kernel :
        cp_async.copy_16(shared_cache[local_idx * 4], global_data[global_offset])
    """
    pass


def commit():
    """
    Valide et regroupe (commit) toutes les requêtes de transfert cp_async lancées précédemment
    dans un groupe de transfert asynchrone (Async Copy Group).

    Exemple d'utilisation dans un kernel :
        cp_async.commit()
    """
    pass


def wait_0():
    """
    Bloque l'exécution des threads du bloc jusqu'à ce que tous les transferts asynchrones
    commités (commit) soient entièrement terminés.
    Le suffixe _0 indique que l'on attend la fin de l'intégralité des groupes en attente (0 groupes restants).

    Exemple d'utilisation dans un kernel :
        cp_async.wait_0()
    """
    pass
