import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import asyncio
from gpusync import gpusync_kernel, AsyncBuffer, use_gpu, get_default_scheduler

@gpusync_kernel
def add_one_test(ctx: int, data: list) -> tuple:
    return data[ctx] + 1.0

def test_auto_scheduler():
    """
    Vérifie que l'on peut exécuter un kernel sans configurer de scheduler à la main
    grâce au lazy default scheduler.
    """
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    # Appel transparent sans passer de scheduler !
    res = add_one_test(data, global_size=len(data))
    
    expected = data + 1.0
    assert np.allclose(res, expected)

def test_context_manager():
    """
    Vérifie le bon fonctionnement du context manager use_gpu.
    """
    data = np.array([5.0, 15.0, 25.0], dtype=np.float32)
    
    with use_gpu(backend="opencl") as sched:
        res = add_one_test(data, global_size=len(data), scheduler=sched)
        
    expected = data + 1.0
    assert np.allclose(res, expected)

def test_opencl_memory_pooling():
    """
    Vérifie que la mémoire VRAM (OpenCL) est effectivement recyclée par le pool de buffers
    lorsqu'un AsyncBuffer est détruit par le Garbage Collector de Python.
    """
    # 1. On s'assure d'avoir initialisé le scheduler OpenCL par défaut
    sched = get_default_scheduler()
    
    # Récupérer le manager de périphérique OpenCL
    device_manager = sched.get_best_device()
    if device_manager.backend != "opencl":
        pytest.skip("Ce test cible spécifiquement la plomberie de pooling du backend OpenCL.")

    # Vider le pool pour le test
    device_manager._buffer_pool.clear()
    
    # 2. Allouer un premier buffer
    data1 = np.ones(1000, dtype=np.float32)
    buf1 = AsyncBuffer.to_device(sched, data1)
    
    # Conserver l'identifiant OpenCL physique d'origine
    raw_buffer_id1 = int(buf1.cl_buffer.int_ptr)
    
    # 3. Supprimer le premier buffer
    del buf1
    
    # 4. Allouer un second buffer de même taille
    buf2 = AsyncBuffer.to_device(sched, data1)
    
    # Récupérer l'identifiant du second buffer
    raw_buffer_id2 = int(buf2.cl_buffer.int_ptr)
    
    # 5. Valider que le buffer brut a été recyclé (les adresses virtuelles concordent)
    assert raw_buffer_id1 == raw_buffer_id2, "Le pool de buffers n'a pas recyclé le buffer physique OpenCL !"
    
    # Nettoyage
    del buf2

def test_structured_debug_and_symbolic_bounds():
    """
    Vérifie que la compilation d'un noyau GPUsync engendre l'écriture
    des fichiers de débuggage et l'analyse symbolique des dimensions sur le disque.
    """
    debug_dir = os.path.join(os.getcwd(), ".gpusync_debug")
    debug_cl = os.path.join(debug_dir, "custom_debug_kernel.cl")
    debug_cu = os.path.join(debug_dir, "custom_debug_kernel.cu")
    
    if os.path.exists(debug_cl):
        os.remove(debug_cl)
    if os.path.exists(debug_cu):
        os.remove(debug_cu)

    # Provoquer la JIT-compilation en décorant un nouveau noyau
    @gpusync_kernel
    def custom_debug_kernel(ctx: int, vector_a: list, vector_b: list) -> tuple:
        return vector_a[ctx * 2 + 1] + vector_b[ctx]

    # Vérifier l'existence des fichiers debug
    assert os.path.exists(debug_dir), "Le dossier .gpusync_debug n'existe pas"
    assert os.path.exists(debug_cl), "Le fichier .cl de débuggage n'existe pas"
    assert os.path.exists(debug_cu), "Le fichier .cu de débuggage n'existe pas"

    # Lire le fichier de debug et vérifier la présence de l'analyse symbolique SymPy
    with open(debug_cl, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "ANALYSE SYMBOLIQUE DES DIMENSIONS" in content
    assert "vector_a" in content
    assert "vector_b" in content
