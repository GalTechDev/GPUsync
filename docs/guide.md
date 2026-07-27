# Guide d'Utilisation

Ce guide détaille le fonctionnement interne et les bonnes pratiques pour exploiter pleinement les fonctionnalités avancées de **GPUsync**.

---

## 🏎️ Memory Pooling VRAM

L'allocation et la libération de buffers de VRAM sont des opérations très coûteuses dans les drivers graphiques. Pour y remédier, GPUsync intègre un gestionnaire de cache interne.

### Fonctionnement automatique

Dès qu'un objet `AsyncBuffer` n'est plus référencé par votre code Python, le Garbage Collector l'élimine, déclenchant son destructeur. Au lieu de désallouer la mémoire physique sur le périphérique, le buffer brut est conservé dans le pool interne du manager de périphérique.

```python
# Premier appel : allocation physique lente sur le GPU
buf_a = AsyncBuffer.to_device(data)

# ... calcul ...

del buf_a  # Libéré par Python, renvoyé dans le pool

# Deuxième appel : réutilisation instantanée du buffer en VRAM (coût = 0 ms)
buf_b = AsyncBuffer.empty_like(data)
```

---

## 🛠️ Auto-Tuning Matériel

Calculer la géométrie optimale des blocs d'exécution (Work-Groups / Thread Blocks) est complexe car elle dépend de la carte de l'utilisateur final.

### Résolution de local_size

Si vous ne spécifiez pas `local_size` dans votre décorateur `@gpusync_kernel`, GPUsync le calcule pour vous :

```python
@gpusync_kernel
def my_kernel(a: list):
    # local_size=None par défaut
    ...
```

À l'exécution, GPUsync va :
1. Interroger le périphérique de calcul pour obtenir `MAX_WORK_GROUP_SIZE`.
2. Interroger le compilateur pour obtenir le `PREFERRED_WORK_GROUP_SIZE_MULTIPLE` (la taille physique du Warp ou du Wavefront).
3. Déterminer une répartition optimale en dimension 1D, 2D ou 3D (ex: $(256, 1, 1)$ ou $(16, 16, 1)$) pour maximiser le taux d'occupation de la carte.

---

## 📦 Compilation SPIR-V & Portabilité

SPIR-V est un langage intermédiaire binaire portable défini par le Khronos Group.

### Comment l'activer ?

Ajoutez le paramètre `compile_spirv=True` à votre décorateur :

```python
@gpusync_kernel(compile_spirv=True)
def heavy_computation(data: list):
    ...
```

GPUsync va chercher le compilateur Google `clspv` :
1. D'abord en local dans le sous-dossier `gpusync/bin/` (mode vendored).
2. Ensuite de façon globale dans le `PATH` système.
3. Si absent, un repli transparent est effectué vers le mode source standard OpenCL C.
