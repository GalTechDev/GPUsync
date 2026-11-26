# Documentation de GPUsync

GPUsync fait tourner sur le GPU des fonctions Python ordinaires : on écrit le calcul d'une case du résultat, la
bibliothèque le traduit pour CUDA, Metal, Vulkan, DX12 ou OpenCL, choisit le périphérique, déplace les données
et découpe les gros calculs. Le même code tourne sur un Mac, un PC avec une carte NVIDIA ou un portable à GPU
intégré.

```py
import numpy as np
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

x = np.random.rand(1_000_000).astype(np.float32)
y = normalize(x, x.mean(), x.std())     # calcul lancé sur le GPU
print(y[:5])                            # attend la fin et rapatrie
```

Cette documentation porte sur la bibliothèque seule. Le moteur graphique `gpusync.gfx`, construit au-dessus,
est un paquet à part ([gpusync-gfx](../gfx/README.md)) et n'y est pas décrit.

## Sommaire

| Page | Contenu |
|---|---|
| [Installation](installation.md) | compiler, vérifier les GPU détectés, CUDA, variables d'environnement |
| [Écrire un kernel](kernels.md) | les trois formes de kernel, les types, ce qu'on peut écrire dedans |
| [Résultats et gros volumes](resultats.md) | `gs.Array`, chaînage, `await`, `out=`, fichiers, flux, plusieurs kernels en une soumission |
| [Périphériques](peripheriques.md) | choix automatique, choix manuel, repli quand une opération manque, mémoire |
| [Mode explicite et atomiques](atomiques.md) | histogrammes, dispersion : un thread par entrée, écritures `gs.atomic_*` |
| [Groupes de threads](groupes.md) | mémoire partagée, barrières, plusieurs cases par thread, produit de matrices d'octets |
| [Fenêtre](fenetre.md) | `gs.Window` : afficher un résultat du GPU sans repasser par la mémoire centrale |
| [Erreurs, débogage, performances](erreurs.md) | exceptions, code généré, pièges, limites connues |
| [Référence](reference.md) | chaque nom public, en une page |

Pour la conception interne (IR, générateurs de code, moteur de transferts, mesures), voir
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Conventions

- Les exemples écrits en `python` sont exécutés par `tests/test_docs.py` : ils marchent tels quels. Ceux écrits
  en `py` (fenêtre, PyTorch, fichiers de plusieurs gigaoctets) ne le sont pas.
- `gs` désigne `gpusync`, `np` désigne NumPy.
- Le site web de la documentation (`docs/pages`, publié sur GitHub Pages) est généré depuis ces pages par
  `python3 tools/build_pages.py` ; un test vérifie qu'il est à jour.
- Les mesures citées viennent de deux machines : un Mac à puce Apple M5 Pro et un PC à RTX 4080 SUPER.
