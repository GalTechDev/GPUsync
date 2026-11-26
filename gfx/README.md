# gpusync-gfx

Moteur graphique et de simulation temps réel écrit avec [GPUsync](../README.md) : rendu 2D et 3D, interface,
fluides (2D, 3D, SPH, surface d'eau), atmosphère et nuages en 3D, météo de grande échelle couplée aux nuages
locaux. Tout le calcul est fait par des kernels GPUsync ; seule dépendance en plus de GPUsync : NumPy.

C'est un paquet séparé de la bibliothèque : `pip install gpusync` donne les kernels, `pip install gpusync-gfx`
ajoute le moteur. Il s'importe sous le même nom qu'avant :

```py
import gpusync as gs
import gpusync.gfx as gfx       # ou gs.gfx
```

Sans `gpusync-gfx` installé, `gs.gfx` lève une `ImportError` qui donne la commande d'installation.

## Développement

Depuis la racine du dépôt, `python3 tools/dev.py setup` installe les deux paquets en mode modifiable, et
`python3 tools/dev.py test` lance aussi les tests du moteur (`gfx/tests`). Les démos sont dans `examples/`
(`gfx_weather.py`, `gfx_clouds.py`…).
