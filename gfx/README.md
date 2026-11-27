# gpusync-gfx

Real-time graphics and simulation engine written with [GPUsync](../README.md): 2D and 3D rendering, user interface,
fluids (2D, 3D, SPH, water surface), 3D atmosphere and clouds with rain and snow, large-scale weather coupled to local clouds. All the
computation is done by GPUsync kernels; the only dependency besides GPUsync is NumPy.

It is a separate package from the library: `pip install gpusync` gives the kernels, `pip install gpusync-gfx` adds
the engine. It is imported under the same name as before:

```py
import gpusync as gs
import gpusync.gfx as gfx       # or gs.gfx
```

Without `gpusync-gfx` installed, `gs.gfx` raises an `ImportError` that gives the install command.

## Development

From the repository root, `python3 tools/dev.py setup` installs both packages in editable mode, and
`python3 tools/dev.py test` also runs the engine tests (`gfx/tests`).
