# Fenêtre

`gs.Window` ouvre une fenêtre native (sans Pygame ni Tkinter) et y affiche une image. Une image calculée par un
kernel sur le même GPU que la fenêtre passe **directement du GPU à l'écran**, sans repasser par la mémoire
centrale : de quoi animer une simulation ou un rendu à plein régime depuis Python.

```py
import math
import numpy as np
import gpusync as gs

@gs.kernel
def waves(xy: gs.f32["h", "w"], t: gs.f32) -> gs.u32["h", "w"]:
    y, x = gs.index()
    v = 0.5 + 0.5 * gs.sin(0.05 * float(x) + t) * gs.cos(0.04 * float(y) - 0.7 * t)
    c = gs.u32(gs.min(255.0 * v, 255.0))
    return c | (c << 8) | (gs.u32(255) << 16) | (gs.u32(255) << 24)   # R | G<<8 | B<<16 | A<<24

@gs.kernel
def blank(x: gs.f32[:]) -> gs.f32[:]:
    return 0.0 * x

with gs.Window(960, 540, title="Vagues") as win:
    t, canvas = 0.0, None
    while win.open:
        for ev in win.poll():
            if ev["type"] == "key_down" and ev["key"] == "escape":
                win.close()
        if canvas is None or canvas.shape != (win.height, win.width):   # au départ et si la fenêtre change
            canvas = blank(np.zeros((win.height, win.width), np.float32), device=win.device)
        win.present(waves(canvas, t, device=win.device))   # `canvas` ne sert qu'à donner la taille
        t += 0.05
```

L'image passe sans copie si deux conditions sont réunies (`win.zero_copy` le dit après chaque `present`) :

- elle est calculée **sur le périphérique de la fenêtre** (`device=win.device`) : la fenêtre utilise wgpu, donc
  Metal, Vulkan ou DX12 ;
- elle **reste sur le GPU** (mode `resident`), donc calculée à partir d'entrées qui y sont déjà. Calculée
  directement à partir d'un tableau NumPy sur un GPU à mémoire unifiée, elle serait écrite dans la mémoire
  centrale (mode `direct`) puis recopiée : c'est pour cela que l'exemple passe par `canvas`, un résultat de
  kernel gardé d'une image à l'autre.

## Images acceptées par `present`

| Image | Format |
|---|---|
| `gs.Array` `uint32` `(H, W)` | RGBA empaqueté dans un entier (rouge dans l'octet de poids faible, voir l'exemple) ; sans copie s'il est resté sur le GPU de la fenêtre |
| `np.ndarray` `uint32` `(H, W)` | même empaquetage |
| `np.ndarray` `uint8` `(H, W, 4)` | RGBA |
| `np.ndarray` `uint8` `(H, W, 3)` | RGB |

## Événements

`win.poll()` traite les événements du système sans bloquer et rend une liste de dictionnaires :

| `type` | Autres clés |
|---|---|
| `"key_down"`, `"key_up"` | `key` : lettre en minuscule (`"z"`, `"+"`…) ou `"space"`, `"escape"`, `"enter"`, `"tab"`, `"backspace"`, `"up"`, `"down"`, `"left"`, `"right"`, `"shift"`, `"ctrl"`, `"alt"`, `"super"` |
| `"mouse_down"`, `"mouse_up"` | `button` : `"left"`, `"right"`, `"middle"` |
| `"mouse_move"` | `x`, `y` (position), `dx`, `dy` (déplacement) |
| `"mouse_wheel"` | `dx`, `dy` |
| `"resize"` | `width`, `height` |
| `"close"` | |

État courant, sans passer par les événements :

| | |
|---|---|
| `win.key_down("z")` | touche enfoncée maintenant (pour les déplacements continus) |
| `win.keys`, `win.mouse_buttons` | ensembles des touches et boutons enfoncés |
| `win.mouse_pos` | position de la souris `(x, y)` (un attribut, pas une fonction) |
| `win.size`, `win.width`, `win.height` | taille logique de la zone d'affichage |
| `win.physical_size` | taille en pixels réels (écrans Retina / HiDPI) |
| `win.open` | `False` une fois fermée |

## Réglages

`gs.Window(largeur, hauteur, title=..., vsync=False, resizable=True, device=None)`. `win.vsync` se change à
chaud ; `win.set_title(...)` change le titre (utile pour afficher les images par seconde) ; `win.close()` ferme.

La fenêtre doit être créée et utilisée sur le thread principal (exigence de macOS).
