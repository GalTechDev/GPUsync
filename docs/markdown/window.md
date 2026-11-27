# Window

`gs.Window` opens a native window (no Pygame, no Tkinter) and shows an image in it. An image computed by a kernel
on the same GPU as the window goes **straight from the GPU to the screen**, without going through main memory:
enough to animate a simulation or a renderer at full speed from Python.

```py
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

with gs.Window(960, 540, title="Waves") as win:
    t, canvas = 0.0, None
    while win.open:
        for ev in win.poll():
            if ev["type"] == "key_down" and ev["key"] == "escape":
                win.close()
        if canvas is None or canvas.shape != (win.height, win.width):   # at start and when the window changes
            canvas = blank(np.zeros((win.height, win.width), np.float32), device=win.device)
        win.present(waves(canvas, t, device=win.device))   # `canvas` only gives the size
        t += 0.05
```

The image goes through without a copy when two conditions hold (`win.zero_copy` tells after each `present`):

- it is computed **on the window's device** (`device=win.device`): the window uses wgpu, hence Metal, Vulkan or
  DX12;
- it **stays on the GPU** (`resident` mode), so it is computed from inputs that are already there. Computed
  straight from a NumPy array on a unified-memory GPU, it would be written to main memory (`direct` mode) and
  then copied: that is why the example goes through `canvas`, a kernel result kept from one frame to the next.

## Images accepted by `present`

| Image | Format |
|---|---|
| `gs.Array` `uint32` `(H, W)` | RGBA packed in one integer (red in the least significant byte, see the example); no copy if it stayed on the window's GPU |
| `np.ndarray` `uint32` `(H, W)` | same packing |
| `np.ndarray` `uint8` `(H, W, 4)` | RGBA |
| `np.ndarray` `uint8` `(H, W, 3)` | RGB |

## Events

`win.poll()` processes system events without blocking and returns a list of dictionaries:

| `type` | Other keys |
|---|---|
| `"key_down"`, `"key_up"` | `key`: lowercase character (`"z"`, `"+"`…) or `"space"`, `"escape"`, `"enter"`, `"tab"`, `"backspace"`, `"up"`, `"down"`, `"left"`, `"right"`, `"shift"`, `"ctrl"`, `"alt"`, `"super"` |
| `"mouse_down"`, `"mouse_up"` | `button`: `"left"`, `"right"`, `"middle"` |
| `"mouse_move"` | `x`, `y` (position), `dx`, `dy` (motion) |
| `"mouse_wheel"` | `dx`, `dy` |
| `"resize"` | `width`, `height` |
| `"close"` | |

Current state, without going through events:

| | |
|---|---|
| `win.key_down("z")` | key held right now (for continuous movement) |
| `win.keys`, `win.mouse_buttons` | sets of keys and buttons held |
| `win.mouse_pos` | mouse position `(x, y)` (an attribute, not a function) |
| `win.size`, `win.width`, `win.height` | logical size of the drawing area |
| `win.physical_size` | size in real pixels (Retina / HiDPI screens) |
| `win.open` | `False` once closed |

## Settings

`gs.Window(width, height, title=..., vsync=False, resizable=True, device=None)`. `win.vsync` can be changed on
the fly; `win.set_title(...)` changes the title (handy to show frames per second); `win.close()` closes.

The window must be created and used on the main thread (a macOS requirement).
