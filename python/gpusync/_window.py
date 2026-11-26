"""`gs.Window`: native GPU window (winit + wgpu), without Pygame or Tkinter.

    import gpusync as gs

    with gs.Window(1024, 576, title="Demo") as win:
        while win.open:
            for ev in win.poll():
                if ev["type"] == "key_down" and ev["key"] == "escape":
                    win.close()
            fb = render_kernel(...)      # gs.Array u32 (H, W): stays on the GPU
            win.present(fb)              # no CPU copy on Metal / Vulkan / DX12!
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._array import Array
from ._device import resolve


class Window:
    """Native GPU window showing a `gs.Array` directly (no copy on the same wgpu device)
    or a NumPy array (`uint32` `(H, W)`, `uint8` `(H, W, 4)` or `uint8` `(H, W, 3)`)."""

    __slots__ = ("_win", "_rgba_scratch")

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "GPUsync",
        *,
        vsync: bool = False,
        resizable: bool = True,
        device=None,
    ) -> None:
        dev_id = None if device is None else resolve(device)
        self._win = _core._Window(int(width), int(height), str(title), bool(vsync), bool(resizable), dev_id)
        self._rgba_scratch: np.ndarray | None = None

    @property
    def open(self) -> bool:
        return self._win.open

    @property
    def device(self) -> str:
        """wgpu device attached to the window (e.g. 'metal:0', 'vulkan:0')."""
        return self._win.device

    @property
    def size(self) -> tuple[int, int]:
        """Logical size `(width, height)` of the drawing area."""
        return self._win.size

    @property
    def physical_size(self) -> tuple[int, int]:
        """Physical size `(width, height)` in real pixels (HiDPI / Retina)."""
        return self._win.physical_size

    @property
    def width(self) -> int:
        return self._win.size[0]

    @property
    def height(self) -> int:
        return self._win.size[1]

    @property
    def vsync(self) -> bool:
        return self._win.vsync

    @vsync.setter
    def vsync(self, value: bool) -> None:
        self._win.vsync = bool(value)

    @property
    def zero_copy(self) -> bool:
        """Did the last presented image go straight from the GPU to the screen, without a CPU copy?"""
        return self._win.zero_copy

    def set_title(self, title: str) -> None:
        self._win.set_title(str(title))

    @property
    def mouse_pos(self) -> tuple[float, float]:
        return self._win.mouse_pos

    @property
    def keys(self) -> set[str]:
        return self._win.keys

    @property
    def mouse_buttons(self) -> set[str]:
        return self._win.mouse_buttons

    def key_down(self, key: str) -> bool:
        return self._win.key_down(key)

    def poll(self) -> list[dict]:
        """Pumps system events (keyboard, mouse, resize, close) without blocking."""
        return self._win.poll()

    def present(self, frame) -> None:
        """Shows an image:
        - `gs.Array` of type `uint32` and shape `(H, W)` (RGBA8 packed `R | G<<8 | B<<16 | A<<24`):
          **no CPU copy** if it lives on the same wgpu GPU as the window;
        - `np.ndarray` of shape `(H, W)` `uint32`, `(H, W, 4)` `uint8` (RGBA) or `(H, W, 3)` `uint8` (RGB).
        """
        if isinstance(frame, Array):
            if frame.ndim != 2 or frame.dtype != np.uint32:
                raise _core.TypeMismatch(
                    f"gs.Window.present: a gs.Array must be 2-D uint32 (H, W) packed RGBA, "
                    f"got {frame.dtype} {frame.shape}"
                )
            h, w = frame.shape
            if frame._handle is not None and not frame._ready:
                self._win.present_array(frame._handle, w, h)
                if frame._check is not None:
                    frame._check()
                return
            host = np.ascontiguousarray(frame.numpy())
            self._win.present_host(host.ctypes.data, host.nbytes, w, h)
            return

        arr = np.asarray(frame)
        if arr.ndim == 2 and arr.dtype == np.uint32:
            h, w = arr.shape
            c = np.ascontiguousarray(arr)
            self._win.present_host(c.ctypes.data, c.nbytes, w, h)
        elif arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
            h, w, _ = arr.shape
            c = np.ascontiguousarray(arr)
            self._win.present_host(c.ctypes.data, c.nbytes, w, h)
        elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
            h, w, _ = arr.shape
            if self._rgba_scratch is None or self._rgba_scratch.shape != (h, w, 4):
                self._rgba_scratch = np.empty((h, w, 4), dtype=np.uint8)
                self._rgba_scratch[:, :, 3] = 255
            self._rgba_scratch[:, :, :3] = arr
            self._win.present_host(self._rgba_scratch.ctypes.data, self._rgba_scratch.nbytes, w, h)
        else:
            raise _core.TypeMismatch(
                f"gs.Window.present: expected a gs.Array uint32 (H, W) or an ndarray uint32 (H, W) / "
                f"uint8 (H, W, 4) / uint8 (H, W, 3), got {arr.dtype} {arr.shape}"
            )

    def close(self) -> None:
        self._win.close()

    def __enter__(self) -> "Window":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
