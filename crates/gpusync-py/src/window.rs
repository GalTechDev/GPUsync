//! PyO3 binding of the native GPU window (`_core._Window` → `gs.Window`).

use gpusync_core::{Window, WindowEvent};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};

use crate::kernel::{DeviceArrayPy, to_py};

/// Native winit + wgpu window (not `Send`: must be used from the main thread).
#[pyclass(name = "_Window", module = "gpusync._core", unsendable)]
pub struct PyWindow {
    inner: Window,
}

#[pymethods]
impl PyWindow {
    #[new]
    #[pyo3(signature = (width, height, title = "GPUsync", vsync = false, resizable = true, device = None))]
    fn new(
        width: u32,
        height: u32,
        title: &str,
        vsync: bool,
        resizable: bool,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let inner = Window::open(width, height, title, vsync, resizable, device).map_err(to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn open(&self) -> bool {
        self.inner.is_open()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device_id()
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        self.inner.size()
    }

    #[getter]
    fn physical_size(&self) -> (u32, u32) {
        self.inner.physical_size()
    }

    #[getter]
    fn vsync(&self) -> bool {
        self.inner.vsync()
    }

    #[setter]
    fn set_vsync(&mut self, vsync: bool) {
        self.inner.set_vsync(vsync);
    }

    #[getter]
    fn zero_copy(&self) -> bool {
        self.inner.last_zero_copy()
    }

    fn set_title(&self, title: &str) {
        self.inner.set_title(title);
    }

    #[getter]
    fn mouse_pos(&self) -> (f64, f64) {
        self.inner.mouse_pos()
    }

    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for k in self.inner.keys_down() {
            s.add(k.as_str())?;
        }
        Ok(s)
    }

    #[getter]
    fn mouse_buttons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for b in self.inner.mouse_buttons() {
            s.add(b.as_str())?;
        }
        Ok(s)
    }

    fn key_down(&self, key: &str) -> bool {
        self.inner.keys_down().contains(&key.to_lowercase())
    }

    fn poll<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for ev in self.inner.poll() {
            let d = PyDict::new(py);
            match ev {
                WindowEvent::Close => {
                    d.set_item("type", "close")?;
                }
                WindowEvent::Resize { width, height } => {
                    d.set_item("type", "resize")?;
                    d.set_item("width", width)?;
                    d.set_item("height", height)?;
                }
                WindowEvent::KeyDown { key } => {
                    d.set_item("type", "key_down")?;
                    d.set_item("key", key)?;
                }
                WindowEvent::KeyUp { key } => {
                    d.set_item("type", "key_up")?;
                    d.set_item("key", key)?;
                }
                WindowEvent::MouseDown { button } => {
                    d.set_item("type", "mouse_down")?;
                    d.set_item("button", button)?;
                }
                WindowEvent::MouseUp { button } => {
                    d.set_item("type", "mouse_up")?;
                    d.set_item("button", button)?;
                }
                WindowEvent::MouseMove { x, y, dx, dy } => {
                    d.set_item("type", "mouse_move")?;
                    d.set_item("x", x)?;
                    d.set_item("y", y)?;
                    d.set_item("dx", dx)?;
                    d.set_item("dy", dy)?;
                }
                WindowEvent::MouseWheel { dx, dy } => {
                    d.set_item("type", "mouse_wheel")?;
                    d.set_item("dx", dx)?;
                    d.set_item("dy", dy)?;
                }
            }
            list.append(d)?;
        }
        Ok(list)
    }

    /// Shows a `DeviceArray` (no CPU copy if it is already on the same wgpu device).
    fn present_array(&mut self, arr: &DeviceArrayPy, width: u32, height: u32) -> PyResult<bool> {
        self.inner
            .present_array(&arr.array, width, height)
            .map_err(to_py)
    }

    /// Shows host bytes `(ptr, nbytes)` (RGBA8 / packed u32).
    fn present_host(&mut self, ptr: usize, nbytes: usize, width: u32, height: u32) -> PyResult<()> {
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };
        self.inner
            .present_bytes(slice, width, height)
            .map_err(to_py)
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWindow>()?;
    Ok(())
}
