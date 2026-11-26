//! Native GPU window (`gs.Window`): shows a `DeviceArray` straight from the GPU
//! (no CPU copy when it lives on the same wgpu device) or from host bytes.

use std::collections::HashSet;
use std::sync::Arc;

pub use gpusync_backend_wgpu::WindowEvent;
use gpusync_backend_wgpu::{WgDevice, WgWindow};

use crate::{CoreError, Device, DeviceArray, R, Storage, dev_err};

/// Native window (main thread only).
pub struct Window {
    inner: WgWindow,
    scratch: Vec<u8>,
    last_zero_copy: bool,
}

impl Window {
    pub fn open(
        width: u32,
        height: u32,
        title: &str,
        vsync: bool,
        resizable: bool,
        device_id: Option<&str>,
    ) -> R<Self> {
        let dev = match device_id {
            Some(id) => match Device::open(id)? {
                Device::Wg(d) => Some(d),
                _ => {
                    return Err(CoreError::Unsupported(format!(
                        "gs.Window displays through wgpu (metal / vulkan / dx12), not \"{id}\" — \
                         leave device=None: results from {id} can still be displayed automatically"
                    )));
                }
            },
            None => None,
        };
        let inner = WgWindow::open(width, height, title, vsync, resizable, dev).map_err(dev_err)?;
        Ok(Self {
            inner,
            scratch: Vec::new(),
            last_zero_copy: false,
        })
    }

    pub fn device_id(&self) -> String {
        self.inner.device_id()
    }

    pub fn is_open(&self) -> bool {
        self.inner.is_open()
    }

    pub fn size(&self) -> (u32, u32) {
        self.inner.size()
    }

    pub fn physical_size(&self) -> (u32, u32) {
        self.inner.physical_size()
    }

    pub fn vsync(&self) -> bool {
        self.inner.vsync()
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.inner.set_vsync(vsync);
    }

    pub fn set_title(&self, title: &str) {
        self.inner.set_title(title);
    }

    pub fn last_zero_copy(&self) -> bool {
        self.last_zero_copy
    }

    pub fn keys_down(&self) -> &HashSet<String> {
        self.inner.keys_down()
    }

    pub fn mouse_buttons(&self) -> &HashSet<String> {
        self.inner.mouse_buttons()
    }

    pub fn mouse_pos(&self) -> (f64, f64) {
        self.inner.mouse_pos()
    }

    pub fn poll(&mut self) -> Vec<WindowEvent> {
        self.inner.poll()
    }

    /// Shows a `DeviceArray` (RGBA8 packed in `u32`, `width × height`).
    /// Returns `true` if the GPU buffer was shown directly, without going through CPU RAM.
    pub fn present_array(&mut self, arr: &DeviceArray, width: u32, height: u32) -> R<bool> {
        let need = (width as usize) * (height as usize) * 4;
        if arr.bytes() < need {
            return Err(CoreError::Shape(format!(
                "gs.Window.present: array of {} bytes for {width}×{height} u32 ({need} expected)",
                arr.bytes()
            )));
        }
        let same_wg = matches!(&arr.device, Device::Wg(d) if Arc::ptr_eq(d, self.inner.device()));
        if same_wg {
            let direct_buf = match &arr.storage {
                Storage::Wg(b) => Some(b),
                Storage::WgDirect { bufs, .. } if bufs.len() == 1 => bufs.first(),
                _ => None,
            };
            if let Some(buf) = direct_buf {
                self.inner.present_wg(buf, width, height).map_err(dev_err)?;
                self.last_zero_copy = true;
                return Ok(true);
            }
        }
        // Other backend (OpenCL, CUDA, CPU): host read into a reused buffer, then GPU upload.
        if self.scratch.len() < arr.bytes() {
            self.scratch.resize(arr.bytes(), 0);
        }
        unsafe { arr.read_into(self.scratch.as_mut_ptr())? };
        self.inner
            .present_bytes(&self.scratch[..need], width, height)
            .map_err(dev_err)?;
        self.last_zero_copy = false;
        Ok(false)
    }

    /// Shows RGBA8 bytes from host memory (NumPy array…).
    pub fn present_bytes(&mut self, rgba: &[u8], width: u32, height: u32) -> R<()> {
        self.inner
            .present_bytes(rgba, width, height)
            .map_err(dev_err)?;
        self.last_zero_copy = false;
        Ok(())
    }

    pub fn close(&mut self) {
        self.inner.close();
    }

    pub fn wg_device(&self) -> &Arc<WgDevice> {
        self.inner.device()
    }
}
