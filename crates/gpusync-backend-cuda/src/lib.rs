//! Backend CUDA (ARCHITECTURE.md §8, jalon M6). Le driver (`nvcuda` / `libcuda`) et NVRTC sont chargés
//! dynamiquement : rien à installer à la compilation, et sur une machine sans GPU NVIDIA le backend est
//! simplement indisponible, avec la raison.

mod exec;
mod sys;

pub use exec::{CuArg, CuBuffer, CuDevice, CuEvent, CuKernel, Pinned};

use std::ffi::{CStr, c_char};

use gpusync_runtime::{Backend, BackendError, DeviceInfo, DeviceKind};

pub struct CudaBackend;

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    /// Sans NVRTC, aucun kernel ne peut être compilé : le backend est indisponible (avec la façon de
    /// l'installer), et le choix automatique passe au backend suivant.
    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        let devs = enumerate()?;
        if !devs.is_empty() && !sys::nvrtc_present() {
            return Err(BackendError(sys::NVRTC_MISSING.into()));
        }
        Ok(devs.into_iter().map(|(_, info)| info).collect())
    }
}

/// Version de NVRTC chargée (ex. « 12.1 »), ou la raison de son absence.
pub fn nvrtc_version() -> Result<String, BackendError> {
    let n = sys::nvrtc()?;
    let (a, b) = n.version();
    Ok(format!("{a}.{b}"))
}

pub(crate) fn enumerate() -> Result<Vec<(sys::CUdevice, DeviceInfo)>, BackendError> {
    let d = sys::driver()?;
    let mut version = 0;
    d.check(
        unsafe { (d.cuDriverGetVersion)(&mut version) },
        "version du driver",
    )?;
    let mut count = 0;
    d.check(
        unsafe { (d.cuDeviceGetCount)(&mut count) },
        "nombre de périphériques",
    )?;
    let mut out = Vec::new();
    for i in 0..count {
        let mut dev = 0;
        d.check(unsafe { (d.cuDeviceGet)(&mut dev, i) }, "périphérique")?;
        let mut name = [0 as c_char; 256];
        d.check(
            unsafe { (d.cuDeviceGetName)(name.as_mut_ptr(), 256, dev) },
            "nom",
        )?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string();
        let mut mem = 0usize;
        d.check(unsafe { (d.cuDeviceTotalMem_v2)(&mut mem, dev) }, "mémoire")?;
        let attr = |a| {
            let mut v = 0;
            unsafe { (d.cuDeviceGetAttribute)(&mut v, a, dev) };
            v
        };
        let integrated = attr(sys::ATTR_INTEGRATED) != 0;
        let (major, minor) = (attr(sys::ATTR_CC_MAJOR), attr(sys::ATTR_CC_MINOR));
        out.push((
            dev,
            DeviceInfo {
                name,
                backend: "cuda",
                index: i as usize,
                kind: if integrated {
                    DeviceKind::IntegratedGpu
                } else {
                    DeviceKind::DiscreteGpu
                },
                vendor: "NVIDIA".into(),
                version: format!(
                    "CUDA {}.{} (capacité {major}.{minor})",
                    version / 1000,
                    (version % 1000) / 10
                ),
                // Tous les GPU NVIDIA calculent en f64, fonctions mathématiques comprises.
                f64: true,
                f64_math: true,
                int64: true,
                // Jetson : mémoire partagée, mais traitée comme un GPU dédié (pas de mode Direct à M6).
                unified_memory: false,
                memory_bytes: Some(mem as u64),
                max_buffer_bytes: Some(mem as u64),
                max_group: attr(sys::ATTR_MAX_THREADS_PER_BLOCK).clamp(1, 1024) as u32,
                shared_bytes: attr(sys::ATTR_MAX_SHARED_MEMORY_PER_BLOCK).max(0) as u64,
            },
        ));
    }
    Ok(out)
}
