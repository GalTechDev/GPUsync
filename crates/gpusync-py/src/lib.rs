//! Liaison Python (PyO3) : module `gpusync._core`. Seul crate du projet qui dépend de Python.

mod kernel;

use std::sync::Mutex;

use gpusync_backend_cuda::CudaBackend;
use gpusync_backend_opencl::OpenClBackend;
use gpusync_backend_wgpu::WgpuBackend;
use gpusync_runtime::{Backend, BackendStatus, DeviceInfo, all_devices, discover};
use pyo3::prelude::*;

/// Résultat de la découverte, gardé en cache : créer une instance wgpu coûte ~100 ms.
static CACHE: Mutex<Option<Vec<BackendStatus>>> = Mutex::new(None);

/// Backends interrogés : CUDA, wgpu (Metal / Vulkan / DX12) puis OpenCL ; le classement des
/// périphériques est fait côté Python (§4.8 de l'architecture).
fn statuses(py: Python<'_>, refresh: bool) -> Vec<BackendStatus> {
    py.detach(|| {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if refresh || cache.is_none() {
            let backends: [&dyn Backend; 3] = [&CudaBackend, &WgpuBackend, &OpenClBackend];
            let mut st = discover(&backends);
            // L'interpréteur de référence, toujours disponible, en dernier recours.
            st.push(BackendStatus {
                name: "cpu",
                devices: vec![::gpusync_core::cpu_info()],
                error: None,
            });
            *cache = Some(st);
        }
        cache.clone().unwrap_or_default()
    })
}

/// Un périphérique de calcul.
#[pyclass(name = "Device", module = "gpusync", frozen, get_all)]
struct PyDevice {
    /// Nom commercial du périphérique.
    name: String,
    /// API utilisée : "cuda", "metal", "vulkan", "dx12" ou "opencl".
    backend: String,
    /// Index parmi les périphériques du même backend.
    index: usize,
    /// "discrete_gpu", "integrated_gpu", "virtual_gpu", "cpu" ou "other".
    kind: String,
    vendor: String,
    /// Version de l'API ou du driver.
    version: String,
    /// Calcul en float64 supporté (arithmétique).
    f64: bool,
    /// Fonctions transcendantes (exp, log, sin…) en float64 supportées.
    f64_math: bool,
    /// Entiers 64 bits (gs.i64, gs.u64) supportés.
    int64: bool,
    /// Mémoire partagée avec l'hôte (zéro-copie possible).
    unified_memory: bool,
    /// Mémoire globale en octets (None si le backend ne l'expose pas).
    memory_bytes: Option<u64>,
    /// Taille max d'un buffer en octets.
    max_buffer_bytes: Option<u64>,
}

impl From<&DeviceInfo> for PyDevice {
    fn from(d: &DeviceInfo) -> Self {
        PyDevice {
            name: d.name.clone(),
            backend: d.backend.to_string(),
            index: d.index,
            kind: d.kind.as_str().to_string(),
            vendor: d.vendor.clone(),
            version: d.version.clone(),
            f64: d.f64,
            f64_math: d.f64_math,
            int64: d.int64,
            unified_memory: d.unified_memory,
            memory_bytes: d.memory_bytes,
            max_buffer_bytes: d.max_buffer_bytes,
        }
    }
}

#[pymethods]
impl PyDevice {
    /// Identifiant court, ex. "opencl:1" : utilisable comme `device="opencl:1"`.
    #[getter]
    fn id(&self) -> String {
        format!("{}:{}", self.backend, self.index)
    }

    fn __repr__(&self) -> String {
        let mut s = format!(
            "Device('{}', backend='{}', f64={}",
            self.name,
            self.backend,
            py_bool(self.f64)
        );
        if self.f64 && !self.f64_math {
            s.push_str(", f64_math=False");
        }
        if !self.int64 {
            s.push_str(", int64=False");
        }
        s.push_str(&format!(
            ", unified_memory={}",
            py_bool(self.unified_memory)
        ));
        if let Some(m) = self.memory_bytes {
            s.push_str(&format!(
                ", memory='{:.1} Go'",
                m as f64 / (1u64 << 30) as f64
            ));
        }
        s.push(')');
        s
    }
}

fn py_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// État d'un backend : disponible ou non, et pourquoi.
#[pyclass(name = "BackendStatus", module = "gpusync", frozen, get_all)]
struct PyBackendStatus {
    name: String,
    available: bool,
    /// Raison de l'indisponibilité (None si disponible).
    error: Option<String>,
    devices: Vec<Py<PyDevice>>,
}

#[pymethods]
impl PyBackendStatus {
    fn __repr__(&self) -> String {
        match &self.error {
            None => format!(
                "BackendStatus('{}', disponible, {} périphérique(s))",
                self.name,
                self.devices.len()
            ),
            Some(e) => format!("BackendStatus('{}', indisponible : {e})", self.name),
        }
    }
}

/// Liste les périphériques de calcul disponibles, dans l'ordre de préférence.
///
/// La découverte est faite une fois puis mise en cache ; `refresh=True` la refait.
#[pyfunction]
#[pyo3(signature = (refresh = false))]
fn devices(py: Python<'_>, refresh: bool) -> Vec<PyDevice> {
    all_devices(&statuses(py, refresh))
        .iter()
        .map(PyDevice::from)
        .collect()
}

/// État de chaque backend (utile pour comprendre pourquoi un GPU n'apparaît pas).
#[pyfunction]
#[pyo3(signature = (refresh = false))]
fn backends(py: Python<'_>, refresh: bool) -> PyResult<Vec<PyBackendStatus>> {
    statuses(py, refresh)
        .iter()
        .map(|s| {
            Ok(PyBackendStatus {
                name: s.name.to_string(),
                available: s.available(),
                error: s.error.clone(),
                devices: s
                    .devices
                    .iter()
                    .map(|d| Py::new(py, PyDevice::from(d)))
                    .collect::<PyResult<_>>()?,
            })
        })
        .collect()
}

#[pymodule]
#[pyo3(name = "_core")]
fn gpusync_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyBackendStatus>()?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(backends, m)?)?;
    kernel::register(m)?;
    Ok(())
}
