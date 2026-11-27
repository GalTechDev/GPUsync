//! Python binding (PyO3): the `gpusync._core` module. The only crate of the project that depends on Python.

mod kernel;
mod window;

use std::sync::Mutex;

use gpusync_backend_cuda::CudaBackend;
use gpusync_backend_opencl::OpenClBackend;
use gpusync_backend_wgpu::WgpuBackend;
use gpusync_runtime::{Backend, BackendStatus, DeviceInfo, all_devices, discover};
use pyo3::prelude::*;

/// Discovery result, cached: creating a wgpu instance costs ~100 ms.
static CACHE: Mutex<Option<Vec<BackendStatus>>> = Mutex::new(None);

/// Backends queried: CUDA, wgpu (Metal / Vulkan / DX12) then OpenCL; devices are ranked
/// on the Python side (§4.8 of the architecture).
fn statuses(py: Python<'_>, refresh: bool) -> Vec<BackendStatus> {
    py.detach(|| {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if refresh || cache.is_none() {
            let backends: [&dyn Backend; 3] = [&CudaBackend, &WgpuBackend, &OpenClBackend];
            let mut st = discover(&backends);
            // The reference interpreter, always available, as a last resort.
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

/// A compute device.
#[pyclass(name = "Device", module = "gpusync", frozen, get_all)]
struct PyDevice {
    /// Commercial name of the device.
    name: String,
    /// API used: "cuda", "metal", "vulkan", "dx12" or "opencl".
    backend: String,
    /// Index among the devices of the same backend.
    index: usize,
    /// "discrete_gpu", "integrated_gpu", "virtual_gpu", "cpu" or "other".
    kind: String,
    vendor: String,
    /// API or driver version.
    version: String,
    /// float64 computation supported (arithmetic).
    f64: bool,
    /// Transcendental functions (exp, log, sin…) supported in float64.
    f64_math: bool,
    /// 64-bit integers (gs.i64, gs.u64) supported.
    int64: bool,
    /// Memory shared with the host (zero-copy possible).
    unified_memory: bool,
    /// Global memory in bytes (None if the backend does not expose it).
    memory_bytes: Option<u64>,
    /// Largest buffer size in bytes.
    max_buffer_bytes: Option<u64>,
    /// Most threads per group (`@gs.kernel(group=n)`).
    max_group: u32,
    /// Most shared memory (`gs.shared`) per group, in bytes.
    shared_bytes: u64,
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
            max_group: d.max_group,
            shared_bytes: d.shared_bytes,
        }
    }
}

#[pymethods]
impl PyDevice {
    /// Short identifier, e.g. "opencl:1": usable as `device="opencl:1"`.
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

/// Status of a backend: available or not, and why.
#[pyclass(name = "BackendStatus", module = "gpusync", frozen, get_all)]
struct PyBackendStatus {
    name: String,
    available: bool,
    /// Reason it is unavailable (None if available).
    error: Option<String>,
    devices: Vec<Py<PyDevice>>,
}

#[pymethods]
impl PyBackendStatus {
    fn __repr__(&self) -> String {
        match &self.error {
            None => format!(
                "BackendStatus('{}', available, {} device(s))",
                self.name,
                self.devices.len()
            ),
            Some(e) => format!("BackendStatus('{}', unavailable: {e})", self.name),
        }
    }
}

/// Lists the available compute devices, in order of preference.
///
/// Discovery runs once and is cached; `refresh=True` runs it again.
#[pyfunction]
#[pyo3(signature = (refresh = false))]
fn devices(py: Python<'_>, refresh: bool) -> Vec<PyDevice> {
    all_devices(&statuses(py, refresh))
        .iter()
        .map(PyDevice::from)
        .collect()
}

/// Status of each backend (useful to understand why a GPU does not show up).
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
    window::register(m)?;
    Ok(())
}
