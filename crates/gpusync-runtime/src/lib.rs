//! GPUsync runtime: devices, backends and (later) buffers, streams and scheduling.
//!
//! M0: device discovery. Each backend implements [`Backend`]; the runtime gathers
//! their devices and keeps each backend's status (available or not, and why).
//! Voir `ARCHITECTURE.md` §8.

use std::fmt;

/// Device category, as reported by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::DiscreteGpu => "discrete_gpu",
            DeviceKind::IntegratedGpu => "integrated_gpu",
            DeviceKind::VirtualGpu => "virtual_gpu",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Other => "other",
        }
    }

    /// Rank for a stable order: discrete, integrated, virtual GPUs, CPU, others.
    pub fn rank(self) -> u8 {
        match self {
            DeviceKind::DiscreteGpu => 0,
            DeviceKind::IntegratedGpu => 1,
            DeviceKind::VirtualGpu => 2,
            DeviceKind::Cpu => 3,
            DeviceKind::Other => 4,
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu | DeviceKind::VirtualGpu
        )
    }
}

/// Description of a compute device discovered by a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    /// Commercial name, e.g. "NVIDIA GeForce RTX 4080 SUPER".
    pub name: String,
    /// API used: "metal", "vulkan", "dx12" (through wgpu) or "opencl".
    pub backend: &'static str,
    /// Index among the devices of the same backend (`"opencl:1"`).
    pub index: usize,
    pub kind: DeviceKind,
    /// Vendor as reported by the driver.
    pub vendor: String,
    /// API or driver version, e.g. "OpenCL 3.0 CUDA" or "610.88".
    pub version: String,
    /// Can the device compute in float64 (arithmetic)?
    pub f64: bool,
    /// … including transcendental functions (exp, log, sin…) in f64?
    pub f64_math: bool,
    /// 64-bit integers (i64, u64) in kernels?
    pub int64: bool,
    /// Memory shared with the host (zero-copy possible, §8.4 of the architecture).
    pub unified_memory: bool,
    /// Global memory in bytes, if the backend exposes it (wgpu does not).
    pub memory_bytes: Option<u64>,
    /// Largest buffer size, in bytes (constrains out-of-core chunking).
    pub max_buffer_bytes: Option<u64>,
    /// Most threads per group (`@gs.kernel(group=n)`).
    pub max_group: u32,
    /// Most shared memory (`gs.shared`) per group, in bytes.
    pub shared_bytes: u64,
}

impl DeviceInfo {
    /// Short identifier: `"opencl:0"`, `"metal:0"`…
    pub fn id(&self) -> String {
        format!("{}:{}", self.backend, self.index)
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.id())
    }
}

/// Backend error (missing library, failing driver…).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

/// A compute backend (wgpu, OpenCL, CUDA…).
///
/// M0 only requires discovery. Operations (allocation, launch, wait) came
/// with M1, following the trait described in §8.2 of the architecture.
pub trait Backend: Send + Sync {
    /// Name of the backend: "wgpu", "opencl"…
    fn name(&self) -> &'static str;

    /// Lists the devices. An error means "backend unavailable on this machine":
    /// it must never crash the caller.
    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError>;
}

/// Status of a backend after discovery.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub name: &'static str,
    pub devices: Vec<DeviceInfo>,
    /// `None` if the backend is available; otherwise the reason.
    pub error: Option<String>,
}

impl BackendStatus {
    pub fn available(&self) -> bool {
        self.error.is_none()
    }
}

/// Queries every backend. A failing backend is reported, never fatal (goal O1).
pub fn discover(backends: &[&dyn Backend]) -> Vec<BackendStatus> {
    backends
        .iter()
        .map(|b| {
            // A broken driver may panic in third-party code: isolate that too.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.devices())) {
                Ok(Ok(devices)) => BackendStatus {
                    name: b.name(),
                    devices,
                    error: None,
                },
                Ok(Err(e)) => BackendStatus {
                    name: b.name(),
                    devices: vec![],
                    error: Some(e.0),
                },
                Err(_) => BackendStatus {
                    name: b.name(),
                    devices: vec![],
                    error: Some("the backend panicked during discovery".into()),
                },
            }
        })
        .collect()
}

/// Flat list of every device, in the order of preference of the given backends.
pub fn all_devices(statuses: &[BackendStatus]) -> Vec<DeviceInfo> {
    statuses
        .iter()
        .flat_map(|s| s.devices.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str, Result<usize, &'static str>);

    impl Backend for Fake {
        fn name(&self) -> &'static str {
            self.0
        }
        fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
            match self.1 {
                Ok(n) => Ok((0..n)
                    .map(|i| DeviceInfo {
                        name: format!("GPU {i}"),
                        backend: self.0,
                        index: i,
                        kind: DeviceKind::DiscreteGpu,
                        vendor: "test".into(),
                        version: "1".into(),
                        f64: true,
                        f64_math: true,
                        int64: true,
                        unified_memory: false,
                        memory_bytes: None,
                        max_buffer_bytes: None,
                        max_group: 1024,
                        shared_bytes: 48 << 10,
                    })
                    .collect()),
                Err(e) => Err(BackendError(e.into())),
            }
        }
    }

    struct Panics;

    impl Backend for Panics {
        fn name(&self) -> &'static str {
            "panique"
        }
        fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
            panic!("broken driver")
        }
    }

    #[test]
    fn failing_or_panicking_backend_does_not_block_others() {
        let ok = Fake("ok", Ok(2));
        let ko = Fake("ko", Err("library not found"));
        let st = discover(&[&ko, &Panics, &ok]);
        assert_eq!(st.len(), 3);
        assert_eq!(st[0].error.as_deref(), Some("library not found"));
        assert!(st[1].error.as_deref().unwrap().contains("panicked"));
        assert!(st[2].available());
        let devs = all_devices(&st);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[1].id(), "ok:1");
    }
}
