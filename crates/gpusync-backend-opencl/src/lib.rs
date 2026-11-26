//! OpenCL backend (≥ 1.2). The OpenCL library is loaded dynamically on first use:
//! if it is missing, the backend is simply unavailable (no error at import).

mod exec;

pub use exec::{ClArg, ClBuffer, ClDevice, ClKernel, Pinned, wait};
pub use opencl3::event::Event as ClEvent;

use gpusync_runtime::{Backend, BackendError, DeviceInfo, DeviceKind};
use opencl3::device::{
    CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, Device, cl_device_id,
    cl_device_type,
};
use opencl3::platform::get_platforms;

pub struct OpenClBackend;

impl Backend for OpenClBackend {
    fn name(&self) -> &'static str {
        "opencl"
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(enumerate()?.into_iter().map(|(_, info)| info).collect())
    }
}

/// OpenCL devices, in a stable order (platforms then devices): the index of
/// `DeviceInfo` is used to reopen the same device.
pub(crate) fn enumerate() -> Result<Vec<(cl_device_id, DeviceInfo)>, BackendError> {
    if let Err(e) = cl3::load_library() {
        return Err(BackendError(format!(
            "OpenCL library not found ({e})"
        )));
    }
    let platforms =
        get_platforms().map_err(|e| BackendError(format!("no OpenCL platform ({e})")))?;
    let mut out = Vec::new();
    for platform in platforms {
        let platform_name = platform.name().unwrap_or_default();
        let ids = match platform.get_devices(CL_DEVICE_TYPE_ALL) {
            Ok(ids) => ids,
            Err(_) => continue, // platform without a device: skip it
        };
        for id in ids {
            let d = Device::new(id);
            let unified = d.host_unified_memory().unwrap_or(false);
            let info = DeviceInfo {
                name: d
                    .name()
                    .unwrap_or_else(|_| platform_name.clone())
                    .trim()
                    .to_string(),
                backend: "opencl",
                index: out.len(),
                kind: kind_of(d.dev_type().unwrap_or(0), unified),
                vendor: d.vendor().unwrap_or_default().trim().to_string(),
                version: d.version().unwrap_or_default().trim().to_string(),
                f64: d.double_fp_config().map(|c| c != 0).unwrap_or(false),
                // OpenCL provides the whole math library in double precision.
                f64_math: d.double_fp_config().map(|c| c != 0).unwrap_or(false),
                // `long` / `ulong` are part of OpenCL C (full profile), except the embedded profile.
                int64: !d.profile().unwrap_or_default().contains("EMBEDDED"),
                unified_memory: unified,
                memory_bytes: d.global_mem_size().ok(),
                max_buffer_bytes: d.max_mem_alloc_size().ok(),
                max_group: d
                    .max_work_group_size()
                    .map(|v| v.min(1024) as u32)
                    .unwrap_or(64),
                shared_bytes: d.local_mem_size().unwrap_or(16 << 10),
            };
            out.push((id, info));
        }
    }
    // The platform order given by the OpenCL loader is not stable from one environment to
    // another (seen on Windows: RTX then UHD 770, then the reverse): sort them, so that
    // `opencl:0` always names the same GPU.
    out.sort_by(|(_, a), (_, b)| {
        (a.kind.rank(), &a.name, &a.vendor).cmp(&(b.kind.rank(), &b.name, &b.vendor))
    });
    for (i, (_, info)) in out.iter_mut().enumerate() {
        info.index = i;
    }
    Ok(out)
}

/// OpenCL does not tell integrated from discrete GPUs: unified memory is used as a hint.
fn kind_of(dev_type: cl_device_type, unified: bool) -> DeviceKind {
    if dev_type & CL_DEVICE_TYPE_GPU != 0 {
        if unified {
            DeviceKind::IntegratedGpu
        } else {
            DeviceKind::DiscreteGpu
        }
    } else if dev_type & CL_DEVICE_TYPE_CPU != 0 {
        DeviceKind::Cpu
    } else {
        DeviceKind::Other // accelerators (FPGA…) and unknown types
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_type() {
        assert_eq!(kind_of(CL_DEVICE_TYPE_GPU, true), DeviceKind::IntegratedGpu);
        assert_eq!(kind_of(CL_DEVICE_TYPE_GPU, false), DeviceKind::DiscreteGpu);
        assert_eq!(kind_of(CL_DEVICE_TYPE_CPU, true), DeviceKind::Cpu);
    }

    #[test]
    fn discovery_never_crashes() {
        // Available or not depending on the machine: only the absence of a panic matters.
        let _ = OpenClBackend.devices();
    }
}
