//! Rust core of GPUsync (no Python): frontend + code generation + execution on a backend.
//!
//! - [`parse`]: Python AST (JSON) → typed IR;
//! - [`Device`]: open device (`cuda:0`, `opencl:0`, `metal:0`, `vulkan:1`, `dx12:0`, `cpu:0`);
//! - [`compile`]: checks capabilities, generates and compiles the code (with a cache);
//! - [`launch`]: runs over `n` elements, in chunks, in the chosen mode (transfer engine,
//!   ARCHITECTURE.md §9) and returns a [`DeviceArray`].

use std::collections::HashMap;
use std::sync::Arc;

use gpusync_backend_cuda::{CuDevice, CuKernel};
use gpusync_backend_opencl::{ClDevice, ClKernel};
use gpusync_backend_wgpu::{WgDevice, WgKernel};
use gpusync_ir::Kernel;
use gpusync_runtime::{DeviceInfo, DeviceKind};

pub use gpusync_frontend::{ErrorKind as FrontendErrorKind, FrontendError};

/// Errors, each one tied to a Python exception (§4.10).
#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    Syntax(String),
    Type(String),
    Shape(String),
    Unsupported(String),
    Device(String),
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (CoreError::Syntax(m)
        | CoreError::Type(m)
        | CoreError::Shape(m)
        | CoreError::Unsupported(m)
        | CoreError::Device(m)) = self;
        f.write_str(m)
    }
}

impl std::error::Error for CoreError {}

type R<T> = Result<T, CoreError>;

fn dev_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Device(e.to_string())
}

/// Python AST (JSON, see gpusync-frontend) → typed IR.
pub fn parse(spec_json: &str) -> R<Arc<Kernel>> {
    gpusync_frontend::lower_json(spec_json)
        .map(Arc::new)
        .map_err(|e| match e.kind {
            FrontendErrorKind::Syntax => CoreError::Syntax(e.to_string()),
            FrontendErrorKind::Type => CoreError::Type(e.to_string()),
        })
}

/// Generated code for a target: "cuda", "opencl", "wgsl" or "ir".
pub fn source(k: &Kernel, target: &str) -> R<String> {
    match target {
        "cuda" => gpusync_codegen::emit_cuda(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "opencl" => gpusync_codegen::emit_opencl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "metal" => gpusync_codegen::emit_wgsl_variant(k, &[], true)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "wgsl" | "vulkan" | "dx12" => gpusync_codegen::emit_wgsl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "ir" => Ok(format!("{k:#?}")),
        t => Err(CoreError::Type(format!(
            "unknown target \"{t}\" (cuda, opencl, wgsl or ir)"
        ))),
    }
}

/// Description of the interpreter, exposed as the `cpu:0` device.
pub fn cpu_info() -> DeviceInfo {
    DeviceInfo {
        name: "GPUsync interpreter (CPU, slow)".into(),
        backend: "cpu",
        index: 0,
        kind: DeviceKind::Cpu,
        vendor: "GPUsync".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        f64: true,
        f64_math: true,
        int64: true,
        unified_memory: true,
        memory_bytes: None,
        max_buffer_bytes: None,
        // One system thread per position in the group; shared memory in RAM.
        max_group: 1024,
        shared_bytes: 1 << 30,
    }
}

/// An open device.
#[derive(Clone)]
pub enum Device {
    Cu(Arc<CuDevice>),
    Cl(Arc<ClDevice>),
    Wg(Arc<WgDevice>),
    Cpu,
}

impl Device {
    /// Opens `cuda:<i>`, `opencl:<i>`, `metal:<i>`, `vulkan:<i>`, `dx12:<i>` or `cpu:0`.
    pub fn open(id: &str) -> R<Device> {
        let (backend, index) = id.split_once(':').unwrap_or((id, "0"));
        let index: usize = index.parse().map_err(|_| {
            CoreError::Device(format!("invalid device identifier: {id}"))
        })?;
        match backend {
            "cuda" => CuDevice::open(index).map(Device::Cu).map_err(dev_err),
            "opencl" => ClDevice::open(index).map(Device::Cl).map_err(dev_err),
            "metal" | "vulkan" | "dx12" => WgDevice::open(backend, index)
                .map(Device::Wg)
                .map_err(dev_err),
            "cpu" if index == 0 => Ok(Device::Cpu),
            _ => Err(CoreError::Device(format!("unknown device: {id}"))),
        }
    }

    /// Same open device (not just the same identifier: a device lost by the
    /// driver is reopened under the same identifier).
    pub fn same(&self, other: &Device) -> bool {
        match (self, other) {
            (Device::Cu(a), Device::Cu(b)) => Arc::ptr_eq(a, b),
            (Device::Cl(a), Device::Cl(b)) => Arc::ptr_eq(a, b),
            (Device::Wg(a), Device::Wg(b)) => Arc::ptr_eq(a, b),
            (Device::Cpu, Device::Cpu) => true,
            _ => false,
        }
    }

    pub fn info(&self) -> DeviceInfo {
        match self {
            Device::Cu(d) => d.info.clone(),
            Device::Cl(d) => d.info.clone(),
            Device::Wg(d) => d.info.clone(),
            Device::Cpu => cpu_info(),
        }
    }

    pub fn id(&self) -> String {
        self.info().id()
    }

    /// Inputs and output in host memory, without copy?
    pub fn zero_copy(&self) -> bool {
        match self {
            Device::Cu(_) => false,
            Device::Cl(d) => d.info.unified_memory,
            Device::Wg(d) => d.info.backend == "metal", // newBufferWithBytesNoCopy
            Device::Cpu => true,
        }
    }

    /// Waits for all queued work.
    pub fn finish(&self) -> R<()> {
        match self {
            Device::Cu(d) => d.finish().map_err(dev_err),
            Device::Cl(d) => d.finish().map_err(dev_err),
            Device::Wg(d) => d.finish().map_err(dev_err),
            Device::Cpu => Ok(()),
        }
    }

    /// Opens a command batch (wgpu: groups several kernels into a single GPU submission).
    pub fn begin_batch(&self) {
        if let Device::Wg(d) = self {
            d.begin_batch();
        }
    }

    /// Closes the current command batch and submits the accumulated work.
    pub fn end_batch(&self) -> R<()> {
        if let Device::Wg(d) = self {
            d.end_batch().map_err(dev_err)?;
        }
        Ok(())
    }
}

/// Kernel ready to launch on a given device.
#[derive(Clone)]
pub struct Compiled {
    pub kernel: Arc<Kernel>,
    pub(crate) imp: Imp,
    /// Variants compiled on demand: 64-bit indices (array of more than 2³¹ − 1 elements) and
    /// sizes known at call time replaced by constants (§5.4).
    pub(crate) variants: Arc<std::sync::Mutex<HashMap<VariantKey, Imp>>>,
}

/// (64-bit indices, specialized parameters).
pub(crate) type VariantKey = (bool, Vec<(usize, i64)>);

/// Most specialized variants kept per kernel and per device; beyond that (sizes that
/// change at every call), the generic version is used, without recompiling.
const MAX_SPECIALIZED: usize = 8;

impl Compiled {
    /// Implementation to use for this call.
    pub(crate) fn variant(&self, dev: &Device, wide: bool, spec: Vec<(usize, i64)>) -> R<Imp> {
        if !wide && spec.is_empty() {
            return Ok(self.imp.clone());
        }
        let mut cache = self.variants.lock().unwrap_or_else(|e| e.into_inner());
        let key = (wide, spec);
        if let Some(imp) = cache.get(&key) {
            return Ok(imp.clone());
        }
        let key = if !key.1.is_empty()
            && cache
                .keys()
                .filter(|(w, s)| *w == wide && !s.is_empty())
                .count()
                >= MAX_SPECIALIZED
        {
            (wide, Vec::new()) // too many different sizes: generic version
        } else {
            key
        };
        if !key.0 && key.1.is_empty() {
            return Ok(self.imp.clone());
        }
        if let Some(imp) = cache.get(&key) {
            return Ok(imp.clone());
        }
        let imp = build(dev, &self.kernel, key.0, &key.1)?;
        cache.insert(key, imp.clone());
        Ok(imp)
    }
}

#[derive(Clone)]
pub(crate) enum Imp {
    Cu(Arc<CuKernel>),
    Cl(Arc<ClKernel>),
    Wg(Arc<WgKernel>),
    Cpu,
}

/// Reason why `info` cannot run `k` (§6.2), or `None` if it can.
pub fn unsupported(info: &DeviceInfo, k: &Kernel) -> Option<String> {
    let req = k.requirements();
    if req.f64 && !info.f64 {
        return Some(format!(
            "kernel '{}' uses f64, not supported by '{}' ({})",
            k.name,
            info.name,
            info.id()
        ));
    }
    if req.int64 && !info.int64 {
        return Some(format!(
            "kernel '{}' uses 64-bit integers (i64, u64), not supported by '{}' ({})",
            k.name,
            info.name,
            info.id()
        ));
    }
    if let Some(g) = k.group.filter(|g| *g > info.max_group) {
        return Some(format!(
            "kernel '{}' asks for groups of {g} threads (group={g}), '{}' ({}) accepts at most {}: \
             lower group= or choose another device",
            k.name,
            info.name,
            info.id(),
            info.max_group
        ));
    }
    let shared = k.shared_bytes() as u64;
    if shared > info.shared_bytes {
        return Some(format!(
            "kernel '{}' uses {shared} bytes of gs.shared per group, '{}' ({}) offers {}: \
             shrink the shared arrays",
            k.name,
            info.name,
            info.id(),
            info.shared_bytes
        ));
    }
    if let Some(op) = req.f64_math_op.filter(|_| !info.f64_math) {
        return Some(format!(
            "kernel '{}' uses {op} in f64, not available on '{}' ({}): only OpenCL, CUDA and cpu:0 \
             provide double-precision math functions",
            k.name,
            info.name,
            info.id()
        ));
    }
    None
}

/// Debugging: with `GPUSYNC_DUMP_DIR`, every compiled code is written to that folder
/// (`<kernel>.<variant>.<cl|cu|wgsl>`).
fn dump(k: &Kernel, wide: bool, spec: &[(usize, i64)], ext: &str, code: &str) {
    if let Some(dir) = std::env::var_os("GPUSYNC_DUMP_DIR") {
        let tag: Vec<String> = spec
            .iter()
            .map(|(p, v)| format!("{}={v}", k.params[*p].name))
            .collect();
        let name = format!(
            "{}.{}{}.{ext}",
            k.name,
            if wide { "i64." } else { "" },
            if tag.is_empty() {
                "generique".into()
            } else {
                tag.join(",")
            }
        );
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(std::path::Path::new(&dir).join(name), code);
    }
}

/// Generates and compiles a variant of `k` for `dev`.
fn build(dev: &Device, k: &Kernel, wide: bool, spec: &[(usize, i64)]) -> R<Imp> {
    let gen_err = |e: gpusync_codegen::CodegenError| CoreError::Unsupported(e.0);
    Ok(match dev {
        Device::Cu(d) => {
            let src = gpusync_codegen::emit_c_variant(k, true, wide, spec).map_err(gen_err)?;
            dump(k, wide, spec, "cu", &src.code);
            Imp::Cu(d.compile(&src.code, &src.entry, k.group).map_err(dev_err)?)
        }
        Device::Cl(d) => {
            let src = gpusync_codegen::emit_c_variant(k, false, wide, spec).map_err(gen_err)?;
            dump(k, wide, spec, "cl", &src.code);
            Imp::Cl(d.compile(&src.code, &src.entry, k.group).map_err(dev_err)?)
        }
        // A wgpu buffer does not exceed a few GiB: an array of more than 2³¹ elements cannot
        // be bound whole; and WGSL has no 64-bit integers.
        Device::Wg(d) if wide => {
            return Err(CoreError::Unsupported(format!(
                "'{}': an array of more than 2³¹ − 1 elements needs 64-bit indices, \
                 which WGSL lacks ({}); use CUDA, OpenCL or cpu:0",
                k.name, d.info.name
            )));
        }
        Device::Wg(d) => {
            let metal = d.info.backend == "metal";
            let src = gpusync_codegen::emit_wgsl_variant(k, spec, metal).map_err(gen_err)?;
            dump(k, wide, spec, "wgsl", &src.code);
            Imp::Wg(
                d.compile(
                    &src.code,
                    k.array_params().count(),
                    k.outputs.len(),
                    gpusync_codegen::has_f64_params(k),
                    &k.array_params()
                        .map(|(i, _)| k.writable().contains(&i))
                        .collect::<Vec<_>>(),
                    k.group,
                )
                .map_err(dev_err)?,
            )
        }
        Device::Cpu => Imp::Cpu,
    })
}

/// Checks the device capabilities (§6.2), generates the code and compiles it.
pub fn compile(dev: &Device, k: &Arc<Kernel>) -> R<Compiled> {
    let info = dev.info();
    if let Some(msg) = unsupported(&info, k) {
        return Err(CoreError::Unsupported(msg));
    }
    Ok(Compiled {
        kernel: k.clone(),
        imp: build(dev, k, false, &[])?,
        variants: Default::default(),
    })
}

mod exec;
mod gpu;
mod window;

pub use exec::{
    DeviceArray, Input, Mode, Storage, TransferLimits, launch, set_transfer_limits, transfer_limits,
};
pub use window::{Window, WindowEvent};
