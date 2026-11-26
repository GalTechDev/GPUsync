//! Transfer engine (ARCHITECTURE.md §9): a single execution path, planned in chunks.
//!
//! | Mode | When | Transfers |
//! |---|---|---|
//! | Direct | OpenCL on unified memory | none: the GPU reads and writes host memory |
//! | Resident | discrete GPU (or wgpu), the call fits in the budget | inputs uploaded in chunks; result kept on the device |
//! | Flux | beyond the budget or a buffer size | chunk by chunk, transfers overlapped with computation; result written straight to its destination |
//!
//! In every mode, the extra host memory is bounded (ring of 3 chunks): no full
//! copy of the data is ever made in RAM.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use gpusync_backend_cuda::{CuBuffer, CuEvent};
use gpusync_backend_opencl::{ClBuffer, ClDevice, ClEvent, ClKernel, wait};

use crate::gpu::{GArg, Gpu};
use gpusync_backend_wgpu::{FluxIn, WgBuffer};
use gpusync_ir::interp::{self, Arg};
use gpusync_ir::{Const, Grid, Kernel, ParamKind, Scalar};
use gpusync_runtime::Backend;

use crate::{Compiled, CoreError, Device, Imp, R, dev_err};

/// Settings of the transfer engine (§9.2, §9.3), adjustable (tests, `gs.set_memory_budget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferLimits {
    /// Target size of a transferred chunk, per array (bytes).
    pub chunk_bytes: usize,
    /// Memory budget of the device (bytes). `None`: 50 % of its memory.
    pub budget_bytes: Option<u64>,
    /// Largest chunk in Direct mode (bytes). `None`: largest allocation.
    pub direct_chunk_bytes: Option<usize>,
}

static LIMITS: RwLock<TransferLimits> = RwLock::new(TransferLimits {
    chunk_bytes: 64 << 20,
    budget_bytes: None,
    direct_chunk_bytes: None,
});

pub fn transfer_limits() -> TransferLimits {
    *LIMITS.read().unwrap_or_else(|e| e.into_inner())
}

pub fn set_transfer_limits(l: TransferLimits) {
    *LIMITS.write().unwrap_or_else(|e| e.into_inner()) = l;
}

/// Execution mode chosen for a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Direct,
    Resident,
    Flux,
    Cpu,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Direct => "direct",
            Mode::Resident => "resident",
            Mode::Flux => "flux",
            Mode::Cpu => "cpu",
        }
    }
}

/// Where the result of a kernel lives.
pub enum Storage {
    /// Direct mode (OpenCL, unified memory): chunks backed by host memory `ptr`.
    /// `done`: end of the last launch (another kernel can read the result without waiting for the host).
    ClDirect {
        bufs: Vec<ClBuffer>,
        ptr: *mut u8,
        done: Option<Arc<ClEvent>>,
    },
    /// Resident mode (OpenCL): in device memory; `done` signals the end of the computation.
    Cl { buf: ClBuffer, done: Arc<ClEvent> },
    /// Resident mode (CUDA).
    Cu { buf: CuBuffer, done: Arc<CuEvent> },
    /// Resident mode (wgpu).
    Wg(WgBuffer),
    /// Direct mode (Metal, `newBufferWithBytesNoCopy`): chunks backed by host memory `ptr`.
    WgDirect { bufs: Vec<WgBuffer>, ptr: *mut u8 },
    /// Already in host memory (Flux mode, interpreter).
    Host { ptr: *mut u8 },
    /// Host memory allocated lazily on the Rust side (Flux / CPU fallback if the caller did not pre-allocate).
    HostOwned(Vec<u8>),
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

/// Result of a kernel.
pub struct DeviceArray {
    pub device: Device,
    pub storage: Storage,
    pub dtype: Scalar,
    pub len: usize,
    pub mode: Mode,
    /// Buffers still use host memory: wait for the computations before
    /// the caller frees it (set back to false as soon as `read_into` has synchronized).
    host_backed_io: AtomicBool,
    /// A later kernel reads this Direct result's host memory directly (zero-copy): wait for the
    /// computations before freeing it. Carried by the source, not by the later kernel's results:
    /// those keep it alive and can be released without waiting.
    read_in_place: AtomicBool,
}

impl DeviceArray {
    pub fn bytes(&self) -> usize {
        self.len * self.dtype.size()
    }

    /// Host pointer if the result is (or will be) directly in host memory.
    pub fn host_ptr(&self) -> Option<*mut u8> {
        match &self.storage {
            Storage::ClDirect { ptr, .. }
            | Storage::WgDirect { ptr, .. }
            | Storage::Host { ptr } => Some(*ptr),
            Storage::HostOwned(v) => Some(v.as_ptr() as *mut u8),
            _ => None,
        }
    }

    /// Waits for the computation and puts the result at `dst` (`bytes()` bytes). If `dst` already is the
    /// result's memory (Direct, Flux), only the synchronization happens.
    ///
    /// # Safety
    /// `dst` must point to `bytes()` writable bytes.
    pub unsafe fn read_into(&self, dst: *mut u8) -> R<()> {
        let n = self.bytes();
        let chunk = transfer_limits().chunk_bytes;
        match (&self.storage, &self.device) {
            (Storage::ClDirect { bufs, ptr, .. }, Device::Cl(d)) => {
                for b in bufs {
                    d.sync_host(b).map_err(dev_err)?;
                }
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
            (Storage::Cl { buf, done }, Device::Cl(d)) => unsafe {
                q_download(&**d, buf, Some(done), dst, n, chunk)?
            },
            (Storage::Cu { buf, done }, Device::Cu(d)) => unsafe {
                q_download(&**d, buf, Some(done), dst, n, chunk)?
            },
            (Storage::Wg(buf), Device::Wg(d)) => unsafe {
                d.download(buf, 0, dst, n, chunk).map_err(dev_err)?
            },
            (Storage::WgDirect { ptr, .. }, Device::Wg(d)) => {
                d.finish().map_err(dev_err)?;
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
            (Storage::Host { ptr }, _) => {
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
            (Storage::HostOwned(v), _) => {
                if !std::ptr::eq(dst, v.as_ptr()) {
                    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), dst, n) };
                }
            }
            _ => {
                return Err(CoreError::Device(
                    "result inconsistent with its device".into(),
                ));
            }
        }
        self.host_backed_io.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        if *self.host_backed_io.get_mut() || *self.read_in_place.get_mut() {
            let _ = self.device.finish();
        }
    }
}

/// An input array.
pub enum Input<'a> {
    /// Host memory (contiguous NumPy array, memmap…), kept alive by the caller.
    Host { ptr: *const u8, bytes: usize },
    /// Result of a previous kernel.
    Device(&'a DeviceArray),
}

/// Input resolved for the target device.
enum In<'a> {
    Host(*const u8, usize),
    Dev(&'a DeviceArray),
}

impl In<'_> {
    fn bytes(&self) -> usize {
        match self {
            In::Host(_, b) => *b,
            In::Dev(a) => a.bytes(),
        }
    }
}

/// Launches `c` over a grid of `n` elements (size of each output); `outs` are the
/// final destinations of the outputs (`n` elements each), in kernel order.
///
/// Element-wise kernel: each input has `n` elements. Indexed kernel: inputs have
/// their own size (dimensions are passed as hidden parameters, in `scalars`).
///
/// # Safety
/// Host pointers must stay valid until the computation ends (results read or
/// returned `DeviceArray`s dropped).
pub unsafe fn launch(
    dev: &Device,
    c: &Compiled,
    inputs: &[Input],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
) -> R<Vec<DeviceArray>> {
    let k = &c.kernel;
    let arrays: Vec<_> = k.array_params().map(|(_, p)| p.ty).collect();
    let n_engine = k.engine_params().iter().flatten().count();
    let n_scalars = k.params.len() - arrays.len() - n_engine;
    if inputs.len() != arrays.len() || scalars.len() != n_scalars {
        return Err(CoreError::Type(format!(
            "'{}' expects {} array(s) and {} scalar(s)",
            k.name,
            arrays.len(),
            n_scalars
        )));
    }
    if outs.len() != k.outputs.len() {
        return Err(CoreError::Type(format!(
            "'{}' has {} output(s), {} destination(s) given",
            k.name,
            k.outputs.len(),
            outs.len()
        )));
    }
    let indexed = matches!(k.grid, Grid::Indexed { .. });
    if let Some(g) = k.group {
        let tile = g as usize * k.cells.max(1) as usize;
        if !n.is_multiple_of(tile) {
            return Err(CoreError::Shape(format!(
                "'{}': group={g}{}, but the output has {n} cells, which do not split into whole tiles of {tile} \
                 (choose dimensions whose product is a multiple of {tile}, e.g. a last dimension that is a multiple of {tile})",
                k.name,
                if k.cells > 1 {
                    format!(", cells={}", k.cells)
                } else {
                    String::new()
                }
            )));
        }
    }
    for (input, (_, p)) in inputs.iter().zip(k.array_params()) {
        let bytes = match input {
            Input::Host { bytes, .. } => *bytes,
            Input::Device(a) => a.bytes(),
        };
        let ok = if indexed {
            bytes % p.ty.size() == 0
        } else {
            bytes == n * p.ty.size()
        };
        if !ok {
            return Err(CoreError::Shape(format!(
                "array '{}': {bytes} bytes for {} elements of type {}",
                p.name,
                if indexed {
                    "its".to_string()
                } else {
                    n.to_string()
                },
                p.ty
            )));
        }
    }

    // Resolving the inputs: a result already in host memory is read in place; a result
    // left on another device goes back through the host.
    let mut staged: Vec<Vec<u8>> = Vec::new();
    let mut idx_staged: Vec<Option<usize>> = Vec::new();
    for i in inputs {
        if let Input::Device(a) = i {
            if a.host_ptr().is_none() && a.device.id() != dev.id() {
                let mut v = vec![0u8; a.bytes()];
                unsafe { a.read_into(v.as_mut_ptr())? };
                staged.push(v);
                idx_staged.push(Some(staged.len() - 1));
                continue;
            }
        }
        idx_staged.push(None);
    }
    let mut ins: Vec<In> = Vec::with_capacity(inputs.len());
    for (i, st) in inputs.iter().zip(&idx_staged) {
        ins.push(match (i, st) {
            (_, Some(s)) => In::Host(staged[*s].as_ptr(), staged[*s].len()),
            (Input::Host { ptr, bytes }, None) => In::Host(*ptr, *bytes),
            // Direct result of this device, in a single chunk: its buffer is reused as is (read-only
            // or copied on the GPU into a fresh buffer for gs.Out), without waiting for the host.
            (Input::Device(a), None) if reusable_direct(a, dev) => {
                a.read_in_place.store(true, Ordering::Release);
                In::Dev(a)
            }
            (Input::Device(a), None) => match a.host_ptr() {
                Some(p) => {
                    unsafe { a.read_into(p)? }; // synchronization only
                    In::Host(p, a.bytes())
                }
                None => In::Dev(a),
            },
        });
    }

    let limits = transfer_limits();
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let scalars = &full_scalars(k, scalars);
    // 64-bit indices if an array read by index exceeds 2³¹ − 1 elements (indexed kernels).
    let wide = indexed
        && ins
            .iter()
            .zip(&arrays)
            .any(|(i, t)| (i.bytes() / t.size()) as u64 > wide_index_limit());
    let imp = c.variant(dev, wide, specialization(k, scalars))?;
    let fits_dev = fits(dev, &ins, n, &out_elems, &limits);
    // If the caller skipped the host allocation (null pointers for the GPU Resident path) and
    // we end up in Flux, Direct or CPU, allocate the fallback host buffers here.
    let mut fallback_bufs: Vec<Option<Vec<u8>>> = Vec::new();
    let mut resolved_outs: Vec<*mut u8> = outs.to_vec();
    let needs_host_fallback = outs.iter().any(|p| p.is_null())
        && (n == 0
            || !fits_dev
            || matches!(dev, Device::Cpu)
            || (matches!(dev, Device::Cl(d) if d.info.unified_memory) && all_host(&ins)));
    if needs_host_fallback {
        fallback_bufs.reserve(outs.len());
        for (o, &elem) in outs.iter().zip(&out_elems) {
            if o.is_null() {
                let mut v = vec![0u8; n * elem];
                resolved_outs[fallback_bufs.len()] = v.as_mut_ptr();
                fallback_bufs.push(Some(v));
            } else {
                fallback_bufs.push(None);
            }
        }
    }
    let outs: &[*mut u8] = &resolved_outs;
    let result: Planned = if n == 0 {
        Ok((
            outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
            Mode::Cpu,
            false,
        ))
    } else if indexed {
        let rows = Rows::new(k, scalars, n, &ins, &out_elems);
        match (dev, &imp) {
            (Device::Cpu, Imp::Cpu) => unsafe { run_cpu(k, &ins, scalars, n, outs) },
            (Device::Cl(d), Imp::Cl(ck)) if d.info.unified_memory && all_host(&ins) => unsafe {
                rows_cl_direct(d, ck, k, &rows, &ins, scalars, outs, &limits)
            },
            (Device::Cl(d), Imp::Cl(ck)) => unsafe {
                rows_q(
                    &**d,
                    ck,
                    k,
                    &rows,
                    &ins,
                    scalars,
                    outs,
                    &limits,
                    fits_dev,
                    budget(dev, &limits),
                )
            },
            (Device::Cu(d), Imp::Cu(ck)) => unsafe {
                rows_q(
                    &**d,
                    ck,
                    k,
                    &rows,
                    &ins,
                    scalars,
                    outs,
                    &limits,
                    fits_dev,
                    budget(dev, &limits),
                )
            },
            (Device::Wg(d), Imp::Wg(wk)) => unsafe {
                match rows_wg_direct(d, wk, k, &rows, &ins, scalars, outs, &limits)? {
                    Some(direct) => Ok(direct),
                    None => rows_wg(
                        d,
                        wk,
                        k,
                        &rows,
                        &ins,
                        scalars,
                        outs,
                        &limits,
                        fits_dev,
                        budget(dev, &limits),
                    ),
                }
            },
            _ => Err(CoreError::Device(
                "kernel compiled for another device".into(),
            )),
        }
    } else {
        match (dev, &imp) {
            (Device::Cpu, Imp::Cpu) => unsafe { run_cpu(k, &ins, scalars, n, outs) },
            (Device::Cl(d), Imp::Cl(ck)) if d.info.unified_memory && all_host(&ins) => unsafe {
                cl_direct(d, ck, k, &ins, &arrays, scalars, n, outs, &limits)
            },
            (Device::Cl(d), Imp::Cl(ck)) => {
                if fits_dev {
                    unsafe { q_resident(&**d, ck, k, &ins, scalars, n, &limits) }
                } else {
                    unsafe { q_flux(&**d, ck, k, &ins, &arrays, scalars, n, outs, &limits) }
                }
            }
            (Device::Cu(d), Imp::Cu(ck)) => {
                if fits_dev {
                    unsafe { q_resident(&**d, ck, k, &ins, scalars, n, &limits) }
                } else {
                    unsafe { q_flux(&**d, ck, k, &ins, &arrays, scalars, n, outs, &limits) }
                }
            }
            (Device::Wg(d), Imp::Wg(wk)) => {
                let (words, fwords) = wg_params(k, scalars)?;
                if let Some(direct) = unsafe {
                    wg_direct(
                        d, wk, &ins, &arrays, &words, &fwords, n, outs, &out_elems, &limits,
                    )?
                } {
                    Ok(direct)
                } else if fits_dev {
                    unsafe { wg_resident(d, wk, &ins, &words, &fwords, n, &out_elems, &limits) }
                } else {
                    let flux_in: Vec<FluxIn> = ins
                        .iter()
                        .zip(&arrays)
                        .map(|(i, ty)| match i {
                            In::Host(p, _) => Ok(FluxIn::Host {
                                ptr: *p,
                                elem: ty.size(),
                            }),
                            In::Dev(a) => match wg_buf(&a.storage) {
                                Some(b) => Ok(FluxIn::Dev {
                                    buf: b,
                                    elem: ty.size(),
                                }),
                                None => Err(other_kind()),
                            },
                        })
                        .collect::<R<_>>()?;
                    let elem_max = elem_max(&arrays, &out_elems);
                    let max_buf = d.info.max_buffer_bytes.unwrap_or(u64::MAX) as usize;
                    let chunk_elems = (limits.chunk_bytes.min(max_buf) / elem_max).max(1);
                    let fouts: Vec<(*mut u8, usize)> = outs
                        .iter()
                        .copied()
                        .zip(out_elems.iter().copied())
                        .collect();
                    unsafe {
                        d.flux(wk, &flux_in, &fouts, &words, &fwords, n, chunk_elems)
                            .map_err(dev_err)?
                    };
                    Ok((
                        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
                        Mode::Flux,
                        false,
                    ))
                }
            }
            _ => Err(CoreError::Device(
                "kernel compiled for another device".into(),
            )),
        }
    };
    let (mut storages, mode, mut host_backed_io) = result?;
    // If fallback host buffers were allocated on the Rust side, move their ownership into Storage::HostOwned.
    if !fallback_bufs.is_empty() {
        if host_backed_io {
            dev.finish()?;
            host_backed_io = false;
        }
        for (st, fb) in storages.iter_mut().zip(fallback_bufs) {
            if let Some(v) = fb {
                *st = Storage::HostOwned(v);
            }
        }
    }
    // Inputs brought back from another device: local copies freed on return.
    if !staged.is_empty() && host_backed_io {
        dev.finish()?;
        host_backed_io = false;
    }
    // Results: the outputs (n cells each), then the gs.Out arrays of the explicit mode.
    let mut shapes: Vec<(Scalar, usize)> = k.outputs.iter().map(|t| (*t, n)).collect();
    for (pos, (i, p)) in k.array_params().enumerate() {
        if k.writable().contains(&i) {
            shapes.push((p.ty, ins[pos].bytes() / p.ty.size()));
        }
    }
    Ok(storages
        .into_iter()
        .zip(shapes)
        .map(|(storage, (ty, len))| DeviceArray {
            device: dev.clone(),
            storage,
            dtype: ty,
            len,
            mode,
            host_backed_io: AtomicBool::new(host_backed_io),
            read_in_place: AtomicBool::new(false),
        })
        .collect())
}

fn host_ptr(i: &In) -> *mut u8 {
    match i {
        In::Host(p, _) => *p as *mut u8,
        In::Dev(_) => unreachable!("gs.Out: always in host memory"),
    }
}

/// Host storage of the gs.Out arrays (explicit mode), in parameter order.
fn writable_host(k: &Kernel, ins: &[In]) -> Vec<Storage> {
    k.array_params()
        .zip(ins)
        .filter(|((i, _), _)| k.writable().contains(i))
        .map(|(_, input)| match input {
            In::Host(p, _) => Storage::Host { ptr: *p as *mut u8 },
            In::Dev(_) => unreachable!("gs.Out: always in host memory"),
        })
        .collect()
}

/// Sizes replaced by constants for this call (§5.4): dimensions of an indexed kernel that
/// are not grid axes (e.g. the length `t` of series) and are at most 1024. The grid
/// axes stay parameters: changing their size recompiles nothing.
/// `GPUSYNC_SPECIALIZE=0` disables specialization (measurements).
fn specialization(k: &Kernel, scalars: &[Const]) -> Vec<(usize, i64)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("GPUSYNC_SPECIALIZE").map_or(true, |v| v != "0"));
    let Grid::Indexed { axes, .. } = &k.grid else {
        return Vec::new();
    };
    if !on {
        return Vec::new();
    }
    let mut pos = 0;
    let mut out = Vec::new();
    for (i, p) in k.params.iter().enumerate() {
        if p.kind != ParamKind::Scalar {
            continue;
        }
        if p.hidden && p.name.starts_with("gs_dim_") && !axes.contains(&i) {
            if let Const::I32(v) = interp::convert(scalars[pos], Scalar::I32) {
                if (1..=1024).contains(&v) {
                    out.push((i, v as i64));
                }
            }
        }
        pos += 1;
    }
    out
}

/// Number of elements beyond which an array needs 64-bit indices: 2³¹ − 1, or the value of
/// `GPUSYNC_WIDE_INDEX_LIMIT` (tests: force the wide variant on small arrays).
fn wide_index_limit() -> u64 {
    static L: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("GPUSYNC_WIDE_INDEX_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(i32::MAX as u64)
    })
}

/// Storage of each output, chosen mode, buffers still backed by host memory.
type Planned = R<(Vec<Storage>, Mode, bool)>;

/// wgpu buffer of a result that another kernel can read: Resident, or Direct in a single chunk (the
/// `NoCopy` buffer already created on host memory is reused as is; wgpu's single queue
/// orders both computations, without waiting for the host).
fn wg_buf(s: &Storage) -> Option<&WgBuffer> {
    match s {
        Storage::Wg(b) => Some(b),
        Storage::WgDirect { bufs, .. } => bufs.first(),
        _ => None,
    }
}

/// Direct result of `dev` covered by a single buffer (the whole array): reusable as
/// the input of a kernel on the same device.
fn reusable_direct(a: &DeviceArray, dev: &Device) -> bool {
    if a.device.id() != dev.id() {
        return false;
    }
    match &a.storage {
        Storage::WgDirect { bufs, .. } => bufs.first().is_some_and(|b| b.bytes >= a.bytes()),
        Storage::ClDirect {
            bufs,
            done: Some(_),
            ..
        } => bufs.first().is_some_and(|b| b.bytes >= a.bytes()),
        _ => false,
    }
}

/// All inputs in host memory: Direct mode possible on unified memory. Otherwise (a chain of
/// kernels), results stay in reused device buffers (Resident mode).
fn all_host(ins: &[In]) -> bool {
    ins.iter().all(|i| matches!(i, In::Host(..)))
}

fn other_kind() -> CoreError {
    CoreError::Device("input on another kind of device".into())
}

fn elem_max(arrays: &[Scalar], out_elems: &[usize]) -> usize {
    arrays
        .iter()
        .map(|t| t.size())
        .chain(out_elems.iter().copied())
        .max()
        .unwrap_or(4)
}

/// Device memory for the budget: known with OpenCL; for wgpu, that of the same GPU as seen
/// by OpenCL if it exists, otherwise 4 GiB (cautious). Cached: avoids an OpenCL FFI enumeration
/// at every wgpu kernel launch.
fn device_memory(dev: &Device) -> u64 {
    let info = dev.info();
    if let Some(m) = info.memory_bytes {
        return m;
    }
    static CL_MEM_BY_NAME: OnceLock<HashMap<String, u64>> = OnceLock::new();
    let map = CL_MEM_BY_NAME.get_or_init(|| {
        gpusync_backend_opencl::OpenClBackend
            .devices()
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|d| d.memory_bytes.map(|m| (d.name, m)))
            .collect()
    });
    map.get(&info.name).copied().unwrap_or(4 << 30)
}

fn budget(dev: &Device, limits: &TransferLimits) -> u64 {
    limits
        .budget_bytes
        .unwrap_or_else(|| device_memory(dev) / 2)
}

/// Does the call fit in Resident mode (budget, largest buffer, a single grid)?
fn fits(dev: &Device, ins: &[In], n: usize, out_elems: &[usize], limits: &TransferLimits) -> bool {
    let budget = limits.budget_bytes.unwrap_or(device_memory(dev) / 2);
    let max_buf = dev.info().max_buffer_bytes.unwrap_or(u64::MAX);
    let outs = out_elems.iter().map(|e| (n * e) as u64);
    let biggest = ins
        .iter()
        .map(|i| i.bytes() as u64)
        .chain(outs.clone())
        .max()
        .unwrap_or(0);
    let new_bytes: u64 = ins
        .iter()
        .filter(|i| matches!(i, In::Host(..)))
        .map(|i| i.bytes() as u64)
        .sum::<u64>()
        + outs.sum::<u64>();
    n <= u32::MAX as usize && new_bytes <= budget && biggest <= max_buf
}

/// Chunks `(start, length)` of the element-wise grid.
fn blocks(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    (0..n)
        .step_by(chunk.max(1))
        .map(|s| (s, chunk.min(n - s)))
        .collect()
}

/// Full list of scalars (parameter order): the caller's, and zeros in place
/// of the engine's parameters (filled in at each launch).
fn full_scalars(k: &Kernel, user: &[Const]) -> Vec<Const> {
    let mut u = user.iter();
    k.params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == ParamKind::Scalar)
        .map(|(i, p)| {
            if k.is_engine_param(i) {
                match p.ty {
                    Scalar::U32 => Const::U32(0),
                    _ => Const::I32(0),
                }
            } else {
                *u.next().unwrap()
            }
        })
        .collect()
}

fn scalar_args(k: &Kernel, scalars: &[Const]) -> Vec<Const> {
    let mut s = scalars.iter();
    k.params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar)
        .map(|p| interp::convert(*s.next().unwrap(), p.ty))
        .collect()
}

fn wg_params(k: &Kernel, scalars: &[Const]) -> R<(Vec<u32>, Vec<f64>)> {
    let (mut w, mut f) = (Vec::new(), Vec::new());
    for c in scalar_args(k, scalars) {
        match c {
            Const::I32(v) => w.push(v as u32),
            Const::U32(v) => w.push(v),
            Const::F32(v) => w.push(v.to_bits()),
            Const::Bool(v) => w.push(v as u32),
            Const::F64(v) => f.push(v),
            // 64-bit integer: two words, low then high (read back by the WGSL code).
            Const::I64(v) => w.extend([v as u64 as u32, ((v as u64) >> 32) as u32]),
            Const::U64(v) => w.extend([v as u32, (v >> 32) as u32]),
        }
    }
    Ok((w, f))
}

fn gscalar<'a, B>(c: Const) -> GArg<'a, B> {
    match c {
        Const::I32(v) => GArg::I32(v),
        Const::U32(v) => GArg::U32(v),
        Const::F32(v) => GArg::F32(v),
        Const::F64(v) => GArg::F64(v),
        Const::Bool(v) => GArg::I32(v as i32),
        Const::I64(v) => GArg::I64(v),
        Const::U64(v) => GArg::U64(v),
    }
}

/// Launch arguments (OpenCL, CUDA) in parameter order: arrays = `bufs` (in
/// order), then the outputs and n.
fn gargs<'a, B>(
    k: &Kernel,
    bufs: &[&'a B],
    scalars: &[Const],
    outs: &[&'a B],
    n: u32,
) -> Vec<GArg<'a, B>> {
    let mut b = bufs.iter();
    let mut s = scalar_args(k, scalars).into_iter();
    let mut args: Vec<GArg<B>> = k
        .params
        .iter()
        .map(|p| match p.kind {
            ParamKind::Array => GArg::Buf(*b.next().unwrap()),
            ParamKind::Scalar => gscalar(s.next().unwrap()),
        })
        .collect();
    args.extend(outs.iter().map(|o| GArg::Buf(*o)));
    args.push(GArg::U32(n));
    args
}

unsafe fn run_cpu(
    k: &Kernel,
    ins: &[In],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
) -> Planned {
    let slices: Vec<&[u8]> = ins
        .iter()
        .map(|i| match i {
            In::Host(p, b) => Ok(unsafe { std::slice::from_raw_parts(*p, *b) }),
            In::Dev(_) => Err(CoreError::Device(
                "cpu:0: input left on a device".into(),
            )),
        })
        .collect::<R<_>>()?;
    let mut sl = slices.into_iter();
    let mut sc = scalar_args(k, scalars).into_iter();
    let args: Vec<Arg> = k
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| match p.kind {
            ParamKind::Array if k.writable().contains(&i) => {
                let s = sl.next().unwrap();
                Arg::Writable(s.as_ptr() as *mut u8, s.len())
            }
            ParamKind::Array => Arg::Array(sl.next().unwrap()),
            ParamKind::Scalar => Arg::Scalar(sc.next().unwrap()),
        })
        .collect();
    let mut out_slices: Vec<&mut [u8]> = outs
        .iter()
        .zip(&k.outputs)
        .map(|(p, t)| unsafe { std::slice::from_raw_parts_mut(*p, n * t.size()) })
        .collect();
    interp::run(k, n, &args, &mut out_slices).map_err(CoreError::Device)?;
    let mut storages: Vec<Storage> = outs.iter().map(|p| Storage::Host { ptr: *p }).collect();
    storages.extend(writable_host(k, ins));
    Ok((storages, Mode::Cpu, false))
}

/// Direct mode (OpenCL, unified memory): each chunk uses host memory directly.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_direct(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let max_alloc = d
        .info
        .max_buffer_bytes
        .unwrap_or(u64::MAX)
        .min(usize::MAX as u64) as usize;
    let chunk_bytes = limits
        .direct_chunk_bytes
        .unwrap_or(max_alloc)
        .min(max_alloc);
    let chunk = (chunk_bytes / elem_max(arrays, &out_elems)).clamp(1, u32::MAX as usize);
    let mut out_bufs: Vec<Vec<ClBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    let mut last: Option<ClEvent> = None;
    for (start, len) in blocks(n, chunk) {
        let mut bufs = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            let In::Host(p, _) = i else {
                return Err(CoreError::Device(
                    "Direct mode: input left on a device".into(),
                ));
            };
            let (off, bytes) = (start * ty.size(), len * ty.size());
            bufs.push(unsafe { d.wrap_host(p.add(off) as *mut u8, bytes) }.map_err(dev_err)?);
        }
        let mut obs = Vec::new();
        for (p, e) in outs.iter().zip(&out_elems) {
            obs.push(unsafe { d.wrap_host(p.add(start * e), len * e) }.map_err(dev_err)?);
        }
        let refs: Vec<&ClBuffer> = bufs.iter().collect();
        let orefs: Vec<&ClBuffer> = obs.iter().collect();
        last = Some(
            Gpu::launch(
                d,
                ck,
                &gargs(k, &refs, scalars, &orefs, len as u32),
                len,
                &[],
            )
            .map_err(dev_err)?,
        );
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    let done = last.map(Arc::new);
    Ok((
        out_bufs
            .into_iter()
            .zip(outs)
            .map(|(bufs, ptr)| Storage::ClDirect {
                bufs,
                ptr: *ptr,
                done: done.clone(),
            })
            .collect(),
        Mode::Direct,
        true,
    ))
}

/// Below this total (inputs and outputs), Metal uses copies rather than Direct mode:
/// such a small copy costs nothing, while freeing memory read in place forces a wait for the GPU.
/// Without this threshold, a small NumPy array that happened to start on a page boundary blocked the host every frame.
const SMALL_DIRECT: usize = 64 << 10;

/// Too small for Direct mode (never the case when tests force the chunk size).
fn too_small_for_direct(ins: &[In], out_bytes: usize, limits: &TransferLimits) -> bool {
    let total = ins.iter().map(In::bytes).sum::<usize>() + out_bytes;
    limits.direct_chunk_bytes.is_none() && total < SMALL_DIRECT
}

/// Direct mode on Metal: each chunk is a `NoCopy` buffer over host memory (no copy).
/// `None` if impossible (not Metal, address not aligned on 16 KiB, input left on a
/// device, indexed kernel too large for a single buffer).
#[allow(clippy::too_many_arguments)]
unsafe fn wg_direct(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    ins: &[In],
    arrays: &[Scalar],
    words: &[u32],
    fwords: &[f64],
    n: usize,
    outs: &[*mut u8],
    out_elems: &[usize],
    limits: &TransferLimits,
) -> R<Option<(Vec<Storage>, Mode, bool)>> {
    const PAGE: usize = 16384;
    if d.info.backend != "metal"
        || outs
            .iter()
            .any(|p| p.is_null() || !(*p as usize).is_multiple_of(PAGE))
    {
        return Ok(None);
    }
    if too_small_for_direct(ins, out_elems.iter().map(|e| e * n).sum(), limits) {
        return Ok(None);
    }
    for i in ins {
        match i {
            In::Host(p, _) if (*p as usize).is_multiple_of(PAGE) => {}
            _ => return Ok(None),
        }
    }
    let max_buf = d
        .info
        .max_buffer_bytes
        .unwrap_or(u64::MAX)
        .min(usize::MAX as u64) as usize;
    let chunk_bytes = limits.direct_chunk_bytes.unwrap_or(max_buf).min(max_buf);
    let step = PAGE / 4; // chunk starts aligned on a page for 4- or 8-byte elements
    let chunk = ((chunk_bytes / elem_max(arrays, out_elems)) / step * step).max(step);
    let refused = || CoreError::Device("Metal refused part of the host memory".into());
    let mut out_bufs: Vec<Vec<WgBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    for (start, len) in blocks(n, chunk) {
        let mut bufs = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            let In::Host(p, _) = i else { unreachable!() };
            let (off, bytes) = (start * ty.size(), len * ty.size());
            match unsafe { d.wrap_host(p.add(off) as *mut u8, bytes) }.map_err(dev_err)? {
                Some(b) => bufs.push(b),
                None if start == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        let mut obs = Vec::new();
        for (p, e) in outs.iter().zip(out_elems) {
            match unsafe { d.wrap_host(p.add(start * e), len * e) }.map_err(dev_err)? {
                Some(b) => obs.push(b),
                None if start == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        let mut w = words.to_vec();
        w.push(len as u32);
        let refs: Vec<&WgBuffer> = bufs.iter().collect();
        let orefs: Vec<&WgBuffer> = obs.iter().collect();
        d.launch(wk, &refs, &orefs, &w, fwords, len)
            .map_err(dev_err)?;
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    Ok(Some((
        out_bufs
            .into_iter()
            .zip(outs)
            .map(|(bufs, ptr)| Storage::WgDirect { bufs, ptr: *ptr })
            .collect(),
        Mode::Direct,
        true,
    )))
}

/// Resident mode (wgpu): inputs uploaded in chunks, outputs kept on the device.
#[allow(clippy::too_many_arguments)]
unsafe fn wg_resident(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    ins: &[In],
    words: &[u32],
    fwords: &[f64],
    n: usize,
    out_elems: &[usize],
    limits: &TransferLimits,
) -> Planned {
    let mut owned = Vec::new();
    for i in ins {
        if let In::Host(p, bytes) = i {
            let b = d.alloc(*bytes).map_err(dev_err)?;
            unsafe {
                d.upload(&b, 0, *p, *bytes, limits.chunk_bytes)
                    .map_err(dev_err)?
            };
            owned.push(b);
        }
    }
    let mut own = owned.iter();
    let bufs: Vec<&WgBuffer> = ins
        .iter()
        .map(|i| match i {
            In::Host(..) => Ok(own.next().unwrap()),
            In::Dev(a) => wg_buf(&a.storage).ok_or_else(other_kind),
        })
        .collect::<R<_>>()?;
    let obufs: Vec<WgBuffer> = out_elems
        .iter()
        .map(|e| d.alloc(n * e).map_err(dev_err))
        .collect::<R<_>>()?;
    let orefs: Vec<&WgBuffer> = obufs.iter().collect();
    let mut w = words.to_vec();
    w.push(n as u32);
    d.launch(wk, &bufs, &orefs, &w, fwords, n)
        .map_err(dev_err)?;
    Ok((
        obufs.into_iter().map(Storage::Wg).collect(),
        Mode::Resident,
        false,
    ))
}

/// Uploads up to this size: without waiting for the host.
const SMALL_UPLOAD: usize = 1 << 20;

/// Ring of 3 pinned buffers to upload `bytes` bytes from `src` to `dst` (transfer queue).
/// Returns the event of the last write.
unsafe fn q_upload<G: Gpu>(
    d: &G,
    dst: &G::Buf,
    src: *const u8,
    bytes: usize,
    chunk: usize,
) -> R<Option<G::Ev>> {
    if bytes == 0 {
        return Ok(None);
    }
    let chunk = chunk.min(bytes).max(4);
    // Small upload: the transfer queue does it without the host waiting; the buffer goes back to the pool
    // with its event (rounded size, for reuse).
    if bytes <= SMALL_UPLOAD && chunk >= bytes {
        let p = d
            .pinned(bytes.next_power_of_two().max(4096))
            .map_err(dev_err)?;
        unsafe { std::ptr::copy_nonoverlapping(src, G::pin_ptr(&p), bytes) };
        let e = unsafe { d.write_at(dst, 0, G::pin_ptr(&p), bytes, &[]) }.map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        d.recycle_after(p, &e).map_err(dev_err)?;
        return Ok(Some(e));
    }
    let mut ring: Vec<(G::Pin, Option<G::Ev>)> = Vec::new();
    for _ in 0..3.min(bytes.div_ceil(chunk)) {
        ring.push((d.pinned(chunk).map_err(dev_err)?, None));
    }
    let mut last = None;
    let mut off = 0;
    let mut i = 0;
    while off < bytes {
        let len = chunk.min(bytes - off);
        let slot = i % ring.len();
        if let Some(e) = ring[slot].1.take() {
            G::wait(&e).map_err(dev_err)?; // the buffer is reusable
        }
        let p = &ring[slot].0;
        unsafe { std::ptr::copy_nonoverlapping(src.add(off), G::pin_ptr(p), len) };
        let e = unsafe { d.write_at(dst, off, G::pin_ptr(p), len, &[]) }.map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        ring[slot].1 = Some(e);
        off += len;
        i += 1;
    }
    for (p, e) in ring {
        if let Some(e) = e {
            G::wait(&e).map_err(dev_err)?;
            last = Some(e);
        }
        d.recycle(p);
    }
    Ok(last)
}

/// Copies a Resident result back through the pinned ring, after `done`.
unsafe fn q_download<G: Gpu>(
    d: &G,
    src: &G::Buf,
    done: Option<&G::Ev>,
    dst: *mut u8,
    bytes: usize,
    chunk: usize,
) -> R<()> {
    if bytes == 0 {
        return done.map_or(Ok(()), |e| G::wait(e).map_err(dev_err));
    }
    let chunk = chunk.min(bytes).max(4);
    let mut ring: Vec<G::Pin> = Vec::new();
    for _ in 0..3.min(bytes.div_ceil(chunk)) {
        ring.push(d.pinned(chunk).map_err(dev_err)?);
    }
    let mut pending: VecDeque<(usize, usize, usize, G::Ev)> = VecDeque::new();
    let mut off = 0;
    let mut i = 0;
    loop {
        if pending.len() == ring.len() || (off >= bytes && !pending.is_empty()) {
            let (slot, o, len, e) = pending.pop_front().unwrap();
            G::wait(&e).map_err(dev_err)?;
            unsafe { std::ptr::copy_nonoverlapping(G::pin_ptr(&ring[slot]), dst.add(o), len) };
            continue;
        }
        if off >= bytes {
            break;
        }
        let len = chunk.min(bytes - off);
        let slot = i % ring.len();
        let waits: Vec<&G::Ev> = done.into_iter().collect();
        let e = unsafe { d.read_at(src, off, G::pin_ptr(&ring[slot]), len, &waits) }
            .map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        pending.push_back((slot, off, len, e));
        off += len;
        i += 1;
    }
    for p in ring {
        d.recycle(p);
    }
    Ok(())
}

/// Resident mode (OpenCL, discrete GPU): inputs uploaded through the pinned ring, outputs kept in VRAM.
#[allow(clippy::too_many_arguments)]
unsafe fn q_resident<G: Gpu>(
    d: &G,
    ck: &G::Kern,
    k: &Kernel,
    ins: &[In],
    scalars: &[Const],
    n: usize,
    limits: &TransferLimits,
) -> Planned {
    let mut owned = Vec::new();
    let mut last: Option<G::Ev> = None;
    for i in ins {
        if let In::Host(p, bytes) = i {
            let b = d.alloc(*bytes).map_err(dev_err)?;
            if let Some(e) = unsafe { q_upload(d, &b, *p, *bytes, limits.chunk_bytes)? } {
                last = Some(e);
            }
            owned.push(b);
        }
    }
    let mut own = owned.iter();
    let bufs: Vec<&G::Buf> = ins
        .iter()
        .map(|i| match i {
            In::Host(..) => Ok(own.next().unwrap()),
            In::Dev(a) => G::resident(&a.storage)
                .map(|(b, _)| b)
                .ok_or_else(other_kind),
        })
        .collect::<R<_>>()?;
    // Inputs from a previous computation on this device: wait until they are ready.
    let mut waits: Vec<&G::Ev> = last.iter().collect();
    for i in ins {
        if let In::Dev(a) = i {
            if let Some((_, done)) = G::resident(&a.storage) {
                waits.push(done);
            }
        }
    }
    let obufs: Vec<G::Buf> = k
        .outputs
        .iter()
        .map(|t| d.alloc(n * t.size()).map_err(dev_err))
        .collect::<R<_>>()?;
    let orefs: Vec<&G::Buf> = obufs.iter().collect();
    let done = Arc::new(
        d.launch(ck, &gargs(k, &bufs, scalars, &orefs, n as u32), n, &waits)
            .map_err(dev_err)?,
    );
    Ok((
        obufs
            .into_iter()
            .map(|buf| G::storage(buf, done.clone()))
            .collect(),
        Mode::Resident,
        false,
    ))
}

/// Flux mode (OpenCL, discrete GPU, element-wise kernels): 3-slot pipeline on two
/// queues; while chunk k is computed, chunk k+1 is uploaded and chunk k−1 comes back to its
/// final destination.
#[allow(clippy::too_many_arguments)]
unsafe fn q_flux<G: Gpu>(
    d: &G,
    ck: &G::Kern,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    const SLOTS: usize = 3;
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let max_alloc = d.info().max_buffer_bytes.unwrap_or(u64::MAX) as usize;
    let chunk = (limits.chunk_bytes.min(max_alloc) / elem_max(arrays, &out_elems)).clamp(1, n);
    struct Slot<G: Gpu> {
        ins: Vec<G::Buf>,
        pins: Vec<Option<G::Pin>>,
        outs: Vec<G::Buf>,
        pin_outs: Vec<G::Pin>,
        /// Chunk in flight: start, length, one read per output.
        busy: Option<(usize, usize, Vec<G::Ev>)>,
    }
    let mut ring: Vec<Slot<G>> = Vec::new();
    for _ in 0..SLOTS.min(n.div_ceil(chunk)) {
        let mut sins = Vec::new();
        let mut pins = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            sins.push(d.alloc(chunk * ty.size()).map_err(dev_err)?);
            pins.push(match i {
                In::Host(..) => Some(d.pinned(chunk * ty.size()).map_err(dev_err)?),
                In::Dev(_) => None,
            });
        }
        let mut souts = Vec::new();
        let mut pouts = Vec::new();
        for e in &out_elems {
            souts.push(d.alloc(chunk * e).map_err(dev_err)?);
            pouts.push(d.pinned(chunk * e).map_err(dev_err)?);
        }
        ring.push(Slot {
            ins: sins,
            pins,
            outs: souts,
            pin_outs: pouts,
            busy: None,
        });
    }
    let finish = |s: &mut Slot<G>| -> R<()> {
        if let Some((start, len, evs)) = s.busy.take() {
            for (o, e) in evs.iter().enumerate() {
                G::wait(e).map_err(dev_err)?;
                let sz = out_elems[o];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        G::pin_ptr(&s.pin_outs[o]),
                        outs[o].add(start * sz),
                        len * sz,
                    )
                };
            }
        }
        Ok(())
    };
    let mut start = 0;
    let mut i = 0;
    while start < n {
        let len = chunk.min(n - start);
        let slot = i % ring.len();
        finish(&mut ring[slot])?;
        let s = &mut ring[slot];
        let mut writes: Vec<G::Ev> = Vec::new();
        for (a, (input, ty)) in ins.iter().zip(arrays).enumerate() {
            let bytes = len * ty.size();
            match input {
                In::Host(p, _) => {
                    let pin = s.pins[a].as_ref().unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            p.add(start * ty.size()),
                            G::pin_ptr(pin),
                            bytes,
                        )
                    };
                    writes.push(
                        unsafe { d.write_at(&s.ins[a], 0, G::pin_ptr(pin), bytes, &[]) }
                            .map_err(dev_err)?,
                    );
                }
                In::Dev(arr) => match G::resident(&arr.storage) {
                    Some((buf, done)) => {
                        writes.push(
                            d.copy_at(buf, start * ty.size(), &s.ins[a], 0, bytes, &[done])
                                .map_err(dev_err)?,
                        );
                    }
                    None => return Err(other_kind()),
                },
            }
        }
        d.flush().map_err(dev_err)?;
        let refs: Vec<&G::Buf> = s.ins.iter().collect();
        let orefs: Vec<&G::Buf> = s.outs.iter().collect();
        let wait_w: Vec<&G::Ev> = writes.iter().collect();
        let ek = d
            .launch(
                ck,
                &gargs(k, &refs, scalars, &orefs, len as u32),
                len,
                &wait_w,
            )
            .map_err(dev_err)?;
        let mut reads = Vec::new();
        for (o, e) in out_elems.iter().enumerate() {
            reads.push(
                unsafe { d.read_at(&s.outs[o], 0, G::pin_ptr(&s.pin_outs[o]), len * e, &[&ek]) }
                    .map_err(dev_err)?,
            );
        }
        d.flush().map_err(dev_err)?;
        s.busy = Some((start, len, reads));
        start += len;
        i += 1;
    }
    let used = ring.len();
    for j in 0..used {
        finish(&mut ring[(i + j) % used])?;
    }
    for s in ring {
        for p in s.pins.into_iter().flatten() {
            d.recycle(p);
        }
        for p in s.pin_outs {
            d.recycle(p);
        }
    }
    Ok((
        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
        Mode::Flux,
        false,
    ))
}

// ───────────────────── Indexed kernels: blocks of rows (§9.7) ─────────────────────

/// Row plan of an indexed kernel: the grid (output shape) is split along
/// its first axis. `chunked` inputs (read only at the current row) follow the same
/// split; the others stay whole on the device.
struct Rows {
    /// Rows of the output (size of the first axis).
    total: usize,
    /// Cells per row.
    rest: usize,
    /// Per input: split by rows?
    chunked: Vec<bool>,
    /// Per input: gs.Out array (explicit mode), whole and returned as a result?
    writable: Vec<bool>,
    /// Bytes per row: of each split input (0 otherwise), of each output.
    in_row: Vec<usize>,
    out_row: Vec<usize>,
    /// Positions of row0, inrow, obase in the list of scalars.
    slots: [usize; 3],
    /// Name of each input (error messages).
    names: Vec<String>,
    /// Rows per launch: a multiple of `granule` (whole groups, `group=`).
    granule: usize,
}

/// Rows of a launch are a multiple of this: each launch covers whole groups.
fn granule(k: &Kernel, rest: usize) -> usize {
    match k.group {
        Some(g) => {
            let tile = g as usize * k.cells.max(1) as usize;
            tile / gcd(tile, rest.max(1))
        }
        None => 1,
    }
}

impl Rows {
    fn new(k: &Kernel, scalars: &[Const], n: usize, ins: &[In], out_elems: &[usize]) -> Rows {
        let Grid::Indexed {
            axes,
            chunked,
            row0,
            inrow,
            obase,
            writable,
        } = &k.grid
        else {
            unreachable!()
        };
        let pos = |param: usize| {
            k.params[..param]
                .iter()
                .filter(|p| p.kind == ParamKind::Scalar)
                .count()
        };
        let total = match interp::convert(scalars[pos(axes[0])], Scalar::I32) {
            Const::I32(v) => v.max(1) as usize,
            _ => 1,
        };
        let rest = n / total;
        let arrays: Vec<(usize, String)> =
            k.array_params().map(|(i, p)| (i, p.name.clone())).collect();
        let is_chunked: Vec<bool> = arrays.iter().map(|(i, _)| chunked.contains(i)).collect();
        Rows {
            total,
            rest,
            in_row: ins
                .iter()
                .zip(&is_chunked)
                .map(|(i, c)| if *c { i.bytes() / total } else { 0 })
                .collect(),
            out_row: out_elems.iter().map(|e| rest * e).collect(),
            chunked: is_chunked,
            writable: arrays.iter().map(|(i, _)| writable.contains(i)).collect(),
            slots: [pos(*row0), pos(*inrow), pos(*obase)],
            names: arrays.into_iter().map(|(_, n)| n).collect(),
            granule: granule(k, rest),
        }
    }

    /// Scalars of a launch: rows `row0…`, split inputs starting at row `inrow`,
    /// outputs written starting at cell `obase` of their buffer.
    fn with(&self, scalars: &[Const], row0: usize, inrow: usize, obase: usize) -> Vec<Const> {
        let mut s = scalars.to_vec();
        s[self.slots[0]] = Const::I32(row0 as i32);
        s[self.slots[1]] = Const::I32(inrow as i32);
        s[self.slots[2]] = Const::U32(obase as u32);
        s
    }

    /// Largest number of bytes per row among the split arrays (inputs and outputs).
    fn max_row(&self) -> usize {
        self.in_row
            .iter()
            .chain(&self.out_row)
            .copied()
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Bytes per row, all split arrays combined.
    fn sum_row(&self) -> usize {
        (self.in_row.iter().sum::<usize>() + self.out_row.iter().sum::<usize>()).max(1)
    }

    /// Blocks of rows `(start, rows)` of at most `rows` rows.
    fn blocks(&self, rows: usize) -> Vec<(usize, usize)> {
        let g = self.granule;
        let rows = (rows.clamp(1, self.total) / g * g).max(g);
        (0..self.total)
            .step_by(rows)
            .map(|s| (s, rows.min(self.total - s)))
            .collect()
    }

    fn resident_too_big(
        &self,
        k: &Kernel,
        i: usize,
        bytes: usize,
        room: u64,
        dev: &str,
    ) -> CoreError {
        CoreError::Unsupported(format!(
            "'{}': input '{}' ({}) is read at any row of the output, so it must \
             fit whole on {dev} ({} available). For it to be split, index it \
             by the current row ({}[i, …] with i, … = gs.index()); otherwise raise gs.set_memory_budget() \
             or use another device",
            k.name,
            self.names[i],
            human(bytes as u64),
            human(room),
            self.names[i],
        ))
    }
}

/// Readable size: "1.50 GiB", "12.0 MiB", "300 B".
fn human(b: u64) -> String {
    let f = b as f64;
    let s = if b >= 1 << 30 {
        format!("{:.2} Gio", f / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} Mio", f / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} Kio", f / 1024.0)
    } else {
        format!("{b} o")
    };
    s.replace('.', ",")
}

/// Cells launched without measuring (short call) or during the first measurement.
const PROBE_CELLS: usize = 1 << 16;
/// Target duration of a launch: well below the drivers' watchdog delay (Windows TDR: ~2 s).
const TARGET_LAUNCH_S: f64 = 0.1;
/// Most cells per launch (u32 indices, wgpu grid limits).
const MAX_LAUNCH_CELLS: usize = 1 << 30;

/// Shortest timed launch for the throughput measurement to be reliable.
const MIN_TIMED_S: f64 = 0.02;

/// Splits the launches of an indexed kernel so that none lasts more than ~0.1 s (the driver
/// watchdog delay): the first ones, small then growing, are timed until the throughput is
/// measured reliably; the next ones, sized from it, are queued without waiting.
#[derive(Default)]
struct Pacer {
    /// Measured throughput (rows per second), and whether it is reliable (a launch of at least MIN_TIMED_S).
    rate: Option<f64>,
    calibrated: bool,
    last: usize,
    /// (kernel, device) and cells per row: the throughput is kept from one call to the next.
    key: Option<((usize, usize), usize)>,
    /// Rows per launch are a multiple of this (whole groups).
    granule: usize,
}

/// Measured throughputs (cells per second, reliable?) per (kernel, device): without them, each call
/// would time its first launches again, waiting for them to finish (no overlap
/// between successive calls, e.g. a loop of 240 NTT stages).
type RateMap = HashMap<(usize, usize), (f64, bool)>;

static RATES: std::sync::LazyLock<Mutex<RateMap>> = std::sync::LazyLock::new(Default::default);

impl Pacer {
    fn for_kernel<D>(k: &Kernel, d: &D, rest: usize) -> Pacer {
        let key = (k as *const Kernel as usize, d as *const D as usize);
        let rest = rest.max(1);
        let known = RATES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .copied();
        Pacer {
            rate: known.map(|(cells, _)| cells / rest as f64),
            calibrated: known.is_some_and(|(_, c)| c),
            last: known.map_or(0, |(cells, _)| {
                ((cells / rest as f64) * MIN_TIMED_S) as usize
            }),
            key: Some((key, rest)),
            granule: granule(k, rest),
        }
    }

    /// Computes rows `[a, b)`; `step(first row, rows, wait for the end)`.
    fn run(
        &mut self,
        a: usize,
        b: usize,
        rest: usize,
        step: &mut dyn FnMut(usize, usize, bool) -> R<()>,
    ) -> R<()> {
        let rest = rest.max(1);
        let max_rows = (MAX_LAUNCH_CELLS / rest).max(1);
        if self.rate.is_none() && (b - a) * rest <= PROBE_CELLS {
            return step(a, b - a, false);
        }
        let mut s = a;
        while s < b {
            let want = match self.rate {
                None => (PROBE_CELLS / rest).max(1),
                Some(r) if self.calibrated => ((r * TARGET_LAUNCH_S) as usize).max(1),
                Some(r) => ((r * TARGET_LAUNCH_S) as usize).clamp(1, self.last.saturating_mul(8)),
            };
            let g = self.granule.max(1);
            let len = (want.min(max_rows) / g * g).max(g).min(b - s);
            let timed = !self.calibrated && s + len < b;
            let t0 = std::time::Instant::now();
            step(s, len, timed)?;
            if timed {
                let dt = t0.elapsed().as_secs_f64().max(1e-6);
                self.rate = Some(len as f64 / dt);
                self.calibrated = dt >= MIN_TIMED_S;
                self.last = len;
            }
            s += len;
        }
        Ok(())
    }
}

impl Drop for Pacer {
    fn drop(&mut self) {
        if let (Some((key, rest)), Some(r)) = (self.key, self.rate) {
            let mut g = RATES.lock().unwrap_or_else(|e| e.into_inner());
            // A reliable measurement is not replaced by a less reliable one.
            if self.calibrated || !g.get(&key).is_some_and(|(_, c)| *c) {
                g.insert(key, (r * rest as f64, self.calibrated));
            }
        }
    }
}

fn max_buffer(info_max: Option<u64>) -> usize {
    info_max.unwrap_or(u64::MAX).min(usize::MAX as u64) as usize
}

/// Indexed kernel, OpenCL on unified memory: each block of rows uses host memory
/// directly; inputs read at every row are imported once.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_cl_direct(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    let max_alloc = max_buffer(d.info.max_buffer_bytes);
    let cap = limits
        .direct_chunk_bytes
        .unwrap_or(max_alloc)
        .min(max_alloc);
    let mut whole: Vec<Option<ClBuffer>> = Vec::new();
    for (i, input) in ins.iter().enumerate() {
        let In::Host(p, bytes) = input else {
            return Err(CoreError::Device(
                "Direct mode: input left on a device".into(),
            ));
        };
        whole.push(if rows.chunked[i] {
            None
        } else {
            if *bytes > max_alloc {
                return Err(rows.resident_too_big(k, i, *bytes, max_alloc as u64, &d.info.name));
            }
            Some(unsafe { d.wrap_host(*p as *mut u8, *bytes) }.map_err(dev_err)?)
        });
    }
    let mut pacer = Pacer::for_kernel(k, d, rows.rest);
    let mut out_bufs: Vec<Vec<ClBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    let mut last: Option<ClEvent> = None;
    for (r0, len) in rows.blocks(cap / rows.max_row()) {
        let mut parts: Vec<Option<ClBuffer>> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            parts.push(match input {
                In::Host(p, _) if rows.chunked[i] => {
                    let rb = rows.in_row[i];
                    Some(
                        unsafe { d.wrap_host(p.add(r0 * rb) as *mut u8, len * rb) }
                            .map_err(dev_err)?,
                    )
                }
                _ => None,
            });
        }
        let mut obs = Vec::new();
        for (p, rb) in outs.iter().zip(&rows.out_row) {
            obs.push(unsafe { d.wrap_host(p.add(r0 * rb), len * rb) }.map_err(dev_err)?);
        }
        let refs: Vec<&ClBuffer> = parts
            .iter()
            .zip(&whole)
            .map(|(a, b)| a.as_ref().or(b.as_ref()).unwrap())
            .collect();
        let orefs: Vec<&ClBuffer> = obs.iter().collect();
        pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
            let cells = l * rows.rest / k.cells.max(1) as usize; // threads (cells / cells per thread)
            let e = Gpu::launch(
                d,
                ck,
                &gargs(k, &refs, &sc, &orefs, cells as u32),
                cells,
                &[],
            )
            .map_err(dev_err)?;
            if sync {
                wait(&e).map_err(dev_err)?;
            }
            last = Some(e);
            Ok(())
        })?;
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
        // The block's input buffers can be released: the queue keeps its references.
    }
    let done = last.map(Arc::new);
    let mut storages: Vec<Storage> = out_bufs
        .into_iter()
        .zip(outs)
        .map(|(bufs, ptr)| Storage::ClDirect {
            bufs,
            ptr: *ptr,
            done: done.clone(),
        })
        .collect();
    // gs.Out: imported whole, they are themselves the results.
    for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
        storages.push(Storage::ClDirect {
            bufs: whole[i].take().into_iter().collect(),
            ptr: host_ptr(&ins[i]),
            done: done.clone(),
        });
    }
    // The other whole imports must outlive the computations: attached to the first result.
    if let Some(Storage::ClDirect { bufs, .. }) = storages.first_mut() {
        bufs.extend(whole.into_iter().flatten());
    }
    Ok((storages, Mode::Direct, true))
}

/// Indexed kernel, OpenCL on a discrete GPU: Resident (everything fits) or blocks of rows, with
/// two slots to overlap the read-back of one block with the computation of the next.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_q<G: Gpu>(
    d: &G,
    ck: &G::Kern,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
    fits: bool,
    budget: u64,
) -> Planned {
    let max_buf = max_buffer(d.info().max_buffer_bytes);
    // Whole inputs (all of them if the call fits, otherwise those read at any row).
    let mut owned: Vec<Option<G::Buf>> = Vec::new();
    let mut waits_owned: Vec<G::Ev> = Vec::new();
    let mut resident_bytes = 0u64;
    for (i, input) in ins.iter().enumerate() {
        let keep_whole = fits || !rows.chunked[i];
        owned.push(match input {
            In::Host(p, bytes) if keep_whole => {
                resident_bytes += *bytes as u64;
                if *bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        *bytes,
                        budget.min(max_buf as u64),
                        &d.info().name,
                    ));
                }
                let b = d.alloc(*bytes).map_err(dev_err)?;
                if let Some(e) = unsafe { q_upload(d, &b, *p, *bytes, limits.chunk_bytes)? } {
                    waits_owned.push(e);
                }
                Some(b)
            }
            In::Dev(a) if rows.writable[i] => {
                let bytes = a.bytes();
                resident_bytes += bytes as u64;
                if bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        bytes,
                        budget.min(max_buf as u64),
                        &d.info().name,
                    ));
                }
                let (src, done) = G::resident(&a.storage).ok_or_else(other_kind)?;
                let b = d.alloc(bytes).map_err(dev_err)?;
                if bytes > 0 {
                    let e = d.copy_at(src, 0, &b, 0, bytes, &[done]).map_err(dev_err)?;
                    waits_owned.push(e);
                }
                Some(b)
            }
            _ => None,
        });
    }
    let dev_buf = |i: usize| -> R<(&G::Buf, &G::Ev)> {
        match &ins[i] {
            In::Dev(a) => G::resident(&a.storage).ok_or_else(other_kind),
            In::Host(..) => unreachable!(),
        }
    };
    let mut pacer = Pacer::for_kernel(k, d, rows.rest);

    if fits {
        let mut refs: Vec<&G::Buf> = Vec::new();
        let mut waits: Vec<&G::Ev> = waits_owned.iter().collect();
        for (i, o) in owned.iter().enumerate() {
            match o {
                Some(b) => refs.push(b),
                None => {
                    let (b, done) = dev_buf(i)?;
                    refs.push(b);
                    waits.push(done);
                }
            }
        }
        let obufs: Vec<G::Buf> = rows
            .out_row
            .iter()
            .map(|rb| d.alloc(rows.total * rb).map_err(dev_err))
            .collect::<R<_>>()?;
        let orefs: Vec<&G::Buf> = obufs.iter().collect();
        let mut last: Option<G::Ev> = None;
        pacer.run(0, rows.total, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, 0, s * rows.rest);
            let cells = l * rows.rest / k.cells.max(1) as usize; // threads (cells / cells per thread)
            let w: &[&G::Ev] = if last.is_none() { &waits } else { &[] };
            let e = d
                .launch(ck, &gargs(k, &refs, &sc, &orefs, cells as u32), cells, w)
                .map_err(dev_err)?;
            if sync {
                G::wait(&e).map_err(dev_err)?;
            }
            last = Some(e);
            Ok(())
        })?;
        drop(refs);
        drop(orefs);
        let done = Arc::new(last.unwrap());
        let mut storages: Vec<Storage> = obufs
            .into_iter()
            .map(|buf| G::storage(buf, done.clone()))
            .collect();
        for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
            storages.push(G::storage(owned[i].take().unwrap(), done.clone()));
        }
        return Ok((storages, Mode::Resident, false));
    }

    // By blocks of rows: two slots, each with its buffers of split inputs and outputs.
    let room = budget.saturating_sub(resident_bytes) as usize;
    let per_row: Vec<usize> = rows.in_row.iter().chain(&rows.out_row).copied().collect();
    let by_buf = per_row
        .iter()
        .filter(|b| **b > 0)
        .map(|b| max_buf / b)
        .min()
        .unwrap_or(rows.total);
    let br = (limits.chunk_bytes / rows.max_row())
        .min(room / (2 * rows.sum_row()))
        .min(by_buf)
        .clamp(1, rows.total);
    struct Slot<G: Gpu> {
        ins: Vec<Option<G::Buf>>,
        outs: Vec<G::Buf>,
        pending: Option<(usize, usize, G::Ev)>,
    }
    let mut slots: Vec<Slot<G>> = Vec::new();
    for _ in 0..2.min(rows.total.div_ceil(br)) {
        slots.push(Slot {
            ins: rows
                .in_row
                .iter()
                .zip(&rows.chunked)
                .map(|(rb, c)| c.then(|| d.alloc(br * rb)).transpose().map_err(dev_err))
                .collect::<R<_>>()?,
            outs: rows
                .out_row
                .iter()
                .map(|rb| d.alloc(br * rb).map_err(dev_err))
                .collect::<R<_>>()?,
            pending: None,
        });
    }
    let mut last_ev: Option<G::Ev> = None;
    let mut drain = |s: &mut Slot<G>| -> R<()> {
        if let Some((r0, len, done)) = s.pending.take() {
            // The block's computation must be finished before reusing the slot, even with
            // no output to read back (explicit mode: only the whole gs.Out are written).
            G::wait(&done).map_err(dev_err)?;
            for (o, buf) in s.outs.iter().enumerate() {
                let rb = rows.out_row[o];
                unsafe {
                    q_download(
                        d,
                        buf,
                        Some(&done),
                        outs[o].add(r0 * rb),
                        len * rb,
                        limits.chunk_bytes,
                    )?
                };
            }
            last_ev = Some(done);
        }
        Ok(())
    };
    for (b, (r0, len)) in rows.blocks(br).into_iter().enumerate() {
        let cur = b % slots.len();
        let mut writes: Vec<G::Ev> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            if !rows.chunked[i] {
                continue;
            }
            let rb = rows.in_row[i];
            let dst = slots[cur].ins[i].as_ref().unwrap();
            match input {
                In::Host(p, _) => {
                    if let Some(e) =
                        unsafe { q_upload(d, dst, p.add(r0 * rb), len * rb, limits.chunk_bytes)? }
                    {
                        writes.push(e);
                    }
                }
                In::Dev(_) => {
                    let (src, done) = dev_buf(i)?;
                    writes.push(
                        d.copy_at(src, r0 * rb, dst, 0, len * rb, &[done])
                            .map_err(dev_err)?,
                    );
                }
            }
        }
        d.flush().map_err(dev_err)?;
        let mut waits: Vec<&G::Ev> = writes.iter().chain(&waits_owned).collect();
        for i in 0..ins.len() {
            if !rows.chunked[i] && owned[i].is_none() {
                waits.push(dev_buf(i)?.1);
            }
        }
        let refs: Vec<&G::Buf> = (0..ins.len())
            .map(|i| match (&slots[cur].ins[i], &owned[i]) {
                (Some(b), _) | (None, Some(b)) => Ok(b),
                (None, None) => dev_buf(i).map(|(b, _)| b),
            })
            .collect::<R<_>>()?;
        let orefs: Vec<&G::Buf> = slots[cur].outs.iter().collect();
        let mut last: Option<G::Ev> = None;
        pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
            let cells = l * rows.rest / k.cells.max(1) as usize; // threads (cells / cells per thread)
            let w: &[&G::Ev] = if last.is_none() { &waits } else { &[] };
            let e = d
                .launch(ck, &gargs(k, &refs, &sc, &orefs, cells as u32), cells, w)
                .map_err(dev_err)?;
            d.flush().map_err(dev_err)?;
            if sync {
                G::wait(&e).map_err(dev_err)?;
            }
            last = Some(e);
            Ok(())
        })?;
        drop(refs);
        drop(orefs);
        // During this computation, the previous block comes back to its destination.
        let prev = (cur + slots.len() - 1) % slots.len();
        if prev != cur {
            drain(&mut slots[prev])?;
        }
        slots[cur].pending = Some((r0, len, last.unwrap()));
    }
    for s in &mut slots {
        drain(s)?;
    }
    // gs.Out: kept whole across every block, copied back (if host) or kept resident (if GPU).
    d.finish().map_err(dev_err)?;
    let mut storages: Vec<Storage> = outs.iter().map(|p| Storage::Host { ptr: *p }).collect();
    let done_arc = last_ev.map(Arc::new);
    for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
        match &ins[i] {
            In::Host(p, bytes) => {
                let buf = owned[i].as_ref().unwrap();
                unsafe { q_download(d, buf, None, *p as *mut u8, *bytes, limits.chunk_bytes)? };
                storages.push(Storage::Host { ptr: *p as *mut u8 });
            }
            In::Dev(_) => {
                storages.push(G::storage(
                    owned[i].take().unwrap(),
                    done_arc.as_ref().unwrap().clone(),
                ));
            }
        }
    }
    Ok((storages, Mode::Flux, false))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Indexed kernel on Metal: blocks of rows as `NoCopy` over host memory (block starts
/// aligned on a page). `None` if impossible (not Metal, unaligned addresses…).
#[allow(clippy::too_many_arguments)]
unsafe fn rows_wg_direct(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> R<Option<(Vec<Storage>, Mode, bool)>> {
    const PAGE: usize = 16384;
    if d.info.backend != "metal"
        || outs
            .iter()
            .any(|p| p.is_null() || !(*p as usize).is_multiple_of(PAGE))
    {
        return Ok(None);
    }
    if too_small_for_direct(ins, rows.out_row.iter().map(|b| b * rows.total).sum(), limits) {
        return Ok(None);
    }
    let max_buf = max_buffer(d.info.max_buffer_bytes);
    for i in ins {
        match i {
            In::Host(p, b) if (*p as usize).is_multiple_of(PAGE) && *b <= max_buf => {}
            _ => return Ok(None),
        }
    }
    let cap = limits.direct_chunk_bytes.unwrap_or(max_buf).min(max_buf);
    // Rows per block: a multiple of `step` so that each block starts on a page.
    let step = rows
        .in_row
        .iter()
        .chain(&rows.out_row)
        .filter(|b| **b > 0)
        .map(|b| PAGE / gcd(PAGE, *b))
        .fold(1, |acc, s| acc / gcd(acc, s) * s);
    let mut br = (cap / rows.max_row()).min(rows.total);
    if br < rows.total {
        let step = step / gcd(step, rows.granule) * rows.granule; // whole pages and groups
        br = br / step * step;
        if br == 0 {
            return Ok(None);
        }
    }
    let mut whole: Vec<Option<WgBuffer>> = Vec::new();
    for (i, input) in ins.iter().enumerate() {
        let In::Host(p, bytes) = input else {
            unreachable!()
        };
        whole.push(if rows.chunked[i] {
            None
        } else {
            match unsafe { d.wrap_host(*p as *mut u8, *bytes) }.map_err(dev_err)? {
                Some(b) => Some(b),
                None => return Ok(None),
            }
        });
    }
    let refused = || CoreError::Device("Metal refused part of the host memory".into());
    let mut pacer = Pacer::for_kernel(k, d, rows.rest);
    let mut out_bufs: Vec<Vec<WgBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    let mut keep: Vec<WgBuffer> = Vec::new();
    for (r0, len) in rows.blocks(br) {
        let mut parts: Vec<Option<WgBuffer>> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            let In::Host(p, _) = input else {
                unreachable!()
            };
            parts.push(if rows.chunked[i] {
                let rb = rows.in_row[i];
                match unsafe { d.wrap_host(p.add(r0 * rb) as *mut u8, len * rb) }
                    .map_err(dev_err)?
                {
                    Some(b) => Some(b),
                    None if r0 == 0 => return Ok(None),
                    None => return Err(refused()),
                }
            } else {
                None
            });
        }
        let mut obs = Vec::new();
        for (p, rb) in outs.iter().zip(&rows.out_row) {
            match unsafe { d.wrap_host(p.add(r0 * rb), len * rb) }.map_err(dev_err)? {
                Some(b) => obs.push(b),
                None if r0 == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        {
            let refs: Vec<&WgBuffer> = parts
                .iter()
                .zip(&whole)
                .map(|(a, b)| a.as_ref().or(b.as_ref()).unwrap())
                .collect();
            let orefs: Vec<&WgBuffer> = obs.iter().collect();
            pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
                let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
                let (mut w, f) = wg_params(k, &sc)?;
                let cells = l * rows.rest / k.cells.max(1) as usize; // threads (cells / cells per thread)
                w.push(cells as u32);
                d.launch(wk, &refs, &orefs, &w, &f, cells)
                    .map_err(dev_err)?;
                if sync {
                    d.finish().map_err(dev_err)?;
                }
                Ok(())
            })?;
        }
        keep.extend(parts.into_iter().flatten());
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    let mut storages: Vec<Storage> = out_bufs
        .into_iter()
        .zip(outs)
        .map(|(bufs, ptr)| Storage::WgDirect { bufs, ptr: *ptr })
        .collect();
    for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
        storages.push(Storage::WgDirect {
            bufs: whole[i].take().into_iter().collect(),
            ptr: host_ptr(&ins[i]),
        });
    }
    if let Some(Storage::WgDirect { bufs, .. }) = storages.first_mut() {
        bufs.extend(keep);
        bufs.extend(whole.into_iter().flatten());
    }
    Ok(Some((storages, Mode::Direct, true)))
}

/// Indexed kernel through wgpu: Resident (everything fits) or blocks of rows (a single queue:
/// upload, computation and read-back of each block follow one another).
#[allow(clippy::too_many_arguments)]
unsafe fn rows_wg(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
    fits: bool,
    budget: u64,
) -> Planned {
    let max_buf = max_buffer(d.info.max_buffer_bytes);
    let mut owned: Vec<Option<WgBuffer>> = Vec::new();
    let mut resident_bytes = 0u64;
    for (i, input) in ins.iter().enumerate() {
        let keep_whole = fits || !rows.chunked[i];
        owned.push(match input {
            In::Host(p, bytes) if keep_whole => {
                resident_bytes += *bytes as u64;
                if *bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        *bytes,
                        budget.min(max_buf as u64),
                        &d.info.name,
                    ));
                }
                let b = d.alloc(*bytes).map_err(dev_err)?;
                unsafe {
                    d.upload(&b, 0, *p, *bytes, limits.chunk_bytes)
                        .map_err(dev_err)?
                };
                Some(b)
            }
            In::Dev(a) if rows.writable[i] => {
                let bytes = a.bytes();
                resident_bytes += bytes as u64;
                if bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        bytes,
                        budget.min(max_buf as u64),
                        &d.info.name,
                    ));
                }
                let src = wg_buf(&a.storage).ok_or_else(other_kind)?;
                let b = d.alloc(bytes).map_err(dev_err)?;
                d.copy(src, 0, &b, 0, bytes).map_err(dev_err)?;
                Some(b)
            }
            _ => None,
        });
    }
    let dev_buf = |i: usize| -> R<&WgBuffer> {
        match &ins[i] {
            In::Dev(a) => wg_buf(&a.storage).ok_or_else(other_kind),
            In::Host(..) => unreachable!(),
        }
    };
    let mut pacer = Pacer::for_kernel(k, d, rows.rest);
    let mut launch = |refs: &[&WgBuffer],
                      orefs: &[&WgBuffer],
                      r0: usize,
                      a: usize,
                      b: usize,
                      whole_out: bool| {
        pacer.run(a, b, rows.rest, &mut |s, l, sync| {
            let obase = if whole_out {
                s * rows.rest
            } else {
                (s - r0) * rows.rest
            };
            let sc = rows.with(scalars, s, r0, obase);
            let (mut w, f) = wg_params(k, &sc)?;
            let cells = l * rows.rest / k.cells.max(1) as usize; // threads (cells / cells per thread)
            w.push(cells as u32);
            d.launch(wk, refs, orefs, &w, &f, cells).map_err(dev_err)?;
            if sync {
                d.finish().map_err(dev_err)?;
            }
            Ok(())
        })
    };

    if fits {
        let refs: Vec<&WgBuffer> = (0..ins.len())
            .map(|i| match &owned[i] {
                Some(b) => Ok(b),
                None => dev_buf(i),
            })
            .collect::<R<_>>()?;
        let obufs: Vec<WgBuffer> = rows
            .out_row
            .iter()
            .map(|rb| d.alloc(rows.total * rb).map_err(dev_err))
            .collect::<R<_>>()?;
        let orefs: Vec<&WgBuffer> = obufs.iter().collect();
        launch(&refs, &orefs, 0, 0, rows.total, true)?;
        drop(orefs);
        drop(refs);
        let mut storages: Vec<Storage> = obufs.into_iter().map(Storage::Wg).collect();
        for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
            storages.push(Storage::Wg(owned[i].take().unwrap()));
        }
        return Ok((storages, Mode::Resident, false));
    }

    let room = budget.saturating_sub(resident_bytes) as usize;
    let by_buf = rows
        .in_row
        .iter()
        .chain(&rows.out_row)
        .filter(|b| **b > 0)
        .map(|b| max_buf / b)
        .min()
        .unwrap_or(rows.total);
    let br = (limits.chunk_bytes / rows.max_row())
        .min(room / rows.sum_row())
        .min(by_buf)
        .clamp(1, rows.total);
    let sins: Vec<Option<WgBuffer>> = rows
        .in_row
        .iter()
        .zip(&rows.chunked)
        .map(|(rb, c)| c.then(|| d.alloc(br * rb)).transpose().map_err(dev_err))
        .collect::<R<_>>()?;
    let souts: Vec<WgBuffer> = rows
        .out_row
        .iter()
        .map(|rb| d.alloc(br * rb).map_err(dev_err))
        .collect::<R<_>>()?;
    for (r0, len) in rows.blocks(br) {
        for (i, input) in ins.iter().enumerate() {
            if !rows.chunked[i] {
                continue;
            }
            let rb = rows.in_row[i];
            let dst = sins[i].as_ref().unwrap();
            match input {
                In::Host(p, _) => unsafe {
                    d.upload(dst, 0, p.add(r0 * rb), len * rb, limits.chunk_bytes)
                        .map_err(dev_err)?
                },
                In::Dev(_) => d
                    .copy(dev_buf(i)?, r0 * rb, dst, 0, len * rb)
                    .map_err(dev_err)?,
            }
        }
        let refs: Vec<&WgBuffer> = (0..ins.len())
            .map(|i| match (&sins[i], &owned[i]) {
                (Some(b), _) | (None, Some(b)) => Ok(b),
                (None, None) => dev_buf(i),
            })
            .collect::<R<_>>()?;
        let orefs: Vec<&WgBuffer> = souts.iter().collect();
        launch(&refs, &orefs, r0, r0, r0 + len, false)?;
        for (o, buf) in souts.iter().enumerate() {
            let rb = rows.out_row[o];
            unsafe {
                d.download(buf, 0, outs[o].add(r0 * rb), len * rb, limits.chunk_bytes)
                    .map_err(dev_err)?
            };
        }
    }
    let mut storages: Vec<Storage> = outs.iter().map(|p| Storage::Host { ptr: *p }).collect();
    for i in (0..ins.len()).filter(|i| rows.writable[*i]) {
        match &ins[i] {
            In::Host(p, bytes) => {
                let buf = owned[i].as_ref().unwrap();
                unsafe {
                    d.download(buf, 0, *p as *mut u8, *bytes, limits.chunk_bytes)
                        .map_err(dev_err)?
                };
                storages.push(Storage::Host { ptr: *p as *mut u8 });
            }
            In::Dev(_) => {
                storages.push(Storage::Wg(owned[i].take().unwrap()));
            }
        }
    }
    Ok((storages, Mode::Flux, false))
}
