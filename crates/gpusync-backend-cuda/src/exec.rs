//! CUDA execution (ARCHITECTURE.md §8, §9), modelled on OpenCL:
//!
//! - **Two streams** per device: `compute` (kernels) and `transfer` (copies), chained by
//!   events: the transfers of one chunk overlap with the computation of another.
//! - **Pinned memory** (`cuMemHostAlloc`): copies from (or to) these buffers go through DMA
//!   at full speed, without blocking.
//! - **Compilation**: CUDA C → PTX with NVRTC (for the GPU's architecture, or the most recent one
//!   NVRTC knows), then PTX → machine code by the driver.
//!
//! The device's primary context is made current at the start of every operation: calls
//! may come from any thread.

use std::collections::{HashMap, VecDeque};
use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};

use crate::sys::{self, CUcontext, CUdeviceptr, CUevent, CUfunction, CUmodule, CUstream, Driver};

const BLOCK: u32 = 256;
/// Buffers kept in reserve for reuse: capacity rounded up to a power of two up to
/// 64 MiB, exact size beyond (a loop reuses the same sizes); 2 GiB in all, given back to the
/// driver if an allocation fails.
const POOL_POW2_MAX: usize = 64 << 20;
const POOL_MAX_TOTAL: usize = 2 << 30;
const POOL_MAX_BUF: usize = POOL_MAX_TOTAL / 4;

/// Reserved buffer: capacity, address, one event per stream.
type Spare = (usize, CUdeviceptr, [CUevent; 2]);

/// Pool of released buffers. `cuMemAlloc` / `cuMemFree` are expensive (`cuMemFree` synchronizes
/// the whole GPU, and fresh memory is paid for at its first write): a returned buffer is kept,
/// with one event per stream. It is preferably taken again once those events have passed;
/// otherwise right away, in stream order, like CuPy's pool: each stream waits
/// (on the GPU, without blocking the host) for the other's event, its own order doing the rest.
/// A loop that submits faster than the GPU computes thus always reuses the same buffers.
struct BufPool {
    d: &'static Driver,
    compute: CUstream,
    transfer: CUstream,
    /// Returned buffers (capacity, address, events) and total capacity.
    free: Mutex<(Vec<Spare>, usize)>,
}

unsafe impl Send for BufPool {}
unsafe impl Sync for BufPool {}

impl BufPool {
    fn take(&self, cap: usize) -> Option<CUdeviceptr> {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let passed = |evs: &[CUevent; 2]| {
            evs.iter()
                .all(|e| unsafe { (self.d.cuEventQuery)(*e) } == 0)
        };
        let ready = g.0.iter().position(|(c, _, evs)| *c == cap && passed(evs));
        let pos = ready.or_else(|| g.0.iter().position(|(c, _, _)| *c == cap))?;
        let (c, ptr, evs) = g.0.swap_remove(pos);
        g.1 -= c;
        if ready.is_none() {
            // Still used by queued commands: each stream waits for the other's.
            let ok = unsafe {
                (self.d.cuStreamWaitEvent)(self.compute, evs[1], 0) == 0
                    && (self.d.cuStreamWaitEvent)(self.transfer, evs[0], 0) == 0
            };
            if !ok {
                unsafe { (self.d.cuCtxSynchronize)() };
            }
        }
        for e in evs {
            unsafe { (self.d.cuEventDestroy_v2)(e) };
        }
        Some(ptr)
    }

    /// Keeps the buffer if there is room; otherwise releases it.
    fn give(&self, cap: usize, ptr: CUdeviceptr) {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if g.1 + cap <= POOL_MAX_TOTAL {
            let mut evs = [std::ptr::null_mut(); 2];
            let ok = [self.compute, self.transfer]
                .iter()
                .zip(evs.iter_mut())
                .all(|(s, e)| unsafe {
                    (self.d.cuEventCreate)(e, sys::EVENT_DISABLE_TIMING) == 0
                        && (self.d.cuEventRecord)(*e, *s) == 0
                });
            if ok {
                g.0.push((cap, ptr, evs));
                g.1 += cap;
                return;
            }
            for e in evs.into_iter().filter(|e| !e.is_null()) {
                unsafe { (self.d.cuEventDestroy_v2)(e) };
            }
        }
        drop(g);
        unsafe { (self.d.cuMemFree_v2)(ptr) };
    }
}

/// Size reserved for `bytes`: power of two (frequent reuse) up to 64 MiB,
/// exact size beyond.
fn capacity(bytes: usize) -> usize {
    let b = bytes.max(256);
    if b <= POOL_POW2_MAX {
        b.next_power_of_two()
    } else {
        b
    }
}

/// An open CUDA device.
pub struct CuDevice {
    pub info: DeviceInfo,
    d: &'static Driver,
    ctx: CUcontext,
    compute: CUstream,
    transfer: CUstream,
    /// Compute capability (e.g. 89 for 8.9).
    cc: i32,
    pool: Arc<BufPool>,
    kernels: Mutex<HashMap<String, Arc<CuKernel>>>,
    pinned_pool: Mutex<HashMap<usize, Vec<Pinned>>>,
    /// Pinned buffers that a queued upload still reads, oldest first.
    pinned_busy: Mutex<VecDeque<(Pinned, CuEvent)>>,
}

// CUDA driver objects can be used from every thread (context made current).
unsafe impl Send for CuDevice {}
unsafe impl Sync for CuDevice {}

/// Compiled kernel (loaded PTX module and its function).
pub struct CuKernel {
    d: &'static Driver,
    ctx: CUcontext,
    module: CUmodule,
    func: CUfunction,
    /// Threads per block: the kernel's group size (`group=`), otherwise `BLOCK`.
    block: u32,
}

unsafe impl Send for CuKernel {}
unsafe impl Sync for CuKernel {}

impl Drop for CuKernel {
    fn drop(&mut self) {
        unsafe {
            (self.d.cuCtxSetCurrent)(self.ctx);
            (self.d.cuModuleUnload)(self.module);
        }
    }
}

/// Buffer in device memory.
pub struct CuBuffer {
    d: &'static Driver,
    ctx: CUcontext,
    pool: Arc<BufPool>,
    cap: usize,
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

unsafe impl Send for CuBuffer {}
unsafe impl Sync for CuBuffer {}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        unsafe { (self.d.cuCtxSetCurrent)(self.ctx) };
        if self.cap <= POOL_MAX_BUF {
            self.pool.give(self.cap, self.ptr);
        } else {
            unsafe { (self.d.cuMemFree_v2)(self.ptr) };
        }
    }
}

/// Event: end of a queued operation.
pub struct CuEvent {
    d: &'static Driver,
    ev: CUevent,
}

unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { (self.d.cuEventDestroy_v2)(self.ev) };
    }
}

impl CuEvent {
    pub fn wait(&self) -> Result<(), BackendError> {
        self.d
            .check(unsafe { (self.d.cuEventSynchronize)(self.ev) }, "attente")
    }
}

/// Pinned host buffer; returned to the pool by `recycle`, freed on drop.
pub struct Pinned {
    d: &'static Driver,
    ctx: CUcontext,
    pub ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            (self.d.cuCtxSetCurrent)(self.ctx);
            (self.d.cuMemFreeHost)(self.ptr.cast());
        }
    }
}

/// Launch argument, in kernel parameter order.
pub enum CuArg<'a> {
    Buf(&'a CuBuffer),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    I64(i64),
    U64(u64),
}

static OPEN: OnceLock<Mutex<HashMap<usize, Arc<CuDevice>>>> = OnceLock::new();

impl CuDevice {
    /// Opens (once per process) the device `cuda:<index>`.
    pub fn open(index: usize) -> Result<Arc<CuDevice>, BackendError> {
        let mut open = OPEN
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(d) = open.get(&index) {
            return Ok(d.clone());
        }
        let (dev, info) = crate::enumerate()?
            .into_iter()
            .nth(index)
            .ok_or_else(|| BackendError(format!("device cuda:{index} not found")))?;
        let d = sys::driver()?;
        let mut ctx = std::ptr::null_mut();
        d.check(
            unsafe { (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev) },
            "contexte",
        )?;
        d.check(unsafe { (d.cuCtxSetCurrent)(ctx) }, "contexte")?;
        let stream = || -> Result<CUstream, BackendError> {
            let mut s = std::ptr::null_mut();
            d.check(
                unsafe { (d.cuStreamCreate)(&mut s, sys::STREAM_NON_BLOCKING) },
                "stream creation",
            )?;
            Ok(s)
        };
        let (compute, transfer) = (stream()?, stream()?);
        let attr = |a| {
            let mut v = 0;
            unsafe { (d.cuDeviceGetAttribute)(&mut v, a, dev) };
            v
        };
        let cc = attr(sys::ATTR_CC_MAJOR) * 10 + attr(sys::ATTR_CC_MINOR);
        let pool = Arc::new(BufPool {
            d,
            compute,
            transfer,
            free: Mutex::new((Vec::new(), 0)),
        });
        let dev = Arc::new(CuDevice {
            info,
            d,
            ctx,
            compute,
            transfer,
            cc,
            pool,
            kernels: Mutex::new(HashMap::new()),
            pinned_pool: Mutex::new(HashMap::new()),
            pinned_busy: Mutex::default(),
        });
        open.insert(index, dev.clone());
        Ok(dev)
    }

    fn bind(&self) -> Result<(), BackendError> {
        self.d
            .check(unsafe { (self.d.cuCtxSetCurrent)(self.ctx) }, "contexte")
    }

    fn record(&self, stream: CUstream) -> Result<CuEvent, BackendError> {
        let mut ev = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuEventCreate)(&mut ev, sys::EVENT_DISABLE_TIMING) },
            "event",
        )?;
        let e = CuEvent { d: self.d, ev };
        self.d
            .check(unsafe { (self.d.cuEventRecord)(ev, stream) }, "event")?;
        Ok(e)
    }

    fn after(&self, stream: CUstream, wait: &[&CuEvent]) -> Result<(), BackendError> {
        for e in wait {
            self.d.check(
                unsafe { (self.d.cuStreamWaitEvent)(stream, e.ev, 0) },
                "waiting for an event",
            )?;
        }
        Ok(())
    }

    /// Compiles (with a cache) the CUDA C code and returns the kernel `entry`.
    /// `group`: threads per block required by the kernel (shared memory, barriers).
    pub fn compile(
        &self,
        code: &str,
        entry: &str,
        group: Option<u32>,
    ) -> Result<Arc<CuKernel>, BackendError> {
        let mut cache = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = cache.get(code) {
            return Ok(k.clone());
        }
        let ptx = match ptx_cache::get(code, self.cc) {
            Some(p) => p,
            None => {
                let p = self.ptx(code)?;
                ptx_cache::put(code, self.cc, &p);
                p
            }
        };
        self.bind()?;
        let mut module = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) },
            "loading the PTX (driver older than NVRTC? update the NVIDIA driver)",
        )?;
        let name = CString::new(entry).unwrap();
        let mut func = std::ptr::null_mut();
        if let Err(e) = self.d.check(
            unsafe { (self.d.cuModuleGetFunction)(&mut func, module, name.as_ptr()) },
            "kernel not found in the module",
        ) {
            unsafe { (self.d.cuModuleUnload)(module) };
            return Err(e);
        }
        let k = Arc::new(CuKernel {
            d: self.d,
            ctx: self.ctx,
            module,
            func,
            block: group.unwrap_or(BLOCK),
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    /// CUDA C → PTX with NVRTC, for the GPU's architecture (or the most recent one NVRTC knows: the
    /// driver then compiles the PTX for the GPU).
    fn ptx(&self, code: &str) -> Result<Vec<u8>, BackendError> {
        let n = sys::nvrtc()?;
        let archs = n.archs();
        let arch = archs
            .iter()
            .copied()
            .filter(|a| *a <= self.cc)
            .max()
            .unwrap_or(self.cc);
        let src = CString::new(code).map_err(|_| BackendError("invalid CUDA code".into()))?;
        let fname = CString::new("gpusync.cu").unwrap();
        let mut prog = std::ptr::null_mut();
        let r = unsafe {
            (n.nvrtcCreateProgram)(
                &mut prog,
                src.as_ptr(),
                fname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != 0 {
            return Err(BackendError(format!("NVRTC: creation ({})", n.error(r))));
        }
        // No --use_fast_math: precision comes first (results compared with NumPy), as with OpenCL.
        let opts: Vec<CString> = [
            format!("--gpu-architecture=compute_{arch}"),
            "-default-device".into(),
        ]
        .into_iter()
        .map(|o| CString::new(o).unwrap())
        .collect();
        let optp: Vec<*const i8> = opts.iter().map(|o| o.as_ptr()).collect();
        let r = unsafe { (n.nvrtcCompileProgram)(prog, optp.len() as i32, optp.as_ptr()) };
        let result = if r != 0 {
            let mut size = 0usize;
            unsafe { (n.nvrtcGetProgramLogSize)(prog, &mut size) };
            let mut log = vec![0u8; size.max(1)];
            unsafe { (n.nvrtcGetProgramLog)(prog, log.as_mut_ptr().cast()) };
            Err(BackendError(format!(
                "CUDA compilation failed on {} ({}):\n{}",
                self.info.name,
                n.error(r),
                String::from_utf8_lossy(&log).trim_end_matches('\0')
            )))
        } else {
            let mut size = 0usize;
            unsafe { (n.nvrtcGetPTXSize)(prog, &mut size) };
            let mut ptx = vec![0u8; size];
            unsafe { (n.nvrtcGetPTX)(prog, ptx.as_mut_ptr().cast()) };
            Ok(ptx)
        };
        unsafe { (n.nvrtcDestroyProgram)(&mut prog) };
        result
    }

    /// Buffer in device memory.
    pub fn alloc(&self, bytes: usize) -> Result<CuBuffer, BackendError> {
        self.bind()?;
        let cap = capacity(bytes);
        let ptr = match self.pool.take(cap) {
            Some(p) => p,
            None => {
                let mut ptr = 0;
                let mut r = unsafe { (self.d.cuMemAlloc_v2)(&mut ptr, cap) };
                if r != 0 {
                    // Memory full: give the pool back to the driver, then try again.
                    self.trim();
                    r = unsafe { (self.d.cuMemAlloc_v2)(&mut ptr, cap) };
                }
                self.d.check(r, "allocation")?;
                ptr
            }
        };
        Ok(CuBuffer {
            d: self.d,
            ctx: self.ctx,
            pool: self.pool.clone(),
            cap,
            ptr,
            bytes,
        })
    }

    /// Frees the reserved buffers (after queued work completes).
    fn trim(&self) {
        unsafe { (self.d.cuCtxSynchronize)() };
        let mut g = self.pool.free.lock().unwrap_or_else(|e| e.into_inner());
        for (_, ptr, evs) in g.0.drain(..) {
            unsafe {
                for e in evs {
                    (self.d.cuEventDestroy_v2)(e);
                }
                (self.d.cuMemFree_v2)(ptr);
            }
        }
        g.1 = 0;
    }

    /// Pinned buffer of at least `bytes` bytes (taken from the pool if possible).
    pub fn pinned(&self, bytes: usize) -> Result<Pinned, BackendError> {
        self.bind()?;
        self.reclaim();
        if let Some(p) = self
            .pinned_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&bytes)
            .and_then(Vec::pop)
        {
            return Ok(p);
        }
        self.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuMemHostAlloc)(&mut p, bytes.max(4), 0) },
            "pinned buffer",
        )?;
        Ok(Pinned {
            d: self.d,
            ctx: self.ctx,
            ptr: p.cast(),
            bytes,
        })
    }

    /// Puts back into the pool the buffers whose upload is done (the transfer stream is ordered).
    fn reclaim(&self) {
        loop {
            let p = {
                let mut busy = self.pinned_busy.lock().unwrap_or_else(|e| e.into_inner());
                match busy.front() {
                    Some((_, e)) if unsafe { (self.d.cuEventQuery)(e.ev) } == 0 => {
                        busy.pop_front().map(|(p, _)| p)
                    }
                    _ => None,
                }
            };
            match p {
                Some(p) => self.recycle(p),
                None => return,
            }
        }
    }

    /// Returns a pinned buffer that upload `_e` (last operation of the transfer stream, or an earlier one)
    /// still reads. Beyond 64 uploads in flight, waits for the oldest (bounded memory).
    pub fn recycle_after(&self, p: Pinned, _e: &CuEvent) -> Result<(), BackendError> {
        self.bind()?;
        let ev = self.record(self.transfer)?;
        let old = {
            let mut busy = self.pinned_busy.lock().unwrap_or_else(|e| e.into_inner());
            busy.push_back((p, ev));
            (busy.len() > 64).then(|| busy.pop_front()).flatten()
        };
        if let Some((p, e)) = old {
            e.wait()?;
            self.recycle(p);
        }
        Ok(())
    }

    /// Returns a pinned buffer to the pool.
    pub fn recycle(&self, p: Pinned) {
        self.pinned_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(p.bytes)
            .or_default()
            .push(p);
    }

    /// Host → device copy (transfer stream, non-blocking), after `wait`.
    ///
    /// # Safety
    /// `src` must stay valid until the returned event completes (and be pinned for the
    /// copy to be asynchronous).
    pub unsafe fn write_at(
        &self,
        dst: &CuBuffer,
        offset: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyHtoDAsync_v2)(
                    dst.ptr + offset as u64,
                    src.cast(),
                    bytes,
                    self.transfer,
                )
            },
            "write",
        )?;
        self.record(self.transfer)
    }

    /// Device → host copy (transfer stream, non-blocking), after `wait`.
    ///
    /// # Safety
    /// `dst` must stay valid until the returned event completes.
    pub unsafe fn read_at(
        &self,
        src: &CuBuffer,
        offset: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyDtoHAsync_v2)(
                    dst.cast(),
                    src.ptr + offset as u64,
                    bytes,
                    self.transfer,
                )
            },
            "read",
        )?;
        self.record(self.transfer)
    }

    /// Device → device copy (transfer stream), after `wait`.
    pub fn copy_at(
        &self,
        src: &CuBuffer,
        src_off: usize,
        dst: &CuBuffer,
        dst_off: usize,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyDtoDAsync_v2)(
                    dst.ptr + dst_off as u64,
                    src.ptr + src_off as u64,
                    bytes,
                    self.transfer,
                )
            },
            "copie",
        )?;
        self.record(self.transfer)
    }

    /// Queues the kernel on the compute stream over `n` elements, after `wait`.
    pub fn launch(
        &self,
        k: &CuKernel,
        args: &[CuArg],
        n: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.compute, wait)?;
        // Argument values (8 bytes each) and array of pointers to them.
        let mut vals: Vec<[u8; 8]> = args
            .iter()
            .map(|a| {
                let mut v = [0u8; 8];
                match a {
                    CuArg::Buf(b) => v.copy_from_slice(&b.ptr.to_ne_bytes()),
                    CuArg::I32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::U32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::F32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::F64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                    CuArg::I64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                    CuArg::U64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                }
                v
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
        let grid = (n as u64).div_ceil(k.block as u64).max(1);
        if grid > i32::MAX as u64 {
            return Err(BackendError(format!(
                "{n} elements: too many for a single CUDA grid"
            )));
        }
        self.d.check(
            unsafe {
                (self.d.cuLaunchKernel)(
                    k.func,
                    grid as u32,
                    1,
                    1,
                    k.block,
                    1,
                    1,
                    0,
                    self.compute,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "kernel launch",
        )?;
        self.record(self.compute)
    }

    /// CUDA commands go out without waiting: nothing to do (for the common interface).
    pub fn flush(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// Waits for everything queued (computation and transfers).
    pub fn finish(&self) -> Result<(), BackendError> {
        self.bind()?;
        self.d
            .check(unsafe { (self.d.cuCtxSynchronize)() }, "synchronisation")
    }
}

/// Disk cache of the PTX produced by NVRTC (~100 ms per kernel and per process otherwise). The rest
/// (PTX → machine code) is already cached by the driver. Folder: `GPUSYNC_CACHE_DIR`, otherwise
/// the user cache (`%LOCALAPPDATA%\gpusync`, `~/.cache/gpusync`, `~/Library/Caches/gpusync`);
/// `GPUSYNC_CACHE=0` disables it. An unreadable file is simply recompiled.
mod ptx_cache {
    use std::path::PathBuf;

    const VERSION: &str = "gpusync-ptx-1";

    fn dir() -> Option<PathBuf> {
        if std::env::var("GPUSYNC_CACHE").is_ok_and(|v| v == "0") {
            return None;
        }
        if let Some(d) = std::env::var_os("GPUSYNC_CACHE_DIR") {
            return Some(PathBuf::from(d).join("cuda"));
        }
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        }?;
        Some(base.join("gpusync").join("cuda"))
    }

    /// 64-bit FNV-1a: key stable across runs and Rust versions.
    fn key(code: &str, cc: i32) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in VERSION.bytes().chain(cc.to_le_bytes()).chain(code.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}.ptx")
    }

    pub fn get(code: &str, cc: i32) -> Option<Vec<u8>> {
        let p = std::fs::read(dir()?.join(key(code, cc))).ok()?;
        // PTX terminated by a zero (expected by cuModuleLoadData).
        (p.last() == Some(&0) && p.starts_with(b"//")).then_some(p)
    }

    pub fn put(code: &str, cc: i32, ptx: &[u8]) {
        let Some(d) = dir() else { return };
        if std::fs::create_dir_all(&d).is_err() {
            return;
        }
        let tmp = d.join(format!("{}.{}.tmp", key(code, cc), std::process::id()));
        if std::fs::write(&tmp, ptx).is_ok()
            && std::fs::rename(&tmp, d.join(key(code, cc))).is_err()
        {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
