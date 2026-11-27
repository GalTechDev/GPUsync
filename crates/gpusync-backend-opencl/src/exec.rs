//! OpenCL execution (ARCHITECTURE.md §8, §9).
//!
//! - **Two queues** per device: `compute` (kernels, zero-copy operations on host memory)
//!   and `transfer` (host ↔ device copies), chained by events: the transfers of one chunk
//!   overlap with the computation of another.
//! - **Unified memory**: `wrap_host` creates a `CL_MEM_USE_HOST_PTR` buffer that uses host memory
//!   directly; `sync_host` (map + unmap) makes it coherent after the computation.
//! - **Pinned memory**: `pinned` provides `CL_MEM_ALLOC_HOST_PTR` buffers mapped once and
//!   for all; a transfer from (or to) their pointer goes through DMA at full speed.

use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{
    Buffer, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_ALLOC_HOST_PTR, CL_MEM_READ_WRITE,
    CL_MEM_USE_HOST_PTR, ClMem,
};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_mem};

fn err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("OpenCL: {what} ({e})"))
}

/// An open OpenCL device.
pub struct ClDevice {
    pub info: DeviceInfo,
    context: Context,
    compute: CommandQueue,
    transfer: CommandQueue,
    kernels: Mutex<HashMap<String, Arc<ClKernel>>>,
    pinned_pool: Mutex<HashMap<usize, Vec<Pinned>>>,
    /// Pinned buffers that a queued upload still reads, oldest first.
    pinned_busy: Mutex<VecDeque<(Pinned, Event)>>,
    pool: Arc<BufPool>,
}

/// Compiled kernel. `set_arg` + launch must be atomic: hence the lock.
pub struct ClKernel {
    kernel: Mutex<Kernel>,
    _program: Program,
    /// Group size required (`group=`); otherwise chosen by the driver.
    local: Option<usize>,
}

/// OpenCL buffer, possibly backed by host memory (non-null `host_ptr`).
pub struct ClBuffer {
    buf: std::mem::ManuallyDrop<Buffer<u8>>,
    pub bytes: usize,
    pub host_ptr: *mut u8,
    /// Actual capacity and pool to return it to (buffers in device memory).
    cap: usize,
    pool: Option<Arc<BufPool>>,
}

impl Drop for ClBuffer {
    fn drop(&mut self) {
        let buf = unsafe { std::mem::ManuallyDrop::take(&mut self.buf) };
        if let Some(p) = &self.pool {
            p.give(self.cap, buf);
        }
    }
}

/// Pooled buffers: capacity rounded up to a power of two up to 64 MiB, exact size
/// beyond, 2 GiB in all (given back to the driver if an allocation fails).
const POOL_POW2_MAX: usize = 64 << 20;
const POOL_MAX_TOTAL: usize = 2 << 30;

fn capacity(bytes: usize) -> usize {
    let b = bytes.max(4);
    if b <= POOL_POW2_MAX {
        b.next_power_of_two()
    } else {
        b
    }
}

/// Pool of released buffers: `clCreateBuffer` gives fresh memory, paid for at the first
/// write. A returned buffer is kept with one marker per queue; it is preferably taken again once
/// the markers have passed, otherwise right away, each queue waiting (a barrier on the
/// device, without blocking the host) for the other's marker, its own order doing the rest.
struct BufPool {
    compute: opencl3::types::cl_command_queue,
    transfer: opencl3::types::cl_command_queue,
    free: Mutex<(Vec<Spare>, usize)>,
}

type Spare = (usize, Buffer<u8>, [Event; 2]);

unsafe impl Send for BufPool {}
unsafe impl Sync for BufPool {}

fn marker(q: opencl3::types::cl_command_queue, wait: &[cl_event]) -> Option<Event> {
    let w = if wait.is_empty() {
        ptr::null()
    } else {
        wait.as_ptr()
    };
    unsafe { opencl3::command_queue::enqueue_marker_with_wait_list(q, wait.len() as u32, w) }
        .ok()
        .map(Event::new)
}

impl BufPool {
    fn give(&self, cap: usize, buf: Buffer<u8>) {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if g.1 + cap > POOL_MAX_TOTAL {
            return; // released by the driver once the queued commands are done
        }
        if let (Some(c), Some(t)) = (marker(self.compute, &[]), marker(self.transfer, &[])) {
            g.0.push((cap, buf, [c, t]));
            g.1 += cap;
        }
    }

    fn flush(&self) {
        let _ = opencl3::command_queue::flush(self.compute);
        let _ = opencl3::command_queue::flush(self.transfer);
    }

    fn take(&self, cap: usize) -> Option<Buffer<u8>> {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let passed = |evs: &[Event; 2]| {
            evs.iter().all(|e| {
                e.command_execution_status().is_ok_and(|s| s.0 == 0) // CL_COMPLETE
            })
        };
        let ready = g.0.iter().position(|(c, _, evs)| *c == cap && passed(evs));
        let pos = ready.or_else(|| g.0.iter().position(|(c, _, _)| *c == cap))?;
        let (c, buf, [ec, et]) = g.0.swap_remove(pos);
        g.1 -= c;
        if ready.is_none() {
            // Wrap the returned cl_events in Event::new so that they get released (clReleaseEvent).
            let b1 = unsafe {
                opencl3::command_queue::enqueue_barrier_with_wait_list(self.compute, 1, &et.get())
            }
            .ok()
            .map(Event::new);
            let b2 = unsafe {
                opencl3::command_queue::enqueue_barrier_with_wait_list(self.transfer, 1, &ec.get())
            }
            .ok()
            .map(Event::new);
            let ok = b1.is_some() && b2.is_some();
            self.flush();
            drop((b1, b2));
            if !ok {
                let _ = (ec.wait(), et.wait());
            }
        }
        Some(buf)
    }

    fn trim(&self) {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        g.0.clear();
        g.1 = 0;
    }
}

// OpenCL objects (cl_mem, cl_event) can be used from any thread.
unsafe impl Send for ClBuffer {}
unsafe impl Sync for ClBuffer {}

impl ClBuffer {
    pub fn is_host_backed(&self) -> bool {
        !self.host_ptr.is_null()
    }

    /// Borrowed "Buffer" view (opencl3 APIs expect `&Buffer<T>`), without taking ownership.
    fn view(&self) -> std::mem::ManuallyDrop<Buffer<u8>> {
        std::mem::ManuallyDrop::new(Buffer::new(self.buf.get()))
    }
}

/// Pinned host buffer, permanently mapped; returned to the pool on drop.
pub struct Pinned {
    _buf: Buffer<u8>, // owner: the buffer stays mapped while it lives
    pub ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

/// Launch argument, in kernel parameter order.
pub enum ClArg<'a> {
    Buf(&'a ClBuffer),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    I64(i64),
    U64(u64),
}

static OPEN: OnceLock<Mutex<HashMap<usize, Arc<ClDevice>>>> = OnceLock::new();

fn wait_list(events: &[&Event]) -> Vec<cl_event> {
    events.iter().map(|e| e.get()).collect()
}

impl ClDevice {
    /// Opens (once per process) the device of index `index` (`opencl:<index>`).
    pub fn open(index: usize) -> Result<Arc<ClDevice>, BackendError> {
        let mut open = OPEN
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(d) = open.get(&index) {
            return Ok(d.clone());
        }
        let (id, info) = crate::enumerate()?
            .into_iter()
            .nth(index)
            .ok_or_else(|| BackendError(format!("device opencl:{index} not found")))?;
        let device = Device::new(id);
        let context = Context::from_device(&device).map_err(|e| err("context creation", e))?;
        #[allow(deprecated)] // clCreateCommandQueue: the only API available in OpenCL 1.2 (Apple)
        let compute =
            CommandQueue::create_default(&context, 0).map_err(|e| err("queue creation", e))?;
        #[allow(deprecated)]
        let transfer =
            CommandQueue::create_default(&context, 0).map_err(|e| err("queue creation", e))?;
        let pool = Arc::new(BufPool {
            compute: compute.get(),
            transfer: transfer.get(),
            free: Mutex::new((Vec::new(), 0)),
        });
        let d = Arc::new(ClDevice {
            pool,
            info,
            context,
            compute,
            transfer,
            kernels: Mutex::new(HashMap::new()),
            pinned_pool: Mutex::new(HashMap::new()),
            pinned_busy: Mutex::default(),
        });
        open.insert(index, d.clone());
        Ok(d)
    }

    /// Compiles (with a cache) the OpenCL C code and returns the kernel `entry`.
    /// `group`: group size required by the kernel (shared memory, barriers).
    pub fn compile(
        &self,
        code: &str,
        entry: &str,
        group: Option<u32>,
    ) -> Result<Arc<ClKernel>, BackendError> {
        let mut cache = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = cache.get(code) {
            return Ok(k.clone());
        }
        // No -cl-fast-relaxed-math: precision comes first (results compared with NumPy).
        let program =
            Program::create_and_build_from_source(&self.context, code, "").map_err(|log| {
                BackendError(format!(
                    "OpenCL compilation failed on {}:\n{log}",
                    self.info.name
                ))
            })?;
        let kernel = Kernel::create(&program, entry).map_err(|e| err("kernel creation", e))?;
        let k = Arc::new(ClKernel {
            kernel: Mutex::new(kernel),
            _program: program,
            local: group.map(|g| g as usize),
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    /// Buffer in device memory.
    pub fn alloc(&self, bytes: usize) -> Result<ClBuffer, BackendError> {
        let cap = capacity(bytes);
        let create = || unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_WRITE, cap, ptr::null_mut())
        };
        let buf = match self.pool.take(cap) {
            Some(b) => b,
            None => match create() {
                Ok(b) => b,
                Err(_) => {
                    // Memory full: give the pool back to the driver, then try again.
                    self.finish()?;
                    self.pool.trim();
                    create().map_err(|e| err("allocation", e))?
                }
            },
        };
        Ok(ClBuffer {
            buf: std::mem::ManuallyDrop::new(buf),
            bytes,
            host_ptr: ptr::null_mut(),
            cap,
            pool: Some(self.pool.clone()),
        })
    }

    /// Buffer that uses host memory directly (zero-copy on unified memory).
    ///
    /// # Safety
    /// `ptr` must stay valid (and not be freed) as long as the buffer may be used.
    pub unsafe fn wrap_host(&self, ptr: *mut u8, bytes: usize) -> Result<ClBuffer, BackendError> {
        let buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                bytes.max(4),
                ptr.cast(),
            )
        }
        .map_err(|e| err("host-backed buffer", e))?;
        Ok(ClBuffer {
            buf: std::mem::ManuallyDrop::new(buf),
            bytes,
            host_ptr: ptr,
            cap: bytes,
            pool: None,
        })
    }

    /// Pinned buffer of at least `bytes` bytes (taken from the pool if possible).
    pub fn pinned(&self, bytes: usize) -> Result<Pinned, BackendError> {
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
        let buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                bytes,
                ptr::null_mut(),
            )
        }
        .map_err(|e| err("pinned buffer", e))?;
        let mut mapped: cl_mem = ptr::null_mut();
        unsafe {
            self.transfer
                .enqueue_map_buffer(
                    &buf,
                    CL_BLOCKING,
                    CL_MAP_READ | CL_MAP_WRITE,
                    0,
                    bytes,
                    &mut mapped,
                    &[],
                )
                .map_err(|e| err("mapping a pinned buffer", e))?;
        }
        Ok(Pinned {
            _buf: buf,
            ptr: mapped.cast(),
            bytes,
        })
    }

    /// Puts back into the pool the buffers whose upload is done (the transfer queue is ordered).
    fn reclaim(&self) {
        loop {
            let p = {
                let mut busy = self.pinned_busy.lock().unwrap_or_else(|e| e.into_inner());
                match busy.front() {
                    // Done (0) or failed (negative).
                    Some((_, e)) if e.command_execution_status().is_ok_and(|s| s.0 <= 0) => {
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

    /// Returns a pinned buffer that upload `e` still reads. Beyond 64 uploads in flight, waits for the
    /// oldest (bounded memory).
    pub fn recycle_after(&self, p: Pinned, e: &Event) -> Result<(), BackendError> {
        unsafe { opencl3::event::retain_event(e.get()) }
            .map_err(|e| err("upload event", opencl3::error_codes::ClError(e)))?;
        let old = {
            let mut busy = self.pinned_busy.lock().unwrap_or_else(|e| e.into_inner());
            busy.push_back((p, Event::new(e.get())));
            (busy.len() > 64).then(|| busy.pop_front()).flatten()
        };
        if let Some((p, e)) = old {
            wait(&e)?;
            self.recycle(p);
        }
        Ok(())
    }

    /// Returns a pinned buffer to the pool (it stays mapped, ready for reuse; bounded per size).
    pub fn recycle(&self, p: Pinned) {
        let mut pool = self.pinned_pool.lock().unwrap_or_else(|e| e.into_inner());
        let slot = pool.entry(p.bytes).or_default();
        if slot.len() < 6 {
            slot.push(p);
        }
    }

    /// Host → device copy (transfer queue, non-blocking), after `wait`.
    ///
    /// # Safety
    /// `src` must stay valid until the returned event completes.
    pub unsafe fn write_at(
        &self,
        dst: &ClBuffer,
        offset: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&Event],
    ) -> Result<Event, BackendError> {
        let mut b = dst.view();
        let data = unsafe { std::slice::from_raw_parts(src, bytes) };
        unsafe {
            self.transfer.enqueue_write_buffer(
                &mut b,
                CL_NON_BLOCKING,
                offset,
                data,
                &wait_list(wait),
            )
        }
        .map_err(|e| err("write", e))
    }

    /// Device → host copy (transfer queue, non-blocking), after `wait`.
    ///
    /// # Safety
    /// `dst` must stay valid until the returned event completes.
    pub unsafe fn read_at(
        &self,
        src: &ClBuffer,
        offset: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&Event],
    ) -> Result<Event, BackendError> {
        let data = unsafe { std::slice::from_raw_parts_mut(dst, bytes) };
        unsafe {
            self.transfer.enqueue_read_buffer(
                &src.view(),
                CL_NON_BLOCKING,
                offset,
                data,
                &wait_list(wait),
            )
        }
        .map_err(|e| err("read", e))
    }

    /// Device → device copy (transfer queue), after `wait`.
    pub fn copy_at(
        &self,
        src: &ClBuffer,
        src_off: usize,
        dst: &ClBuffer,
        dst_off: usize,
        bytes: usize,
        wait: &[&Event],
    ) -> Result<Event, BackendError> {
        let mut d = dst.view();
        unsafe {
            self.transfer.enqueue_copy_buffer(
                &src.view(),
                &mut d,
                src_off,
                dst_off,
                bytes,
                &wait_list(wait),
            )
        }
        .map_err(|e| err("copie", e))
    }

    /// Queues the kernel on the compute queue over `n` elements, after `wait`.
    pub fn launch(
        &self,
        k: &ClKernel,
        args: &[ClArg],
        n: usize,
        wait: &[&Event],
    ) -> Result<Event, BackendError> {
        let kernel = k.kernel.lock().unwrap_or_else(|e| e.into_inner());
        let mut exec = ExecuteKernel::new(&kernel);
        let mems: Vec<cl_mem> = args
            .iter()
            .filter_map(|a| {
                if let ClArg::Buf(b) = a {
                    Some(b.buf.get())
                } else {
                    None
                }
            })
            .collect();
        let mut m = mems.iter();
        let waits = wait_list(wait);
        let ev = unsafe {
            for a in args {
                match a {
                    ClArg::Buf(_) => exec.set_arg(m.next().unwrap()),
                    ClArg::I32(v) => exec.set_arg(v),
                    ClArg::U32(v) => exec.set_arg(v),
                    ClArg::F32(v) => exec.set_arg(v),
                    ClArg::F64(v) => exec.set_arg(v),
                    ClArg::I64(v) => exec.set_arg(v),
                    ClArg::U64(v) => exec.set_arg(v),
                };
            }
            exec.set_global_work_size(n);
            if let Some(l) = k.local {
                exec.set_local_work_size(l);
            }
            exec.set_event_wait_list(&waits)
                .enqueue_nd_range(&self.compute)
        }
        .map_err(|e| err("kernel launch", e))?;
        self.compute.flush().map_err(|e| err("flush", e))?;
        Ok(ev)
    }

    /// Sends the commands of both queues to the device without waiting.
    pub fn flush(&self) -> Result<(), BackendError> {
        self.transfer.flush().map_err(|e| err("flush", e))?;
        self.compute.flush().map_err(|e| err("flush", e))
    }

    /// Makes the host memory of a `wrap_host` buffer coherent (waits for the computations).
    pub fn sync_host(&self, buf: &ClBuffer) -> Result<(), BackendError> {
        let mut mapped: cl_mem = ptr::null_mut();
        unsafe {
            self.compute
                .enqueue_map_buffer(
                    &buf.view(),
                    CL_BLOCKING,
                    CL_MAP_READ,
                    0,
                    buf.bytes,
                    &mut mapped,
                    &[],
                )
                .map_err(|e| err("map", e))?;
            self.compute
                .enqueue_unmap_mem_object(buf.buf.get(), mapped.cast(), &[])
                .map_err(|e| err("unmap", e))?;
        }
        self.compute.finish().map_err(|e| err("finish", e))
    }

    /// Waits for everything queued (computation and transfers).
    pub fn finish(&self) -> Result<(), BackendError> {
        self.compute.finish().map_err(|e| err("finish", e))?;
        self.transfer.finish().map_err(|e| err("finish", e))
    }
}

pub fn wait(e: &Event) -> Result<(), BackendError> {
    e.wait().map_err(|e| err("attente", e))
}
