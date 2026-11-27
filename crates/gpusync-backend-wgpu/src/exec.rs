//! wgpu execution: one `Device` per adapter, cached pipelines, storage buffers.
//!
//! Binding layout (see gpusync-codegen): input arrays, output, then `gs_params`
//! (scalars and `n`, one 32-bit word each). The grid is 2-D to go beyond 65,535 groups.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};

pub const WORKGROUP_SIZE: u32 = 256;

pub struct WgDevice {
    pub info: DeviceInfo,
    pub(crate) adapter_idx: usize,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    max_groups_per_dim: u32,
    kernels: Mutex<HashMap<String, Arc<WgKernel>>>,
    staging_pool: Mutex<HashMap<(usize, bool), Vec<Staging>>>,
    pool: Arc<BufPool>,
    /// Grouping of several dispatches into a single CommandEncoder (`gs.batch()`).
    batch: Mutex<BatchState>,
    /// Key in `OPEN` (e.g. `vulkan:1`).
    key: String,
    /// Why the driver lost the device (Windows TDR, crash…), if lost.
    lost: Arc<Mutex<Option<String>>>,
    /// Last uncaptured wgpu error (validation / OOM outside an explicit scope).
    uncaptured: Arc<Mutex<Option<String>>>,
    /// Batches rejected by wgpu at submission: reading or reusing what they were meant to write is an
    /// error (never a silently wrong result, goal O2).
    failed: Mutex<HashMap<u64, String>>,
    any_failed: AtomicBool,
}

/// Command batch of the device, shared by every thread. Its lock covers submission:
/// as soon as a thread gets it, every withdrawn batch is already queued, in order.
#[derive(Default)]
struct BatchState {
    depth: usize,
    /// Number of the current batch (from 1), changed at each new encoder.
    id: u64,
    encoder: Option<wgpu::CommandEncoder>,
    keep: Vec<WgBuffer>,
}

impl BatchState {
    fn open(&mut self, device: &wgpu::Device) -> &mut wgpu::CommandEncoder {
        if self.encoder.is_none() {
            self.id += 1;
        }
        self.encoder
            .get_or_insert_with(|| device.create_command_encoder(&Default::default()))
    }
}

pub struct WgKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_inputs: usize,
    n_outputs: usize,
    has_fparams: bool,
    /// Inputs written by the kernel (explicit-mode gs.Out).
    writable: Vec<bool>,
    /// Threads per group (`@workgroup_size` of the module).
    workgroup: u32,
}

/// Launch ready to record: bindings, grid, and parameter buffers to keep alive until
/// submission.
struct Call {
    bind: wgpu::BindGroup,
    groups: (u32, u32),
    keep: Vec<WgBuffer>,
}

impl Call {
    fn record(&self, enc: &mut wgpu::CommandEncoder, k: &WgKernel) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&k.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
    }
}

pub struct WgBuffer {
    pub(crate) buf: wgpu::Buffer,
    pub bytes: usize,
    /// Pool to return the buffer to (allocated by `alloc`); `None` for wrapped host memory.
    pool: Option<Arc<BufPool>>,
    /// Last batch that uses this buffer, and last batch that writes it (0: none).
    used_lot: AtomicU64,
    written_lot: AtomicU64,
}

impl WgBuffer {
    fn new(buf: wgpu::Buffer, bytes: usize, pool: Option<Arc<BufPool>>) -> Self {
        WgBuffer {
            buf,
            bytes,
            pool,
            used_lot: AtomicU64::new(0),
            written_lot: AtomicU64::new(0),
        }
    }
}

impl Drop for WgBuffer {
    fn drop(&mut self) {
        if let Some(p) = &self.pool {
            p.give(self.buf.clone());
        }
    }
}

/// Released buffers, reused by later calls: a fresh buffer is zero-filled by wgpu
/// on first use (expensive on unified memory). wgpu's single queue orders
/// submissions and tracks usage: a returned buffer can be reused right away. Size rounded up
/// to a power of two up to 64 MiB, exact beyond; 2 GiB in all.
#[derive(Default)]
struct BufPool {
    free: Mutex<(HashMap<u64, Vec<wgpu::Buffer>>, u64)>,
    /// During a batch (`true`), returned buffers wait for submission: recorded
    /// commands may still use them.
    deferred: Mutex<(bool, Vec<wgpu::Buffer>)>,
}

const POOL_POW2_MAX: u64 = 64 << 20;
const POOL_MAX_TOTAL: u64 = 2 << 30;
const STAGING_MAX_PER_KEY: usize = 6;
const FAST_UPLOAD_MAX: usize = 4 << 20;

fn capacity(bytes: usize) -> u64 {
    let b = aligned(bytes);
    if b <= POOL_POW2_MAX {
        b.next_power_of_two()
    } else {
        b
    }
}

impl BufPool {
    fn push_free(&self, buf: wgpu::Buffer) {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let size = buf.size();
        if g.1 + size <= POOL_MAX_TOTAL {
            g.0.entry(size).or_default().push(buf);
            g.1 += size;
        }
    }

    fn give(&self, buf: wgpu::Buffer) {
        {
            let mut d = self.deferred.lock().unwrap_or_else(|e| e.into_inner());
            if d.0 {
                d.1.push(buf);
                return;
            }
        }
        self.push_free(buf);
    }

    /// Puts pending buffers back into the pool, then keeps (`on`) or stops deferring the next ones.
    /// To be called under the batch lock, once its commands are submitted.
    fn drain_deferred(&self, on: bool) {
        let items = {
            let mut d = self.deferred.lock().unwrap_or_else(|e| e.into_inner());
            d.0 = on;
            std::mem::take(&mut d.1)
        };
        for buf in items {
            self.push_free(buf);
        }
    }

    fn take(&self, size: u64) -> Option<wgpu::Buffer> {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let b = g.0.get_mut(&size)?.pop()?;
        g.1 -= size;
        Some(b)
    }
}

static OPEN: OnceLock<Mutex<HashMap<String, Arc<WgDevice>>>> = OnceLock::new();

fn err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("wgpu: {what} ({e})"))
}

/// Size aligned to 4 bytes (a wgpu copy constraint), never zero.
fn aligned(bytes: usize) -> u64 {
    (bytes.max(4) as u64).div_ceil(4) * 4
}

impl WgDevice {
    /// Clear error if the driver lost the device (it is then forgotten and will be reopened
    /// on the next `open`), or if an uncaptured wgpu error happened.
    fn check(&self) -> Result<(), BackendError> {
        let lost = self.lost.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(reason) = lost {
            if let Some(open) = OPEN.get() {
                let mut open = open.lock().unwrap_or_else(|e| e.into_inner());
                if open
                    .get(&self.key)
                    .is_some_and(|d| std::ptr::eq(&**d, self))
                {
                    open.remove(&self.key);
                }
            }
            return Err(BackendError(format!(
                "wgpu: the driver lost the device {} ({reason}). On Windows, a computation longer \
                 than ~2 s on a GPU is interrupted (TDR): reduce the size of the call or choose a faster \
                 GPU. The device will be reopened on the next call.",
                self.info.name
            )));
        }
        if let Some(msg) = self
            .uncaptured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            return Err(BackendError(format!(
                "wgpu: error on {} ({msg})",
                self.info.name
            )));
        }
        Ok(())
    }

    /// Storage buffer initialized from the pool (no GPU allocation once warm).
    fn storage_init(&self, contents: &[u8]) -> Result<WgBuffer, BackendError> {
        let buf = self.alloc(contents.len())?;
        if contents.len().is_multiple_of(4) {
            if !contents.is_empty() {
                self.queue.write_buffer(&buf.buf, 0, contents);
            }
        } else {
            let mut data = contents.to_vec();
            data.resize(aligned(contents.len()) as usize, 0);
            self.queue.write_buffer(&buf.buf, 0, &data);
        }
        Ok(buf)
    }

    fn lock_batch(&self) -> std::sync::MutexGuard<'_, BatchState> {
        self.batch.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` under wgpu error scopes (per thread, without waiting for the GPU): a validation
    /// or memory error becomes the error of this operation.
    fn guarded<T>(&self, what: &str, f: impl FnOnce() -> T) -> Result<T, BackendError> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let r = f();
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(oom.pop()),
        ];
        if let Some(e) = errors.into_iter().flatten().next() {
            self.check()?;
            return Err(BackendError(format!(
                "wgpu: {what} on {} ({e})",
                self.info.name
            )));
        }
        Ok(r)
    }

    /// Error if `buf` was to be written by a rejected batch.
    pub(crate) fn check_buf(&self, buf: &WgBuffer) -> Result<(), BackendError> {
        if !self.any_failed.load(Ordering::Acquire) {
            return Ok(());
        }
        let lot = buf.written_lot.load(Ordering::Acquire);
        match self
            .failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&lot)
        {
            Some(m) => Err(BackendError(format!("result not computed: {m}"))),
            None => Ok(()),
        }
    }

    /// Submits the current batch, under its lock: no other thread can record, read or
    /// take back a buffer between removing the encoder and queuing it.
    fn submit_lot(&self, b: &mut BatchState, keep_deferring: bool) -> Result<(), BackendError> {
        let mut r = Ok(());
        if let Some(enc) = b.encoder.take() {
            r = self.guarded("command batch rejected", || {
                self.queue.submit([enc.finish()]);
            });
            if let Err(e) = &r {
                self.failed
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(b.id, e.0.clone());
                self.any_failed.store(true, Ordering::Release);
            }
        }
        b.keep.clear();
        self.pool.drain_deferred(keep_deferring);
        r
    }

    /// Opens a command batch: the following `launch` and `copy` share a single `CommandEncoder`.
    pub fn begin_batch(&self) {
        let mut b = self.lock_batch();
        if b.depth == 0 {
            self.pool.drain_deferred(true); // no pending command: defer from here
        }
        b.depth += 1;
    }

    /// Submits the current `CommandEncoder` immediately (if a batch is active), without leaving the batch.
    pub fn flush_batch(&self) -> Result<(), BackendError> {
        let mut b = self.lock_batch();
        let deferring = b.depth > 0;
        self.submit_lot(&mut b, deferring)
    }

    /// Closes the current command batch and submits the `CommandEncoder` when the depth drops back to 0.
    pub fn end_batch(&self) -> Result<(), BackendError> {
        {
            let mut b = self.lock_batch();
            b.depth = b.depth.saturating_sub(1);
            if b.depth > 0 {
                return Ok(());
            }
            self.submit_lot(&mut b, false)?;
        }
        self.check()
    }

    /// Opens (once per process) the device `<api>:<index>` (e.g. `metal:0`, `vulkan:1`).
    pub fn open(api: &str, index: usize) -> Result<Arc<WgDevice>, BackendError> {
        let key = format!("{api}:{index}");
        let mut open = OPEN
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(d) = open.get(&key) {
            return Ok(d.clone());
        }
        let st = crate::state()?;
        let (adapter_idx, info) = st
            .devices
            .iter()
            .find(|(_, d)| d.backend == api && d.index == index)
            .ok_or_else(|| BackendError(format!("device {key} not found")))?;
        let adapter = &st.adapters[*adapter_idx];
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpusync"),
            // f64 in WGSL (Vulkan): native wgpu extension, requested only when available.
            required_features: {
                let mut f = wgpu::Features::empty();
                if info.f64 {
                    f |= wgpu::Features::SHADER_F64;
                }
                if info.int64 {
                    f |= wgpu::Features::SHADER_INT64;
                }
                f
            },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| err("opening the device", e))?;
        // By default wgpu panics on an uncaptured error: capture it to report it in Python.
        let uncaptured: Arc<Mutex<Option<String>>> = Arc::default();
        let uflag = uncaptured.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            let mut g = uflag.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_none() {
                *g = Some(e.to_string());
            }
        }));
        let lost: Arc<Mutex<Option<String>>> = Arc::default();
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            *flag.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{reason:?}: {msg}"));
        });
        let d = Arc::new(WgDevice {
            info: info.clone(),
            adapter_idx: *adapter_idx,
            device,
            queue,
            max_groups_per_dim: limits.max_compute_workgroups_per_dimension.max(1),
            kernels: Mutex::new(HashMap::new()),
            staging_pool: Mutex::new(HashMap::new()),
            pool: Arc::default(),
            batch: Mutex::default(),
            key: key.clone(),
            lost,
            uncaptured,
            failed: Mutex::default(),
            any_failed: AtomicBool::new(false),
        });
        open.insert(key, d.clone());
        Ok(d)
    }

    /// Compiles (with a cache) a WGSL module whose entry point is `main`. `writable`: inputs
    /// bound read-write (explicit-mode gs.Out outputs).
    pub fn compile(
        &self,
        code: &str,
        n_inputs: usize,
        n_outputs: usize,
        has_fparams: bool,
        writable: &[bool],
        group: Option<u32>,
    ) -> Result<Arc<WgKernel>, BackendError> {
        let mut cache = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = cache.get(code) {
            return Ok(k.clone());
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Explicit layout: a declared but unused binding stays valid.
        let mut entries: Vec<_> = (0..n_inputs as u32)
            .map(|b| storage(b, !writable.get(b as usize).copied().unwrap_or(false)))
            .collect();
        let first_param = (n_inputs + n_outputs) as u32;
        entries.extend((n_inputs as u32..first_param).map(|b| storage(b, false)));
        entries.push(storage(first_param, true));
        if has_fparams {
            entries.push(storage(first_param + 1, true));
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpusync"),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpusync"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(BackendError(format!(
                "WGSL compilation failed on {}:\n{e}",
                self.info.name
            )));
        }
        let k = Arc::new(WgKernel {
            pipeline,
            layout,
            n_inputs,
            n_outputs,
            has_fparams,
            writable: writable.to_vec(),
            workgroup: group.unwrap_or(WORKGROUP_SIZE),
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    pub fn alloc(&self, bytes: usize) -> Result<WgBuffer, BackendError> {
        self.check()?;
        let size = capacity(bytes);
        let buf = match self.pool.take(size) {
            Some(buf) => buf,
            None => self.guarded("allocation rejected", || {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })?,
        };
        Ok(WgBuffer::new(buf, bytes, Some(self.pool.clone())))
    }

    /// Host → device copy (the data is copied immediately by wgpu).
    pub fn write(&self, buf: &WgBuffer, data: &[u8]) -> Result<(), BackendError> {
        self.write_at(buf, 0, data)
    }

    /// `queue.write_buffer` runs before any command submitted afterwards: if the current batch already
    /// uses `dst`, it is submitted first (otherwise its computations would read the new data).
    fn write_at(&self, dst: &WgBuffer, off: usize, data: &[u8]) -> Result<(), BackendError> {
        {
            let mut b = self.lock_batch();
            if b.encoder.is_some() && dst.used_lot.load(Ordering::Acquire) == b.id {
                let deferring = b.depth > 0;
                let _ = self.submit_lot(&mut b, deferring); // the failure stays attached to its results
            }
        }
        self.guarded("upload rejected", || {
            if data.len().is_multiple_of(4) {
                self.queue.write_buffer(&dst.buf, off as u64, data);
            } else {
                let mut padded = data.to_vec();
                padded.resize(data.len().div_ceil(4) * 4, 0);
                self.queue.write_buffer(&dst.buf, off as u64, &padded);
            }
        })
    }

    /// Queues the kernel over `n` elements (or records it in the active `gs.batch()`).
    pub fn launch(
        &self,
        k: &WgKernel,
        inputs: &[&WgBuffer],
        outs: &[&WgBuffer],
        params: &[u32],
        fparams: &[f64],
        n: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        if inputs.len() != k.n_inputs || outs.len() != k.n_outputs {
            return Err(BackendError(
                "inconsistent number of input arrays".into(),
            ));
        }
        for b in inputs.iter().chain(outs) {
            self.check_buf(b)?;
        }
        // `gs.batch()` open: encode into it (under its lock).
        {
            let mut b = self.lock_batch();
            if b.depth > 0 {
                let call = self.prepare(k, inputs, outs, params, fparams, n)?;
                call.record(b.open(&self.device), k);
                let id = b.id;
                for buf in inputs.iter().chain(outs) {
                    buf.used_lot.store(id, Ordering::Release);
                }
                let written = inputs
                    .iter()
                    .zip(&k.writable)
                    .filter_map(|(b, w)| w.then_some(b));
                for buf in written.chain(outs) {
                    buf.written_lot.store(id, Ordering::Release);
                }
                b.keep.extend(call.keep);
                return Ok(());
            }
        }
        // Any validation or memory error becomes a Python error (never a silently wrong
        // result, goal O2).
        let call = self.prepare(k, inputs, outs, params, fparams, n)?;
        self.guarded("launch rejected", || {
            let mut enc = self.device.create_command_encoder(&Default::default());
            call.record(&mut enc, k);
            self.queue.submit([enc.finish()]);
        })?;
        Ok(())
    }

    /// Prepares the launch of `k` over `n` elements (parameters, bindings), without recording anything: an
    /// error here does not affect the current batch.
    fn prepare(
        &self,
        k: &WgKernel,
        inputs: &[&WgBuffer],
        outs: &[&WgBuffer],
        params: &[u32],
        fparams: &[f64],
        n: usize,
    ) -> Result<Call, BackendError> {
        let params_buf = self.storage_init(bytemuck_u32(params))?;
        let fparams_buf = if k.has_fparams {
            Some(self.storage_init(unsafe {
                std::slice::from_raw_parts(fparams.as_ptr().cast(), std::mem::size_of_val(fparams))
            })?)
        } else {
            None
        };
        let mut entries: Vec<wgpu::BindGroupEntry> =
            Vec::with_capacity(inputs.len() + outs.len() + 2);
        for (i, b) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buf.as_entire_binding(),
            });
        }
        for (o, b) in outs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (inputs.len() + o) as u32,
                resource: b.buf.as_entire_binding(),
            });
        }
        let first_param = (inputs.len() + outs.len()) as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: first_param,
            resource: params_buf.buf.as_entire_binding(),
        });
        if let Some(b) = &fparams_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: first_param + 1,
                resource: b.buf.as_entire_binding(),
            });
        }
        let groups = (n as u64).div_ceil(k.workgroup as u64);
        let gx = groups.min(self.max_groups_per_dim as u64) as u32;
        let gy = groups.div_ceil(gx.max(1) as u64);
        if gy > self.max_groups_per_dim as u64 {
            return Err(BackendError(format!(
                "{n} elements: too many for a single grid on {}",
                self.info.name
            )));
        }
        let bind = self.guarded("binding rejected", || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &k.layout,
                entries: &entries,
            })
        })?;
        Ok(Call {
            bind,
            groups: (gx, gy as u32),
            keep: std::iter::once(params_buf).chain(fparams_buf).collect(),
        })
    }

    /// Device → device copy, queued (offsets and size multiples of 4).
    pub fn copy(
        &self,
        src: &WgBuffer,
        src_off: usize,
        dst: &WgBuffer,
        dst_off: usize,
        bytes: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        if bytes == 0 {
            return Ok(());
        }
        self.check_buf(src)?;
        self.check_buf(dst)?;
        let copy_len = aligned(bytes);
        let record = |enc: &mut wgpu::CommandEncoder| {
            enc.copy_buffer_to_buffer(&src.buf, src_off as u64, &dst.buf, dst_off as u64, copy_len);
        };
        {
            let mut b = self.lock_batch();
            if b.depth > 0 {
                record(b.open(&self.device));
                let id = b.id;
                src.used_lot.store(id, Ordering::Release);
                dst.used_lot.store(id, Ordering::Release);
                dst.written_lot.store(id, Ordering::Release);
                return Ok(());
            }
        }
        self.guarded("copy rejected", || {
            let mut enc = self.device.create_command_encoder(&Default::default());
            record(&mut enc);
            self.queue.submit([enc.finish()]);
        })
    }

    /// Blocking device → host copy (waits for submitted work).
    pub fn read(&self, buf: &WgBuffer, out: &mut [u8]) -> Result<(), BackendError> {
        self.check()?;
        let _ = self.flush_batch(); // a rejected batch only affects its own results
        self.check_buf(buf)?;
        let size = aligned(buf.bytes);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::default();
        let r = result.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |res| {
            *r.lock().unwrap() = Some(res)
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| err("waiting for the GPU", e))?;
        match result.lock().unwrap().take() {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(err("reading the result", e)),
            None => return Err(BackendError("wgpu: read not finished".into())),
        }
        {
            let view = staging
                .get_mapped_range(..)
                .map_err(|e| err("accessing the result", e))?;
            out.copy_from_slice(&view[..out.len()]);
        }
        staging.unmap();
        self.check()
    }

    /// Metal buffer that uses host memory directly (zero-copy, `newBufferWithBytesNoCopy`).
    ///
    /// Returns `Ok(None)` when that is not possible (an API other than Metal, address not aligned on a
    /// 16 KiB page, memory refused by Metal): the caller then goes through transfers.
    ///
    /// # Safety
    /// `ptr..ptr + bytes` rounded up to the next page must stay valid while the buffer exists.
    pub unsafe fn wrap_host(
        &self,
        ptr: *mut u8,
        bytes: usize,
    ) -> Result<Option<WgBuffer>, BackendError> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice, MTLResourceOptions};
            const PAGE: usize = 16384;
            if self.info.backend != "metal" || !(ptr as usize).is_multiple_of(PAGE) || bytes == 0 {
                return Ok(None);
            }
            let len = bytes.div_ceil(PAGE) * PAGE; // the last whole page stays within the same mapped page
            let Some(hal_dev) = (unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() }) else {
                return Ok(None);
            };
            let Some(raw) = (unsafe {
                hal_dev
                    .raw_device()
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        std::ptr::NonNull::new_unchecked(ptr.cast()),
                        len,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            }) else {
                return Ok(None);
            };
            let hal_buf = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw, len as u64) };
            let buf = unsafe {
                self.device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    hal_buf,
                    &wgpu::BufferDescriptor {
                        label: Some("gpusync-nocopy"),
                        size: len as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };
            Ok(Some(WgBuffer::new(buf, bytes, None)))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ptr, bytes);
            Ok(None)
        }
    }

    // ───────────────────── chunked transfers (ARCHITECTURE.md §9) ─────────────────────

    fn staging(&self, bytes: usize, upload: bool) -> Staging {
        if let Some(st) = self
            .staging_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&(bytes, upload))
            .and_then(Vec::pop)
        {
            return st;
        }
        let usage = if upload {
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        };
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpusync-staging"),
            size: aligned(bytes),
            usage,
            mapped_at_creation: upload, // an upload buffer is born ready to be filled
        });
        Staging {
            buf,
            bytes,
            mapped: Arc::new(AtomicBool::new(upload)),
            submission: None,
        }
    }

    fn recycle(&self, st: Staging, upload: bool) {
        let mut pool = self.staging_pool.lock().unwrap_or_else(|e| e.into_inner());
        let slot = pool.entry((st.bytes, upload)).or_default();
        if slot.len() < STAGING_MAX_PER_KEY {
            slot.push(st);
        }
    }

    /// Waits until `st` is mapped (its last copy done), without waiting for later submissions.
    fn wait_mapped(&self, st: &Staging) -> Result<(), BackendError> {
        if st.mapped.load(Ordering::Acquire) {
            return Ok(());
        }
        let ty = match &st.submission {
            Some(idx) => wgpu::PollType::Wait {
                submission_index: Some(idx.clone()),
                timeout: None,
            },
            None => wgpu::PollType::wait_indefinitely(),
        };
        self.device
            .poll(ty)
            .map_err(|e| err("attente d'un transfert", e))?;
        while !st.mapped.load(Ordering::Acquire) {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| err("attente d'un transfert", e))?;
            std::thread::yield_now();
        }
        Ok(())
    }

    fn remap(&self, st: &mut Staging, mode: wgpu::MapMode, idx: &wgpu::SubmissionIndex) {
        st.mapped.store(false, Ordering::Release);
        st.submission = Some(idx.clone());
        let flag = st.mapped.clone();
        st.buf.map_async(mode, .., move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
    }

    /// Sends `bytes` bytes from the host to `dst` (starting at `dst_off`) in chunks of `chunk` bytes,
    /// through a ring of 3 buffers: the extra host memory stays bounded to 3 chunks.
    ///
    /// # Safety
    /// `src` must point to `bytes` readable bytes.
    pub unsafe fn upload(
        &self,
        dst: &WgBuffer,
        dst_off: usize,
        src: *const u8,
        bytes: usize,
        chunk: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        if bytes == 0 {
            return Ok(());
        }
        // Fast path for small and medium transfers (≤ 4 MiB): `queue.write_buffer` avoids
        // creating 3 staging buffers, a CommandEncoder and a dedicated queue.submit.
        if bytes <= FAST_UPLOAD_MAX && bytes <= chunk.max(FAST_UPLOAD_MAX) {
            let data = unsafe { std::slice::from_raw_parts(src, bytes) };
            return self.write_at(dst, dst_off, data);
        }
        let _ = self.flush_batch(); // a rejected batch only affects its own results
        let chunk = chunk.min(bytes).max(4);
        let mut ring: Vec<Staging> = (0..3).map(|_| self.staging(chunk, true)).collect();
        let mut off = 0;
        let mut i = 0;
        while off < bytes {
            let len = chunk.min(bytes - off);
            let st = &mut ring[i % 3];
            self.wait_mapped(st)?;
            {
                let mut view = st
                    .buf
                    .get_mapped_range_mut(..len as u64)
                    .map_err(|e| err("writing a buffer", e))?;
                view.copy_from_slice(unsafe { std::slice::from_raw_parts(src.add(off), len) });
            }
            st.buf.unmap();
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&st.buf, 0, &dst.buf, (dst_off + off) as u64, len as u64);
            let idx = self.queue.submit([enc.finish()]);
            self.remap(st, wgpu::MapMode::Write, &idx);
            off += len;
            i += 1;
        }
        for st in ring {
            self.recycle(st, true);
        }
        self.check()
    }

    /// Copies `bytes` bytes of `src` (starting at `src_off`) back to the host, in chunks (ring of 3).
    ///
    /// # Safety
    /// `dst` must point to `bytes` writable bytes.
    pub unsafe fn download(
        &self,
        src: &WgBuffer,
        src_off: usize,
        dst: *mut u8,
        bytes: usize,
        chunk: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        let _ = self.flush_batch(); // a rejected batch only affects its own results
        self.check_buf(src)?;
        if bytes == 0 {
            return Ok(());
        }
        let chunk = chunk.min(bytes).max(4);
        let mut ring: Vec<Staging> = (0..3).map(|_| self.staging(chunk, false)).collect();
        let mut pending: std::collections::VecDeque<(usize, usize, usize)> = Default::default(); // (slot, off, len)
        let mut off = 0;
        let mut i = 0;
        loop {
            // Finishes the oldest chunk when the ring is full (or at the end).
            if pending.len() == 3 || (off >= bytes && !pending.is_empty()) {
                let (slot, o, len) = pending.pop_front().unwrap();
                let st = &ring[slot];
                self.wait_mapped(st)?;
                {
                    let view = st
                        .buf
                        .get_mapped_range(..len as u64)
                        .map_err(|e| err("reading a buffer", e))?;
                    unsafe { std::ptr::copy_nonoverlapping(view.as_ptr(), dst.add(o), len) };
                }
                st.buf.unmap();
                continue;
            }
            if off >= bytes {
                break;
            }
            let len = chunk.min(bytes - off);
            let slot = i % 3;
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                &src.buf,
                (src_off + off) as u64,
                &ring[slot].buf,
                0,
                len.div_ceil(4) as u64 * 4,
            );
            let idx = self.queue.submit([enc.finish()]);
            self.remap(&mut ring[slot], wgpu::MapMode::Read, &idx);
            pending.push_back((slot, off, len));
            off += len;
            i += 1;
        }
        for st in ring {
            self.recycle(st, false);
        }
        self.check()
    }

    /// Runs `k` in Flux mode: chunk by chunk, uploading chunk k+1 and reading back chunk k−1 during the
    /// computation of chunk k; each chunk of each output is written straight to its destination
    /// (`outs`: (host address, bytes per element)).
    ///
    /// # Safety
    /// The host pointers (`Host` inputs, destinations) must cover `n` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn flux(
        &self,
        k: &WgKernel,
        inputs: &[FluxIn],
        outs: &[(*mut u8, usize)],
        scalars: &[u32],
        fparams: &[f64],
        n: usize,
        chunk_elems: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        let _ = self.flush_batch(); // a rejected batch only affects its own results
        for i in inputs {
            if let FluxIn::Dev { buf, .. } = i {
                self.check_buf(buf)?;
            }
        }
        const SLOTS: usize = 3;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let r =
            unsafe { self.flux_inner(k, inputs, outs, scalars, fparams, n, chunk_elems, SLOTS) };
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(oom.pop()),
        ];
        r?;
        if let Some(e) = errors.into_iter().flatten().next() {
            return Err(BackendError(format!(
                "wgpu: chunked processing rejected on {} ({e})",
                self.info.name
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn flux_inner(
        &self,
        k: &WgKernel,
        inputs: &[FluxIn],
        outs: &[(*mut u8, usize)],
        scalars: &[u32],
        fparams: &[f64],
        n: usize,
        chunk_elems: usize,
        slots: usize,
    ) -> Result<(), BackendError> {
        struct Slot {
            ins: Vec<WgBuffer>,
            ups: Vec<Option<Staging>>,
            outs: Vec<WgBuffer>,
            backs: Vec<Staging>,
            busy: Option<(usize, usize)>, // (first element, number of elements)
        }
        let chunk_elems = chunk_elems.max(1).min(n);
        let mut ring: Vec<Slot> = Vec::with_capacity(slots);
        for _ in 0..slots.min(n.div_ceil(chunk_elems)) {
            let mut ins = Vec::new();
            let mut ups = Vec::new();
            for i in inputs {
                let bytes = chunk_elems * i.elem();
                ins.push(self.alloc(bytes)?);
                ups.push(matches!(i, FluxIn::Host { .. }).then(|| self.staging(bytes, true)));
            }
            let mut obufs = Vec::new();
            let mut backs = Vec::new();
            for (_, elem) in outs {
                obufs.push(self.alloc(chunk_elems * elem)?);
                backs.push(self.staging(chunk_elems * elem, false));
            }
            ring.push(Slot {
                ins,
                ups,
                outs: obufs,
                backs,
                busy: None,
            });
        }
        let finish = |dev: &WgDevice, s: &mut Slot| -> Result<(), BackendError> {
            if let Some((start, len)) = s.busy.take() {
                for (back, (dst, elem)) in s.backs.iter().zip(outs) {
                    dev.wait_mapped(back)?;
                    let bytes = len * elem;
                    {
                        let view = back
                            .buf
                            .get_mapped_range(..bytes as u64)
                            .map_err(|e| err("reading a buffer", e))?;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                view.as_ptr(),
                                dst.add(start * elem),
                                bytes,
                            )
                        };
                    }
                    back.buf.unmap();
                }
            }
            Ok(())
        };
        let mut start = 0;
        let mut i = 0;
        while start < n {
            let len = chunk_elems.min(n - start);
            let slot = i % ring.len();
            finish(self, &mut ring[slot])?;
            let s = &mut ring[slot];
            let mut enc = self.device.create_command_encoder(&Default::default());
            for (a, input) in inputs.iter().enumerate() {
                let bytes = len * input.elem();
                match input {
                    FluxIn::Host { ptr, elem } => {
                        let st = s.ups[a].as_mut().unwrap();
                        self.wait_mapped(st)?;
                        {
                            let mut view = st
                                .buf
                                .get_mapped_range_mut(..bytes as u64)
                                .map_err(|e| err("writing a buffer", e))?;
                            view.copy_from_slice(unsafe {
                                std::slice::from_raw_parts(ptr.add(start * elem), bytes)
                            });
                        }
                        st.buf.unmap();
                        enc.copy_buffer_to_buffer(&st.buf, 0, &s.ins[a].buf, 0, bytes as u64);
                    }
                    FluxIn::Dev { buf, elem } => {
                        enc.copy_buffer_to_buffer(
                            &buf.buf,
                            (start * elem) as u64,
                            &s.ins[a].buf,
                            0,
                            bytes as u64,
                        );
                    }
                }
            }
            let mut params = scalars.to_vec();
            params.push(len as u32);
            let ins: Vec<&WgBuffer> = s.ins.iter().collect();
            let obufs: Vec<&WgBuffer> = s.outs.iter().collect();
            let call = self.prepare(k, &ins, &obufs, &params, fparams, len)?;
            call.record(&mut enc, k);
            for ((ob, back), (_, elem)) in s.outs.iter().zip(&s.backs).zip(outs) {
                enc.copy_buffer_to_buffer(
                    &ob.buf,
                    0,
                    &back.buf,
                    0,
                    (len * elem).div_ceil(4) as u64 * 4,
                );
            }
            let idx = self.queue.submit([enc.finish()]);
            drop(call);
            for st in s.ups.iter_mut().flatten() {
                self.remap(st, wgpu::MapMode::Write, &idx);
            }
            for back in s.backs.iter_mut() {
                self.remap(back, wgpu::MapMode::Read, &idx);
            }
            s.busy = Some((start, len));
            start += len;
            i += 1;
        }
        // Finishes the chunks still in flight, in order.
        let used = ring.len();
        for j in 0..used {
            let slot = (i + j) % used;
            finish(self, &mut ring[slot])?;
        }
        for s in ring {
            for st in s.ups.into_iter().flatten() {
                self.recycle(st, true);
            }
            for back in s.backs {
                self.recycle(back, false);
            }
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<(), BackendError> {
        self.check()?;
        self.flush_batch()?;
        let r = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.check()?;
        r.map_err(|e| err("waiting for the GPU", e))?;
        Ok(())
    }
}

/// Transfer buffer (mappable), with its mapping state.
struct Staging {
    buf: wgpu::Buffer,
    bytes: usize,
    mapped: Arc<AtomicBool>,
    submission: Option<wgpu::SubmissionIndex>,
}

/// Input of a chunked computation.
pub enum FluxIn<'a> {
    /// Array in host memory (`elem` bytes per element).
    Host { ptr: *const u8, elem: usize },
    /// Buffer already on the device (result of a previous kernel).
    Dev { buf: &'a WgBuffer, elem: usize },
}

impl FluxIn<'_> {
    fn elem(&self) -> usize {
        match self {
            FluxIn::Host { elem, .. } | FluxIn::Dev { elem, .. } => *elem,
        }
    }
}

fn bytemuck_u32(words: &[u32]) -> &[u8] {
    // u32 → bytes (native order), without an extra dependency.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpusync_runtime::Backend;

    const ADD1: &str = "
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < p[0]) { y[g.x] = x[g.x] + 1u; }
}";

    /// Device, kernel y = x + 1 already launched once (None: no wgpu GPU on this machine).
    fn warm() -> Option<(Arc<WgDevice>, Arc<WgKernel>)> {
        let info = crate::WgpuBackend.devices().ok()?.into_iter().next()?;
        let d = WgDevice::open(info.backend, info.index).unwrap();
        let k = d.compile(ADD1, 1, 1, false, &[false], None).unwrap();
        let (a, b) = (filled(&d, 1), d.alloc(64).unwrap());
        d.launch(&k, &[&a], &[&b], &[16], &[], 16).unwrap();
        assert_eq!(values(&d, &b).unwrap(), [2; 16]);
        Some((d, k))
    }

    fn filled(d: &WgDevice, v: u32) -> WgBuffer {
        let b = d.alloc(64).unwrap();
        d.write(&b, bytemuck_u32(&[v; 16])).unwrap();
        b
    }

    /// Input buffer destroyed: any submission that uses it is rejected by wgpu.
    fn destroyed(d: &WgDevice) -> WgBuffer {
        let buf = d.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut b = filled(d, 0);
        b.pool = None;
        b.buf = buf;
        b
    }

    fn values(d: &WgDevice, b: &WgBuffer) -> Result<Vec<u32>, BackendError> {
        let mut out = vec![0u8; 64];
        unsafe { d.download(b, 0, out.as_mut_ptr(), 64, 64)? };
        Ok(out
            .chunks(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// These tests share the device (and its batch): one at a time.
    static ONE: Mutex<()> = Mutex::new(());

    #[test]
    fn refused_launch_is_reported_at_once() {
        let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some((d, k)) = warm() else { return };
        let (bad, out) = (destroyed(&d), d.alloc(64).unwrap());
        bad.buf.destroy();
        assert!(d.launch(&k, &[&bad], &[&out], &[16], &[], 16).is_err());
        // The device stays usable, with no leftover error.
        let (a, b) = (filled(&d, 4), d.alloc(64).unwrap());
        d.launch(&k, &[&a], &[&b], &[16], &[], 16).unwrap();
        assert_eq!(values(&d, &b).unwrap(), [5; 16]);
    }

    #[test]
    fn refused_batch_poisons_its_results_only() {
        let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some((d, k)) = warm() else { return };
        let (bad, out, next) = (destroyed(&d), d.alloc(64).unwrap(), d.alloc(64).unwrap());
        let other = filled(&d, 7);
        d.begin_batch();
        d.launch(&k, &[&bad], &[&out], &[16], &[], 16).unwrap();
        bad.buf.destroy();
        assert!(d.end_batch().is_err());
        assert_eq!(values(&d, &other).unwrap(), [7; 16]);
        for _ in 0..2 {
            assert!(
                values(&d, &out).is_err(),
                "result never computed, read without error"
            );
            assert!(d.launch(&k, &[&out], &[&next], &[16], &[], 16).is_err());
        }
        let (a, b) = (filled(&d, 4), d.alloc(64).unwrap());
        d.launch(&k, &[&a], &[&b], &[16], &[], 16).unwrap();
        assert_eq!(values(&d, &b).unwrap(), [5; 16]);
    }

    #[test]
    fn write_waits_for_the_pending_batch() {
        let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some((d, k)) = warm() else { return };
        let (a, b, c) = (filled(&d, 10), d.alloc(64).unwrap(), d.alloc(64).unwrap());
        d.begin_batch();
        d.launch(&k, &[&a], &[&b], &[16], &[], 16).unwrap();
        let twenty = [20u32; 16];
        unsafe {
            d.upload(&a, 0, twenty.as_ptr().cast(), 64, 1 << 20)
                .unwrap()
        };
        d.launch(&k, &[&a], &[&c], &[16], &[], 16).unwrap();
        d.end_batch().unwrap();
        assert_eq!(values(&d, &b).unwrap(), [11; 16]);
        assert_eq!(values(&d, &c).unwrap(), [21; 16]);
    }
}
