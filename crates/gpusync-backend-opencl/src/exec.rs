//! Exécution OpenCL (ARCHITECTURE.md §8, §9).
//!
//! - **Deux files** par périphérique : `compute` (kernels, opérations sur mémoire hôte en zéro-copie)
//!   et `transfer` (copies hôte ↔ périphérique), chaînées par événements : les transferts d'un bloc
//!   se recouvrent avec le calcul d'un autre.
//! - **Mémoire unifiée** : `wrap_host` crée un buffer `CL_MEM_USE_HOST_PTR` qui utilise directement la
//!   mémoire de l'hôte ; `sync_host` (map + unmap) la rend cohérente après calcul.
//! - **Mémoire épinglée** : `pinned` fournit des tampons `CL_MEM_ALLOC_HOST_PTR` mappés une fois pour
//!   toutes ; un transfert depuis (ou vers) leur pointeur se fait par DMA à pleine vitesse.

use std::collections::HashMap;
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
    BackendError(format!("OpenCL : {what} ({e})"))
}

/// Un périphérique OpenCL ouvert.
pub struct ClDevice {
    pub info: DeviceInfo,
    context: Context,
    compute: CommandQueue,
    transfer: CommandQueue,
    kernels: Mutex<HashMap<String, Arc<ClKernel>>>,
    pinned_pool: Mutex<HashMap<usize, Vec<Pinned>>>,
    pool: Arc<BufPool>,
}

/// Kernel compilé. `set_arg` + lancement doivent être atomiques : d'où le verrou.
pub struct ClKernel {
    kernel: Mutex<Kernel>,
    _program: Program,
    /// Taille de groupe imposée (`group=`) ; sinon choisie par le driver.
    local: Option<usize>,
}

/// Buffer OpenCL, éventuellement adossé à la mémoire de l'hôte (`host_ptr` non nul).
pub struct ClBuffer {
    buf: std::mem::ManuallyDrop<Buffer<u8>>,
    pub bytes: usize,
    pub host_ptr: *mut u8,
    /// Capacité réelle et réserve où le rendre (buffers en mémoire du périphérique).
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

/// Buffers mis en réserve : capacité arrondie à la puissance de deux jusqu'à 64 Mio, taille exacte
/// au-delà, 2 Gio en tout (rendus au driver si une allocation échoue).
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

/// Réserve de buffers libérés : `clCreateBuffer` donne une mémoire neuve, payée à la première
/// écriture. Un buffer rendu est gardé avec un marqueur par file ; il est repris de préférence une
/// fois les marqueurs passés, sinon tout de suite, chaque file attendant (barrière côté
/// périphérique, sans bloquer l'hôte) le marqueur de l'autre, son propre ordre faisant le reste.
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
            return; // libéré par le driver une fois les commandes en file terminées
        }
        if let (Some(c), Some(t)) = (marker(self.compute, &[]), marker(self.transfer, &[])) {
            self.flush(); // un marqueur jamais envoyé bloquerait la barrière de l'autre file
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
            let ok = unsafe {
                opencl3::command_queue::enqueue_barrier_with_wait_list(self.compute, 1, &et.get())
                    .is_ok()
                    && opencl3::command_queue::enqueue_barrier_with_wait_list(
                        self.transfer,
                        1,
                        &ec.get(),
                    )
                    .is_ok()
            };
            self.flush();
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

// Les objets OpenCL (cl_mem, cl_event) sont utilisables depuis n'importe quel thread.
unsafe impl Send for ClBuffer {}
unsafe impl Sync for ClBuffer {}

impl ClBuffer {
    pub fn is_host_backed(&self) -> bool {
        !self.host_ptr.is_null()
    }

    /// Vue « Buffer » empruntée (les API opencl3 attendent `&Buffer<T>`), sans prise de propriété.
    fn view(&self) -> std::mem::ManuallyDrop<Buffer<u8>> {
        std::mem::ManuallyDrop::new(Buffer::new(self.buf.get()))
    }
}

/// Tampon hôte épinglé, mappé en permanence ; rendu au pool à la destruction.
pub struct Pinned {
    _buf: Buffer<u8>, // propriétaire : le tampon reste mappé tant qu'il vit
    pub ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

/// Argument de lancement, dans l'ordre des paramètres du kernel.
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
    /// Ouvre (une seule fois par processus) le périphérique d'index `index` (`opencl:<index>`).
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
            .ok_or_else(|| BackendError(format!("périphérique opencl:{index} introuvable")))?;
        let device = Device::new(id);
        let context = Context::from_device(&device).map_err(|e| err("création du contexte", e))?;
        #[allow(deprecated)] // clCreateCommandQueue : seule API disponible en OpenCL 1.2 (Apple)
        let compute =
            CommandQueue::create_default(&context, 0).map_err(|e| err("création de la file", e))?;
        #[allow(deprecated)]
        let transfer =
            CommandQueue::create_default(&context, 0).map_err(|e| err("création de la file", e))?;
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
        });
        open.insert(index, d.clone());
        Ok(d)
    }

    /// Compile (avec cache) le code OpenCL C et renvoie le kernel `entry`.
    /// `group` : taille de groupe imposée par le kernel (mémoire partagée, barrières).
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
        // Pas de -cl-fast-relaxed-math : la précision prime (résultats comparés à NumPy).
        let program =
            Program::create_and_build_from_source(&self.context, code, "").map_err(|log| {
                BackendError(format!(
                    "échec de compilation OpenCL sur {} :\n{log}",
                    self.info.name
                ))
            })?;
        let kernel = Kernel::create(&program, entry).map_err(|e| err("création du kernel", e))?;
        let k = Arc::new(ClKernel {
            kernel: Mutex::new(kernel),
            _program: program,
            local: group.map(|g| g as usize),
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    /// Buffer en mémoire du périphérique.
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
                    // Mémoire pleine : rendre la réserve au driver, puis réessayer.
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

    /// Buffer qui utilise directement la mémoire de l'hôte (zéro-copie sur mémoire unifiée).
    ///
    /// # Safety
    /// `ptr` doit rester valide (et ne pas être libéré) tant que le buffer peut être utilisé.
    pub unsafe fn wrap_host(&self, ptr: *mut u8, bytes: usize) -> Result<ClBuffer, BackendError> {
        let buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                bytes.max(4),
                ptr.cast(),
            )
        }
        .map_err(|e| err("buffer adossé à l'hôte", e))?;
        Ok(ClBuffer {
            buf: std::mem::ManuallyDrop::new(buf),
            bytes,
            host_ptr: ptr,
            cap: bytes,
            pool: None,
        })
    }

    /// Tampon épinglé d'au moins `bytes` octets (pris dans le pool si possible).
    pub fn pinned(&self, bytes: usize) -> Result<Pinned, BackendError> {
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
        .map_err(|e| err("tampon épinglé", e))?;
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
                .map_err(|e| err("map d'un tampon épinglé", e))?;
        }
        Ok(Pinned {
            _buf: buf,
            ptr: mapped.cast(),
            bytes,
        })
    }

    /// Rend un tampon épinglé au pool (il reste mappé, prêt à resservir).
    pub fn recycle(&self, p: Pinned) {
        self.pinned_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(p.bytes)
            .or_default()
            .push(p);
    }

    /// Copie hôte → périphérique (file de transfert, non bloquante), après `wait`.
    ///
    /// # Safety
    /// `src` doit rester valide jusqu'à la fin de l'événement renvoyé.
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
        .map_err(|e| err("écriture", e))
    }

    /// Copie périphérique → hôte (file de transfert, non bloquante), après `wait`.
    ///
    /// # Safety
    /// `dst` doit rester valide jusqu'à la fin de l'événement renvoyé.
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
        .map_err(|e| err("lecture", e))
    }

    /// Copie périphérique → périphérique (file de transfert), après `wait`.
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

    /// Met le kernel en file de calcul sur `n` éléments, après `wait`.
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
        .map_err(|e| err("lancement du kernel", e))?;
        self.compute.flush().map_err(|e| err("flush", e))?;
        Ok(ev)
    }

    /// Envoie les commandes des deux files au périphérique sans attendre.
    pub fn flush(&self) -> Result<(), BackendError> {
        self.transfer.flush().map_err(|e| err("flush", e))?;
        self.compute.flush().map_err(|e| err("flush", e))
    }

    /// Rend la mémoire hôte d'un buffer `wrap_host` cohérente (attend la fin des calculs).
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

    /// Attend la fin de tout ce qui est en file (calcul et transferts).
    pub fn finish(&self) -> Result<(), BackendError> {
        self.compute.finish().map_err(|e| err("finish", e))?;
        self.transfer.finish().map_err(|e| err("finish", e))
    }
}

pub fn wait(e: &Event) -> Result<(), BackendError> {
    e.wait().map_err(|e| err("attente", e))
}
