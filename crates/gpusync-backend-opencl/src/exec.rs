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
}

/// Kernel compilé. `set_arg` + lancement doivent être atomiques : d'où le verrou.
pub struct ClKernel {
    kernel: Mutex<Kernel>,
    _program: Program,
}

/// Buffer OpenCL, éventuellement adossé à la mémoire de l'hôte (`host_ptr` non nul).
pub struct ClBuffer {
    buf: Buffer<u8>,
    pub bytes: usize,
    pub host_ptr: *mut u8,
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
        let d = Arc::new(ClDevice {
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
    pub fn compile(&self, code: &str, entry: &str) -> Result<Arc<ClKernel>, BackendError> {
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
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    /// Buffer en mémoire du périphérique.
    pub fn alloc(&self, bytes: usize) -> Result<ClBuffer, BackendError> {
        let buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                bytes.max(4),
                ptr::null_mut(),
            )
        }
        .map_err(|e| err("allocation", e))?;
        Ok(ClBuffer {
            buf,
            bytes,
            host_ptr: ptr::null_mut(),
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
            buf,
            bytes,
            host_ptr: ptr,
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
            exec.set_global_work_size(n)
                .set_event_wait_list(&waits)
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
