//! Exécution OpenCL : contexte et file ordonnée par périphérique, compilation, buffers, lancement.
//!
//! Mémoire unifiée (§8.4) : `wrap_host` crée un buffer `CL_MEM_USE_HOST_PTR` qui utilise directement
//! la mémoire de l'hôte (tableau NumPy) ; `sync_host` (map + unmap) la rend cohérente après calcul.

use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MAP_READ, CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR, ClMem};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_mem};

fn err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("OpenCL : {what} ({e})"))
}

/// Un périphérique OpenCL ouvert.
pub struct ClDevice {
    pub info: DeviceInfo,
    context: Context,
    queue: CommandQueue,
    kernels: Mutex<HashMap<String, Arc<ClKernel>>>,
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

// Les objets OpenCL (cl_mem) sont utilisables depuis n'importe quel thread (spécification §A.2).
unsafe impl Send for ClBuffer {}
unsafe impl Sync for ClBuffer {}

impl ClBuffer {
    pub fn is_host_backed(&self) -> bool {
        !self.host_ptr.is_null()
    }
}

/// Argument de lancement, dans l'ordre des paramètres du kernel.
pub enum ClArg<'a> {
    Buf(&'a ClBuffer),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
}

static OPEN: OnceLock<Mutex<HashMap<usize, Arc<ClDevice>>>> = OnceLock::new();

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
        let queue =
            CommandQueue::create_default(&context, 0).map_err(|e| err("création de la file", e))?;
        let d = Arc::new(ClDevice {
            info,
            context,
            queue,
            kernels: Mutex::new(HashMap::new()),
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
    /// `ptr` doit rester valide (et ne pas être libéré) tant que le buffer existe.
    pub unsafe fn wrap_host(&self, ptr: *mut u8, bytes: usize) -> Result<ClBuffer, BackendError> {
        let buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                bytes,
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

    /// Copie bloquante hôte → périphérique.
    pub fn write(&self, buf: &ClBuffer, data: &[u8]) -> Result<(), BackendError> {
        let mut b = buf.buf_ref();
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut b, CL_BLOCKING, 0, data, &[])
        }
        .map_err(|e| err("écriture", e))?;
        Ok(())
    }

    /// Met le kernel en file sur `n` éléments (non bloquant).
    pub fn launch(&self, k: &ClKernel, args: &[ClArg], n: usize) -> Result<(), BackendError> {
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
        unsafe {
            for a in args {
                match a {
                    ClArg::Buf(_) => exec.set_arg(m.next().unwrap()),
                    ClArg::I32(v) => exec.set_arg(v),
                    ClArg::U32(v) => exec.set_arg(v),
                    ClArg::F32(v) => exec.set_arg(v),
                    ClArg::F64(v) => exec.set_arg(v),
                };
            }
            exec.set_global_work_size(n).enqueue_nd_range(&self.queue)
        }
        .map_err(|e| err("lancement du kernel", e))?;
        self.queue.flush().map_err(|e| err("flush", e))?;
        Ok(())
    }

    /// Copie bloquante périphérique → hôte (attend la fin des calculs en file).
    pub fn read(&self, buf: &ClBuffer, out: &mut [u8]) -> Result<(), BackendError> {
        unsafe {
            self.queue
                .enqueue_read_buffer(&buf.buf_ref(), CL_BLOCKING, 0, out, &[])
        }
        .map_err(|e| err("lecture", e))?;
        Ok(())
    }

    /// Rend la mémoire hôte d'un buffer `wrap_host` cohérente (attend la fin des calculs).
    pub fn sync_host(&self, buf: &ClBuffer) -> Result<(), BackendError> {
        let mut mapped: cl_mem = ptr::null_mut();
        unsafe {
            self.queue
                .enqueue_map_buffer(
                    &buf.buf_ref(),
                    CL_BLOCKING,
                    CL_MAP_READ,
                    0,
                    buf.bytes,
                    &mut mapped,
                    &[],
                )
                .map_err(|e| err("map", e))?;
            self.queue
                .enqueue_unmap_mem_object(buf.buf.get(), mapped.cast(), &[])
                .map_err(|e| err("unmap", e))?;
        }
        self.finish()?;
        Ok(())
    }

    /// Attend la fin de tout ce qui est en file.
    pub fn finish(&self) -> Result<(), BackendError> {
        self.queue.finish().map_err(|e| err("finish", e))
    }
}

impl ClBuffer {
    /// Vue « Buffer » empruntée (les API opencl3 attendent `&Buffer<T>`), sans prise de propriété.
    fn buf_ref(&self) -> std::mem::ManuallyDrop<Buffer<u8>> {
        std::mem::ManuallyDrop::new(Buffer::new(self.buf.get()))
    }
}
